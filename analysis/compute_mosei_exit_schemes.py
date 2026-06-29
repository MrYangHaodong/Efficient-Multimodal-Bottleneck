"""Adaptive early-exit on CMU-MOSEI seqA: ORIGINAL training vs PREFIX-SUPERVISION,
across the 5 new seeds. Schemes: entropy / patience(continuous) / bottleneck-saturation.
M=3 (vision/audio/text), Acc-2, T=50 float. Uses best_model_cold_val.pth (val-LOO order
matches the deployment exit order). Output -> results_3p3/mosei/exit_schemes_mosei.json"""
from __future__ import annotations
import glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import f1_score
_HERE = os.path.dirname(os.path.abspath(__file__)); _CM = os.path.dirname(_HERE); sys.path.insert(0, _CM)
from utils.helper_function import set_seed
from data.CMU_MOSEI.get_data import get_dataloader, CMUMOSEIDataset
from models.seqA import DualVideoBottleneckModelV6Downsample

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); set_seed(42)
MODS = list(CMUMOSEIDataset.ALL_MODALITIES); M = len(MODS); NUMC = 2; T = 50
SEEDS = [21, 365, 1024]
RES = os.path.join(_CM, 'train', 'results_mosei_5seed')
GRID_ENT = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 10.0]
GRID_MSP = [1.01, 0.999, 0.99, 0.98, 0.96, 0.93, 0.9, 0.85, 0.8, 0.72, 0.62, 0.5, 0.38, 0.25, 0.0]


def variant_dir(tag, seed):
    pat = sorted(glob.glob(os.path.join(RES, f'*_{tag}_acc2_s{seed}_*')))
    pat = [d for d in pat if (tag == 'seqA_ps') == ('seqA_ps' in d)]
    return pat[-1]


def build(C, VAR):
    class _Cfg:
        modalities = list(MODS); variates = dict(VAR)
        video_high_dim = VAR[MODS[0]]; video_low_dim = VAR[MODS[0]]
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=_Cfg(), output_dim=NUMC, input_length=T, d_model=C['d_model'], nhead=C['nhead'],
        num_layers_per_modal=C['num_layers_per_modal'], num_layers=C['num_layers'], dropout=C['dropout'], verbose=False,
        video_low_dim=VAR[MODS[0]], video_high_dim=VAR[MODS[0]], use_bottleneck=True,
        n_bottlenecks=C['n_bottlenecks'], fusion_layer=C['fusion_layer'], use_sparse_moe=False,
        num_experts=C['num_experts'], expert_k=1, internal_dim=C['internal_dim'], bottleneck_head_pos=True,
        use_sparse_attn=C['use_sparse_attn'], factor=C['base_factor'], selector_video_source='high',
        encoder_video_source='high', no_selector=True, use_weighted_factor=False, use_triton=False, num_classes=NUMC,
        sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
        downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
        per_modal_distill=C['per_modal_distill'], per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
        fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full', fusion_mlp_ratio=1.0,
        fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'], seq_modality_order=C['seq_modality_order'],
        seq_random_order=C['seq_random_order'], bottleneck_agg_mode=C.get('bottleneck_agg_mode', 'gate'),
        n_fusion_distill=C.get('n_fusion_distill', -1)).to(dev).float()


def preload(loader):
    out = []
    for batch in loader:
        *feats, label = batch
        high = {m: feats[2 * j].to(dev).float() for j, m in enumerate(MODS)}
        low = {m: feats[2 * j + 1].to(dev).float() for j, m in enumerate(MODS)}
        out.append((high, low, label.squeeze(-1).long()))
    return out


def _fwd(model, high, low, keep):
    o = model({m: high[m] for m in keep}, {m: low[m] for m in keep}, training=False, return_selection_info=False)
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def acc_with(model, batches, keep):
    cor = n = 0
    for high, low, y in batches:
        cor += int((_fwd(model, high, low, keep).argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


@torch.no_grad()
def prefix_LB(model, batches, order):
    fusion = model.bottleneck_fusion
    by, bby, labels = [[] for _ in range(M)], [[] for _ in range(M)], []
    for high, low, y in batches:
        labels.append(y)
        for k in range(1, M + 1):
            o = _fwd(model, high, low, order[:k]); by[k - 1].append(o.cpu())
            bs = fusion._last_bn_state; bd = bs[max(bs)]
            bby[k - 1].append(bd.reshape(bd.shape[0], -1).cpu())
    labels = torch.cat(labels).numpy()
    L = torch.stack([torch.cat(by[k]) for k in range(M)], 1).float()
    BD = torch.stack([torch.cat(bby[k]) for k in range(M)], 1).float()
    return labels, L, BD


def signals(L, BD):
    P = F.softmax(L, -1)
    Hs = -(P * P.clamp_min(1e-9).log()).sum(-1).numpy()
    msp = P.max(-1).values.numpy(); preds = L.argmax(-1).numpy()
    rl = np.ones((preds.shape[0], M), dtype=int)
    for k in range(1, M):
        rl[:, k] = np.where(preds[:, k] == preds[:, k - 1], rl[:, k - 1] + 1, 1)
    BDn = F.normalize(BD, dim=-1)
    cos = (BDn[:, 1:] * BDn[:, :-1]).sum(-1).numpy()
    bn = np.full((preds.shape[0], M), -1.0, np.float32); bn[:, 1:] = cos
    return Hs, msp, preds, rl, bn


def exit_thr(score, preds, labels, tau, ge):
    N = len(labels); ek = np.full(N, M)
    for k in range(M):
        cond = (score[:, k] >= tau) if ge else (score[:, k] <= tau)
        ek = np.where((ek == M) & cond, k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    return float(ek.mean()), float((pe == labels).mean()), float(f1_score(labels, pe, average='macro'))


def run_variant(tag):
    per = {'entropy': [], 'maxprob': [], 'patience': [], 'bottleneck': []}; gpk = None
    taus_pat = np.linspace(1.0, M + 1.0, 24)
    for seed in SEEDS:
        d = variant_dir(tag, seed); C = json.load(open(os.path.join(d, 'config.json')))
        VAR = {m: C['feature_dims'][m] for m in MODS}
        _, vl, tl = get_dataloader(batch_size=128, num_workers=2, modalities=MODS, split_id=0,
                                   max_seq_len=T, time_compression_ratio=4, task='classification2')
        vb, tb = preload(vl), preload(tl)
        m = build(C, VAR)
        sd = torch.load(os.path.join(d, 'best_model_cold_val.pth'), map_location='cpu', weights_only=True)
        m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); m.eval()
        full = acc_with(m, vb, MODS)
        loo = {mm: full - acc_with(m, vb, [x for x in MODS if x != mm]) for mm in MODS}
        order = sorted(MODS, key=lambda mm: -loo[mm])
        labels, L, BD = prefix_LB(m, tb, order)
        Hs, msp, preds, rl, bn = signals(L, BD)
        per['entropy'].append([exit_thr(Hs, preds, labels, t, ge=False) for t in GRID_ENT])
        per['maxprob'].append([exit_thr(msp, preds, labels, t, ge=True) for t in GRID_MSP])
        per['patience'].append([exit_thr(rl + msp, preds, labels, t, ge=True) for t in taus_pat])
        bn_taus = np.quantile(bn[:, 1:].reshape(-1), np.linspace(1.0, 0.0, 16))
        per['bottleneck'].append([exit_thr(bn, preds, labels, t, ge=True) for t in bn_taus])
        if gpk is None:
            h1 = {mm: tb[0][0][mm][:1] for mm in MODS}; l1 = {mm: tb[0][1][mm][:1] for mm in MODS}
            consensus = order; gpk = []
            for k in range(1, M + 1):
                fcm = FlopCounterMode(display=False)
                with fcm, torch.no_grad():
                    _fwd(m, h1, l1, consensus[:k])
                gpk.append(fcm.get_total_flops() / 1e9)
        print(f'  {tag} s{seed}: full_val={full:.4f} order={order}')
        del m; torch.cuda.empty_cache()
    schemes = {nm: np.mean(np.array(v), axis=0).tolist() for nm, v in per.items()}
    return {'M': M, 'gflops_per_k': gpk, 'schemes': schemes}


out = {}
for tag in ['seqA', 'seqA_ps']:
    print(f'=== {tag} ===')
    out[tag] = run_variant(tag)
    for nm, c in out[tag]['schemes'].items():
        c = np.array(c); print(f'  {nm:10s}: acc {c[:,1].min():.3f}-{c[:,1].max():.3f}  meanK {c[:,0].min():.2f}-{c[:,0].max():.2f}')
os.makedirs(os.path.join(_HERE, 'mosei'), exist_ok=True)
json.dump(out, open(os.path.join(_HERE, 'mosei', 'exit_schemes_mosei_3seed.json'), 'w'), indent=2)
print('saved -> results_3p3/mosei/exit_schemes_mosei_3seed.json')
