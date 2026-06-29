"""Entropy-head seqA adaptive-exit curves on 4 datasets (same val-LOO + entropy
early-exit method as the gap-head exit scripts, but head_mode='gap' and reading
the results_3p3/entropy_head/<ds>/seqA models). Output -> entropy_head/exit_entropy2_4ds.json
with per-dataset {M, exit_curve:[meanK,acc,f1], gflops_per_k, consensus_order}.
Run from clean_models/train/:  python ../results_3p3/compute_entropy_exit_4ds.py"""
from __future__ import annotations
import glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from utils.helper_function import set_seed
from models.seqA import DualVideoBottleneckModelV6Downsample

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); set_seed(42)
TAUS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 10.0]
ENT = os.path.join(_ROOT, 'results_3p3', 'entropy_head2')


def make_build(C, NUMC, T, vhigh, vlow, modalities_cfg):
    class _VC:
        def __init__(s, mods):
            s.modalities = list(mods); s.variates = {m: modalities_cfg[m] for m in mods}
            s.video_high_dim = vhigh; s.video_low_dim = vlow
    def build(mods):
        return DualVideoBottleneckModelV6Downsample(
            head_mode='gap', cfg=_VC(mods), output_dim=NUMC, input_length=T, d_model=C['d_model'], nhead=C['nhead'],
            num_layers_per_modal=C['num_layers_per_modal'], num_layers=C['num_layers'], dropout=C['dropout'], verbose=False,
            video_low_dim=vlow, video_high_dim=vhigh, use_bottleneck=True,
            n_bottlenecks=C['n_bottlenecks'], fusion_layer=C['fusion_layer'], use_sparse_moe=False,
            num_experts=C['num_experts'], expert_k=1, internal_dim=C['internal_dim'], bottleneck_head_pos=True,
            use_sparse_attn=C['use_sparse_attn'], factor=C['base_factor'], selector_video_source='high',
            encoder_video_source='high', no_selector=True, use_weighted_factor=False, use_triton=False, num_classes=NUMC,
            sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
            downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
            per_modal_distill=C['per_modal_distill'], per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
            fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full',
            fusion_mlp_ratio=1.0, fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'],
            seq_modality_order=C['seq_modality_order'], seq_random_order=C['seq_random_order'],
            bottleneck_agg_mode=C.get('bottleneck_agg_mode', 'gate'), n_fusion_distill=C.get('n_fusion_distill', -1)).to(dev).float()
    return build


def load(build, ckpt):
    m = build(['x'] if False else None) if False else build
    return None


@torch.no_grad()
def acc_with(model, batches, keep):
    cor = n = 0
    for high, low, y in batches:
        h = {m: high[m] for m in keep}; l = {m: low[m] for m in keep}
        o = model(h, l, training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        cor += int((o.argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


@torch.no_grad()
def prefix(model, batches, order, M):
    by, labels = [], []
    perk = [[] for _ in range(M)]
    for high, low, y in batches:
        labels.append(y)
        for k in range(1, M + 1):
            keep = order[:k]
            h = {m: high[m] for m in keep}; l = {m: low[m] for m in keep}
            o = model(h, l, training=False, return_selection_info=False)
            o = o[0] if isinstance(o, tuple) else o
            perk[k - 1].append(o.cpu())
    labels = torch.cat(labels).numpy()
    L = torch.stack([torch.cat(perk[k]) for k in range(M)], 1)   # [N,M,C]
    P = F.softmax(L, -1); Hs = -(P * P.clamp_min(1e-9).log()).sum(-1).numpy(); preds = L.argmax(-1).numpy()
    return labels, Hs, preds


def simulate(labels, Hs, preds, tau, M):
    N = len(labels); ek = np.full(N, M)
    for k in range(M):
        ek = np.where((ek == M) & (Hs[:, k] <= tau), k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    return float((pe == labels).mean()), float(f1_score(labels, pe, average='macro')), float(ek.mean())


# ----- per-dataset cached batches (high/low dicts + labels on CPU? keep on dev for speed) -----
def har_batches(loader, MODS, sl):
    out = []
    for x, y in loader:
        x = x.to(dev).float()
        high = {m: x[:, :, sl[m][0]:sl[m][1]] for m in MODS}
        out.append((high, dict(high), y))
    return out


def run_dataset(ds):
    seqa_dir = sorted(glob.glob(os.path.join(_ROOT, 'results_3p3', f'seqA_{ds}', '*seqA_gap_dual*')))[0]
    C = json.load(open(os.path.join(seqa_dir, 'config.json')))
    if ds == 'iemocap':
        from data.IEMOCAP.get_data import get_dataloader, FEATURE_DIMS, NUM_CLASSES
        MODS = list(C['modalities']); M = len(MODS); NUMC = NUM_CLASSES
        T = C['max_seq_len'] if C['max_seq_len'] > 0 else 600
        vdim = FEATURE_DIMS.get('video', 1024)
        build = make_build(C, NUMC, T, vdim, vdim, {m: FEATURE_DIMS[m] for m in MODS})
        folds = range(3)

        def loaders(fold):
            _, vl, el = get_dataloader(data_root='/files1/haodong/data/IEMOCAP', batch_size=64, num_workers=4,
                modalities=MODS, split_id=fold, train_shuffle=False, max_seq_len=C['max_seq_len'],
                time_compression_ratio=C['time_compression_ratio'], use_batched_fusion=C['use_batched_fusion'],
                available_sessions=None, split_mode='cross_subject')

            def conv(loader):
                out = []
                for batch in loader:
                    *feats, label = batch
                    feats = [f.to(dev).float() for f in feats]
                    y = label.squeeze(-1).long()
                    high = {m: feats[2 * j] for j, m in enumerate(MODS)}
                    low = {m: feats[2 * j + 1] for j, m in enumerate(MODS)}
                    out.append((high, low, y))
                return out
            return conv(vl), conv(el)
    else:
        from utils.dataset_cfg import DaliaHAR, PAMAP2, DSADS
        if ds == 'daliahar':
            cfg = DaliaHAR(); TRANSFORM = 'sax'; T = (cfg.duration * cfg.base_sample_rate) // 1; SAX = {'alphabet_size': 20, 'word_length': 1}
        elif ds == 'pamap2':
            cfg = PAMAP2(); TRANSFORM = 'sax_noisy'; T = 512; SAX = {'alphabet_size': 20, 'word_length': 1}
        else:
            cfg = DSADS(); TRANSFORM = 'sax'; T = (cfg.duration * cfg.base_sample_rate) // 1; SAX = {'alphabet_size': 20, 'word_length': 1}
        MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
        sl, off = {}, 0
        for m in MODS:
            sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
        v0 = cfg.variates[MODS[0]]
        build = make_build(C, NUMC, T, v0, v0, {m: cfg.variates[m] for m in MODS})
        folds = range(4) if ds == 'dsads' else range(3)

        def loaders(fold):
            if ds == 'dsads':
                DATA = '/files1/haodong/data/DSADS/'
                from data.dataset_builder import DSADSDataset
                _src = torch.load(DATA + f'dsad_diversify_dict_scenario{fold}_src.pt', weights_only=False)
                _trg = torch.load(DATA + f'dsad_diversify_dict_scenario{fold}_trg.pt', weights_only=False)
                _, _vax, _, _vay = train_test_split(_src['samples'], _src['labels'], test_size=0.2, random_state=2711, stratify=_src['labels'])
                vl = DataLoader(DSADSDataset({'samples': _vax, 'labels': _vay}, MODS, cfg, 'sax', sax_params=SAX), batch_size=64, shuffle=False, num_workers=4)
                el = DataLoader(DSADSDataset(_trg, MODS, cfg, 'sax', sax_params=SAX), batch_size=64, shuffle=False, num_workers=4)
            elif ds == 'pamap2':
                from data.pamap2_crosssubject import load_pamap2_subjects
                fc = cfg.folds[fold]
                vl = DataLoader(load_pamap2_subjects(cfg.cross_subject_dir, MODS, cfg.val_set, cfg, TRANSFORM, sax_params=SAX), batch_size=64, shuffle=False, num_workers=4)
                el = DataLoader(load_pamap2_subjects(cfg.cross_subject_dir, MODS, fc['eval_set'], cfg, TRANSFORM, sax_params=SAX), batch_size=64, shuffle=False, num_workers=4)
            else:
                from data.dataset_builder import HARDataset
                ROOT = '/files1/haodong/data/processed_dalia_activity'; fc = cfg.folds[fold]
                vl = DataLoader(HARDataset(ROOT, MODS, cfg.val_set, cfg, 'sax', sax_params=SAX), batch_size=64, shuffle=False, num_workers=4)
                el = DataLoader(HARDataset(ROOT, MODS, fc['eval_set'], cfg, 'sax', sax_params=SAX), batch_size=64, shuffle=False, num_workers=4)
            return har_batches(vl, MODS, sl), har_batches(el, MODS, sl)

    # ---- per-fold val-LOO order + entropy exit on test ----
    perfold = {t: {'acc': [], 'f1': [], 'mk': []} for t in TAUS}
    borda = {m: 0.0 for m in MODS}
    for fold in folds:
        vb, eb = loaders(fold)
        model = build(MODS)
        sd = torch.load(os.path.join(seqa_dir, f'best_model_loo_fold{fold}.pth'), map_location='cpu', weights_only=True)
        model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()
        full = acc_with(model, vb, MODS)
        loo = {m: full - acc_with(model, vb, [x for x in MODS if x != m]) for m in MODS}
        order = sorted(MODS, key=lambda m: -loo[m])
        for r, m in enumerate(order):
            borda[m] += (M - r)
        labels, Hs, preds = prefix(model, eb, order, M)
        for t in TAUS:
            a, f, mk = simulate(labels, Hs, preds, t, M)
            perfold[t]['acc'].append(a); perfold[t]['f1'].append(f); perfold[t]['mk'].append(mk)
        print(f'  {ds} fold{fold} full_val={full:.4f} order={order}')
    consensus = sorted(MODS, key=lambda m: -borda[m])
    exit_curve = sorted([(float(np.mean(perfold[t]['mk'])), float(np.mean(perfold[t]['acc'])), float(np.mean(perfold[t]['f1']))) for t in TAUS])

    # per-K gflops at T (build once, single sample)
    vb0, _ = loaders(list(folds)[0])
    x_high, x_low, _ = vb0[0]
    one_h = {m: x_high[m][:1] for m in MODS}; one_l = {m: x_low[m][:1] for m in MODS}
    mm = build(MODS).eval(); gpk = []
    for k in range(1, M + 1):
        keep = consensus[:k]
        fcm = FlopCounterMode(display=False)
        with fcm, torch.no_grad():
            mm({m: one_h[m] for m in keep}, {m: one_l[m] for m in keep}, training=False, return_selection_info=False)
        gpk.append(fcm.get_total_flops() / 1e9)
    return {'M': M, 'exit_curve': exit_curve, 'gflops_per_k': gpk, 'consensus_order': consensus}


out = {}
for ds in ['iemocap', 'daliahar', 'pamap2', 'dsads']:
    print(f'=== {ds} (entropy head) ===')
    out[ds] = run_dataset(ds)
    ec = out[ds]['exit_curve']
    print(f"  -> exit acc {ec[0][1]:.3f}->{ec[-1][1]:.3f} | gflops_per_k={[round(g,3) for g in out[ds]['gflops_per_k']]}")
json.dump(out, open(os.path.join(_ROOT, 'results_3p3', 'seqA_gap_dual_exit_4ds.json'), 'w'), indent=2)
print('saved -> results_3p3/seqA_gap_dual_exit_4ds.json')
