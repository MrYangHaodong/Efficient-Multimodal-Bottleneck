"""Compare adaptive-early-exit CONFIDENCE SCHEMES (entropy vs alternatives) for seqA,
on the deployed RANDOM-order-trained (gap_dual) models, 4 datasets, cross-subject.
All schemes are post-hoc on the same per-prefix logits (no retraining). Per-fold val-LOO
order; sweep each scheme's threshold -> (meanK, acc, f1). Output -> seqA_exit_schemes_4ds.json
with per-dataset {M, gflops_per_k, consensus_order, schemes:{name:[[meanK,acc,f1],...]}}."""
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
from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import DualVideoBottleneckModelV6Downsample

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); set_seed(42)

# ---- threshold grids per scheme (fixed so folds aggregate cleanly) ----
GRID_ENTROPY = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 10.0]   # exit if H<=tau
GRID_MSP = [1.01, 0.999, 0.99, 0.98, 0.96, 0.93, 0.9, 0.85, 0.8, 0.72, 0.62, 0.5, 0.38, 0.25, 0.0]   # exit if maxp>=tau
GRID_MARGIN = [1.01, 0.99, 0.97, 0.94, 0.9, 0.83, 0.75, 0.65, 0.55, 0.42, 0.3, 0.2, 0.1, 0.03, -0.01]  # exit if p1-p2>=tau
GRID_ENERGY_Q = np.linspace(1.0, 0.0, 15)   # quantile levels of logsumexp(logits); high q -> high threshold
PATIENCES = [2, 3]


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


@torch.no_grad()
def acc_with(model, batches, keep):
    cor = n = 0
    for high, low, y in batches:
        o = model({m: high[m] for m in keep}, {m: low[m] for m in keep}, training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        cor += int((o.argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


@torch.no_grad()
def prefix_logits(model, batches, order, M):
    perk = [[] for _ in range(M)]; bperk = [[] for _ in range(M)]; labels = []
    fusion = model.bottleneck_fusion
    for high, low, y in batches:
        labels.append(y)
        for k in range(1, M + 1):
            keep = order[:k]
            o = model({m: high[m] for m in keep}, {m: low[m] for m in keep}, training=False, return_selection_info=False)
            o = o[0] if isinstance(o, tuple) else o
            perk[k - 1].append(o.cpu())
            bs = fusion._last_bn_state                      # {lyr: [B,K,H]} carried board after k modalities
            board = bs[max(bs)]                             # deepest fusion layer's board
            bperk[k - 1].append(board.reshape(board.shape[0], -1).cpu())   # [B, K*H]
    labels = torch.cat(labels).numpy()
    L = torch.stack([torch.cat(perk[k]) for k in range(M)], 1).float()   # [N,M,C]
    BD = torch.stack([torch.cat(bperk[k]) for k in range(M)], 1).float() # [N,M,K*H]
    return labels, L, BD


# ---- confidence signals from logits L [N,M,C] ----
def signals(L):
    P = F.softmax(L, -1)
    H = -(P * P.clamp_min(1e-9).log()).sum(-1).numpy()                 # [N,M]  low=confident
    top2 = P.topk(2, dim=-1).values
    msp = top2[..., 0].numpy()                                          # max prob
    margin = (top2[..., 0] - top2[..., 1]).numpy()                     # top1-top2
    energy = torch.logsumexp(L, -1).numpy()                            # high=confident
    preds = L.argmax(-1).numpy()
    return H, msp, margin, energy, preds


def exit_by_threshold(score, preds, labels, tau, M, ge=True):
    N = len(labels); ek = np.full(N, M)
    for k in range(M):
        cond = (score[:, k] >= tau) if ge else (score[:, k] <= tau)
        ek = np.where((ek == M) & cond, k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    return float(ek.mean()), float((pe == labels).mean()), float(f1_score(labels, pe, average='macro'))  # (meanK, acc, f1)


def exit_by_patience(preds, labels, p, M):
    N, _ = preds.shape; ek = np.full(N, M); cnt = np.ones(N, dtype=int)  # 1 prediction seen at k=1
    for k in range(1, M):
        agree = preds[:, k] == preds[:, k - 1]
        cnt = np.where(agree, cnt + 1, 1)
        ek = np.where((ek == M) & (cnt >= p), k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    return float(ek.mean()), float((pe == labels).mean()), float(f1_score(labels, pe, average='macro'))  # (meanK, acc, f1)


def bottleneck_saturation(BD):
    """Cosine similarity between the carried bottleneck board at prefix k and k-1 (per
    sample). High = board stopped changing when modality k was added -> fusion saturated.
    sim[:,0] = -1 (cannot measure change at k=1 -> never exit on saturation before k=2)."""
    BDn = torch.nn.functional.normalize(BD, dim=-1)              # [N,M,D]
    cos = (BDn[:, 1:, :] * BDn[:, :-1, :]).sum(-1).numpy()       # [N,M-1]
    N, M = BD.shape[0], BD.shape[1]
    sim = np.full((N, M), -1.0, dtype=np.float32); sim[:, 1:] = cos
    return sim


def run_length(preds, M):
    """Per-prefix run length of the current prediction (consecutive prefixes ending at k
    whose argmax == argmax at k). r in [1, k+1]."""
    N = preds.shape[0]; r = np.ones((N, M), dtype=int)
    for k in range(1, M):
        r[:, k] = np.where(preds[:, k] == preds[:, k - 1], r[:, k - 1] + 1, 1)
    return r
# Continuous patience = exit_by_threshold with score = run_length + maxprob (ge), tau in [1, M+1].
# At integer tau=p this is exactly patience-p; between integers, maxprob tie-breaks run-length-p samples.


def run_dataset(ds):
    seqa_dir = sorted(glob.glob(os.path.join(_ROOT, 'results_3p3', f'seqA_{ds}', '*seqA_gap_dual*')))[0]
    C = json.load(open(os.path.join(seqa_dir, 'config.json')))
    if ds == 'iemocap':
        from data.IEMOCAP.get_data import get_dataloader, FEATURE_DIMS, NUM_CLASSES
        MODS = list(C['modalities']); M = len(MODS); NUMC = NUM_CLASSES
        T = C['max_seq_len'] if C['max_seq_len'] > 0 else 600; vdim = FEATURE_DIMS.get('video', 1024)
        build = make_build(C, NUMC, T, vdim, vdim, {m: FEATURE_DIMS[m] for m in MODS}); folds = range(3)

        def loaders(fold):
            _, vl, el = get_dataloader(data_root='/files1/haodong/data/IEMOCAP', batch_size=48, num_workers=4,
                modalities=MODS, split_id=fold, train_shuffle=False, max_seq_len=C['max_seq_len'],
                time_compression_ratio=C['time_compression_ratio'], use_batched_fusion=C['use_batched_fusion'],
                available_sessions=None, split_mode='cross_subject')
            def conv(loader):
                out = []
                for batch in loader:
                    *feats, label = batch; feats = [f.to(dev).float() for f in feats]
                    y = label.squeeze(-1).long()
                    out.append(({m: feats[2 * j] for j, m in enumerate(MODS)},
                                {m: feats[2 * j + 1] for j, m in enumerate(MODS)}, y))
                return out
            return conv(vl), conv(el)
    else:
        from utils.dataset_cfg import DaliaHAR, PAMAP2, DSADS
        if ds == 'daliahar':
            cfg = DaliaHAR(); TRANSFORM = 'sax'; T = (cfg.duration * cfg.base_sample_rate) // 1
        elif ds == 'pamap2':
            cfg = PAMAP2(); TRANSFORM = 'sax_noisy'; T = 512
        else:
            cfg = DSADS(); TRANSFORM = 'sax'; T = (cfg.duration * cfg.base_sample_rate) // 1
        SAX = {'alphabet_size': 20, 'word_length': 1}
        MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
        sl, off = {}, 0
        for m in MODS:
            sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
        v0 = cfg.variates[MODS[0]]
        build = make_build(C, NUMC, T, v0, v0, {m: cfg.variates[m] for m in MODS}); folds = range(4) if ds == 'dsads' else range(3)

        def har_batches(loader):
            out = []
            for x, y in loader:
                x = x.to(dev).float(); high = {m: x[:, :, sl[m][0]:sl[m][1]] for m in MODS}
                out.append((high, dict(high), y))
            return out

        def loaders(fold):
            if ds == 'dsads':
                from data.dataset_builder import DSADSDataset
                DATA = '/files1/haodong/data/DSADS/'
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
            return har_batches(vl), har_batches(el)

    scheme_names = ['entropy', 'maxprob', 'margin', 'energy', 'patience', 'bottleneck'] + [f'patience{p}' for p in PATIENCES]
    taus_pat = np.linspace(1.0, M + 1.0, 30)                  # continuous patience threshold grid (M-dependent)
    perfold = {nm: [] for nm in scheme_names}                 # list of per-fold curves: [(meanK,acc,f1),...]
    borda = {m: 0.0 for m in MODS}
    raw = {'labels': [], 'preds': [], 'msp': [], 'bn_sim': [], 'runlen': []}   # per-sample, pooled over folds
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
        labels, L, BD = prefix_logits(model, eb, order, M)
        H, msp, margin, energy, preds = signals(L)
        bn_sim = bottleneck_saturation(BD)                    # [N,M] cosine board-stability
        perfold['entropy'].append([exit_by_threshold(H, preds, labels, t, M, ge=False) for t in GRID_ENTROPY])
        perfold['maxprob'].append([exit_by_threshold(msp, preds, labels, t, M, ge=True) for t in GRID_MSP])
        perfold['margin'].append([exit_by_threshold(margin, preds, labels, t, M, ge=True) for t in GRID_MARGIN])
        e_taus = np.quantile(energy.reshape(-1), GRID_ENERGY_Q)
        perfold['energy'].append([exit_by_threshold(energy, preds, labels, t, M, ge=True) for t in e_taus])
        pat_score = run_length(preds, M) + msp                # [N,M], continuous patience evidence
        perfold['patience'].append([exit_by_threshold(pat_score, preds, labels, t, M, ge=True) for t in taus_pat])
        bn_taus = np.quantile(bn_sim[:, 1:].reshape(-1), np.linspace(1.0, 0.0, 22))  # cosine quantiles (high=exit late)
        perfold['bottleneck'].append([exit_by_threshold(bn_sim, preds, labels, t, M, ge=True) for t in bn_taus])
        for p in PATIENCES:
            perfold[f'patience{p}'].append([exit_by_patience(preds, labels, p, M)])
        raw['labels'].append(labels); raw['preds'].append(preds); raw['msp'].append(msp)
        raw['bn_sim'].append(bn_sim); raw['runlen'].append(run_length(preds, M))
        print(f'  {ds} fold{fold} full_val={full:.4f} order={order}')
    # average each scheme's curve across folds (same #points per fold)
    schemes = {}
    for nm in scheme_names:
        arrs = np.array(perfold[nm])            # [folds, npts, 3]
        schemes[nm] = np.mean(arrs, axis=0).tolist()
    consensus = sorted(MODS, key=lambda m: -borda[m])
    # per-K GFLOPs at T (single sample, consensus order)
    vb0, _ = loaders(list(folds)[0]); x_high, x_low, _ = vb0[0]
    one_h = {m: x_high[m][:1] for m in MODS}; one_l = {m: x_low[m][:1] for m in MODS}
    mm = build(MODS).eval(); gpk = []
    for k in range(1, M + 1):
        keep = consensus[:k]; fcm = FlopCounterMode(display=False)
        with fcm, torch.no_grad():
            mm({m: one_h[m] for m in keep}, {m: one_l[m] for m in keep}, training=False, return_selection_info=False)
        gpk.append(fcm.get_total_flops() / 1e9)
    raw_pooled = {k: np.concatenate(v, axis=0) for k, v in raw.items()}   # pool samples over folds
    return {'M': M, 'gflops_per_k': gpk, 'consensus_order': consensus, 'schemes': schemes}, raw_pooled


out = {}; RAW = {}
for ds in ['iemocap', 'daliahar', 'pamap2', 'dsads']:
    print(f'=== {ds} ===')
    out[ds], _raw = run_dataset(ds)
    RAW[f'{ds}_gflops_per_k'] = np.array(out[ds]['gflops_per_k']); RAW[f'{ds}_M'] = np.array(out[ds]['M'])
    for kk, vv in _raw.items():
        RAW[f'{ds}_{kk}'] = vv
    for nm, cur in out[ds]['schemes'].items():
        c = np.array(cur)
        print(f"  {nm:10s}: acc {c[:,1].min():.3f}-{c[:,1].max():.3f}  meanK {c[:,0].min():.2f}-{c[:,0].max():.2f}")
json.dump(out, open(os.path.join(_ROOT, 'results_3p3', 'seqA_exit_schemes_4ds.json'), 'w'), indent=2)
print('saved -> results_3p3/seqA_exit_schemes_4ds.json')
np.savez(os.path.join(_ROOT, 'results_3p3', 'seqA_exit_signals_raw_4ds.npz'), **RAW)
print('saved -> results_3p3/seqA_exit_signals_raw_4ds.npz')
