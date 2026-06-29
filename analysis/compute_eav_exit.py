"""seqA EAV adaptive-exit curve + per-K modality GFLOPs (val-LOO modality order,
strong-first; same method as compute_mosei_exit.py).

EAV: M=3 (eeg/audio/video), FLOAT input T=128, 5-class emotion. The 3 cross-subject
folds play the role of folds. Uses the seqA_prefixsup best_model_loo_fold{f}.pth
checkpoints (wrapper state, 'model.' prefix stripped to the inner V6 model).

Output -> results_3p3/eav/exit_eav.json.
Run from clean_models/train/:  python ../results_3p3/compute_eav_exit.py
"""
from __future__ import annotations
import glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_CM = os.path.dirname(_HERE)
sys.path.insert(0, _CM)
from utils.helper_function import set_seed                                    # noqa: E402
from data.EAV.get_data import get_dataloader, EAVDataset, FEATURE_DIMS         # noqa: E402
from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import (  # noqa: E402
    DualVideoBottleneckModelV6Downsample)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
set_seed(42)
TAUS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 10.0]
MODS = list(EAVDataset.ALL_MODALITIES)            # ['eeg','audio','video']
M = len(MODS); NUMC = 5; T = 128
FOLDS = [0, 1, 2]
SEQA_DIR = os.path.join(_CM, 'train', 'results_eav', 'seqA_prefixsup',
                        '2026-06-28_EAV_seqA_prefixsup_seqA')
# argparse defaults not stored in config.json:
_DEFAULTS = dict(bottleneck_agg_mode='gate', bottleneck_init_mode='random',
                 fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256,
                 fusion_pos_embed_mode='full', fusion_mlp_ratio=1.0,
                 share_fusion_params=False, head_mode='gap')
_C = {**_DEFAULTS, **json.load(open(os.path.join(SEQA_DIR, 'config.json')))}
VAR = {m: FEATURE_DIMS[m] for m in MODS}


class _V6Cfg:
    def __init__(s):
        s.modalities = list(MODS); s.variates = dict(VAR)
        v = FEATURE_DIMS.get('video', 1024)       # matches _EAVV6Cfg
        s.video_high_dim = v; s.video_low_dim = v


def build():
    return DualVideoBottleneckModelV6Downsample(
        head_mode=_C['head_mode'], cfg=_V6Cfg(), output_dim=NUMC, input_length=T,
        d_model=_C['d_model'], nhead=_C['nhead'],
        num_layers_per_modal=_C['num_layers_per_modal'], num_layers=_C['num_layers'],
        dropout=_C['dropout'], verbose=False,
        video_low_dim=VAR['video'], video_high_dim=VAR['video'], use_bottleneck=True,
        n_bottlenecks=_C['n_bottlenecks'], fusion_layer=_C['fusion_layer'], use_sparse_moe=False,
        num_experts=_C['num_experts'], expert_k=1, internal_dim=_C['internal_dim'],
        bottleneck_head_pos=True, use_sparse_attn=_C['use_sparse_attn'], factor=_C['base_factor'],
        selector_video_source='high', encoder_video_source='high', no_selector=True,
        use_weighted_factor=False, use_triton=False, num_classes=NUMC,
        sparse_attn_variant=_C['sparse_attn_variant'], strat_block_size=_C['strat_block_size'],
        downsample_min_len=_C['downsample_min_len'], use_batched_fusion=_C['use_batched_fusion'],
        per_modal_distill=_C['per_modal_distill'],
        per_modal_downsample_min_len=_C['per_modal_downsample_min_len'],
        bottleneck_init_mode=_C['bottleneck_init_mode'],
        fusion_add_pos_embeds=_C['fusion_add_pos_embeds'],
        fusion_pos_embeds_max_len=_C['fusion_pos_embeds_max_len'],
        fusion_pos_embed_mode=_C['fusion_pos_embed_mode'], fusion_mlp_ratio=_C['fusion_mlp_ratio'],
        share_fusion_params=_C['share_fusion_params'],
        fusion_mode=_C['fusion_mode'], seq_depth_flow=_C['seq_depth_flow'],
        seq_modality_order=_C['seq_modality_order'], seq_random_order=_C['seq_random_order'],
        bottleneck_agg_mode=_C['bottleneck_agg_mode'], n_fusion_distill=_C['n_fusion_distill']).to(dev).float()


def preload(loader):
    out = []
    for batch in loader:
        *feats, label = batch
        high = {m: feats[2 * j].to(dev).float() for j, m in enumerate(MODS)}
        low = {m: feats[2 * j + 1].to(dev).float() for j, m in enumerate(MODS)}
        out.append((high, low, label.squeeze(-1).long()))
    return out


def _fwd(model, high, low, keep):
    o = model({m: high[m] for m in keep}, {m: low[m] for m in keep},
              training=False, return_selection_info=False)
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def acc_with(model, batches, keep):
    cor = n = 0
    for high, low, y in batches:
        p = _fwd(model, high, low, keep).argmax(-1).cpu()
        cor += int((p == y).sum()); n += y.shape[0]
    return cor / n


@torch.no_grad()
def prefix_logits(model, batches, order):
    by, labels = [], None
    for k in range(1, M + 1):
        outs, ys = [], []
        for high, low, y in batches:
            outs.append(_fwd(model, high, low, order[:k]).cpu()); ys.append(y)
        by.append(torch.cat(outs))
        labels = torch.cat(ys).numpy() if labels is None else labels
    L = torch.stack(by, 1); P = F.softmax(L, -1)
    Hs = -(P * P.clamp_min(1e-9).log()).sum(-1).numpy()
    msp = P.max(-1).values.numpy()                    # [N, M] max-softmax per prefix
    return labels, Hs, msp, L.argmax(-1).numpy()


def simulate(labels, Hs, preds, tau):
    N = len(labels); ek = np.full(N, M)
    for k in range(M):
        ek = np.where((ek == M) & (Hs[:, k] <= tau), k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    return float((pe == labels).mean()), float(f1_score(labels, pe, average='macro')), float(ek.mean())


# ---- continuous patience (PABEE-style): score = run_length + maxprob, exit when >= tau;
#      at integer tau=p this is exactly patience-p (same method as compute_exit_schemes_4ds.py) ----
TAUS_PAT = np.linspace(1.0, M + 1.0, 30)


def run_length(preds):
    N = preds.shape[0]; r = np.ones((N, M), dtype=int)
    for k in range(1, M):
        r[:, k] = np.where(preds[:, k] == preds[:, k - 1], r[:, k - 1] + 1, 1)
    return r


def patience_sim(preds, msp, labels, tau):
    score = run_length(preds) + msp; N = len(labels); ek = np.full(N, M)
    for k in range(M):
        ek = np.where((ek == M) & (score[:, k] >= tau), k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    return float((pe == labels).mean()), float(f1_score(labels, pe, average='macro')), float(ek.mean())


perfold = {t: {'acc': [], 'f1': [], 'mk': []} for t in TAUS}
perfold_pat = {i: {'acc': [], 'f1': [], 'mk': []} for i in range(len(TAUS_PAT))}
borda = {m: 0.0 for m in MODS}; orders = []
print('=== seqA_prefixsup EAV val-LOO modality order (strong-first) ===')
for fold in FOLDS:
    _, val_loader, test_loader = get_dataloader(
        batch_size=64, num_workers=2, modalities=MODS, split_id=fold, max_seq_len=T,
        time_compression_ratio=_C['time_compression_ratio'], split_mode='cross_subject')
    val_b, test_b = preload(val_loader), preload(test_loader)
    model = build()
    ckpt = os.path.join(SEQA_DIR, f'best_model_loo_fold{fold}.pth')
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    inner = {k[6:]: v for k, v in sd.items() if k.startswith('model.')}
    miss, unexp = model.load_state_dict(inner, strict=False)
    miss = [k for k in miss if 'singleton' not in k]
    if miss:
        print(f'  [warn] fold{fold} missing keys: {miss[:5]} ...')
    model.eval()
    full = acc_with(model, val_b, MODS)
    loo = {m: full - acc_with(model, val_b, [x for x in MODS if x != m]) for m in MODS}
    order = sorted(MODS, key=lambda m: -loo[m]); orders.append(order)
    for r, m in enumerate(order):
        borda[m] += (M - r)
    print(f'  fold{fold} full_val={full:.4f}  LOO: ' + ', '.join(f'{m}={loo[m]:+.3f}' for m in order))
    print(f'          order: {order}')
    labels, Hs, msp, preds = prefix_logits(model, test_b, order)
    for t in TAUS:
        a, f, mk = simulate(labels, Hs, preds, t)
        perfold[t]['acc'].append(a); perfold[t]['f1'].append(f); perfold[t]['mk'].append(mk)
    for i, t in enumerate(TAUS_PAT):
        a, f, mk = patience_sim(preds, msp, labels, t)
        perfold_pat[i]['acc'].append(a); perfold_pat[i]['f1'].append(f); perfold_pat[i]['mk'].append(mk)

consensus = sorted(MODS, key=lambda m: -borda[m])
print(f'  >>> CONSENSUS val-LOO order (strong->weak): {consensus}')
exit_curve = sorted([(float(np.mean(perfold[t]['mk'])), float(np.mean(perfold[t]['acc'])),
                      float(np.mean(perfold[t]['f1']))) for t in TAUS])
exit_curve_patience = sorted([(float(np.mean(perfold_pat[i]['mk'])), float(np.mean(perfold_pat[i]['acc'])),
                               float(np.mean(perfold_pat[i]['f1']))) for i in range(len(TAUS_PAT))])

# per-K modality GFLOPs at T=128 (consensus strong-first subset). FlopCounterMode only
# (matches compute_mosei_exit.py methodology for the seqA curve).
h1 = {m: test_b[0][0][m][:1] for m in MODS}; l1 = {m: test_b[0][1][m][:1] for m in MODS}
mm = build().eval(); gpk = []
for k in range(1, M + 1):
    fcm = FlopCounterMode(display=False)
    with fcm, torch.no_grad():
        _fwd(mm, h1, l1, consensus[:k])
    gpk.append(fcm.get_total_flops() / 1e9)

out = {'eav': {'M': M, 'exit_curve': exit_curve, 'exit_curve_patience': exit_curve_patience,
               'gflops_per_k': gpk, 'consensus_order': consensus, 'per_fold_order': orders}}
os.makedirs(os.path.join(_HERE, 'eav'), exist_ok=True)
json.dump(out, open(os.path.join(_HERE, 'eav', 'exit_eav.json'), 'w'), indent=2)
print(f"EAV @T128: exit acc {exit_curve[0][1]:.3f}->{exit_curve[-1][1]:.3f} "
      f"@ meanK {exit_curve[0][0]:.2f}->{exit_curve[-1][0]:.2f} | gflops_per_k={[round(g, 4) for g in gpk]}")
print('saved -> results_3p3/eav/exit_eav.json')
