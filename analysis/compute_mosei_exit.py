"""seqA CMU-MOSEI adaptive-exit curve + per-K modality GFLOPs (val-LOO modality
order, strong-first; same method as compute_dsads_exit.py).

MOSEI: M=3 (vision/audio/text), FLOAT input T=50, binary Acc-2. The 3 training
seeds play the role of folds. Uses the deepcopy-fixed seqA checkpoints
(best_model_cold_val.pth -> wrapper state, 'model.' prefix stripped).

Output -> results_3p3/mosei/exit_mosei.json (keys mirror exit_dsads.json).
Run from clean_models/train/:  python ../results_3p3/compute_mosei_exit.py
"""
from __future__ import annotations
import glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_CM = os.path.dirname(_HERE)                      # clean_models
sys.path.insert(0, _CM)
from utils.helper_function import set_seed                                    # noqa: E402
from data.CMU_MOSEI.get_data import get_dataloader, CMUMOSEIDataset            # noqa: E402
from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import (  # noqa: E402
    DualVideoBottleneckModelV6Downsample)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
set_seed(42)
TAUS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 10.0]
MODS = list(CMUMOSEIDataset.ALL_MODALITIES)       # ['vision','audio','text']
M = len(MODS); NUMC = 2; T = 50
SEEDS = [42, 2025]            # best-2 of 3 seqA seeds (drop early-stopped s1337)
RES = os.path.join(_CM, 'train', 'results_mosei_3seed')


def seqa_dir(seed):
    ds = sorted(glob.glob(os.path.join(RES, f'2026-*_seqA_lr2e5e30_acc2_s{seed}_seqA')))
    return ds[-1]


_C0 = json.load(open(os.path.join(seqa_dir(42), 'config.json')))
VAR = {m: _C0['feature_dims'][m] for m in MODS}


class _V6Cfg:
    def __init__(s):
        s.modalities = list(MODS); s.variates = dict(VAR)
        f = MODS[0]; s.video_high_dim = VAR[f]; s.video_low_dim = VAR[f]


def build(C):
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=_V6Cfg(), output_dim=NUMC, input_length=T,
        d_model=C['d_model'], nhead=C['nhead'],
        num_layers_per_modal=C['num_layers_per_modal'], num_layers=C['num_layers'],
        dropout=C['dropout'], verbose=False,
        video_low_dim=VAR[MODS[0]], video_high_dim=VAR[MODS[0]], use_bottleneck=True,
        n_bottlenecks=C['n_bottlenecks'], fusion_layer=C['fusion_layer'], use_sparse_moe=False,
        num_experts=C['num_experts'], expert_k=1, internal_dim=C['internal_dim'],
        bottleneck_head_pos=True, use_sparse_attn=C['use_sparse_attn'], factor=C['base_factor'],
        selector_video_source='high', encoder_video_source='high', no_selector=True,
        use_weighted_factor=False, use_triton=False, num_classes=NUMC,
        sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
        downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
        per_modal_distill=C['per_modal_distill'],
        per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
        fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full',
        fusion_mlp_ratio=1.0, fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'],
        seq_modality_order=C['seq_modality_order'], seq_random_order=C['seq_random_order'],
        bottleneck_agg_mode='gate', n_fusion_distill=-1).to(dev).float()


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
    return labels, Hs, L.argmax(-1).numpy()


def simulate(labels, Hs, preds, tau):
    N = len(labels); ek = np.full(N, M)
    for k in range(M):
        ek = np.where((ek == M) & (Hs[:, k] <= tau), k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    return float((pe == labels).mean()), float(f1_score(labels, pe, average='macro')), float(ek.mean())


perfold = {t: {'acc': [], 'f1': [], 'mk': []} for t in TAUS}
borda = {m: 0.0 for m in MODS}; orders = []
print('=== seqA CMU-MOSEI val-LOO modality order (strong-first; drives early-exit prefix) ===')
for seed in SEEDS:
    d = seqa_dir(seed)
    C = json.load(open(os.path.join(d, 'config.json')))
    _, val_loader, test_loader = get_dataloader(batch_size=128, num_workers=2,
                                                modalities=MODS, split_id=0, max_seq_len=T,
                                                time_compression_ratio=4, task='classification2')
    val_b, test_b = preload(val_loader), preload(test_loader)
    model = build(C)
    sd = torch.load(os.path.join(d, 'best_model_cold_val.pth'), map_location='cpu', weights_only=True)
    model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False)
    model.eval()
    full = acc_with(model, val_b, MODS)
    loo = {m: full - acc_with(model, val_b, [x for x in MODS if x != m]) for m in MODS}
    order = sorted(MODS, key=lambda m: -loo[m]); orders.append(order)
    for r, m in enumerate(order):
        borda[m] += (M - r)
    print(f'  seed{seed} full_val={full:.4f}  LOO: ' + ', '.join(f'{m}={loo[m]:+.3f}' for m in order))
    print(f'          order: {order}')
    labels, Hs, preds = prefix_logits(model, test_b, order)
    for t in TAUS:
        a, f, mk = simulate(labels, Hs, preds, t)
        perfold[t]['acc'].append(a); perfold[t]['f1'].append(f); perfold[t]['mk'].append(mk)
consensus = sorted(MODS, key=lambda m: -borda[m])
print(f'  >>> CONSENSUS val-LOO order (strong->weak): {consensus}')
exit_curve = sorted([(float(np.mean(perfold[t]['mk'])), float(np.mean(perfold[t]['acc'])),
                      float(np.mean(perfold[t]['f1']))) for t in TAUS])

# per-K modality GFLOPs at T=50 (consensus strong-first subset).
h1 = {m: test_b[0][0][m][:1] for m in MODS}; l1 = {m: test_b[0][1][m][:1] for m in MODS}
mm = build(_C0).eval(); gpk = []
for k in range(1, M + 1):
    keep = consensus[:k]
    fcm = FlopCounterMode(display=False)
    with fcm, torch.no_grad():
        _fwd(mm, h1, l1, keep)
    gpk.append(fcm.get_total_flops() / 1e9)

out = {'mosei': {'M': M, 'exit_curve': exit_curve, 'gflops_per_k': gpk,
                 'consensus_order': consensus, 'per_fold_order': orders}}
os.makedirs(os.path.join(_HERE, 'mosei'), exist_ok=True)
json.dump(out, open(os.path.join(_HERE, 'mosei', 'exit_mosei.json'), 'w'), indent=2)
print(f"MOSEI @T50: exit acc {exit_curve[0][1]:.3f}->{exit_curve[-1][1]:.3f} "
      f"@ meanK {exit_curve[0][0]:.2f}->{exit_curve[-1][0]:.2f} | gflops_per_k={[round(g, 4) for g in gpk]}")
print('saved -> results_3p3/mosei/exit_mosei.json')
