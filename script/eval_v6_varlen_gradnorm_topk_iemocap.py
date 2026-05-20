"""Top-K modality eval on IEMOCAP driven by per-modality gradient norms
recorded during ``main_v6_iemocap_fusion_only_dyna.py`` training.

For each K in ``--ks``:
  * gradnorm top-K: keep the K modalities with largest ``final_grad_norms``;
                    evaluate test accuracy on the frozen fusion forward.
  * random K (n seeds): reference baseline.

Mirrors ``eval_v6_dyna_gradnorm_topk_dsads.py`` but uses ``IEMOCAPDataset``
(returns 12 per-modality tensors + label).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed
from data.IEMOCAP.get_data import (
    IEMOCAPDataset, FEATURE_DIMS, NUM_CLASSES, collate_fn,
)
from multimodal_model.v6_downsample_orig import (
    DualVideoBottleneckModelV6Downsample,
)


class _IEMOCAPV6Cfg:
    def __init__(self, modalities):
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        v_dim = FEATURE_DIMS.get('video', 1024)
        self.video_high_dim = v_dim
        self.video_low_dim = v_dim


def _build_v6(args, modalities, input_length):
    cfg = _IEMOCAPV6Cfg(modalities)
    return DualVideoBottleneckModelV6Downsample(
        cfg=cfg, output_dim=NUM_CLASSES, input_length=input_length,
        d_model=args.d_model, nhead=args.nhead,
        num_layers_per_modal=args.num_layers_per_modal,
        num_layers=args.num_layers, dropout=args.dropout, verbose=False,
        video_low_dim=cfg.video_low_dim, video_high_dim=cfg.video_high_dim,
        use_bottleneck=True, n_bottlenecks=args.n_bottlenecks,
        fusion_layer=args.fusion_layer, use_sparse_moe=False,
        num_experts=args.num_experts, expert_k=1,
        internal_dim=args.internal_dim, bottleneck_head_pos=True,
        use_sparse_attn=args.use_sparse_attn, factor=args.base_factor,
        selector_video_source='high', encoder_video_source='high',
        no_selector=True, use_weighted_factor=False, use_triton=False,
        num_classes=NUM_CLASSES,
        sparse_attn_variant=args.sparse_attn_variant,
        strat_block_size=args.strat_block_size,
        downsample_min_len=args.downsample_min_len,
        use_batched_fusion=False,
    )


def _unpack(batch, modalities, device):
    *feats, label = batch
    high, low = {}, {}
    for j, m in enumerate(modalities):
        high[m] = feats[2 * j].to(device).float()
        low[m] = feats[2 * j + 1].to(device).float()
    y = label.squeeze(-1).long().to(device)
    return high, low, y


@torch.no_grad()
def _eval_at_subset(model, loader, modalities, kept, device):
    model.eval()
    all_pred, all_y = [], []
    for batch in loader:
        high, low, y = _unpack(batch, modalities, device)
        h = {m: high[m] for m in kept}
        l = {m: low[m] for m in kept}
        out = model(h, l, training=False, return_selection_info=False)
        logits = out[0] if isinstance(out, tuple) else out
        all_pred.append(logits.argmax(1).cpu())
        all_y.append(y.cpu())
    y_arr = torch.cat(all_y).numpy()
    p_arr = torch.cat(all_pred).numpy()
    return float(accuracy_score(y_arr, p_arr)), float(f1_score(y_arr, p_arr, average='macro'))


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in ('y', 'yes', 't', 'true', '1'):
        return True
    if s in ('n', 'no', 'f', 'false', '0', ''):
        return False
    raise argparse.ArgumentTypeError(f'Boolean expected, got {v!r}')


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--results_json', required=True,
                    help='Per-fold results.json with final_grad_norms dict.')
parser.add_argument('--rank_key', default='final_grad_norms',
                    help='Top-level key in results_json holding the per-mod '
                         'ranking signal.  Default ``final_grad_norms``.  '
                         'For CGGM aux-acc ranking pass ``final_aux_accs``.')
parser.add_argument('--fold', default=0, type=int, choices=[0, 1, 2])
parser.add_argument('--cuda_pick', default='cuda:0')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_seq_len', default=384, type=int)
parser.add_argument('--time_compression_ratio', default=4, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--data_root', default='/files1/haodong/data/IEMOCAP')
parser.add_argument('--ks', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6])
parser.add_argument('--n_random_seeds', default=5, type=int)
parser.add_argument('--output_json', default=None)

# V6 (must match dyna training)
parser.add_argument('--d_model', default=256, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--internal_dim', default=256, type=int)
parser.add_argument('--n_bottlenecks', default=8, type=int)
parser.add_argument('--fusion_layer', default=0, type=int)
parser.add_argument('--use_sparse_attn', default=True, type=_str2bool)
parser.add_argument('--sparse_attn_variant', default='opt', type=str)
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--num_experts', default=4, type=int)

args = parser.parse_args()

device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed_num)
modalities = list(IEMOCAPDataset.ALL_MODALITIES)
input_length = args.max_seq_len if args.max_seq_len > 0 else 600

# Test dataloader
suffix = '' if args.fold == 0 else f'_split{args.fold}'
test_csv = os.path.join(args.data_root, f'test{suffix}.csv')
ds = IEMOCAPDataset(
    data_root=args.data_root, csv_path=test_csv, modalities=modalities,
    max_seq_len=args.max_seq_len, time_compression_ratio=args.time_compression_ratio,
    use_batched_fusion=False)  # varlen: keep native modality lengths
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, drop_last=False,
                    collate_fn=collate_fn)
print(f'Device={device}  Fold={args.fold}  N_test={len(ds)}')

# Build & load V6 (strip "model." prefix from wrapper-saved ckpt)
model = _build_v6(args, modalities, input_length)
state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
miss, unexp = model.load_state_dict(state, strict=False)
print(f'Loaded {args.ckpt}  missing={len(miss)}  unexpected={len(unexp)}')
model.eval().to(device)

# grad norm ranking
gn = json.load(open(args.results_json)).get(args.rank_key, {})
if not gn:
    raise ValueError(f'No entries under "{args.rank_key}" in {args.results_json}')
gn_vec = np.array([gn[m] for m in modalities], dtype=np.float32)
gn_rank = np.argsort(-gn_vec).tolist()
print(f'Ranking by {args.rank_key} (high → low):')
for j in gn_rank:
    print(f'  {modalities[j]:<14}  {gn_vec[j]:.4f}')

rng = np.random.default_rng(args.seed_num)
per_k_results = {}

print('\n' + '=' * 80)
for K in args.ks:
    # gradnorm top-K subset
    gn_subset = sorted(modalities[j] for j in gn_rank[:K])
    gn_acc, gn_f1 = _eval_at_subset(model, loader, modalities, gn_subset, device)

    # random K (n seeds)
    rand_accs, rand_f1s = [], []
    for s in range(args.n_random_seeds):
        gen = np.random.default_rng(args.seed_num + s)
        idx = gen.choice(len(modalities), K, replace=False)
        sub = sorted([modalities[j] for j in idx])
        a, f = _eval_at_subset(model, loader, modalities, sub, device)
        rand_accs.append(a); rand_f1s.append(f)
    rand_acc_mean = float(np.mean(rand_accs)); rand_acc_std = float(np.std(rand_accs, ddof=1))
    rand_f1_mean = float(np.mean(rand_f1s)); rand_f1_std = float(np.std(rand_f1s, ddof=1))

    print(f'K={K}  gradnorm top-K ({gn_subset}): acc={gn_acc:.4f}  f1={gn_f1:.4f}')
    print(f'      random-{K} ({args.n_random_seeds} seeds): '
          f'acc={rand_acc_mean:.4f}±{rand_acc_std:.4f}  '
          f'f1={rand_f1_mean:.4f}±{rand_f1_std:.4f}')
    per_k_results[str(K)] = {
        'K': K,
        'gradnorm': {'kept': list(gn_subset), 'acc': gn_acc, 'f1': gn_f1},
        'random_k': {
            'acc_mean': rand_acc_mean, 'acc_std': rand_acc_std,
            'f1_mean': rand_f1_mean, 'f1_std': rand_f1_std,
            'n_seeds': args.n_random_seeds,
        },
    }

out = {
    'ckpt': args.ckpt, 'fold': args.fold,
    'modalities': modalities, 'ks': args.ks,
    'grad_norms': {m: float(gn_vec[j]) for j, m in enumerate(modalities)},
    'ranked_modalities': [modalities[j] for j in gn_rank],
    'n_random_seeds': args.n_random_seeds,
    'results': per_k_results,
}
if args.output_json:
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote: {args.output_json}')
