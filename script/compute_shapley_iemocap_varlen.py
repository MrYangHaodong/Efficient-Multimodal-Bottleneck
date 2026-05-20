"""Compute per-sample Shapley modality importance on a frozen IEMOCAP
V6 fusion checkpoint.  IEMOCAP sibling of ``compute_shapley_daliahar.py``
and ``compute_shapley_dsads.py``.

Key differences:
  - 6 modalities (video / audio / text / mocap_hand/head/rotated) ->
    2^6 = 64 subsets per batch.
  - ``IEMOCAPDataset`` returns a list of (high, low) tensor pairs +
    label rather than a single concatenated tensor.  We forward V6
    by feeding the high dicts directly (no channel-split required).
  - ``--split`` selects train / val / test from the per-split CSV.

Run:
    python script/compute_shapley_iemocap.py --fold=0 --cuda_pick=cuda:0 \\
        --ckpt=./model_chkpt/IEMOCAP/multimodal_model/2026-05-17_IEMOCAP_v6_fusion_only_dyna/best_model_fold0.pth \\
        --split=train --save_phi \\
        --output_json=./model_chkpt/IEMOCAP/multimodal_model/2026-05-17_IEMOCAP_v6_fusion_only_dyna/shapley_train_fold0.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from itertools import combinations
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


# --------------------------------------------------------------------- #
#                              Helpers                                  #
# --------------------------------------------------------------------- #


class _IEMOCAPV6Cfg:
    def __init__(self, modalities):
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        v_dim = FEATURE_DIMS.get('video', 1024)
        self.video_high_dim = v_dim
        self.video_low_dim = v_dim


def _build_v6_model(args, modalities, input_length):
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


def _unpack_batch(batch, modalities, device):
    """IEMOCAPDataset collate gives (h0, l0, h1, l1, ..., label[B,1]).
    Return high/low dicts of all-modality features + label.

    We don't channel-split (already per-modality from the dataset).
    """
    *feats, label = batch
    assert len(feats) == 2 * len(modalities)
    high, low = {}, {}
    for j, m in enumerate(modalities):
        high[m] = feats[2 * j].to(device).float()
        low[m] = feats[2 * j + 1].to(device).float()
    y = label.squeeze(-1).long().to(device)
    return high, low, y


@torch.no_grad()
def _forward_subset(model, high, low, subset, modalities):
    keep = [modalities[j] for j in subset]
    high_S = {m: high[m] for m in keep}
    low_S = {m: low[m] for m in keep}
    out = model(high_S, low_S, training=False, return_selection_info=False)
    return out[0] if isinstance(out, tuple) else out


def _shapley_weights(p):
    fact = [math.factorial(i) for i in range(p + 1)]
    return {s: fact[s] * fact[p - s - 1] / fact[p] for s in range(p)}


def _enumerate_subsets(p):
    return [tuple(s) for r in range(p + 1) for s in combinations(range(p), r)]


@torch.no_grad()
def exact_shapley_batch(model, high, low, y, modalities, num_classes):
    p = len(modalities)
    B = y.shape[0]
    device = y.device
    subsets = _enumerate_subsets(p)
    subset_to_idx = {S: i for i, S in enumerate(subsets)}
    v = torch.zeros(B, len(subsets), device=device)
    baseline_loss = math.log(num_classes)
    for k, S in enumerate(subsets):
        if len(S) == 0:
            continue
        logits = _forward_subset(model, high, low, S, modalities)
        loss_S = F.cross_entropy(logits, y, reduction='none')
        v[:, k] = baseline_loss - loss_S
    weights = _shapley_weights(p)
    Phi = torch.zeros(B, p, device=device)
    for j in range(p):
        for k, S in enumerate(subsets):
            if j in S:
                continue
            k_with = subset_to_idx[tuple(sorted(S + (j,)))]
            Phi[:, j] += weights[len(S)] * (v[:, k_with] - v[:, k])
    return Phi


# --------------------------------------------------------------------- #
#                                CLI                                    #
# --------------------------------------------------------------------- #


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
parser.add_argument('--ckpt', required=True, type=str)
parser.add_argument('--fold', default=0, type=int, choices=[0, 1, 2])
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--data_root', default='/files1/haodong/data/IEMOCAP', type=str)
parser.add_argument('--split', default='train', choices=['train', 'val', 'eval'])
parser.add_argument('--max_seq_len', default=384, type=int)
parser.add_argument('--time_compression_ratio', default=4, type=int)
parser.add_argument('--num_workers', default=2, type=int)

parser.add_argument('--output_json', default=None, type=str)
parser.add_argument('--save_phi', action='store_true', default=False)
parser.add_argument('--grad_norms_json', default=None, type=str)

# Modalities
parser.add_argument('--modalities', nargs='+',
                    default=list(IEMOCAPDataset.ALL_MODALITIES),
                    choices=list(IEMOCAPDataset.ALL_MODALITIES))

# V6 backbone (must match training)
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

modalities = list(args.modalities)
p = len(modalities)
num_classes = NUM_CLASSES
input_length = args.max_seq_len if args.max_seq_len > 0 else 600

print(f'Device: {device}  fold={args.fold}  split={args.split}')
print(f'Modalities ({p}): {modalities}')
print(f'num_classes={num_classes}  input_length={input_length}')

# Build dataset directly (skip get_dataloader so we control shuffle order)
suffix = '' if args.fold == 0 else f'_split{args.fold}'
csv_name = {'train': 'train', 'val': 'val', 'eval': 'test'}[args.split]
csv_path = os.path.join(args.data_root, f'{csv_name}{suffix}.csv')
print(f'CSV: {csv_path}')

ds = IEMOCAPDataset(
    data_root=args.data_root, csv_path=csv_path,
    modalities=modalities, max_seq_len=args.max_seq_len,
    time_compression_ratio=args.time_compression_ratio,
    use_batched_fusion=False,  # varlen: keep native modality lengths
)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, drop_last=False,
                    collate_fn=collate_fn)
print(f'N = {len(ds)} samples')

# Build & load
model = _build_v6_model(args, modalities, input_length)
state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
# Wrapper saves with "model." prefix → strip
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k, v in state.items()
             if k.startswith('model.')}
missing, unexpected = model.load_state_dict(state, strict=False)
n_missing_nonselector = len([k for k in missing if 'modality_selector' not in k])
print(f'Loaded ckpt: missing={len(missing)} (non-selector={n_missing_nonselector}) '
      f'unexpected={len(unexpected)}')
if unexpected:
    print(f'  first 3 unexpected: {unexpected[:3]}')
model.eval().to(device)

# Forward all subsets per batch
all_phi = []
all_labels = []
t0 = time.time()
print(f'\nComputing Shapley over {len(loader)} batches ('
      f'{2**p}={len(_enumerate_subsets(p))} subsets per batch)...')
for i, batch in enumerate(loader):
    high, low, y = _unpack_batch(batch, modalities, device)
    Phi = exact_shapley_batch(model, high, low, y, modalities, num_classes)
    all_phi.append(Phi.cpu())
    all_labels.append(y.cpu())
    if i % 5 == 0 or i == len(loader) - 1:
        print(f'  batch {i+1:>3}/{len(loader)}  elapsed={time.time()-t0:6.1f}s')

Phi = torch.cat(all_phi, dim=0).numpy()
labels = torch.cat(all_labels, dim=0).numpy()
N = Phi.shape[0]
print(f'\nShapley matrix:  {Phi.shape}')

# Per-modality summary
mean_phi = Phi.mean(axis=0)
std_phi = Phi.std(axis=0, ddof=1)
print('\nPer-modality mean Shapley:')
for j, m in enumerate(modalities):
    print(f'  {m:<14}  mean={mean_phi[j]:+.4f}  std={std_phi[j]:.4f}')

top1 = np.argmax(Phi, axis=1)
top1_counts = np.bincount(top1, minlength=p)
print('\nPer-sample top-1 hist:')
for j, m in enumerate(modalities):
    print(f'  {m:<14}  count={top1_counts[j]:>4}  '
          f'frac={top1_counts[j]/N:.3f}')

mean_argmax = int(mean_phi.argmax())
print(f'\nper-sample top-1 matches population top-1: '
      f'{(top1 == mean_argmax).mean()*100:.1f}%')

# Save
out = {
    'ckpt': args.ckpt, 'fold': args.fold, 'split': args.split,
    'estimator': 'exact', 'n_samples': int(N),
    'modalities': modalities,
    'per_modality': [
        {'modality': m, 'mean_shapley': float(mean_phi[j]),
         'std_shapley': float(std_phi[j])}
        for j, m in enumerate(modalities)
    ],
    'shapley_ranking': [
        modalities[j] for j in np.argsort(-mean_phi).tolist()
    ],
    'top1_hist': {m: float(top1_counts[j] / N)
                  for j, m in enumerate(modalities)},
}
if args.output_json:
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote summary to {args.output_json}')

if args.save_phi:
    npz_path = (args.output_json.replace('.json', '_phi.npz')
                if args.output_json else
                f'shapley_phi_iemocap_fold{args.fold}_{args.split}.npz')
    np.savez(npz_path, phi=Phi, labels=labels, modalities=np.array(modalities))
    print(f'Wrote per-sample Phi to {npz_path}')

print(f'\nDone in {time.time()-t0:.1f}s.')
