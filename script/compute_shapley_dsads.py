"""Compute per-sample Shapley modality importance on the DSADS
fusion-only-dyna checkpoint.  DSADS sibling of
``compute_shapley_daliahar.py``.

Differences from the DaliaHAR variant:
  * 4 scenarios (folds 0..3) instead of 3 subject folds.
  * Data is loaded from pre-saved ``dsad_diversify_dict_scenario{F}_src/trg.pt``
    files; train split is the source data, optionally stratified-split
    80/20 to mirror the dyna training pipeline.
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed
from utils.dataset_cfg import DSADS
from data.dataset_builder import DSADSDataset
from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)


# --------------------------------------------------------------------- #
#                              Helpers                                  #
# --------------------------------------------------------------------- #


class _DSADSV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


def _build_v6_model(args, dataset_cfg, input_length):
    v6_cfg = _DSADSV6Cfg(dataset_cfg)
    return DualVideoBottleneckModelV6Downsample(
        cfg=v6_cfg,
        output_dim=dataset_cfg.num_classes,
        input_length=input_length,
        d_model=args.d_model, nhead=args.nhead,
        num_layers_per_modal=args.num_layers_per_modal,
        num_layers=args.num_layers, dropout=args.dropout, verbose=False,
        video_low_dim=v6_cfg.video_low_dim, video_high_dim=v6_cfg.video_high_dim,
        use_bottleneck=True, n_bottlenecks=args.n_bottlenecks,
        fusion_layer=args.fusion_layer, use_sparse_moe=False,
        num_experts=args.num_experts, expert_k=1,
        internal_dim=args.internal_dim, bottleneck_head_pos=True,
        use_sparse_attn=args.use_sparse_attn, factor=args.base_factor,
        selector_video_source='high', encoder_video_source='high',
        no_selector=True, use_weighted_factor=False,
        use_triton=False,
        num_classes=dataset_cfg.num_classes,
        sparse_attn_variant=args.sparse_attn_variant,
        strat_block_size=args.strat_block_size,
        downsample_min_len=args.downsample_min_len,
        use_batched_fusion=True,
        per_modal_distill=False,
        per_modal_downsample_min_len=args.downsample_min_len,
    )


def _split_x(x, modalities, variates):
    high, low = {}, {}
    offset = 0
    for m in modalities:
        n = variates[m]
        chunk = x[:, :, offset:offset + n]
        high[m] = chunk
        low[m] = chunk
        offset += n
    return high, low


@torch.no_grad()
def _forward_subset(model, high, low, subset, modalities):
    keep = [modalities[j] for j in subset]
    high_S = {m: high[m] for m in keep if m in high}
    low_S = {m: low[m] for m in keep if m in low}
    out = model(high_S, low_S, training=False, return_selection_info=False)
    return out[0] if isinstance(out, tuple) else out


def _shapley_weights(p):
    fact = [math.factorial(i) for i in range(p + 1)]
    return {s: fact[s] * fact[p - s - 1] / fact[p] for s in range(p)}


def _enumerate_subsets(p):
    return [tuple(s) for r in range(p + 1) for s in combinations(range(p), r)]


@torch.no_grad()
def exact_shapley_batch(model, x, y, modalities, variates, num_classes):
    p = len(modalities)
    B = x.shape[0]
    device = x.device
    high, low = _split_x(x, modalities, variates)
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
parser.add_argument('--fold', default=0, type=int, choices=[0, 1, 2, 3])
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--val_split_ratio', default=0.2, type=float)
parser.add_argument('--split', default='train',
                    choices=['train', 'val', 'eval'])
parser.add_argument('--data_root', default='/files1/haodong/data/DSADS/', type=str)

parser.add_argument('--grad_norms_json', default=None, type=str)
parser.add_argument('--output_json', default=None, type=str)
parser.add_argument('--save_phi', action='store_true', default=False)

parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--internal_dim', default=64, type=int)
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

dataset_cfg = DSADS()
modalities = dataset_cfg.modalities
p = len(modalities)
num_classes = dataset_cfg.num_classes
input_length = dataset_cfg.duration * dataset_cfg.base_sample_rate

scenario_idx = args.fold
src_path = os.path.join(args.data_root, f'dsad_diversify_dict_scenario{scenario_idx}_src.pt')
trg_path = os.path.join(args.data_root, f'dsad_diversify_dict_scenario{scenario_idx}_trg.pt')
train_data = torch.load(src_path, weights_only=False)
test_data = torch.load(trg_path, weights_only=False)

tr_x, va_x, tr_y, va_y = train_test_split(
    train_data['samples'], train_data['labels'],
    test_size=args.val_split_ratio, random_state=args.seed_num,
    stratify=train_data['labels'])

if args.split == 'train':
    payload = {'samples': tr_x, 'labels': tr_y}
elif args.split == 'val':
    payload = {'samples': va_x, 'labels': va_y}
else:
    payload = test_data

ds = DSADSDataset(payload, modalities, dataset_cfg, args.transform)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, drop_last=False)
print(f'Device: {device}  Fold(scenario)={args.fold}  split={args.split}  N={len(ds)}')
print(f'Modalities ({p}): {modalities}  num_classes={num_classes}  input_length={input_length}')

model = _build_v6_model(args, dataset_cfg, input_length)
state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k, v in state.items()
             if k.startswith('model.')}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f'Loaded {args.ckpt}  missing={len(missing)}  unexpected={len(unexpected)}')
model.eval().to(device)

all_phi = []
all_labels = []
t0 = time.time()
for i, (x, y) in enumerate(loader):
    x = x.to(device).float()
    y = y.to(device)
    Phi = exact_shapley_batch(model, x, y, modalities, dataset_cfg.variates, num_classes)
    all_phi.append(Phi.cpu())
    all_labels.append(y.cpu())
    if i % 5 == 0 or i == len(loader) - 1:
        print(f'  batch {i+1:>3}/{len(loader)}  elapsed={time.time()-t0:6.1f}s')

Phi = torch.cat(all_phi, dim=0).numpy()
labels = torch.cat(all_labels, dim=0).numpy()
N = Phi.shape[0]
print(f'Shapley matrix:  {Phi.shape}')

mean_phi = Phi.mean(axis=0)
std_phi = Phi.std(axis=0, ddof=1)
per_modality = [{'modality': m, 'mean_shapley': float(mean_phi[j]),
                 'std_shapley': float(std_phi[j])}
                for j, m in enumerate(modalities)]
ranked = sorted(per_modality, key=lambda d: -d['mean_shapley'])
print('Per-modality ranking by mean Shapley:')
for r, e in enumerate(ranked):
    print(f'  rank {r+1}: {e["modality"]:<14}  mean={e["mean_shapley"]:+.4f}  std={e["std_shapley"]:.4f}')

top1 = np.argmax(Phi, axis=1)
top1_counts = np.bincount(top1, minlength=p)
print('Per-sample top-1 hist:')
for j, m in enumerate(modalities):
    print(f'  {m:<14}  count={top1_counts[j]:>4}  frac={top1_counts[j]/N:.3f}')

out = {'ckpt': args.ckpt, 'fold': args.fold, 'split': args.split,
       'estimator': 'exact', 'n_samples': int(N), 'modalities': modalities,
       'per_modality': per_modality,
       'shapley_ranking': [e['modality'] for e in ranked]}
if args.output_json:
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote summary to {args.output_json}')
if args.save_phi:
    npz_path = (args.output_json.replace('.json', '_phi.npz')
                if args.output_json else f'shapley_phi_dsads_fold{args.fold}.npz')
    np.savez(npz_path, phi=Phi, labels=labels, modalities=np.array(modalities))
    print(f'Wrote per-sample Phi to {npz_path}')
print(f'Done in {time.time()-t0:.1f}s.')
