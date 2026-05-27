"""Compute per-sample Shapley value for ARBITRARY MODALITY SUBSETS on IEMOCAP V6.

Extends the standard per-modality Shapley (compute_shapley_iemocap.py) to every
non-empty subset T ⊆ M.  φ_T treats T as a single "super-player" coalescing with
the |M\T| remaining individual modalities; for singleton T this reduces exactly
to the standard Shapley value.

Outputs (per fold × split):
  * `v_subsets` shape (N, 2^M)    — value v(S) for all subsets S, indexed by bitmask
  * `phi_subsets` shape (N, 2^M-1)— Shapley value φ_T for all non-empty subsets T,
                                    indexed by (bitmask - 1)
  * `subset_index` list[tuple] of modality-index tuples (one per bitmask)
  * `labels`, `modalities`

Value function (same as singleton Shapley):
  v(S; x) = log(num_classes) − CE(logits_S(x), y),  v(∅) = 0
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
from typing import List

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
from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)


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
        per_modal_distill=False,
        per_modal_downsample_min_len=4,
        bottleneck_init_mode='random',
    )


def _unpack_iemocap_batch(batch, modalities, device):
    feats, label = batch[:-1], batch[-1]
    high, low = {}, {}
    for j, m in enumerate(modalities):
        high[m] = feats[2 * j].to(device).float()
        low[m] = feats[2 * j + 1].to(device).float()
    y = label.squeeze(-1).long().to(device)
    return high, low, y


@torch.no_grad()
def _forward_subset(model, high, low, subset_idx, modalities):
    keep = [modalities[j] for j in subset_idx]
    high_S = {m: high[m] for m in keep}
    low_S = {m: low[m] for m in keep}
    out = model(high_S, low_S, training=False, return_selection_info=False)
    return out[0] if isinstance(out, tuple) else out


def _bitmask_to_tuple(b, p):
    return tuple(j for j in range(p) if (b >> j) & 1)


@torch.no_grad()
def exact_subset_shapley_batch(model, high, low, y, modalities, num_classes):
    """Returns per-sample:
        v: (B, 2^p)     — value of every subset by bitmask
        phi: (B, 2^p-1) — Shapley value of every NON-EMPTY subset T (indexed by bitmask-1)
        preds: (B, 2^p) int8 — argmax(logits) per subset, -1 for empty subset
    """
    p = len(modalities)
    P = 1 << p   # 2^p
    B = y.shape[0]
    device = y.device
    baseline_loss = math.log(num_classes)

    # ---- 1. evaluate v(S) for every subset (bitmask order, empty = 0) ----
    v = torch.zeros(B, P, device=device)
    preds = torch.full((B, P), -1, dtype=torch.int8, device=device)
    for b in range(1, P):
        S = _bitmask_to_tuple(b, p)
        logits = _forward_subset(model, high, low, S, modalities)
        v[:, b] = baseline_loss - F.cross_entropy(logits, y, reduction='none')
        preds[:, b] = logits.argmax(dim=-1).to(torch.int8)

    # ---- 2. compute φ_T for every non-empty T (group-Shapley) ----
    # φ_T = Σ_{S: S ∩ T = ∅} w(|S|, p-|T|) · [v(S∪T) - v(S)]
    # w(s, k) = (s! · (k-s)!) / (k+1)!  — Shapley weight treating T as 1 super-player
    fact = [math.factorial(i) for i in range(p + 2)]
    phi = torch.zeros(B, P - 1, device=device)

    # cache bit count
    popcount = torch.tensor([bin(b).count('1') for b in range(P)], dtype=torch.long)

    for T_mask in range(1, P):
        t_size = bin(T_mask).count('1')
        k = p - t_size               # |M\T|
        T_complement = ((P - 1) ^ T_mask)
        # enumerate subsets of T_complement
        # iterate using submask trick
        S = T_complement
        # gather all submasks (including 0) by walking
        submasks = []
        sm = T_complement
        while True:
            submasks.append(sm)
            if sm == 0: break
            sm = (sm - 1) & T_complement
        # weights
        for sm in submasks:
            s_size = bin(sm).count('1')
            w = fact[s_size] * fact[k - s_size] / fact[k + 1]
            phi[:, T_mask - 1] += w * (v[:, sm | T_mask] - v[:, sm])

    return v, phi, preds


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in ('y','yes','t','true','1'): return True
    if s in ('n','no','f','false','0',''): return False
    raise argparse.ArgumentTypeError(f'Boolean expected, got {v!r}')


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, type=str)
parser.add_argument('--fold', default=0, type=int, choices=[0, 1, 2])
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--data_root', default='/files1/haodong/data/IEMOCAP', type=str)
parser.add_argument('--split', default='train', choices=['train','val','eval'])
parser.add_argument('--max_seq_len', default=128, type=int)
parser.add_argument('--time_compression_ratio', default=4, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--output_npz', required=True, type=str,
                    help='Output .npz path for v_subsets, phi_subsets, labels, modalities, subset_index')
parser.add_argument('--modalities', nargs='+',
                    default=list(IEMOCAPDataset.ALL_MODALITIES),
                    choices=list(IEMOCAPDataset.ALL_MODALITIES))
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--internal_dim', default=128, type=int)
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
input_length = args.max_seq_len if args.max_seq_len > 0 else 600

print(f"[subset-Shapley] fold={args.fold} split={args.split} modalities={modalities} (p={p})")
print(f"[subset-Shapley] enumerating 2^p = {1 << p} subsets per sample")

model = _build_v6_model(args, modalities, input_length).to(device)
state = torch.load(args.ckpt, map_location=device)
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
clean = { (k[len('model.'):] if k.startswith('model.') else k): v for k, v in state.items() }
missing, unexpected = model.load_state_dict(clean, strict=False)
if missing:
    print(f"  [warn] missing keys (first 5): {missing[:5]}")
if unexpected:
    print(f"  [warn] unexpected keys (first 5): {unexpected[:5]}")
model.eval()

suffix = '' if args.fold == 0 else f'_split{args.fold}'
csv_name = {'train': 'train', 'val': 'val', 'eval': 'test'}[args.split]
csv_path = os.path.join(args.data_root, f'{csv_name}{suffix}.csv')
print(f"[subset-Shapley] CSV: {csv_path}")

ds = IEMOCAPDataset(
    data_root=args.data_root, csv_path=csv_path,
    modalities=modalities, max_seq_len=args.max_seq_len,
    time_compression_ratio=args.time_compression_ratio,
    use_batched_fusion=True,
)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, drop_last=False,
                    collate_fn=collate_fn)
print(f"[subset-Shapley] dataset size N={len(ds)}")

all_v, all_phi, all_y, all_preds = [], [], [], []
t0 = time.perf_counter()
for bi, batch in enumerate(loader):
    high, low, y = _unpack_iemocap_batch(batch, modalities, device)
    v, phi, preds = exact_subset_shapley_batch(model, high, low, y, modalities, NUM_CLASSES)
    all_v.append(v.cpu().numpy().astype(np.float32))
    all_phi.append(phi.cpu().numpy().astype(np.float32))
    all_y.append(y.cpu().numpy().astype(np.int64))
    all_preds.append(preds.cpu().numpy().astype(np.int8))
    if bi % 20 == 0:
        print(f"  [{bi+1}/{len(loader)}] elapsed={time.perf_counter()-t0:.1f}s", flush=True)

v_all = np.concatenate(all_v, axis=0)
phi_all = np.concatenate(all_phi, axis=0)
y_all = np.concatenate(all_y, axis=0)
preds_all = np.concatenate(all_preds, axis=0)

# subset_index: list of tuples (modality indices) per bitmask
subset_index = np.empty(1 << p, dtype=object)
for b in range(1 << p):
    subset_index[b] = tuple(j for j in range(p) if (b >> j) & 1)

os.makedirs(os.path.dirname(args.output_npz) or '.', exist_ok=True)
np.savez_compressed(
    args.output_npz,
    v_subsets=v_all,           # (N, 2^p)
    phi_subsets=phi_all,       # (N, 2^p - 1)  T_mask = idx + 1
    preds_subsets=preds_all,   # (N, 2^p) int8, argmax(logits) per subset; -1 for empty
    labels=y_all,
    modalities=np.array(modalities, dtype='<U16'),
    subset_index=subset_index, # subset_index[b] = tuple of modality indices in bitmask b
)
print(f"[subset-Shapley] saved → {args.output_npz}")
print(f"  v_subsets shape:   {v_all.shape}")
print(f"  phi_subsets shape: {phi_all.shape}")
print(f"  preds_subsets:     {preds_all.shape}")
print(f"  N={len(y_all)}  elapsed={time.perf_counter()-t0:.1f}s")
