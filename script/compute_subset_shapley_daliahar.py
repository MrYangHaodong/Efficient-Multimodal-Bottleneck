"""Compute per-sample Shapley over ARBITRARY subsets on DaliaHAR V6.

5 modalities → 2^5 = 32 subsets per sample. 3 folds with fixed val_set (S11,S12).

Outputs (per fold × split):
  v_subsets, phi_subsets, preds_subsets, labels, modalities, subset_index
"""
from __future__ import annotations

import argparse, json, math, os, sys, time, warnings
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
from utils.dataset_cfg import DaliaHAR
from data.dataset_builder import HARDataset
from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)


class _DaliaV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


def _build_v6_model(args, dataset_cfg, input_length):
    cfg = _DaliaV6Cfg(dataset_cfg)
    return DualVideoBottleneckModelV6Downsample(
        cfg=cfg, output_dim=dataset_cfg.num_classes,
        input_length=input_length,
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
def _forward_subset(model, high, low, subset_idx, modalities):
    keep = [modalities[j] for j in subset_idx]
    high_S = {m: high[m] for m in keep}
    low_S = {m: low[m] for m in keep}
    out = model(high_S, low_S, training=False, return_selection_info=False)
    return out[0] if isinstance(out, tuple) else out


def _bitmask_to_tuple(b, p):
    return tuple(j for j in range(p) if (b >> j) & 1)


@torch.no_grad()
def exact_subset_shapley_batch(model, x, y, modalities, variates, num_classes):
    p = len(modalities)
    P = 1 << p
    B = y.shape[0]
    device = y.device
    baseline_loss = math.log(num_classes)

    high, low = _split_x(x, modalities, variates)
    v = torch.zeros(B, P, device=device)
    preds = torch.full((B, P), -1, dtype=torch.int8, device=device)
    for b in range(1, P):
        S = _bitmask_to_tuple(b, p)
        logits = _forward_subset(model, high, low, S, modalities)
        v[:, b] = baseline_loss - F.cross_entropy(logits, y, reduction='none')
        preds[:, b] = logits.argmax(dim=-1).to(torch.int8)

    # group-Shapley for every non-empty T
    fact = [math.factorial(i) for i in range(p + 2)]
    phi = torch.zeros(B, P - 1, device=device)
    for T_mask in range(1, P):
        t_size = bin(T_mask).count('1')
        k = p - t_size
        T_complement = ((P - 1) ^ T_mask)
        sm = T_complement
        submasks = []
        while True:
            submasks.append(sm)
            if sm == 0: break
            sm = (sm - 1) & T_complement
        for sm in submasks:
            s_size = bin(sm).count('1')
            w = fact[s_size] * fact[k - s_size] / fact[k + 1]
            phi[:, T_mask - 1] += w * (v[:, sm | T_mask] - v[:, sm])
    return v, phi, preds


def _str2bool(v):
    s = str(v).strip().lower()
    if s in ('y','yes','t','true','1'): return True
    if s in ('n','no','f','false','0',''): return False
    raise argparse.ArgumentTypeError(f'Boolean expected, got {v!r}')


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, type=str)
parser.add_argument('--fold', required=True, type=int, choices=[0, 1, 2])
parser.add_argument('--split', required=True, choices=['train', 'val', 'eval'])
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--data_root', default='/files1/haodong/data/processed_dalia_activity', type=str)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--output_npz', required=True, type=str)
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

dataset_cfg = DaliaHAR()
modalities = list(dataset_cfg.modalities)
variates = dataset_cfg.variates
p = len(modalities)
input_length = ((dataset_cfg.duration * dataset_cfg.base_sample_rate) // 2
                if args.transform == 'sax'
                else dataset_cfg.duration * dataset_cfg.base_sample_rate)
fold_cfg = dataset_cfg.folds[args.fold]
train_subjects, eval_subjects = fold_cfg['train_set'], fold_cfg['eval_set']
val_subjects = dataset_cfg.val_set

if args.split == 'train':   split_subjects = train_subjects
elif args.split == 'val':   split_subjects = val_subjects
else:                       split_subjects = eval_subjects

print(f"[subset-Shapley/DaliaHAR] fold={args.fold} split={args.split} subjects={split_subjects}")
print(f"[subset-Shapley/DaliaHAR] modalities={modalities} (p={p}), 2^p={1<<p} subsets")

ds = HARDataset(args.data_root, modalities, split_subjects, dataset_cfg, args.transform)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, drop_last=False)
print(f"[subset-Shapley/DaliaHAR] N={len(ds)}")

model = _build_v6_model(args, dataset_cfg, input_length).to(device)
state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"[subset-Shapley/DaliaHAR] ckpt: missing={len(missing)} unexpected={len(unexpected)}")
if unexpected: print(f"  first 3 unexpected: {unexpected[:3]}")
model.eval()

all_v, all_phi, all_y, all_preds = [], [], [], []
t0 = time.perf_counter()
for bi, (x, y) in enumerate(loader):
    x = x.to(device).float()
    y = y.to(device).long()
    v, phi, preds = exact_subset_shapley_batch(model, x, y, modalities, variates, dataset_cfg.num_classes)
    all_v.append(v.cpu().numpy().astype(np.float32))
    all_phi.append(phi.cpu().numpy().astype(np.float32))
    all_y.append(y.cpu().numpy().astype(np.int64))
    all_preds.append(preds.cpu().numpy().astype(np.int8))
    if bi % 10 == 0:
        print(f"  [{bi+1}/{len(loader)}] elapsed={time.perf_counter()-t0:.1f}s", flush=True)

v_all = np.concatenate(all_v, axis=0)
phi_all = np.concatenate(all_phi, axis=0)
y_all = np.concatenate(all_y, axis=0)
preds_all = np.concatenate(all_preds, axis=0)
subset_index = np.empty(1 << p, dtype=object)
for b in range(1 << p):
    subset_index[b] = tuple(j for j in range(p) if (b >> j) & 1)

os.makedirs(os.path.dirname(args.output_npz) or '.', exist_ok=True)
np.savez_compressed(args.output_npz,
    v_subsets=v_all, phi_subsets=phi_all, preds_subsets=preds_all,
    labels=y_all, modalities=np.array(modalities, dtype='<U16'),
    subset_index=subset_index)
print(f"[subset-Shapley/DaliaHAR] saved → {args.output_npz}")
print(f"  v_subsets: {v_all.shape}  phi_subsets: {phi_all.shape}  preds_subsets: {preds_all.shape}")
print(f"  N={len(y_all)}  elapsed={time.perf_counter()-t0:.1f}s")
