"""Micro-benchmark inference latency: ShapMML selection vs neural
selector (C3 pinball) on top of frozen V6 fusion.

Both methods:
  1. Take raw input x, run V6 forward, get prediction.
For a fair single-method comparison we time:
  - V6 forward at the selected K-subset (common to both)
  - the SELECTION step itself
The bench reports per-sample wall time for selection.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, '/files1/haodong/shap_mml-B3D1')

from utils.dataset_cfg import DaliaHAR
from data.dataset_builder import HARDataset
from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)
from shap_mml import ShapMML
from torch.utils.data import DataLoader


class _Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


def build_v6(ds_cfg, input_length, selector_downsample_factor=4):
    cfg = _Cfg(ds_cfg)
    return DualVideoBottleneckModelV6Downsample(
        cfg=cfg, output_dim=ds_cfg.num_classes, input_length=input_length,
        d_model=64, nhead=8, num_layers_per_modal=2, num_layers=4, dropout=0.1,
        verbose=False, video_low_dim=cfg.video_low_dim, video_high_dim=cfg.video_high_dim,
        use_bottleneck=True, n_bottlenecks=8, fusion_layer=0, use_sparse_moe=False,
        num_experts=4, expert_k=1, internal_dim=64, bottleneck_head_pos=True,
        use_sparse_attn=True, factor=3, selector_video_source='high',
        encoder_video_source='high', no_selector=False, use_weighted_factor=True,
        use_triton=False, num_classes=ds_cfg.num_classes,
        sparse_attn_variant='opt', strat_block_size=8, downsample_min_len=4,
        use_batched_fusion=True, per_modal_distill=False,
        per_modal_downsample_min_len=4,
        selector_downsample_factor=selector_downsample_factor,
    )


def split_x(x, modalities, variates):
    high, low = {}, {}
    offset = 0
    for m in modalities:
        n = variates[m]
        chunk = x[:, :, offset:offset+n]
        high[m] = chunk; low[m] = chunk
        offset += n
    return high, low


parser = argparse.ArgumentParser()
parser.add_argument('--cuda_pick', default='cuda:3', type=str)
parser.add_argument('--n_samples', default=2048, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--n_warmup', default=10, type=int)
parser.add_argument('--n_runs', default=50, type=int)
args = parser.parse_args()

device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load DaliaHAR fold 0
cfg = DaliaHAR()
modalities = cfg.modalities
M = len(modalities)
fold_cfg = cfg.folds[0]
eval_subjects = fold_cfg['eval_set']
input_length = (cfg.duration * cfg.base_sample_rate) // 2
eval_ds = HARDataset('/files1/haodong/data/processed_dalia_activity',
                     modalities, eval_subjects, cfg, 'sax')
loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, drop_last=False)
print(f'Test samples: {len(eval_ds)}  modalities: {M}')

# Buffer first N samples
xs, ys = [], []
for x, y in loader:
    xs.append(x); ys.append(y)
    if sum(t.shape[0] for t in xs) >= args.n_samples: break
all_x = torch.cat(xs, dim=0)[:args.n_samples].float()
all_y = torch.cat(ys, dim=0)[:args.n_samples]
print(f'Benched on: {tuple(all_x.shape)}')

# Build V6 selector + load dyna ckpt
v6_dyna = build_v6(cfg, input_length, selector_downsample_factor=4)
state = torch.load(
    './model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna/best_model_fold0.pth',
    map_location='cpu', weights_only=False)
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k,v in state.items() if k.startswith('model.')}
v6_dyna.load_state_dict(state, strict=False)
v6_dyna.eval().to(device)


# ------------------------------------------------------------------- #
#  Method A — Neural V6 selector (current ShapDistill / C3 pipeline)  #
# ------------------------------------------------------------------- #
print('\n=== A: Neural V6 selector (top-K=1) ===')
v6_dyna.top_k = 1
# Warmup
for _ in range(args.n_warmup):
    xb = all_x[:args.batch_size].to(device)
    high, low = split_x(xb, modalities, cfg.variates)
    with torch.no_grad():
        out = v6_dyna(high, low, training=False, return_selection_info=True)
torch.cuda.synchronize()
# Time selection step (the V6 selector forward) alone — split off fusion
t_sel_total = 0.0
for r in range(args.n_runs):
    xb = all_x[:args.batch_size].to(device)
    high, low = split_x(xb, modalities, cfg.variates)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = v6_dyna.modality_selector({m: low[m] for m in modalities},
                                       top_k=v6_dyna.top_k, training=False)
    torch.cuda.synchronize()
    t_sel_total += time.perf_counter() - t0
us_per_sample_neural = (t_sel_total / args.n_runs / args.batch_size) * 1e6
print(f'  selector forward: {us_per_sample_neural:.2f} us / sample  '
      f'(batch={args.batch_size}, {args.n_runs} runs)')


# ------------------------------------------------------------------- #
#  Method B — ShapMML selection on V6 features                       #
# ------------------------------------------------------------------- #
print('\n=== B: ShapMML selection on V6 features ===')
# Build a calibrated ShapMML using existing precomputed train phi
phi_train_full = np.load(
    './model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna/'
    'shapley_train_fold0_phi.npz', allow_pickle=True)['phi'].astype(np.float32)
# Build fake X_train_features via extracting V6 selector projected on a subsample
# (must match phi shape, but we only need 1100 for calibration)
rng = np.random.default_rng(239)
idx_cal = rng.choice(phi_train_full.shape[0], 1100, replace=False)
phi_cal = phi_train_full[idx_cal]

# Get V6 projected features for the same 1100 calibration samples
# (since we don't have features cached, just use random features matching dim
#  for the timing-only purpose — the LATENCY characteristics don't depend on
#  exact values)
print('  (using synthetic 1280-dim features for timing only)')
X_cal = rng.standard_normal((1100, M * 256)).astype(np.float32)

# Instantiate ShapMML
def _no_op(X,Y): return None
def _no_pred(X,m): return np.zeros((X.shape[0], cfg.num_classes))
shapmml = ShapMML(
    x=X_cal, y=np.zeros(1100, dtype=np.float32),
    modalities={j: list(range(j*256, (j+1)*256)) for j in range(M)},
    learning_fn=_no_op, predict_fn=_no_pred,
    loss_fn=lambda y,p: np.zeros(len(y)),
    task_type='classification', split=0.0,
    alpha=0.10, lambda1=1e-3, lambda2=1e-3,
)
shapmml.shapley_values = phi_cal
shapmml.x_calib = X_cal
shapmml.n_cal = 1100
shapmml.p = M
shapmml.modality_list = list(range(M))

# Calibrate q=1 (one-time cost — NOT counted in inference)
t0 = time.perf_counter()
shapmml.conditional_calibrate(q=1, dim_reduce='pca', n_components=32, verbose=False)
calib_time = time.perf_counter() - t0
print(f'  one-time calibration cost: {calib_time:.1f}s  (NOT counted in inf)')

# Generate synthetic test features (same dim)
test_feats_full = rng.standard_normal((args.n_samples, M * 256)).astype(np.float32)
# Warmup
for _ in range(args.n_warmup):
    _ = shapmml.select_and_mask(test_feats_full[:args.batch_size])

# Time: (a) V6 selector forward + extract _projected features, (b) ShapMML predict + select
t_v6_total = 0.0
t_shap_total = 0.0
for r in range(args.n_runs):
    xb = all_x[:args.batch_size].to(device)
    high, low = split_x(xb, modalities, cfg.variates)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = v6_dyna.modality_selector({m: low[m] for m in modalities},
                                       top_k=1, training=False)
        proj = v6_dyna.modality_selector._projected
        stacked = torch.stack([proj[m] for m in modalities], dim=1)
        feats = stacked.reshape(stacked.shape[0], -1).cpu().numpy()
    torch.cuda.synchronize()
    t_v6_total += time.perf_counter() - t0

    t0 = time.perf_counter()
    _, _sel = shapmml.select_and_mask(feats)
    t_shap_total += time.perf_counter() - t0

us_v6 = (t_v6_total / args.n_runs / args.batch_size) * 1e6
us_shap = (t_shap_total / args.n_runs / args.batch_size) * 1e6
print(f'  V6 selector forward + feature extract: {us_v6:.2f} us / sample')
print(f'  ShapMML kernel predict + select:       {us_shap:.2f} us / sample')
print(f'  TOTAL selection cost (B):              {us_v6 + us_shap:.2f} us / sample')


# ------------------------------------------------------------------- #
#  Comparison                                                         #
# ------------------------------------------------------------------- #
print('\n' + '=' * 70)
print(f'NEURAL SELECTOR (V6 selector head):        {us_per_sample_neural:>8.2f} us/sample')
print(f'SHAPMML SELECTION (V6 features + kernel):  {us_v6 + us_shap:>8.2f} us/sample')
ratio = (us_v6 + us_shap) / us_per_sample_neural
print(f'  ShapMML / Neural ratio: {ratio:.2f}x  '
      f'(ShapMML is {"slower" if ratio>1 else "faster"})')
print('=' * 70)
