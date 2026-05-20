"""Test-Time Calibration (TTCal) for IEMOCAP ShapDistill selector.

Diagnosis (prior session): train→test Shapley distribution shifts on
IEMOCAP — text mean phi: +0.69 (train) → -0.92 (test). The selector
trained on train-Shapley is therefore *miscalibrated* at test time.

This script applies a zero-train post-hoc fix:
  1. Load a ShapDistill selector ckpt (frozen fusion + trained selector).
  2. Precompute leave-one-out (LOO) `phi_proxy` on val (true labels)
     and test (pseudo-labels from full-modality forward).
  3. Grid search `alpha` on val per K to maximise val_acc when shifting
     selector logits by `alpha * phi_proxy`.
  4. Apply best `alpha[K]` on test → final TTCal test_acc per K.

phi_proxy[i, j] = p_all(y_i) - p_minus_j(y_i)
  where p_all is full-modality softmax on the true / pseudo label,
  and p_minus_j drops modality j.

Selector logit hook: `inner.modality_selector._tt_logit_bias` is added
to logits before sigmoid (see ``v6_selector.py``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
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


def _build_model(args, modalities, input_length):
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
        no_selector=False, use_weighted_factor=True, use_triton=False,
        num_classes=NUM_CLASSES,
        sparse_attn_variant=args.sparse_attn_variant,
        strat_block_size=args.strat_block_size,
        downsample_min_len=args.downsample_min_len,
        use_batched_fusion=True, per_modal_distill=False,
        per_modal_downsample_min_len=args.downsample_min_len,
        selector_downsample_factor=args.selector_downsample_factor,
    )


def _unpack(batch, modalities, device):
    *feats, label = batch
    high, low = {}, {}
    for j, m in enumerate(modalities):
        high[m] = feats[2 * j].to(device).float()
        low[m] = feats[2 * j + 1].to(device).float()
    y = label.squeeze(-1).long().to(device)
    return high, low, y


def _forward(model, high, low, top_k, training=False):
    """Wrap V6 forward with explicit top_k override (eval semantics)."""
    saved = model.top_k
    model.top_k = top_k
    out = model(high, low, training=training, return_selection_info=False)
    model.top_k = saved
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def _compute_phi_proxy(model, loader, modalities, device, use_true_label):
    """Return phi_proxy [N, M], pseudo_y [N], and true_y [N].

    phi_proxy[i, j] = p_all(target_i) - p_minus_j(target_i)
    target_i = true label if use_true_label else pseudo label.
    """
    model.eval()
    M = len(modalities)
    phi_chunks: List[np.ndarray] = []
    yp_chunks: List[np.ndarray] = []
    yt_chunks: List[np.ndarray] = []
    for batch in loader:
        high, low, y = _unpack(batch, modalities, device)

        # Full-modality forward (top_k=M => no selection, all 6 used)
        logits_full = _forward(model, high, low, top_k=M)
        prob_full = F.softmax(logits_full, dim=-1)
        pseudo_y = prob_full.argmax(dim=-1)
        target = y if use_true_label else pseudo_y

        B = prob_full.shape[0]
        p_all_t = prob_full.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]

        phi_batch = torch.zeros(B, M, device=device)
        for j, m in enumerate(modalities):
            h_drop = {k: v for k, v in high.items() if k != m}
            l_drop = {k: v for k, v in low.items() if k != m}
            logits_j = _forward(model, h_drop, l_drop, top_k=M - 1)
            prob_j = F.softmax(logits_j, dim=-1)
            p_minus_j_t = prob_j.gather(1, target.unsqueeze(1)).squeeze(1)
            phi_batch[:, j] = p_all_t - p_minus_j_t

        phi_chunks.append(phi_batch.cpu().numpy())
        yp_chunks.append(pseudo_y.cpu().numpy())
        yt_chunks.append(y.cpu().numpy())

    return (np.concatenate(phi_chunks, axis=0),
            np.concatenate(yp_chunks, axis=0),
            np.concatenate(yt_chunks, axis=0))


@torch.no_grad()
def _eval_with_bias(model, loader, modalities, K, device,
                    phi_proxy, alpha, pseudo_labels=None):
    """Iterate loader (shuffle=False) and apply alpha*phi_proxy as
    test-time logit bias.

    Returns dict with:
      acc, f1                 — vs true labels (from loader)
      pseudo_acc              — vs ``pseudo_labels`` if provided (TTA target)
    """
    model.eval()
    selector = model.modality_selector
    M = len(modalities)
    cursor = 0
    all_pred, all_y = [], []

    for batch in loader:
        high, low, y = _unpack(batch, modalities, device)
        B = y.shape[0]
        bias = torch.from_numpy(phi_proxy[cursor:cursor + B]).float().to(device)
        bias = alpha * bias
        try:
            selector._tt_logit_bias = bias
            logits = _forward(model, high, low, top_k=K)
        finally:
            selector._tt_logit_bias = None
        all_pred.append(logits.argmax(dim=-1).cpu())
        all_y.append(y.cpu())
        cursor += B
    assert cursor == phi_proxy.shape[0], (
        f'Iteration order mismatch: consumed {cursor}, phi has {phi_proxy.shape[0]}')
    y_arr = torch.cat(all_y).numpy()
    p_arr = torch.cat(all_pred).numpy()
    out = {
        'acc': float(accuracy_score(y_arr, p_arr)),
        'f1': float(f1_score(y_arr, p_arr, average='macro')),
    }
    if pseudo_labels is not None:
        out['pseudo_acc'] = float(accuracy_score(pseudo_labels, p_arr))
    return out


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
parser.add_argument('--ckpt_dir', required=True,
                    help='Dir containing best_model_fold{F}_k{K}.pth files.')
parser.add_argument('--fold', default=0, type=int, choices=[0, 1, 2])
parser.add_argument('--ks', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6])
parser.add_argument('--alpha_grid', nargs='+', type=float,
                    default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
parser.add_argument('--cuda_pick', default='cuda:0')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_seq_len', default=128, type=int)
parser.add_argument('--time_compression_ratio', default=4, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--data_root', default='/files1/haodong/data/IEMOCAP')
parser.add_argument('--output_json', default=None)
parser.add_argument('--use_pseudo_for_val', default=False, type=_str2bool,
                    help='If True, use pseudo-labels also for val phi_proxy '
                         '(simulates fully unlabeled calibration set). '
                         'Default uses val true labels.')
parser.add_argument('--alpha_select_on', default='val_true',
                    choices=['val_true', 'test_pseudo'],
                    help='val_true: select alpha by val_acc against true labels '
                         '(standard cross-domain calibration). '
                         'test_pseudo: select alpha by test_acc against '
                         'pseudo-labels from full-modality teacher '
                         '(TTA-style consistency, no test label leakage).')

# V6 (must match training)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--internal_dim', default=128, type=int)
parser.add_argument('--n_bottlenecks', default=8, type=int)
parser.add_argument('--fusion_layer', default=0, type=int)
parser.add_argument('--use_sparse_attn', default=True, type=_str2bool)
parser.add_argument('--sparse_attn_variant', default='opt', type=str)
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--selector_downsample_factor', default=4, type=int)

args = parser.parse_args()

device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed_num)
modalities = list(IEMOCAPDataset.ALL_MODALITIES)
input_length = args.max_seq_len if args.max_seq_len > 0 else 600
M = len(modalities)

# Loaders (shuffle=False so phi_proxy can be indexed by iteration order)
suffix = '' if args.fold == 0 else f'_split{args.fold}'
val_csv = os.path.join(args.data_root, f'val{suffix}.csv')
test_csv = os.path.join(args.data_root, f'test{suffix}.csv')
ds_kwargs = dict(
    data_root=args.data_root, modalities=modalities,
    max_seq_len=args.max_seq_len,
    time_compression_ratio=args.time_compression_ratio,
    use_batched_fusion=True,
)
val_ds = IEMOCAPDataset(csv_path=val_csv, **ds_kwargs)
test_ds = IEMOCAPDataset(csv_path=test_csv, **ds_kwargs)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False,
                        collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, drop_last=False,
                         collate_fn=collate_fn)
print(f'Device={device}  Fold={args.fold}  N_val={len(val_ds)}  N_test={len(test_ds)}')

# Build model template (reused per K, just reload weights)
model = _build_model(args, modalities, input_length).to(device)
model.eval()

per_k_results = {}
print('\n' + '=' * 80)
print(f'TT-Calibration sweep (alpha grid={args.alpha_grid})')
print('=' * 80)

for K in args.ks:
    ckpt_path = os.path.join(args.ckpt_dir, f'best_model_fold{args.fold}_k{K}.pth')
    if not os.path.exists(ckpt_path):
        print(f'[K={K}] missing ckpt {ckpt_path} — skipping')
        continue
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if any(k.startswith('model.') for k in state.keys()):
        state = {k[len('model.'):]: v for k, v in state.items()
                 if k.startswith('model.')}
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f'\n[K={K}] loaded {ckpt_path}  miss={len(miss)} unexp={len(unexp)}')

    # ---- Precompute phi_proxy (val with true labels, test with pseudo)
    phi_val, _, _ = _compute_phi_proxy(
        model, val_loader, modalities, device,
        use_true_label=not args.use_pseudo_for_val)
    phi_test, pseudo_y_test, _ = _compute_phi_proxy(
        model, test_loader, modalities, device, use_true_label=False)
    print(f'  phi_val  mean={phi_val.mean():+.4f}  std={phi_val.std():.4f}'
          f'   per-mod mean=' + ', '.join(
              f'{modalities[j][:5]}={phi_val[:, j].mean():+.3f}'
              for j in range(M)))
    print(f'  phi_test mean={phi_test.mean():+.4f}  std={phi_test.std():.4f}'
          f'   per-mod mean=' + ', '.join(
              f'{modalities[j][:5]}={phi_test[:, j].mean():+.3f}'
              for j in range(M)))

    # ---- Baseline (no calibration)
    base_val = _eval_with_bias(model, val_loader, modalities, K, device,
                                np.zeros((len(val_ds), M), dtype=np.float32), 0.0)
    base_test = _eval_with_bias(model, test_loader, modalities, K, device,
                                 np.zeros((len(test_ds), M), dtype=np.float32),
                                 0.0, pseudo_labels=pseudo_y_test)
    base_val_acc, base_val_f1 = base_val['acc'], base_val['f1']
    base_test_acc, base_test_f1 = base_test['acc'], base_test['f1']
    base_test_pseudo_acc = base_test['pseudo_acc']
    print(f'  baseline   val_acc={base_val_acc:.4f}  test_acc={base_test_acc:.4f}  '
          f'test_pseudo_acc={base_test_pseudo_acc:.4f}')

    # ---- alpha grid search
    grid_results = {}
    select_key = args.alpha_select_on  # 'val_true' or 'test_pseudo'
    best_alpha = 0.0
    if select_key == 'val_true':
        best_select_score = base_val_acc
    else:
        best_select_score = base_test_pseudo_acc

    for alpha in args.alpha_grid:
        if alpha == 0.0:
            v = base_val; t = base_test
        else:
            v = _eval_with_bias(model, val_loader, modalities, K, device,
                                phi_val, alpha)
            t = _eval_with_bias(model, test_loader, modalities, K, device,
                                phi_test, alpha, pseudo_labels=pseudo_y_test)
        grid_results[f'{alpha:.2f}'] = {
            'val_acc': v['acc'], 'val_f1': v['f1'],
            'test_acc': t['acc'], 'test_f1': t['f1'],
            'test_pseudo_acc': t['pseudo_acc'],
        }
        score = v['acc'] if select_key == 'val_true' else t['pseudo_acc']
        if score > best_select_score + 1e-9:
            best_select_score = score
            best_alpha = alpha
        print(f'    alpha={alpha:>5.2f}  val_acc={v["acc"]:.4f}  '
              f'test_acc={t["acc"]:.4f}  test_pseudo_acc={t["pseudo_acc"]:.4f}')
    print(f'  -> best alpha by {select_key}: {best_alpha} '
          f'(score={best_select_score:.4f})')

    # ---- Final TTCal test result (pulled directly from the grid sweep so
    # there's no GPU-nondeterminism drift between alpha lookup and reporting)
    grid_best = grid_results[f'{best_alpha:.2f}']
    ttcal_test_acc = grid_best['test_acc']
    ttcal_test_f1 = grid_best['test_f1']
    ttcal_val_acc = grid_best['val_acc']
    print(f'  TTCal      val_acc={ttcal_val_acc:.4f}  test_acc={ttcal_test_acc:.4f}  '
          f'(delta={ttcal_test_acc - base_test_acc:+.4f})')

    per_k_results[str(K)] = {
        'K': K,
        'ckpt': ckpt_path,
        'baseline': {
            'val_acc': base_val_acc, 'val_f1': base_val_f1,
            'test_acc': base_test_acc, 'test_f1': base_test_f1,
            'test_pseudo_acc': base_test_pseudo_acc,
        },
        'best_alpha': best_alpha,
        'select_on': select_key,
        'ttcal': {
            'val_acc': ttcal_val_acc,
            'test_acc': ttcal_test_acc, 'test_f1': ttcal_test_f1,
        },
        'delta_test_acc': ttcal_test_acc - base_test_acc,
        'alpha_grid': grid_results,
        'phi_val_mean_per_mod': phi_val.mean(axis=0).tolist(),
        'phi_test_mean_per_mod': phi_test.mean(axis=0).tolist(),
    }

print('\n' + '=' * 80)
print('SUMMARY (fold={})'.format(args.fold))
print(f'  {"K":<4}{"baseline":<12}{"alpha*":<10}{"TTCal":<12}{"delta":<10}')
for K in args.ks:
    r = per_k_results.get(str(K))
    if r is None:
        continue
    print(f'  {K:<4}{r["baseline"]["test_acc"]:<12.4f}{r["best_alpha"]:<10}'
          f'{r["ttcal"]["test_acc"]:<12.4f}{r["delta_test_acc"]:+.4f}')

out = {
    'ckpt_dir': args.ckpt_dir,
    'fold': args.fold,
    'modalities': modalities,
    'ks': args.ks,
    'alpha_grid': args.alpha_grid,
    'use_pseudo_for_val': args.use_pseudo_for_val,
    'results': per_k_results,
}
if args.output_json:
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote: {args.output_json}')
