"""Apply shap_mml's selection algorithm (RKHS quantile regression
+ ``select_and_mask``) on top of the frozen V6 fusion model.

This is the apples-to-apples comparison: same V6 fusion as all our
other selectors; the *only* difference is the selection mechanism.

Pipeline:
  1. Build V6 with the 5.14 dyna checkpoint (frozen).  Selector head is
     present (no_selector=False, selector_downsample_factor=4) but we
     only use it to extract per-modality projected features.
  2. Compute per-sample V6 features  Z  =  concat_j projected_j(x) on
     the *calibration* set (subsample of training samples for which we
     already pre-computed Shapley targets).
  3. Pass Z + precomputed Shapley into shap_mml.conditional_calibrate
     to fit per-modality RKHS quantile regressors  h_j.
  4. At test time:  features Z_test → h_j(Z_test) → select_and_mask
     returns ``S(i)`` per sample → V6 fusion forward at ``S(i)`` gives
     the prediction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, '/files1/haodong/shap_mml-B3D1')

from utils.dataset_cfg import DaliaHAR, DSADS
from data.dataset_builder import HARDataset, DSADSDataset
from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)
from shap_mml import ShapMML


# --------------------------------------------------------------------- #
#                                Helpers                               #
# --------------------------------------------------------------------- #


class _V6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


def _build_v6_model(args, dataset_cfg, input_length):
    v6_cfg = _V6Cfg(dataset_cfg)
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
        no_selector=False, use_weighted_factor=True, use_triton=False,
        num_classes=dataset_cfg.num_classes,
        sparse_attn_variant=args.sparse_attn_variant,
        strat_block_size=args.strat_block_size,
        downsample_min_len=args.downsample_min_len,
        use_batched_fusion=True, per_modal_distill=False,
        per_modal_downsample_min_len=args.downsample_min_len,
        selector_downsample_factor=args.selector_downsample_factor,
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
def extract_projected_features(model, loader, modalities, variates, device):
    """For each batch, run V6 forward and read off
    model.modality_selector._projected[m] (shape [B, uniform_dim=256]).
    Concatenate per modality -> [B, M*256].  Returns ndarray [N, M*256]
    and labels."""
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device).float()
        high, low = _split_x(x, modalities, variates)
        _ = model(high, low, training=False, return_selection_info=True, labels=None)
        proj = model.modality_selector._projected
        stacked = torch.stack([proj[m] for m in modalities], dim=1)
        flat = stacked.reshape(stacked.shape[0], -1)
        feats.append(flat.cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


@torch.no_grad()
def v6_forward_with_subset(model, x, subsets_per_sample, modalities, variates,
                            num_classes, device):
    """Run V6 fusion on each sample with its own modality subset.
    Returns logits [B, C].  Groups by subset to batch forwards."""
    B = x.shape[0]
    high, low = _split_x(x, modalities, variates)
    groups = {}
    for i, S in enumerate(subsets_per_sample):
        # S is a tuple of modality indices into ``modalities`` list
        key = tuple(sorted(S))
        groups.setdefault(key, []).append(i)
    logits_buf = torch.zeros(B, num_classes, device=device)
    for key, idxs in groups.items():
        if len(key) == 0:
            # Empty subset -> uniform output (one modality fallback would
            # be unfair; uniform is the principled "no info" prediction).
            logits_buf[idxs] = (-torch.log(torch.tensor(
                float(num_classes), device=device)) *
                torch.ones(num_classes, device=device))
            continue
        keep = [modalities[j] for j in key]
        sub_high = {m: high[m][idxs] for m in keep}
        sub_low = {m: low[m][idxs] for m in keep}
        out = model(sub_high, sub_low, training=False,
                    return_selection_info=False)
        sub_logits = out[0] if isinstance(out, tuple) else out
        logits_buf[idxs] = sub_logits
    return logits_buf


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
parser.add_argument('--dataset', required=True, choices=['DaliaHAR', 'DSADS'])
parser.add_argument('--fold', required=True, type=int)
parser.add_argument('--frozen_ckpt', required=True, type=str)
parser.add_argument('--shapley_train_npz', required=True, type=str,
                    help='Path to precomputed V6-Shapley .npz for the train split.')
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--data_root_dsads', default='/files1/haodong/data/DSADS/', type=str)
parser.add_argument('--val_split_ratio', default=0.2, type=float)

# ShapMML selection knobs
parser.add_argument('--q_list', nargs='+', type=int, default=[1, 2, 3, 4, 5])
parser.add_argument('--alpha', default=0.10, type=float)
parser.add_argument('--lambda1', default=1e-3, type=float)
parser.add_argument('--lambda2', default=1e-3, type=float)
parser.add_argument('--n_components', default=32, type=int)
parser.add_argument('--n_cal_max', default=1100, type=int,
                    help='Random sub-sample of training set to use as '
                         'calibration for the QP solver.')

parser.add_argument('--output_json', default=None, type=str)

# Selector / backbone
parser.add_argument('--selector_downsample_factor', default=4, type=int)
parser.add_argument('--target_k', default=4, type=int)
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
np.random.seed(args.seed_num)
torch.manual_seed(args.seed_num)


# --------------------------------------------------------------------- #
#                                Data                                  #
# --------------------------------------------------------------------- #


print(f'Dataset: {args.dataset}  Fold: {args.fold}')
if args.dataset == 'DaliaHAR':
    cfg = DaliaHAR()
    modalities = cfg.modalities
    fold_cfg = cfg.folds[args.fold]
    train_subjects = fold_cfg['train_set']
    eval_subjects = fold_cfg['eval_set']

    input_length = ((cfg.duration * cfg.base_sample_rate) // 2
                    if args.transform == 'sax'
                    else cfg.duration * cfg.base_sample_rate)
    root_dir = '/files1/haodong/data/processed_dalia_activity'
    train_ds = HARDataset(root_dir, modalities, train_subjects, cfg, args.transform)
    eval_ds = HARDataset(root_dir, modalities, eval_subjects, cfg, args.transform)
else:
    cfg = DSADS()
    modalities = cfg.modalities
    input_length = cfg.duration * cfg.base_sample_rate
    sc = args.fold
    src_path = os.path.join(args.data_root_dsads,
                             f'dsad_diversify_dict_scenario{sc}_src.pt')
    trg_path = os.path.join(args.data_root_dsads,
                             f'dsad_diversify_dict_scenario{sc}_trg.pt')
    train_data = torch.load(src_path, weights_only=False)
    test_data = torch.load(trg_path, weights_only=False)
    tr_x, _, tr_y, _ = train_test_split(
        train_data['samples'], train_data['labels'],
        test_size=args.val_split_ratio, random_state=args.seed_num,
        stratify=train_data['labels'])
    train_split = {'samples': tr_x, 'labels': tr_y}
    train_ds = DSADSDataset(train_split, modalities, cfg, args.transform)
    eval_ds = DSADSDataset(test_data, modalities, cfg, args.transform)

num_classes = cfg.num_classes
print(f'  modalities ({len(modalities)}): {modalities}')
print(f'  num_classes={num_classes}  input_length={input_length}')

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=2, drop_last=False)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=2, drop_last=False)


# --------------------------------------------------------------------- #
#                                Model                                 #
# --------------------------------------------------------------------- #


inner = _build_v6_model(args, cfg, input_length)
state = torch.load(args.frozen_ckpt, map_location='cpu', weights_only=False)
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k, v in state.items()
             if k.startswith('model.')}
missing, unexpected = inner.load_state_dict(state, strict=False)
print(f'Loaded {args.frozen_ckpt}  '
      f'(missing={len(missing)} [selector head], unexpected={len(unexpected)})')
inner.eval().to(device)


# --------------------------------------------------------------------- #
#               Extract V6 projected features (train / test)           #
# --------------------------------------------------------------------- #


print('\nExtracting V6 projected features...')
t0 = time.time()
X_train_feats, y_train = extract_projected_features(
    inner, train_loader, modalities, cfg.variates, device)
print(f'  train features: {X_train_feats.shape}  ({time.time()-t0:.1f}s)')

t0 = time.time()
X_test_feats, y_test = extract_projected_features(
    inner, eval_loader, modalities, cfg.variates, device)
print(f'  test  features: {X_test_feats.shape}  ({time.time()-t0:.1f}s)')

# Precomputed Shapley targets for the train set
phi_pack = np.load(args.shapley_train_npz, allow_pickle=True)
phi_train = phi_pack['phi'].astype(np.float32)
assert phi_train.shape[0] == X_train_feats.shape[0], (
    f'Phi rows {phi_train.shape[0]} != train features rows '
    f'{X_train_feats.shape[0]}.  Make sure both use the same '
    f'shuffle=False ordering.')
print(f'\nLoaded V6 Shapley phi: {phi_train.shape}  '
      f'mean={phi_train.mean():+.4f}  std={phi_train.std():.4f}')

# Sub-sample for ShapMML's QP solver
if X_train_feats.shape[0] > args.n_cal_max:
    rng = np.random.default_rng(args.seed_num)
    idx = rng.choice(X_train_feats.shape[0], args.n_cal_max, replace=False)
    X_cal = X_train_feats[idx]
    y_cal = y_train[idx]
    Phi_cal = phi_train[idx]
    print(f'\nSub-sampled calibration set to {args.n_cal_max} '
          f'(from {X_train_feats.shape[0]})')
else:
    X_cal = X_train_feats
    y_cal = y_train
    Phi_cal = phi_train


# --------------------------------------------------------------------- #
#                  Initialise ShapMML with V6 features                 #
# --------------------------------------------------------------------- #


M = len(modalities)
uniform_dim = 256
modalities_flat = {j: list(range(j * uniform_dim, (j + 1) * uniform_dim))
                   for j in range(M)}

print(f'\nInstantiating ShapMML with V6 features...')

# Dummy callbacks — we never call ``train()`` so ``mu_S`` predictors
# are not built.  We only use ShapMML for the RKHS quantile regression
# and the selection rule.
def _dummy_learn(X, Y):
    return None
def _dummy_predict(X, model):
    return np.zeros((X.shape[0], num_classes))

shapmml = ShapMML(
    x=X_cal, y=y_cal.astype(np.float32),
    modalities=modalities_flat,
    learning_fn=_dummy_learn,
    predict_fn=_dummy_predict,
    loss_fn=lambda y, p: np.zeros(len(y)),
    task_type='classification',
    split=0.0,           # all samples go to ``x_calib``
    alpha=args.alpha,
    lambda1=args.lambda1, lambda2=args.lambda2,
)
# Override: feed our V6-based Shapley directly
shapmml.shapley_values = Phi_cal
shapmml.x_calib = X_cal
shapmml.n_cal = X_cal.shape[0]
shapmml.p = M
shapmml.modality_list = list(modalities_flat.keys())
print(f'  n_cal={shapmml.n_cal}  p={shapmml.p}  '
      f'feature_dim={X_cal.shape[1]}')


# --------------------------------------------------------------------- #
#              Conditional calibrate + per-q test eval                 #
# --------------------------------------------------------------------- #


# We need full test sample data (not just features) so we can run V6
# fusion at the selected subsets.  Pre-buffer test x tensors.
all_test_x = []
all_test_y = []
for x, y in eval_loader:
    all_test_x.append(x.float())
    all_test_y.append(y)
all_test_x = torch.cat(all_test_x, dim=0)        # CPU [N_test, T, C]
all_test_y = torch.cat(all_test_y, dim=0)        # CPU [N_test]
print(f'\nTest tensors buffered: x={tuple(all_test_x.shape)}, '
      f'y={tuple(all_test_y.shape)}')


results = {
    'dataset': args.dataset, 'fold': args.fold,
    'modalities': modalities, 'num_classes': num_classes,
    'q_list': args.q_list, 'n_cal': int(shapmml.n_cal),
    'per_q': {},
}

print('\n' + '=' * 80)
for q in args.q_list:
    if q < 1 or q > M:
        continue
    print(f'\n=== q = {q} ===')
    t1 = time.time()
    shapmml.conditional_calibrate(q=q, dim_reduce='pca',
                                  n_components=args.n_components,
                                  verbose=False)
    cal_time = time.time() - t1
    print(f'  conditional calibration: {cal_time:.1f}s')

    # Per-sample selection on test features
    t1 = time.time()
    _, sel_list = shapmml.select_and_mask(X_test_feats)
    sel_time = time.time() - t1

    # Run V6 fusion at the selected subsets (batched)
    t1 = time.time()
    all_preds = []
    BS = args.batch_size
    for start in range(0, all_test_x.shape[0], BS):
        end = min(start + BS, all_test_x.shape[0])
        xb = all_test_x[start:end].to(device)
        sel_batch = sel_list[start:end]
        logits = v6_forward_with_subset(
            inner, xb, sel_batch, modalities, cfg.variates,
            num_classes, device)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(all_preds)
    fwd_time = time.time() - t1

    acc = float(accuracy_score(all_test_y.numpy(), y_pred))
    f1 = float(f1_score(all_test_y.numpy(), y_pred, average='macro'))
    ks = [len(s) for s in sel_list]
    mean_k = float(np.mean(ks))
    min_k = int(min(ks)) if ks else 0
    max_k = int(max(ks)) if ks else 0
    n_zero = sum(1 for k in ks if k == 0)
    print(f'  q={q}  test_acc={acc:.4f}  test_f1={f1:.4f}')
    print(f'      mean|S|={mean_k:.2f}  (min={min_k}, max={max_k}, '
          f'#empty={n_zero})  sel_time={sel_time:.1f}s  '
          f'fwd_time={fwd_time:.1f}s')

    results['per_q'][str(q)] = {
        'q': q, 'test_acc': acc, 'test_f1': f1,
        'mean_k_used': mean_k, 'min_k': min_k, 'max_k': max_k,
        'n_empty_subsets': n_zero,
        'cal_time_s': cal_time, 'sel_time_s': sel_time,
        'fwd_time_s': fwd_time,
    }

print('\n' + '=' * 80)
print('Summary:')
for q in args.q_list:
    if str(q) in results['per_q']:
        r = results['per_q'][str(q)]
        print(f'  q={q}: acc={r["test_acc"]:.4f}  '
              f'f1={r["test_f1"]:.4f}  mean|S|={r["mean_k_used"]:.2f}')

if args.output_json:
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote: {args.output_json}')
