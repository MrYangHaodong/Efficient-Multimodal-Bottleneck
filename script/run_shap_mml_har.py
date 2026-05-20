"""Apply the off-the-shelf ShapMML (shap_mml-B3D1) algorithm to
HAR datasets.

This is an independent baseline: instead of our V6 fusion + neural
selector, we use the ShapMML library *as-is* — XGBoost predictors
per subset + conditional-conformal RKHS quantile regression for the
selection rule.  The point is to see how well their algorithm does
on time-series HAR data, head-to-head with our neural pipeline.

Time series ``(N, T, C)`` are flattened to ``(N, T * C)`` so that
modalities map to *flat* feature indices.  Each modality j owns the
flat indices ``[t * C + c for t in range(T) for c in chans(j)]``.

Run example:
    python script/run_shap_mml_har.py --dataset=DaliaHAR --fold=0 --q=2
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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, '/files1/haodong/shap_mml-B3D1')

from utils.dataset_cfg import DaliaHAR, DSADS
from data.dataset_builder import HARDataset, DSADSDataset
from sklearn.model_selection import train_test_split
from shap_mml import ShapMML


# --------------------------------------------------------------------- #
#                     Learning / predict / loss fns                    #
# --------------------------------------------------------------------- #


def xgb_classifier_learning_fn(num_class):
    """Returns a learning_fn closure for XGBoost multi-class."""
    import xgboost as xgb

    def _learn(X, Y):
        params = {
            'objective': 'multi:softprob',
            'num_class': int(num_class),
            'eval_metric': 'mlogloss',
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.5,
            'verbosity': 0,
            'tree_method': 'hist',
            'nthread': 8,
            'seed': 42,
        }
        dtrain = xgb.DMatrix(X, label=Y)
        bst = xgb.train(params, dtrain, num_boost_round=50)
        return bst

    def _predict(X, model):
        import xgboost as xgb
        dx = xgb.DMatrix(X)
        return model.predict(dx)

    return _learn, _predict


def ce_per_sample(y_true, y_pred):
    """Per-sample cross-entropy loss for classification."""
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    n_classes = y_pred.shape[1]
    y_onehot = label_binarize(y_true.astype(int),
                               classes=np.arange(n_classes))
    if n_classes == 2 and y_onehot.ndim == 1:
        y_onehot = np.column_stack([1 - y_onehot, y_onehot])
    return -np.sum(y_onehot * np.log(y_pred), axis=1)


# --------------------------------------------------------------------- #
#                            Data loading                              #
# --------------------------------------------------------------------- #


def _load_daliahar(fold, transform='sax'):
    cfg = DaliaHAR()
    fold_cfg = cfg.folds[fold]
    train_subjects, eval_subjects = fold_cfg['train_set'], fold_cfg['eval_set']
    val_subjects = cfg.val_set
    root_dir = '/files1/haodong/data/processed_dalia_activity'

    train_ds = HARDataset(root_dir, cfg.modalities, train_subjects, cfg, transform)
    val_ds = HARDataset(root_dir, cfg.modalities, val_subjects, cfg, transform)
    eval_ds = HARDataset(root_dir, cfg.modalities, eval_subjects, cfg, transform)

    def _to_np(ds):
        xs, ys = [], []
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append(np.asarray(x))
            ys.append(int(y))
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)

    Xtr, Ytr = _to_np(train_ds)
    Xva, Yva = _to_np(val_ds)
    Xte, Yte = _to_np(eval_ds)
    return cfg, (Xtr, Ytr), (Xva, Yva), (Xte, Yte)


def _load_dsads(fold, data_root='/files1/haodong/data/DSADS/', transform='sax',
                seed=239, val_ratio=0.2):
    cfg = DSADS()
    src_path = os.path.join(data_root, f'dsad_diversify_dict_scenario{fold}_src.pt')
    trg_path = os.path.join(data_root, f'dsad_diversify_dict_scenario{fold}_trg.pt')
    train_data = torch.load(src_path, weights_only=False)
    test_data = torch.load(trg_path, weights_only=False)
    tr_x, va_x, tr_y, va_y = train_test_split(
        train_data['samples'], train_data['labels'],
        test_size=val_ratio, random_state=seed,
        stratify=train_data['labels'])

    train_ds = DSADSDataset({'samples': tr_x, 'labels': tr_y},
                             cfg.modalities, cfg, transform)
    val_ds = DSADSDataset({'samples': va_x, 'labels': va_y},
                           cfg.modalities, cfg, transform)
    eval_ds = DSADSDataset(test_data, cfg.modalities, cfg, transform)

    def _to_np(ds):
        xs, ys = [], []
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append(np.asarray(x))
            ys.append(int(y))
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)

    Xtr, Ytr = _to_np(train_ds)
    Xva, Yva = _to_np(val_ds)
    Xte, Yte = _to_np(eval_ds)
    return cfg, (Xtr, Ytr), (Xva, Yva), (Xte, Yte)


# --------------------------------------------------------------------- #
#                              Driver                                  #
# --------------------------------------------------------------------- #


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['DaliaHAR', 'DSADS'])
parser.add_argument('--fold', required=True, type=int)
parser.add_argument('--q_list', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                    help='List of q (target K) values to evaluate.')
parser.add_argument('--alpha', default=0.10, type=float)
parser.add_argument('--lambda1', default=1e-3, type=float)
parser.add_argument('--lambda2', default=1e-3, type=float)
parser.add_argument('--n_components', default=32, type=int,
                    help='PCA components for RKHS quantile reg.')
parser.add_argument('--split', default=0.5, type=float,
                    help='Train/calibration split ratio (default 0.5 like ShapMML default).')
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--output_json', default=None, type=str)
parser.add_argument('--features', default='raw',
                    choices=['raw', 'stats', 'stats_fft'],
                    help='raw = flatten time series; '
                         'stats = mean/std/min/max/q25/q50/q75 per channel; '
                         'stats_fft = stats + FFT energy in 5 bands per channel.')

args = parser.parse_args()
np.random.seed(args.seed)


def _extract_stats(X, fft_bands=False):
    """Extract per-channel statistical features.
    Input  X: (N, T, C).  Output: (N, n_feat_per_channel * C)
    Channels are interleaved: [stat0_c0, stat0_c1, ..., stat6_c0, ...]
    -> actually we lay them out as [c0_stat0..stat6, c1_stat0..stat6, ...]
    so modality 'c0' owns a contiguous slice.
    """
    N, T, C = X.shape
    stats = [X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1),
             np.percentile(X, 25, axis=1),
             np.percentile(X, 50, axis=1),
             np.percentile(X, 75, axis=1)]
    feat = np.stack(stats, axis=-1)  # (N, C, 7)

    if fft_bands:
        # FFT per channel, energy in 5 evenly-spaced frequency bands
        Xf = np.abs(np.fft.rfft(X, axis=1))           # (N, F, C), F=T/2+1
        Fbins = Xf.shape[1]
        band_edges = np.linspace(0, Fbins, 6).astype(int)
        band_energies = np.stack([
            Xf[:, band_edges[b]:band_edges[b+1], :].sum(axis=1)
            for b in range(5)
        ], axis=-1)                                    # (N, C, 5)
        feat = np.concatenate([feat, band_energies], axis=-1)  # (N, C, n_stats)

    feat_flat = feat.reshape(N, -1)                   # (N, C * n_stats)
    n_stats_per_channel = feat.shape[-1]
    return feat_flat.astype(np.float32), n_stats_per_channel


# --------------------------------------------------------------------- #
#                                Load                                  #
# --------------------------------------------------------------------- #


print(f'Dataset: {args.dataset}  Fold: {args.fold}')
if args.dataset == 'DaliaHAR':
    cfg, (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = _load_daliahar(args.fold, args.transform)
else:
    cfg, (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = _load_dsads(args.fold, transform=args.transform)

modalities = cfg.modalities
variates = cfg.variates
num_classes = cfg.num_classes
T = Xtr.shape[1]
C = Xtr.shape[2]
print(f'  modalities: {modalities}  num_classes={num_classes}')
print(f'  train: {Xtr.shape}, val: {Xva.shape}, eval: {Xte.shape}')
print(f'  T={T}  C={C}  flat features = {T*C}')

# Build modality dict (flat indices)
# Channel offsets per modality
offsets = []
off = 0
for m in modalities:
    offsets.append((off, off + variates[m]))
    off += variates[m]

if args.features == 'raw':
    # Raw flatten: each (T, C) -> T*C features, modality owns
    # [t * C + c for t in range(T) for c in chans(j)]
    modalities_flat = {}
    for j, m in enumerate(modalities):
        s, e = offsets[j]
        chans = list(range(s, e))
        idxs = [t * C + c for t in range(T) for c in chans]
        modalities_flat[j] = idxs
    X_flat_tr = Xtr.reshape(Xtr.shape[0], -1)
    X_flat_te = Xte.reshape(Xte.shape[0], -1)
    print(f'  Feature scheme: raw flatten  dim={X_flat_tr.shape[1]}')
else:
    # Stats / stats+fft per channel.  Each channel owns n_stats consecutive
    # features.  Channels are laid out in the same order as the original
    # (T, C) i.e. channel index 0..C-1.  Modality j owns the slice of
    # channels [offsets[j][0], offsets[j][1]).
    fft_bands = (args.features == 'stats_fft')
    X_flat_tr, n_stats = _extract_stats(Xtr, fft_bands=fft_bands)
    X_flat_te, _ = _extract_stats(Xte, fft_bands=fft_bands)
    modalities_flat = {}
    for j, m in enumerate(modalities):
        s, e = offsets[j]
        idxs = []
        for c in range(s, e):
            base = c * n_stats
            idxs.extend(list(range(base, base + n_stats)))
        modalities_flat[j] = idxs
    # Standardize features (XGBoost is scale-invariant but cvxpy is sensitive)
    mu_x = X_flat_tr.mean(axis=0, keepdims=True)
    sd_x = X_flat_tr.std(axis=0, keepdims=True) + 1e-8
    X_flat_tr = (X_flat_tr - mu_x) / sd_x
    X_flat_te = (X_flat_te - mu_x) / sd_x
    print(f'  Feature scheme: {args.features}  '
          f'n_stats_per_channel={n_stats}  dim={X_flat_tr.shape[1]}')
    for j, m in enumerate(modalities):
        print(f'    modality {j}={m}  feature_count={len(modalities_flat[j])}')


# --------------------------------------------------------------------- #
#                            Build ShapMML                              #
# --------------------------------------------------------------------- #


learn_fn, predict_fn = xgb_classifier_learning_fn(num_classes)

t0 = time.time()
print('\nInstantiating ShapMML...')
shapmml = ShapMML(
    x=X_flat_tr,
    y=Ytr.astype(np.float32),
    modalities=modalities_flat,
    learning_fn=learn_fn,
    predict_fn=predict_fn,
    loss_fn=ce_per_sample,
    task_type='classification',
    alpha=args.alpha,
    split=args.split,
    lambda1=args.lambda1,
    lambda2=args.lambda2,
)
print(f'  n_train={shapmml.m}  n_cal={shapmml.n_cal}  p={shapmml.p}')

# Train 2^p XGBoost predictors
print('\nTraining 2^p XGBoost predictors...')
t1 = time.time()
shapmml.train()
print(f'  done in {time.time() - t1:.1f}s')

# Marginal calibration computes per-sample Shapley values + marginal CIs
print('\nMarginal calibration (computes Shapley values + marginal CIs)...')
t1 = time.time()
marginal = shapmml.marginal_calibrate()
print(f'  done in {time.time() - t1:.1f}s')
print('  marginal table:')
print(marginal)

# Save Phi for downstream
phi_calib = shapmml.shapley_values  # [n_cal, p]
print(f'\nShapley values (calibration set): {phi_calib.shape}')
print('  per-modality mean phi:')
for j, m in enumerate(modalities):
    print(f'    {m:<14}  mean={phi_calib[:, j].mean():+.4f}  std={phi_calib[:, j].std():.4f}')


# --------------------------------------------------------------------- #
#               Conditional calibration + per-q evaluation             #
# --------------------------------------------------------------------- #


results = {'dataset': args.dataset, 'fold': args.fold,
           'modalities': modalities, 'num_classes': num_classes,
           'q_list': args.q_list,
           'phi_mean': {m: float(phi_calib[:, j].mean())
                         for j, m in enumerate(modalities)},
           'per_q': {}}

for q in args.q_list:
    if q < 0 or q > shapmml.p:
        continue
    print(f'\n=== q = {q} ===')
    t1 = time.time()
    shapmml.conditional_calibrate(q=q, dim_reduce='pca',
                                   n_components=args.n_components,
                                   verbose=False)
    cal_time = time.time() - t1
    print(f'  conditional calibration: {cal_time:.1f}s')

    # Predict on test set
    t1 = time.time()
    if q == 0:
        y_pred_probs = np.full((X_flat_te.shape[0], num_classes), 1.0 / num_classes)
        selected_S_list = [tuple()] * X_flat_te.shape[0]
    else:
        y_pred_probs, selected_S_list = shapmml.predict_optimal_modalities(X_flat_te)
    pred_time = time.time() - t1
    y_pred = y_pred_probs.argmax(axis=1)

    acc = float(accuracy_score(Yte, y_pred))
    f1 = float(f1_score(Yte, y_pred, average='macro'))

    ks = [len(s) for s in selected_S_list]
    mean_k = float(np.mean(ks))
    min_k = int(min(ks)) if ks else 0
    max_k = int(max(ks)) if ks else 0
    print(f'  q={q}  test_acc={acc:.4f}  test_f1={f1:.4f}  '
          f'mean|S|={mean_k:.2f} (min={min_k}, max={max_k})  '
          f'pred_time={pred_time:.1f}s')

    results['per_q'][str(q)] = {
        'q': q, 'test_acc': acc, 'test_f1': f1,
        'mean_k_used': mean_k,
        'cal_time_s': cal_time, 'pred_time_s': pred_time,
    }

print(f'\nTotal time: {time.time() - t0:.1f}s')

if args.output_json:
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote results to {args.output_json}')
