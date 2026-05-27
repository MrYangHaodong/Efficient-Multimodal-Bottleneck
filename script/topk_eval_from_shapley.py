"""Top-K modality selection & inference accuracy from pre-computed Shapley npz.

Loads the per-sample subset-Shapley files produced by
``compute_subset_shapley_{iemocap,daliahar,emognition}.py`` and reports test
accuracy + macro F1 for five selection strategies × K = 1..M-1:

  (POP-SING) Pop singleton-topK:
      Rank singletons by mean φ from {train,val,train+val}, pick top-K, apply
      that fixed subset to every test sample.

  (POP-SUB)  Pop subset-argmax:
      Argmax over all C(M,K) size-K subsets by mean φ from {train,val,train+val},
      apply that fixed subset to every test sample.

  (PER-SING) Per-sample singleton-topK (oracle, uses test-set φ):
      For each test sample, pick its own top-K modalities by per-sample
      singleton φ.

  (PER-SUB)  Per-sample subset-argmax (oracle, uses test-set φ):
      For each test sample, pick its own argmax size-K subset by per-sample
      group φ_T.

  (OUT-UB)   Output-best (true upper bound, uses test labels):
      For each test sample, score 1 if ANY size-K subset predicts correctly.

Per dataset:
  - IEMOCAP / DaliaHAR  →  averages over 3 folds (fold0_*.npz / fold1_*.npz / fold2_*.npz).
  - Emognition (within-subject) → single split (train.npz / val.npz / test.npz).

Example:
    python script/topk_eval_from_shapley.py \\
        --dataset=IEMOCAP \\
        --shapley_dir=./model_chkpt/IEMOCAP/multimodal_model/2026-05-23_IEMOCAP_v6_iemocap/subset_shapley
"""
from __future__ import annotations

import argparse
import os
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score


DATASET_CFGS: Dict[str, dict] = {
    'IEMOCAP':    dict(M=6, folds=[0, 1, 2],
                        modalities=['video', 'audio', 'text', 'mocap_hand', 'mocap_head', 'mocap_rotated']),
    'DaliaHAR':   dict(M=5, folds=[0, 1, 2],
                        modalities=['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ACC']),
    'Emognition': dict(M=8, folds=None,
                        modalities=['eeg_band', 'acc_head', 'bvp', 'eda', 'skt',
                                    'acc_wrist', 'hr', 'face_au']),
}


def load_npz(base: str, fold: Optional[int], split: str) -> dict:
    if fold is None:
        path = os.path.join(base, f'{split}.npz')
    else:
        path = os.path.join(base, f'fold{fold}_{split}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing Shapley file: {path}')
    return dict(np.load(path, allow_pickle=True).items())


def eval_mask(preds: np.ndarray, y: np.ndarray, mask) -> tuple:
    if np.isscalar(mask):
        pred = preds[:, int(mask)]
    else:
        pred = preds[np.arange(len(y)), mask]
    return (pred == y).mean(), f1_score(y, pred, average='macro', zero_division=0)


def select_pop_singleton(phi: np.ndarray, p: int, K: int) -> int:
    sing = np.array([phi[(1 << j) - 1] for j in range(p)])
    order = np.argsort(-sing)
    return sum(1 << i for i in sorted(order[:K]))


def select_pop_subset(phi: np.ndarray, p: int, K: int) -> int:
    masks = np.array([sum(1 << j for j in c) for c in combinations(range(p), K)])
    return int(masks[phi[masks - 1].argmax()])


def select_per_singleton(phi_te: np.ndarray, p: int, K: int) -> np.ndarray:
    sing_idx = np.array([(1 << j) - 1 for j in range(p)])
    sing = phi_te[:, sing_idx]
    topK = np.argsort(-sing, axis=1)[:, :K]
    return (1 << topK).sum(axis=1)


def select_per_subset(phi_te: np.ndarray, p: int, K: int) -> np.ndarray:
    masks = np.array([sum(1 << j for j in c) for c in combinations(range(p), K)])
    idx = phi_te[:, masks - 1].argmax(axis=1)
    return masks[idx]


def select_output_best(preds_te: np.ndarray, y_te: np.ndarray, p: int, K: int) -> np.ndarray:
    masks = np.array([sum(1 << j for j in c) for c in combinations(range(p), K)])
    correct = (preds_te == y_te[:, None]).astype(np.int32)
    first_correct_idx = correct[:, masks].argmax(axis=1)
    return masks[first_correct_idx]


def evaluate_dataset(shapley_dir: str, dataset: str, source: str) -> dict:
    """Run all five strategies × K=1..M-1, return per-K results.
    source ∈ {'train','val','train+val'} selects which split's φ feeds Pop methods.
    """
    cfg = DATASET_CFGS[dataset]
    M, folds = cfg['M'], cfg['folds']
    iter_folds = folds if folds is not None else [None]
    eval_split = 'eval' if folds is not None else 'test'

    out: Dict[int, Dict[str, dict]] = {K: {} for K in range(1, M)}
    full_metrics = {'acc': [], 'f1': []}

    for fold in iter_folds:
        ev = load_npz(shapley_dir, fold, eval_split)
        preds = ev['preds_subsets']
        y = ev['labels']

        # full-K (all modalities)
        full_mask = (1 << M) - 1
        a, f1 = eval_mask(preds, y, full_mask)
        full_metrics['acc'].append(a)
        full_metrics['f1'].append(f1)

        # population-level: load source phi for selection
        if source == 'train':
            phi_pop = load_npz(shapley_dir, fold, 'train')['phi_subsets'].mean(0)
        elif source == 'val':
            phi_pop = load_npz(shapley_dir, fold, 'val')['phi_subsets'].mean(0)
        elif source == 'train+val':
            t = load_npz(shapley_dir, fold, 'train')['phi_subsets']
            v = load_npz(shapley_dir, fold, 'val')['phi_subsets']
            phi_pop = np.concatenate([t, v], axis=0).mean(0)
        else:
            raise ValueError(f"Unknown source={source!r}")

        phi_te = ev['phi_subsets']
        for K in range(1, M):
            for name, mask in [
                ('POP-SING', select_pop_singleton(phi_pop, M, K)),
                ('POP-SUB',  select_pop_subset(phi_pop, M, K)),
                ('PER-SING', select_per_singleton(phi_te, M, K)),
                ('PER-SUB',  select_per_subset(phi_te, M, K)),
                ('OUT-UB',   select_output_best(preds, y, M, K)),
            ]:
                a, f1 = eval_mask(preds, y, mask)
                d = out[K].setdefault(name, {'acc': [], 'f1': []})
                d['acc'].append(a)
                d['f1'].append(f1)

    return {'per_K': out, 'full': full_metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASET_CFGS.keys()))
    ap.add_argument('--shapley_dir', required=True, type=str,
                    help='Directory containing the per-fold/per-split *.npz files.')
    ap.add_argument('--source', default='train+val', choices=['train', 'val', 'train+val'],
                    help='Which split feeds population-level selection.')
    args = ap.parse_args()

    res = evaluate_dataset(args.shapley_dir, args.dataset, args.source)
    M = DATASET_CFGS[args.dataset]['M']

    full_a = np.mean(res['full']['acc'])
    full_f = np.mean(res['full']['f1'])
    full_sa = np.std(res['full']['acc'])
    full_sf = np.std(res['full']['f1'])
    n = len(res['full']['acc'])
    print()
    print('=' * 110)
    print(f"Dataset: {args.dataset}  (M={M}, folds={n})    selection source = {args.source!r}")
    print(f"Reference K={M} (full): acc = {full_a:.4f}" + (f' ± {full_sa:.4f}' if n > 1 else '') +
          f",  F1 = {full_f:.4f}" + (f' ± {full_sf:.4f}' if n > 1 else ''))
    print('=' * 110)
    header = (f"  {'K':>3} | "
              f"{'POP-SING':>21} | {'POP-SUB':>21} | "
              f"{'PER-SING':>21} | {'PER-SUB':>21} | {'OUT-UB':>21}")
    print(header)
    print('-' * len(header))
    for K in range(1, M):
        cells = []
        for name in ['POP-SING', 'POP-SUB', 'PER-SING', 'PER-SUB', 'OUT-UB']:
            d = res['per_K'][K][name]
            ma = np.mean(d['acc']); mf = np.mean(d['f1'])
            cells.append(f"{ma:.4f}/{mf:.4f}")
        print(f"  {K:>3} | " + ' | '.join(f"{c:>21}" for c in cells))
    print()
    print('  (each cell: acc / macro F1; per-K mean across folds)')


if __name__ == '__main__':
    main()
