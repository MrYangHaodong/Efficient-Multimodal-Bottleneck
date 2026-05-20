"""Aggregate IEMOCAP TT-Calibration results across 3 folds.

Reads ttcal_fold{0,1,2}.json from one or two output dirs (typical pair:
ttcal_pseudo + ttcal_val) and prints per-K mean+/-std for baseline vs
TTCal test_acc, plus optional comparison vs gradnorm and val-Shapley.

Usage:
    python script/aggregate_ttcal_iemocap.py \
        --pseudo_dir=./model_chkpt/IEMOCAP/selector/2026-05-17_IEMOCAP_v6_ttcal_pseudo \
        --val_dir=./model_chkpt/IEMOCAP/selector/2026-05-17_IEMOCAP_v6_ttcal_val
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def _load_fold_dir(d):
    out = {}
    for fold in (0, 1, 2):
        p = os.path.join(d, f'ttcal_fold{fold}.json')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            out[fold] = json.load(f)
    return out


def _summarize(label, fold_data, ks):
    """Pull TTCal test_acc from alpha_grid[best_alpha] (the original sweep
    numbers) rather than the re-run TTCal field which can drift slightly
    due to GPU non-determinism."""
    print(f'\n[{label}]')
    print(f'  {"K":<3}{"baseline":<25}{"TTCal":<25}{"delta":<12}{"alpha*":<14}')
    for K in ks:
        base_accs, ttcal_accs, alphas = [], [], []
        for fold, d in sorted(fold_data.items()):
            r = d.get('results', {}).get(str(K))
            if r is None:
                continue
            base_accs.append(r['baseline']['test_acc'])
            a = r['best_alpha']
            grid = r.get('alpha_grid', {}) or {}
            akey = f'{a:.2f}'
            if akey in grid:
                ttcal_accs.append(grid[akey]['test_acc'])
            else:
                ttcal_accs.append(r['ttcal']['test_acc'])
            alphas.append(a)
        if not base_accs:
            continue
        b = np.array(base_accs); t = np.array(ttcal_accs)
        delta = t - b
        print(f'  {K:<3}'
              f'{b.mean():.4f} +/- {b.std(ddof=0):.4f}   '
              f'{t.mean():.4f} +/- {t.std(ddof=0):.4f}   '
              f'{delta.mean():+.4f}     '
              f'{alphas}')


parser = argparse.ArgumentParser()
parser.add_argument('--pseudo_dir', default=None)
parser.add_argument('--val_dir', default=None)
parser.add_argument('--ks', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6])
args = parser.parse_args()

print('=' * 80)
print('IEMOCAP ShapDistill + TT-Calibration aggregation')
print('=' * 80)

if args.pseudo_dir:
    fd = _load_fold_dir(args.pseudo_dir)
    if fd:
        _summarize(f'test_pseudo selection  ({args.pseudo_dir})', fd, args.ks)
if args.val_dir:
    fd = _load_fold_dir(args.val_dir)
    if fd:
        _summarize(f'val_true selection     ({args.val_dir})', fd, args.ks)
