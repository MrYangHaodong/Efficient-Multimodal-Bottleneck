#!/usr/bin/env python
"""Aggregate cross-subject (alpha=0) full-modality acc/F1 across all baselines.
Trained methods (seqA/crossattn/multimodn/dymo/decalign/shaspec): mean+/-std of
best_test_{acc,f1} over results_fold{0,1,2}.json.  DynMM = shift_sweep 'gate';
ADMN = shift_sweep 'admn.controller_*'."""
import json, glob, os, re
import numpy as np

R = os.path.dirname(os.path.abspath(__file__))
DS = ['daliahar', 'pamap2', 'wesad', 'iemocap']


def folds_stat(method_ds_glob, acc_key='best_test_acc', f1_key='best_test_f1'):
    """mean,std of acc/f1 over per-fold jsons; returns (n, acc_m, acc_s, f1_m, f1_s)."""
    # dedupe by fold number (newest file per fold across dated dirs)
    by_fold = {}
    for f in glob.glob(method_ds_glob):
        mo = re.search(r'results_fold(\d+)\.json$', f)
        if not mo:
            continue
        k = int(mo.group(1))
        if k not in by_fold or os.path.getmtime(f) > os.path.getmtime(by_fold[k]):
            by_fold[k] = f
    accs, f1s = [], []
    for k in sorted(by_fold):
        d = json.load(open(by_fold[k]))
        if acc_key in d and d[acc_key] is not None:
            accs.append(float(d[acc_key])); f1s.append(float(d.get(f1_key, np.nan)))
    if not accs:
        return (0, None, None, None, None)
    return (len(accs), np.mean(accs), np.std(accs), np.nanmean(f1s), np.nanstd(f1s))


def shift_stat(subdir, ds, sub, acc_k, f1_k):
    accs, f1s = [], []
    for fold in (0, 1, 2):
        p = os.path.join(R, subdir, f'{ds}_a0.0_f{fold}.json')
        if os.path.exists(p):
            d = json.load(open(p)).get(sub, {})
            if acc_k in d:
                accs.append(d[acc_k]); f1s.append(d.get(f1_k, np.nan))
    if not accs:
        return (0, None, None, None, None)
    return (len(accs), np.mean(accs), np.std(accs), np.nanmean(f1s), np.nanstd(f1s))


def get(method, ds):
    if method == 'seqA':
        return folds_stat(os.path.join(R, f'seqA_{ds}/*/results_fold*.json'))
    if method in ('crossattn', 'multimodn', 'dymo', 'decalign', 'shaspec',
                  'qmf', 'pdf', 'dymo_proto'):
        return folds_stat(os.path.join(R, f'{method}_{ds}/*/results_fold*.json'))
    if method == 'DynMM':
        return shift_stat('shift_sweep_3p3', ds, 'gate', 'acc', 'f1')
    if method == 'ADMN':
        return shift_stat('shift_sweep_admn3', ds, 'admn', 'controller_acc', 'controller_f1')


METHODS = ['seqA', 'crossattn', 'multimodn', 'dymo', 'DynMM', 'ADMN',
           'decalign', 'shaspec', 'qmf', 'pdf', 'dymo_proto']

for metric, mi in [('Accuracy', (1, 2)), ('Macro-F1', (3, 4))]:
    print(f'\n===== {metric} (cross-subject alpha=0, full modalities, mean+/-std / 3 folds) =====')
    print(f'{"method":<11}' + ''.join(f'{ds:>16}' for ds in DS))
    for m in METHODS:
        row = f'{m:<11}'
        for ds in DS:
            n, am, as_, fm, fs_ = get(m, ds)
            mean = (am, as_) if metric == 'Accuracy' else (fm, fs_)
            if n == 0:
                row += f'{"--":>16}'
            elif n < 3:
                row += f'{f"{mean[0]*100:.1f}({n}/3)":>16}'
            else:
                row += f'{f"{mean[0]*100:.1f}+/-{mean[1]*100:.1f}":>16}'
        print(row)
print('\n(DynMM=learned gate; ADMN=depth controller; both on shared 3enc+3fus backbone.'
      '\n dymo=confidence-greedy; dymo_proto=faithful DyMo (prototype+Gaussian greedy).)')

# ---- faithful-DyMo: full-modality vs greedy-select + avg modalities kept ----
print('\n===== Faithful DyMo (dymo_proto): FULL vs greedy-SELECT (acc) + avg-K =====')
print(f'{"dataset":<11}{"FULL acc":>16}{"SELECT acc":>16}{"avg kept":>14}')
for ds in DS:
    g = os.path.join(R, f'dymo_proto_{ds}/*/results_fold*.json')
    nf, fam, fas, _, _ = folds_stat(g, 'best_test_acc', 'best_test_f1')
    ns, sam, sas, _, _ = folds_stat(g, 'select_test_acc', 'select_test_f1')
    nk, avk, avks, _, _ = folds_stat(g, 'avg_selected', 'avg_selected')
    if nf == 0:
        print(f'{ds:<11}{"--":>16}{"--":>16}{"--":>14}'); continue
    print(f'{ds:<11}{f"{fam*100:.1f}+/-{fas*100:.1f}":>16}'
          f'{f"{sam*100:.1f}+/-{sas*100:.1f}":>16}{f"{avk:.2f}":>14}')
