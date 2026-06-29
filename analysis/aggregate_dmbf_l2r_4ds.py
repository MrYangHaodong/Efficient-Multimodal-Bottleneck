#!/usr/bin/env python
"""Aggregate DMBF + L2R on the 4 HAR/IEMOCAP datasets: mean +/- std test acc / F1 over folds."""
import glob, json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
METH = {'dmbf': ('results_dmbf_4ds', 'dmbf'), 'l2r': ('results_l2r_4ds', 'l2r')}
NFOLD = {'iemocap': 3, 'daliahar': 3, 'pamap2': 3, 'dsads': 4}
for meth, (rd, exp) in METH.items():
    print(f'\n===== {meth.upper()} (4 datasets, cross-subject folds) =====')
    for ds in ['iemocap', 'daliahar', 'pamap2', 'dsads']:
        accs, f1s = [], []
        for f in range(NFOLD[ds]):
            p = os.path.join(HERE, rd, f'{exp}_{ds}', f'fold{f}', f'results_fold{f}.json')
            if os.path.isfile(p):
                r = json.load(open(p)); accs.append(r['best_test_acc']); f1s.append(r['best_test_f1'])
        if accs:
            print(f'  {ds:9s}: acc {np.mean(accs)*100:5.2f} ± {np.std(accs)*100:4.2f}  '
                  f'F1 {np.mean(f1s)*100:5.2f}  (n={len(accs)}/{NFOLD[ds]})')
        else:
            print(f'  {ds:9s}: (no results yet)')
