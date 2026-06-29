#!/usr/bin/env python
"""Aggregate MMEE + CREMA-style results across 5 datasets.
HAR/IEMOCAP: mean +/- std over cross-subject folds. MOSEI: mean +/- std over 5 seeds.
Also reports the anytime exit-curve endpoints (cheapest -> full)."""
import os, glob, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METH = {'mmee': ('results_mmee', 'mmee'), 'crema': ('results_crema', 'crema_style')}
NFOLD = {'iemocap': 3, 'daliahar': 3, 'pamap2': 3, 'dsads': 4}
MOSEI_SEEDS = [7, 21, 88, 365, 1024]
CURVE_KEY = {'mmee': 'exit_curve_meanLayer_acc_f1', 'crema': 'exit_curve_meanMods_acc_f1'}


def grab(paths):
    accs, f1s, curves = [], [], []
    for p in paths:
        if not os.path.isfile(p):
            continue
        r = json.load(open(p))
        accs.append(r['best_test_acc']); f1s.append(r['best_test_f1'])
        ck = r.get('exit_curve_meanLayer_acc_f1') or r.get('exit_curve_meanMods_acc_f1')
        if ck:
            curves.append(np.array(ck))
    return accs, f1s, curves


out = {}
for meth, (rd, exp) in METH.items():
    rd = os.path.join(HERE, rd)
    print(f'\n===== {meth.upper()} =====')
    for ds in ['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei']:
        if ds == 'mosei':
            paths = [os.path.join(rd, f'{exp}_s{s}_mosei', 'run', 'results.json') for s in MOSEI_SEEDS]
            unit = f'{len(MOSEI_SEEDS)} seeds'
        else:
            paths = [os.path.join(rd, f'{exp}_{ds}', f'fold{f}', f'results_fold{f}.json') for f in range(NFOLD[ds])]
            unit = f'{NFOLD[ds]} folds'
        accs, f1s, curves = grab(paths)
        if not accs:
            print(f'  {ds:9s}: (no results yet, 0/{len(paths)})'); continue
        # cheapest & full operating points averaged across runs (curves may differ in length only by ties)
        cheap = np.mean([c[0, 1] for c in curves]) if curves else float('nan')
        full = np.mean([c[-1, 1] for c in curves]) if curves else float('nan')
        ck0 = np.mean([c[0, 0] for c in curves]) if curves else float('nan')
        ckN = np.mean([c[-1, 0] for c in curves]) if curves else float('nan')
        print(f'  {ds:9s}: full acc {np.mean(accs)*100:5.2f} ± {np.std(accs)*100:4.2f}  '
              f'F1 {np.mean(f1s)*100:5.2f}   exit acc {cheap*100:5.2f}->{full*100:5.2f} '
              f'@ K {ck0:.2f}->{ckN:.2f}  (n={len(accs)}/{len(paths)} {unit})')
        out[f'{meth}_{ds}'] = {'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
                               'f1_mean': float(np.mean(f1s)), 'n': len(accs),
                               'exit_cheap_acc': float(cheap), 'exit_full_acc': float(full)}
json.dump(out, open(os.path.join(HERE, 'mmee_crema_summary.json'), 'w'), indent=2)
print(f'\nWrote {os.path.join(HERE, "mmee_crema_summary.json")}')
