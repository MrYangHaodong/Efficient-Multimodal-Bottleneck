#!/usr/bin/env python
"""Aggregate CMU-MOSEI 5-seed runs under results_mosei_5seed/.

For each model tag, finds the per-seed exp dirs ({date}_CMU_MOSEI_{tag}_acc2_s{seed}),
reads results.json, and reports mean +/- std of test acc / F1 across seeds.
"""
import os, re, json, glob
import numpy as np

RD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_mosei_5seed')
TAGS = ['seqA', 'seqA_ps', 'crossattn', 'multimodn', 'decalign', 'dymo', 'shaspec']


def _pick(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def main():
    print(f'Aggregating under {RD}\n')
    rows = []
    for tag in TAGS:
        # exp dirs look like 2026-06-21_CMU_MOSEI_{tag}_acc2_s{seed}
        dirs = sorted(glob.glob(os.path.join(RD, f'*_{tag}_acc2_s*')))
        accs, f1s, seeds = [], [], []
        for d in dirs:
            rj = os.path.join(d, 'results.json')
            if not os.path.isfile(rj):
                continue
            with open(rj) as f:
                r = json.load(f)
            acc = _pick(r, 'best_test_acc', 'test_acc')
            f1 = _pick(r, 'best_test_f1', 'test_f1')
            m = re.search(r'_s(\d+)(?:_|$)', d)   # seqA dir ends with _seqA suffix
            seeds.append(m.group(1) if m else '?')
            if acc is not None:
                accs.append(float(acc))
            if f1 is not None:
                f1s.append(float(f1))
        if not accs:
            print(f'{tag:12s}: no results yet ({len(dirs)} dirs found)')
            continue
        amean, astd = np.mean(accs), np.std(accs)
        fmean = np.mean(f1s) if f1s else float('nan')
        fstd = np.std(f1s) if f1s else float('nan')
        print(f'{tag:12s}: Acc-2 {amean:.4f} +/- {astd:.4f}   '
              f'F1 {fmean:.4f} +/- {fstd:.4f}   '
              f'(n={len(accs)} seeds={seeds})')
        rows.append({'model': tag, 'acc_mean': amean, 'acc_std': astd,
                     'f1_mean': fmean, 'f1_std': fstd, 'n': len(accs),
                     'accs': accs, 'f1s': f1s, 'seeds': seeds})
    out = os.path.join(RD, 'summary_5seed.json')
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
