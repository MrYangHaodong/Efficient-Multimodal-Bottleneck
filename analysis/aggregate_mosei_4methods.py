#!/usr/bin/env python
"""Aggregate CMU-MOSEI Acc-2 (mean±std over seeds 7,21,365) for the 4 per-sample/adaptive
baselines: DMBF, MMEE, DynMM, AdaMML. + seqA / seqA_ps for reference."""
import glob, json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [7, 21, 365]


def acc_f1(p):
    try: d = json.load(open(p))
    except Exception: return None
    a = next((d[k] for k in ['best_test_acc', 'test_acc', 'best_acc', 'acc'] if k in d), None)
    f = next((d[k] for k in ['best_test_f1', 'test_f1', 'best_f1', 'f1'] if k in d), None)
    return None if a is None else (a, f)


# method -> per-seed glob template (use {s})
SRC = {
    'DMBF':    'results_dmbf_mosei/*dmbf_mosei_s{s}/results.json',
    'MMEE':    'results_mmee/mmee_s{s}_mosei/run/results.json',
    'DynMM':   '../results_3p3/dynmm_mosei_s{s}/dynmm_seqA/dynmm_seqA_mosei/results_fold0.json',
    'AdaMML':  '../results_3p3/adamml_mosei/adamml_s{s}/results_fold0.json',
    'seqA':    'results_mosei_5seed/*seqA_acc2_s{s}_seqA/results*.json',
    'seqA_ps': 'results_mosei_5seed/*seqA_ps_acc2_s{s}_seqA/results*.json',
}
print(f"{'method':9s} | {'Acc-2 mean±std':>16s} | {'F1 mean±std':>16s} | per-seed acc (7/21/365)  [n]")
for m, tmpl in SRC.items():
    accs, f1s, per = [], [], []
    for s in SEEDS:
        hits = glob.glob(os.path.join(HERE, tmpl.format(s=s)))
        r = acc_f1(sorted(hits)[0]) if hits else None
        if r:
            accs.append(r[0]); per.append(f"{r[0]:.3f}")
            if r[1] is not None: f1s.append(r[1])
        else:
            per.append("  -  ")
    if accs:
        fa = f"{np.mean(f1s)*100:5.2f}±{np.std(f1s)*100:4.2f}" if f1s else "   n/a   "
        print(f"{m:9s} | {np.mean(accs)*100:6.2f} ± {np.std(accs)*100:4.2f}  | {fa:>16s} | {' / '.join(per)}  [{len(accs)}/3]")
    else:
        print(f"{m:9s} | (no results found) | tmpl={tmpl}")
