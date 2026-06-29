#!/usr/bin/env python
"""Aggregate faithful original-DynMM-3branch (no distill) over folds, 5 datasets, + compare to
seqA-DynMM where available. Reads results_3p3/dynmm_orig_<ds>/fold<f>/gate/dynmm_orig_gate/results_fold<f>.json"""
import glob, json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R3 = os.path.join(HERE, '..', 'results_3p3')
NF = {'iemocap': 3, 'daliahar': 3, 'pamap2': 3, 'dsads': 4, 'eav': 3}
print(f"{'dataset':9s} | {'orig-DynMM Acc':>14s} {'F1':>6s} | {'usage 1mod/part/full':>22s} | {'GFLOPs':>7s} | seqA-DynMM Acc")
for ds, nf in NF.items():
    A, F, U, G = [], [], [], []
    for f in range(nf):
        p = os.path.join(R3, f'dynmm_orig_{ds}', f'fold{f}', 'gate', 'dynmm_orig_gate', f'results_fold{f}.json')
        if os.path.isfile(p):
            d = json.load(open(p)); A.append(d['best_test_acc']); F.append(d['best_test_f1'])
            U.append(d['usage_1mod_partial_full']); G.append(d['mean_gflops'])
    # seqA-DynMM comparison (gate results, if present)
    sp = glob.glob(os.path.join(R3, f'dynmm_{ds}', 'dynmm_seqA', '*', 'results_fold*.json'))
    sa = [json.load(open(q)).get('best_test_acc') for q in sp]
    sa = [x for x in sa if x is not None]
    if A:
        um = np.array(U).mean(0)
        seqa = f"{np.mean(sa)*100:.2f} (n={len(sa)})" if sa else "—"
        print(f"{ds:9s} | {np.mean(A)*100:8.2f}±{np.std(A)*100:4.2f} {np.mean(F)*100:5.2f} | "
              f"{um[0]*100:4.0f}/{um[1]*100:4.0f}/{um[2]*100:4.0f}%       | {np.mean(G):7.3f} | {seqa}  [n={len(A)}/{nf}]")
    else:
        print(f"{ds:9s} | (no results yet)")
