#!/usr/bin/env python
"""Aggregate the CMU-MOSEI 30-epoch / 5-seed benchmark across all 10 methods
(GloVe Acc-2). Reports mean +/- std test Acc-2 and Macro-F1."""
import glob, json, os, re, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = ['7', '21', '88', '365', '1024']

# method -> list of results.json glob patterns (relative to HERE)
PAT = {
    'seqA':      ['results_mosei_5seed/*_seqA_acc2_s*/results.json'],
    'seqA_ps':   ['results_mosei_5seed/*_seqA_ps_acc2_s*/results.json'],
    'crossattn': ['results_mosei_5seed/*_crossattn_acc2_s*/results.json'],
    'multimodn': ['results_mosei_5seed/*_multimodn_acc2_s*/results.json'],
    'shaspec':   ['results_mosei_5seed/*_shaspec_acc2_s*/results.json'],
    'decalign':  ['results_mosei_5seed/*_decalign_acc2_s*/results.json'],
    'dymo':      ['results_mosei_5seed/*_dymo_acc2_s*/results.json'],
    'mmee':      ['results_mmee/mmee_s*_mosei/run/results.json'],
    'dmbf':      ['results_dmbf_mosei/*_dmbf_mosei_s*/results.json'],
    'l2r':       ['results_l2r_mosei/*_l2r_mosei_s*/results.json'],
}


def seed_of(p):
    m = re.search(r'_s(\d+)', p)
    return m.group(1) if m else '?'


rows = []
for meth, pats in PAT.items():
    accs, f1s, used = [], [], []
    for pat in pats:
        for p in glob.glob(os.path.join(HERE, pat)):
            if meth == 'seqA' and 'seqA_ps' in p:
                continue
            s = seed_of(p)
            if s not in SEEDS:
                continue
            r = json.load(open(p))
            a = r.get('best_test_acc'); f = r.get('best_test_f1')
            if a is not None:
                accs.append(a); f1s.append(f); used.append(s)
    if not accs:
        print(f'  {meth:11s}: (no results yet)'); continue
    rows.append((meth, np.mean(accs), np.std(accs), np.mean(f1s), sorted(used)))

rows.sort(key=lambda r: -r[1])
print(f"=== CMU-MOSEI 30-epoch / 5-seed benchmark (Acc-2) ===")
for meth, am, asd, fm, used in rows:
    print(f"  {meth:11s}: Acc-2 {am:.4f} ± {asd:.4f}   F1 {fm:.4f}   (n={len(used)} seeds={used})")
json.dump([{'method': m, 'acc_mean': a, 'acc_std': s, 'f1_mean': f, 'n': len(u)} for m, a, s, f, u in rows],
          open(os.path.join(HERE, 'results_mosei_5seed', 'summary_30ep_10methods.json'), 'w'), indent=2)
print('\nwrote results_mosei_5seed/summary_30ep_10methods.json')
