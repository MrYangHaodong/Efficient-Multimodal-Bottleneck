"""seqA adaptive early-exit: ENTROPY head vs GAP head, GFLOPs vs Acc / Macro-F1 on
4 datasets (cross-subject, val-LOO modality order). Both curves are the same model
config; only the classifier head differs (gap = global avg pool; entropy = per-
modality entropy-weighted late fusion). Output -> entropy_vs_gap_exit_4ds.png."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads']
M_OF = {'iemocap': 6, 'daliahar': 5, 'pamap2': 4, 'dsads': 5}
TAG = {'iemocap': 'T=128, float', 'daliahar': 'T=256, sax', 'pamap2': 'T=512, sax_noisy', 'dsads': 'T=125, sax'}

EXIT_IE = json.load(open(os.path.join(R, 'seqA_statick_3p3.json')))
GF_IE = json.load(open(os.path.join(R, 'seqA3p3_gflops.json')))
CMP256 = json.load(open(os.path.join(R, 'compare_seqexit_vs_admn_T256.json')))
P2EX = json.load(open(os.path.join(R, 'seqA_pamap2', 'exit_T512_saxnoisy.json')))
DSEX = json.load(open(os.path.join(R, 'dsads', 'exit_dsads.json')))
ENT = json.load(open(os.path.join(R, 'entropy_head', 'exit_entropy_4ds.json')))


def gap_curve(ds, which):       # which: 'acc'/'f1'
    M = M_OF[ds]
    if ds == 'iemocap':
        gpk = GF_IE[ds]['gflops_per_k']; mk = np.array(EXIT_IE[ds]['exit_meanK']); y = np.array(EXIT_IE[ds]['exit_' + which])
    else:
        src = {'daliahar': CMP256.get('daliahar'), 'pamap2': P2EX['pamap2'], 'dsads': DSEX['dsads']}[ds]
        ec = np.array(src['exit_curve']); gpk = src['gflops_per_k']; mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    gf = np.interp(mk, np.arange(1, M + 1), gpk); o = np.argsort(gf)
    return gf[o], y[o]


def ent_curve(ds, which):
    M = M_OF[ds]; o = ENT[ds]; ec = np.array(o['exit_curve']); gpk = o['gflops_per_k']
    mk = ec[:, 0]; y = ec[:, 1 if which == 'acc' else 2]
    gf = np.interp(mk, np.arange(1, M + 1), gpk); idx = np.argsort(gf)
    return gf[idx], y[idx]


fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
for col, ds in enumerate(DS):
    for row, (which, ylab) in enumerate([('acc', 'Accuracy'), ('f1', 'Macro-F1')]):
        ax = axes[row, col]
        gx, gy = gap_curve(ds, which); ex, ey = ent_curve(ds, which)
        ax.plot(gx, gy, '-o', color='tab:blue', lw=2.4, ms=4, zorder=6, label='GAP head')
        ax.plot(ex, ey, '--s', color='tab:red', lw=2.0, ms=4, zorder=7, label='ENTROPY head')
        ax.set_xscale('log'); ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
        ax.set_title(f'{ds} (M={M_OF[ds]}, {TAG[ds]})' + ('  [seqA exit, cross-subject]' if row == 0 else ''), fontsize=10)
        ax.grid(alpha=0.25, which='both')
        if col == 0 and row == 0:
            ax.legend(fontsize=9, loc='lower right')
fig.suptitle('seqA adaptive early-exit: ENTROPY head vs GAP head (GFLOPs vs Acc / Macro-F1, cross-subject, val-LOO modality order).\n'
             'Same backbone/config; only the classifier head differs. Both sweep the compute axis via per-sample entropy early-exit.', fontsize=11)
fig.subplots_adjust(left=0.045, right=0.997, bottom=0.08, top=0.88, wspace=0.27, hspace=0.30)
o = os.path.join(R, 'entropy_vs_gap_exit_4ds.png')
fig.savefig(o, dpi=140); print('saved ->', o)
for ds in DS:
    gx, gy = gap_curve(ds, 'acc'); ex, ey = ent_curve(ds, 'acc')
    print(f"{ds:9s}: GAP acc {gy.min():.3f}->{gy.max():.3f} | ENTROPY {ey.min():.3f}->{ey.max():.3f} @ GF {ex.min():.2f}-{ex.max():.2f}")
