"""Order-insensitivity figure: the spread of test macro-F1 across acquisition ORDERS of a
FIXED modality subset, for a random-order-trained vs a fixed-order-trained IEMOCAP seqA
backbone.

Left: per-order F1, centred on each (model, |S|, fold) mean so the SPREAD is the visible
quantity rather than the level. Right: that spread itself -- the std across orderings,
averaged over folds.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'order_variance_iemocap')
MODELS = ['random-order trained', 'fixed-order trained']
COL = {'random-order trained': '#228833', 'fixed-order trained': '#77324C'}


def load():
    """training -> subset size -> fold -> [macro-F1, one per ordering of that subset]."""
    D = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(open(f'{HERE}/data/order_variance_iemocap.csv')):
        D[r['training']][r['subset_size']][r['fold']].append(float(r['macro_f1']))
    return D


def main():
    D = load()
    KS = sorted(int(k) for k in D[MODELS[0]])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.5, 2.6),
                                   gridspec_kw=dict(width_ratios=[1.35, 1]))
    rng = np.random.default_rng(0)

    # ---- left: per-order deviation from each (model, k, fold) mean ----
    for mi, name in enumerate(MODELS):
        for k in KS:
            dev = []
            for fold, f1s in D[name][str(k)].items():
                a = np.array(f1s); dev.append(a - a.mean())
            dev = np.concatenate(dev)
            x = k + (mi - 0.5) * 0.34
            axL.scatter(x + rng.normal(0, .045, dev.size), dev * 100, s=4.5, alpha=.55,
                        color=COL[name], edgecolors='none',
                        label=name if k == KS[0] else None)
            axL.plot([x - .12, x + .12], [dev.std() * 100] * 2, color=COL[name], lw=1.1)
            axL.plot([x - .12, x + .12], [-dev.std() * 100] * 2, color=COL[name], lw=1.1)
    axL.axhline(0, color='0.6', lw=.6, ls=':')
    axL.set_xticks(KS); axL.set_xlabel('modalities in the fixed subset  $|S|$', fontsize=7.4)
    axL.set_ylabel('macro-F1 deviation from\nthe subset mean (pts)', fontsize=7.4)
    axL.tick_params(labelsize=6.8, length=2, pad=1.5)
    axL.set_title('every ordering of the same subset', fontsize=8)
    axL.legend(fontsize=6.2, loc='lower left', framealpha=.9, borderpad=.3, handletextpad=.4)
    for s in ('top', 'right'): axL.spines[s].set_visible(False)

    # ---- right: std across orders ----
    w = 0.36
    for mi, name in enumerate(MODELS):
        stds = [np.mean([np.std(v) for v in D[name][str(k)].values()]) * 100 for k in KS]
        axR.bar(np.arange(len(KS)) + (mi - .5) * w, stds, w, color=COL[name],
                edgecolor='black', linewidth=.5, label=name)
        for xi, v in zip(np.arange(len(KS)) + (mi - .5) * w, stds):
            axR.text(xi, v + .06, f'{v:.2f}', ha='center', va='bottom', fontsize=5.8)
    axR.set_xticks(np.arange(len(KS))); axR.set_xticklabels(KS, fontsize=6.8)
    axR.set_xlabel('modalities in the fixed subset  $|S|$', fontsize=7.4)
    axR.set_ylabel('std across orderings (pts)', fontsize=7.4)
    axR.tick_params(labelsize=6.8, length=2, pad=1.5)
    axR.set_title('spread is $\\sim$5$\\times$ smaller', fontsize=8)
    axR.grid(axis='y', ls=':', lw=.45, alpha=.55); axR.set_axisbelow(True)
    for s in ('top', 'right'): axR.spines[s].set_visible(False)

    fig.tight_layout(w_pad=1.4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', dpi=300, bbox_inches='tight')
    print('wrote', OUT + '.pdf')

    print('\n=== std across orderings (F1 points), mean over folds ===')
    print(f"{'|S|':>4s} {'random-order':>13s} {'fixed-order':>12s} {'ratio':>6s}")
    for k in KS:
        a = np.mean([np.std(v) for v in D[MODELS[0]][str(k)].values()]) * 100
        b = np.mean([np.std(v) for v in D[MODELS[1]][str(k)].values()]) * 100
        print(f'{k:4d} {a:13.3f} {b:12.3f} {b / a:6.1f}x')
    allr = np.concatenate([np.array(v) - np.mean(v)
                           for k in KS for v in D[MODELS[0]][str(k)].values()])
    allf = np.concatenate([np.array(v) - np.mean(v)
                           for k in KS for v in D[MODELS[1]][str(k)].values()])
    print(f'\npooled std: random-order {allr.std() * 100:.3f} pts   '
          f'fixed-order {allf.std() * 100:.3f} pts   ratio {allf.std() / allr.std():.1f}x')
    print(f'pooled worst-case range: random-order {(allr.max() - allr.min()) * 100:.2f} pts'
          f'   fixed-order {(allf.max() - allf.min()) * 100:.2f} pts')


if __name__ == '__main__':
    main()
