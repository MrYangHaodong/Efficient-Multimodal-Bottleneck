"""Order-insensitivity figure: the spread of test macro-F1 across acquisition ORDERS of a
FIXED modality subset, for a random-order-trained vs a fixed-order-trained IEMOCAP seqA
backbone.

Left: per-order F1, centred on each (model, |S|, fold) mean so the SPREAD is the visible
quantity rather than the level. Right: that spread itself -- the std across orderings,
averaged over folds.

Publication layout with a shared legend, boxed axes, and editable vector text.
Each run updates the same PDF, SVG, and PNG files next to this script.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import iclr_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'order_variance_iemocap')
MODELS = ['random-order trained', 'fixed-order trained']
# Match the paper's purple/orange comparison palette used by plot_cascade.py.
COL = {'random-order trained': '#AA4499', 'fixed-order trained': '#EE7733'}
MARKERS = {'random-order trained': 'o', 'fixed-order trained': 's'}


def load():
    """training -> subset size -> fold -> [macro-F1, one per ordering of that subset]."""
    D = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(open(f'{HERE}/data/order_variance_iemocap.csv')):
        D[r['training']][r['subset_size']][r['fold']].append(float(r['macro_f1']))
    return D


def main():
    D = load()
    KS = sorted(int(k) for k in D[MODELS[0]])
    iclr_style.apply()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.edgecolor': 'black',
        'svg.fonttype': 'none',
        'savefig.facecolor': 'white',
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.9),
                                   gridspec_kw=dict(width_ratios=[1.28, 1]))
    fig.subplots_adjust(left=.105, right=.99, bottom=.19, top=.845, wspace=.36)
    rng = np.random.default_rng(0)

    # ---- left: per-order deviation from each (model, k, fold) mean ----
    for mi, name in enumerate(MODELS):
        for k in KS:
            dev = []
            for fold, f1s in D[name][str(k)].items():
                a = np.array(f1s); dev.append(a - a.mean())
            dev = np.concatenate(dev)
            x = k + (mi - 0.5) * 0.34
            axL.scatter(x + rng.normal(0, .045, dev.size), dev * 100, s=15, alpha=.62,
                        color=COL[name], marker=MARKERS[name], edgecolors='none', zorder=3)
            axL.plot([x - .12, x + .12], [dev.std() * 100] * 2,
                     color=COL[name], lw=1.7, zorder=4)
            axL.plot([x - .12, x + .12], [-dev.std() * 100] * 2,
                     color=COL[name], lw=1.7, zorder=4)
    axL.axhline(0, color='#888888', lw=.8, ls=':', zorder=2)
    axL.set_xticks(KS)
    axL.set_ylabel('Macro-F1 deviation from\nsubset mean (pts)', labelpad=7)
    axL.margins(x=.06, y=.09)
    axL.set_yticks([-6, -4, -2, 0, 2, 4])

    # ---- right: std across orders ----
    w = .30
    for mi, name in enumerate(MODELS):
        stds = [np.mean([np.std(v) for v in D[name][str(k)].values()]) * 100 for k in KS]
        # Leave room for the short-bar labels beside the taller neighboring bars.
        positions = np.arange(len(KS)) + (mi - .5) * .5
        axR.bar(positions, stds, w, color=COL[name],
                edgecolor='black', linewidth=.65,
                hatch='///' if mi else None, zorder=3)
        for xi, v in zip(positions, stds):
            axR.text(xi, v + .055, f'{v:.2f}', ha='center', va='bottom',
                     fontsize=13.5, zorder=4)
    axR.set_xticks(np.arange(len(KS)), KS)
    # With the y-label rotated 90 degrees, a left arrow points down on the page.
    axR.set_ylabel('Order-wise Macro-F1 SD\n(pts, mean over folds)\n'
                   '\u2190 lower is better', labelpad=7)
    axR.set_ylim(0, 3.3)
    axR.set_yticks(np.arange(0, 3.1, .5))

    for panel, ax in zip(('a', 'b'), (axL, axR)):
        ax.grid(axis='y', color='#E5E7E9', linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out', length=3, width=.8, pad=4, colors='black')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        ax.text(.025, .96, f'({panel})', transform=ax.transAxes,
                ha='left', va='top', fontsize=16, color='black')

    handles = [
        Line2D([], [], linestyle='none', marker=MARKERS[name], markersize=7,
               markerfacecolor=COL[name], markeredgecolor=COL[name],
               label=name.capitalize())
        for name in MODELS
    ]
    handles.append(Line2D([], [], color='#444444', linewidth=1.7,
                          label=r'$\pm$1 SD (a)'))
    legend = fig.legend(
        handles=handles, ncol=len(handles), loc='lower center',
        bbox_to_anchor=(.5475, .859), frameon=True, fancybox=False,
        framealpha=1, facecolor='white', edgecolor='#B9B9B9',
        borderaxespad=0, borderpad=.3, columnspacing=1.2,
        handlelength=1.1, handletextpad=.45,
    )
    legend.get_frame().set_linewidth(.6)
    fig.supxlabel('Number of modalities in the fixed subset, $|S|$',
                  x=.5475, y=.026, fontsize=20)

    for ext in ('pdf', 'svg', 'png'):
        path = f'{OUT}.{ext}'
        fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=.05)
        print('wrote', path)
    plt.close(fig)

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
