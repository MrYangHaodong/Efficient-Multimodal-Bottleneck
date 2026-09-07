#!/usr/bin/env python
"""Class-conditional acquisition, restyled for the paper (see iclr_style).

    Bar-top value labels and error bars are deliberately omitted: the figure carries the
    class-conditional PATTERN, and the exact probabilities and the fold-to-fold spread are
    quoted in the caption instead, so nothing is lost by keeping the panels clean.

Same 1x4 layout and same numbers as before; what changes is the styling -- boxed dark spines,
dashed grid behind the bars, the colourblind-safe Tol palette, faded error bars so the bar
heights read first, and a shared y-axis so the four classes are visually comparable.
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import iclr_style as S


def _dark(hexcolor, k=0.55):
    """Darker shade of a bar colour, so its whisker stays legible against the bar itself."""
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % (int(r * k), int(g * k), int(b * k))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'selection_by_class_iemocap')
MODS  = ['video', 'audio', 'text', 'mocap_hand', 'mocap_head', 'mocap_rotated']
SHORT = ['video', 'audio', 'text', 'm-hand', 'm-head', 'm-rot']

def main():
    S.apply()
    rows = list(csv.DictReader(open(os.path.join(HERE, 'data', 'selection_by_class_iemocap.csv'))))
    fig, axes = plt.subplots(1, len(rows), figsize=(7.6, 2.05), sharey=True)
    x = np.arange(len(MODS))
    for ax, r in zip(axes, rows):
        f = np.array([float(r[f'freq_{m}']) for m in MODS])
        S.grid(ax, axis='y')
        bars = ax.bar(x, f, width=0.72, color=S.CATEGORICAL, edgecolor='white',
                      linewidth=0.6, zorder=3)
        ax.set_title(r['class_name'], fontsize=8.2, pad=3)
        ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=38, ha='right')
        ax.tick_params(axis='x', labelsize=6.8)
        ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    axes[0].set_ylabel(r'$P(j \in S \mid y)$')
    fig.subplots_adjust(left=0.072, right=0.995, top=0.885, bottom=0.235, wspace=0.10)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', dpi=300)
    print('wrote', OUT + '.pdf')
    for r in rows:
        print(f"  {r['class_name']:8s} n={r['n']:>4s} |S|={float(r['mean_k']):.2f} "
              f"acc={float(r['acc']):.3f}")

if __name__ == '__main__':
    main()
