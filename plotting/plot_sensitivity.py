#!/usr/bin/env python
"""Sensitivity figure for the seqA backbone knobs, styled after ICLR_94/experts_vs_wordlength.pdf.

Three panels -- the two non-CE loss weights and the bottleneck width. F1 is order-averaged
over 24 permutations of the full modality set. The canonical-order series was dropped: it ran
almost parallel to the order-averaged one, so it cost a second line and a legend to say
something the appendix already states in a sentence.
Error bars are the std across folds; they are the point of the figure, because the
apparent zigzag in F1 is smaller than the fold-to-fold spread.

Reads runs/csvs/sensitivity.csv so the figure can never drift from the table.
"""
import csv, os, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                   'sensitivity_iemocap.csv')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sensitivity_iemocap')
BLUE, RED = '#4477AA', '#AA3355'          # paper's tolblue + a burgundy-leaning red

def load():
    """data/sensitivity_iemocap.csv -> one panel spec per swept parameter."""
    LABEL = {'lambda_DS': r'$\lambda_{\mathrm{DS}}$  (deep supervision)',
             'lambda_KD': r'$\lambda_{\mathrm{KD}}$  (distillation)',
             'K':         r'$K$  (bottleneck tokens)'}
    panels, seen = [], {}
    for r in csv.DictReader(open(CSV)):
        k = r['param']
        if k not in seen:
            seen[k] = dict(key=k, xlabel=LABEL[k], v=[])
            panels.append(seen[k])
        seen[k]['v'].append(dict(x=float(r['value']),
                                 f1=float(r['f1_order_averaged']),
                                 sd=float(r['f1_std_folds']),
                                 dep=r['deployed'] == 'yes'))
    return panels


def main():
    P = load()
    assert len(P) == 3, [p['key'] for p in P]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.05))
    for ax, p in zip(axes, P):
        v = sorted(p['v'], key=lambda d: d['x'])
        xs = np.arange(len(v))
        f1 = np.array([d['f1'] for d in v]); sd = np.array([d['sd'] for d in v])
        ax.errorbar(xs, f1, yerr=sd, color=BLUE, ecolor=BLUE, alpha=1.0, lw=1.9,
                    marker='o', ms=5.4, mec='white', mew=1.0, elinewidth=0.9,
                    capsize=2.0, capthick=0.9, zorder=3)
        for c in ax.containers:                       # fade only the bars, not the line
            if hasattr(c, 'lines') and c.lines[2]:
                for seg in c.lines[2]: seg.set_alpha(0.32)
                for cap in c.lines[1]: cap.set_alpha(0.45)
        ax.set_xticks(xs)
        ax.set_xticklabels([('%g' % d['x']) for d in v])
        for d, lab in zip(v, ax.get_xticklabels()):          # flag the shipped setting
            if d['dep']:
                lab.set_color(BLUE); lab.set_fontweight('bold')
        ax.grid(True, which='major', ls='--', lw=0.7, color='0.82', zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_linewidth(1.0); s.set_color('0.25')
        ax.set_xlabel(p['xlabel'], fontsize=9, labelpad=2)
        ax.tick_params(labelsize=8, length=2.5, pad=2)
        ax.set_ylim(0.605, 0.722); ax.set_yticks([0.62, 0.65, 0.68, 0.71])
    axes[0].set_ylabel('Macro-F1', fontsize=9, labelpad=2)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.subplots_adjust(left=0.082, right=0.996, top=0.97, bottom=0.235, wspace=0.08)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', dpi=300)
    print('wrote', OUT + '.pdf')
    for p in P:
        v = sorted(p['v'], key=lambda d: d['x'])
        print(f"  {p['key']:7s} x={[d['x'] for d in v]}")
        print(f"          f1={[round(d['f1'],4) for d in v]}")

if __name__ == '__main__':
    main()
