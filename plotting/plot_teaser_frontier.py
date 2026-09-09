"""Teaser: the accuracy/cost frontier on IEMOCAP as modalities go missing.

Each method is a short polyline over three missingness rates (0/20/40%), so the reader sees
not just where a method sits but which way it MOVES as inputs degrade. Static-cost methods
travel straight down; SPARQ travels down-and-LEFT, because the policy buys fewer streams when
fewer are available.
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import iclr_style as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'teaser_frontier')

# hand-placed so the nine baseline labels do not collide on the log axis
LABEL_OFF = {'Vanilla MBT (R)': (23, 0), 'MAESTRO (R)': (0, -13), 'ShaSpec (R)': (0, 9),
             'DecALign (R)': (-27, -1), 'MultiModN (R, best static order)': (0, 9),
             'DyMo (E)': (16, 0), 'AdaMML (E)': (-22, 2), 'DyMM (E)': (15, -4),
             'MMEE': (14, 5)}


def load():
    """method -> (gflops, macro-F1) over the three missingness rates, in file order."""
    out = {}
    for r in csv.DictReader(open(f'{HERE}/data/teaser_frontier_iemocap.csv')):
        gf, f1 = out.setdefault(r['method'], ([], []))
        gf.append(float(r['gflops']))
        f1.append(float(r['f1']))
    return {k: (np.array(g), np.array(f)) for k, (g, f) in out.items()}


def main():
    S.apply()
    D = load()
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    S.grid(ax)
    for name, (gf, f1) in D.items():
        if name.startswith('Ours'):
            continue
        ax.plot(gf, f1, color='0.62', lw=1.2, marker='o', ms=3.2, mec='white', mew=0.7, zorder=2)
        dx, dy = LABEL_OFF.get(name, (0, 7))
        ax.annotate(name.split(' (')[0], (gf[0], f1[0]), textcoords='offset points',
                    xytext=(dx, dy), ha='center', fontsize=6.2, color='0.35', zorder=5)
    gf, f1 = D['Ours w/o RL']
    ax.plot(gf, f1, color=S.RED, lw=1.7, ls='--', marker='s', ms=4.6, mec='white', mew=1.0,
            zorder=3, label='SPARQ w/o RL (all modalities)')
    gf, f1 = D['Ours']
    ax.plot(gf, f1, color=S.BLUE, lw=2.4, marker='o', ms=6.0, mec='white', mew=1.2,
            zorder=4, label='SPARQ (Ours)')
    ax.text(0.015, 0.955, 'missingness $0\\!\\rightarrow\\!40\\%$:\naccuracy holds, cost shrinks',
            transform=ax.transAxes, fontsize=6.6, color=S.BLUE, ha='left', va='top', zorder=5)
    ax.set_xscale('log')
    ax.set_xlabel('Inference cost (GFLOPs, log scale)')
    ax.set_ylabel('Macro-F1')
    ax.set_xlim(0.09, 20)
    ax.set_ylim(0.33, 0.745)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_xticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10'])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7])
    ax.legend(loc='lower center', fontsize=6.6)
    fig.subplots_adjust(left=0.108, right=0.985, top=0.97, bottom=0.165)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', dpi=300)
    print('wrote', OUT + '.pdf')
    for k, (g, f) in D.items():
        print(f'  {k:34s} GF={np.round(g, 3)}  F1={np.round(f, 3)}')


if __name__ == '__main__':
    main()
