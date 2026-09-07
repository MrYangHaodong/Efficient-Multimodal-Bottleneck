#!/usr/bin/env python
"""Availability cascade, restyled for the paper (see iclr_style).

Replaces the 2x3 diagnostic dump with two panels. The five first-pick bar panels collapse
into annotations on the accuracy panel -- the first pick is deterministic within a fold under
a fixed availability mask, so a whole panel per config was spending a lot of ink on one label.
Left: macro-F1 as the available set shrinks, for SPARQ and for the same frozen backbone
consuming everything available. Right: what that costs, as modality usage.
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import iclr_style as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'firstpick_cascade_iemocap')
SHORT = {'text': 'text', 'audio': 'audio', 'video': 'video',
         'mocap_hand': 'm‑hand', 'mocap_head': 'm‑head', 'mocap_rotated': 'm‑rot'}

def load():
    """data/firstpick_cascade_iemocap.csv -- the RL walk and the consume-all reference."""
    rows = list(csv.DictReader(open(os.path.join(HERE, 'data',
                                                 'firstpick_cascade_iemocap.csv'))))
    f = lambda k: np.array([float(r[k]) for r in rows])
    return (rows, [int(r['n_avail']) for r in rows], f('f1_mean'), f('f1_std'),
            f('ref_f1_mean'), f('ref_f1_std'), f('mu_mean'),
            [r['first_pick'] for r in rows],
            ['all' if not r['removed_to_reach'] else '\u2212' + r['removed_to_reach']
             for r in rows])


def main():
    S.apply()
    rows, nA, f1, sd, rf1, rsd, mu, first, removed = load()
    x = np.arange(len(rows))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 2.15))

    axL.errorbar(x, f1, yerr=sd, color=S.BLUE, ecolor=S.BLUE, marker='o',
                 elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=3, label='SPARQ')
    axL.errorbar(x, rf1, yerr=rsd, color=S.RED, ecolor=S.RED, ls='--', marker='s',
                 ms=4.6, elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=2,
                 label='consumes all available')
    S.fade_errorbars(axL)
    for xi, yi, m in zip(x, f1, first):                 # the first pick, per config
        axL.annotate(m, (xi, yi), textcoords='offset points', xytext=(0, 11),
                     ha='center', fontsize=6.6, color=S.BLUE, fontweight='bold')
    axL.set_ylabel('Macro-F1'); axL.set_ylim(0.15, 0.80)
    axL.set_yticks([0.2, 0.4, 0.6, 0.8])

    axR.plot(x, mu, color=S.BLUE, marker='o', zorder=3, label='SPARQ')
    axR.axhline(1.0, color=S.RED, ls='--', lw=1.9, zorder=2, label='consumes all available')
    axR.set_ylabel('Modality usage'); axR.set_ylim(0.0, 1.12)
    axR.set_yticks([0.0, 0.5, 1.0])
    axR.legend(loc='lower center')

    for ax in (axL, axR):
        S.grid(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{n}\n{r}' for n, r in zip(nA, removed)])
        ax.set_xlim(-0.42, len(x) - 0.58)          # headroom so the annotations do not clip
        ax.set_xlabel(r'available modalities $|\mathcal{A}|$', labelpad=1)
    fig.subplots_adjust(left=0.072, right=0.988, top=0.955, bottom=0.30, wspace=0.17)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', dpi=300)
    print('wrote', OUT + '.pdf')
    print('  |A| :', nA, '\n  first:', first, '\n  removed:', removed)

if __name__ == '__main__':
    main()
