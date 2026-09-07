#!/usr/bin/env python
"""Class-conditional acquisition for the datasets other than IEMOCAP (see iclr_style).

Companion to plot_selection_by_class.py, which draws the IEMOCAP figure used in the paper.

IEMOCAP has C=4, so one bar panel per class fits. The others run C=5..27, where that layout
collapses -- so the same quantity, P(j in S | y), is drawn as a class x modality heatmap.
EAV (C=5) keeps the bar form so it stays directly comparable with the IEMOCAP figure.
Right-hand strip on each heatmap is the per-class mean acquired-set size |S|.
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import iclr_style as S

HERE = os.path.dirname(os.path.abspath(__file__))
PRETTY = {'mocap_hand': 'm-hand', 'mocap_head': 'm-head', 'mocap_rotated': 'm-rot',
          'tof_1': 'ToF$_1$', 'tof_2': 'ToF$_2$', 'tof_3': 'ToF$_3$', 'tof_4': 'ToF$_4$',
          'tof_5': 'ToF$_5$', 'thm': 'THM', 'imu': 'IMU'}
TITLE = {'mmfi': 'MM-Fi', 'cmi': 'CMI', 'czu_mhad': 'CZU-MHAD', 'dsads': 'DSADS',
         'eav': 'EAV', 'utd_mhad': 'UTD-MHAD'}
# white -> tolblue, so "never acquired" reads as empty and "always" as solid
CMAP = LinearSegmentedColormap.from_list('tolblue_seq', ['#ffffff', '#c6d9ea', '#4477AA', '#204060'])


def load(ds):
    rows = list(csv.DictReader(open(os.path.join(HERE, 'data',
                                                 f'selection_by_class_{ds}.csv'))))
    mods = [k[5:] for k in rows[0] if k.startswith('freq_')]
    P = np.array([[float(r[f'freq_{m}']) for m in mods] for r in rows])
    k = np.array([float(r['mean_k']) for r in rows])
    n = np.array([int(r['n']) for r in rows])
    return mods, P, k, n


def heatmap(ds):
    S.apply()
    mods, P, k, n = load(ds)
    C, M = P.shape
    fig, (ax, axk) = plt.subplots(
        1, 2, figsize=(0.42 * M + 1.9, max(1.9, 0.118 * C + 0.85)),
        gridspec_kw=dict(width_ratios=[M, 0.9], wspace=0.05))
    im = ax.imshow(P, aspect='auto', cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(M)); ax.set_xticklabels([PRETTY.get(m, m) for m in mods],
                                                rotation=38, ha='right', fontsize=7)
    ax.set_yticks(range(C)); ax.set_yticklabels(range(C), fontsize=5.6)
    ax.set_ylabel('true class', fontsize=8.5, labelpad=2)
    ax.set_title(f'{TITLE[ds]}   $P(j\\in S\\mid y)$', fontsize=8.5, pad=4)
    for s in ax.spines.values(): s.set_linewidth(1.0); s.set_color('0.25')
    axk.barh(range(C), k, color='#AA3355', height=0.78)
    axk.set_ylim(ax.get_ylim()); axk.set_yticks([])
    axk.set_xlim(0, M); axk.set_xticks([0, M])
    axk.set_xlabel('$|S|$', fontsize=8)
    axk.tick_params(labelsize=7)
    axk.grid(True, axis='x', ls='--', lw=0.6, color=S.GREY); axk.set_axisbelow(True)
    for s in axk.spines.values(): s.set_linewidth(1.0); s.set_color('0.25')
    cb = fig.colorbar(im, ax=axk, fraction=0.30, pad=0.12, ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.8)
    fig.subplots_adjust(left=0.135, right=0.845, top=0.905, bottom=0.235)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{HERE}/selection_by_class_{ds}.{ext}', dpi=300)
    plt.close(fig)
    print(f'  {ds:9s} heatmap  C={C} M={M}  |S| {k.min():.2f}-{k.max():.2f}')


def bars(ds):
    """EAV: C=5, so the IEMOCAP bar-panel layout still works."""
    S.apply()
    mods, P, k, n = load(ds)
    C, M = P.shape
    fig, axes = plt.subplots(1, C, figsize=(7.6, 2.05), sharey=True)
    x = np.arange(M)
    for c, ax in enumerate(axes):
        S.grid(ax, axis='y')
        ax.bar(x, P[c], width=0.72, color=S.CATEGORICAL[:M], edgecolor='white',
               linewidth=0.6, zorder=3)
        ax.set_title(f'class {c}', fontsize=8.2, pad=3)
        ax.set_xticks(x); ax.set_xticklabels([PRETTY.get(m, m) for m in mods],
                                             rotation=38, ha='right')
        ax.tick_params(axis='x', labelsize=6.8)
        ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        for s in ax.spines.values(): s.set_linewidth(1.0); s.set_color('0.25')
    axes[0].set_ylabel(r'$P(j \in S \mid y)$')
    fig.subplots_adjust(left=0.075, right=0.995, top=0.885, bottom=0.235, wspace=0.10)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{HERE}/selection_by_class_{ds}.{ext}', dpi=300)
    plt.close(fig)
    print(f'  {ds:9s} bars     C={C} M={M}  |S| {k.min():.2f}-{k.max():.2f}')


if __name__ == '__main__':
    bars('eav')
    for d in ('mmfi', 'cmi', 'czu_mhad', 'dsads', 'utd_mhad'):
        heatmap(d)
