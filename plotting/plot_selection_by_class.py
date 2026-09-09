#!/usr/bin/env python
"""Class-conditional modality selection, P(j in S | y), for every dataset (see iclr_style).

One implementation per FORM, not per dataset:

  bars(ds)     one panel per class. Readable while the label space is small, so it is the
               form used for IEMOCAP (C=4, the main-text figure) and EAV (C=5).
  heatmap(ds)  the same quantity as a class x modality image, with a side strip for the
               per-class mean acquired-set size. The only form that survives C=18..27.

`--form auto` (the default) picks bars for C<=5 and heatmap above it, which reproduces every
figure in the paper. Asking for a form a dataset would not choose writes a suffixed file
(e.g. selection_by_class_iemocap_heatmap.pdf) so the published figures are never overwritten.

  python plot_selection_by_class.py                          # all 7, paper forms
  python plot_selection_by_class.py --form heatmap --datasets iemocap eav
"""
import argparse, csv, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import iclr_style as S

HERE = os.path.dirname(os.path.abspath(__file__))
ALL = ['iemocap', 'mmfi', 'cmi', 'czu_mhad', 'dsads', 'eav', 'utd_mhad']
PRETTY = {'mocap_hand': 'm-hand', 'mocap_head': 'm-head', 'mocap_rotated': 'm-rot',
          'tof_1': 'ToF$_1$', 'tof_2': 'ToF$_2$', 'tof_3': 'ToF$_3$', 'tof_4': 'ToF$_4$',
          'tof_5': 'ToF$_5$', 'thm': 'THM', 'imu': 'IMU'}
TITLE = {'iemocap': 'IEMOCAP', 'mmfi': 'MM-Fi', 'cmi': 'CMI', 'czu_mhad': 'CZU-MHAD',
         'dsads': 'DSADS', 'eav': 'EAV', 'utd_mhad': 'UTD-MHAD'}
# IEMOCAP's panel row carries class names, so its y-label needs marginally more room. Kept
# per-dataset so both published bar figures reproduce byte-for-byte.
BARS_LEFT = {'iemocap': 0.072}
CMAP = LinearSegmentedColormap.from_list('tolblue_seq', ['#ffffff', '#c6d9ea', '#4477AA', '#204060'])


def load(ds):
    """-> modality names, P[class, modality], mean |S| per class, class labels."""
    rows = list(csv.DictReader(open(os.path.join(HERE, 'data', f'selection_by_class_{ds}.csv'))))
    mods = [k[5:] for k in rows[0] if k.startswith('freq_')]
    P = np.array([[float(r[f'freq_{m}']) for m in mods] for r in rows])
    k = np.array([float(r['mean_k']) for r in rows])
    names = ([r['class_name'] for r in rows] if 'class_name' in rows[0]
             else [f'class {i}' for i in range(len(rows))])
    return mods, P, k, names


def _out(ds, form, auto):
    return os.path.join(HERE, f'selection_by_class_{ds}' + ('' if form == auto else f'_{form}'))


def bars(ds, auto):
    S.apply()
    mods, P, k, names = load(ds)
    C, M = P.shape
    fig, axes = plt.subplots(1, C, figsize=(7.6, 2.05), sharey=True)
    x = np.arange(M)
    for c, ax in enumerate(np.atleast_1d(axes)):
        S.grid(ax, axis='y')
        ax.bar(x, P[c], width=0.72, color=S.CATEGORICAL[:M], edgecolor='white',
               linewidth=0.6, zorder=3)
        ax.set_title(names[c], fontsize=8.2, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY.get(m, m) for m in mods], rotation=38, ha='right')
        ax.tick_params(axis='x', labelsize=6.8)
        ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    np.atleast_1d(axes)[0].set_ylabel(r'$P(j \in S \mid y)$')
    fig.subplots_adjust(left=BARS_LEFT.get(ds, 0.075), right=0.995,
                        top=0.885, bottom=0.235, wspace=0.10)
    out = _out(ds, 'bars', auto)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300)
    plt.close(fig)
    print(f'  {ds:9s} bars     C={C} M={M}  |S| {k.min():.2f}-{k.max():.2f}  -> {os.path.basename(out)}.pdf')


def heatmap(ds, auto):
    S.apply()
    mods, P, k, names = load(ds)
    C, M = P.shape
    fig, (ax, axk) = plt.subplots(
        1, 2, figsize=(0.42 * M + 1.9, max(1.9, 0.118 * C + 0.85)),
        gridspec_kw=dict(width_ratios=[M, 0.9], wspace=0.05))
    im = ax.imshow(P, aspect='auto', cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(M))
    ax.set_xticklabels([PRETTY.get(m, m) for m in mods], rotation=38, ha='right', fontsize=7)
    ax.set_yticks(range(C))
    # named classes are worth spelling out; a 27-way numeric index is not
    ax.set_yticklabels(names if 'class_name' in open(
        os.path.join(HERE, 'data', f'selection_by_class_{ds}.csv')).readline() else range(C),
        fontsize=5.6 if C > 8 else 7)
    ax.set_ylabel('true class', fontsize=8.5, labelpad=2)
    ax.set_title(f'{TITLE[ds]}   $P(j\\in S\\mid y)$', fontsize=8.5, pad=4)
    for s in ax.spines.values(): s.set_linewidth(1.0); s.set_color('0.25')
    axk.barh(range(C), k, color='#AA3355', height=0.78)
    axk.set_ylim(ax.get_ylim()); axk.set_yticks([])
    axk.set_xlim(0, M); axk.set_xticks([0, M])
    axk.set_xlabel('$|S|$', fontsize=8); axk.tick_params(labelsize=7)
    axk.grid(True, axis='x', ls='--', lw=0.6, color=S.GREY); axk.set_axisbelow(True)
    for s in axk.spines.values(): s.set_linewidth(1.0); s.set_color('0.25')
    cb = fig.colorbar(im, ax=axk, fraction=0.30, pad=0.12, ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.8)
    fig.subplots_adjust(left=0.135, right=0.845, top=0.905, bottom=0.235)
    out = _out(ds, 'heatmap', auto)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300)
    plt.close(fig)
    print(f'  {ds:9s} heatmap  C={C} M={M}  |S| {k.min():.2f}-{k.max():.2f}  -> {os.path.basename(out)}.pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=ALL)
    ap.add_argument('--form', choices=['auto', 'bars', 'heatmap'], default='auto')
    a = ap.parse_args()
    for ds in a.datasets:
        C = len(load(ds)[1])
        auto = 'heatmap'      # the form the paper uses for every dataset
        form = auto if a.form == 'auto' else a.form
        (bars if form == 'bars' else heatmap)(ds, auto)


if __name__ == '__main__':
    main()
