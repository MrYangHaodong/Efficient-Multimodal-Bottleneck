#!/usr/bin/env python3
"""Publication figures for class-conditional selection, P(j in S | y).

The default is a compact heatmap for every dataset, with a common probability
scale and no mean acquired-set-size strip. Optional bars retain one panel per
class and use the paper's categorical colors. CSV values and class
order are unchanged; the figures show means, not uncertainty estimates.

No figure title or outer frame is added. PDF fonts are embedded, and PNGs are
saved at 600 dpi. Dimensions and font sizes are in inches and points, so they
can be set for the final manuscript placement without rescaling the fonts.

  python plotting/plot_selection_by_class.py
  python plotting/plot_selection_by_class.py --datasets iemocap --annotate
  python plotting/plot_selection_by_class.py --form bars --datasets iemocap eav
  python plotting/plot_selection_by_class.py --width-inches 5.5 --font-size 11

Outputs stay next to this script unless --output-dir is supplied. Heatmaps
keep selection_by_class_<dataset>; bar figures keep the _bars suffix.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter

HERE = Path(__file__).resolve().parent
ALL = ['iemocap', 'mmfi', 'cmi', 'czu_mhad', 'dsads', 'eav', 'utd_mhad']
PRETTY = {'mocap_hand': 'm-hand', 'mocap_head': 'm-head', 'mocap_rotated': 'm-rot',
          'tof_1': 'ToF$_1$', 'tof_2': 'ToF$_2$', 'tof_3': 'ToF$_3$', 'tof_4': 'ToF$_4$',
          'tof_5': 'ToF$_5$', 'thm': 'THM', 'imu': 'IMU', 'eeg': 'EEG', 'rgb': 'RGB',
          'wifi': 'Wi-Fi', 'mmwave': 'mmWave',
          'right_arm': 'r-arm', 'left_arm': 'l-arm',
          'right_leg': 'r-leg', 'left_leg': 'l-leg',
          'sensor_head': 'head', 'sensor_torso': 'torso',
          'sensor_rarm': 'r-arm', 'sensor_larm': 'l-arm', 'sensor_leg': 'leg'}
PURPLE = '#AA4499'
PALETTE = [PURPLE, '#EE7733', '#4477AA', '#882255', '#228833', '#CC6677', '#BBBBBB']
# Keep a modality's bar color stable across datasets, even when CSV columns differ.
MODALITY_COLORS = {
    'video': PALETTE[0], 'audio': PALETTE[1], 'text': PALETTE[2],
    'mocap_hand': PALETTE[3], 'mocap_head': PALETTE[4], 'mocap_rotated': PALETTE[5],
    'eeg': PALETTE[2], 'infra1': PALETTE[0], 'wifi': PALETTE[1],
    'mmwave': PALETTE[2], 'lidar': PALETTE[3], 'depth': PALETTE[4],
    'imu': PALETTE[0], 'thm': PALETTE[1],
    **{f'tof_{i}': PALETTE[i + 1] for i in range(1, 6)},
    'skeleton': PALETTE[0], 'sensor_head': PALETTE[1], 'sensor_torso': PALETTE[2],
    'sensor_rarm': PALETTE[3], 'sensor_larm': PALETTE[5], 'sensor_leg': PALETTE[6],
    'torso': PALETTE[0], 'right_arm': PALETTE[1], 'left_arm': PALETTE[2],
    'right_leg': PALETTE[3], 'left_leg': PALETTE[4],
    'inertial': PALETTE[1], 'rgb': PALETTE[2],
}
CMAP = LinearSegmentedColormap.from_list(
    'semarc_probability', ['#FFFFFF', '#EAD6E7', PURPLE, '#602658'])


def apply_style(font_size=10.0):
    """Local styling: other scripts using iclr_style are deliberately unaffected."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': font_size,
        'axes.labelsize': font_size + 1, 'axes.titlesize': font_size,
        'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
        'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.pad': 2, 'ytick.major.pad': 2,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'savefig.bbox': None,
    })


def frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)


def load(ds):
    """-> modality names, P[class, modality], mean |S| per class, class labels."""
    if ds not in ALL:
        raise ValueError(f'Unknown dataset: {ds}')
    with (HERE / 'data' / f'selection_by_class_{ds}.csv').open(newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f'{ds}: no class-selection measurements')
    mods = [k[5:] for k in rows[0] if k.startswith('freq_')]
    P = np.array([[float(r[f'freq_{m}']) for m in mods] for r in rows])
    k = np.array([float(r['mean_k']) for r in rows])
    names = ([r['class_name'] for r in rows] if 'class_name' in rows[0]
             else [f'class {i}' for i in range(len(rows))])
    if not mods or not np.isfinite(P).all() or np.any((P < 0) | (P > 1)):
        raise ValueError(f'{ds}: selection probabilities must be finite and in [0, 1]')
    if not np.isfinite(k).all() or np.any((k < 0) | (k > len(mods))):
        raise ValueError(f'{ds}: mean acquired-set size must be in [0, {len(mods)}]')
    return mods, P, k, names


def _out(ds, form, auto, output_dir=None):
    directory = HERE if output_dir is None else Path(output_dir)
    return directory / (f'selection_by_class_{ds}' + ('' if form == auto else f'_{form}'))


def save(fig, out, dpi=600):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)


def bars(ds, auto='heatmap', *, output_dir=None, font_size=10.0,
         width=None, height=None, dpi=600, annotate=False):
    apply_style(font_size)
    mods, P, k, names = load(ds)
    C, M = P.shape
    # Wrap large label spaces; the default heatmap remains the compact form.
    columns = C if C <= 5 else 4
    rows = int(np.ceil(C / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                            figsize=(width or 5.5, height or 2.35 * rows),
                            sharey=True, layout='constrained')
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, wspace=0.025, hspace=0.07)
    x = np.arange(M)
    for c, ax in enumerate(axes.flat):
        if c >= C:
            ax.set_visible(False)
            continue
        frame(ax)
        ax.grid(True, axis='y', color='#DEDEDE', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.bar(x, P[c], width=0.74,
               color=[MODALITY_COLORS.get(m, PURPLE) for m in mods],
               edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_title(names[c], pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY.get(m, m) for m in mods], rotation=60,
                           ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', labelsize=font_size - 1)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    fig.supylabel(r'$P(j \in S \mid y)$', fontsize=font_size + 1)
    out = _out(ds, 'bars', auto, output_dir)
    save(fig, out, dpi)
    print(f'  {ds:9s} bars     C={C} M={M}  mean |S| {k.min():.2f}-{k.max():.2f}  -> {out}.pdf')


def heatmap(ds, auto='heatmap', *, output_dir=None, font_size=10.0,
            width=None, height=None, dpi=600, annotate=False):
    apply_style(font_size)
    mods, P, k, names = load(ds)
    C, M = P.shape
    # Keep dense class labels readable at their native print size.
    row_height = max(0.17, 1.4 * font_size / 72)
    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(width or max(3.6, 0.40 * M + 1.4),
                       height or max(2.0, row_height * C + 0.82)),
        gridspec_kw=dict(width_ratios=[M, 0.16]), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, wspace=0.035, hspace=0.02)
    im = ax.imshow(P, aspect='auto', cmap=CMAP, vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(M))
    ax.set_xticklabels([PRETTY.get(m, m) for m in mods], rotation=38,
                       ha='right', rotation_mode='anchor')
    ax.set_yticks(range(C))
    ax.set_yticklabels([name.removeprefix('class ') for name in names])
    ax.set_ylabel('True class', labelpad=3)
    ax.tick_params(length=0, pad=3)
    frame(ax)
    # Thin white separators aid row tracking without changing any cell values.
    ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(C + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.45)
    ax.tick_params(which='minor', length=0)
    if annotate:
        for c, m in np.ndindex(P.shape):
            ax.text(m, c, f'{P[c, m]:.2f}', ha='center', va='center',
                    fontsize=font_size - 1,
                    color='white' if P[c, m] >= 0.6 else 'black')
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.set_label(r'$P(j \in S \mid y)$', fontsize=font_size + 1, labelpad=5)
    cb.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    cb.ax.tick_params(labelsize=font_size, length=2.5)
    cb.outline.set_linewidth(0.8)
    cb.outline.set_edgecolor('black')
    out = _out(ds, 'heatmap', auto, output_dir)
    save(fig, out, dpi)
    print(f'  {ds:9s} heatmap  C={C} M={M}  mean |S| {k.min():.2f}-{k.max():.2f}  -> {out}.pdf')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', choices=ALL, default=ALL)
    ap.add_argument('--form', choices=['auto', 'bars', 'heatmap'], default='auto')
    ap.add_argument('--output-dir', type=Path, default=HERE)
    ap.add_argument('--font-size', type=float, default=10.0, help='Base font size in points.')
    ap.add_argument('--width-inches', type=float, default=None)
    ap.add_argument('--height-inches', type=float, default=None)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--annotate', action='store_true', help='Print two-decimal probabilities in heatmap cells.')
    a = ap.parse_args()
    for name in ('font_size', 'width_inches', 'height_inches', 'dpi'):
        value = getattr(a, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            ap.error(f'--{name.replace("_", "-")} must be finite and positive')
    if a.font_size <= 1:
        ap.error('--font-size must exceed 1 point')
    for ds in a.datasets:
        auto = 'heatmap'      # the form the paper uses for every dataset
        form = auto if a.form == 'auto' else a.form
        (bars if form == 'bars' else heatmap)(
            ds, auto, output_dir=a.output_dir, font_size=a.font_size,
            width=a.width_inches, height=a.height_inches, dpi=a.dpi, annotate=a.annotate)


if __name__ == '__main__':
    main()
