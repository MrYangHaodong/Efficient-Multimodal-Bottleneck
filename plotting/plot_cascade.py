#!/usr/bin/env python3
"""Publication-ready IEMOCAP availability cascade.

Replaces the 2x3 diagnostic dump with two panels. The five first-pick bar panels collapse
into annotations on the accuracy panel -- the first pick is deterministic within a fold under
a fixed availability mask, so a whole panel per config was spending a lot of ink on one label.
Top: macro-F1 as the available set shrinks, for SeMARC and for the same frozen backbone
consuming everything available. Bottom: what that costs, as modality usage.

With no arguments, write firstpick_cascade_iemocap.pdf and .png beside this script.
The 4.0 x 2.60 inch canvas is 2400 x 1560 pixels at 600 dpi. Panels (a) and (b)
show macro-F1 and modality usage, respectively. Both panels share cumulative
availability labels; w/o means without all listed modalities, and parentheses
give the remaining modality count.
Axis titles use 8 pt and tick labels 7 pt at the default base font size.
Modality-usage annotations show mean acquired / available modality counts;
the mean count is rounded to one decimal, while the plotted fractions are unchanged.
The source CSV and its recorded mean/std values are never modified.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
OUT = HERE / 'firstpick_cascade_iemocap'
PURPLE = '#AA4499'
ORANGE = '#EE7733'
DEFAULT_WIDTH = 4.0
DEFAULT_HEIGHT = 2.60
SHORT = {'text': 'text', 'audio': 'audio', 'video': 'video',
         'mocap_hand': 'm‑hand', 'mocap_head': 'm‑head', 'mocap_rotated': 'm‑rot'}


def load():
    """data/firstpick_cascade_iemocap.csv -- the RL walk and the consume-all reference."""
    with (HERE / 'data' / 'firstpick_cascade_iemocap.csv').open(
        encoding='utf-8', newline=''
    ) as source:
        rows = list(csv.DictReader(source))
    f = lambda k: np.array([float(r[k]) for r in rows])
    return (rows, [int(r['n_avail']) for r in rows], f('f1_mean'), f('f1_std'),
            f('ref_f1_mean'), f('ref_f1_std'), f('mu_mean'),
            [r['first_pick'] for r in rows],
            ['all' if not r['removed_to_reach'] else '\u2212' + r['removed_to_reach']
             for r in rows])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--font-size', type=float, default=10.0,
                        help='Base print font size in points (default: 10).')
    parser.add_argument('--width-inches', type=float, default=DEFAULT_WIDTH)
    parser.add_argument('--height-inches', type=float, default=DEFAULT_HEIGHT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    for name in ('font_size', 'width_inches', 'height_inches', 'dpi'):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive and finite.')
    if args.font_size < 8:
        parser.error('--font-size must be at least 8 pt to keep secondary labels readable.')
    return args


def availability_labels(rows):
    """List every accumulated removal, not only the last transition's removal."""
    removed = []
    labels = []
    for row in rows:
        modality = row['removed_to_reach']
        if modality:
            removed.append(SHORT.get(modality, modality))
        count = int(row['n_avail'])
        if not removed:
            labels.append(f'All ({count})')
        elif len(removed) == 1:
            labels.append(f'w/o {removed[0]}\n({count})')
        else:
            lines = [f'w/o {removed[0]},']
            remainder = f"{', '.join(removed[1:])} ({count})"
            lines.extend(textwrap.wrap(remainder, width=14,
                                       break_long_words=False, break_on_hyphens=False))
            # Balance a trailing count onto a modality line, keeping adjacent
            # cumulative labels visually separated at the narrower width.
            if lines[-1] == f'({count})' and ' ' in lines[-2]:
                prefix, last_modality = lines[-2].rsplit(' ', 1)
                lines[-2] = prefix
                lines[-1] = f'{last_modality} ({count})'
            labels.append('\n'.join(lines))
    return labels


def build_figure(font_size=10.0, width_inches=DEFAULT_WIDTH, height_inches=DEFAULT_HEIGHT):
    """Draw both panels without altering or rounding the source observations."""
    label_size = max(7.0, font_size - 2)
    tick_size = max(6.5, font_size - 3)
    annotation_size = max(6.0, font_size - 4)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': font_size,
        'mathtext.fontset': 'dejavusans',
        'axes.labelsize': label_size, 'axes.labelcolor': 'black',
        'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size,
        'axes.linewidth': 0.65, 'axes.edgecolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'text.color': 'black', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    })
    rows, nA, f1, sd, rf1, rsd, mu, first, removed = load()
    x = np.arange(len(rows))

    fig, (axL, axR) = plt.subplots(2, 1, figsize=(width_inches, height_inches), sharex=True)
    line_options = dict(lw=1.35, markeredgecolor='white', markeredgewidth=0.6)
    for values, std, color, marker, style, label in (
        (f1, sd, PURPLE, 'o', '-', 'SeMARC'),
        (rf1, rsd, ORANGE, 's', '--', 'SeMA (all modalities)'),
    ):
        container = axL.errorbar(
            x, values, yerr=std, color=color, ecolor=color, marker=marker,
            ls=style, ms=5.0 if marker == 'o' else 4.4, elinewidth=0.85,
            capsize=2.0, capthick=0.85, zorder=3 if marker == 'o' else 2,
            label=label, **line_options,
        )
        for artist in (*container.lines[1], *container.lines[2]):
            artist.set_alpha(0.60)

    # Place first-pick labels beyond both error bars, not over the uncertainty.
    annotation_y = np.maximum(f1 + sd, rf1 + rsd) + 0.026
    for xi, yi, m in zip(x, annotation_y, first):
        axL.annotate(SHORT.get(m, m), (xi, yi), ha='center', va='bottom',
                     fontsize=annotation_size, color='black')
    axL.set_ylabel('Macro-F1', labelpad=2)
    lower = min(0.10, float(np.min(np.minimum(f1 - sd, rf1 - rsd))) - 0.025)
    axL.set_ylim(lower, max(1.0, float(annotation_y.max()) + 0.23))
    axL.set_yticks([0.2, 0.5, 0.8])

    reference_mu = np.array([float(row['ref_mu_mean']) for row in rows])
    axR.plot(x, mu, color=PURPLE, marker='o', ms=5.0, zorder=3, **line_options)
    axR.plot(x, reference_mu, color=ORANGE, ls='--', lw=1.35, zorder=2)
    for xi, yi, row, available in zip(x, mu, rows, nA):
        acquired = float(row['mean_k'])
        axR.annotate(f'{acquired:.1f}/{available}', (xi, yi), xytext=(0, -5),
                     textcoords='offset points', ha='center', va='top',
                     fontsize=annotation_size, color='black')
    axR.set_ylabel('Modality\nusage', labelpad=2)
    axR.set_ylim(0.0, max(1.10, float(max(mu.max(), reference_mu.max())) + 0.10))
    axR.set_yticks([0.0, 0.5, 1.0])

    for panel_label, ax in zip(('(a)', '(b)'), (axL, axR)):
        ax.text(0.02, 0.96, panel_label, transform=ax.transAxes,
                ha='left', va='top', fontsize=label_size, color='black',
                bbox=dict(facecolor='white', edgecolor='none', pad=0.1))
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', color='#E6E6E6', linewidth=0.4, linestyle='-')
        ax.set_xticks(x)
        ax.set_xlim(-0.48, len(x) - 0.52)
        ax.tick_params(direction='out', length=2.2, width=0.55, pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.65)
    axR.set_xticklabels(availability_labels(rows))
    for label in axR.get_xticklabels():
        label.set_linespacing(1.0)
    axR.set_xlabel('Availability (remaining count)', labelpad=2)
    axL.tick_params(axis='x', labelbottom=False, length=0)
    fig.align_ylabels((axL, axR))

    handles = [
        Line2D([], [], color=PURPLE, marker='o', ms=4.2,
               label='SeMARC', **line_options),
        Line2D([], [], color=ORANGE, ls='--', marker='s', ms=3.8,
               label='SeMA (all modalities)', **line_options),
    ]
    left_margin = 0.51 / width_inches
    right_margin = 1 - 0.035 / width_inches
    legend = fig.legend(
        handles=handles, loc='upper center',
        bbox_to_anchor=((left_margin + right_margin) / 2, 1 - 0.02 / height_inches),
        ncol=2, fontsize=max(6.0, font_size - 3), frameon=True, fancybox=False,
        framealpha=1.0, facecolor='#FAFAFA', edgecolor='#B8B8B8',
        columnspacing=0.8, handlelength=1.3, handletextpad=0.4, borderpad=0.25,
        borderaxespad=0,
    )
    legend.get_frame().set_linewidth(0.55)
    # Keep physical margins tight while the added height goes to the data panels.
    fig.subplots_adjust(left=left_margin, right=right_margin,
                        top=1 - 0.21 / height_inches,
                        bottom=0.53 / height_inches, hspace=0.10)
    return fig


def validate_figure(fig):
    """Reject clipped labels and overlapping tick labels before exporting."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if (box.x0 < canvas.x0 - 0.5 or box.y0 < canvas.y0 - 0.5
                or box.x1 > canvas.x1 + 0.5 or box.y1 > canvas.y1 + 0.5):
            raise ValueError(f'Clipped label {text.get_text()!r}; increase figure dimensions.')
    for ax in fig.axes:
        labels = [label for label in ax.get_xticklabels() if label.get_visible()]
        boxes = [label.get_window_extent(renderer) for label in labels]
        if any(left.overlaps(right) for left, right in zip(boxes, boxes[1:])):
            raise ValueError('Overlapping x tick labels; increase --width-inches.')
    labels = [text for text in fig.findobj(Text)
              if text.get_visible() and text.get_text()]
    for index, left in enumerate(labels):
        left_box = left.get_window_extent(renderer)
        for right in labels[index + 1:]:
            if left_box.overlaps(right.get_window_extent(renderer)):
                raise ValueError(f'Overlapping labels {left.get_text()!r} and '
                                 f'{right.get_text()!r}; increase figure dimensions.')


def main():
    args = parse_args()
    fig = build_figure(args.font_size, args.width_inches, args.height_inches)
    try:
        validate_figure(fig)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for ext in ('pdf', 'png'):
            path = args.output_dir / f'{OUT.name}.{ext}'
            fig.savefig(path, dpi=args.dpi, bbox_inches=None, pad_inches=0)
            print('wrote', path.resolve())
    finally:
        plt.close(fig)
    rows, nA, _, _, _, _, _, first, removed = load()
    print('  |A| :', nA, '\n  first:', first, '\n  removed:', removed)

if __name__ == '__main__':
    main()
