#!/usr/bin/env python3
"""Publication-ready IEMOCAP availability cascade.

Replaces the 2x3 diagnostic dump with two panels, omitting first-pick annotations.
Top: macro-F1 as the available set shrinks, for SeMARC and for the same frozen backbone
consuming everything available. Bottom: what that costs, as modality usage.

With no arguments, write firstpick_cascade_iemocap.pdf and .png beside this script.
Both exports default to exactly 6.0 x 3.20 inches (3600 x 1920 at 600 dpi).
The IEMOCAP selection and modality-weighting figures remain 4.5 x 3.2 inches,
so all three have equal displayed heights in a 30% / 40% / 30% manuscript row.
Use --tight for a content-cropped standalone export with a one-pixel guard.
Panels (a) and (b)
show macro-F1 and modality usage, respectively. Both panels share recursive
availability labels: A_0 is the full available set, and A_i is obtained by
removing the named modality from A_(i-1). The first tick is All (A_0), followed
by A_0 - text, A_1 - audio, A_2 - video, and A_3 - m-head.
Each tick puts the remaining modality count inline on its second line, (6) to (2).
Axis titles and panel labels use 12 pt, tick and legend labels 11 pt, and
annotations 10 pt at the default base font size. Counts and the matching
"(modality count)" axis-title suffix use smaller 9 pt text.
Modality-usage annotations show mean acquired / available modality counts;
the mean count is rounded to one decimal, while the plotted fractions are unchanged.
The source CSV and its recorded mean/std values are never modified. The upper
panel ends at 0.8 and extends below its 0.2 tick to keep the full error bars visible.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
OUT = HERE / 'firstpick_cascade_iemocap'
PURPLE = '#AA4499'
ORANGE = '#EE7733'
DEFAULT_WIDTH = 6.0
DEFAULT_HEIGHT = 3.20
DEFAULT_FONT_SIZE = 14.0
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
    parser.add_argument('--font-size', type=float, default=DEFAULT_FONT_SIZE,
                        help='Base print font size in points (default: 14).')
    parser.add_argument('--width-inches', type=float, default=DEFAULT_WIDTH)
    parser.add_argument('--height-inches', type=float, default=DEFAULT_HEIGHT)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--tight', action='store_true',
                        help='Crop to content instead of preserving the shared row dimensions.')
    args = parser.parse_args()
    for name in ('font_size', 'width_inches', 'height_inches', 'dpi'):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive and finite.')
    if args.font_size < 8:
        parser.error('--font-size must be at least 8 pt to keep secondary labels readable.')
    return args


def availability_labels(rows):
    """Show each recursive available set and its remaining modality count."""
    labels = []
    for step, row in enumerate(rows):
        count = int(row['n_avail'])
        if step == 0:
            labels.append('All\n' + r'$A_0$' + f' ({count})')
        else:
            modality = row['removed_to_reach']
            name = SHORT.get(modality, modality).replace('\u2011', '-')
            labels.append(rf'$A_{{{step - 1}}} -$' + f'\n{name} ({count})')
    return labels


def add_availability_labels(ax, rows, tick_size, label_size, count_size):
    """Pack real point-sized count text inline, without shrinking the modality names."""
    def text(value, size):
        return TextArea(value, textprops={'fontsize': size, 'color': 'black'})

    labels = availability_labels(rows)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelbottom=False)
    for position, label in enumerate(labels):
        heading, detail = label.split('\n')
        modality, count = detail.rsplit(' ', 1)
        second_line = HPacker(
            children=[text(modality, tick_size), text(count, count_size)],
            align='baseline', pad=0, sep=1)
        block = VPacker(children=[text(heading, tick_size), second_line],
                        align='center', pad=0, sep=1)
        artist = AnnotationBbox(
            block, (position, 0), xycoords=ax.get_xaxis_transform(),
            xybox=(0, -4), boxcoords='offset points', box_alignment=(0.5, 1),
            frameon=False, pad=0, annotation_clip=False)
        artist.set_gid(f'availability-tick-{position}')
        ax.add_artist(artist)

    # Keep the standard axis label as metadata; draw its two font sizes together.
    ax.set_xlabel(r'Available set, $A_n$ (modality count)', labelpad=2)
    ax.xaxis.label.set_visible(False)
    title = HPacker(
        children=[text(r'Available set, $A_n$', label_size),
                  text('(modality count)', count_size)],
        align='baseline', pad=0, sep=3)
    artist = AnnotationBbox(
        title, (0.5, 0), xycoords=ax.transAxes,
        xybox=(0, -31), boxcoords='offset points', box_alignment=(0.5, 1),
        frameon=False, pad=0, annotation_clip=False)
    artist.set_gid('availability-axis-label')
    ax.add_artist(artist)


def build_figure(font_size=DEFAULT_FONT_SIZE, width_inches=DEFAULT_WIDTH, height_inches=DEFAULT_HEIGHT):
    """Draw both panels without altering or rounding the source observations."""
    label_size = max(7.0, font_size - 2)
    tick_size = max(6.5, font_size - 3)
    annotation_size = max(6.0, font_size - 4)
    count_size = max(6.0, font_size - 5)
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
        (rf1, rsd, ORANGE, 's', '--', r'SeMA (use all $A_n$)'),
    ):
        container = axL.errorbar(
            x, values, yerr=std, color=color, ecolor=color, marker=marker,
            ls=style, ms=5.0 if marker == 'o' else 4.4, elinewidth=0.85,
            capsize=2.0, capthick=0.85, zorder=3 if marker == 'o' else 2,
            label=label, **line_options,
        )
        for artist in (*container.lines[1], *container.lines[2]):
            artist.set_alpha(0.60)

    axL.set_ylabel('Macro-F1', labelpad=2)
    lower_error = float(np.min(np.minimum(f1 - sd, rf1 - rsd)))
    axL.set_ylim(min(0.2, lower_error - 0.025), 0.8)
    axL.set_yticks([0.2, 0.5, 0.8])

    reference_mu = np.array([float(row['ref_mu_mean']) for row in rows])
    axR.plot(x, mu, color=PURPLE, marker='o', ms=5.0, zorder=3, **line_options)
    axR.plot(x, reference_mu, color=ORANGE, ls='--', lw=1.35, zorder=2)
    for xi, yi, row, available in zip(x, mu, rows, nA):
        acquired = float(row['mean_k'])
        axR.annotate(f'{acquired:.1f}/{available}', (xi, yi), xytext=(0, -5),
                     textcoords='offset points', ha='center', va='top',
                     fontsize=annotation_size, color='black')
    axR.set_ylabel('Modality usage', labelpad=2)
    axR.set_ylim(0.0, max(1.10, float(max(mu.max(), reference_mu.max())) + 0.10))
    axR.set_yticks([0.0, 0.5, 1.0])

    for panel_label, ax in zip(('(a)', '(b)'), (axL, axR)):
        ax.text(0.90, 0.96, panel_label, transform=ax.transAxes,
                ha='left', va='top', fontsize=label_size, color='black',
                bbox=dict(facecolor='white', edgecolor='none', pad=0.1))
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', color='#E6E6E6', linewidth=0.4, linestyle='-')
        ax.set_xticks(x)
        ax.set_xlim(-0.25, len(x) - 0.48)
        ax.tick_params(direction='out', length=2.2, width=0.55, pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.65)
    add_availability_labels(axR, rows, tick_size, label_size, count_size)
    axL.tick_params(axis='x', labelbottom=False, length=0)
    fig.align_ylabels((axL, axR))

    handles = [
        Line2D([], [], color=PURPLE, marker='o', ms=4.2,
               label='SeMARC', **line_options),
        Line2D([], [], color=ORANGE, ls='--', marker='s', ms=3.8,
               label=r'SeMA (use all $A_n$)', **line_options),
    ]
    left_margin = 0.50 / width_inches
    right_margin = 1 - 0.015 / width_inches
    legend = fig.legend(
        handles=handles, loc='upper center',
        bbox_to_anchor=(0.5, 1 - 0.02 / height_inches),
        ncol=2, fontsize=max(6.0, font_size - 3), frameon=True, fancybox=False,
        framealpha=1.0, facecolor='#FAFAFA', edgecolor='#B8B8B8',
        columnspacing=0.8, handlelength=1.3, handletextpad=0.4, borderpad=0.25,
        borderaxespad=0,
    )
    legend.get_frame().set_linewidth(0.55)
    # Keep physical margins tight while the added height goes to the data panels.
    fig.subplots_adjust(left=left_margin, right=right_margin,
                        top=1 - 0.33 / height_inches,
                        bottom=0.62 / height_inches, hspace=0.10)
    return fig


def validate_figure(fig):
    """Reject clipped uncertainty, clipped labels, and overlapping labels."""
    lower, upper = fig.axes[0].get_ylim()
    for collection in fig.axes[0].collections:
        for segment in collection.get_segments():
            if np.any(segment[:, 1] < lower) or np.any(segment[:, 1] > upper):
                raise ValueError('Clipped Macro-F1 error bar; expand the y limits.')
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
    packed_ticks = [artist for artist in fig.findobj(AnnotationBbox)
                    if (artist.get_gid() or '').startswith('availability-tick-')]
    boxes = [artist.get_window_extent(renderer) for artist in packed_ticks]
    if any(left.overlaps(right) for left, right in zip(boxes, boxes[1:])):
        raise ValueError('Overlapping packed availability labels; increase --width-inches.')
    labels = [text for text in fig.findobj(Text)
              if text.get_visible() and text.get_text()]
    for index, left in enumerate(labels):
        left_box = left.get_window_extent(renderer)
        for right in labels[index + 1:]:
            if left_box.overlaps(right.get_window_extent(renderer)):
                raise ValueError(f'Overlapping labels {left.get_text()!r} and '
                                 f'{right.get_text()!r}; increase figure dimensions.')


def tight_export_bbox(fig):
    """Crop both formats to rendered ink, including full strokes and packed labels.

    Text layout boxes include unused font-descent space, so use the already
    validated Agg rendering rather than a generic tight bounding box. Keep one
    export pixel outside the ink to protect antialiased strokes from clipping.
    The PDF stays vector; only its page boundary uses these measured bounds.
    """
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    ink = np.any(rgba[:, :, :3] < 255, axis=2) & (rgba[:, :, 3] > 0)
    ys, xs = np.nonzero(ink)
    if not len(xs):
        raise ValueError('Cannot crop an empty figure.')
    height = rgba.shape[0]
    return Bbox.from_extents(
        (xs.min() - 1) / fig.dpi, (height - ys.max() - 2) / fig.dpi,
        (xs.max() + 2) / fig.dpi, (height - ys.min() + 1) / fig.dpi)


def main():
    args = parse_args()
    fig = build_figure(args.font_size, args.width_inches, args.height_inches)
    try:
        for dpi in (100, args.dpi):
            fig.set_dpi(dpi)
            validate_figure(fig)
        export_bbox = tight_export_bbox(fig) if args.tight else None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for ext in ('pdf', 'png'):
            path = args.output_dir / f'{OUT.name}.{ext}'
            fig.savefig(path, dpi=args.dpi, bbox_inches=export_bbox, pad_inches=0)
            print('wrote', path.resolve())
    finally:
        plt.close(fig)
    rows, nA, _, _, _, _, _, first, removed = load()
    print('  |A| :', nA, '\n  first:', first, '\n  removed:', removed)

if __name__ == '__main__':
    main()
