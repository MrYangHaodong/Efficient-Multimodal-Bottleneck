#!/usr/bin/env python3
"""Publication-style modality-weighting sensitivity at fixed lambda=0.5.

Source: the user's supplied penalty sweep, weight sweep, modality-penalty table,
and deployed-controller reference. All values below are preserved as supplied;
this script does not download data, train a policy, or infer missing uncertainty.

Default: one compact panel with purple Macro-F1 bars (+/-1 SD) and orange
latency bars. It contains all seven gamma-sweep rows supplied on 2026-09-21.
No title, black boxed axes, a light one-row legend, and zero-based bar axes.
The default 4.5 x 3.2 inch export matches class selection (2700 x 1920 pixels
at 600 dpi). With the 6.0 x 3.2 inch cascade between them, all three have equal
displayed heights in a 30% / 40% / 30% manuscript row.
The separate deployed-controller references are omitted from this sweep view.
Use --secondary mu, gflops, or none for another view of the same measurements.
Use --sweep both to reproduce the older two-panel figure for appendix work;
that legacy penalty panel contains only the earlier uniform-cost measurements.

Important: lambda=0.5, gamma=0 has different results in the two studies. The
weight-sweep uniform control is NOT replaced with the deployed reference.
SD is descriptive standard deviation, not a confidence interval. Latency SD
and gamma-sweep sample counts were not supplied. The old normalized penalty
allocation table is retained for provenance only: the newer raw modality
weights do NOT sum to one, and no normalization formula is inferred here.
The gamma=0.75 sweep point is labeled deployed in the Gamma CSV, but is not
merged with the separate Lambda CSV deployed record (fold-specific lambdas).

Examples:
    python plotting/reward-sensitivity.py
    python plotting/reward-sensitivity.py --secondary mu --stem weight_sensitivity_mu
    python plotting/reward-sensitivity.py --sweep both

Dependencies: matplotlib, numpy. Outputs: modality_weight_sensitivity_iemocap.pdf and .png
beside this script, unless --output-dir or --stem is supplied.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
PURPLE = '#AA4499'
ORANGE = '#EE7733'


@dataclass(frozen=True)
class Measurement:
    lambda_: float
    gamma: float
    weight_scheme: str
    macro_f1: float
    f1_sd: float
    mu: float
    latency_ms: float
    gflops: float


PENALTY_SWEEP = (
    Measurement(0.05, 0, 'Uniform', 0.646, 0.026, 0.574, 1442, 0.251),
    Measurement(0.1, 0, 'Uniform', 0.651, 0.021, 0.674, 1435, 0.290),
    Measurement(0.2, 0, 'Uniform', 0.644, 0.024, 0.533, 945, 0.230),
    Measurement(0.5, 0, 'Uniform', 0.668, 0.030, 0.558, 1373, 0.244),
    Measurement(1, 0, 'Uniform', 0.630, 0.032, 0.411, 654, 0.180),
    Measurement(2, 0, 'Uniform', 0.608, 0.011, 0.331, 948, 0.146),
)
WEIGHT_SWEEP = (
    Measurement(0.5, 0, 'Uniform control', 0.5893, 0.0848, 0.347, 318, 0.1533),
    Measurement(0.5, 0.25, 'Non-uniform', 0.6430, 0.0349, 0.511, 615, 0.2190),
    Measurement(0.5, 0.5, 'Non-uniform', 0.6662, 0.0182, 0.732, 222, 0.3088),
    Measurement(0.5, 0.75, 'Non-uniform', 0.6668, 0.0154, 0.718, 165, 0.3015),
    Measurement(0.5, 1, 'Non-uniform', 0.6497, 0.0160, 0.705, 143, 0.2946),
    Measurement(0.5, 1.5, 'Non-uniform', 0.6164, 0.0135, 0.473, 32, 0.1943),
    Measurement(0.5, 2, 'Non-uniform', 0.6114, 0.0100, 0.500, 32, 0.2049),
)

MODALITIES = ('Video', 'Audio', 'Text', 'Hand', 'Head', 'Rotation')
# LEGACY input from the earlier request, not the newer unnormalized raw weights.
# gamma, lambda, penalties in MODALITIES order, supplied total penalty.
PENALTY_ALLOCATIONS = (
    (0, 0.5, (0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333), 0.5),
    (0.25, 0.5, (0.20285008, 0.10004362, 0.05608066, 0.04170908, 0.04040037, 0.05891619), 0.5),
    (0.5, 0.5, (0.33648473, 0.08185370, 0.02571287, 0.01422533, 0.01334475, 0.02837862), 0.5),
    (0.75, 0.5, (0.42294607, 0.05074215, 0.00893623, 0.00367668, 0.00334007, 0.01035881), 0.5),
    (1, 0.5, (0.46490659, 0.02750987, 0.00271555, 0.00083063, 0.00073108, 0.00330629), 0.5),
    (1.5, 0.5, (0.49233046, 0.00708661, 0.00021977, 0.00003720, 0.00003070, 0.00029527), 0.5),
    (2, 0.5, (0.49821054, 0.00174444, 0.00001700, 0.00000159, 0.00000123, 0.00002520), 0.5),
)
REFERENCE_F1 = 0.668
REFERENCE_LATENCY_MS = 1373


def validate_data() -> None:
    """Validate the supplied tables without changing rounded input values."""
    assert len(PENALTY_SWEEP) == 6 and len(WEIGHT_SWEEP) == 7
    for row in (*PENALTY_SWEEP, *WEIGHT_SWEEP):
        assert all(np.isfinite(value) for value in (
            row.lambda_, row.gamma, row.macro_f1, row.f1_sd,
            row.mu, row.latency_ms, row.gflops))
        assert 0 <= row.macro_f1 <= 1 and 0 <= row.mu <= 1
        assert min(row.lambda_, row.gamma, row.f1_sd, row.latency_ms, row.gflops) >= 0
    assert all(row.gamma == 0 and row.weight_scheme == 'Uniform' for row in PENALTY_SWEEP)
    assert all(row.lambda_ == 0.5 for row in WEIGHT_SWEEP)
    assert [row.gamma for row in WEIGHT_SWEEP] == [row[0] for row in PENALTY_ALLOCATIONS]
    for _, penalty, weights, total in PENALTY_ALLOCATIONS:
        assert len(weights) == len(MODALITIES)
        assert all(np.isfinite(weights)) and min(weights) >= 0
        assert abs(sum(weights) - total) <= 3.1e-8  # six rounded eight-decimal inputs
        assert penalty == total == 0.5
    deployed = next(row for row in PENALTY_SWEEP if row.lambda_ == 0.5)
    assert deployed.macro_f1 == REFERENCE_F1
    assert deployed.latency_ms == REFERENCE_LATENCY_MS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep', choices=('weights', 'both'), default='weights',
                        help='Weights only (default), or the older two-panel appendix view.')
    parser.add_argument('--secondary', choices=('latency', 'mu', 'gflops', 'none'),
                        default='latency', help='Metric for orange bars (default: latency).')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--stem', default=None)
    parser.add_argument('--width-inches', type=float, default=None)
    parser.add_argument('--height-inches', type=float, default=None)
    parser.add_argument('--font-size', type=float, default=None)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    weights_only = args.sweep == 'weights'
    if args.stem is None:
        args.stem = 'modality_weight_sensitivity_iemocap' if weights_only else 'reward_sensitivity'
    defaults = {'width_inches': 4.5 if weights_only else 6.8,
                'height_inches': 3.2 if weights_only else 2.6,
                'font_size': 12 if weights_only else 8}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ('width_inches', 'height_inches', 'font_size', 'dpi'):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and positive.')
    if Path(args.stem).name != args.stem or args.stem in ('', '.', '..'):
        parser.error('--stem must be a filename stem, not a path.')
    return args


def secondary_spec(name: str):
    return {
        'latency': ('latency_ms', 'Latency (ms)', 1600, [0, 400, 800, 1200, 1600]),
        'mu': ('mu', 'Modality usage', 1.0, [0, 0.25, 0.5, 0.75, 1.0]),
        'gflops': ('gflops', 'GFLOPs', 0.4, [0, 0.1, 0.2, 0.3, 0.4]),
    }[name]


def build_weight_figure(secondary='latency', width=4.5, height=3.2, font_size=12):
    """Single-panel gamma sweep; separate deployed records are not substituted."""
    validate_data()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': font_size,
        'mathtext.fontset': 'dejavusans', 'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 1, 'ytick.labelsize': font_size - 1,
        'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    fig, ax = plt.subplots(figsize=(width, height))
    rows = WEIGHT_SWEEP
    x = np.arange(len(rows))
    paired = secondary != 'none'
    f1_bars = ax.bar(
        x - 0.18 if paired else x, [row.macro_f1 for row in rows],
        width=0.34 if paired else 0.62, color=PURPLE,
        yerr=[row.f1_sd for row in rows], capsize=2.2, zorder=3,
        error_kw={'ecolor': 'black', 'elinewidth': 0.85, 'capthick': 0.85},
    )
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_ylabel('Macro-F1', labelpad=3)
    # These are discrete tested configurations; spacing is categorical.
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_xticks(x, [f'{row.gamma:g}' for row in rows])
    ax.set_xlabel(r'Modality weighting, $\gamma$ ($\lambda = 0.5$)', labelpad=4)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E6E6E6', linewidth=0.45)
    ax.tick_params(direction='out', length=2.6, width=0.7, pad=2)
    handles = [Patch(facecolor=PURPLE, label='Macro-F1 (±1 SD)')]
    other_bars = None
    other_axes = []
    if paired:
        field, ylabel, ymax, ticks = secondary_spec(secondary)
        if secondary == 'latency':
            ymax, ticks = 700, [0, 200, 400, 600]
        other = ax.twinx()
        other_bars = other.bar(
            x + 0.18, [getattr(row, field) for row in rows], width=0.34,
            color=ORANGE, zorder=2,
        )
        other.set_ylim(0, ymax)
        other.set_yticks(ticks)
        other.set_ylabel(ylabel, labelpad=4)
        other.tick_params(direction='out', length=2.6, width=0.7, pad=2)
        other_axes.append(other)
        handles.append(Patch(facecolor=ORANGE, label=ylabel))
    for axis in (ax, *other_axes):
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
    legend = fig.legend(
        handles=handles, ncol=len(handles), loc='upper center',
        bbox_to_anchor=(0.5, 1 - 0.035 / height), borderaxespad=0,
        fontsize=font_size - 1, frameon=True, fancybox=False,
        facecolor='#FAFAFA', edgecolor='#CBCBCB', framealpha=1,
        columnspacing=1.0, handlelength=1.2, handletextpad=0.4, borderpad=0.3,
    )
    legend.get_frame().set_linewidth(0.55)
    fig.subplots_adjust(left=0.58 / width,
                        right=1 - (0.70 if paired else 0.08) / width,
                        bottom=0.49 / height, top=1 - 0.36 / height)
    return fig, [ax], other_axes, [(rows, f1_bars, other_bars)]


def build_figure(secondary='latency', width=6.8, height=2.6, font_size=8):
    validate_data()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': font_size,
        'mathtext.fontset': 'dejavusans', 'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 1, 'ytick.labelsize': font_size - 1,
        'axes.edgecolor': 'black', 'axes.linewidth': 0.65,
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    paired = secondary != 'none'
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True)
    secondary_axes = []
    metadata = []
    for index, (ax, rows, parameter, title) in enumerate(zip(
        axes, (PENALTY_SWEEP, WEIGHT_SWEEP), ('lambda_', 'gamma'),
        (r'(a) Penalty sweep ($\gamma=0$, uniform)',
         r'(b) Weight sweep ($\lambda=0.5$)'),
    )):
        x = np.arange(len(rows))
        f1 = np.array([row.macro_f1 for row in rows])
        sd = np.array([row.f1_sd for row in rows])
        bar_width = 0.34 if paired else 0.62
        f1_bars = ax.bar(
            x - 0.18 if paired else x, f1, width=bar_width,
            color=PURPLE, yerr=sd, capsize=2.1, zorder=3,
            error_kw={'ecolor': 'black', 'elinewidth': 0.7, 'capthick': 0.7},
        )
        ax.axhline(REFERENCE_F1, color=PURPLE, linewidth=1.05,
                   linestyle=(0, (4, 2)), zorder=4)
        ax.set_ylim(0, 0.82)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xlim(-0.55, len(rows) - 0.45)
        ax.set_xticks(x, [f'{getattr(row, parameter):g}' for row in rows])
        ax.set_xlabel('Penalty strength ' + r'$\lambda$' if index == 0
                      else 'Weight concentration ' + r'$\gamma$', labelpad=3)
        ax.set_title(title, loc='left', fontsize=font_size, pad=5)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E6E6E6', linewidth=0.4)
        ax.tick_params(direction='out', length=2.2, width=0.55, pad=2)
        if index == 0:
            ax.set_ylabel('Macro-F1', labelpad=3)
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)
        other_bars = None
        if paired:
            field, ylabel, ymax, ticks = secondary_spec(secondary)
            other = ax.twinx()
            other_bars = other.bar(
                x + 0.18, [getattr(row, field) for row in rows],
                width=bar_width, color=ORANGE, zorder=2,
            )
            other.set_ylim(0, ymax)
            other.set_yticks(ticks)
            other.tick_params(direction='out', length=2.2, width=0.55, pad=2,
                              right=index == 1, labelright=index == 1)
            if index == 1:
                other.set_ylabel(ylabel, labelpad=3)
            if secondary == 'latency':
                other.axhline(REFERENCE_LATENCY_MS, color=ORANGE, linewidth=1.05,
                              linestyle=(0, (1.5, 1.5)), zorder=4)
            # Put F1/error bars and its reference above the second axes' background.
            ax.set_zorder(other.get_zorder() + 1)
            ax.patch.set_visible(False)
            secondary_axes.append(other)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.65)
        metadata.append((rows, f1_bars, other_bars))

    handles = [Patch(facecolor=PURPLE, label='Macro-F1 (±1 SD)')]
    if paired:
        handles.append(Patch(facecolor=ORANGE, label=secondary_spec(secondary)[1]))
    handles.append(Line2D([], [], color=PURPLE, linestyle=(0, (4, 2)), linewidth=1.05,
                          label=f'Deployed F1: {REFERENCE_F1:.3f}'))
    if secondary == 'latency':
        handles.append(Line2D([], [], color=ORANGE, linestyle=(0, (1.5, 1.5)), linewidth=1.05,
                              label=f'Deployed latency: {REFERENCE_LATENCY_MS:g} ms'))
    legend = fig.legend(handles=handles, ncol=len(handles), loc='upper center',
                        bbox_to_anchor=(0.5, 1 - 0.025 / height), borderaxespad=0,
                        fontsize=font_size - 1, frameon=True, fancybox=False,
                        facecolor='#FAFAFA', edgecolor='#B8B8B8', framealpha=1,
                        columnspacing=1.0, handlelength=1.7, handletextpad=0.45,
                        borderpad=0.3)
    legend.get_frame().set_linewidth(0.55)
    fig.subplots_adjust(left=0.43 / width, right=1 - (0.55 if paired else 0.05) / width,
                        bottom=0.36 / height, top=1 - 0.46 / height, wspace=0.13)
    return fig, axes, secondary_axes, metadata


def validate_figure(fig, metadata, secondary):
    """Verify bar heights and readable layout before writing the figure."""
    for rows, f1_bars, other_bars in metadata:
        np.testing.assert_array_equal([bar.get_height() for bar in f1_bars],
                                      [row.macro_f1 for row in rows])
        ax = f1_bars[0].axes
        lo, hi = ax.get_ylim()
        if lo > 0 or any(row.macro_f1 - row.f1_sd < lo or
                         row.macro_f1 + row.f1_sd > hi for row in rows):
            raise ValueError('The zero bar baseline and full F1 error bars must be visible.')
        segments = f1_bars.errorbar.lines[2][0].get_segments()
        np.testing.assert_allclose(
            [[segment[0, 1], segment[1, 1]] for segment in segments],
            [[row.macro_f1 - row.f1_sd, row.macro_f1 + row.f1_sd] for row in rows])
        if secondary != 'none':
            field = secondary_spec(secondary)[0]
            np.testing.assert_array_equal([bar.get_height() for bar in other_bars],
                                          [getattr(row, field) for row in rows])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    for text in visible:
        box = text.get_window_extent(renderer)
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
            raise ValueError(f'Clipped label {text.get_text()!r}; increase figure dimensions.')
    for index, left in enumerate(visible):
        for right in visible[index + 1:]:
            if left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer)):
                raise ValueError(f'Overlapping labels {left.get_text()!r} and {right.get_text()!r}.')


def main():
    args = parse_args()
    builder = build_weight_figure if args.sweep == 'weights' else build_figure
    fig, axes, other_axes, metadata = builder(
        args.secondary, args.width_inches, args.height_inches, args.font_size)
    try:
        for dpi in (100, args.dpi):
            fig.set_dpi(dpi)
            validate_figure(fig, metadata, args.secondary)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ('pdf', 'png'):
            path = args.output_dir / f'{args.stem}.{suffix}'
            fig.savefig(path, dpi=args.dpi, bbox_inches=None, pad_inches=0)
            print(f'Wrote {path.resolve()}')
    finally:
        plt.close(fig)
    print(f'Plotted {sum(len(rows) for rows, _, _ in metadata)} supplied measurements.')
    print('Error bars: ±1 SD for Macro-F1 only; no latency uncertainty was supplied.')
    print('NOTE: uniform controls differ across studies; no cross-study equivalence is assumed.')


if __name__ == '__main__':
    main()
