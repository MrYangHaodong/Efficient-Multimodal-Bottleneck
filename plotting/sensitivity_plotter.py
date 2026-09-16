#!/usr/bin/env python3
"""Compact, single-row IEMOCAP sensitivity plots from the supplied F1 values.

No measurements are loaded from the existing sensitivity CSV: the numbers below
are the user's exact inputs. In that CSV, the supplied K sweep matches canonical
F1, whereas the lambda sweeps match order-averaged F1. Do not describe all three
as the same evaluation protocol without confirming their provenance. No matching
uncertainties were supplied, so this plot intentionally has no error bars.

The default puts all three panels together within ``0.3 text width``:
  row:       three approximately 0.3-textwidth panels in one 5.5-inch row;
  miniature: the entire three-panel figure in 0.3 textwidth (1.65 inches).
The miniature keeps every data point but labels fewer x ticks for readability;
the quarter-value tick is written as ¼ to accommodate larger print-size fonts.
The default writes the miniature as PDF and 600-dpi PNG without an outer frame.

The 5.5-inch text width is specified in the ICLR 2027 template:
https://raw.githubusercontent.com/ICLR/Master-Template/master/iclr2027/iclr2027_conference.sty

Examples:
    python plotting/sensitivity_plotter.py
    python plotting/sensitivity_plotter.py --layout row
    python plotting/sensitivity_plotter.py --layout row --width-inches 4.5

Dependency: matplotlib. No downloads, LaTeX installation, or training required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullLocator


COLOR = "#AA4499"
ICLR_TEXT_WIDTH_INCHES = 5.5
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class Sweep:
    name: str
    label: str
    values: tuple[float, ...]
    f1: tuple[float, ...]


SWEEPS = (
    Sweep(
        "bottleneck_tokens", r"$K$ (tokens)",
        (4, 8, 16, 32),
        (0.6593, 0.6598, 0.6680, 0.6346),
    ),
    Sweep(
        "lambda_kd", r"$\lambda_{\mathrm{KD}}$",
        (0.25, 0.5, 1, 2, 4),
        (0.6461, 0.6748, 0.6546, 0.6572, 0.6561),
    ),
    Sweep(
        "lambda_ds", r"$\lambda_{\mathrm{DS}}$",
        (0.25, 0.5, 1, 2, 4),
        (0.6734, 0.6455, 0.6748, 0.6459, 0.6528),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("row", "miniature", "both"), default="miniature")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--width-inches", type=float, default=None,
                        help="Override width for one layout; cannot be used with 'both'.")
    parser.add_argument("--height-inches", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.layout == "both" and (args.width_inches is not None or args.height_inches is not None):
        parser.error("Choose row or miniature when overriding physical dimensions.")
    if args.width_inches is not None and args.width_inches <= 0:
        parser.error("--width-inches must be positive.")
    if args.height_inches is not None and args.height_inches <= 0:
        parser.error("--height-inches must be positive.")
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    return args


def build_figure(layout: str, width: float | None = None, height: float | None = None):
    """Use shared linear F1 limits and true base-two parameter spacing."""
    miniature = layout == "miniature"
    width = width if width is not None else ICLR_TEXT_WIDTH_INCHES * (0.3 if miniature else 1.0)
    height = height if height is not None else (0.72 if miniature else 1.35)
    tick_size = 7.0 if miniature else 7.5
    label_size = 8.5
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": tick_size,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.45 if miniature else 0.65,
        "text.color": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(width, height), sharey=True)
    for index, (ax, sweep) in enumerate(zip(axes, SWEEPS)):
        ax.set_xscale("log", base=2)
        ax.plot(
            sweep.values, sweep.f1,
            color=COLOR, linewidth=0.8 if miniature else 1.2,
            marker="o", markersize=2.3 if miniature else 3.6,
            markerfacecolor=COLOR, markeredgecolor="white",
            markeredgewidth=0.25 if miniature else 0.4,
            clip_on=True, zorder=3,
        )
        ticks = sweep.values
        if miniature:
            ticks = ticks[::2] if len(ticks) == 5 else (ticks[0], ticks[-1])
        tick_labels = ["¼" if miniature and value == 0.25 else f"{value:g}" for value in ticks]
        ax.set_xticks(ticks, tick_labels)
        if miniature and index == 0:
            ax.get_xticklabels()[-1].set_horizontalalignment("right")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(sweep.values[0] / 2**0.32, sweep.values[-1] * 2**0.32)
        ax.set_ylim(0.630, 0.682)
        ax.set_yticks((0.64, 0.66, 0.68))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(
            direction="out", length=1.7 if miniature else 2.2,
            width=0.4 if miniature else 0.55,
            pad=1.0 if miniature else 1.6,
            labelleft=index == 0,
        )
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.35 if miniature else 0.45)
        label = r"$K$" if miniature and index == 0 else sweep.label
        ax.set_xlabel(label, labelpad=1.0 if miniature else 2.0)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_visible(True)
    axes[0].set_ylabel("F1", labelpad=1.0 if miniature else 2.2)

    # Physical margins preserve print-size typography and the exact export width.
    left = 0.40 if miniature else 0.44
    right = 0.025 if miniature else 0.035
    bottom = 0.31 if miniature else 0.33
    top = 0.055 if miniature else 0.045
    if width <= left + right + 0.45 or height <= bottom + top + 0.20:
        raise ValueError("Requested dimensions leave too little room for three readable panels.")
    fig.subplots_adjust(
        left=left / width, right=1 - right / width,
        bottom=bottom / height, top=1 - top / height,
        wspace=0.14 if miniature else 0.085,
    )
    return fig, axes


def validate_figure(fig, axes) -> None:
    """Catch data drift and labels clipped by the exact-size PDF canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for ax, sweep in zip(axes, SWEEPS):
        line = ax.lines[0]
        if tuple(line.get_xdata()) != sweep.values or tuple(line.get_ydata()) != sweep.f1:
            raise AssertionError(f"Plotted data differ from supplied values: {sweep.name}")
        texts = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]
        for text in texts:
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < canvas.x0 - 0.5 or box.x1 > canvas.x1 + 0.5 or box.y0 < canvas.y0 - 0.5 or box.y1 > canvas.y1 + 0.5:
                raise ValueError(f"Label {text.get_text()!r} is clipped; increase the figure dimensions.")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layouts = ("row", "miniature") if args.layout == "both" else (args.layout,)
    for layout in layouts:
        fig, axes = build_figure(layout, args.width_inches, args.height_inches)
        validate_figure(fig, axes)
        stem = f"iemocap_sensitivity_{layout}"
        for suffix in ("pdf", "png"):
            path = args.output_dir / f"{stem}.{suffix}"
            # No tight-bbox crop: the requested physical width must stay exact.
            fig.savefig(path, dpi=args.dpi, bbox_inches=None, pad_inches=0)
            print(path.resolve())
        print(f"  Size: {fig.get_figwidth():.2f} x {fig.get_figheight():.2f} inches")
        plt.close(fig)
    print("Used all 14 supplied F1 values exactly; no error bars were inferred.")
    print("Provenance check: supplied K values match canonical-order F1 in the existing CSV; lambda values match order-averaged F1.")


if __name__ == "__main__":
    main()
