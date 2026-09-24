#!/usr/bin/env python3
"""Generate the Android-only plot for Figure 1(d), without a panel label.

Use the same workbook reader, units, colors, markers, and F1 rounding as
energy_latency_full_plotter.py. Show IEMOCAP Android INT8 measurements for
SeMARC, MBT, AdaMML, and DyMo; omit DyMM and the no-RL ablation for this teaser.
There is no zoom inset or overall title. F1 annotations are the shared
full-availability reference scores, not Android-specific accuracy measurements.

Example:
    python intro_teaser_android_plotter.py --input results.xlsx

Dependencies: matplotlib, openpyxl. Outputs: PDF, PNG, and editable SVG.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

from energy_latency_full_plotter import (
    METHOD_COLORS,
    METHOD_MARKERS,
    SHEET_URL,
    Measurement,
    f1_label,
    load_workbook,
    read_f1_scores,
    read_measurements,
)


METHODS = ("SeMARC", "MBT", "AdaMML", "DyMo")
PLATFORM = "Android (INT8)"
DATASET = "IEMOCAP"
FONT_INCREMENT = 4.0
LEGEND_DESCRIPTORS = {
    "SeMARC": "Ours",
    "MBT": "Monolithic",
    "AdaMML": "Selective",
    "DyMo": "Selective",
}


def make_legend(fig):
    """Two borderless rows with smaller, inline parenthetical descriptors."""
    entries = {}
    for method in METHODS:
        symbol = DrawingArea(19, 18, 0, 4)
        symbol.add_artist(Line2D(
            [9], [7], linestyle="none", marker=METHOD_MARKERS[method],
            markerfacecolor=METHOD_COLORS[method], markeredgecolor="white",
            markeredgewidth=0.8, markersize=16 if method == "SeMARC" else 9,
        ))
        name = TextArea(method, textprops={"fontsize": 16.6, "color": "black"})
        descriptor = TextArea(
            f"({LEGEND_DESCRIPTORS[method]})",
            textprops={"fontsize": 11.6, "color": "black"},
        )
        entries[method] = HPacker(
            children=[symbol, name, descriptor], align="baseline", pad=0, sep=3,
        )
    columns = [
        VPacker(children=[entries[top], entries[bottom]], align="left", pad=0, sep=3)
        for top, bottom in (("SeMARC", "AdaMML"), ("MBT", "DyMo"))
    ]
    box = HPacker(children=columns, align="top", pad=0, sep=16)
    legend = AnchoredOffsetbox(
        loc="upper center", child=box, frameon=False, pad=0, borderpad=0,
        bbox_to_anchor=(0.56, 0.995), bbox_transform=fig.transFigure,
    )
    fig.add_artist(legend)
    return legend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=SHEET_URL, help="Local XLSX or XLSX export URL.")
    parser.add_argument("--sheet", default="Energy_Study")
    parser.add_argument("--results-sheet", default="Main_ResultsFull")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--stem", default="intro_teaser_android")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def select_measurements(measurements: list[Measurement]) -> list[Measurement]:
    selected = [
        point for point in measurements
        if point.dataset.casefold() == DATASET.casefold()
        and point.platform == PLATFORM
        and point.method in METHODS
    ]
    by_method = {point.method: point for point in selected}
    if len(selected) != len(by_method):
        raise ValueError("Duplicate Android IEMOCAP measurements.")
    missing = set(METHODS) - set(by_method)
    if missing:
        raise ValueError(f"Missing Android IEMOCAP measurements: {sorted(missing)}")
    for point in selected:
        if not all(math.isfinite(value) and value > 0 for value in (point.latency_s, point.energy_j)):
            raise ValueError(f"Invalid measurement: {point}")
    return [by_method[method] for method in METHODS]


def build_figure(points: list[Measurement], f1_scores: dict[str, float]):
    """Return the single axes and its figure, with all F1 labels inside the axes."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15.0 + FONT_INCREMENT,
        "axes.labelsize": 17.2 + FONT_INCREMENT,
        "xtick.labelsize": 14.5 + FONT_INCREMENT,
        "ytick.labelsize": 14.5 + FONT_INCREMENT,
        "legend.fontsize": 12.6 + FONT_INCREMENT,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(5.7, 3.7))
    offsets = {"SeMARC": (12, 12), "MBT": (-12, 0), "AdaMML": (-12, -16), "DyMo": (-12, 16)}
    annotations = []
    for point in points:
        ours = point.method == "SeMARC"
        ax.plot(
            point.latency_s, point.energy_j, linestyle="none",
            marker=METHOD_MARKERS[point.method],
            markersize=25.5 if ours else 12.0,
            markerfacecolor=METHOD_COLORS[point.method],
            markeredgecolor="white", markeredgewidth=1.1,
            zorder=6 if ours else 5,
        )
        dx, dy = offsets[point.method]
        annotations.append(ax.annotate(
            f1_label(f1_scores[point.method]),
            xy=(point.latency_s, point.energy_j),
            xytext=(dx, dy), textcoords="offset points",
            ha="left" if dx > 0 else "right", va="center",
            fontsize=14.0 + FONT_INCREMENT,
            color=METHOD_COLORS[point.method],
            arrowprops={
                "arrowstyle": "-", "color": METHOD_COLORS[point.method],
                "linewidth": 0.6, "shrinkA": 2, "shrinkB": 10 if ours else 6,
            },
            zorder=8,
        ))

    xs = [point.latency_s for point in points]
    ys = [point.energy_j for point in points]
    x_span, y_span = max(xs) - min(xs), max(ys) - min(ys)
    ax.set_xlim(max(0, min(xs) - 0.14 * x_span), max(xs) + 0.20 * x_span)
    ax.set_ylim(max(0, min(ys) - 0.16 * y_span), max(ys) + 0.25 * y_span)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel("Latency per inference (s)", labelpad=4)
    ax.set_ylabel("Energy per inference (J)", labelpad=5)
    ax.tick_params(direction="out", length=3, width=0.7, pad=2)
    ax.grid(True, color="#D8D8D8", linewidth=0.55, alpha=0.82)
    ax.set_axisbelow(True)

    legend = make_legend(fig)
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.21, top=0.86)
    # Reserve only the compact legend's measured height plus a small gap.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bottom = legend.get_window_extent(renderer).y0
    fig.subplots_adjust(top=(legend_bottom - 4) / fig.bbox.height)
    return fig, ax, legend, annotations


def validate_layout(fig, ax, legend, annotations) -> None:
    """Reject clipped/overlapping annotations before exporting the paper panel."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    if not (fig.bbox.x0 <= legend_box.x0 and legend_box.x1 <= fig.bbox.x1
            and fig.bbox.y0 <= legend_box.y0 and legend_box.y1 <= fig.bbox.y1):
        raise ValueError("The legend extends beyond the export canvas.")
    if legend_box.y0 < axes_box.y1 + 2:
        raise ValueError("The legend overlaps the plot; increase its top spacing.")
    boxes = [Text.get_window_extent(label, renderer) for label in annotations]
    for label, box in zip(annotations, boxes):
        if not (axes_box.x0 <= box.x0 and box.x1 <= axes_box.x1
                and axes_box.y0 <= box.y0 and box.y1 <= axes_box.y1):
            raise ValueError(f"F1 annotation is outside the plot: {label.get_text()}")
    for index, box in enumerate(boxes):
        if any(box.overlaps(other) for other in boxes[index + 1:]):
            raise ValueError("F1 annotations overlap.")


def render(points, f1_scores, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    fig, ax, legend, annotations = build_figure(points, f1_scores)
    validate_layout(fig, ax, legend, annotations)
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.010)
    # Give the right border a small exterior margin while keeping other sides tight.
    export_bounds = Bbox.from_extents(bounds.x0, bounds.y0, bounds.x1 + 0.050, bounds.y1)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches=export_bounds, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    book = load_workbook(args.input)
    measurements, _, _ = read_measurements(book, args.sheet)
    points = select_measurements(measurements)
    scores = read_f1_scores(book, args.results_sheet, DATASET)
    paths = render(points, scores, args.output_dir, args.stem, args.dpi)
    print(f"{DATASET}, {PLATFORM}; F1 uses full-availability reference scores.")
    for point in points:
        print(f"  {point.method}: {point.latency_s:g} s, {point.energy_j:g} J, {f1_label(scores[point.method])}")
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
