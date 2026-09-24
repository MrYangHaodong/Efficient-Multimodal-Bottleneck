#!/usr/bin/env python3
"""Selected IEMOCAP ablations at 0% missingness, relative to full-model F1.

Source: the user's supplied ablation table and final selection (2026-09-20).
Full-model Macro-F1 = 0.668, as specified in the supplied reference script.
Preserve the supplied four-decimal variant scores; do not substitute other
experiments or infer uncertainty. In particular, no RL is 0.6443, not the
separate no-learned-stop experiment (0.6459). Display labels follow the user's
final terminology; source_label retains the original supplied descriptions.

Bars show 100 * (variant F1 - full-model F1) / full-model F1. These are relative
percentage changes, not percentage-point differences or gains with a variant
denominator. Ablated F1 scores are displayed to two decimals in parentheses;
calculations retain the original four-decimal scores. No uncertainty
or sample counts were supplied, so no error bars/significance claims are made.

Layout: downward bars and endpoint score labels inspired by Figure 6, page 9,
https://arxiv.org/pdf/2509.25278. That reference uses absolute accuracy drops;
this figure instead uses the explicitly requested relative Macro-F1 changes.
All bars use the paper's Macro-F1 purple. A dotted divider separates ARC (RL)
from SeMA (backbone); plain Times New Roman labels sit inside their sections.
Category labels appear above the plot. Times New Roman must be installed;
the script refuses a silent font substitution.
All text is enlarged by another 4 points: ticks 36 pt and the axis label
39.5 pt; values/group labels are 27.5 pt. Display rounding never changes
bar heights. Top labels use one row tilted 45 degrees, with compact 0.95 line
spacing for multiline labels. The y-axis title uses 1.0 line
spacing and wraps onto two lines. The 16.15-inch width is retained, with an
8.8-inch working height to fit the larger text and complete mask expression;
there are no staggered label levels or guide lines.
All three exports are cropped to the visible content with a 1.5-point safety
margin for vector-backend font metrics. Cropping does not rescale fonts or the plot;
PDF and SVG retain vector graphics.

Run: python plotting/plot_ablation.py
Updates the same ablation_iemocap.pdf, .svg, and .png beside this script.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import Affine2D, Bbox


HERE = Path(__file__).resolve().parent
STEM = "ablation_iemocap"
BASELINE_F1 = Decimal("0.668")
BAR_COLOR = "#AA4499"
FILL_ALPHA = 0.75
GROUP_FONT = "Times New Roman"
GROUP_DIVIDER = 8.0
X_POSITIONS = (0.0, 2.09, 3.86, 6.35, 8.99, 10.84, 12.69)
DEFAULT_WIDTH = 16.15
DEFAULT_HEIGHT = 8.8
DEFAULT_FONT_SIZE = 28.0
EXPORT_PAD_PT = 1.5
LABEL_PAD = 7
LABEL_ROTATION = 45
CATEGORY_LINE_SPACING = 0.95
AXIS_TITLE_LINE_SPACING = 1.0


@dataclass(frozen=True)
class Ablation:
    group: str
    label: str
    source_label: str
    f1: Decimal

    @property
    def relative_change_pct(self) -> Decimal:
        return 100 * (self.f1 - BASELINE_F1) / BASELINE_F1


ABLATIONS = (
    Ablation("RL", "Heuristic\n(no RL)",
             "No RL: Shapley ordering + 80% confidence exit", Decimal("0.6443")),
    Ablation("RL", "No Q-prior\n" + r"($Q_0=0$)",
             "v(s) zero-initialized", Decimal("0.6173")),
    Ablation("RL", r"$[\mathbf{1}_{S_s}\Vert\mathbf{1}_{\mathcal{A}}]$ only" + "\n(state)",
             "RL state: masks only", Decimal("0.6447")),
    Ablation("RL", r"$p_s$ only" + "\n(state)",
             "RL state: p_s derivatives only", Decimal("0.6651")),
    Ablation("Backbone", "Fixed-order\ntraining",
             "No random-order training", Decimal("0.6406")),
    Ablation("Backbone", "Bottleneck\nsuffix",
             "Bottleneck position", Decimal("0.6463")),
    Ablation("Backbone", "Final-prefix\nCE only",
             "Only CE loss", Decimal("0.6491")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--width-inches", type=float, default=DEFAULT_WIDTH)
    parser.add_argument("--height-inches", type=float, default=DEFAULT_HEIGHT)
    parser.add_argument("--font-size", type=float, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    for name in ("width_inches", "height_inches", "font_size", "dpi"):
        value = getattr(args, name)
        if not isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.font_size < 6:
        parser.error("--font-size must be at least 6 for readable labels")
    return args


def validate_data() -> None:
    expected_scores = ("0.6443", "0.6173", "0.6447", "0.6651",
                       "0.6406", "0.6463", "0.6491")
    expected_percentages = ("-3.55", "-7.59", "-3.49", "-0.43",
                            "-4.10", "-3.25", "-2.83")
    if tuple(str(row.f1) for row in ABLATIONS) != expected_scores:
        raise ValueError("The selected scores do not match the supplied table")
    if tuple(f"{row.relative_change_pct:.2f}" for row in ABLATIONS) != expected_percentages:
        raise ValueError("Relative changes do not match the independent calculation")
    if tuple(row.group for row in ABLATIONS) != ("RL",) * 4 + ("Backbone",) * 3:
        raise ValueError("Expected four RL and three backbone ablations")
    if not all(0 < row.f1 < BASELINE_F1 <= 1 for row in ABLATIONS):
        raise ValueError("Expected valid Macro-F1 scores below the supplied baseline")


def build_figure(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, font_size=DEFAULT_FONT_SIZE):
    validate_data()
    findfont(FontProperties(family=GROUP_FONT), fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": font_size,
        "mathtext.fontset": "dejavusans", "axes.labelsize": font_size + 11.5,
        "xtick.labelsize": font_size + 8, "ytick.labelsize": font_size + 8,
        "axes.edgecolor": "black", "axes.linewidth": 0.8,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": STEM, "axes.unicode_minus": True,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(width, height))
    x = X_POSITIONS
    values = [float(row.relative_change_pct) for row in ABLATIONS]
    bars = ax.bar(x, values, width=1.0,
                  color=to_rgba(BAR_COLOR, FILL_ALPHA),
                  edgecolor="black", linewidth=0.65, zorder=3)
    for xi, value, row in zip(x, values, ABLATIONS):
        label = f"{row.relative_change_pct:+.2f}%\n({row.f1:.2f})"
        ax.annotate(label.replace("-", "\N{MINUS SIGN}"), (xi, value),
                    xytext=(0, -5), textcoords="offset points",
                    ha="center", va="top", fontsize=font_size - 0.5,
                    linespacing=1.1, color="black", zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92,
                          "pad": 0.3})

    ax.set_xlim(-1.23, 14.7)
    ax.set_ylim(-9.25, 0)
    ax.set_xticks(x, [row.label for row in ABLATIONS])
    ax.set_yticks([-8, -6, -4, -2, 0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Relative Macro-F1\nchange (%)", labelpad=6,
                  linespacing=AXIS_TITLE_LINE_SPACING)
    ax.grid(axis="y", color="#D0D0D0", linestyle="--", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0, pad=LABEL_PAD, top=True, labeltop=True,
                   bottom=False, labelbottom=False)
    ax.tick_params(axis="y", direction="out", length=3, width=0.65, pad=3)
    for label in ax.get_xticklabels():
        label.set_linespacing(CATEGORY_LINE_SPACING)
        label.set_rotation(LABEL_ROTATION)
        label.set_rotation_mode("anchor")
        label.set_ha("left")
        label.set_va("bottom")
        label.set_multialignment("left")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    # Both sections use the same metric color; names and the divider identify them.
    ax.axvline(GROUP_DIVIDER, color="black", linestyle=":", linewidth=1.0, zorder=2)
    for right_edge, group in ((7.65, "ARC"), (13.15, "SeMA")):
        ax.text(right_edge, 0.07, group, transform=ax.get_xaxis_transform(),
                ha="right", va="bottom", fontsize=font_size - 0.5,
                fontfamily=GROUP_FONT, color="black")
    fig.subplots_adjust(left=2.55 / width, right=1 - 0.35 / width,
                        bottom=0.2 / height, top=1 - 3.3 / height)
    return fig, ax, bars


def text_outline(text, renderer):
    """Return the actual rotated text rectangle, not its axis-aligned envelope."""
    rotation = text.get_rotation()
    if rotation % 90 == 0:
        box = text.get_window_extent(renderer)
        return [(box.x0, box.y0), (box.x1, box.y0),
                (box.x1, box.y1), (box.x0, box.y1)]
    if text.get_rotation_mode() != "anchor":
        raise ValueError("Tilted labels must rotate around their text anchor")
    try:
        text.set_rotation(0)
        box = text.get_window_extent(renderer)
    finally:
        text.set_rotation(rotation)
    anchor = text.get_transform().transform(text.get_unitless_position())
    return Affine2D().rotate_deg_around(*anchor, rotation).transform(
        [(box.x0, box.y0), (box.x1, box.y0),
         (box.x1, box.y1), (box.x0, box.y1)]
    )


def outlines_overlap(first, second):
    """Separating-axis test avoids false collisions between tilted labels."""
    for polygon in (first, second):
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            normal = (start[1] - end[1], end[0] - start[0])
            projections = [[x * normal[0] + y * normal[1] for x, y in shape]
                           for shape in (first, second)]
            left, right = projections
            if max(left) <= min(right) or max(right) <= min(left):
                return False
    return True


def validate_figure(fig, ax, bars, font_size=DEFAULT_FONT_SIZE) -> None:
    if len(fig.axes) != 1 or len(bars) != 7 or ax.get_title():
        raise ValueError("Expected one untitled panel with exactly seven bars")
    for bar, row in zip(bars, ABLATIONS):
        if bar.get_y() != 0 or bar.get_height() != float(row.relative_change_pct):
            raise ValueError(f"Incorrect bar value/baseline: {row.source_label}")
        if bar.get_facecolor() != to_rgba(BAR_COLOR, FILL_ALPHA):
            raise ValueError(f"Incorrect Macro-F1 color: {row.source_label}")
    expected_labels = [f"{row.relative_change_pct:+.2f}%\n({row.f1:.2f})".replace(
        "-", "\N{MINUS SIGN}") for row in ABLATIONS]
    if [text.get_text() for text in ax.texts[:7]] != expected_labels:
        raise ValueError("Expected two-decimal percentage and F1 annotations")
    group_labels = ax.texts[7:]
    if [text.get_text() for text in group_labels] != ["ARC", "SeMA"]:
        raise ValueError("Expected ARC and SeMA section labels")
    for label in group_labels:
        if label.get_bbox_patch() is not None or label.get_fontfamily() != [GROUP_FONT]:
            raise ValueError("Section labels must use unboxed Times New Roman")
    if len(ax.lines) != 1 or ax.lines[0].get_linestyle() != ":" or any(
        x != GROUP_DIVIDER for x in ax.lines[0].get_xdata()
    ):
        raise ValueError("Expected one dotted divider between the ablation groups")
    if any(tick.label1.get_visible() or not tick.label2.get_visible()
           for tick in ax.xaxis.get_major_ticks()):
        raise ValueError("Category labels must appear only above the plot")
    if [label.get_text() for label in ax.get_xticklabels()] != [row.label for row in ABLATIONS]:
        raise ValueError("Category labels must preserve the requested text and order")
    if any(tick.get_pad() != LABEL_PAD for tick in ax.xaxis.get_major_ticks()):
        raise ValueError("Category labels must share one top level")
    if any(label.get_rotation() != LABEL_ROTATION
           or label.get_rotation_mode() != "anchor"
           or label.get_ha() != "left" or label.get_va() != "bottom"
           for label in ax.get_xticklabels()):
        raise ValueError("Category labels must use the requested 45-degree tilt")
    if ax.collections:
        raise ValueError("The compact layout must not retain staggered-label guides")
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        if label.get_fontsize() != font_size + 8:
            raise ValueError("Tick labels have an unexpected font size")
    if ax.yaxis.label.get_fontsize() != font_size + 11.5:
        raise ValueError("Axis label has an unexpected font size")
    if any(label.get_fontsize() != font_size - 0.5 for label in ax.texts):
        raise ValueError("Value/group labels have an unexpected font size")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Enlarged endpoint labels must stay inside the plot, not erase its border.
    for label in ax.texts[:7]:
        box = label.get_window_extent(renderer)
        padding = 0.3 * fig.dpi / 72
        if box.y0 - padding < ax.bbox.y0 or box.y1 + padding > ax.bbox.y1:
            raise ValueError(f"Endpoint label crosses the plot border: {label.get_text()!r}")
    texts = [text for text in fig.findobj(Text) if text.get_visible() and text.get_text()]
    boxes = [text.get_window_extent(renderer) for text in texts]
    outlines = [text_outline(text, renderer) for text in texts]
    for text, box in zip(texts, boxes):
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
            raise ValueError(f"Clipped label {text.get_text()!r}; increase figure size")
    for index, left in enumerate(outlines):
        for other_index in range(index + 1, len(outlines)):
            if outlines_overlap(left, outlines[other_index]):
                raise ValueError(f"Overlapping labels: {texts[index].get_text()!r}, "
                                 f"{texts[other_index].get_text()!r}")


def tight_export_bbox(fig):
    """Remove blank margins while including rotated glyphs and complete strokes.

    Measure rendered ink, not the extra font-descent space in text layout boxes.
    The vector exports use the same measured page boundary, not a raster image.
    Keep a small point-based margin for antialiasing and vector-font differences.
    """
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    ink = np.any(rgba[:, :, :3] < 255, axis=2) & (rgba[:, :, 3] > 0)
    # Axis reductions avoid allocating a coordinate pair for every ink pixel.
    xs = np.flatnonzero(ink.any(axis=0))
    ys = np.flatnonzero(ink.any(axis=1))
    if not len(xs):
        raise ValueError("Cannot crop an empty figure")
    height = rgba.shape[0]
    padding = EXPORT_PAD_PT / 72
    return Bbox.from_extents(
        xs[0] / fig.dpi - padding, (height - ys[-1] - 1) / fig.dpi - padding,
        (xs[-1] + 1) / fig.dpi + padding, (height - ys[0]) / fig.dpi + padding)


def main() -> None:
    args = parse_args()
    fig, ax, bars = build_figure(args.width_inches, args.height_inches, args.font_size)
    try:
        for dpi in (100, args.dpi):
            fig.set_dpi(dpi)
            validate_figure(fig, ax, bars, args.font_size)
        export_bbox = tight_export_bbox(fig)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "svg", "png"):
            path = args.output_dir / f"{STEM}.{extension}"
            fig.savefig(path, dpi=args.dpi, bbox_inches=export_bbox, pad_inches=0)
            print(f"Created {path.resolve()}")
    finally:
        plt.close(fig)
    print(f"IEMOCAP, 0% missingness. Full-model Macro-F1: {BASELINE_F1}")
    print("Relative change (%) = 100 * (variant F1 - full-model F1) / full-model F1")
    for row in ABLATIONS:
        print(f"  {row.source_label}: F1={row.f1:.4f}, {row.relative_change_pct:+.2f}%")
    print("All seven supplied scores, computed bar heights, and label layout verified.")


if __name__ == "__main__":
    main()
