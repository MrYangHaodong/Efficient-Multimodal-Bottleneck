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
Category labels and y ticks are 36 pt; the axis title is 39.5 pt, and values/
group labels are 27.5 pt. Category parentheticals are 2 pt smaller (34 pt),
inline on the same baseline. All seven category labels are horizontal and
single-line, centered over their bars. The wider 36.8 x 6.6 inch canvas fits
the complete labels without reducing the main fonts. Bars are 0.5 inches
wide, about 40% thinner than the previous 0.832-inch bars. Label widths
determine category spacing; there are no staggered levels or guide lines.
The y-axis title retains its two-line wrap and 1.0 line spacing.
Display rounding never changes bar heights.
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
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import Bbox


HERE = Path(__file__).resolve().parent
STEM = "ablation_iemocap"
BASELINE_F1 = Decimal("0.668")
BAR_COLOR = "#AA4499"
FILL_ALPHA = 0.75
GROUP_FONT = "Times New Roman"
DEFAULT_WIDTH = 36.8
DEFAULT_HEIGHT = 6.6
DEFAULT_FONT_SIZE = 28.0
EXPORT_PAD_PT = 1.5
BAR_WIDTH_INCHES = 0.5
LABEL_BASELINE_PAD_PT = 20
PARENTHETICAL_REDUCTION_PT = 2
CATEGORY_EDGE_PAD_INCHES = 0.2
MIN_CATEGORY_GAP_INCHES = 0.35
AXIS_TITLE_LINE_SPACING = 1.0


@dataclass(frozen=True)
class Ablation:
    group: str
    label: str
    source_label: str
    f1: Decimal
    qualifier: str = ""

    @property
    def relative_change_pct(self) -> Decimal:
        return 100 * (self.f1 - BASELINE_F1) / BASELINE_F1


ABLATIONS = (
    Ablation("RL", "Heuristic",
             "No RL: Shapley ordering + 80% confidence exit", Decimal("0.6443"), "(no RL)"),
    Ablation("RL", "No Q-prior",
             "v(s) zero-initialized", Decimal("0.6173"), r"($Q_0=0$)"),
    Ablation("RL", r"$[\mathbf{1}_{S_s}\Vert\mathbf{1}_{\mathcal{A}}]$ only",
             "RL state: masks only", Decimal("0.6447"), "(state)"),
    Ablation("RL", r"$p_s$ only",
             "RL state: p_s derivatives only", Decimal("0.6651"), "(state)"),
    Ablation("Backbone", "Fixed-order training",
             "No random-order training", Decimal("0.6406")),
    Ablation("Backbone", "Bottleneck suffix",
             "Bottleneck position", Decimal("0.6463")),
    Ablation("Backbone", "Final-prefix CE only",
             "Only CE loss", Decimal("0.6491")),
)


class InlineCategoryLabel(AnnotationBbox):
    """Anchor mixed-size text to one baseline across raster/vector backends."""

    def update_positions(self, renderer):
        box = self.offsetbox.get_bbox(renderer)
        self.xybox = (0, LABEL_BASELINE_PAD_PT + box.y0 / renderer.points_to_pixels(1))
        super().update_positions(renderer)


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
    fig.subplots_adjust(left=2.55 / width, right=1 - 0.35 / width,
                        bottom=0.2 / height, top=1 - 1.1 / height)
    # A horizontal data unit is one physical inch, so widening the plot does
    # not accidentally widen the bars. Labels retain their native point sizes.
    axes_width = ax.get_position().width * width
    ax.set_xlim(0, axes_width)
    renderer = fig.canvas.get_renderer()
    label_boxes = []
    for row in ABLATIONS:
        parts = [TextArea(row.label, textprops={"fontsize": font_size + 8})]
        if row.qualifier:
            parts.append(TextArea(row.qualifier, textprops={
                "fontsize": font_size + 8 - PARENTHETICAL_REDUCTION_PT}))
        packed = HPacker(children=parts, align="baseline", pad=0, sep=7)
        packed.set_figure(fig)
        label_boxes.append(packed)
    label_widths = [box.get_bbox(renderer).width / fig.dpi for box in label_boxes]
    gap = (axes_width - sum(label_widths) - 2 * CATEGORY_EDGE_PAD_INCHES) / 6
    if gap < MIN_CATEGORY_GAP_INCHES:
        plt.close(fig)
        raise ValueError("The one-row labels need more width; increase --width-inches")
    x = []
    cursor = CATEGORY_EDGE_PAD_INCHES
    for label_width in label_widths:
        x.append(cursor + label_width / 2)
        cursor += label_width + gap
    group_divider = x[3] + label_widths[3] / 2 + gap / 2
    values = [float(row.relative_change_pct) for row in ABLATIONS]
    bars = ax.bar(x, values, width=BAR_WIDTH_INCHES,
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

    ax.set_ylim(-9.25, 0)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(x))
    ax.set_yticks([-8, -6, -4, -2, 0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Relative Macro-F1\nchange (%)", labelpad=6,
                  linespacing=AXIS_TITLE_LINE_SPACING)
    ax.grid(axis="y", color="#D0D0D0", linestyle="--", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0, top=False, labeltop=False,
                   bottom=False, labelbottom=False)
    ax.tick_params(axis="y", direction="out", length=3, width=0.65, pad=3)
    for xi, packed in zip(x, label_boxes):
        label = InlineCategoryLabel(
            packed, (xi, 1), xycoords=ax.get_xaxis_transform(),
            xybox=(0, LABEL_BASELINE_PAD_PT), boxcoords="offset points",
            box_alignment=(0.5, 0), frameon=False, pad=0,
            annotation_clip=False)
        ax.add_artist(label)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    # Both sections use the same metric color; names and the divider identify them.
    ax.axvline(group_divider, color="black", linestyle=":", linewidth=1.0, zorder=2)
    for right_edge, group in ((group_divider - 0.4, "ARC"), (axes_width - 0.4, "SeMA")):
        ax.text(right_edge, 0.07, group, transform=ax.get_xaxis_transform(),
                ha="right", va="bottom", fontsize=font_size - 0.5,
                fontfamily=GROUP_FONT, color="black")
    return fig, ax, bars


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
    centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    if (len(ax.lines) != 1 or ax.lines[0].get_linestyle() != ":"
            or not centers[3] < ax.lines[0].get_xdata()[0] < centers[4]):
        raise ValueError("Expected one dotted divider between the ablation groups")
    if any(tick.label1.get_visible() or tick.label2.get_visible()
           for tick in ax.xaxis.get_major_ticks()):
        raise ValueError("Native category labels must be hidden behind the mixed-size labels")
    category_labels = [artist for artist in ax.artists if isinstance(artist, InlineCategoryLabel)]
    if len(category_labels) != 7:
        raise ValueError("Expected exactly seven inline category labels")
    for label, row in zip(category_labels, ABLATIONS):
        text_areas = label.offsetbox.get_children()
        expected = [row.label] + ([row.qualifier] if row.qualifier else [])
        if [area.get_text() for area in text_areas] != expected:
            raise ValueError("Category labels must preserve the requested text and order")
        for index, area in enumerate(text_areas):
            text = area.get_children()[0]
            expected_size = font_size + 8 - (PARENTHETICAL_REDUCTION_PT if index else 0)
            if text.get_fontsize() != expected_size or text.get_rotation() != 0 or "\n" in text.get_text():
                raise ValueError("Category text must be horizontal, one-line, with smaller parentheticals")
    if ax.collections:
        raise ValueError("The compact layout must not retain staggered-label guides")
    for label in ax.get_yticklabels():
        if label.get_fontsize() != font_size + 8:
            raise ValueError("Tick labels have an unexpected font size")
    if ax.yaxis.label.get_fontsize() != font_size + 11.5:
        raise ValueError("Axis label has an unexpected font size")
    if any(label.get_fontsize() != font_size - 0.5 for label in ax.texts):
        raise ValueError("Value/group labels have an unexpected font size")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    baseline = ax.bbox.y1 + renderer.points_to_pixels(LABEL_BASELINE_PAD_PT)
    for label, center, bar in zip(category_labels, centers, bars):
        box = label.get_window_extent(renderer)
        raw_box = label.offsetbox.get_bbox(renderer)
        if not np.isclose((box.x0 + box.x1) / 2, ax.transData.transform((center, 0))[0], atol=0.5):
            raise ValueError("Each complete category label must be centered over its bar")
        if not np.isclose(box.y0 - raw_box.y0, baseline, atol=0.5):
            raise ValueError("All category labels must share one text baseline")
        physical_width = bar.get_window_extent(renderer).width / fig.dpi
        if not np.isclose(physical_width, BAR_WIDTH_INCHES):
            raise ValueError("Bar widths must stay fixed in physical inches")
    # Enlarged endpoint labels must stay inside the plot, not erase its border.
    for label in ax.texts[:7]:
        box = label.get_window_extent(renderer)
        padding = 0.3 * fig.dpi / 72
        if box.y0 - padding < ax.bbox.y0 or box.y1 + padding > ax.bbox.y1:
            raise ValueError(f"Endpoint label crosses the plot border: {label.get_text()!r}")
    texts = [text for text in fig.findobj(Text) if text.get_visible() and text.get_text()]
    boxes = [text.get_window_extent(renderer) for text in texts]
    for text, box in zip(texts, boxes):
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
            raise ValueError(f"Clipped label {text.get_text()!r}; increase figure size")
    for index, left in enumerate(boxes):
        for other_index in range(index + 1, len(boxes)):
            if left.overlaps(boxes[other_index]):
                raise ValueError(f"Overlapping labels: {texts[index].get_text()!r}, "
                                 f"{texts[other_index].get_text()!r}")


def tight_export_bbox(fig):
    """Remove blank margins while including complete glyphs and strokes.

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
