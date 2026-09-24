#!/usr/bin/env python3
"""Plot equal-fold class-mean Shapley contributions with between-fold SDs.

Uses the exact same verified IEMOCAP test-split source as the heatmap. Each
dot is the supplied class mean and each horizontal whisker spans mean +/- SD.
The 2026-09-23 raw-data audit identified these as population SDs (ddof=0) of
three fold-specific class means, NOT individual-sample SDs, confidence intervals,
or standard errors. No synthetic samples are generated.
All four class panels share a scale wide enough to include every full interval.

Run: python plotting/plot_shapley_dot_sd.py
Updates shapley_iemocap_test_dot_sd.pdf, .svg, and 300-dpi .png in plotting/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

from plot_shapley_heatmap import (
    HERE, POSITIVE, SOURCE_RANGE, SOURCE_SHEET, SOURCE_URL, read_source,
)


STEM = "shapley_iemocap_test_dot_sd"
FIGSIZE = (8.0, 3.1)
FONT_SIZE = 17
XLIM = (-0.30, 0.45)
MEAN_MARKER_SIZE = 7.2
SD_LINEWIDTH = 2.8
SD_CAPWIDTH = 2.0
SD_CAPSIZE = 4.5
PLOT_COLOR = POSITIVE
# Display aliases retain the source order: hand, head, rotated.
MODALITIES = ("Video", "Audio", "Text", "MoCap1", "MoCap2", "MoCap3")
CAPTION = (
    "Modality Shapley contributions on the IEMOCAP test split. Dots indicate "
    "the equally weighted mean of three fold-specific class means; horizontal "
    "whiskers show plus/minus one population standard deviation of those fold "
    "means, not individual-sample variation or confidence intervals. "
    "MoCap1, MoCap2, and MoCap3 denote hand, head, and rotated, respectively."
)


def build_figure(rows):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "text.color": "black", "axes.labelcolor": "black",
        "axes.edgecolor": "black", "axes.linewidth": 1.1,
        "xtick.color": "black", "ytick.color": "black",
        "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": STEM,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 4, figsize=FIGSIZE, sharex=True, sharey=True)
    containers = []
    for ax, row in zip(axes, rows):
        panel_containers = []
        for y, (mean, sd) in enumerate(zip(row.means, row.sds)):
            color = PLOT_COLOR
            container = ax.errorbar(float(mean), y, xerr=float(sd), fmt="o",
                                   markersize=MEAN_MARKER_SIZE, color=color, ecolor=color,
                                   elinewidth=SD_LINEWIDTH, capsize=SD_CAPSIZE,
                                   markeredgecolor="white", markeredgewidth=0.7,
                                   zorder=3)
            # Set caps independently: markeredgewidth otherwise overrides capthick.
            for cap in container.lines[1]:
                cap.set_markeredgewidth(SD_CAPWIDTH)
            panel_containers.append(container)
        containers.append(panel_containers)
        ax.set_xlim(*XLIM)
        ax.set_ylim(5.5, -0.5)
        ax.set_yticks(range(6), MODALITIES)
        ax.set_xticks((-0.25, 0, 0.25), ("−0.25", "0", "0.25"))
        ax.tick_params(axis="x", length=3, width=1.1, pad=3)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.set_title(row.name.title(), pad=6, fontweight="normal")
        ax.axvline(0, color="#777777", linewidth=1.25, linestyle=":", zorder=1)
        ax.grid(axis="y", color="#E4E4E4", linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
    fig.text(4.675 / FIGSIZE[0], 2.943 / FIGSIZE[1], "Classes →",
             ha="center", va="center", fontsize=FONT_SIZE)
    # Rotating a left arrow by 90 degrees makes it point down the modality rows.
    fig.text(0.163 / FIGSIZE[0], 1.575 / FIGSIZE[1], "← Modalities",
             ha="center", va="center", rotation=90, fontsize=FONT_SIZE)
    fig.supxlabel("Mean Shapley contribution",
                  x=4.675 / FIGSIZE[0], y=0.035 / FIGSIZE[1], fontsize=FONT_SIZE)
    fig.subplots_adjust(left=1.40 / FIGSIZE[0], right=1 - 0.05 / FIGSIZE[0],
                        bottom=0.65 / FIGSIZE[1], top=2.50 / FIGSIZE[1],
                        wspace=0.125)
    return fig, axes, containers


def validate_figure(fig, axes, containers, rows):
    assert len(axes) == 4 and tuple(fig.get_size_inches()) == FIGSIZE
    assert [label.get_text() for label in axes[0].get_yticklabels()] == list(MODALITIES)
    assert {"Classes →", "← Modalities"}.issubset(text.get_text() for text in fig.texts)
    assert fig._supxlabel.get_text() == "Mean Shapley contribution"
    for ax, panel_containers, row in zip(axes, containers, rows):
        assert tuple(ax.get_xlim()) == XLIM
        assert len(panel_containers) == 6
        assert ax.get_title() == row.name.title()
        for y, (container, mean, sd) in enumerate(zip(panel_containers, row.means, row.sds)):
            point, caps, intervals = container.lines
            np.testing.assert_allclose(np.asarray(point.get_xdata(), dtype=float),
                                       [float(mean)], rtol=0, atol=1e-15)
            np.testing.assert_array_equal(point.get_ydata(), [y])
            assert point.get_color() == PLOT_COLOR
            np.testing.assert_allclose(intervals[0].get_colors(),
                                       [matplotlib.colors.to_rgba(PLOT_COLOR)])
            assert all(cap.get_color() == PLOT_COLOR for cap in caps)
            lower, upper = float(mean - sd), float(mean + sd)
            np.testing.assert_allclose(intervals[0].get_segments()[0],
                                       ((lower, y), (upper, y)), rtol=0, atol=1e-15)
            assert XLIM[0] < lower <= upper < XLIM[1]
            assert len(caps) == 2
            assert point.get_markersize() == MEAN_MARKER_SIZE
            np.testing.assert_allclose(intervals[0].get_linewidths(), [SD_LINEWIDTH])
            assert all(cap.get_markeredgewidth() == SD_CAPWIDTH for cap in caps)
            assert all(cap.get_markersize() == 2 * SD_CAPSIZE for cap in caps)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [text for text in fig.findobj(Text) if text.get_visible() and text.get_text()]
    assert all(text.get_fontsize() == FONT_SIZE for text in texts)
    boxes = [text.get_window_extent(renderer) for text in texts]
    for text, box in zip(texts, boxes):
        if (box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width
                or box.y1 > fig.bbox.height):
            raise ValueError(f"Clipped label: {text.get_text()!r}")
    for i, left in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            if left.overlaps(boxes[j]):
                raise ValueError(f"Overlapping labels: {texts[i].get_text()!r}, "
                                 f"{texts[j].get_text()!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "data/shapley_iemocap_test.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    rows = read_source(args.input)
    fig, axes, containers = build_figure(rows)
    try:
        for dpi in (100, 300):
            fig.set_dpi(dpi)
            validate_figure(fig, axes, containers, rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "svg", "png"):
            metadata = {"Title": "IEMOCAP test-split Shapley equal-fold means and fold SDs"}
            metadata["Subject" if extension == "pdf" else "Description"] = (
                f'{SOURCE_URL}; "{SOURCE_SHEET}"!{SOURCE_RANGE}. {CAPTION}'
            )
            path = args.output_dir / f"{STEM}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0, metadata=metadata)
            print(f"Created {path.resolve()}")
    finally:
        plt.close(fig)
    print("Verified all 24 equal-fold means and all 24 full mean ± fold SD intervals.")


if __name__ == "__main__":
    main()
