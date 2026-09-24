#!/usr/bin/env python3
"""Standalone IEMOCAP test-split Shapley heatmap, exported at 7.7 x 2.8 inches.

Source: "Shapley scores"!A4:H8 in the Google Sheet linked below, verified
2026-09-21. The CSV preserves all four-decimal mean | SD pairs verbatim.
The raw-data audit on 2026-09-23 identified equal-fold means and population SDs
of three fold-specific class means, not individual-sample SDs. No normalization, resampling,
synthetic observations, or aggregation is applied. Row 9 is not included.

Cells have no numeric annotations. Colors use unrounded signed means with one
fixed, zero-centered [-0.30, +0.30] scale and the paper's orange/purple palette.
Every font is 4 pt larger: class/modality labels 17 pt, colorbar ticks 13 pt,
and colorbar label 14.5 pt. Titles and explanatory notes are omitted.
All heatmap cells and the colorbar remain vector in PDF/SVG. SVG text remains
editable. Running this script updates the same three output files.

Run: python plotting/plot_shapley_heatmap.py
Optional current-workbook input: --input path/to/workbook.xlsx
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.text import Text


HERE = Path(__file__).resolve().parent
STEM = "shapley_iemocap_test_heatmap"
SOURCE_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1ZsKgZztYMPOj76LL7eYWT07JJ4k0cWdaC1zmOlIy3q0/edit"
)
SOURCE_SHEET = "Shapley scores"
SOURCE_RANGE = "A4:H8"
CLASS_ORDER = ("neutral", "angry", "happy", "sad")
HEADERS = ("class", "n samples", "video", "audio", "text",
           "mocap_hand", "mocap_head", "mocap_rotated")
COLUMN_LABELS = ("Video", "Audio", "Text", "MoCap\nhand",
                 "MoCap\nhead", "MoCap\nrotated")
FIGSIZE = (7.7, 2.8)
LIMIT = Decimal("0.30")
# Same endpoints as the current paper figures (plot_cascade.py and
# plot_selection_by_class.py); the diverging midpoint preserves signed values.
NEGATIVE, ZERO, POSITIVE = "#EE7733", "#FAF9F6", "#AA4499"
AXIS_LABEL_SIZE = 17
COLORBAR_TICK_SIZE = 13
COLORBAR_LABEL_SIZE = 14.5
CAPTION = (
    "Modality Shapley contributions on the IEMOCAP test split. Colour represents "
    "the signed class mean averaged equally over three folds; row labels include "
    "the total number of test samples in each class."
)


@dataclass(frozen=True)
class ClassScores:
    name: str
    n_samples: int
    means: tuple[Decimal, ...]
    sds: tuple[Decimal, ...]


def read_source(path: Path) -> tuple[ClassScores, ...]:
    """Read only the supplied summary cells; never reconstruct samples."""
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            sheet = workbook[SOURCE_SHEET]
            if sheet["A3"].value != "TEST split":
                raise ValueError("The selected range is not labelled TEST split")
            rows = list(sheet.iter_rows(min_row=4, max_row=8, max_col=8,
                                        values_only=True))
        finally:
            workbook.close()
    else:
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file))
    if len(rows) != 5 or tuple(rows[0]) != HEADERS:
        raise ValueError("Expected exactly the A4:H8 header and four class rows")
    parsed = []
    for row, expected_class in zip(rows[1:], CLASS_ORDER):
        if len(row) != 8 or row[0] != expected_class:
            raise ValueError("Unexpected class order or modality columns")
        count = Decimal(str(row[1]))
        if not count.is_finite() or count <= 0 or count != count.to_integral_value():
            raise ValueError(f"Invalid sample count for {expected_class}")
        means, sds = [], []
        for cell in row[2:]:
            parts = str(cell).split("|")
            if len(parts) != 2:
                raise ValueError(f"Expected mean | SD, received {cell!r}")
            mean, sd = (Decimal(part.strip()) for part in parts)
            if not mean.is_finite() or not -LIMIT <= mean <= LIMIT:
                raise ValueError(f"Mean outside the fixed color scale: {mean}")
            if not sd.is_finite() or sd < 0:
                raise ValueError(f"Invalid supplied between-fold SD: {sd}")
            means.append(mean)
            sds.append(sd)
        parsed.append(ClassScores(expected_class, int(count), tuple(means), tuple(sds)))
    return tuple(parsed)


def build_figure(rows):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": AXIS_LABEL_SIZE,
        "xtick.labelsize": AXIS_LABEL_SIZE, "ytick.labelsize": AXIS_LABEL_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE, "text.color": "#111111",
        "axes.labelcolor": "#111111", "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": STEM, "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    width, height = FIGSIZE
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes((1.08 / width, 0.10 / height, 5.22 / width, 2.08 / height))
    cax = fig.add_axes((6.42 / width, 0.15 / height, 0.105 / width, 2.00 / height))
    means = np.asarray([[float(value) for value in row.means] for row in rows])
    cmap = LinearSegmentedColormap.from_list(
        "paper_orange_white_purple", [NEGATIVE, ZERO, POSITIVE], N=257,
    )
    norm = TwoSlopeNorm(vmin=-float(LIMIT), vcenter=0, vmax=float(LIMIT))
    mesh = ax.pcolormesh(np.arange(7), np.arange(5), means, cmap=cmap, norm=norm,
                        shading="flat", edgecolors="white", linewidth=0.55,
                        antialiased=False, rasterized=False)
    ax.set(xlim=(0, 6), ylim=(4, 0))
    ax.set_xticks(np.arange(6) + 0.5, COLUMN_LABELS)
    ax.set_yticks(np.arange(4) + 0.5,
                  [f"{row.name.title()}\n(n={row.n_samples:,})" for row in rows])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3.5)
    ax.tick_params(axis="y", length=0, pad=4)
    for label in ax.get_xticklabels():
        label.set_linespacing(1.05)
    for label in ax.get_yticklabels():
        label.set_linespacing(1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(mesh, cax=cax, ticks=(-0.30, -0.15, 0, 0.15, 0.30))
    colorbar.ax.set_yticklabels(("−0.30", "−0.15", "0", "+0.15", "+0.30"))
    colorbar.ax.tick_params(length=2, width=0.45, pad=2, labelsize=COLORBAR_TICK_SIZE)
    colorbar.set_label("Mean Shapley\ncontribution", fontsize=COLORBAR_LABEL_SIZE, labelpad=2)
    colorbar.ax.yaxis.label.set_linespacing(1.0)
    colorbar.outline.set_linewidth(0.45)
    colorbar.outline.set_edgecolor("#444444")
    colorbar.solids.set_rasterized(False)

    return fig, ax, mesh, colorbar


def validate_figure(fig, ax, mesh, colorbar, rows):
    expected = np.asarray([[float(value) for value in row.means] for row in rows])
    np.testing.assert_array_equal(np.asarray(mesh.get_array()).reshape(4, 6), expected)
    assert (mesh.norm.vmin, mesh.norm.vcenter, mesh.norm.vmax) == (-0.30, 0, 0.30)
    assert colorbar.mappable is mesh and not mesh.get_rasterized()
    assert not colorbar.solids.get_rasterized()
    assert tuple(fig.get_size_inches()) == FIGSIZE
    assert not ax.texts and not fig.texts and not ax.get_title()
    assert all(label.get_fontsize() == AXIS_LABEL_SIZE
               for label in (*ax.get_xticklabels(), *ax.get_yticklabels()))
    assert all(label.get_fontsize() == COLORBAR_TICK_SIZE
               for label in colorbar.ax.get_yticklabels())
    assert colorbar.ax.yaxis.label.get_fontsize() == COLORBAR_LABEL_SIZE
    assert [label.get_text() for label in ax.get_xticklabels()] == list(COLUMN_LABELS)
    assert [label.get_text() for label in ax.get_yticklabels()] == [
        f"{row.name.title()}\n(n={row.n_samples:,})" for row in rows
    ]
    for value, color in ((-0.30, NEGATIVE), (0, ZERO), (0.30, POSITIVE)):
        np.testing.assert_allclose(mesh.cmap(mesh.norm(value)), to_rgba(color))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [text for text in fig.findobj(Text) if text.get_visible() and text.get_text()]
    boxes = [text.get_window_extent(renderer) for text in texts]
    for text, box in zip(texts, boxes):
        if (box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width
                or box.y1 > fig.bbox.height):
            raise ValueError(f"Clipped text: {text.get_text()!r}")
    for index, box in enumerate(boxes):
        for other_index in range(index + 1, len(boxes)):
            if box.overlaps(boxes[other_index]):
                raise ValueError(f"Overlapping text: {texts[index].get_text()!r}, "
                                 f"{texts[other_index].get_text()!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "data/shapley_iemocap_test.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    rows = read_source(args.input)
    fig, ax, mesh, colorbar = build_figure(rows)
    try:
        for dpi in (100, 300):
            fig.set_dpi(dpi)
            validate_figure(fig, ax, mesh, colorbar, rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        provenance = f'{SOURCE_URL}; "{SOURCE_SHEET}"!{SOURCE_RANGE}. {CAPTION}'
        for extension in ("pdf", "svg", "png"):
            metadata = {"Title": "IEMOCAP test-split Shapley contributions"}
            metadata["Subject" if extension == "pdf" else "Description"] = provenance
            path = args.output_dir / f"{STEM}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0, metadata=metadata)
            print(f"Created {path.resolve()}")
    finally:
        plt.close(fig)
    print(f"Verified {len(rows)} classes, {sum(row.n_samples for row in rows):,} samples, 24 mean/SD pairs.")
    print("Signed unnormalized means; SD across samples within class; fixed scale [-0.30, +0.30].")


if __name__ == "__main__":
    main()
