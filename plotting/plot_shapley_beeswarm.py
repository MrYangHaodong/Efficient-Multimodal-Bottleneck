#!/usr/bin/env python3
"""Class-wise sample Shapley distributions, with optional mean +/- sample SD.

Each small dot is one supplied random/avg/test attribution. X values are never
changed or sampled. Deterministic vertical stacking within 0.025-wide x bins is
only a display offset, scaled separately for each class/modality row. Swarm
widths therefore show relative density within a row, not sample count across
rows. This is a binned beeswarm, not a collision-free swarm or a KDE.

Run: python plotting/plot_shapley_beeswarm.py
For a signed-color beeswarm without summary markers:
  python plotting/plot_shapley_beeswarm.py --style gradient
To import the full supplied CSV once:
  python plotting/plot_shapley_beeswarm.py --input /path/to/all.csv --save-filtered

The earlier summary CSV is preserved. Its values are audited separately because
its equal-fold means and between-fold SDs differ from pooled sample statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.text import Text


HERE = Path(__file__).resolve().parent
STEM = "shapley_iemocap_test_beeswarm"
SAMPLE_PATH = HERE / "data/shapley_iemocap_random_avg_test_samples.csv"
SUMMARY_PATH = HERE / "data/shapley_iemocap_test.csv"
CLASS_ORDER = ("neutral", "angry", "happy", "sad")
MODALITIES = ("video", "audio", "text", "mocap_hand", "mocap_head", "mocap_rotated")
DISPLAY_LABELS = ("Video", "Audio", "Text", "MoCap1", "MoCap2", "MoCap3")
PHI_COLUMNS = tuple(f"phi_{name}" for name in MODALITIES)
HEADERS = ("backbone", "vS", "split", "fold", "idx", "y", "class", *PHI_COLUMNS)
FILTER = {"backbone": "random", "vS": "avg", "split": "test"}
EXPECTED_COUNTS = {"neutral": 958, "angry": 724, "happy": 334, "sad": 686}
EXPECTED_FOLDS = {0: 909, 1: 793, 2: 1000}
FIGSIZE = (8.0, 3.6)
FONT_SIZE = 17
GRADIENT_FIGSIZE = (8.0, 4.4)
GRADIENT_FONT_SIZE = 24
COLOR = "#AA4499"
# Paul Tol's colour-blind-friendly diverging sunset scheme, Fig. 13.
# The author explicitly permits linear interpolation of these sRGB anchors.
GRADIENT_PALETTE = "Paul Tol sunset"
GRADIENT_PALETTE_SOURCE = "https://sronpersonalpages.nl/~pault/"
GRADIENT_COLORS = (
    "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF", "#EAECCC",
    "#FEDA8B", "#FDB366", "#F67E4B", "#DD3D2D", "#A50026",
)
GRADIENT_EDGE_COLOR = "none"
GRADIENT_EDGE_WIDTH = 0
GRADIENT_SAMPLE_SIZE = 9.0  # points squared; larger borderless gradient markers
XLIM = (-1.08, 1.08)
SAMPLE_SIZE = 3.8  # points squared
SAMPLE_ALPHA = 0.38
MEAN_SIZE = 7.2  # points
SD_WIDTH = 2.8
CAPTION = (
    "Sample-wise modality Shapley contributions for the random-order backbone "
    "(vS=avg) on the IEMOCAP test split. Small dots represent individual "
    "attributions; large dots and capped whiskers show the sample-weighted class "
    "mean and plus/minus one sample standard deviation (ddof=1), pooled across "
    "the three test folds. SD is descriptive spread, not a confidence interval. "
    "Binned vertical stacks are scaled separately per row, so their heights are "
    "not comparable sample counts; all x values are unaltered. "
    "MoCap1, MoCap2, and MoCap3 denote hand, head, and rotated, respectively. "
    "No feature-value color encoding is used."
)
GRADIENT_CAPTION = (
    "Sample-wise modality Shapley contributions for the random-order backbone "
    "(vS=avg) on the IEMOCAP test split, pooled across three test folds. Each dot "
    "is one supplied attribution. Both horizontal position and color encode "
    "signed Shapley contribution, using Paul Tol's colorblind-friendly sunset "
    "diverging palette: blue for negative values, pale cream at zero, and "
    "orange/red for positive values, with a shared scale from -1 to +1. "
    "Position and the zero-reference line provide a color-independent sign cue. "
    "Color does not encode input feature values or sample "
    "density. No mean markers or SD whiskers are shown. Binned vertical stacks "
    "are scaled separately per row, so their heights are not comparable sample "
    "counts; all x values are unaltered. MoCap1, MoCap2, and MoCap3 denote hand, "
    "head, and rotated, respectively."
)


def read_samples(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HEADERS:
            raise ValueError(f"Unexpected source columns: {reader.fieldnames}")
        all_rows = list(reader)
    seen = set()
    selected = []
    groups = Counter()
    for row in all_rows:
        if set(row) != set(HEADERS) or any(v is None or not v.strip() for v in row.values()):
            raise ValueError("Missing or malformed fields")
        fold, idx, label = (int(row[k]) for k in ("fold", "idx", "y"))
        if fold < 0 or idx < 0 or not 0 <= label < len(CLASS_ORDER):
            raise ValueError("Invalid fold, index, or class label")
        if CLASS_ORDER[label] != row["class"]:
            raise ValueError("Class label/name mismatch")
        key = tuple(row[k] for k in ("backbone", "vS", "split", "fold", "idx"))
        if key in seen:
            raise ValueError(f"Duplicate sample key: {key}")
        seen.add(key)
        phi = np.asarray([float(row[k]) for k in PHI_COLUMNS])
        if not np.isfinite(phi).all():
            raise ValueError(f"Nonfinite score: {key}")
        groups[tuple(row[k] for k in ("backbone", "vS", "split", "fold"))] += 1
        if all(row[k] == value for k, value in FILTER.items()):
            selected.append(row)
    if Counter(r["class"] for r in selected) != EXPECTED_COUNTS:
        raise ValueError("Unexpected test class counts")
    if Counter(int(r["fold"]) for r in selected) != EXPECTED_FOLDS:
        raise ValueError("Unexpected test fold counts")
    for fold, count in EXPECTED_FOLDS.items():
        if sorted(int(r["idx"]) for r in selected if int(r["fold"]) == fold) != list(range(count)):
            raise ValueError(f"Noncontiguous sample indices in fold {fold}")
    # Sentinels from the user's pasted sample table, spanning all test folds.
    sentinels = {
        (0, 0): (-.26819, -.66528, .19014, .14083, -.19042, -.20708),
        (0, 908): (.03333, .03333, .03333, .03333, .03333, -.16667),
        (1, 0): (.01528, .44444, .44444, -.03194, .05139, .07639),
        (1, 792): (.01847, -.29458, .54472, .02736, .32597, -.45528),
        (2, 0): (-.08639, .00694, -.00139, .42361, -.41222, .06944),
        (2, 999): (.08333, .41667, .41667, .0, .0, .08333),
    }
    lookup = {(int(r["fold"]), int(r["idx"])): r for r in selected}
    for key, expected in sentinels.items():
        np.testing.assert_array_equal([float(lookup[key][k]) for k in PHI_COLUMNS], expected)
    arrays = {
        name: np.asarray([[float(r[k]) for k in PHI_COLUMNS]
                          for r in selected if r["class"] == name])
        for name in CLASS_ORDER
    }
    return selected, arrays, {
        "input_file": str(path.resolve()),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_rows": len(all_rows),
        "input_groups": {"/".join(k): v for k, v in sorted(groups.items())},
        "filter": FILTER,
        "selected_rows": len(selected),
        "class_counts": EXPECTED_COUNTS,
        "fold_counts": EXPECTED_FOLDS,
        "unique_composite_keys": True,
        "finite_complete_scores": True,
        "class_labels_verified": True,
        "pasted_sentinels_verified": len(sentinels),
        "all_zero_rows_retained": sum(all(float(r[k]) == 0 for k in PHI_COLUMNS)
                                      for r in selected),
    }


def audit_statistics(rows, arrays):
    with SUMMARY_PATH.open(encoding="utf-8-sig", newline="") as stream:
        previous = {r["class"]: r for r in csv.DictReader(stream)}
    statistics = []
    for name in CLASS_ORDER:
        values = arrays[name]
        fold_means = np.asarray([
            np.mean([[float(r[k]) for k in PHI_COLUMNS]
                     for r in rows if r["class"] == name and int(r["fold"]) == fold], axis=0)
            for fold in EXPECTED_FOLDS
        ])
        for j, modality in enumerate(MODALITIES):
            old_mean, old_sd = (float(x.strip()) for x in previous[name][modality].split("|"))
            statistics.append({
                "class": name, "modality": modality, "n": len(values),
                "pooled_mean": float(values[:, j].mean()),
                "pooled_sample_sd_ddof1": float(values[:, j].std(ddof=1)),
                "minimum": float(values[:, j].min()),
                "maximum": float(values[:, j].max()),
                "fold_means": fold_means[:, j].tolist(),
                "equal_fold_mean": float(fold_means[:, j].mean()),
                "sd_of_fold_means_ddof0": float(fold_means[:, j].std(ddof=0)),
                "previous_mean": old_mean, "previous_sd": old_sd,
                "previous_mean_matches_equal_fold_mean_4dp":
                    f"{old_mean:.4f}" == f"{fold_means[:, j].mean():.4f}",
                "previous_sd_matches_fold_sd_4dp":
                    f"{old_sd:.4f}" == f"{fold_means[:, j].std(ddof=0):.4f}",
            })
    return statistics


def swarm_offsets(values, seed):
    """Visual-only, deterministic, locally binned stacking; retains exact x."""
    bins = np.floor(values / 0.025 + 0.5).astype(int)
    offsets = np.zeros(len(values))
    rng = np.random.default_rng(seed)
    for bin_number in np.unique(bins):
        indices = np.flatnonzero(bins == bin_number)
        rng.shuffle(indices)
        rank = np.arange(len(indices))
        stack = np.where(rank % 2, (rank + 1) // 2, -(rank // 2)).astype(float)
        offsets[indices] = stack - stack.mean()
    max_offset = np.abs(offsets).max()
    if max_offset:
        offsets *= .29 / max_offset
    return offsets


def build_figure(arrays, style="summary"):
    if style not in ("summary", "gradient"):
        raise ValueError(f"Unknown style: {style}")
    gradient = style == "gradient"
    font_size = GRADIENT_FONT_SIZE if gradient else FONT_SIZE
    figsize = GRADIENT_FIGSIZE if gradient else FIGSIZE
    sample_size = GRADIENT_SAMPLE_SIZE if gradient else SAMPLE_SIZE
    cmap = LinearSegmentedColormap.from_list("tol_sunset", GRADIENT_COLORS, N=511)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": font_size,
        "xtick.labelsize": font_size, "ytick.labelsize": font_size,
        "axes.labelsize": font_size, "axes.titlesize": font_size,
        "axes.linewidth": 1.1, "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": STEM, "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)
    marks = []
    for class_index, (ax, name) in enumerate(zip(axes, CLASS_ORDER)):
        values = arrays[name]
        panel_marks = []
        for j in range(6):
            x = values[:, j]
            offsets = swarm_offsets(x, 23092026 + 10 * class_index + j)
            color_args = ({"c": x, "cmap": cmap, "norm": norm, "alpha": 1.0}
                          if gradient else {"color": COLOR, "alpha": SAMPLE_ALPHA})
            edge_args = ({"edgecolors": GRADIENT_EDGE_COLOR,
                          "linewidths": GRADIENT_EDGE_WIDTH}
                         if gradient else {"linewidths": 0})
            dots = ax.scatter(x, j + offsets, s=sample_size, **color_args, **edge_args,
                              rasterized=False, zorder=2)
            summary = None
            if not gradient:
                mean, sd = x.mean(), x.std(ddof=1)
                summary = ax.errorbar(mean, j, xerr=sd, fmt="o", markersize=MEAN_SIZE,
                                     color=COLOR, ecolor=COLOR, elinewidth=SD_WIDTH,
                                     capsize=3.5, markeredgecolor="white", markeredgewidth=1,
                                     zorder=4)
                for cap in summary.lines[1]:
                    cap.set_markeredgewidth(1.8)
                # White halo separates the summary from same-hue individual samples.
                summary.lines[2][0].set_path_effects([
                    pe.Stroke(linewidth=SD_WIDTH + 1.5, foreground="white"), pe.Normal()
                ])
            panel_marks.append((dots, summary))
        marks.append(panel_marks)
        ax.set_xlim(*XLIM)
        ax.set_ylim(5.5, -.5)
        ax.set_yticks(range(6), DISPLAY_LABELS)
        ax.set_xticks((-1, 0, 1), ("−1", "0", "1"))
        ax.tick_params(axis="x", length=3, width=1.1, pad=3)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.set_title(name.title(), pad=6, fontweight="normal")
        ax.axvline(0, color="#777777", linewidth=1.2, linestyle=":", zorder=1)
        ax.grid(axis="y", color="#E4E4E4", linewidth=.55)
        ax.set_axisbelow(True)
    if gradient:
        # Extra height and spacing accommodate larger text without scaling it down.
        fig.subplots_adjust(left=1.90 / 8, right=1 - .12 / 8,
                            bottom=1.50 / 4.4, top=3.55 / 4.4, wspace=.30)
        fig.text(4.89 / 8, 4.18 / 4.4, "Classes →", ha="center", va="center")
        fig.text(.22 / 8, 2.525 / 4.4, "← Modalities",
                 ha="center", va="center", rotation=90)
        # One shared label describes the identical x-position and color encoding.
        color_ax = fig.add_axes([3.615 / 8, .46 / 4.4, 2.55 / 8, .10 / 4.4])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax,
                               orientation="horizontal", ticks=(-1, 0, 1))
        colorbar.set_ticklabels(("−1", "0", "+1"))
        colorbar.ax.tick_params(length=2.5, width=.8, pad=3, labelsize=font_size)
        colorbar.ax.set_title("Shapley contribution", pad=9, fontsize=font_size)
        colorbar.outline.set_linewidth(.6)
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")  # Avoid vector-renderer segment seams.
    else:
        fig.subplots_adjust(left=1.40 / 8, right=1 - .05 / 8,
                            bottom=.93 / 3.6, top=3.04 / 3.6, wspace=.125)
        fig.text(4.675 / 8, 3.45 / 3.6, "Classes →", ha="center", va="center")
        fig.text(.163 / 8, ((.93 + 3.04) / 2) / 3.6, "← Modalities",
                 ha="center", va="center", rotation=90)
        fig.supxlabel("Shapley contribution", x=4.675 / 8, y=.35 / 3.6, fontsize=FONT_SIZE)
        sample_handle = Line2D([], [], color=COLOR, marker="o", linestyle="none",
                               markersize=3.8, alpha=SAMPLE_ALPHA)
        fig.legend([sample_handle, marks[0][0][1]], ["Samples", "Mean ± SD"],
                   loc="lower center", bbox_to_anchor=(4.675 / 8, .005 / 3.6),
                   frameon=False, ncol=2, handlelength=1.6, handletextpad=.35,
                   columnspacing=1.1, borderpad=0, borderaxespad=0, fontsize=FONT_SIZE)
    return fig, axes, marks


def validate_figure(fig, axes, marks, arrays, style="summary"):
    assert tuple(fig.get_size_inches()) == (GRADIENT_FIGSIZE if style == "gradient" else FIGSIZE)
    total_points = 0
    for ax, name, panel_marks in zip(axes, CLASS_ORDER, marks):
        assert tuple(ax.get_xlim()) == XLIM and ax.get_title() == name.title()
        values = arrays[name]
        for j, (dots, summary) in enumerate(panel_marks):
            plotted = np.asarray(dots.get_offsets())
            np.testing.assert_array_equal(plotted[:, 0], values[:, j])
            assert np.abs(plotted[:, 1] - j).max() <= .2900000001
            assert XLIM[0] < plotted[:, 0].min() <= plotted[:, 0].max() < XLIM[1]
            assert len(plotted) == EXPECTED_COUNTS[name]
            total_points += len(plotted)
            if style == "gradient":
                assert summary is None and not ax.containers
                assert len(ax.lines) == 1  # Only the zero-reference line remains.
                assert dots.get_alpha() == 1.0
                np.testing.assert_array_equal(dots.get_sizes(), [GRADIENT_SAMPLE_SIZE])
                np.testing.assert_array_equal(dots.get_linewidths(), [GRADIENT_EDGE_WIDTH])
                assert dots.get_edgecolors().size == 0  # No marker outline.
                np.testing.assert_array_equal(dots.get_array(), values[:, j])
                assert (dots.norm.vmin, dots.norm.vcenter, dots.norm.vmax) == (-1, 0, 1)
                np.testing.assert_allclose(dots.cmap(dots.norm([-1., 0., 1.])),
                                           [to_rgba(GRADIENT_COLORS[k]) for k in (0, 5, 10)],
                                           atol=1e-14)
                np.testing.assert_allclose(dots.cmap(np.linspace(0, 1, 11)),
                                           [to_rgba(c) for c in GRADIENT_COLORS], atol=1e-14)
                continue
            mean, sd = values[:, j].mean(), values[:, j].std(ddof=1)
            point, caps, bars = summary.lines
            np.testing.assert_allclose(np.asarray(point.get_xdata(), dtype=float),
                                       [mean], rtol=0, atol=1e-14)
            np.testing.assert_allclose(bars[0].get_segments()[0],
                                       [[mean - sd, j], [mean + sd, j]], rtol=0, atol=1e-14)
            assert XLIM[0] < mean - sd <= mean + sd < XLIM[1]
            assert point.get_color() == COLOR and point.get_markersize() == MEAN_SIZE
            assert all(cap.get_color() == COLOR for cap in caps)
    assert total_points == 2702 * 6
    fig.canvas.draw()
    if style == "gradient":
        assert len(fig.axes) == 5 and not fig.legends
        for panel_marks in marks:
            for dots, _ in panel_marks:
                np.testing.assert_allclose(dots.get_facecolors(),
                                           dots.cmap(dots.norm(dots.get_array())), atol=1e-14)
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    font_size = GRADIENT_FONT_SIZE if style == "gradient" else FONT_SIZE
    assert all(t.get_fontsize() == font_size for t in texts)
    boxes = [t.get_window_extent(renderer) for t in texts]
    for text, box in zip(texts, boxes):
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            raise ValueError(f"Clipped text: {text.get_text()}")
    for i, box in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            if box.overlaps(boxes[j]):
                raise ValueError(f"Overlapping text: {texts[i].get_text()} / {texts[j].get_text()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--save-filtered", action="store_true")
    parser.add_argument("--style", choices=("summary", "gradient"), default="summary")
    args = parser.parse_args()
    stem = f"{STEM}_gradient" if args.style == "gradient" else STEM
    caption = GRADIENT_CAPTION if args.style == "gradient" else CAPTION
    rows, arrays, audit = read_samples(args.input)
    audit["statistics"] = audit_statistics(rows, arrays)
    audit["caption"] = caption
    audit["style"] = args.style
    if args.style == "gradient":
        audit["appearance"] = {"font_size_pt": GRADIENT_FONT_SIZE,
                               "figure_size_inches": list(GRADIENT_FIGSIZE),
                               "marker_size_pt_squared": GRADIENT_SAMPLE_SIZE,
                               "marker_edge_color": GRADIENT_EDGE_COLOR,
                               "marker_edge_width_pt": GRADIENT_EDGE_WIDTH}
        audit["color_encoding"] = {"quantity": "signed Shapley contribution",
                                   "palette": GRADIENT_PALETTE,
                                   "palette_source": GRADIENT_PALETTE_SOURCE,
                                   "limits": [-1, 1], "center": 0,
                                   "colors": list(GRADIENT_COLORS), "alpha": 1.0,
                                   "shared_across_all_panels": True,
                                   "color_independent_sign_cue": "x position and zero line",
                                   "mean_sd_overlay": False}
    if args.save_filtered:
        if args.input.resolve() == SAMPLE_PATH.resolve():
            raise ValueError("Do not rewrite the input in place")
        with SAMPLE_PATH.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HEADERS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        reread, _, _ = read_samples(SAMPLE_PATH)
        assert reread == rows
    audit["filtered_data_sha256"] = hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest()
    fig, axes, marks = build_figure(arrays, args.style)
    try:
        for dpi in (100, 300):
            fig.set_dpi(dpi)
            validate_figure(fig, axes, marks, arrays, args.style)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "svg", "png"):
            metadata = {"Title": "IEMOCAP sample-wise Shapley contributions"}
            metadata["Subject" if extension == "pdf" else "Description"] = caption
            path = args.output_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0, metadata=metadata)
            print(f"Created {path}")
    finally:
        plt.close(fig)
    audit_filename = ("shapley_iemocap_gradient_audit.json" if args.style == "gradient"
                      else "shapley_iemocap_sample_audit.json")
    audit_path = args.output_dir / "data" / audit_filename
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Verified 16,212 unaltered sample scores and",
          "shared signed-contribution colors without summaries." if args.style == "gradient"
          else "24 pooled means/sample SDs.")
    print("Previous summary matches equal-fold mean / fold-population SD:",
          sum(s["previous_mean_matches_equal_fold_mean_4dp"] for s in audit["statistics"]),
          sum(s["previous_sd_matches_fold_sd_4dp"] for s in audit["statistics"]), "of 24 each")


if __name__ == "__main__":
    main()
