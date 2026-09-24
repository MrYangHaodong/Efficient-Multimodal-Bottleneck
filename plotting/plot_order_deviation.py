#!/usr/bin/env python3
"""Standalone panel (a): order-wise Macro-F1 deviations for IEMOCAP.

Uses the same CSV, fold-specific centering, deterministic jitter, and pooled
population-SD marks as plot_order_variance.py. This is not a TV-distance plot.
The original two-panel figure is left unchanged. Exports PDF, SVG, and a
600-dpi PNG beside this script by default.

Fixed-order training uses a darker neutral grey (#888888), adjusted from
Paul Tol's original bright/vibrant grey (#BBBBBB):
https://sronpersonalpages.nl/~pault/#sec:qualitative
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

import iclr_style
from plot_order_variance import MARKERS, MODELS, load


HERE = Path(__file__).resolve().parent
COLORS = {MODELS[0]: "#AA4499", MODELS[1]: "#888888"}
EDGE_COLORS = {MODELS[0]: "#6B285F", MODELS[1]: "#444444"}
# Saturated purple stays visibly purple after downsampling; both shades remain
# darker than their marker borders without becoming indistinguishably black.
SD_COLORS = {MODELS[0]: "#660066", MODELS[1]: "#333333"}


def centered_groups(data):
    """Return fold-centered deviations in F1 points, grouped by model/size."""
    sizes = sorted(int(k) for k in data[MODELS[0]])
    groups = {}
    for model in MODELS:
        if sorted(int(k) for k in data[model]) != sizes:
            raise ValueError("Training schemes must contain the same subset sizes.")
        for size in sizes:
            centered = []
            for scores in data[model][str(size)].values():
                values = np.asarray(scores, dtype=float)
                if not np.isfinite(values).all() or values.size < 2:
                    raise ValueError("Each fold needs at least two finite order scores.")
                centered.append(values - values.mean())
            groups[model, size] = np.concatenate(centered) * 100
    return sizes, groups


def build_figure(data, width=6.8, height=4.7):
    sizes, groups = centered_groups(data)
    iclr_style.apply()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 24,
        "axes.labelsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 21.5,
        "axes.edgecolor": "black",
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=.175, right=.988, bottom=.17, top=.865)
    rng = np.random.default_rng(0)

    for model_index, model in enumerate(MODELS):
        for size in sizes:
            deviations = groups[model, size]
            x = size + (model_index - .5) * .34
            ax.scatter(
                x + rng.normal(0, .045, deviations.size), deviations,
                s=23, marker=MARKERS[model],
                facecolors=to_rgba(COLORS[model], .72 if model_index == 0 else .88),
                edgecolors=EDGE_COLORS[model], linewidths=.6, zorder=3,
            )
            spread = deviations.std(ddof=0)
            for y in (spread, -spread):
                ax.plot([x - .12, x + .12], [y, y],
                        color=SD_COLORS[model], linewidth=3.0, zorder=4)

    ax.axhline(0, color="#888888", linewidth=.8, linestyle=":", zorder=2)
    ax.set_xticks(sizes)
    ax.set_yticks([-6, -4, -2, 0, 2, 4])
    ax.set_xlabel("Modalities in the fixed subset, $|S|$", labelpad=9)
    ax.set_ylabel("Macro-F1 deviation from\nsubset mean (pts)", labelpad=8)
    ax.margins(x=.06, y=.09)
    ax.grid(axis="y", color="#E5E7E9", linewidth=.65)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3, width=.8, pad=4, colors="black")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)

    handles = [
        Line2D([], [], linestyle="none", marker=MARKERS[model], markersize=7,
               markerfacecolor=COLORS[model], markeredgecolor=EDGE_COLORS[model],
               markeredgewidth=.7,
               label=model.capitalize())
        for model in MODELS
    ]
    legend = fig.legend(
        handles=handles, ncol=2, loc="lower center",
        bbox_to_anchor=(.54, .89), frameon=True, fancybox=False,
        framealpha=1, facecolor="white", edgecolor="#CCCCCC",
        borderaxespad=0, borderpad=.3, columnspacing=.85,
        handlelength=.85, handletextpad=.35,
    )
    legend.get_frame().set_linewidth(.6)
    return fig, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--stem", default="order_deviation_iemocap")
    parser.add_argument("--width-inches", type=float, default=6.8)
    parser.add_argument("--height-inches", type=float, default=4.7)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    fig, groups = build_figure(load(), args.width_inches, args.height_inches)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        path = args.output_dir / f"{args.stem}.{suffix}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=.035)
        print(f"Wrote {path.resolve()}")
    plt.close(fig)
    print("Pooled population SD of fold-centered scores (F1 points):")
    for (model, size), deviations in groups.items():
        print(f"  {model}, |S|={size}: n={len(deviations)}, SD={deviations.std():.6f}")


if __name__ == "__main__":
    main()
