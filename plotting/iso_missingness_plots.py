#!/usr/bin/env python3
"""Plot accuracy under low-, middle-, and high-GFLOP method strata."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from plot_missingness_pareto import (
    MISSINGNESS,
    SHEET_URL,
    interpolate_log_gflops,
    load_workbook,
    read_results,
)


OURS = "SeMARQ"
METHOD_MARKERS = {
    "SeMARQ": "o",
    "MBT": "s",
    "MAESTRO": "^",
    "ShaSpec": "D",
    "DecALign": "v",
    "MultiModN": "P",
    "DyMo": "X",
    "AdaMML": "<",
    "DyMM": ">",
    "MMEE": "h",
}
TIER_SPECS = (
    ("Low GFLOP third", slice(0, 3), "#228833", 2.1),
    ("Middle GFLOP third", slice(3, 6), "#CCBB44", 1.25),
    ("High GFLOP third", slice(6, 10), "#EE6677", 1.25),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=SHEET_URL)
    parser.add_argument("--sheet", default="Main_ResultsFull")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/figures")
    )
    parser.add_argument(
        "--stem", default="semarc_compute_tier_envelopes"
    )
    parser.add_argument(
        "--html-fragment",
        type=Path,
        default=None,
        help="Optional destination for an inline visualization fragment.",
    )
    parser.add_argument(
        "--bands-only",
        action="store_true",
        help="Omit envelope lines and show only tier ribbons plus observations.",
    )
    parser.add_argument(
        "--band-span",
        choices=("top2", "full"),
        default="top2",
        help="Vertical extent of each ribbon at a missingness level.",
    )
    return parser.parse_args()


def tier_envelopes(methods):
    envelopes = {
        name: {"f1": [], "runner_up": [], "minimum": [], "winner": []}
        for name, *_ in TIER_SPECS
    }
    for index in range(len(MISSINGNESS)):
        ranked = sorted(
            methods,
            key=lambda method: methods[method]["gflops"][index],
        )
        for name, tier_slice, _color, _width in TIER_SPECS:
            candidates = ranked[tier_slice]
            performance_rank = sorted(
                candidates,
                key=lambda method: methods[method]["f1"][index],
                reverse=True,
            )
            winner = performance_rank[0]
            runner_up = performance_rank[1]
            envelopes[name]["winner"].append(winner)
            envelopes[name]["f1"].append(methods[winner]["f1"][index])
            envelopes[name]["runner_up"].append(
                methods[runner_up]["f1"][index]
            )
            envelopes[name]["minimum"].append(
                min(methods[method]["f1"][index] for method in candidates)
            )
    for values in envelopes.values():
        values["f1"] = np.asarray(values["f1"], dtype=float)
        values["runner_up"] = np.asarray(values["runner_up"], dtype=float)
        values["minimum"] = np.asarray(values["minimum"], dtype=float)
    return envelopes


def plot(
    results,
    output_dir: Path,
    stem: str,
    bands_only: bool = False,
    band_span: str = "top2",
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.12))
    low_tier_ours = 0
    low_tier_total = 0
    all_envelopes = {}

    for panel_index, (ax, (dataset, methods)) in enumerate(
        zip(axes.flat, results.items())
    ):
        envelopes = tier_envelopes(methods)
        all_envelopes[dataset] = envelopes

        # Show every evaluated method at every missingness level. Marker shape
        # identifies the method; marker edge colour identifies its GFLOP tier.
        for point_index, missingness in enumerate(MISSINGNESS):
            ranked = sorted(
                methods,
                key=lambda method: methods[method]["gflops"][point_index],
            )
            for name, tier_slice, color, _width in TIER_SPECS:
                winner = envelopes[name]["winner"][point_index]
                for method in ranked[tier_slice]:
                    if method == winner and not bands_only:
                        continue
                    is_ours = method == OURS
                    ax.plot(
                        missingness,
                        methods[method]["f1"][point_index],
                        linestyle="none",
                        marker=METHOD_MARKERS[method],
                        markersize=4.0 if is_ours else 3.1,
                        markerfacecolor="#4477AA" if is_ours else "white",
                        markeredgecolor="#4477AA" if is_ours else color,
                        markeredgewidth=0.75 if is_ours else 0.65,
                        alpha=0.95 if is_ours else 0.72,
                        zorder=4 if is_ours else 2,
                    )

        for tier_index, (name, _tier_slice, color, width) in enumerate(TIER_SPECS):
            values = envelopes[name]
            lower = (
                values["minimum"]
                if band_span == "full"
                else values["runner_up"]
            )
            ax.fill_between(
                MISSINGNESS,
                lower,
                values["f1"],
                color=color,
                alpha=(0.13 if tier_index == 0 else 0.09)
                if bands_only
                else (0.16 if tier_index == 0 else 0.11),
                linewidth=0,
                zorder=0.7,
            )
            if not bands_only:
                ax.plot(
                    MISSINGNESS,
                    values["f1"],
                    color=color,
                    linewidth=width,
                    alpha=1.0 if tier_index == 0 else 0.86,
                    zorder=3 - tier_index,
                )
            for point_index, (missingness, f1, winner) in enumerate(
                zip(MISSINGNESS, values["f1"], values["winner"])
            ):
                is_ours = winner == OURS
                if not bands_only:
                    ax.plot(
                        missingness,
                        f1,
                        linestyle="none",
                        marker=METHOD_MARKERS[winner],
                        markersize=4.8 if is_ours else 4.0,
                        markerfacecolor="#4477AA" if is_ours else color,
                        markeredgecolor="white" if is_ours else "#555555",
                        markeredgewidth=0.7 if is_ours else 0.55,
                        alpha=1.0 if is_ours else 0.9,
                        zorder=8 if is_ours else 5,
                    )

                if tier_index == 0:
                    low_tier_total += 1
                    low_tier_ours += int(is_ours)

        ax.set_xlim(-1.0, 41.0)
        ax.set_xticks(MISSINGNESS)
        all_f1 = np.concatenate(
            [values["f1"] for values in methods.values()]
        )
        span = max(all_f1) - min(all_f1)
        pad = max(0.018, 0.085 * span)
        ax.set_ylim(min(all_f1) - pad, max(all_f1) + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(True, color="#E1E1E1", linewidth=0.45, alpha=0.72)
        ax.tick_params(direction="out", length=2.2, width=0.55, pad=1.5)

        panel_label = chr(ord("a") + panel_index)
        ax.text(
            0.025,
            0.04,
            f"({panel_label})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            fontweight="normal",
            color="#333333",
            zorder=20,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#444444")
            spine.set_linewidth(0.7)

    tier_handles = [
        Patch(
            facecolor=color,
            edgecolor=color,
            linewidth=0.7,
            alpha=0.22,
            label=name,
        )
        for name, _tier_slice, color, width in TIER_SPECS
    ]
    tier_legend = fig.legend(
        handles=tier_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="#FAFAFA",
        edgecolor="#D0D0D0",
        columnspacing=1.15,
        handlelength=1.55,
        handletextpad=0.35,
        borderpad=0.35,
    )
    tier_legend.get_frame().set_linewidth(0.55)

    winner_methods = list(METHOD_MARKERS)
    method_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=METHOD_MARKERS[method],
            markerfacecolor="#4477AA" if method == OURS else "#F4F4F4",
            markeredgecolor="white" if method == OURS else "#555555",
            markeredgewidth=0.7,
            markersize=4.8,
            label="SeMARC" if method == OURS else method,
        )
        for method in winner_methods
    ]
    method_legend = fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=len(method_handles),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="#FAFAFA",
        edgecolor="#D0D0D0",
        fontsize=5.2,
        columnspacing=0.52,
        handlelength=0.72,
        handletextpad=0.18,
        borderpad=0.3,
    )
    method_legend.get_frame().set_linewidth(0.55)

    fig.text(
        0.985,
        0.872,
        (
            f"SeMARC reaches the top of the low-GFLOP band in "
            f"{low_tier_ours}/{low_tier_total} conditions"
            if bands_only
            else f"SeMARC defines {low_tier_ours}/{low_tier_total} "
            "low-compute envelope points"
        ),
        ha="right",
        va="top",
        fontsize=6.2,
        color="#444444",
    )
    fig.supxlabel("Modality missingness (%)", x=0.52, y=0.004, fontsize=8.2)
    fig.supylabel("Macro-F1", x=0.006, y=0.47, fontsize=8.2)
    fig.subplots_adjust(
        left=0.058,
        right=0.995,
        bottom=0.085,
        top=0.835,
        wspace=0.17,
        hspace=0.16,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for extension, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 400}),
    ):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        output_paths.append(path)
    plt.close(fig)
    return output_paths, all_envelopes, low_tier_ours, low_tier_total


def write_html_fragment(svg_path: Path, destination: Path) -> None:
    svg = svg_path.read_text(encoding="utf-8")
    svg = re.sub(r"^<\?xml[^>]*>\s*", "", svg)
    svg = re.sub(r"^<!DOCTYPE[^>]*(?:\[[\s\S]*?\]\s*)?>\s*", "", svg)
    fragment = (
        '<div id="compute-tier-envelope-mock" '
        'style="width:100%;max-width:1200px;margin:0 auto;">\n'
        '<style>#compute-tier-envelope-mock svg{display:block;width:100%;'
        'height:auto;}</style>\n'
        f"{svg}\n"
        "</div>\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(fragment, encoding="utf-8")


def main() -> None:
    args = parse_args()
    workbook = load_workbook(args.input)
    results = read_results(workbook[args.sheet])
    interpolate_log_gflops(results)
    paths, envelopes, ours_count, total = plot(
        results,
        args.output_dir,
        args.stem,
        bands_only=args.bands_only,
        band_span=args.band_span,
    )
    if args.html_fragment is not None:
        svg_path = next(path for path in paths if path.suffix == ".svg")
        write_html_fragment(svg_path, args.html_fragment)
        paths.append(args.html_fragment)

    print(f"SeMARC low-compute envelope points: {ours_count}/{total}")
    print("Created:")
    for path in paths:
        print(f"  {path.resolve()}")
    print("Low-compute winners:")
    low_name = TIER_SPECS[0][0]
    for dataset, dataset_envelopes in envelopes.items():
        winners = ", ".join(dataset_envelopes[low_name]["winner"])
        print(f"  {dataset}: {winners}")


if __name__ == "__main__":
    main()
