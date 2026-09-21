# plotting — paper figures and the values behind them

Self-contained: each script reads only its CSV in `data/` and writes the PDF/PNG next to
itself. The order-variance script also exports editable SVG. No repo imports outside this
directory, no absolute paths, no GPU. Regenerate everything with

    python plot_cascade.py && python plot_selection_by_class.py && \
    python plot_sensitivity.py && python plot_teaser_frontier.py && \
    python plot_order_variance.py

| script | figure in the paper | data |
|---|---|---|
| `plot_cascade.py` | `firstpick_cascade_iemocap.pdf` | `data/firstpick_cascade_iemocap.csv` |
| `plot_selection_by_class.py` | `selection_by_class_<ds>.pdf`, all 7 datasets | `data/selection_by_class_<ds>.csv` |
| `plot_sensitivity.py` | `sensitivity_iemocap.pdf` | `data/sensitivity_iemocap.csv` |
| `plot_teaser_frontier.py` | `teaser_frontier.pdf` | `data/teaser_frontier_iemocap.csv` |
| `plot_order_variance.py` | `order_variance_iemocap.pdf` | `data/order_variance_iemocap.csv` |
| `iclr_style.py` | shared style (colours, grid, spines, error bars) | — |

`plot_selection_by_class.py` holds **one implementation per form, not per dataset** —
`bars(ds)` draws a panel per class, `heatmap(ds)` draws a class x modality image with a
probability colorbar. `--form auto` (the default) draws a heatmap for
every dataset, which is what the paper uses throughout: one comparable view across label
spaces from $C{=}4$ to $C{=}27$, where a panel per class stops being readable. `--form bars`
is kept because it is the more direct reading when the label space is small.

Asking for the other form writes a suffixed file rather than overwriting the published one:

    python plot_selection_by_class.py --form bars --datasets iemocap eav
    # -> selection_by_class_iemocap_bars.pdf, selection_by_class_eav_bars.pdf

Except for the restyled order-variance, cascade, and class-selection figures, these figures reproduce the ones in the paper
**pixel-identically**; that was checked against
the originals rather than assumed, which is why the cascade CSV stores full float precision
and uses the non-breaking hyphen (U+2011) in `m‑hand` / `m‑head` / `m‑rot`. Rounding those
values to 4 decimals, or using an ordinary hyphen, both change the rendering.

The cascade and class-selection scripts now use SeMARC purple, black axes, larger print-size
labels, no overall title, tight spacing, embedded PDF fonts, and 600-dpi PNG output. Their
CSV measurements are unchanged; the cascade retains its supplied standard-deviation bars.
Both scripts accept `--font-size`, `--width-inches`, `--height-inches`, `--dpi`, and
`--output-dir`. Dimensions refer to the Matplotlib canvas: class-selection exports trim
the margins, while the cascade preserves the exact canvas size.
Class-selection heatmaps omit the redundant mean acquired-set-size strip and use a
narrower canvas (3.8 inches for IEMOCAP), keeping the fonts and probability scale
unchanged. They also accept `--annotate` for two-decimal probability labels:

    python plot_selection_by_class.py --datasets iemocap --annotate
    python plot_cascade.py --width-inches 5.5 --font-size 10

## What the data columns mean

**`firstpick_cascade_iemocap.csv`** — one row per availability configuration, each removing
the modality the policy opened with in the previous one.
`first_pick` is the modality acquired first under that configuration; `removed_to_reach` is
what was taken away to get there. `f1_*` / `mu_mean` / `mean_k` are SPARQ; `ref_*` is the same
frozen backbone consuming every available modality.

**`selection_by_class_<ds>.csv`** (7 datasets) — one row per true class. The IEMOCAP file
carries an extra `class_name` column; the others index classes numerically. `freq_<mod>` is
P(modality in S | y), `std_<mod>` its standard deviation across folds, `mean_k` the mean
acquired-set size, `acc` the backbone's accuracy on that class.

**`teaser_frontier_iemocap.csv`** — three rows per method, one per missingness rate
(0 / 0.2 / 0.4) on IEMOCAP, which is what makes each method a short polyline rather than a
point: `f1` macro-F1, `mu` the modality-usage fraction, `gflops` the inference cost the figure
puts on the log x-axis. Static-cost methods hold `gflops` fixed across the three rates; SPARQ
moves left as well as down.

**`order_variance_iemocap.csv`** — one row per (training scheme, subset size, fold, ordering):
`macro_f1` is the test macro-F1 of that one acquisition ORDER of that fixed modality subset.
`subset_size` runs 2--6 and the orderings per subset are capped at 24, so |S|=2 has 2 rows per
fold, |S|=3 has 6, and |S|>=4 has 24. The figure never plots the level, only the deviation from
each (training, |S|, fold) mean, so what it shows is the spread the ordering induces.
Panel (a) retains all 480 observations, with horizontal marks at plus/minus one population
standard deviation of the fold-centred observations pooled within each training/subset group.
Panel (b) shows the population standard deviation across orders within each fold, averaged
over the three folds. These are different summaries of spread, not confidence intervals.

**`sensitivity_iemocap.csv`** — one row per swept value of the three fixed hyperparameters
(`lambda_DS`, `lambda_KD`, `K`). `f1_order_averaged` is the mean over 24 permutations of the
full modality set, `f1_std_folds` the spread across folds, `std_across_orders_pts` the spread
across orderings in F1 points, `f1_canonical` the single canonical-order protocol (measured,
retained for reference, not plotted), and `deployed` marks the shipped setting.

## Provenance

The values are measurements, not re-derivations: the cascade, class-conditional and teaser
numbers come from the deployed policy walk in `energy_adaptive/` (the teaser is the IEMOCAP
column of the main results table), and the sensitivity and order-variance numbers from the
backbone retrains in `runs/seqA_ablations_0903/` and `runs/seqA_losssens_0903/`. This directory
is the plotting layer only.

Two of these figures are not currently placed in the paper: `teaser_frontier.pdf` is in
`figures/` but its `figure` environment in `introduction.tex` is commented out, and
`order_variance_iemocap.pdf` supports the order-insensitivity appendix (`app:order`), which is
still a stub. The teaser script retains its original pixel-identical rendering.
`plot_order_variance.py` was restyled for publication on 2026-09-16, using `iclr_style.py`
with larger type, black boxed axes, a light one-row shared legend, and no subplot titles.
The source CSV and statistical calculations are unchanged. Running the script updates the
same `order_variance_iemocap.pdf`, `.svg`, and 400-dpi `.png` files in this directory.

## Selected IEMOCAP ablations

`python plotting/plot_ablation.py` (from the repository root) updates the same
`ablation_iemocap.pdf`, `.svg`, and 600-dpi `.png` files beside the script. This is one
F1-only panel containing the user's final selection: four RL and three backbone
ablations at 0% missingness. The seven scores are preserved inline in the script from
the supplied table (2026-09-20); the supplied baseline is Macro-F1 = 0.668.

Bars show `100 * (F1_variant - 0.668) / 0.668`, with F1 in parentheses. Both are
displayed to two decimals; calculations retain the original four-decimal scores. Negative
percentages are relative decreases from the full model, not percentage-point changes
or full-model gains divided by an ablated score. No uncertainty was supplied.
The downward-bar layout follows [Figure 6, page 9](https://arxiv.org/pdf/2509.25278#page=9),
but uses relative Macro-F1 instead of that reference's absolute accuracy drops.
All bars use the paper's Macro-F1 purple (`#AA4499`). A dotted vertical divider
separates ARC (RL) from SeMA (backbone); these group names appear as unboxed
Times New Roman text inside the plot. Category labels sit above the plot.
Times New Roman must be installed; the script rejects silent font substitution.
Axis lettering is enlarged by another 4 pt: 22 pt x/y tick labels and a 25.5 pt
axis label; values/group labels remain 19.5 pt. The 16.15 x 6.45 inch canvas and
category spacing fit the exact two-line labels without overlap or clipping.
The labels, left to right, are Heuristic policy (no RL), No Q-prior
($Q_0=0$), Masks-only state, Predictive-summary state only, Fixed-order training,
Bottleneck tokens last, and Final-prefix CE only. The original source descriptions
and full-precision measurements remain in the script. The figure retains black
boundaries and no title or source footnote. The no-RL heuristic uses F1 = 0.6443,
not the separate no-learned-stop row.
