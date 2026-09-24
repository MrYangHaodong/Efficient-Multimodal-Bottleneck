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
| `intro_teaser_android_plotter.py` | `intro_teaser_android.pdf`, Figure 1(d) panel | `Energy_Study` and `Main_ResultsFull` workbook sheets |
| `plot_order_variance.py` | `order_variance_iemocap.pdf` | `data/order_variance_iemocap.csv` |
| `plot_order_deviation.py` | `order_deviation_iemocap.pdf`, standalone panel (a) | `data/order_variance_iemocap.csv` |
| `iclr_style.py` | shared style (colours, grid, spines, error bars) | — |

`intro_teaser_android_plotter.py` is the standalone Android INT8 teaser panel,
derived from the three-platform energy/latency plot. It shows SeMARC, MBT,
AdaMML, and DyMo with no zoom inset, title, or panel-letter label. It preserves
the paper colors, distinct markers, two-decimal F1 annotations, black axes, and
large print-size fonts. The borderless legend has exactly two rows, with smaller
inline `(Ours)`, `(Monolithic)`, and `(Selective)` descriptors and no `Fusion`
suffix. The compact canvas and tight export reduce exterior whitespace. The F1 labels
are full-availability reference scores, not Android-specific accuracy results.
Unlike the CSV-based plots above, it reads the Google Sheets XLSX export by
default, or a local workbook with `--input results.xlsx`, and shares parsing and
F1 rounding with `energy_latency_full_plotter.py`. It writes PDF, 600-dpi PNG,
and editable SVG beside the script. It does not modify `figures/intro_teaser.pptx`.

    python intro_teaser_android_plotter.py --input results.xlsx

`energy_latency_full_plotter.py` increases fonts in the IEMOCAP linear
three-panel view by 16 pt over the shared publication style (five 4 pt increases
followed by two 2 pt reductions). Main-axis and inset tick labels, plus the
legend, have an additional 2 pt reduction; F1 annotations and other text are unchanged.
All main, inset, and legend marker
sizes are increased by 16 pt (four 4 pt increases). The 18.6237-inch canvas width is
held fixed so the size increase remains visible at a fixed manuscript width.
The compact legend is centered above the panels in two rows (three methods
followed by two), preserving method order across each row. Its descriptions are
shortened to `(Monolithic)` and `(Selective)`, while `SeMARC (Ours)` is retained.
The legend sits directly above the panels, and unused upper whitespace is
cropped from the export. Enlarged F1 labels are staggered within the zoom panels, without
leader lines. The Android inset includes extra lower padding to keep its blue
marker separate from its label; its dotted source box matches those limits.
Panel (a) is shifted left by 2% of the canvas width to leave clear space for
panel (b)'s four-digit y-axis ticks; panel sizes and the other positions are unchanged.
The all-dataset, other-dataset, and log-scale font defaults are unchanged.
Reproduce the existing figure's values without refreshing the live sheet:

    python plotting/energy_latency_full_plotter.py --dataset IEMOCAP --scale linear \
        --snapshot plotting/data/energy_latency_iemocap_3panel_linear.json --dpi 600

The snapshot preserves the original plotted GPU SeMARC point, which differs
from the later live-sheet value; its provenance and exact SVG round-trip check
are recorded in the JSON. Outputs remain in `plotting/output/` as PDF, SVG, and PNG.

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
`--output-dir`. For a **30% / 40% / 30%** manuscript row, the IEMOCAP figures use
these exact, fixed-canvas dimensions in both PDF and 600-dpi PNG:

| Figure (left to right) | Native size (inches) | PNG size | Row width |
|---|---|---|---|
| Class selection | 4.5 x 3.2 | 2700 x 1920 | 0.30 |
| Availability cascade | 6.0 x 3.2 | 3600 x 1920 | 0.40 |
| Modality weighting | 4.5 x 3.2 | 2700 x 1920 | 0.30 |

Their width ratio is 3:4:3, so all three scale by the same factor and have equal
displayed heights. Native axes layouts adapt without stretching raster images;
fonts, observations, and error bars are unchanged. Small exterior margins
protect text and strokes. Place these inside an existing figure environment:

```latex
\includegraphics[width=0.30\linewidth]{plotting/selection_by_class_iemocap.pdf}%
\includegraphics[width=0.40\linewidth]{plotting/firstpick_cascade_iemocap.pdf}%
\includegraphics[width=0.30\linewidth]{plotting/modality_weight_sensitivity_iemocap.pdf}
```

The trailing `%` removes inter-image whitespace, since the widths sum to one.
If extra gaps are needed, reduce all three widths proportionally. Do not set
an independent height or use `trim`/`clip`, which would alter the matched ratios.
The cascade's optional
`--tight` restores content cropping for standalone use. Other class-selection
datasets retain their existing cropped exports.
The IEMOCAP cascade defaults to a 14 pt base (all text 4 pt larger): 12 pt axis/panel
labels, 11 pt tick/legend labels, and 10 pt annotations. The inline counts and matching
"(modality count)" axis-title suffix use smaller 9 pt text. Its 6.0-inch width
and 3.2-inch height suit the middle 40% slot in the row.
The top panel ends at Macro-F1 0.8, with ticks at 0.2, 0.5, and 0.8. Its lower
bound extends 0.025 below the lowest error-bar endpoint (about 0.12), so all error
bars remain visible. The underlying mean and standard-deviation values are unchanged.
Availability ticks use compact two-line recursive labels: All ($A_0$), $A_0$ - text,
$A_1$ - audio, $A_2$ - video, and $A_3$ - m-head, with the remaining modality count
inline on the second line: `$A_0$ (6)`, `text (5)`, `audio (4)`, `video (3)`, and
`m-head (2)`. Counts come directly from
the CSV's `n_avail` column. Here $A_0$ is the full available
set and $A_i$ is the set after the first $i$ removals, so each removal applies to
the preceding set, not independently to all modalities. The axis title is
"Available set, $A_n$ (modality count)"; the reference legend reads
"SeMA (use all $A_n$)", meaning all modalities in the current available set.
The old `w/o` labels and top-panel first-pick
annotations are omitted. Bottom-panel acquired/available annotations and all error
bars remain unchanged. Panel labels sit at the upper right; the one-row legend is
centered above the panels. The working bottom margin is 0.62 inches; the x-axis
title sits 6 pt closer to the tick labels. Layout
validation runs at both 100 dpi and the export resolution.
Class-selection heatmaps omit the redundant mean acquired-set-size strip,
keeping the probability scale unchanged.
The IEMOCAP heatmap now uses a 14 pt base (4 pt larger): 14 pt class/modality and
colorbar tick labels, 15 pt axis/colorbar titles, and 13 pt optional cell annotations.
Other datasets and bar plots retain their 10 pt base; `--font-size` overrides either
default. Regenerate only this figure with `python plot_selection_by_class.py --datasets iemocap`.
The IEMOCAP heatmap's height is increased to fill the shared 4.5 x 3.2 inch
canvas; its 14/15 pt fonts, class order, cell values, and probability scale remain
unchanged. Heatmaps also accept `--annotate` for two-decimal probability labels:

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

`python plotting/plot_order_deviation.py` produces only panel (a), with random-order
training in purple (`#AA4499`) and fixed-order training in a darker grey (`#888888`),
adjusted from Paul Tol's original bright/vibrant grey (`#BBBBBB`). It reuses the
original CSV loader, preserving all 480 observations, fold-specific centering,
and pooled population-SD marks; the legend lists only the two training schemes.
Fonts are 24 pt for ticks, 26 pt for axis labels, and 21.5 pt for the legend.
Markers have opaque darker borders; 3 pt SD marks use saturated dark purple
(`#660066`) and charcoal (`#333333`) to remain visible at preview size.
This is Macro-F1 variation,
not posterior total-variation distance. It writes `order_deviation_iemocap.pdf`,
`.svg`, and a 600-dpi `.png`, leaving the original two-panel figure unchanged.

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
Main category labels and y ticks retain 36 pt type; the y-axis title is 39.5 pt,
and values/group labels remain 27.5 pt. Category parentheticals are 2 pt smaller
(34 pt), inline rather than on a second line. All seven category labels are
horizontal on a shared text baseline and centered over their corresponding bars.
The working canvas is widened to 36.8 x 6.6 inches to fit the full single-line
labels without reducing the main fonts; the y-axis title retains its two-line
wrap and 1.0 line spacing. At a fixed manuscript display width, this wider
figure will naturally scale down more than the earlier compact version.
Label widths determine the category spacing. Bars have a fixed physical width
of 0.5 inches, about 40% thinner than the previous 0.832-inch bars; increasing
the canvas width does not widen them. Layout checks verify the common baseline,
label/bar centering, smaller parentheticals, physical bar width, text bounds,
and lack of overlap at both 100 and 600 dpi. Endpoint labels stay inside the plot.
All three ablation exports crop to the rendered content with a 1.5-point
safety margin for vector-font differences, removing blank outer margins.
Cropping does not rescale
fonts or the plot; the underlying scores are unchanged. PDF and SVG remain vector
graphics and use the same crop as the PNG.
The labels, left to right, are Heuristic (no RL), No Q-prior
($Q_0=0$), $[\mathbf{1}_{S_s}\Vert\mathbf{1}_{\mathcal A}]$ only (state),
$p_s$ only (state), Fixed-order training, Bottleneck suffix, and Final-prefix CE only.
The original source descriptions
and full-precision measurements remain in the script. The figure retains black
boundaries and no title or source footnote. The no-RL heuristic uses F1 = 0.6443,
not the separate no-learned-stop row.

## Standalone IEMOCAP Shapley heatmap

`python plotting/plot_shapley_heatmap.py` updates the same
`shapley_iemocap_test_heatmap.pdf`, editable `.svg`, and 300-dpi `.png` files
beside the script. The standalone canvas is **7.7 x 2.8 inches**
(2310 x 840 pixels), adjusted for the larger fonts. It is separate from the class-selection heatmap and does
not change any manuscript files.

Source: [ICLR Results, Shapley scores](https://docs.google.com/spreadsheets/d/1ZsKgZztYMPOj76LL7eYWT07JJ4k0cWdaC1zmOlIy3q0/edit),
**A4:H8**, downloaded and verified on 2026-09-21. The offline source snapshot
`data/shapley_iemocap_test.csv` retains every original four-decimal `mean | SD`
pair and the class counts (Neutral 958, Angry 724, Happy 334, Sad 686).
The parser also accepts `--input path/to/workbook.xlsx` and reads only this
test-split block, excluding the all-class and split-level summaries below it.

Colors encode the original signed means, without row normalization, on one
zero-centered scale from -0.30 to +0.30: the paper's orange (`#EE7733`) for
negative values, near-white at zero, and purple (`#AA4499`) for positive values.
The figure has no cell numbers, title, or explanatory footnote. Class and modality
labels use 17 pt type (increased by 4 pt); sample counts remain in the row labels,
wrapped onto a second line. The colorbar has 13 pt ticks and a 14.5 pt label
(both also increased by 4 pt); its label wraps to retain a compact layout.
The original SDs are retained in the source CSV but are not displayed.
The sample-data audit on 2026-09-23 established that the supplied means are
**equally weighted means of three fold-specific class means**, and the SDs are
the **population SD of those three fold means** (`ddof=0`), not SD across
individual samples. All 24 mean/SD pairs match this calculation at four decimals.
The heatmap remains the original summary-based figure. Thin white cell separators
and the colorbar remain vector objects, and SVG text stays editable.

Suggested caption: “Modality Shapley contributions on the IEMOCAP test split.
Colour represents the signed class mean averaged equally over three folds;
row labels include the total number of test samples in each class.”

## IEMOCAP Shapley mean and between-fold variation plot

`python plotting/plot_shapley_dot_sd.py` creates the separate
`shapley_iemocap_test_dot_sd.pdf`, editable `.svg`, and 300-dpi `.png`.
It shares the heatmap's verified CSV and source parser; neither the data nor
the manuscript is modified. The compact canvas is 8 x 3.1 inches
(2400 x 930 pixels), with tight outer margins and panel spacing.

The four class panels contain six modalities each. Dots show the original
signed equal-fold class means and capped horizontal whiskers show **mean ±1
population SD of the three fold-specific class means** (`ddof=0`). The original
description as individual-sample variation was incorrect, as confirmed by the
sample-data audit below. These are not standard errors or confidence intervals.
The existing figure files have not been regenerated. All panels share the x-range
[-0.30, +0.45], including every full interval; no whisker is clipped at the
heatmap's color limit. The extreme endpoints are -0.2448 (Sad, Video) and
+0.4069 (Happy, Text). All means and SD whiskers use paper purple (`#AA4499`);
their position relative to the dotted zero line indicates the contribution's
sign. Class headings omit sample counts; 2.8 pt whiskers, 2 pt caps, and 7.2 pt
dots improve print visibility. The background grid remains light.
Direction labels are centered above the class panels and vertically beside the
modality rows, with rightward and downward arrows. All visible text uses 17 pt.
MoCap1, MoCap2, and MoCap3 denote hand, head, and rotated, respectively;
these are display aliases only, with the original source keys unchanged.
No synthetic samples or
distributional assumptions are used. Source-to-marker and source-to-whisker
validation runs at 100 and 300 dpi before export.

Suggested caption: “Modality Shapley contributions on the IEMOCAP test split.
Dots indicate the equally weighted average of three fold-specific class means;
horizontal whiskers show ±1 population standard deviation of those fold means,
not individual-sample variation or confidence intervals.”

## IEMOCAP sample-wise Shapley beeswarm

`python plotting/plot_shapley_beeswarm.py` exports
`shapley_iemocap_test_beeswarm.pdf`, editable `.svg`, and 300-dpi `.png`.
The compact canvas is 8 x 3.6 inches (2400 x 1080 pixels), with four class panels,
17 pt text, and the same purple (`#AA4499`) for all data marks. Prior figures and
the original summary CSV remain unchanged.

The raw source was `/Users/paymoh01/Downloads/iemocap_persample_shapley_all.csv`,
supplied with the sample table. Its SHA-256 is
`1137459a285bd151dcfb5afd07d37d248167a056ffaeff2e50cbbb9e29556d0d`.
Of 10,500 input rows, the figure uses only **2,702 `random / avg / test` rows**;
static-backbone and validation rows are excluded. The original five-decimal
input strings are preserved in `data/shapley_iemocap_random_avg_test_samples.csv`.
Keys are `(backbone, vS, split, fold, idx)`, not `idx` alone. Checks cover unique
keys, finite and complete scores, class-label consistency, contiguous indices,
and six first/last-fold sentinels from the supplied table. Test-fold counts are
909, 793, and 1,000. Class counts are Neutral 958, Angry 724, Happy 334, and Sad 686.
All 38 all-zero rows are retained.

Each of the **16,212 small dots** is one supplied modality score, with its exact
x coordinate retained. The binned beeswarm adds deterministic vertical offsets
within 0.025-wide x bins; row height is scaled independently to show relative
density within each class/modality row. Height is not comparable sample count
across classes, and the swarm is not a collision-free packing or smoothed KDE.
No observations are simulated, dropped, resampled, or horizontally jittered.
Feature values were not supplied, so color does not encode feature magnitude.
All panels share [-1.08, +1.08], containing the full observed [-0.95, +1.00] range.

Large dots and capped whiskers show the **pooled sample-weighted mean ±1 sample
SD** (`ddof=1`) within each class, using all three test folds. The SD describes
the pooled sample spread, including any between-fold differences, not a
confidence interval or an isolated estimate of within-fold variation. The
statistics and comparison to the previous summary are recorded in
`data/shapley_iemocap_sample_audit.json`. For example, Happy–Video is
0.086713 ± 0.260828 across samples; the old 0.0872 ± 0.0038 was an equal-fold
summary. MoCap1, MoCap2, and MoCap3 denote hand, head, and rotated, respectively.

Before export, every plotted x value, all 24 mean/SD intervals, labels, and
text bounding boxes are validated at 100 and 300 dpi. PDF/SVG retain vector
points and text. The figure is descriptive attribution evidence; it does not
by itself establish the benefit of an online acquisition policy.

Suggested caption: “Sample-wise modality Shapley contributions for the
random-order backbone (`vS=avg`) on the IEMOCAP test split. Small dots represent
individual attributions; large dots and whiskers show the pooled class mean
and ±1 sample standard deviation across the three test folds, not confidence
intervals. Binned vertical stacks separate points visually and are scaled per
row; their heights are not comparable sample counts. MoCap1/2/3 denote
hand/head/rotated motion capture.”

### Signed-gradient beeswarm without summary markers

`python plotting/plot_shapley_beeswarm.py --style gradient` exports the separate
`shapley_iemocap_test_beeswarm_gradient.pdf`, editable `.svg`, and 300-dpi `.png`.
It preserves all 16,212 sample scores, deterministic vertical offsets, four
class panels and modality order. Typography is 24 pt, with an 8 x 4.4 inch
canvas and wider label/panel spacing to avoid crowding. Sample markers use
9 pt² gradient fills (increased from 3.8 pt²), with no border or outline. The
mean dots, SD whiskers, and summary legend are omitted. A compact horizontal
colorbar shares the label “Shapley contribution” with the horizontal positions.

All panels and modalities use the same linear, zero-centered color scale with
[Paul Tol's colorblind-friendly Sunset diverging palette](https://sronpersonalpages.nl/~pault/)
(Fig. 13): blue `#364B9A` at -1, pale cream `#EAECCC` at 0, and orange/red
leading to `#A50026` at +1. All eleven published anchors are linearly
interpolated as recommended by the author. The scheme supports common
red/green color-vision deficiencies; it is not a claim that all shades remain
equally distinguishable for every viewer. Horizontal position and the zero
line retain a color-independent sign cue, including in monochrome.
Points are opaque so their hue matches the colorbar without density-dependent
alpha blending. Color redundantly encodes the signed attribution itself,
**not original input-feature magnitude or sample density**. Vertical row heights
remain independently scaled as described above. The existing summary-overlay
exports and their audit are not overwritten; this version writes a separate
`data/shapley_iemocap_gradient_audit.json`.

Before export, checks verify exact point counts and x positions, per-point
color values and rendered RGBA, larger borderless markers, 24 pt fonts,
identical color normalization across panels, absence of mean/SD markers,
and text bounds at 100 and 300 dpi.

Suggested caption: “Sample-wise modality Shapley contributions for the
random-order backbone (`vS=avg`) on the IEMOCAP test split, pooled across three
test folds. Each dot represents one sample; horizontal position and the shared
colorblind-friendly Sunset diverging palette encode signed Shapley
contribution, with blue for negative values and orange/red for positive values.
Binned vertical
stacks separate points visually and are scaled per row; their heights are not
comparable sample counts. MoCap1/2/3 denote hand/head/rotated motion capture.”

## Modality-weighting sensitivity

`python plotting/reward-sensitivity.py` now exports only the seven-point gamma
sweep at fixed lambda = 0.5 to `modality_weight_sensitivity_iemocap.pdf` and
600-dpi `.png`. The row-matched 4.5 x 3.2 inch figure uses purple Macro-F1 bars with
the supplied plus/minus one standard deviation, orange latency bars (ms),
black axes, a light one-row legend, and no title. The tested gamma values are
categorical configurations, not a uniformly spaced continuous scale. Both bar
axes begin at zero; all uncertainty endpoints remain visible. No latency
uncertainty was supplied. `--secondary mu`, `gflops`, or `none` changes the
secondary metric; dimensions and base font size can also be overridden.

The inline gamma measurements match the user's 2026-09-21 table (Gamma CSV rows
3-9) exactly. Gamma = 0 retains F1 = 0.5893, not the separate deployed uniform
controller's 0.668. Gamma = 0.75 is marked deployed in the Gamma CSV, but this
does not equate it with the separate Lambda CSV deployment, which uses
fold-specific lambdas and a different delta field. The figure does not merge
these records or include their horizontal references. The newer raw modality
weights are unnormalized; the script does not infer a cost-normalization rule
from them. Its older normalized penalty-allocation table remains explicitly
labeled as legacy input, not the new raw weights.

The original two-panel view remains available via `--sweep both` for later
appendix work, with its earlier uniform-only penalty table unchanged. It does
not yet incorporate the newly supplied paired uniform/per-modality lambda
sweep, which is deferred. Its existing output files are not overwritten by
the default single-sweep command.
