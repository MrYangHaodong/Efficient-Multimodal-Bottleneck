# plotting — paper figures and the values behind them

Self-contained: each script reads only its CSV in `data/` and writes the PDF/PNG next to
itself. No repo imports, no absolute paths, no GPU. Regenerate everything with

    python plot_cascade.py && python plot_selection_by_class.py && python plot_sensitivity.py

| script | figure in the paper | data |
|---|---|---|
| `plot_cascade.py` | `firstpick_cascade_iemocap.pdf` | `data/firstpick_cascade_iemocap.csv` |
| `plot_selection_by_class.py` | `selection_by_class_iemocap.pdf` | `data/selection_by_class_iemocap.csv` |
| `plot_sensitivity.py` | `sensitivity_iemocap.pdf` | `data/sensitivity_iemocap.csv` |
| `iclr_style.py` | shared style (colours, grid, spines, error bars) | — |

All three reproduce the figures in the paper **pixel-identically**; that was checked against
the originals rather than assumed, which is why the cascade CSV stores full float precision
and uses the non-breaking hyphen (U+2011) in `m‑hand` / `m‑head` / `m‑rot`. Rounding those
values to 4 decimals, or using an ordinary hyphen, both change the rendering.

## What the data columns mean

**`firstpick_cascade_iemocap.csv`** — one row per availability configuration, each removing
the modality the policy opened with in the previous one.
`first_pick` is the modality acquired first under that configuration; `removed_to_reach` is
what was taken away to get there. `f1_*` / `mu_mean` / `mean_k` are SPARQ; `ref_*` is the same
frozen backbone consuming every available modality.

**`selection_by_class_iemocap.csv`** — one row per true class. `freq_<mod>` is
P(modality in S | y), `std_<mod>` its standard deviation across folds, `mean_k` the mean
acquired-set size, `acc` the backbone's accuracy on that class.

**`sensitivity_iemocap.csv`** — one row per swept value of the three fixed hyperparameters
(`lambda_DS`, `lambda_KD`, `K`). `f1_order_averaged` is the mean over 24 permutations of the
full modality set, `f1_std_folds` the spread across folds, `std_across_orders_pts` the spread
across orderings in F1 points, `f1_canonical` the single canonical-order protocol (measured,
retained for reference, not plotted), and `deployed` marks the shipped setting.

## Provenance

The values are measurements, not re-derivations: the cascade and class-conditional numbers
come from the deployed policy walk in `energy_adaptive/`, and the sensitivity numbers from the
backbone retrains in `runs/seqA_ablations_0903/` and `runs/seqA_losssens_0903/`. This directory
is the plotting layer only.
