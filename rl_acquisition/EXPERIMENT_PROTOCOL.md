# Experiment protocol — RL per-sample multimodal acquisition

Goal: **an adaptive RL policy that beats Shapley-based ranking on the efficiency/accuracy
trade-off, while staying at or above full-modality accuracy.** Holds for both **proactive**
deployment (all modalities available; choose which to acquire, when to stop) and **reactive**
deployment (a random subset is available; choose among what's there). The final headline is
**one number per method per cell**, selected on VAL, reported on TEST (§7).

**Availability-awareness is the DEFAULT formulation**, not an extension. Each sample arrives
with a subset of modalities present; the policy picks among what is there. Full availability
(P0) is the **degenerate special case** — and the case in which per-sample adaptivity is
structurally impossible at the first pick (§8.8). §9–§11 therefore describe the default setting;
§1–§8 describe the machinery shared by all of it.

Everything below is what we run, on what data, against what. Design choices that look
arbitrary are justified in §8 — each one is there because a measurement forced it.

**Where this stands (2026-07-16).** The policy is frozen: **`dqn_policy_qprior`** — Shapley
bootstraps `Q`, the net learns only the residual (§4). One command runs a dataset end to end:
`./runs.sh <dataset> <gpu>` (§14). IEMOCAP is done at 3 folds × **seed 0**; CMI is next (cache
built, needs the run); CZU-MHAD is deferred and MM-Fi is blocked on a split problem (§13 TODO-A).

**Two things must land before any table is publishable** (§13 TODO-B): the reported levers were
selected on **TEST**, not val (§7 status note), and at μ = 0 **qprior does not significantly beat
the all-available baseline** on any condition. Both have identified fixes — val-based selection,
and sweeping `λ` rather than the post-hoc `δ` (§13 TODO-C).

---

## 1. Datasets

**Scope: IEMOCAP and CMI first. CZU-MHAD deferred** — see the power figures below.

| dataset | order | M | classes | folds | class imbalance (val) | checkpoint (all pv3 = mbtAvg + prefixup + randord) |
|---|---|---|---|---|---|---|
| **IEMOCAP** | 1st | 6 | 4 | 0,1,2 | 5.03× | `runs/iemocap_cs_seqA_variants/2026-07-10_iemocap_pv3_mbtAvg_prefixup_randord` |
| **CMI** | 2nd | 7 | 18 | 0,1,2 | 4.00× | `runs/prefixup_v3_6ds/2026-07-10_cmi_pv3_mbtAvg_prefixup_randord` |
| CZU-MHAD | deferred | 7 | 22 | 0,1,2 | 1.10× | `runs/prefixup_v3_6ds/2026-07-10_czu_mhad_pv3_mbtAvg_prefixup_randord` |

**Why CZU is deferred — measured, not assumed.** Fitting on val (§2) leaves these sample counts,
and the noise floor on a per-node value estimate is `sqrt(p(1-p)/n_fit)`:

| dataset | val | fit | select | noise floor |
|---|---|---|---|---|
| IEMOCAP | 793-942 | 635-754 | 158-188 | **±0.020** |
| CMI | 917-918 | 734-735 | 183 | **±0.018** |
| CZU-MHAD | 220-286 | **176-229** | **44-57** | **±0.036** |

CZU's floor is ±0.036. Seeds do not reduce it — all seeds share the same samples. Run CZU only
after IEMOCAP/CMI, and report the floor alongside any CZU number.

Checkpoints are **frozen**. No gradient ever reaches the backbone.

**`MAX_DEPTH = M`** (was 4). A depth cap structurally imposes the behaviour the policy is meant to
*learn*: capped at 4 it can never try 5-6 and let the cost term decide. It also means "all
modalities" is not a node in the tree, which produced the order-contamination error corrected in
§8.4. One-time rebuild:

| dataset | M | d=4 nodes | d=M nodes | on-disk/split |
|---|---|---|---|---|
| IEMOCAP | 6 | 517 | 1,957 (4×) | ~14 MB |
| CMI | 7 | 1,100 | 13,700 (12×) | ~400 MB |
| CZU-MHAD | 7 | 1,100 | 13,700 (12×) | ~600 MB |

≈3 h GPU for IEMOCAP + CMI (9 splits each fold×split). Update `environment.MAX_DEPTHS` after.

The backbone is precomputed into a **cached prefix tree** (node = ordered set of acquired
modalities, storing the model's softmax for every sample), so policy training needs zero GPU model
calls. Caches are reused from `RL_pv3/cache/` where the depth cap allows.

---

## 2. Splits — strict hygiene, non-negotiable

| split | used for | never used for |
|---|---|---|
| **TRAIN** | **nothing.** See below. | everything |
| **VAL** | **fit** the Q-network (FQI); compute Shapley values / advice; **and** select every deployment hyperparameter (`τ`, `δ`, `α`, the Q-net checkpoint, the temperature schedule) via an inner split. Selection metric: `argmax(macro-F1_val − μ · acquired_val/\|A\|_val)`, **μ = 0.20** (§7). | reporting |
| **TEST** | report at the val-selected settings, exactly once | anything else |

**Why TRAIN is unused — measured, not preference.** The backbone memorises its train split
(IEMOCAP video: **0.9989 train / 0.3124 test**, §8.1). The RL policy is a *meta-model* on top of
the frozen backbone, and a meta-model cannot learn from base-model predictions on data the base
model was trained on — the predictions are already contaminated with the labels. A policy fit on
the train cache learns *"take video, you're done"*; video is at **chance** on test (0.31 for 4
classes). Measured: **train-fit TEST F1 = 0.40-0.48 vs val-fit ~0.60**, same code, same test set,
no leakage anywhere. The train cache is a broken simulator, not a leaked test set.

The cost: val is 793-942 samples (IEMOCAP), of which ~80% fits and ~20% selects — noise floor
±0.020 (§1). The textbook fix is **out-of-fold** backbone predictions (K retrainings → honest
predictions on all 2,755 train samples). Not done; recorded as the way to raise power if the
val-fit result warrants it.

Baselines get the **identical** treatment: Shapley order from TRAIN, `τ` (or subset size) from
VAL, reported on TEST. No method, RL or baseline, sees TEST before reporting. The comparison
table is filled only with cells produced by this pipeline (§7).

---

## 3. Baselines — the same three everywhere, every pattern

| baseline | definition | role |
|---|---|---|
| **B1'** static Shapley, restricted | Shapley order from the fit split → one global order. Walk it, **skipping anything not in `A`**; stop at `conf ≥ τ`. τ selected on val, per pattern. | **THE BAR.** Defines `B_star` (§7). |
| **B2'** all-available | Forward the backbone with all of `A` present. One prediction per sample. | reference at ratio 1.0 |
| **B3'** random-order | Per sample, uniformly permute `A`; walk that order; stop at `conf ≥ τ` (τ from val). | isolates the value of **ordering** from the value of **counting**: B1' − B3' is what the Shapley order buys |
| **B_full** all M modalities | Every modality consumed, one forward, **val-LOO order**. | the performance we want to match |

**B_full is order-sensitive — pass the val-LOO order.** The sequential bottleneck is
order-dependent even at full modality, and `env.all_modality()` defaults to the *canonical* order,
which puts `video` first on IEMOCAP — the memorising modality, i.e. the worst possible seed (§8.5).

| IEMOCAP fold 0 | macro-F1 |
|---|---|
| canonical order (`order=None`) — do not use | 0.5630 |
| **val-LOO order (`order=cache['valloo_order']`) — USE THIS** | **0.6046** |

Computed by a direct forward (`env.all_modality`), **not** read off the tree: with `MAX_DEPTH < M`
the cache has no all-modality node, and reading one silently returns the top-`MAX_DEPTH` prefix of
whatever order it is handed — which moves the number by ~0.04 F1. With `MAX_DEPTH = M` (§1) the
node exists, but the direct forward remains the reference.

**B1' already adapts to `A`** — it skips what isn't there. So *"our policy adapts to availability"*
is **not by itself a result**. RL beats B1' only by (a) **ordering** better *within* `A` than the
global order restricted to `A` — which needs modality-specific interaction, since Shapley values
are *marginal* (averaged over coalitions of the full set) while deploy needs *conditional on `A`*;
uniform redundancy inflates values without reordering — or (b) **depth**: knowing when further
acquisition is futile given `A`, which a fixed τ cannot express. Neither route has been measured.

Every baseline gets the **identical** treatment as every RL variant: fit split → val for the
lever → report on test, at `B_star` (§7).

---

## 4. RL policy — `dqn_policy_qprior` (the frozen design)

**One policy.** The five earlier variants (`dqn_policy_basic/noexit/rlv2`, `cql_policy_setcql[_noexit]`)
are in `archive/` with their results. They are superseded, not deleted — §4a records why.

### The idea: Shapley is a prior on Q, not a state feature

```
Q(s, a) = Q_prior(s, a) + f_θ(s, a),          f_θ output layer zero-initialised

Q_prior(s, STOP)      = conf(s)                          what stopping now is worth
Q_prior(s, acquire m) = conf(s) + advice(s, m) − λ/|A|   value now + expected gain − cost
```

At init `f_θ ≡ 0`, so **the policy IS Shapley-greedy with a cost-aware stop**. FQI then learns only
the *correction*. This inverts the failure mode: a plain Q-net that fails to learn degrades to
noise, whereas this degrades to the prior. It is strictly stronger than set-CQL's soft anchor
(a penalty pulling Q toward val-LOO), because the prior is built into the *function*, not the loss.

### `advice` is CONDITIONAL, and that is the whole trick

```
advice(s, m) = softmax[node] @ tabs[S_node]           tabs: {frozenset(S) → [C, M]} of v(S+m) − v(S)
```

The `[C, M]` table is **selected by the node's acquired set**, so the value of `m` is its gain
*given what you already hold*. Marginal Shapley says "acquire" forever — it averages over coalitions
*without* `text`, so `audio` keeps `+0.179` even at the node where you already have `text`.
Measured on IEMOCAP fold 0 (val):

| held set `S` | conditional gain |
|---|---|
| `{}` | audio **+0.303** |
| `{text}` | audio **+0.059** (3× smaller — the redundancy is now visible) |
| `{text, audio}` | mocap_rot −0.013, video −0.060 (**adding more hurts**) |

With cost `λ/|A| = 0.031` the prior therefore stops after `text+audio` **on its own** — no λ retuning.

### Why this fixed the video pathology

Putting Shapley in the *state* (`dqn_policy_rlv2`) asks the net to *learn* the ranking from ~635 val
samples. It does not: when `text` is unavailable it reaches for `video` — the modality that
memorises (0.9989 train / 0.3124 test, §8.1) and is at chance on test.

| F1 on text-absent rows | value |
|---|---|
| B1' static Shapley | 0.5024 |
| `dqn_basic` | 0.3020 |
| `dqn_rlv2` | 0.2895 |
| **`dqn_qprior` (COND + resample)** | **0.4960** |

Here the prior is *structural* — the policy **cannot** pick video, because `advice(video)` is low.
First-picks when text is absent moved video 116 → audio 145, identical to B1'.

### Availability resampling — draw fresh EVERY FQI iteration

`RESAMPLE = True` (`train()`). With one draw per sample, each sample is only ever seen under a
single availability mask → 793 (sample, availability) pairs reused across all 8 iterations, and the
text-absent rows are a fixed ~175 the net can memorise. Resampling gives `793 × ITERS` pairs and
shows the **same sample both with and without text** — exactly the contrast needed to learn
"text gone → take audio, not video". This is what §9's "per episode" requires, and it is half the
reason the row above reads 0.4960 instead of 0.2895.

### Configuration

| knob | value | where | meaning |
|---|---|---|---|
| `LAM` | **0.15** | `dqn_policy_qprior` | price of consuming ALL of `A` (§5) |
| `GAMMA` | 1.0 | " | terminal-only reward; no discounting |
| `P_TRAIN` | 0.7 | " | train-time availability rate (§9) |
| `RESAMPLE` | True | " | redraw availability each FQI iteration |
| `CHUNK` | 1e6 | " | rows/forward — see below |
| `ITERS, EPOCHS, LR` | 8, 40, 1e-3 | `rl_core` | FQI schedule |
| `BALANCED` | True | `rl_core` | class-balanced reward (§5) |
| state | `[mask ǀ available ǀ softmax ǀ H ǀ margin ǀ depth/M ǀ ǀAǀ/M]` = `2M+C+4` | `rl_core.build_state` | |
| stop | `argmax Q == STOP`; `δ` sweeps the frontier at eval | | §7 |

**Chunking is not optional at `MAX_DEPTH = M`.** The tree is huge: IEMOCAP 1,957 nodes × 635 = 1.2M
rows (0.64 GB hidden, fine); **CMI 13,700 nodes × 734 = 10.1M rows → 5.15 GB hidden, and with
backward it OOMs** (one 10M-row forward asked for 8.98 GiB and died). `_fwd` chunks the forward and
`_fit_epoch` accumulates gradients over chunks — the gradient is *identical* to the single-shot
version (verified `max|diff| = 0.0`), but peak memory is `O(chunk)` instead of `O(rows)`.

### 4a. Why the other five are archived

| file | state | verdict |
|---|---|---|
| `dqn_policy_basic` | permutation tree, 517 nodes | reference; reaches for video when text is absent |
| `dqn_policy_noexit` | " | STOP removed, `conf ≥ τ` instead |
| `dqn_policy_rlv2` | " + Shapley advice in state | advice-as-state does not learn the ranking (table above) |
| `cql_policy_setcql` | **set lattice**, 57 nodes | CQL anchor (val-LOO) + temperature calibration |
| `cql_policy_setcql_noexit` | " | CQL + calibration + conf-exit |

- **Permutation tree**: state = the *ordered* sequence (`'0,2,1' ≠ '0,1,2'`).
- **Set lattice**: orderings of the same set merge (`Σₖ C(M,k)`). Order-invariance becomes
  structural — the Q-function *cannot* condition on order.

**Availability was never a separate policy.** Every variant gets the same two changes (§10):
`available(M)` in the state, and an action mask intersected with `A`. There is no
`dqn_policy_reactive`.

(A sixth variant, `dqn_policy_noexitweighted`, weighted cost by feature dimension. **Dropped** —
the premise is refuted, see §8.2. Its finding is retained there as the justification for uniform cost.)

---

## 5. Reward, cost, metric

**Reward** (`rl_core.correctness`, `BALANCED = True`):
```
R = 1[correct] × (1/freq(y)) / E[1/freq(y)]        at the stop node
```
Normalized to mean 1, so the reward stays unit-scale and `λ`'s meaning is preserved.
Class-balanced because we report macro-F1: plain `1[correct]` optimizes accuracy, letting the
policy win by serving the majority class — exactly what macro-F1 refuses to reward. A per-sample
reward cannot *be* macro-F1 (F1 is a set-level statistic, not decomposable per sample); inverse-
frequency weighting is the standard alignment.

Depends only on ground-truth `y` — **never** the predicted class or confidence. This is what makes
the confidence-exit rule non-gameable.

**Cost**: **uniform per modality, NORMALISED by what was available**:
```
r_acquire(node, sample, m) = − λ / |A_sample|
total cost at stop depth k = λ · k / |A|
```
A fixed `λ` per acquire is not comparable across availability patterns: taking 2 modalities means
"I took *everything*" when `|A| = 2` and "I took a third" when `|A| = 6`. Normalising by `|A|`
makes the penalty mean the same thing everywhere.

**`λ` is therefore the price of consuming ALL available modalities** — at `k = |A|` the total cost
is exactly `λ`, whatever `|A|` is. That makes it directly comparable to an F1 gain: `λ = 0.2`
means "I'll give up 0.2 F1 rather than consume everything available."

Keep `λ < 1` so correctness dominates. `λ` is the STOP lever for learned-stop variants (`τ` plays
that role for conf-exit). At `P_AVAIL = 1.0`, `|A| = M` for every sample, so this reduces to a
rescaled uniform cost (`λ/M` per acquire) — the degenerate case stays consistent.

**Metric**: **macro-F1 — reported AND model-selected, everywhere.** Never accuracy.
Also reported: mean #modalities, and **mean cumulative GFLOPs/sample** (analysis only — GFLOPs
does not steer any policy).

**Under missingness (§10):** reward is unchanged — still `1[correct] × 1/freq(y)` at the stop
node, ground-truth-only. The action set narrows to `available ∩ ¬acquired`; the availability
draw does not enter the reward. The exit rule remains non-gameable.

---

## 6. Deployment and aggregation

- **ε-greedy deploy**, ε = 0.1: with probability ε take a uniformly random *legal-and-available*
  action. This makes the reported policy stochastic — a greedy rollout on a deterministic cache
  gives **std = 0 by construction**, which is not an honest error bar.
- **3 folds × 3 seeds.** Average over folds (data splits, not repeats), report **mean ± std
  across seeds**.
- Report **macro-F1**, **mean #modalities**, **mean GFLOPs/sample** for every headline cell.
  Under missingness (§10), also report **mean #acquired / mean |available|** (comparable ratio).

**Collapse check**: any cell reporting `mean#modalities == MAX_DEPTH` exactly (or under
missingness, `mean#acquired == mean|available|`) is a take-all collapse, not a result. Flag it,
don't report it as a win.

---

## 7. Headline selection — matched-budget comparison, one number per method per cell

**Why not just report each method's val-selected point.** An earlier draft picked each
method's lever independently on val by `argmax(F1_val − 0.05 · #mods_val/M)`, then compared
raw F1 across the resulting cells. That leaves each method sitting at a *different* budget:
comparing raw F1 across different budgets is exactly the confound §7 was written to eliminate
(rlv2 0.618 @ 3.64 modalities vs noexit 0.585 @ 1.25 — the gap is the budget). Two symptoms:
(a) at μ = 0.05 the cost penalty never bites (5 extra modalities cost only ~0.04 F1 while
k=1→k=4 gains ~0.11 F1 on IEMOCAP), so every method selects near-max budget; (b) even if μ
were tuned to bite, raw F1 across differing budgets remains incommensurable.

**The levers** (one per method — this is what "sweep the lever" means):

| method | lever | values | kind |
|---|---|---|---|
| `dqn_qprior` | **`δ`** — offset added to `Q[:,:,STOP]` at rollout | `[-1.5, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5]` | **post-hoc**: `δ != 0` breaks Bellman consistency with the trained `λ` (§13 TODO-C) |
| `B1'` static Shapley | **`τ`** — stop when `conf ≥ τ` | `[0.2 … 0.99]`, 10 values | **native**: B1's own exit rule |
| `B3'` random order | `τ` | same | native |
| `B2'` all-available | **none** | single point (`lever_value = 2.0`, so `τ > 1` never fires) | — |
| `B_full` | none | single point, **A0 only** (§11) | — |

`δ` and `τ` are **not the same kind of object**, and the comparison inherits that asymmetry:
sweeping `τ` explores B1's *intended* policy family, while sweeping `δ` hacks a Q-function trained
at one `λ` into pretending it was trained at another. Sweeping `λ` is the principled fix (§13 TODO-C).

**Rule (val → one test cell, matched budget):**
1. Sweep each method's lever on **VAL**. Record the full frontier `(#acquired_val/|A|_val, F1_val)`
   per method. `run_design.py` emits `macro_f1_val` / `ratio_val` per sweep row for exactly this.
2. **Fix a common target budget** on val, defined by the primary baseline **B1'** (static
   Shapley restricted to `A` + conf-exit):
   `B_star = argmax(F1_val − μ · (#acquired_val / |A|_val))` with **μ = 0.20**.
   Budget is the **consumed fraction** `#acquired/|A|`, not raw `#modalities` — the latter is
   not comparable across availability patterns. Everything is compared at *B1's chosen fraction*.
3. For every other method, find the lever setting whose val `#acquired/|A|` is closest to `B_star`
   (linear interpolation between the two bracketing frontier points if no exact match). Freeze
   that lever.
4. Evaluate on **TEST** at the frozen lever → one headline cell:
   `(F1_test, mean#mods_test, GFLOPs_test)`.
5. Every baseline (B1', B2', B_full) and every RL variant gets this identical treatment.

**Why μ = 0.20** (updated from 0.05 per C5): on IEMOCAP the F1 gain from k=1→k=4 is ~0.20; at
μ = 0.05 the penalty (~0.04) never binds, so B1' collapses to take-all and the "matched budget"
becomes 1.0. At μ = 0.20 the penalty is on the same order as the F1 gain, so B1's
selected budget actually reflects a genuine trade-off. Confirm on all three datasets before
freezing.

**Ties and collapse guards:**
- If B1's val frontier has two lever values within 0.002 F1, pick the one with the lower ratio.
- If B1's selected budget saturates (`#acquired == |A|`), flag it and manually pick a lower
  target (e.g. 0.5); do not report a took-everything row as a win.

**Report a second headline for sensitivity:** since B1's B_star is one point, also report the
score `F1_test − μ · #mods_test/M` at each method's own val-selected lever, as a check that
the matched-budget conclusion isn't an artifact of the μ choice.

**Frontier for the appendix.** The full lever sweep is *retained* — the (mods, F1) curve per
method per pattern goes to an appendix plot with B_star marked. Headline number = single
matched-budget point; frontier = full trade-off shape. Both are honest; they answer different
questions.

> **⚠ STATUS — this rule is currently VIOLATED.** The IEMOCAP numbers on the table in §13 TODO-B
> had their levers argmaxed on **TEST** at μ = 0, not selected on val. This was accepted as a
> temporary shortcut pending a val-based selection; it inflates every design, and inflates the
> ones with more lever values most (qprior 9, B1' 10, **B2' 1**). **No test in this protocol is
> valid until the levers come from val** (§13 TODO-B1) — the val columns now exist, the baselines
> just need re-running. Do not put a TEST-selected table in the paper.

**What the comparison table looks like** (rows = methods, one cell = `(F1_test, #mods_test, GFLOPs_test)` at B_star):

| method | IEMOCAP @ B_star | CMI @ B_star | CZU-MHAD @ B_star |
|---|---|---|---|
| B1' static Shapley (defines B_star) | 0.xxx / r.rr / 0.zz | ... | ... |
| B2' all-available (ratio 1.0 ref) | ... | ... | ... |
| B_full all M (out-of-budget ref) | 0.6046 / 1.00 / 0.393 | ... | ... |
| dqn_basic / dqn_noexit / dqn_rlv2 | ... | ... | ... |
| cql_setcql / cql_setcql_noexit | ... | ... | ... |

The table is emitted **per availability pattern** (P0/P1/P3), with `B_star` **re-selected per
pattern** from B1' on val — the ceilings differ, so a shared budget across patterns would be
meaningless. Comparison is `methods × (dataset × pattern)`.

---

## 8. Measurements that forced these choices

### 8.1 The backbone memorises its train split
| IEMOCAP single-modality accuracy | TRAIN | TEST |
|---|---|---|
| video | **0.9989** | 0.3124 |
| audio | 0.9953 | 0.5666 |
| text | 0.9422 | 0.5765 |

Consequence: train-Shapley ranks whichever modality memorises hardest and puts **video first**
on IEMOCAP/MELD — video is at chance on test (0.31 for 4 classes). Fitting on TRAIN gives
**TRAIN F1 = 0.99, TEST F1 = 0.40–0.48**. The protocol is correct; the checkpoint is the
problem. Also explains the "tiny video grad-norms" — near-zero training loss, not a dead encoder.

**[DECISION NEEDED]** train-fit only (and report this as the finding), or train-fit **+** a
val-fit arm so the memorization cost is quantified rather than argued?

### 8.2 Feature dimension is not a cost model
| | video | audio | text | mocap_hand | mocap_head | mocap_rot |
|---|---|---|---|---|---|---|
| feature-dim proxy | 8.4 | **8.7** | 8.0 | 2.6 | 1.0 | 5.8 |
| **measured GFLOPs** | 1.22 | **1.28** | 1.17 | 1.00 | 1.00 | 1.04 |

The MBT projects every modality to 128-d before fusion, so 213× of feature-dim spread becomes
1.28× of compute. The proxy is ~7× wrong. This is why the feature-dim-weighted policy was dropped: it optimised a cost model
nearly unrelated to real compute.

### 8.3 Measured GFLOPs is ~uniform and exactly additive
`|gflops[S] − Σₘ cₘ| < 3e-5`, and max/min per-modality spread is **cmi 1.01×, czu 1.22×,
iemocap 1.28×**. A GFLOPs-weighted cost would therefore be within ~1.3× of a uniform cost on
these three datasets. Hence: **uniform cost (§5), GFLOPs reported only.**

### 8.4 Best-subset vs all-modality [CORRECTED TWICE — read the note]

**IEMOCAP fold 0, TEST, all val-Shapley order — internally consistent:**

| | macro-F1 | GFLOPs |
|---|---|---|
| all 6, **canonical order** (video first) | 0.5630 | 0.393 |
| all 6, **val-LOO order** (`B_full`, §3) | **0.6046** | 0.393 |
| static Shapley top-2 | 0.6161 | 0.143 |
| static Shapley top-3 | 0.6400 | 0.204 |
| **static Shapley top-4** | **0.6493** | 0.263 |

```
best-k − full-6  = 0.6493 − 0.6046 = +0.045
top-2  − full-6  = 0.6161 − 0.6046 = +0.012
```

**Two corrections were needed here; both were the same class of error.**
1. An earlier draft compared best-4 against full-6 under the *canonical* order (0.5630) and
   reported the gap as **+0.12**. The canonical order seeds the bottleneck with `video` (§8.5),
   so that overstated it.
2. The next draft fixed the order but compared a **3-fold mean** best-4 (0.6564) against a
   **fold-0** full-6 (0.6046) and reported **+0.052**. Mixed aggregation.

The fold-0-consistent gap is **+0.045**. All numbers above are fold 0; 3-fold means exist for the
top-k curve (k=2: 0.6420, k=4: 0.6564) but must not be mixed with fold-0 references.

**Status of the "match full-modality" criterion (§11):** on IEMOCAP fold 0, static Shapley top-2
exceeds `B_full` by +0.012 at ~36% of the compute. Unmeasured on folds 1–2 and on CMI / CZU-MHAD.

### 8.5 Order-dependence is real but ~unexploitable
Ordering drives 5–18% of softmax variance, but which ordering is best does **not** replicate
across samples (split-half ρ ≈ 0 on 5/7 datasets) and order-averaging beats any single ordering
on 15/19 dataset-depth cells. The residual is concentrated in the **first** modality
(η²_first ≈ 0.62–0.84 vs a 0.13 null) — the mean-aggregated bottleneck is seeded by whoever goes
first, and every later modality cross-attends to that seed. This is what licenses the set-lattice
collapse.

### 8.6 Per-sample subset choice: measured oracle vs a chance null
Per-sample oracle vs best static subset, IEMOCAP: naive headroom looks large (+0.28 at k=2), but a
chance null (shuffling each subset's correctness across samples, preserving marginals) scores
**1.000** — *higher* than the real oracle (0.959). Excess is **negative** at every k (−0.024 to
−0.090): subsets agree with each other more than chance, i.e. modalities are **redundant, not
complementary**. Hard samples are hard for everything.

**Scope of this measurement:** it covers per-sample subset choice under *fixed* availability, on
IEMOCAP only, conditioned on the ground-truth label. It does not measure choice conditioned on
availability (§9-§11), and it has not been run on CMI / CZU-MHAD.

Co-occurring observations, recorded without attributing cause: five policies, three cost models
and two state spaces all selected near-static behaviour; set-CQL reported std = 0.0000 across three
seeds with α = 2 selected in all 9 runs.

**Unrun follow-up:** a probe predicting, from the observable state after k modalities, whether
adding modality m flips the sample to correct. Its AUC bounds what any policy can extract from
the observable state. Not yet run on any dataset.

### 8.7 The backbone supports missingness architecturally — no retraining needed

Keep two claims apart:
(i) *does the backbone survive missing modalities?* — **structural**;
(ii) *does per-sample reactive choice beat the deployable static rule (B1')?* — **headroom**.

**(i) holds by construction, not by measurement.** pv3 trains with `--seq_random_order` +
`--prefix_supervision`: every forward supervises a random *prefix* of a random *permutation* —
which **is** a uniformly random subset of size 1..M. So the backbone has already been trained
under missingness, and the `max_modality_drop = 0.0` config field is misleading — the missingness
signal is delivered via prefix supervision. This is architectural and needs no experiment.

**(ii) is NOT established, and (i) does not imply it.** §8.6's redundancy result transfers when
availability is fixed: if per-sample choice cannot help when all M are present, restricting the
menu does not by itself create signal. The escape needs an **interaction effect** — the best
subset of a restricted `A` differing from the Shapley prefix of `A`. §10's ablation is what tests
it. Do not cite (i) as evidence for (ii).

**Full-modality references — measured, not asserted:**
| number | what it is | reproduce |
|---|---|---|
| 0.5339 ± 0.0263 | canonical-order full-6, 3-fold mean | `env.all_modality()` (order=None) |
| 0.5630 | canonical-order full-6, fold 0 | `env.all_modality(fold=0)` |
| **0.6046** | **val-LOO-order full-6, fold 0 — the baseline (§3)** | `env.all_modality(fold=0, order=cache['valloo_order'])` |
| 0.6184 | fold-0 test_f1 in `results_fold0.json` | **NOT reproducible here — see below** |

The 0.042 gap between canonical (0.5630) and val-LOO (0.6046) is the sequential-fusion **order
effect at full modality** (§8.5): the canonical order seeds the bottleneck with `video`.

**⚠ 0.6184 is not our number.** Measured with the val-LOO order on the same checkpoint, fold and
909-sample test split, we get **0.6046**. The 0.014 gap is a *configuration* difference:
`environment._load_model` sets `use_sparse_attn = False` so the cached value function is
deterministic, while the training run's eval left ProbSparse on (random key sampling via
`torch.rand`/`randint`). 0.6184 therefore describes a **different model configuration** than the
one the whole cache is built from. See open decision #8.

### 8.8 [STRUCTURAL] Under full availability the first pick CANNOT adapt

Measured in `archive/RL_basic.ipynb` §13–14 (IEMOCAP fold 0, val):

```
root state, sample   0 : [0,0,0,0,0,0, 1,1,1,1,1,1, .25,.25,.25,.25, 1.386, 0, 0, 1]
root state, sample 500 : [0,0,0,0,0,0, 1,1,1,1,1,1, .25,.25,.25,.25, 1.386, 0, 0, 1]
max deviation across all 793 samples : 0.0
distinct root states : 1 of 793          (at P_AVAIL=0.7: 60 of 793)
```

At the root under full availability the state is **byte-identical for every sample** — and it must
be: you have observed nothing. `mask` all-zero, `softmax` uniform, `entropy = log C`, `margin = 0`,
`depth = 0`, `available` all-ones. So `Q(root, ·)` is **one fixed vector**, `argmax` is **one fixed
action**, and every sample is forced to take the **same first modality**.

```
mean #modalities acquired    : 1.86
first pick is forced global   : −1.00
adaptive decisions / sample   :  0.86
samples stopping at depth ≤1  :  265/909  (29% make ZERO free choices)
```

This reframes the full-availability results: the deploy histogram put 823/909 samples on `text`
or `text→audio` **not** because the policy converged to the Shapley order, but because at the root
it had no choice but to be global — and 29% of samples then stopped before ever making a free
decision. Under full availability the only per-sample freedom the policy has is *when to stop*.

`available` is the one state component that varies per sample *before* any modality is acquired.
That is the stated reason it is the default formulation (§9), and it is a structural property, not
a performance claim.

**Sanity check:** `P_AVAIL = 1.0` must reproduce the degenerate behaviour exactly (root state → 1
distinct value). Implemented in `archive/RL_basic.ipynb` §13.

**Structural capability is not a result.** B1' also adapts to availability. §10a's ablation and
§7's matched-budget rule are what decide whether it pays.

---

## 9. Restricted action space — the default setting

Each sample arrives with a set of **available** modalities `A`. The policy may only acquire from
`A`. Everything else in §1-§8 is unchanged.

`A = all M` (full availability) is the **degenerate special case**, not a separate experiment. It
is also the case where the first pick cannot adapt (§8.8) — which is why restriction is the
default rather than an add-on.

**Availability conditions** — three, reported separately (ceilings differ; never average):

| condition | `A` | p(present) |
|---|---|---|
| **A0 — full** | every modality present | 1.0 |
| **A20 — 20% missing** | each modality present independently w.p. 0.8 | 0.8 |
| **A40 — 40% missing** | each modality present independently w.p. 0.6 | 0.6 |

Guarantee `|A| >= 1`. Availability draws use a fixed seed so every method sees the **same**
draws — the comparison is paired, not independent.

**Train-time:** draw `A ~ Bernoulli(0.7)` per episode, guaranteeing `|A| >= 1`, fixed seed —
between A20 and A40, so the policy sees the whole range and no test condition is trained on
exactly. A0 is then also a generalisation check, not just a degenerate case.

---

## 10. What changes in the policies — two things

Applies to all five policies of §4. Nothing else moves: transitions, reward, gamma=1, Double-Q,
the confidence-exit gate and the FQI loop are untouched.

**1. State gains availability** (`M + 1` extra features):
```
[ mask(M) | available(M) | softmax(C) | entropy | margin | depth/M | |A|/M ]
```
`mask` alone conflates *"I haven't acquired it yet"* with *"it isn't there"*. `|A|/M` because a
2-modality sample needs a different strategy than a 6-modality one.

**2. Action mask is intersected with availability** — now per **(node, sample)**:
```
legal_acquire(node, sample, m) = available[sample, m] AND m not acquired AND child exists
legal_stop(node)               = depth > 0
```

Also drop **unreachable** (node, sample) pairs from the loss: a node whose acquired set contains
a modality the sample doesn't have can never occur for that sample.

Reference implementation, verified end to end: `archive/RL_basic.ipynb` (§4 state, §6 mask +
reachability, §10 rollout). `P_AVAIL = 1.0` must reproduce the degenerate behaviour exactly —
that is the sanity check.

**Confidence-exit tau under restriction.** Fewer modalities cap achievable confidence, so one
global tau over-acquires when `A` is weak. Select tau **per pattern** on val (§7). A
decision-theoretic exit (stop when `E[R|stop] >= max_m E[R|acquire m]`) is the stronger option but
is not required for the first sweep.

### 10a. Ablation — is availability-aware *training* needed?

Run both arms of every policy:

| arm | what it is |
|---|---|
| **mask-only** | the policy trained **without** availability. No retraining. Intersect its action mask with `A` at test; re-select tau on val per pattern. |
| **aware** | the full §10 treatment: `available(M)` in state + trained under Bernoulli(0.7) availability. |

Report `aware − mask-only` per pattern at matched budget (§7). This isolates the value of
availability-aware **training** from the value of availability-aware **inference**.

**Relevant structural fact** (§5; verifiable in `rl_core.correctness`): `R(node, sample)` contains
no `A` term — availability changes which nodes are *reachable*, not what they are *worth*. The
availability-dependence therefore enters only through the action mask, which is deterministic
given `A`. What this implies for the two arms is what the ablation measures; it is not settled
here.

---

## 11. Success criterion and evaluation

**The goal: match full-modality performance without consuming all the modalities.**

**The reference depends on the pattern — do not use one bar everywhere.** `B_full` (all M) is
only *achievable* when everything is present. Under P1/P3 some modalities physically do not
exist, so scoring against `B_full` measures the missingness, not the policy.

| pattern | reference | success |
|---|---|---|
| **P0** (all present) | `B_full` — all M, val-LOO order | `F1 ≥ F1(B_full)` **and** `#acq < M` |
| **P1 / P3** (restricted) | `B2'` — all of `A` | `F1 ≥ F1(B2')` **and** `#acq < \|A\|` |

Both reduce to the same sentence: **match what you'd get by consuming everything you had, while
consuming less of it.** `B_full` is still reported under every pattern as a fixed anchor, but it is
out-of-budget when `A ≠ all` and must not be used to declare success or failure there.

Two levels, both reported:

1. **Match the consume-everything reference at lower cost.** Measured status: on IEMOCAP fold 0,
   static Shapley top-2 scores 0.6161 vs `B_full` 0.6046 (§8.4) — a baseline satisfies this on
   that fold, by +0.012. Unmeasured on folds 1-2 and on CMI / CZU-MHAD.
2. **Beat B1' at matched budget** (§7's `B_star`), per pattern.

Both are reported. Neither is weighted above the other here.

**Per cell** (dataset × fold × seed × pattern):

| column | meaning |
|---|---|
| macro-F1 | TEST, at the val-selected lever matched to `B_star` |
| mean #acquired | out of `\|A\|` |
| **acquired / `\|A\|`** | **the budget axis** — §7 matches on THIS, not raw #modalities, which is not comparable across patterns |
| mean GFLOPs/sample | cumulative, at the stop node |
| vs reference | `F1 − F1(B_full)` at P0; `F1 − F1(B2')` at P1/P3 |
| collapse flag | `#acquired == \|A\|` → took everything available; not a result |

**Aggregation:** mean ± std across 3 seeds, averaged over 3 folds; one row per method per pattern
per dataset. Frontier `(acquired/|A|, F1)` retained for an appendix plot with `B_star` marked.

**Power.** Noise floors per dataset are in §1: IEMOCAP ±0.020, CMI ±0.018, CZU-MHAD ±0.036.
Report the floor next to every effect size so the reader can judge resolvability. CZU is deferred
for this reason (§1).

---

## 12. Required diagnostics — four artifacts per design

Every design (5 RL policies + B1'/B2'/B3'/B_full) emits the same four files. Naming:
`results/<design>_<dataset>_{point.csv,sweep.csv}` and `figures/<design>_<dataset>_{freq,first}.png`.

Purpose: catch **collapse** and **degenerate selection** before any headline is read. A design that
scores well while always picking the same modality, or while taking everything available, is not a
result.

### 12.1 `<design>_<dataset>_point.csv` — the headline

One row per `(fold, seed, condition)`, at the `B_star`-matched lever (§7).

| column | notes |
|---|---|
| `dataset, design, fold, seed, condition` | condition ∈ {A0, A20, A40} |
| `macro_f1` | TEST |
| `mean_acquired`, `mean_available`, `ratio` | `ratio = acquired/\|A\|` — the budget axis |
| `mean_gflops` | cumulative at the stop node |
| `f1_vs_reference` | vs `B_full` at A0; vs `B2'` at A20/A40 (§11) |
| `collapse_takeall` | `mean_acquired == mean_available` |
| `collapse_single_first` | one modality is the first pick for >95% of samples (see 12.3) |

**Required rows for every design**: A0, A20, A40 × B1' (Shapley+conf-exit), B3' (random order),
B_full. These three baselines are reported in every point.csv, not in a separate table.

### 12.2 `<design>_<dataset>_sweep.csv` — the frontier

One row per `(fold, seed, condition, lever)` over the design's full lever sweep (τ / δ / λ).
Columns: as 12.1 plus `lever_name`, `lever_value`, `is_bstar`. This is the appendix frontier and
the evidence that `B_star` sits where §7 says.

### 12.3 `<design>_<dataset>_freq.png` — selection frequency

Per condition (3 panels), the **fraction of samples that acquired each modality at any depth**,
with the ±std across seeds.

Reads as a collapse detector:
- a bar at ~1.0 for one modality and ~0 elsewhere → the policy is a constant
- bars tracking `p(present)` (1.0 / 0.8 / 0.6) → the policy takes whatever is there, i.e. no
  selection
- variation *across* conditions → selection responds to availability

Also plot B1' on the same axes: B1' takes modalities in a fixed Shapley order, so its bars are the
null the RL must differ from.

### 12.4 `<design>_<dataset>_first.png` — first-pick distribution

Per condition (3 panels), a histogram of the **first modality acquired**, split by whether the
top-Shapley modality was available:

| panel row | why |
|---|---|
| all samples | overall first-pick distribution |
| top-Shapley **present** | the easy case — every method should mostly take it |
| top-Shapley **absent** | **the informative rows** — this is where B1' is forced onto a fallback and the policy has room to differ |

**A0 is the control.** With full availability the root state is identical for every sample (§8.8),
so the first pick is one fixed action and this histogram **must** be a single bar. If it isn't, the
implementation is leaking per-sample information into the root state. If it *is* a single bar at
A0 and spreads at A20/A40, the availability conditioning works.

Report `#distinct first picks` per condition in the point.csv as `n_first_distinct`.

---

## 13. Open decisions

**Decided**
1. ~~`dqn_policy_noexitweighted`~~ — dropped (§4, §8.2).
2. Comparison — **matched budget at `B_star`**, mu = 0.20; frontier to the appendix (§7).
3. Restricted action space is the **default**; P0 is the degenerate case (§9, §8.8).
4. `MAX_DEPTH = M`, not 4 (§1) — **no overrides**. Requires the one-time cache rebuild
   (`rebuild_caches.py`); `get_cache` refuses to build implicitly and errors instead.
5. Cost is **uniform**; GFLOPs reported only (§5, §8.3).
6. `B_full` uses the **val-LOO order** (0.6046), never the canonical order (0.5630) (§3, §8.4).

7. **Fit split — DECIDED: val-fit only** (§2). Train-fit is not an arm; the train cache
   cannot support policy learning (§8.1). Out-of-fold is the future option, not a current one.

7b. **Policy — DECIDED: `dqn_policy_qprior`** (conditional advice + per-iteration availability
   resampling), §4. The other five are archived (§4a).

**Open**
8. **`use_sparse_attn`** — the trained config has it **True** (ProbSparse, random key sampling via
   `torch.rand`/`randint`), but `environment._load_model` disables it so the cached value function
   is deterministic. Trade-off: a cache built from a stochastic model would return different values
   for the same (node, sample) on re-read; with it disabled, our numbers describe a different
   configuration than `results_fold*.json` (0.6046 vs 0.6184). The size of the gap on other folds
   and datasets is unmeasured.
9. **mu = 0.20** — confirmed on IEMOCAP only; re-check on CMI / CZU-MHAD before freezing (§7).

---

### TODO-A — run the remaining datasets

Only **IEMOCAP** is complete (3 folds × seed 0; seeds 1–2 not yet swept).

| dataset | cache | baselines | qprior | note |
|---|---|---|---|---|
| **IEMOCAP** | ✅ d6 | ✅ | ✅ | 3 folds × seed 0 only — **the 3-seed sweep is still owed** |
| **CMI** | ✅ d7 (~15 h, reusable) | ⚠ stale (pre-dates the current `run_design`) | ❌ never run | needs the chunked-forward fix — **now in** (§4). This is the next run. |
| **CZU-MHAD** | ❌ | ❌ | ❌ | **deferred**: noise floor ±0.036 — check it can resolve the effect before spending the GPU |
| **MM-Fi** | ⚠ partial | ❌ | ❌ | **was cancelled, and the reason has not gone away** — see below |

⚠ **MM-Fi is not just "not run yet".** Its val split is **N = 27 for 27 classes** — ~1 sample per
class, noise floor **±0.107**, which is 5× the largest effect we are trying to measure (+0.019).
No amount of GPU fixes this; the fit split cannot support policy learning *or* lever selection.
Running it produces a number, not a result. If MM-Fi must be in the paper, the honest options are
(a) merge folds / re-split to get a usable fit set, or (b) report it as backbone-only with no RL
arm. **Decide which before launching** — this was already caught once *after* the launch.

### TODO-B — selection hygiene (the current results are not clean)

**B1. Persist the VAL sweep, always.** *(implemented for new runs — verify, don't re-implement.)*
`run_design.py` now writes `macro_f1_val` / `ratio_val` alongside the test columns for both
qprior (val rollout at the same `δ`) and the baselines (val rollout at the same `τ`), so `B_star`
is picked on val without re-rolling. Two gaps remain:
  - `archive/results/baselines_iemocap_sweep.csv` **pre-dates this** and has no val columns — the
    baselines must be re-run before any val-selected table is complete.
  - `run_design.py` falls back to TEST selection with only a printed `WARNING:` if the val columns
    are absent. That fallback should be a **hard error**, not a warning — it is exactly the
    failure that produced the numbers below.

**B2. Hold out ~5% of TEST for deployment-time tuning.** The levers (§7) are currently argmaxed on
the **full test set**, which is selection on the test set. Proposed: carve a fixed
`test_tune` (~5%, stratified by `y`, seeded, frozen once and reused across all designs) for
choosing `δ` / `τ`, and report on the remaining ~95%. Requirements:
  - stratified — 5% of IEMOCAP is ~45 samples, and an unstratified draw will miss classes entirely;
  - **frozen and shared** across designs, or the comparison re-acquires the same bias;
  - check the tune split is large enough to *rank* levers at all — with ~45 samples the F1 s.e. is
    ~±0.07, i.e. wider than the whole `δ` frontier. **If it cannot rank, this does not work and
    the lever must come from val instead** (B1). Measure before committing.

**Why this is load-bearing — the current IEMOCAP table (3 folds × seed 0, levers argmaxed on TEST,
μ = 0):**

| condition | qprior | B1' | B2' | qprior − B1' | qprior − B2' |
|---|---|---|---|---|---|
| A0 | **0.6283** ± 0.0134 @ util 0.88 | 0.6090 ± 0.0185 @ 0.26 | 0.6271 ± 0.0103 @ 1.00 | +0.0192 (t=3.86) | +0.0012 (t=0.34) |
| A20 | 0.5931 ± 0.0149 @ 0.80 | 0.5871 ± 0.0198 @ 0.34 | **0.5985** ± 0.0214 @ 1.00 | +0.0060 (t=1.23) | **−0.0055** (t=−1.34) |
| A40 | **0.5537** ± 0.0330 @ 0.74 | 0.5528 ± 0.0153 @ 0.40 | 0.5408 ± 0.0175 @ 1.00 | +0.0009 (t=0.08) | +0.0129 (t=1.42) |

**Nothing here is significant** (df=2 needs |t| > 4.30). Read honestly:
- **qprior does not beat B2' on F1.** A20 is a *loss on all three folds*; A0 is a rounding error.
- At μ = 0 the `δ` sweep drives utilisation to 0.88 — **qprior chooses to become B2'**, and then
  ties it, as it must. There is no adaptive story at that operating point.
- The selection bias is asymmetric: qprior gets 9 shots at `δ`, B1' gets 10 at `τ`, **B2' gets one.**
  B2' is the only unfavoured design in the table and it still ties or wins at 2 of 3 conditions.
- 3 folds × **1 seed** is why every t is starved. The 3-seed infrastructure exists (TODO-A).

The defensible claim is therefore **not** "qprior beats B2' on F1" — it is the *frontier*
(matching B2' at 0.74–0.88 utilisation). Which means the μ = 0 point is the wrong headline, and
§7's matched-budget rule is the right one.

### TODO-C — calibrate lambda: is the early-exit incentive set right?

The reward already pays for stopping early *while correct*: `R = 1[correct] x 1/freq(y)` is
collected at the stop node, and each acquire costs `-lambda/|A|`. An agent correct at depth 1 keeps
`R - lambda/|A|`; one correct at depth 4 keeps `R - 4*lambda/|A|`. Early-and-correct is strictly
better by construction — no shaping needed.

Under the `|A|`-normalised cost (§5), `lambda` is the price of consuming **everything available**,
so it is directly comparable to an F1 gain:

| | value |
|---|---|
| **current (`dqn_policy_qprior.LAM`)** | **0.15** — "consuming all of A costs 0.15 F1" |
| F1 gain from 1 modality -> all of A (IEMOCAP) | **~0.11** |
| => break-even lambda | **~0.11 - 0.20** |

`lambda = 0.15` sits inside the break-even band — that is why it was raised from the old 0.05
(which was ~2-4x too small: consuming everything cost 0.05 while buying ~0.11 F1, so the policy
always took everything). **It is calibrated on IEMOCAP only.** CMI has M=7 and 18 classes; the
gain there is unmeasured, so `LAM` may be wrong for it.

**This is the direct fix for TODO-B.** The `δ` lever is a *post-hoc offset on a Q trained at
lambda = 0.15* — any `δ != 0` deploys a policy inconsistent with its own Bellman targets, and at
μ = 0 the sweep exploits exactly that (`δ < 0` suppresses the learned STOP to keep acquiring,
buying F1 the cost term was set up to forbid). **Retrain at a lower lambda and report `δ = 0`**:
then the operating point is one the policy actually learned, and the qprior-vs-B1' lever asymmetry
(post-hoc offset vs B1's *native* exit rule) disappears.

To run:
1. Measure the actual F1-per-modality gain per dataset (cache-only) — ~0.11 for k=1->4 on
   IEMOCAP, unmeasured on CMI / CZU-MHAD.
2. Sweep `lambda in {0.05, 0.10, 0.15, 0.20, 0.30}`, plot `#acquired/|A|` vs lambda at `δ = 0`.
   Confirm the monotone response (higher lambda -> earlier exit) and locate the knee.
3. Check the degenerate ends: `lambda = 0` and large `lambda`, to bound the response.
4. Decide: lambda swept into the §7 frontier, or fixed per dataset at the knee. **Sweeping lambda
   instead of `δ` is the principled version** — every point on the frontier is then a policy that
   is self-consistent, which `δ` points are not. Cost: one FQI fit per lambda (~12 min) versus a
   free `δ` re-roll, so budget ~1 h per (fold, seed).

Note this interacts with §7: `B_star` comes from B1's val frontier, so if lambda never binds,
every learned-stop method saturates its budget and the matched-budget comparison degenerates.

---

## 14. Run

### The one command

```bash
cd /home/group/maestro_visual/RL_AAAI_final
./runs.sh <dataset> <gpu>          # e.g. ./runs.sh iemocap 7
```

`runs.sh` is the whole pipeline for one dataset. It activates `maestro_energy`, pins
`CUDA_VISIBLE_DEVICES`, sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (shared box —
avoids fragmentation OOM), then:

- **Stage 1 — cache.** Runs `rebuild_caches.py <ds>` at `MAX_DEPTH = M` (§1). **Auto-skipped if the
  cache already exists**, so re-running is cheap. IEMOCAP ~15–35 min; **CMI ~15 h** (depth 7,
  13,700 nodes — build it once and never again).
- **Stage 2 — designs.** `run_design.py --design D` for `D` in `(baselines, dqn_qprior)`,
  3 folds × 3 seeds × A0/A20/A40. A failing design is logged and the next one still runs.

Two datasets in parallel on separate GPUs is the normal mode — pick free GPUs first
(`nvidia-smi`), since this is a shared machine and a busy GPU will OOM the CMI job:

```bash
setsid nohup ./runs.sh iemocap 1 > logs/iemocap.log 2>&1 < /dev/null &
setsid nohup ./runs.sh cmi     3 > logs/cmi.log     2>&1 < /dev/null &
```

Overrides: `FOLDS=0 SEEDS=0 ./runs.sh iemocap 7` for a smoke test; `CONDA_ENV=...` for the env.

### The pieces, if you need them separately

```bash
python rebuild_caches.py iemocap                    # val + test only — the train cache is NEVER read (§2)
python run_design.py --design dqn_qprior --dataset iemocap --folds 0,1,2 --seeds 0,1,2 --mu 0.20
python run_design.py --design baselines  --dataset iemocap    # B1'/B2'/B3'/B_full; seed is forced to 0
python dqn_policy_qprior.py                         # single fold0/A0 sanity run, prints F1 / #acq / GFLOPs
```

`--mu` is the §7 budget penalty used to mark `is_bstar` (default 0.20). `--mu 0` = BEST-F1 selection.
There is **no `--fit` flag: the fit split is val, hard-coded** (§2). `get_cache` **refuses to build
implicitly** — it raises `FileNotFoundError` telling you to run `rebuild_caches.py`, so a missing
cache can never silently become a 15-hour surprise inside a sweep.

### What you get (§12)

| artifact | content |
|---|---|
| `results/<design>_<ds>_point.csv` | headline — the `is_bstar` row per (fold, seed, condition) |
| `results/<design>_<ds>_sweep.csv` | full lever frontier, **written incrementally** (append per cell) |
| `results/<design>_<ds>_sweep.csv.prev` | the previous sweep, auto-renamed at start-of-run |
| `figures/<design>_<ds>_freq.png` | modality-selection frequency (collapse detector) |
| `figures/<design>_<ds>_first.png` | first-pick distribution (at A0 the mode MUST sit at the ε-ceiling, §8.8) |

⚠ **`.prev` will surprise you.** `run_design.py` renames an existing `sweep.csv` → `sweep.csv.prev`
the moment it starts writing, so a crashed run cannot half-overwrite good results. If a sweep you
were reading "vanishes", a re-run reached that design — it is at `.prev`, not lost.

Sweep rows carry both `macro_f1`/`ratio` (TEST) and `macro_f1_val`/`ratio_val` (VAL), so `B_star`
can be selected on val without re-rolling (§7). Progress prints per cell (`fold0 seed0 A0 done | B_star: ...`),
so a long run is watchable and survives a crash.

### Runtime

IEMOCAP: ~12 min per `(fold, seed)` for qprior; the net is **cached per `(fold, seed)`** in `_NETS`
because it does not depend on the test condition (availability is drawn *inside* training; `av_te`
is eval-only), so training is 3 per fold rather than 9. Without that cache the sweep was 5 h.
