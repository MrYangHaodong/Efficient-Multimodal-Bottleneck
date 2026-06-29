#!/usr/bin/env python
"""(1) Judge whether PATIENCE and BOTTLENECK-saturation exit are correct on DIFFERENT
samples (per-sample complementarity at matched compute). (2) If so, COMBINE them and
test if the combination beats both. Operates on dumped per-sample signals (no model)."""
import os, numpy as np
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(R, 'seqA_exit_signals_raw_4ds.npz'))
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads']
TAG = {'iemocap': 'M=6', 'daliahar': 'M=5', 'pamap2': 'M=4', 'dsads': 'M=5'}


def rank01(S):
    """percentile-rank each entry of S[N,M] within the pooled value distribution -> [0,1)."""
    flat = S.reshape(-1); order = flat.argsort(kind='mergesort')
    r = np.empty_like(order, dtype=float); r[order] = np.arange(len(order)) / len(order)
    return r.reshape(S.shape)


def simulate(score, preds, labels, tau, ge=True):
    N, M = score.shape; ek = np.full(N, M)
    for k in range(M):
        cond = (score[:, k] >= tau) if ge else (score[:, k] <= tau)
        ek = np.where((ek == M) & cond, k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    return ek, (pe == labels)


def curve(score, preds, labels, taus, ge=True):
    out = []
    for t in taus:
        ek, corr = simulate(score, preds, labels, t, ge)
        out.append((ek.mean(), corr.mean(), f1_score(labels, preds[np.arange(len(labels)), ek - 1], average='macro')))
    return np.array(out)


def tau_for_meanK(score, preds, labels, target, taus, ge=True):
    mks = np.array([simulate(score, preds, labels, t, ge)[0].mean() for t in taus])
    return taus[np.argmin(np.abs(mks - target))]


def gflops(ds, mk):
    M = int(Z[f'{ds}_M']); g = Z[f'{ds}_gflops_per_k']
    return np.interp(mk, np.arange(1, M + 1), g)


print('=' * 78)
print('(1) PER-SAMPLE COMPLEMENTARITY at matched compute (patience vs bottleneck)')
print('=' * 78)
comp = {}
for ds in DS:
    labels = Z[f'{ds}_labels']; preds = Z[f'{ds}_preds']; msp = Z[f'{ds}_msp']
    bn = Z[f'{ds}_bn_sim']; runlen = Z[f'{ds}_runlen']; M = int(Z[f'{ds}_M'])
    pat = runlen + msp                                  # patience evidence
    taus_p = np.linspace(1.0, M + 1.0, 200); taus_b = np.linspace(-1.0, 1.0, 200)
    # matched budgets in the range both schemes can reach (bottleneck floor ~2)
    budgets = [b for b in [2.3, 0.5 * (2 + M), M - 0.4] if 2.0 < b < M]
    rows = []
    for B in budgets:
        tp = tau_for_meanK(pat, preds, labels, B, taus_p, ge=True)
        tb = tau_for_meanK(bn, preds, labels, B, taus_b, ge=True)
        ekp, cp = simulate(pat, preds, labels, tp, True)
        ekb, cb = simulate(bn, preds, labels, tb, True)
        both = (cp & cb).mean(); onlyp = (cp & ~cb).mean(); onlyb = (~cp & cb).mean(); none = (~cp & ~cb).mean()
        oracle = (cp | cb).mean()
        rows.append((B, ekp.mean(), ekb.mean(), cp.mean(), cb.mean(), both, onlyp, onlyb, none, oracle))
    comp[ds] = rows
    print(f'\n{ds} ({TAG[ds]}):')
    print(f"  {'budget':>6} {'mkP':>5} {'mkB':>5} | {'accP':>6} {'accB':>6} | {'both✓':>6} {'onlyP':>6} {'onlyB':>6} {'both✗':>6} | {'oracle':>6} {'head':>6}")
    for (B, mkp, mkb, ap, ab, bo, op, ob, no, orc) in rows:
        head = orc - max(ap, ab)
        print(f"  {B:6.1f} {mkp:5.2f} {mkb:5.2f} | {ap:6.3f} {ab:6.3f} | {bo:6.3f} {op:6.3f} {ob:6.3f} {no:6.3f} | {orc:6.3f} {head:+6.3f}")

print('\n' + '=' * 78)
print('(2) COMBINED schemes vs individuals + oracle upper bound')
print('=' * 78)
COMB = {}
for ds in DS:
    labels = Z[f'{ds}_labels']; preds = Z[f'{ds}_preds']; msp = Z[f'{ds}_msp']
    bn = Z[f'{ds}_bn_sim']; runlen = Z[f'{ds}_runlen']; M = int(Z[f'{ds}_M'])
    pat = runlen + msp
    rp = rank01(pat); rb = rank01(bn)                   # rank-normalized readiness in [0,1]
    taus_p = np.linspace(1.0, M + 1.0, 60); taus_b = np.linspace(-1.0, 1.0, 60); taus01 = np.linspace(0.0, 1.0, 60)
    cur = {}
    cur['patience'] = curve(pat, preds, labels, taus_p, True)
    cur['bottleneck'] = curve(bn, preds, labels, taus_b, True)
    cur['comb_sum'] = curve(0.5 * rp + 0.5 * rb, preds, labels, taus01, True)
    cur['comb_and'] = curve(np.minimum(rp, rb), preds, labels, taus01, True)
    cur['comb_or'] = curve(np.maximum(rp, rb), preds, labels, taus01, True)
    # oracle-either upper bound: at matched meanK B, correct if patience OR bottleneck correct
    orc = []
    for B in np.linspace(2.05, M, 18):
        tp = tau_for_meanK(pat, preds, labels, B, taus_p, True); tb = tau_for_meanK(bn, preds, labels, B, taus_b, True)
        _, cp = simulate(pat, preds, labels, tp, True); _, cb = simulate(bn, preds, labels, tb, True)
        orc.append((B, (cp | cb).mean(), 0.0))
    cur['oracle'] = np.array(orc)
    COMB[ds] = cur

# ---- plot ----
STYLE = {'patience': ('Patience', 'tab:purple', '-'), 'bottleneck': ('Bottleneck sat.', 'tab:cyan', '--'),
         'comb_sum': ('Combined (rank-sum)', 'tab:red', '-'), 'comb_and': ('Combined (AND)', 'tab:green', '-'),
         'oracle': ('Oracle (either-right)', 'k', ':')}
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for col, ds in enumerate(DS):
    ax = axes[col]
    for nm, (lab, c, ls) in STYLE.items():
        cur = COMB[ds][nm]; x = gflops(ds, cur[:, 0]); y = cur[:, 1]; o = np.argsort(x)
        lw = 2.6 if nm.startswith('comb') else (1.6 if nm == 'oracle' else 2.0)
        ax.plot(x[o], y[o], ls, color=c, lw=lw, label=lab, zorder=6 if nm.startswith('comb') else 4)
    ax.set_xscale('log'); ax.set_xlabel('Mean GFLOPs / sample (log)'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{ds} ({TAG[ds]}) [cross-subject]', fontsize=10); ax.grid(alpha=0.25, which='both')
    if col == 0:
        ax.legend(fontsize=8, loc='lower right')
fig.suptitle('Combining PATIENCE + BOTTLENECK-saturation early exit for seqA (cross-subject). Combined (rank-sum / AND) vs each alone; dotted = oracle upper bound.', fontsize=11)
fig.subplots_adjust(left=0.04, right=0.997, bottom=0.13, top=0.88, wspace=0.24)
op = os.path.join(R, 'gflops_combine_pat_bn_4ds.png'); fig.savefig(op, dpi=140); print('saved ->', op)

# combined vs best-individual over shared GFLOPs
print(f"\n{'dataset':9s} {'combined':12s} {'mean Δacc vs best(pat,bn)':>26s}")
for ds in DS:
    pa = COMB[ds]['patience']; bnc = COMB[ds]['bottleneck']
    for nm in ['comb_sum', 'comb_and']:
        cc = COMB[ds][nm]
        xs = [gflops(ds, c[:, 0]) for c in (cc, pa, bnc)]
        lo = max(x.min() for x in xs); hi = min(x.max() for x in xs); g = np.linspace(lo, hi, 40)
        def itp(c):
            x = gflops(ds, c[:, 0]); o = np.argsort(x); return np.interp(g, x[o], c[o, 1])
        d = itp(cc) - np.maximum(itp(pa), itp(bnc))
        print(f"{ds:9s} {nm:12s} {100*d.mean():+23.2f}pp")
