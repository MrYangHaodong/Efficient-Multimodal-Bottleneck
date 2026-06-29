"""#2: PROVE whether redundancy HELPS or HARMS as a modality-selection criterion.
Matched-compute 2-bucket policy: pick the most-"exitable" fraction p of samples to run on
1 modality (val-LOO top-1), the rest on all M; choose WHICH p by each signal. Measure test
acc vs mean #modalities. If the R curve is BELOW patience, R's info harms selection.
Signals (higher=more exitable): oracle(-exit_k), patience(-patience_k), R(+R), maxprob1(+msp1),
R+patience(rank-sum), random. Reads results_3p3/lsmi_exit/arrays_<ds>.npz."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R_ = os.path.dirname(os.path.abspath(__file__))
DSETS = ['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei']


def rankpct(a):
    o = a.argsort(kind='mergesort'); r = np.empty(len(a)); r[o] = np.arange(len(a)) / max(1, len(a) - 1); return r


def curve(exitability, correct, gpk, nstep=21):
    """2-bucket: top-p exitable -> k=1 (correct[:,0]); rest -> k=M (correct[:,-1])."""
    N, M = correct.shape; order = np.argsort(-exitability)            # most exitable first
    c1 = correct[:, 0]; cM = correct[:, -1]; xs, ys = [], []
    for p in np.linspace(0, 1, nstep):
        e = int(round(p * N)); early = order[:e]; rest = order[e:]
        acc = (c1[early].sum() + cM[rest].sum()) / N
        meank = (e * 1 + (N - e) * M) / N
        gf = (e * gpk[0] + (N - e) * gpk[-1]) / N
        xs.append(gf); ys.append(acc)
    return np.array(xs), np.array(ys)


SIG = {'oracle': ('k', ':'), 'patience': ('tab:purple', '-'), 'R (redundancy)': ('tab:red', '-'),
       'R+patience': ('tab:green', '-'), 'maxprob@1': ('tab:orange', '--'), 'random': ('tab:gray', '--')}
fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
summary = {}
for col, ds in enumerate(DSETS):
    f = os.path.join(R_, 'lsmi_exit', f'arrays_{ds}.npz')
    if not os.path.isfile(f):
        print(f'{ds}: no arrays'); continue
    z = np.load(f); cor = z['correct']; gpk = z['gflops_per_k']; M = cor.shape[1]
    R, pat, ek, msp = z['R'], z['patience_k'], z['exit_k'], z['msp1']
    sigs = {'oracle': -ek, 'patience': -pat, 'R (redundancy)': R, 'maxprob@1': msp,
            'R+patience': rankpct(R) + rankpct(-pat), 'random': np.random.RandomState(0).rand(len(R))}
    ax = axes[col]; rows = {}
    for nm, (c, ls) in SIG.items():
        x, y = curve(sigs[nm], cor, gpk); ax.plot(x, y, ls, color=c, lw=2.2 if nm in ('R (redundancy)', 'patience', 'R+patience') else 1.5, label=nm)
        rows[nm] = (x, y)
    # acc at matched compute = midpoint GFLOPs (interp each onto common grid)
    xfull, x1 = gpk[-1], gpk[0]; grid = np.linspace(x1, xfull, 50)
    def at(nm): xx, yy = rows[nm]; o = np.argsort(xx); return np.interp(grid, xx[o], yy[o])
    aR, aPat, aRP, aOr = at('R (redundancy)'), at('patience'), at('R+patience'), at('oracle')
    summary[ds] = {'mean_dAcc_R_minus_patience': float(np.mean(aR - aPat)),
                   'mean_dAcc_Rpat_minus_patience': float(np.mean(aRP - aPat)),
                   'mean_dAcc_oracle_minus_patience': float(np.mean(aOr - aPat))}
    ax.set_xscale('log'); ax.set_xlabel('mean GFLOPs/sample (log)'); ax.set_ylabel('test acc'); ax.set_title(ds); ax.grid(alpha=.25, which='both')
    if col == 0: ax.legend(fontsize=7.5, loc='lower right')
fig.suptitle('#2 Modality-selection POLICY test (seqA_ps): run most-"exitable" p% on 1 modality, rest on all M; choose who by each signal. '
             'Does R beat patience? (R/R+patience use ORACLE redundancy = upper bound.)', fontsize=11)
fig.subplots_adjust(left=0.04, right=0.997, bottom=0.13, top=0.86, wspace=0.26)
o = os.path.join(R_, 'lsmi_exit', 'policy_R_vs_patience.png'); fig.savefig(o, dpi=140); print('saved ->', o)
print('\n=== mean Δacc over matched-compute range (vs patience policy) ===')
print(f"{'dataset':9s} {'R−patience':>12s} {'R+pat−patience':>16s} {'oracle−patience':>16s}")
for ds, s in summary.items():
    print(f"{ds:9s} {100*s['mean_dAcc_R_minus_patience']:>+11.2f}pp {100*s['mean_dAcc_Rpat_minus_patience']:>+15.2f}pp {100*s['mean_dAcc_oracle_minus_patience']:>+15.2f}pp")
json.dump(summary, open(os.path.join(R_, 'lsmi_exit', 'policy_summary.json'), 'w'), indent=2)
