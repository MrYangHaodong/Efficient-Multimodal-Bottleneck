"""Can combining ENTROPY (per-prefix confidence) + PATIENCE (run-length stability) beat either
alone for modality early-exit? Reads exit_raw_<ds>.npz (per-prefix H, run-length rl, maxprob, preds,
y) for 5 seqA_ps datasets. Tests:
  - pure entropy            : exit when H_k <= tau
  - pure patience           : exit when (rl_k + maxprob_k) >= tau   (current continuous patience)
  - combined score (fixed)  : S_k = rl_k + lambda*(1 - H_k/logC)    (1D sweep; deployable)
  - AND-gate Pareto (UB)    : exit when rl_k>=p AND H_k<=e          (2D grid -> Pareto frontier)
  - OR-gate Pareto (UB)     : exit when rl_k>=p OR  H_k<=e
AUC compared over the common GFLOPs range. UB curves are upper bounds (post-hoc threshold pick),
not deployable; the fixed score is the honest deployable combination."""
import os, json
import numpy as np
R_ = os.path.dirname(os.path.abspath(__file__))
DS = ['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei']


def exit_curve(score, preds, y, taus, ge, M, gpk):
    N = score.shape[0]; out = []
    for t in taus:
        ek = np.full(N, M)
        for k in range(M):
            cond = (score[:, k] >= t) if ge else (score[:, k] <= t)
            ek = np.where((ek == M) & cond, k + 1, ek)
        pe = preds[np.arange(N), ek - 1]
        out.append((np.interp(ek, np.arange(1, M + 1), gpk).mean(), (pe == y).mean()))
    return np.array(out)


def gate_pareto(rl, H, preds, y, M, gpk, mode):
    N = len(y); pts = []
    egrid = np.quantile(H, np.linspace(0, 1, 30))
    for p in range(1, M + 1):
        for e in egrid:
            ek = np.full(N, M)
            for k in range(M):
                a = rl[:, k] >= p; b = H[:, k] <= e
                cond = (a & b) if mode == 'and' else (a | b)
                ek = np.where((ek == M) & cond, k + 1, ek)
            pe = preds[np.arange(N), ek - 1]
            pts.append((np.interp(ek, np.arange(1, M + 1), gpk).mean(), (pe == y).mean()))
    return np.array(pts)


def auc_on(curve, grid):
    c = curve[np.argsort(curve[:, 0])]; return np.interp(grid, c[:, 0], c[:, 1]).mean()


def pareto_envelope(pts, grid):
    """max acc achievable at compute <= each grid point (monotone upper-left frontier)."""
    c = pts[np.argsort(pts[:, 0])]; env = np.maximum.accumulate(c[:, 1]); return np.interp(grid, c[:, 0], env)


summary = {}
for ds in DS:
    z = np.load(os.path.join(R_, 'lsmi_exit', f'exit_raw_{ds}.npz'))
    H = z['H']; rl = z['rl'].astype(float); mp = z['maxprob']; preds = z['preds']; y = z['y']; M = int(z['M']); gpk = z['gpk']
    C = preds.max() + 1; logC = float(np.log(C))
    conf = 1 - H / logC                      # entropy-confidence in [0,1]
    cur_ent = exit_curve(H, preds, y, np.quantile(H, np.linspace(0, 1, 30)), False, M, gpk)
    cur_pat = exit_curve(rl + mp, preds, y, np.linspace(1, M + 1, 30), True, M, gpk)
    # fixed deployable combined score, a few lambda
    best_lam = None; best_combo = None; best_auc = -1
    grid = np.linspace(gpk[0], gpk[-1], 60)
    for lam in [0.5, 1.0, 2.0, 3.0, 5.0]:
        cur = exit_curve(rl + lam * conf, preds, y, np.linspace(1, M + 1 + 5 * lam, 40), True, M, gpk)
        a = auc_on(cur, grid)
        if a > best_auc: best_auc, best_lam, best_combo = a, lam, cur
    aE = auc_on(cur_ent, grid); aP = auc_on(cur_pat, grid)
    envAND = pareto_envelope(gate_pareto(rl, H, preds, y, M, gpk, 'and'), grid).mean()
    envOR = pareto_envelope(gate_pareto(rl, H, preds, y, M, gpk, 'or'), grid).mean()
    base = max(aE, aP)
    summary[ds] = dict(M=M, entropy=aE, patience=aP, combo_fixed=best_auc, combo_lam=best_lam,
                       AND_UB=float(envAND), OR_UB=float(envOR), best_pure='patience' if aP >= aE else 'entropy')
    print(f"{ds:9s} M={M} | entropy={aE:.3f} patience={aP:.3f} | combo(λ={best_lam})={best_auc:.3f} "
          f"(vs best-pure {base:.3f}: {100*(best_auc-base):+.2f}pp) | UB AND={envAND:.3f} OR={envOR:.3f} "
          f"(OR-UB vs best-pure {100*(envOR-base):+.2f}pp)")

print("\n=== verdict: does the FIXED combo beat the better pure signal? ===")
for ds, s in summary.items():
    base = max(s['entropy'], s['patience']); d = s['combo_fixed'] - base
    print(f"  {ds:9s}: {'YES' if d>0.001 else 'no '} ({100*d:+.2f}pp vs best-pure={s['best_pure']}); headroom (OR-UB)={100*(s['OR_UB']-base):+.2f}pp")
json.dump(summary, open(os.path.join(R_, 'lsmi_exit', 'combo_summary.json'), 'w'), indent=2)
