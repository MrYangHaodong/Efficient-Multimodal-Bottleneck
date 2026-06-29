"""U-based pruning vs val-LOO pruning at MATCHED COMPUTE (keep top-K modalities), IEMOCAP seqA_ps.
Exhaustive: enumerate all 63 subsets, eval TEST acc for each (processed in val-LOO-strong-first order
restricted to the subset -> identical processing order regardless of which criterion picked it).
Then per K: acc of U-top-K set vs val-LOO-top-K set vs oracle(best K-subset) vs random(mean K-subset).
Reuses consistent_unique_iemocap.npz (Un + val-LOO drop). Run from clean_models/train/."""
import os, sys
from itertools import combinations
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_3p3'))
from lsmi_exit_seqps_4ds import setup, fwd, UNITS

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def acc(model, batches, keep):
    c = n = 0
    for h, l, y in batches:
        c += int((fwd(model, h, l, keep).argmax(-1).cpu() == y.cpu()).sum()); n += y.shape[0]
    return c / n


z = np.load(os.path.join(HERE, 'lsmi_exit', 'consistent_unique_iemocap.npz'), allow_pickle=True)
mods_npz = list(z['mods']); Un_all = z['Un']; drop_all = z['drop']   # [M, nfold]
M = len(mods_npz)
# accumulate per-fold curves
curves = {k: [] for k in ['valoo', 'U', 'oracle', 'random']}
for fi, fold in enumerate(UNITS['iemocap']):
    model, (trb, vb, teb), mods, C = setup('iemocap', fold)
    assert list(mods) == mods_npz, (mods, mods_npz)
    Un = {m: Un_all[i, fi] for i, m in enumerate(mods_npz)}
    drop = {m: drop_all[i, fi] for i, m in enumerate(mods_npz)}
    order_full = sorted(mods, key=lambda m: -drop[m])           # val-LOO strong-first (processing order)
    # exhaustive subset test acc, processed in val-LOO order restricted to subset
    tacc = {}
    for K in range(1, M + 1):
        for sub in combinations(mods, K):
            keep = [m for m in order_full if m in sub]
            tacc[frozenset(sub)] = acc(model, teb, keep)
    cv, cu, co, cr = [], [], [], []
    for K in range(1, M + 1):
        Uset = frozenset(sorted(mods, key=lambda m: -Un[m])[:K])
        Lset = frozenset(sorted(mods, key=lambda m: -drop[m])[:K])
        ksubs = [frozenset(s) for s in combinations(mods, K)]
        cu.append(tacc[Uset]); cv.append(tacc[Lset])
        co.append(max(tacc[s] for s in ksubs)); cr.append(np.mean([tacc[s] for s in ksubs]))
    curves['U'].append(cu); curves['valoo'].append(cv); curves['oracle'].append(co); curves['random'].append(cr)
    print(f"fold {fold}: U-rank={sorted(mods,key=lambda m:-Un[m])}")
    print(f"          LOO-rank={order_full}")
    del model; torch.cuda.empty_cache()

Ks = list(range(1, M + 1))
agg = {k: (np.array(v).mean(0), np.array(v).std(0)) for k, v in curves.items()}
print(f"\n=== IEMOCAP seqA_ps: test acc vs #modalities kept (matched compute), mean over 3 folds ===")
print(f"{'K':>2s} | {'val-LOO':>14s} {'consistent-U':>14s} {'Δ(U−LOO)':>9s} | {'oracle':>8s} {'random':>8s}")
for i, K in enumerate(Ks):
    v = agg['valoo'][0][i]; u = agg['U'][0][i]
    print(f"{K:>2d} | {v:.3f}±{agg['valoo'][1][i]:.3f}  {u:.3f}±{agg['U'][1][i]:.3f}  {100*(u-v):>+7.2f}pp | {agg['oracle'][0][i]:.3f}   {agg['random'][0][i]:.3f}")
auc_v = agg['valoo'][0].mean(); auc_u = agg['U'][0].mean()
print(f"\nmean-over-K acc: val-LOO={auc_v:.3f}  consistent-U={auc_u:.3f}  -> U {100*(auc_u-auc_v):+.2f}pp vs val-LOO")

plt.figure(figsize=(7, 5))
for k, lab, c, ls in [('oracle', 'oracle (best K-subset)', 'k', ':'), ('valoo', 'val-LOO pruning', 'tab:blue', '-o'),
                      ('U', 'consistent-U pruning', 'tab:red', '-s'), ('random', 'random subset (mean)', 'tab:gray', '--')]:
    m, s = agg[k]; plt.plot(Ks, m, ls, color=c, lw=2.2, label=lab)
    if k in ('valoo', 'U'): plt.fill_between(Ks, m - s, m + s, color=c, alpha=0.12)
plt.xlabel('# modalities kept (matched compute)'); plt.ylabel('Test accuracy (cross-subject)')
plt.title('IEMOCAP seqA_ps: U-based pruning vs val-LOO pruning at matched compute')
plt.grid(alpha=0.3); plt.legend()
o = os.path.join(HERE, 'lsmi_exit', 'unique_vs_valoo_pruning_iemocap.png'); plt.savefig(o, dpi=140, bbox_inches='tight')
print('saved ->', o)
