"""Add R (LSMI redundancy) as an exit signal to the gflops_exit_schemes scheme, on seqA_ps.
Per-prefix threshold-sweep exit curves for entropy/patience/bottleneck (exit at the prefix
where the signal crosses tau), PLUS an R-exit curve (per-sample budget: high-R -> 1 modality,
low-R -> all M, swept by R-threshold; R is the oracle LSMI redundancy from arrays_<ds>.npz).
acc & f1 vs mean GFLOPs, 4 datasets. Run from clean_models/train/."""
from __future__ import annotations
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
from itertools import combinations
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
_HERE = os.path.dirname(os.path.abspath(__file__)); _CM = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_CM, 'results_3p3'))
from lsmi_exit_seqps_4ds import setup, fwd, dev, UNITS

GRID_ENT = [0.0, .02, .05, .08, .12, .18, .25, .35, .5, .7, .9, 1.2, 1.6, 2.0, 10.0]


@torch.no_grad()
def prefix_logits_boards(model, batches, order):
    M = len(order); per = [[] for _ in range(M)]; bper = [[] for _ in range(M)]; ys = []
    fusion = model.bottleneck_fusion
    for high, low, y in batches:
        ys.append(y.cpu())
        for k in range(1, M + 1):
            o = fwd(model, high, low, order[:k]); per[k - 1].append(o.cpu())
            bs = fusion._last_bn_state; bd = bs[max(bs)]; bper[k - 1].append(bd.reshape(bd.shape[0], -1).cpu())
    L = torch.stack([torch.cat(per[k]) for k in range(M)], 1).float()
    BD = torch.stack([torch.cat(bper[k]) for k in range(M)], 1).float()
    return L, BD, torch.cat(ys).numpy()


def acc_v(model, batches, keep):
    cor = n = 0
    for high, low, y in batches:
        cor += int((fwd(model, high, low, keep).argmax(-1).cpu() == y.cpu()).sum()); n += y.shape[0]
    return cor / n


def exitsim(score, preds, y, taus, ge):
    """per-prefix threshold exit -> (meanK, acc, f1) per tau."""
    N, M = score.shape; out = []
    for t in taus:
        ek = np.full(N, M)
        for k in range(M):
            cond = (score[:, k] >= t) if ge else (score[:, k] <= t)
            ek = np.where((ek == M) & cond, k + 1, ek)
        pe = preds[np.arange(N), ek - 1]
        out.append((ek.mean(), (pe == y).mean(), f1_score(y, pe, average='macro')))
    return np.array(sorted(out))


def Rbucket(R, correct, y_preds_1, y_preds_M, y, M):
    """per-sample 2-bucket: R>=tau -> k=1 else k=M. sweep tau -> (meanK, acc, f1)."""
    N = len(R); order = np.argsort(-R); out = []
    c1, cM = correct[:, 0], correct[:, -1]
    for p in np.linspace(0, 1, 21):
        e = int(round(p * N)); early = order[:e]; rest = order[e:]
        pe = np.where(np.isin(np.arange(N), early), y_preds_1, y_preds_M)
        acc = (c1[early].sum() + cM[rest].sum()) / N
        out.append(((e * 1 + (N - e) * M) / N, acc, f1_score(y, pe, average='macro')))
    return np.array(sorted(out))


DS = ['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei']
SCH = {'entropy': ('tab:gray', '-'), 'patience': ('tab:purple', '-'), 'bottleneck': ('tab:cyan', '--'), 'R (oracle)': ('tab:red', '-o')}
res = {}
for ds in DS:
    arr = np.load(os.path.join(_HERE, 'lsmi_exit', f'arrays_{ds}.npz')); R = arr['R']; gpk = arr['gflops_per_k']; M = int(arr['M']); ek_saved = arr['exit_k']
    Ls, BDs, ys = [], [], []
    for unit in UNITS[ds]:
        model, (trb, vb, teb), mods, NUMC = setup(ds, unit)
        full = acc_v(model, vb, mods); loo = {m: full - acc_v(model, vb, [x for x in mods if x != m]) for m in mods}
        order = sorted(mods, key=lambda m: -loo[m])
        L, BD, y = prefix_logits_boards(model, teb, order); Ls.append(L); BDs.append(BD); ys.append(y)
        del model; torch.cuda.empty_cache()
    L = torch.cat(Ls); BD = torch.cat(BDs); y = np.concatenate(ys); N = len(y)
    P = F.softmax(L, -1); preds = L.argmax(-1).numpy()
    H = -(P * P.clamp_min(1e-9).log()).sum(-1).numpy()
    rl = np.ones((N, M), int)
    for k in range(1, M):
        rl[:, k] = np.where(preds[:, k] == preds[:, k - 1], rl[:, k - 1] + 1, 1)
    BDn = F.normalize(BD, dim=-1); cos = (BDn[:, 1:] * BDn[:, :-1]).sum(-1).numpy()
    bn = np.full((N, M), -1.0); bn[:, 1:] = cos
    correct = (preds == y[:, None]).astype(float)
    np.savez(os.path.join(_HERE, 'lsmi_exit', f'exit_raw_{ds}.npz'), H=H, rl=rl, maxprob=P.max(-1).values.numpy(),
             bn=bn, preds=preds, y=y, R=R, gpk=np.array(gpk), M=M)   # raw per-prefix signals for combo experiments
    ek_now = np.full(N, M)
    for k in range(M):
        ek_now = np.where((ek_now == M) & correct[:, k].astype(bool), k + 1, ek_now)
    align = float(np.mean(ek_now == ek_saved)) if len(ek_saved) == N else -1
    curves = {}
    curves['entropy'] = exitsim(H, preds, y, GRID_ENT, ge=False)
    curves['patience'] = exitsim(rl + P.max(-1).values.numpy(), preds, y, np.linspace(1, M + 1, 24), ge=True)
    curves['bottleneck'] = exitsim(bn, preds, y, np.quantile(bn[:, 1:], np.linspace(1, 0, 16)), ge=True)
    curves['R (oracle)'] = Rbucket(R, correct, preds[:, 0], preds[:, -1], y, M)
    res[ds] = {'curves': curves, 'gpk': gpk.tolist(), 'M': M, 'N': N, 'align': align}
    print(f"{ds}: N={N} align(exit_k vs saved)={align:.3f}  "
          f"R-exit acc {curves['R (oracle)'][:,1].min():.3f}-{curves['R (oracle)'][:,1].max():.3f}")

fig, axes = plt.subplots(2, len(DS), figsize=(4.7 * len(DS), 8.5))
for col, ds in enumerate(DS):
    g = np.array(res[ds]['gpk']); M = res[ds]['M']
    for row, yi, ylab in [(0, 1, 'Accuracy'), (1, 2, 'Macro-F1')]:
        ax = axes[row, col]
        for nm, (c, ls) in SCH.items():
            cu = res[ds]['curves'][nm]; x = np.interp(cu[:, 0], np.arange(1, M + 1), g)
            o = np.argsort(x); ax.plot(x[o], cu[o, yi], ls, color=c, lw=2.4 if nm.startswith('R') else 1.8,
                                       ms=5 if nm.startswith('R') else 0, label=nm)
        ax.set_xscale('log'); ax.set_xlabel('mean GFLOPs/sample (log)'); ax.set_ylabel(ylab)
        ax.set_title(f'{ds} (M={M})' + ('  [seqA_ps]' if row == 0 else '')); ax.grid(alpha=.25, which='both')
        if col == 0 and row == 0: ax.legend(fontsize=8, loc='lower right')
fig.suptitle('Adaptive exit on seqA_ps: per-prefix schemes (entropy/patience/bottleneck) vs R-as-exit-signal '
             '(per-sample 2-bucket on ORACLE LSMI redundancy). Does R make a usable exit signal?', fontsize=11)
fig.subplots_adjust(left=.045, right=.997, bottom=.08, top=.90, wspace=.27, hspace=.30)
o = os.path.join(_HERE, 'lsmi_exit', 'gflops_exit_schemes_withR_4ds.png'); fig.savefig(o, dpi=140); print('saved ->', o)
json.dump({ds: {k: v.tolist() for k, v in res[ds]['curves'].items()} | {'gpk': res[ds]['gpk'], 'M': res[ds]['M']} for ds in DS},
          open(os.path.join(_HERE, 'lsmi_exit', 'exit_withR.json'), 'w'))
