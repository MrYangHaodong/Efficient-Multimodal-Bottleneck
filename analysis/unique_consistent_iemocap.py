"""Prototype: CONSISTENT unique-information estimator (Lyu/Clark/Raviv 2508.05530, Def. 6) at M=6,
reusing LSMI's subset-softmax engine on IEMOCAP seqA_ps. For each modality i:
  Un(S_i -> T | S_{[N]\\i}) = I(S'_i; T | S_rest) = H(T|S_rest) - H(T|S'_i, S_rest)
where the primed posterior (derived):  p(T | S'_i, S_rest) ∝ p(T|S_i)·p(T|S_rest)/p(T)
and the primed expectation is estimated by WITHIN-CLASS PERMUTATION (draw S_i from an independent
same-class sample). Discrete T -> no KNIFE. Validates per-modality U against val-LOO acc-importance.
Run from clean_models/train/:  python ../results_3p3/unique_consistent_iemocap.py"""
import os, sys
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_3p3'))
from lsmi_exit_seqps_4ds import setup, fwd, UNITS

K = 30          # within-class permutation draws per base sample
EPS = 1e-12


@torch.no_grad()
def subset_softmax(model, batches, keep):
    return torch.cat([F.softmax(fwd(model, h, l, keep), -1).cpu() for h, l, _ in batches]).numpy()


@torch.no_grad()
def acc(model, batches, keep):
    c = n = 0
    for h, l, y in batches:
        c += int((fwd(model, h, l, keep).argmax(-1).cpu() == y.cpu()).sum()); n += y.shape[0]
    return c / n


def H(P):                       # row-wise Shannon entropy (nats)
    return -(P * np.log(np.clip(P, EPS, 1))).sum(-1)


rng = np.random.RandomState(0)
folds = UNITS['iemocap']
Un_f = {}                       # modality -> [Un per fold]
drop_f = {}                     # modality -> [val-LOO acc drop per fold]
diag = {'H_full': [], 'C': None, 'mods': None}
for fold in folds:
    model, (trb, vb, teb), mods, NUMC = setup('iemocap', fold)
    diag['mods'] = mods; diag['C'] = NUMC
    y = np.concatenate([yy.cpu().numpy() for _, _, yy in teb])
    C = NUMC
    full = subset_softmax(model, teb, mods)
    sing = {m: subset_softmax(model, teb, [m]) for m in mods}
    loo = {m: subset_softmax(model, teb, [x for x in mods if x != m]) for m in mods}
    prior = np.bincount(y, minlength=C).astype(float); prior /= prior.sum()
    logpri = np.log(np.clip(prior, EPS, 1))
    diag['H_full'].append(H(full).mean())
    # val-LOO importance (deployable, on val set)
    vfull = acc(model, vb, mods)
    for m in mods:
        drop_f.setdefault(m, []).append(vfull - acc(model, vb, [x for x in mods if x != m]))
    idx_by_c = {c: np.where(y == c)[0] for c in range(C)}
    for m in mods:
        H_rest = H(loo[m]).mean()
        loga = np.log(np.clip(sing[m], EPS, 1))      # log p(T|S_m=a)
        logb = np.log(np.clip(loo[m], EPS, 1))       # log p(T|S_rest=b)
        per_j = []
        for c in range(C):
            Jc = idx_by_c[c]
            if len(Jc) == 0:
                continue
            ks = rng.choice(Jc, size=(len(Jc), K), replace=True)        # same-class draws for S_m
            lp = loga[ks] + logb[Jc][:, None, :] - logpri[None, None, :]  # [|Jc|,K,C], PoE/prior in log
            lp -= lp.max(-1, keepdims=True)
            P = np.exp(lp); P /= P.sum(-1, keepdims=True)
            h = -(P * np.log(np.clip(P, EPS, 1))).sum(-1)               # [|Jc|,K]
            per_j.append(h.mean(1))                                     # mean over K -> per base j
        H_primed = np.concatenate(per_j).mean()
        Un_f.setdefault(m, []).append(H_rest - H_primed)
    del model; torch.cuda.empty_cache()

mods = diag['mods']; C = diag['C']
print(f"\nIEMOCAP seqA_ps, consistent UNIQUE info (Def.6, M={len(mods)}, C={C}); entropy in nats (max ln{C}={np.log(C):.3f})")
print(f"mean H(T|S_all)={np.mean(diag['H_full']):.3f}\n")
print(f"{'modality':14s} {'Un (mean±std)':>16s} {'val-LOO acc-drop':>16s}")
Un_mean = {m: np.mean(Un_f[m]) for m in mods}
for m in sorted(mods, key=lambda m: -Un_mean[m]):
    print(f"{m:14s} {np.mean(Un_f[m]):+.4f} ± {np.std(Un_f[m]):.4f}   {np.mean(drop_f[m]):+.4f}")
u = np.array([Un_mean[m] for m in mods]); d = np.array([np.mean(drop_f[m]) for m in mods])
rho, p = spearmanr(u, d)
print(f"\nSpearman(consistent-Un, val-LOO acc-drop) over {len(mods)} modalities = {rho:+.3f} (p={p:.2f})")
print(f"non-negativity check: min Un = {u.min():+.4f}  (should be >= 0 up to estimation noise)")
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lsmi_exit', 'consistent_unique_iemocap.npz'),
         mods=np.array(mods), Un=np.array([Un_f[m] for m in mods]), drop=np.array([drop_f[m] for m in mods]))
print("saved -> results_3p3/lsmi_exit/consistent_unique_iemocap.npz")
