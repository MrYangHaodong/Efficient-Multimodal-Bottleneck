"""Does a test sample's LSMI interaction type (R/U/S) explain its early-exitability?
On the seqA_ps model: compute per-sample R/U/S (LSMI, oracle w.r.t. y_true AND label-free
w.r.t. the model's full-modality posterior) and correlate with the per-sample ORACLE EXIT-K
= min #top-(val-LOO) modalities needed for a correct prefix prediction. PoC: IEMOCAP (M=6).
Run from clean_models/train/:  python ../results_3p3/lsmi_exit_seqps.py --dataset iemocap"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn.functional as F
from itertools import combinations
from scipy.stats import spearmanr
_HERE = os.path.dirname(os.path.abspath(__file__)); _CM = os.path.dirname(_HERE)
sys.path.insert(0, _CM); sys.path.insert(0, os.path.join(_CM, 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(_CM), 'v6_organized', 'script'))
from lsmi.entropy_estimation import MargKernel
from compute_vlo_vs_random import build_iemocap, unpack_iemocap, dev
from torch.optim import AdamW

ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='iemocap'); args = ap.parse_args()
SEQPS = {'iemocap': sorted(glob.glob(os.path.join(_CM, 'results_3p3/seqA_prefixsup/iemocap/2026-*')))[0]}


def extract_z_and_subsets(model, batches, mods, pairs):
    """Run per-subset forwards; return per-modality z (mean-pooled encoder feat),
    full/singleton/pair softmaxes, prefix logits (val-LOO order set later), labels."""
    M = len(mods)
    z = {m: [] for m in mods}; labels = []
    full_sm, sing_sm = [], {m: [] for m in mods}; pair_sm = {pi: [] for pi in range(len(pairs))}
    with torch.no_grad():
        for high, low, y in batches:
            labels.append(y.cpu())
            # full forward -> z (per-modality encoder features) + full softmax
            o = model(high, low, training=False, return_selection_info=False)
            o = o[0] if isinstance(o, tuple) else o
            full_sm.append(F.softmax(o, -1).cpu())
            proc = model._last_processed_modalities
            for m in mods:
                z[m].append(proc[m].mean(1).cpu())
            for m in mods:
                om = model({m: high[m]}, {m: low[m]}, training=False, return_selection_info=False)
                om = om[0] if isinstance(om, tuple) else om
                sing_sm[m].append(F.softmax(om, -1).cpu())
            for pi, (a, b) in enumerate(pairs):
                keep = [mods[a], mods[b]]
                op = model({m: high[m] for m in keep}, {m: low[m] for m in keep}, training=False, return_selection_info=False)
                op = op[0] if isinstance(op, tuple) else op
                pair_sm[pi].append(F.softmax(op, -1).cpu())
    cat = lambda L: torch.cat(L, 0)
    return ({m: cat(z[m]) for m in mods}, cat(full_sm), {m: cat(sing_sm[m]) for m in mods},
            {pi: cat(pair_sm[pi]) for pi in pair_sm}, torch.cat(labels).numpy())


def prefix_correct_k(model, batches, mods, order):
    """Per-sample oracle exit-k = min k s.t. top-k (val-LOO order) prefix prediction == y_true.
    M+1 if never correct."""
    M = len(mods); by, ys = [], []
    with torch.no_grad():
        for k in range(1, M + 1):
            keep = order[:k]; outs = []
            yk = []
            for high, low, y in batches:
                o = model({m: high[m] for m in keep}, {m: low[m] for m in keep}, training=False, return_selection_info=False)
                o = o[0] if isinstance(o, tuple) else o
                outs.append(o.argmax(-1).cpu()); yk.append(y.cpu())
            by.append(torch.cat(outs));
            if k == 1: ys = torch.cat(yk).numpy()
    preds = torch.stack(by, 1).numpy()                      # [N, M]
    correct = (preds == ys[:, None])                        # [N, M]
    ek = np.full(len(ys), M + 1)
    for k in range(M):
        ek = np.where((ek == M + 1) & correct[:, k], k + 1, ek)
    return ek, ys, preds


def train_knife(zper, mods, epochs=30, bs=128, lr=1e-3):
    est = {}
    for m in mods:
        Z = zper[m]; mdl = MargKernel(dim=Z.shape[1]).to(dev); opt = AdamW(mdl.parameters(), lr=lr)
        for ep in range(epochs):
            perm = torch.randperm(Z.shape[0])
            for b0 in range(0, Z.shape[0], bs):
                x = Z[perm[b0:b0 + bs]].to(dev); opt.zero_grad(); mdl(x).mean().backward(); opt.step()
        est[m] = mdl
    return est


@torch.no_grad()
def H_of(mdl, Z, bs=256):
    mdl.eval()   # eval mode -> per-sample entropy (train mode returns the scalar mean)
    return torch.cat([mdl(Z[b0:b0 + bs].to(dev)).cpu() for b0 in range(0, Z.shape[0], bs)], 0)


def rus(zte, full_sm, sing, pair, labels, knife, mods, pairs, target):
    """R/U/S per (sample,modality), averaged over partner. target: 'ytrue' or 'full'."""
    M = len(mods); N = len(labels); C = full_sm.shape[1]; logC = float(np.log(C))
    tgt = full_sm if target == 'full' else F.one_hot(torch.tensor(labels), C).float()
    i_m = {m: logC + (sing[m].clamp_min(1e-12).log() * tgt).sum(-1).numpy() for m in mods}
    i12 = {pi: (logC + (pair[pi].clamp_min(1e-12).log() * tgt).sum(-1).numpy()) for pi in pair}
    H = {m: H_of(knife[m], zte[m]).numpy() for m in mods}
    I = np.stack([i_m[m] for m in mods])
    R = np.zeros((M, N)); U = np.zeros((M, N)); S = np.zeros((M, N))
    for pi, (a, b) in enumerate(pairs):
        r = np.minimum(H[mods[a]], H[mods[b]]) - np.minimum(H[mods[a]] - I[a], H[mods[b]] - I[b])
        ua = I[a] - r; ub = I[b] - r; s = i12[pi] - I[a] - I[b] + r
        R[a] += r; R[b] += r; U[a] += ua; U[b] += ub; S[a] += s; S[b] += s
    R /= (M - 1); U /= (M - 1); S /= (M - 1)
    # per-sample aggregate
    return {'R': R.mean(0), 'U': U.max(0), 'S': S.mean(0)}   # [N] each


def run(dsname):
    d = SEQPS[dsname]; C = json.load(open(d + '/config.json'))
    from data.IEMOCAP.get_data import get_dataloader, FEATURE_DIMS, NUM_CLASSES
    mods = list(C['modalities']); M = len(mods); variates = {m: FEATURE_DIMS[m] for m in mods}
    T = C['max_seq_len']; pairs = list(combinations(range(M), 2))
    agg = {'ytrue': {'R': [], 'U': [], 'S': []}, 'full': {'R': [], 'U': [], 'S': []}, 'ek': [], 'corr_ek1': []}
    for fold in range(3):
        tl, vl, te = get_dataloader(data_root='/files1/haodong/data/IEMOCAP', batch_size=64, num_workers=4,
            modalities=mods, split_id=fold, train_shuffle=False, max_seq_len=C['max_seq_len'],
            time_compression_ratio=C['time_compression_ratio'], use_batched_fusion=C['use_batched_fusion'],
            available_sessions=None, split_mode='cross_subject')
        def conv(loader):
            out = []
            for batch in loader:
                high, low, y = unpack_iemocap(batch, mods); out.append((high, low, y))
            return out
        trb, vb, teb = conv(tl), conv(vl), conv(te)
        m = build_iemocap(C, mods, variates, T, NUM_CLASSES)
        sd = torch.load(d + f'/best_model_loo_fold{fold}.pth', map_location='cpu', weights_only=True)
        m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); m.eval()
        # val-LOO order
        def acc(keep, bs):
            cor = n = 0
            for high, low, y in bs:
                o = m({mm: high[mm] for mm in keep}, {mm: low[mm] for mm in keep}, training=False, return_selection_info=False)
                o = o[0] if isinstance(o, tuple) else o; cor += int((o.argmax(-1).cpu() == y.cpu()).sum()); n += y.shape[0]
            return cor / n
        full = acc(mods, vb); loo = {mm: full - acc([x for x in mods if x != mm], vb) for mm in mods}
        order = sorted(mods, key=lambda mm: -loo[mm])
        zper_tr, *_ = extract_z_and_subsets(m, trb, mods, [])           # z only (no pairs) for KNIFE train
        zte, full_sm, sing, pair, labels = extract_z_and_subsets(m, teb, mods, pairs)
        knife = train_knife(zper_tr, mods)
        ek, ys, preds = prefix_correct_k(m, teb, mods, order)
        for tgt in ('ytrue', 'full'):
            rr = rus(zte, full_sm, sing, pair, labels, knife, mods, pairs, tgt)
            for k in ('R', 'U', 'S'): agg[tgt][k].append(rr[k])
        agg['ek'].append(ek.astype(float))
        print(f'  fold{fold} order={order} exit-k mean={ek.mean():.2f} (M={M}); '
              f'frac exit@1={np.mean(ek==1):.2f} need-all={np.mean(ek>=M):.2f}')
        del m; torch.cuda.empty_cache()
    # pool folds
    ek = np.concatenate(agg['ek'])
    print(f"\n=== {dsname} seqA_ps: R/U/S vs ORACLE exit-k (pooled {len(ek)} test samples, 3 folds) ===")
    print("Hypothesis: Synergy↑ -> exit-k↑ (need more modalities); Redundancy↑ -> exit-k↓ (exit early).")
    for tgt in ('ytrue', 'full'):
        print(f"\n-- target = {tgt} ({'oracle LSMI' if tgt=='ytrue' else 'label-free (model full posterior)'}) --")
        for k in ('R', 'U', 'S'):
            v = np.concatenate(agg[tgt][k]); rho, p = spearmanr(v, ek)
            print(f"   Spearman({k}, exit-k) = {rho:+.3f}  (p={p:.1e})")
        # mean exit-k by dominant interaction type
        Rv, Uv, Sv = (np.concatenate(agg[tgt][x]) for x in ('R', 'U', 'S'))
        dom = np.argmax(np.stack([Rv, Uv, Sv]), 0)   # 0=R,1=U,2=S
        for di, nm in [(0, 'R-dom'), (1, 'U-dom'), (2, 'S-dom')]:
            mask = dom == di
            if mask.sum(): print(f"   {nm}: n={mask.sum():4d}  mean exit-k={ek[mask].mean():.2f}  frac-exit@1={np.mean(ek[mask]==1):.2f}")
    out = {'dataset': dsname, 'exit_k': ek.tolist(),
           'ytrue': {k: np.concatenate(agg['ytrue'][k]).tolist() for k in ('R', 'U', 'S')},
           'full': {k: np.concatenate(agg['full'][k]).tolist() for k in ('R', 'U', 'S')}}
    os.makedirs(os.path.join(_HERE, 'lsmi_exit'), exist_ok=True)
    json.dump(out, open(os.path.join(_HERE, 'lsmi_exit', f'{dsname}.json'), 'w'))
    print(f"\nsaved -> results_3p3/lsmi_exit/{dsname}.json")


run(args.dataset)
