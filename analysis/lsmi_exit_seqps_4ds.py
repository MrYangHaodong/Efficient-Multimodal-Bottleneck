"""#1: Does LSMI redundancy add anything OVER patience for predicting per-sample
early-exitability? On seqA_ps, per dataset: compute oracle exit-k (min top-val-LOO
modalities for a correct prefix), per-sample R/U/S (LSMI, oracle w.r.t. y_true), and
DEPLOYABLE label-free signals (maxprob@1, entropy@1, patience-stabilize-k). Report
Spearman(each, exit-k) and PARTIAL Spearman(R, exit-k | patience/maxprob) — if ~0,
redundancy is subsumed by patience. Datasets via --ds. Run from clean_models/train/."""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn.functional as F
from itertools import combinations
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
_HERE = os.path.dirname(os.path.abspath(__file__)); _CM = os.path.dirname(_HERE)
sys.path.insert(0, _CM); sys.path.insert(0, os.path.join(_CM, 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(_CM), 'v6_organized', 'script'))
from lsmi.entropy_estimation import MargKernel
from compute_vlo_vs_random import build_har, build_iemocap, sub_har, unpack_iemocap, dev

ap = argparse.ArgumentParser(); ap.add_argument('--ds', nargs='+', default=['daliahar', 'pamap2', 'dsads', 'mosei']); args = ap.parse_args()
PS = os.path.join(_CM, 'results_3p3/seqA_prefixsup')
CKPT_KIND = os.environ.get('LSMI_CKPT_KIND', 'ps')   # 'ps' = prefix-sup (seqA_ps); 'gapdual' = random-order deployed model
def _ckpt_dir(ds):
    if CKPT_KIND == 'gapdual':
        return sorted(glob.glob(os.path.join(_CM, 'results_3p3', f'seqA_{ds}', '*seqA_gap_dual*')))[0]
    return sorted(glob.glob(f'{PS}/{ds}/2026-*'))[0]
SUF = 'gapdual_' if CKPT_KIND == 'gapdual' else ''


# ---------- per-dataset: model + (train/val/test) per-modality batches ----------
def setup(ds, unit):
    """unit = fold (HAR/iemocap) or seed (mosei). Returns model, (trb,vb,teb), mods, NUMC, order-helper."""
    if ds in ('daliahar', 'pamap2', 'dsads'):
        from utils.dataset_cfg import DaliaHAR, PAMAP2, DSADS
        cfg = {'daliahar': DaliaHAR, 'pamap2': PAMAP2, 'dsads': DSADS}[ds]()
        MODS = list(cfg.modalities); NUMC = cfg.num_classes; variates = dict(cfg.variates)
        T = {'daliahar': 256, 'pamap2': 512, 'dsads': 125}[ds]
        TR = {'daliahar': 'sax', 'pamap2': 'sax_noisy', 'dsads': 'sax'}[ds]; SAX = {'alphabet_size': 20, 'word_length': 1}
        sl, off = {}, 0
        for m in MODS:
            sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
        d = _ckpt_dir(ds); C = json.load(open(d + '/config.json'))
        model = build_har(C, MODS, variates, T, NUMC, C['num_layers_per_modal'])
        sd = torch.load(f'{d}/best_model_loo_fold{unit}.pth', map_location='cpu', weights_only=True)
        model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()

        def har_b(loader):
            out = []
            for x, y in loader:
                x = x.to(dev).float(); high = {m: x[:, :, sl[m][0]:sl[m][1]] for m in MODS}
                out.append((high, dict(high), y))
            return out
        if ds == 'dsads':
            from data.dataset_builder import DSADSDataset
            src = torch.load(f'/files1/haodong/data/DSADS/dsad_diversify_dict_scenario{unit}_src.pt', weights_only=False)
            trg = torch.load(f'/files1/haodong/data/DSADS/dsad_diversify_dict_scenario{unit}_trg.pt', weights_only=False)
            trx, vax, tryy, vay = train_test_split(src['samples'], src['labels'], test_size=0.2, random_state=2711, stratify=src['labels'])
            trb = har_b(DataLoader(DSADSDataset({'samples': trx, 'labels': tryy}, MODS, cfg, 'sax', sax_params=SAX), batch_size=64, num_workers=4))
            vb = har_b(DataLoader(DSADSDataset({'samples': vax, 'labels': vay}, MODS, cfg, 'sax', sax_params=SAX), batch_size=64, num_workers=4))
            teb = har_b(DataLoader(DSADSDataset(trg, MODS, cfg, 'sax', sax_params=SAX), batch_size=64, num_workers=4))
        elif ds == 'pamap2':
            from data.pamap2_crosssubject import load_pamap2_subjects
            fc = cfg.folds[unit]
            trb = har_b(DataLoader(load_pamap2_subjects(cfg.cross_subject_dir, MODS, fc['train_set'], cfg, TR, sax_params=SAX), batch_size=64, num_workers=4))
            vb = har_b(DataLoader(load_pamap2_subjects(cfg.cross_subject_dir, MODS, cfg.val_set, cfg, TR, sax_params=SAX), batch_size=64, num_workers=4))
            teb = har_b(DataLoader(load_pamap2_subjects(cfg.cross_subject_dir, MODS, fc['eval_set'], cfg, TR, sax_params=SAX), batch_size=64, num_workers=4))
        else:
            from data.dataset_builder import HARDataset
            R = '/files1/haodong/data/processed_dalia_activity'; fc = cfg.folds[unit]
            trb = har_b(DataLoader(HARDataset(R, MODS, fc['train_set'], cfg, 'sax', sax_params=SAX), batch_size=64, num_workers=4))
            vb = har_b(DataLoader(HARDataset(R, MODS, cfg.val_set, cfg, 'sax', sax_params=SAX), batch_size=64, num_workers=4))
            teb = har_b(DataLoader(HARDataset(R, MODS, fc['eval_set'], cfg, 'sax', sax_params=SAX), batch_size=64, num_workers=4))
        return model, (trb, vb, teb), MODS, NUMC
    if ds == 'mosei':
        from data.CMU_MOSEI.get_data import get_dataloader, CMUMOSEIDataset, FEATURE_DIMS, NUM_CLASSES
        MODS = list(CMUMOSEIDataset.ALL_MODALITIES); NUMC = NUM_CLASSES; variates = {m: FEATURE_DIMS[m] for m in MODS}
        d = sorted(glob.glob(f'{_CM}/train/results_mosei_5seed/*_seqA_ps_acc2_s{unit}_*'))[0]; C = json.load(open(d + '/config.json'))
        model = build_iemocap(C, MODS, variates, C['max_seq_len'], NUMC)   # generic float builder
        sd = torch.load(f'{d}/best_model_cold_val.pth', map_location='cpu', weights_only=True)
        model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()
        tl, vl, te = get_dataloader(batch_size=64, num_workers=4, modalities=MODS, split_id=0, max_seq_len=50,
                                    time_compression_ratio=4, task='classification2')
        def conv(loader):
            out = []
            for batch in loader:
                *feats, label = batch
                high = {m: feats[2 * j].to(dev).float() for j, m in enumerate(MODS)}
                low = {m: feats[2 * j + 1].to(dev).float() for j, m in enumerate(MODS)}
                out.append((high, low, label.squeeze(-1).long()))
            return out
        return model, (conv(tl), conv(vl), conv(te)), MODS, NUMC
    if ds == 'iemocap':
        from data.IEMOCAP.get_data import get_dataloader, FEATURE_DIMS, NUM_CLASSES
        MODS = list(['video', 'audio', 'text', 'mocap_hand', 'mocap_head', 'mocap_rotated']); NUMC = NUM_CLASSES
        d = _ckpt_dir('iemocap'); C = json.load(open(d + '/config.json'))
        MODS = list(C['modalities']); variates = {m: FEATURE_DIMS[m] for m in MODS}
        model = build_iemocap(C, MODS, variates, C['max_seq_len'], NUMC)
        sd = torch.load(f'{d}/best_model_loo_fold{unit}.pth', map_location='cpu', weights_only=True)
        model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()
        tl, vl, te = get_dataloader(data_root='/files1/haodong/data/IEMOCAP', batch_size=64, num_workers=4, modalities=MODS,
            split_id=unit, train_shuffle=False, max_seq_len=C['max_seq_len'], time_compression_ratio=C['time_compression_ratio'],
            use_batched_fusion=C['use_batched_fusion'], available_sessions=None, split_mode='cross_subject')
        def conv(loader):
            out = []
            for batch in loader:
                high, low, y = unpack_iemocap(batch, MODS); out.append((high, low, y))
            return out
        return model, (conv(tl), conv(vl), conv(te)), MODS, NUMC


def fwd(model, high, low, keep):
    o = model({m: high[m] for m in keep}, {m: low[m] for m in keep}, training=False, return_selection_info=False)
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def acc(model, batches, keep):
    cor = n = 0
    for high, low, y in batches:
        cor += int((fwd(model, high, low, keep).argmax(-1).cpu() == y.cpu()).sum()); n += y.shape[0]
    return cor / n


@torch.no_grad()
def extract_z(model, batches, mods):
    z = {m: [] for m in mods}
    for high, low, y in batches:
        fwd(model, high, low, mods)              # full forward sets _last_processed_modalities
        proc = model.bottleneck_fusion._last_processed_modalities if hasattr(model.bottleneck_fusion, '_last_processed_modalities') else model._last_processed_modalities
        for m in mods:
            z[m].append(proc[m].mean(1).cpu())
    return {m: torch.cat(z[m]) for m in mods}


@torch.no_grad()
def subsets(model, batches, mods, pairs):
    full, sing, pair = [], {m: [] for m in mods}, {pi: [] for pi in range(len(pairs))}
    labels = []
    for high, low, y in batches:
        labels.append(y.cpu())
        full.append(F.softmax(fwd(model, high, low, mods), -1).cpu())
        for m in mods:
            sing[m].append(F.softmax(fwd(model, high, low, [m]), -1).cpu())
        for pi, (a, b) in enumerate(pairs):
            pair[pi].append(F.softmax(fwd(model, high, low, [mods[a], mods[b]]), -1).cpu())
    cat = lambda L: torch.cat(L, 0)
    return cat(full), {m: cat(sing[m]) for m in mods}, {pi: cat(pair[pi]) for pi in pair}, torch.cat(labels).numpy()


def train_knife(zper, mods, epochs=20, bs=128):
    est = {}
    for m in mods:
        Z = zper[m]; mdl = MargKernel(dim=Z.shape[1]).to(dev); opt = AdamW(mdl.parameters(), lr=1e-3)
        for ep in range(epochs):
            perm = torch.randperm(Z.shape[0])
            for b0 in range(0, Z.shape[0], bs):
                x = Z[perm[b0:b0 + bs]].to(dev); opt.zero_grad(); mdl(x).mean().backward(); opt.step()
        mdl.eval(); est[m] = mdl
    return est


@torch.no_grad()
def H_of(mdl, Z, bs=256):
    return torch.cat([mdl(Z[b0:b0 + bs].to(dev)).cpu() for b0 in range(0, Z.shape[0], bs)], 0).numpy()


def rus(zte, full_sm, sing, pair, labels, knife, mods, pairs):
    M = len(mods); N = len(labels); C = full_sm.shape[1]; logC = float(np.log(C))
    tgt = F.one_hot(torch.tensor(labels), C).float()                    # oracle: y_true
    i_m = {m: logC + (sing[m].clamp_min(1e-12).log() * tgt).sum(-1).numpy() for m in mods}
    i12 = {pi: logC + (pair[pi].clamp_min(1e-12).log() * tgt).sum(-1).numpy() for pi in pair}
    H = {m: H_of(knife[m], zte[m]) for m in mods}
    I = np.stack([i_m[m] for m in mods]); R = np.zeros((M, N)); U = np.zeros((M, N)); S = np.zeros((M, N))
    for pi, (a, b) in enumerate(pairs):
        r = np.minimum(H[mods[a]], H[mods[b]]) - np.minimum(H[mods[a]] - I[a], H[mods[b]] - I[b])
        R[a] += r; R[b] += r; U[a] += I[a] - r; U[b] += I[b] - r
        s = i12[pi] - I[a] - I[b] + r; S[a] += s; S[b] += s
    R /= (M - 1); U /= (M - 1); S /= (M - 1)
    return R.mean(0), U.max(0), S.mean(0)


@torch.no_grad()
def exit_and_deployable(model, batches, mods, order):
    """oracle exit-k + deployable signals (maxprob@1, entropy@1, patience-stabilize-k)."""
    M = len(mods); per = [[] for _ in range(M)]; ys = []
    for high, low, y in batches:
        ys.append(y.cpu())
        for k in range(1, M + 1):
            per[k - 1].append(fwd(model, high, low, order[:k]).cpu())
    ys = torch.cat(ys).numpy(); L = torch.stack([torch.cat(per[k]) for k in range(M)], 1)  # [N,M,C]
    P = F.softmax(L, -1); preds = L.argmax(-1).numpy()
    correct = (preds == ys[:, None]); ek = np.full(len(ys), M + 1)
    for k in range(M):
        ek = np.where((ek == M + 1) & correct[:, k], k + 1, ek)
    msp1 = P[:, 0].max(-1).values.numpy(); H1 = -(P[:, 0] * P[:, 0].clamp_min(1e-9).log()).sum(-1).numpy()
    # patience-stabilize-k: first k where pred == final pred (label-free convergence)
    pat = np.full(len(ys), M)
    for k in range(M - 1, -1, -1):
        pat = np.where(preds[:, k] == preds[:, M - 1], k + 1, pat)
    return ek.astype(float), msp1, H1, pat.astype(float), correct.astype(np.float32), preds.astype(np.int64), ys.astype(np.int64)


@torch.no_grad()
def gflops_per_k(model, batches, mods, order):
    from torch.utils.flop_counter import FlopCounterMode
    high, low, _ = batches[0]; M = len(mods); gpk = []
    h1 = {m: high[m][:1] for m in mods}; l1 = {m: low[m][:1] for m in mods}
    for k in range(1, M + 1):
        keep = order[:k]; fcm = FlopCounterMode(display=False)
        with fcm:
            fwd(model, h1, l1, keep)
        gpk.append(fcm.get_total_flops() / 1e9)
    return gpk


def srank(a):
    o = a.argsort(kind='mergesort'); r = np.empty_like(o, float); r[o] = np.arange(len(a)); return r
def spear(a, b):
    ra, rb = srank(a), srank(b); return float(np.corrcoef(ra, rb)[0, 1])
def partial_spear(x, y, z):   # corr(x,y | z) on ranks
    rxy, rxz, ryz = spear(x, y), spear(x, z), spear(y, z)
    den = np.sqrt(max(1e-9, (1 - rxz**2) * (1 - ryz**2)))
    return (rxy - rxz * ryz) / den


UNITS = {'daliahar': [0, 1, 2], 'pamap2': [0, 1, 2], 'dsads': [0, 1, 2, 3], 'iemocap': [0, 1, 2], 'mosei': [7, 21, 365]}

if __name__ == "__main__":
    OUT = {}
    for ds in args.ds:
        print(f"\n{'='*64}\n=== {ds} ===\n{'='*64}")
        acc_R = []; acc_U = []; acc_S = []; acc_ek = []; acc_msp = []; acc_H1 = []; acc_pat = []; acc_cor = []; acc_pred = []; acc_y = []; gpk = None
        for unit in UNITS[ds]:
            model, (trb, vb, teb), mods, NUMC = setup(ds, unit); M = len(mods); pairs = list(combinations(range(M), 2))
            full = acc(model, vb, mods); loo = {m: full - acc(model, vb, [x for x in mods if x != m]) for m in mods}
            order = sorted(mods, key=lambda m: -loo[m])
            ztr = extract_z(model, trb, mods); zte = extract_z(model, teb, mods)
            knife = train_knife(ztr, mods)
            full_sm, sing, pair, labels = subsets(model, teb, mods, pairs)
            R, U, S = rus(zte, full_sm, sing, pair, labels, knife, mods, pairs)
            ek, msp1, H1, pat, correct, preds_k, y_k = exit_and_deployable(model, teb, mods, order)
            if gpk is None:
                gpk = gflops_per_k(model, teb, mods, order)
            acc_R.append(R); acc_U.append(U); acc_S.append(S); acc_ek.append(ek); acc_msp.append(msp1); acc_H1.append(H1); acc_pat.append(pat); acc_cor.append(correct); acc_pred.append(preds_k); acc_y.append(y_k)
            print(f"  unit {unit}: order={order} mean exit-k={ek.mean():.2f} (M={M})")
            del model; torch.cuda.empty_cache()
        R, U, S, ek, msp, H1, pat = (np.concatenate(x) for x in (acc_R, acc_U, acc_S, acc_ek, acc_msp, acc_H1, acc_pat))
        correct = np.vstack(acc_cor)   # [N, M] per-prefix correctness (val-LOO order)
        preds_km = np.vstack(acc_pred)  # [N, M] per-prefix predicted class (val-LOO order)
        y_all = np.concatenate(acc_y)   # [N] true labels
        os.makedirs(os.path.join(_HERE, 'lsmi_exit'), exist_ok=True)
        np.savez(os.path.join(_HERE, 'lsmi_exit', f'arrays_{SUF}{ds}.npz'), R=R, U=U, S=S, exit_k=ek, msp1=msp,
                 entropy1=H1, patience_k=pat, correct=correct, preds_km=preds_km, y_all=y_all, gflops_per_k=np.array(gpk), M=len(acc_cor[0][0]))
        print(f"  pooled N={len(ek)}")
        print(f"  Spearman vs exit-k:  R={spear(R,ek):+.3f}  U={spear(U,ek):+.3f}  S={spear(S,ek):+.3f}  | "
              f"maxprob@1={spear(msp,ek):+.3f}  entropy@1={spear(H1,ek):+.3f}  patience-k={spear(pat,ek):+.3f}")
        print(f"  PARTIAL Spearman(R, exit-k | control):  |patience-k={partial_spear(R,ek,pat):+.3f}  |maxprob@1={partial_spear(R,ek,msp):+.3f}")
        print(f"  PARTIAL Spearman(patience-k, exit-k | R)={partial_spear(pat,ek,R):+.3f}   (asymmetry test)")
        OUT[ds] = {'N': len(ek), 'spear': {'R': spear(R, ek), 'U': spear(U, ek), 'S': spear(S, ek),
                   'maxprob1': spear(msp, ek), 'entropy1': spear(H1, ek), 'patience_k': spear(pat, ek)},
                   'partial_R_given_patience': partial_spear(R, ek, pat), 'partial_R_given_maxprob': partial_spear(R, ek, msp),
                   'partial_patience_given_R': partial_spear(pat, ek, R)}
    os.makedirs(os.path.join(_HERE, 'lsmi_exit'), exist_ok=True)
    json.dump(OUT, open(os.path.join(_HERE, 'lsmi_exit', f'redundancy_vs_patience_{SUF or "ps"}.json'), 'w'), indent=2)
    print('\nsaved -> results_3p3/lsmi_exit/redundancy_vs_patience.json')
