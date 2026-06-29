"""Controlled SHIFT SWEEP: does per-sample adaptive selection degrade vs a static
schedule as train-test (subject) shift grows, while an ORACLE stays flat?

Shift knob alpha in [0,1] = fraction of TRAINING data drawn from the (held-out)
test subjects. alpha=0 -> training has NONE of the test subjects (full cross-
subject shift); alpha=1 -> training is mostly the test subjects' OTHER windows
(near within-subject, low shift). Training size is FIXED; the TEST set (held-out
windows of the test subjects) is FIXED across all alpha -> only train composition
changes, no confound.

Backbone: GenericMMExitClassifier (supports top-k subset eval + per-sample exit).
At a matched compute budget we compare on the fixed test set:
  static top-k (val-frozen importance order)  | per-sample entropy early-exit
  ORACLE (per-sample smallest-correct-k)
Run: python script/shift_sweep.py --dataset daliahar --alpha 0.5 --fold 0
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
from utils.dataset_cfg import DaliaHAR, PAMAP2, WESAD                       # noqa: E402
from data.dataset_builder import HARDataset                                # noqa: E402
from data.pamap2_crosssubject import load_pamap2_subjects                  # noqa: E402
from multimodal_model.mmexit_baseline import GenericMMExitClassifier, mmexit_losses  # noqa: E402
from data.IEMOCAP.get_data import (IEMOCAPDataset, FEATURE_DIMS as IEMO_FDIMS,        # noqa: E402
                                   NUM_CLASSES as IEMO_NUMC)
import pandas as _pd                                                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='daliahar', choices=['daliahar', 'pamap2', 'wesad', 'iemocap'])
ap.add_argument('--alpha', default=0.0, type=float)
ap.add_argument('--fold', default=0, type=int)
ap.add_argument('--num_epochs', default=80, type=int)
ap.add_argument('--batch_size', default=64, type=int)
ap.add_argument('--lr', default=1e-4, type=float)
ap.add_argument('--d_model', default=64, type=int)
ap.add_argument('--max_modality_drop', default=0.4, type=float)
ap.add_argument('--fusion_input', default='pool', choices=['pool', 'token'],
                help="'pool'=MMExit-faithful (mean-pool each modality to 1 token); "
                     "'token'=fuse RAW per-modality encoder token sequences (concat M*T tokens).")
ap.add_argument('--num_fusion_layers', default=2, type=int)
ap.add_argument('--num_enc_layers', default=2, type=int, help='per-modality encoder depth')
ap.add_argument('--fusion_distill', action='store_true', default=False,
                help='token mode: halve the modality token sequence after each fusion layer '
                     '(seqA-style per-layer distill; cls bottleneck reads distilled tokens).')
ap.add_argument('--admn', action='store_true', default=False,
                help='ADMN-style adaptive ENCODER-layer experiment: a controller picks which '
                     'modalities get their 2nd encoder layer (budget B), per-sample, vs static '
                     'best-B (val-frozen). Tests whether per-sample DEPTH allocation transfers '
                     'under cross-subject shift. Requires --fusion_input token --fusion_distill.')
ap.add_argument('--admn_budget', default=-1, type=int, help='# modalities to get 2nd enc layer (-1=ceil(M/2))')
ap.add_argument('--ctrl_epochs', default=40, type=int, help='ADMN controller training epochs')
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--seed', default=0, type=int)
ap.add_argument('--out_dir', default=os.path.join(_HERE, 'results_perk_shapley', 'shift_sweep'))
args = ap.parse_args()

device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed); np.random.seed(args.seed)

IS_FLOAT = (args.dataset == 'iemocap')   # IEMOCAP = float feats-list (others = SAX int tokens)
# token-level fusion attends over M*T tokens -> O(L^2) memory; shrink the EVAL batch
# (no_grad forwards) so wesad(1280)/pamap2(2048)-token fusion doesn't OOM. Training batch
# (args.batch_size) is unchanged for comparability with pool mode.
EVAL_BS = 32 if args.fusion_input == 'token' else 256


def _inp(x):
    """Cast model input: long for SAX-tokenized HAR, float for IEMOCAP features."""
    return x.float() if IS_FLOAT else x.long()


if IS_FLOAT:
    cfg, root, transform = None, '/files1/haodong/data/IEMOCAP', 'float'
    MODS = list(IEMOCAPDataset.ALL_MODALITIES); M = len(MODS); NUMC = IEMO_NUMC
    VARIATES = {m: IEMO_FDIMS[m] for m in MODS}            # per-modality channel dims
    IEMO_MAX_T = 128
    # canonical cross_subject folds (session-disjoint): fold0 test=S5, fold1=S4, fold2=S3
    IEMO_TEST_SESSION = {0: [5], 1: [4], 2: [3]}
    _IEMO_CACHE = os.path.join(_HERE, 'results_perk_shapley',
                               f'_iemocap_session_cache_T{IEMO_MAX_T}.pt')
else:
    CFG = {'daliahar': (DaliaHAR(), '/files1/haodong/data/processed_dalia_activity', 'sax'),
           'wesad': (WESAD(), '/files1/haodong/data/WESAD/processed_wesad_activity', 'sax'),
           'pamap2': (PAMAP2(), None, 'sax_noisy')}[args.dataset]
    cfg, root, transform = CFG
    MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
    VARIATES = dict(cfg.variates)


def _iemo_load_all():
    """Build {session: (X[N,T,sumC] float32, y[N])} for IEMOCAP, concatenating the
    per-modality HIGH features along the channel dim (channel_map order = MODS order).
    Loading 4457 utterances x 6 modality files is slow, so cache to disk once."""
    if os.path.exists(_IEMO_CACHE):
        d = torch.load(_IEMO_CACHE)
        return {s: (X.float(), y) for s, (X, y) in d.items()}
    print(f'[iemocap] building session cache (one-time disk load) -> {_IEMO_CACHE}')
    dfs = [_pd.read_csv(os.path.join(root, f'{p}_cs_split0.csv')) for p in ('train', 'val', 'test')]
    master = _pd.concat(dfs).reset_index(drop=True)            # all valid samples (union of one split)
    ds = IEMOCAPDataset(data_root=root, csv_path=os.path.join(root, 'train_cs_split0.csv'),
                        modalities=MODS, max_seq_len=IEMO_MAX_T, use_batched_fusion=True)
    ds.df = master; ds.samples = master.to_dict('records')
    hi = [2 * j for j in range(M)]                            # high-feature positions in feats-list
    by = {}
    for i in range(len(ds)):
        item = ds[i]; feats, lab = item[:-1], int(item[-1])
        sess = int(ds.samples[i]['session'])
        x = torch.cat([feats[h] for h in hi], dim=-1)         # [T, sumC]
        by.setdefault(sess, ([], []))
        by[sess][0].append(x.half()); by[sess][1].append(lab)
        if (i + 1) % 500 == 0:
            print(f'  loaded {i + 1}/{len(ds)}')
    out = {s: (torch.stack(v[0]), torch.tensor(v[1])) for s, v in by.items()}  # X float16 on disk
    os.makedirs(os.path.dirname(_IEMO_CACHE), exist_ok=True)
    torch.save(out, _IEMO_CACHE)
    return {s: (X.float(), y) for s, (X, y) in out.items()}


def _all_subjects():
    if args.dataset == 'iemocap':
        return [1, 2, 3, 4, 5]
    if args.dataset in ('daliahar', 'wesad'):
        s = set()
        for f in cfg.folds:
            s |= set(f['train_set']) | set(f['eval_set'])
        s |= set(cfg.val_set)
        return sorted(s)
    return list(cfg.subjects)


def _load_subject(s):
    if args.dataset in ('daliahar', 'wesad'):
        d = HARDataset(root, MODS, [s], cfg, transform)
    else:
        d = load_pamap2_subjects(cfg.cross_subject_dir, MODS, [s], cfg, transform)
    if len(d) == 0:
        return None
    X = torch.stack([d[i][0] for i in range(len(d))])
    y = torch.tensor([int(d[i][1]) if np.ndim(d[i][1]) == 0 else int(np.argmax(d[i][1]))
                      for i in range(len(d))])
    return X, y


# preload every subject ONCE, drop empties / load errors (e.g. DaliaHAR S9)
if IS_FLOAT:
    _ALL = _iemo_load_all()
else:
    _ALL = {}
    for _s in _all_subjects():
        try:
            _r = _load_subject(_s)
        except Exception:
            _r = None
        if _r is not None:
            _ALL[_s] = _r
print(f'[load] {args.dataset}: {len(_ALL)} usable subjects: {sorted(_ALL)} '
      f'(sizes={ {s: len(v[1]) for s, v in _ALL.items()} })')


def make_split():
    subs = sorted(_ALL)
    rng = np.random.RandomState(1000 + args.fold)
    # test subjects = cfg fold eval_set (HAR) or a deterministic 1/3 (pamap2)
    if args.dataset == 'iemocap':
        test_subs = [s for s in IEMO_TEST_SESSION[args.fold] if s in _ALL]
    elif args.dataset in ('daliahar', 'wesad'):
        test_subs = [s for s in cfg.folds[args.fold]['eval_set'] if s in _ALL]
    else:
        perm = list(rng.permutation(subs)); k = max(1, len(subs) // 3)
        test_subs = perm[:k]
    other_subs = [s for s in subs if s not in set(test_subs)]
    # per-test-subject: split 50/50 into leakable pool vs fixed held-out test
    Tpool_X, Tpool_y, Ttest_X, Ttest_y = [], [], [], []
    for s in test_subs:
        X, y = _ALL[s]
        idx = rng.permutation(len(X)); h = len(X) // 2
        Tpool_X.append(X[idx[:h]]); Tpool_y.append(y[idx[:h]])
        Ttest_X.append(X[idx[h:]]); Ttest_y.append(y[idx[h:]])
    Tpool_X = torch.cat(Tpool_X); Tpool_y = torch.cat(Tpool_y)
    test_X = torch.cat(Ttest_X); test_y = torch.cat(Ttest_y)
    # other-subject pool
    OX, OY = [], []
    for s in other_subs:
        X, y = _ALL[s]; OX.append(X); OY.append(y)
    OX = torch.cat(OX); OY = torch.cat(OY)
    # TRUE fixed training size N (no size confound): with-replacement sampling so
    # n_leak + n_other == N exactly at every alpha, regardless of pool sizes.
    N = min(len(OX), 6000)
    n_leak = int(round(args.alpha * N)); n_other = N - n_leak

    def _samp(PX, PY, n):
        if n <= 0:
            return PX[:0], PY[:0]
        idx = rng.choice(len(PX), size=n, replace=(n > len(PX)))
        return PX[idx], PY[idx]
    lX, lY = _samp(Tpool_X, Tpool_y, n_leak)
    oX2, oY2 = _samp(OX, OY, n_other)
    trX = torch.cat([oX2, lX]); trY = torch.cat([oY2, lY])
    p = rng.permutation(len(trX)); trX, trY = trX[p], trY[p]
    nv = max(1, int(0.15 * len(trX)))
    valX, valY = trX[:nv], trY[:nv]; trX, trY = trX[nv:], trY[nv:]
    return (trX, trY), (valX, valY), (test_X, test_y), test_subs


(trX, trY), (valX, valY), (teX, teY), test_subs = make_split()
T = trX.shape[1]
print(f'[{args.dataset} a={args.alpha} f{args.fold}] train={len(trX)} val={len(valX)} '
      f'test={len(teX)} (test_subs={test_subs}) T={T}')

model = GenericMMExitClassifier(
    cfg=type('C', (), {'modalities': MODS, 'variates': VARIATES, 'num_classes': NUMC})(),
    input_length=T, input_mode=('float' if IS_FLOAT else 'sax'),
    d_model=(128 if IS_FLOAT else args.d_model), nhead=8,                 # IEMOCAP=128 (huge feats), HAR=64
    num_layers=args.num_enc_layers, num_fusion_layers=args.num_fusion_layers, dropout=0.1,
    fusion_input=args.fusion_input, fusion_distill=args.fusion_distill).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)


def batches(X, Y, bs, shuffle):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X) - (bs if shuffle else 0) + (0 if shuffle else 1), bs):
        j = idx[i:i + bs]
        if len(j) == 0:
            continue
        yield X[j].to(device), Y[j].to(device)


@torch.no_grad()
def prefix_logits(X, Y, order):
    """[N,M,C] logits via forward_subset(top-k) for k=1..M, on (X,Y)."""
    model.eval()
    outs = [[] for _ in range(M)]
    ys = []
    for xb, yb in batches(X, Y, EVAL_BS, False):
        xl = _inp(xb)
        for k in range(1, M + 1):
            outs[k - 1].append(model.forward_subset(xl, set(order[:k])).cpu())
        ys.append(yb.cpu())
    L = torch.stack([torch.cat(o) for o in outs], dim=1)        # [N,M,C]
    return L, torch.cat(ys).numpy()


@torch.no_grad()
def all_tokens(X):
    """Encode all M modalities for every sample -> [N, M, d] (the fusion inputs)."""
    model.eval()
    outs = []
    for i in range(0, len(X), 256):
        xl = _inp(X[i:i + 256].to(device))
        toks = torch.stack([model._encode_modality(xl[:, :, model.channel_map[m]], m)
                            for m in MODS], dim=1)               # [b,M,d]
        outs.append(toks.cpu())
    return torch.cat(outs)                                       # [N,M,d]


@torch.no_grad()
def attn_sparsity_scores(tokens_NMd):
    """Per-modality importance = the ProbSparse 'sparsity measurement'
    M-score = max_j(QK[i,j]) - mean_j(QK[i,j]) of each modality token in the
    fusion's FIRST self-attention layer, using the model's learned q/k geometry.
    (L = M+1 tokens is tiny, so all keys are used -> exact M-score, no sampling.)
    Returns [N, M]: how sharply each modality's token attends across the set."""
    model.eval()
    d = model.d_model; nh = 8; dh = d // nh
    layer0 = model.fusion.layers[0]
    Wi, bi = layer0.self_attn.in_proj_weight, layer0.self_attn.in_proj_bias
    Wq, Wk = Wi[:d], Wi[d:2 * d]; bq, bk = bi[:d], bi[d:2 * d]
    out = []
    for i in range(0, len(tokens_NMd), 512):
        t = tokens_NMd[i:i + 512].to(device)                    # [b,M,d]
        b = t.shape[0]
        toks = t + model.modality_emb.weight.unsqueeze(0)       # add modality emb (as fusion does)
        seq = torch.cat([model.cls_token.expand(b, -1, -1), toks], dim=1)   # [b, M+1, d]
        h = layer0.norm1(seq)                                   # norm_first=True
        q = F.linear(h, Wq, bq).view(b, M + 1, nh, dh).transpose(1, 2)      # [b,nh,M+1,dh]
        k = F.linear(h, Wk, bk).view(b, M + 1, nh, dh).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-1, -2)) / (dh ** 0.5)            # [b,nh,M+1,M+1]
        ms = (sc.max(-1).values - sc.mean(-1)).mean(1)          # [b, M+1] (avg over heads)
        out.append(ms[:, 1:].cpu())                             # drop cls -> [b, M]
    return torch.cat(out)                                       # [N,M]


@torch.no_grad()
def exit_head_pred(tokens_NMd, mask_NM):
    """Replicate forward_subset's k<M path (cheap exit head on masked-mean of the
    chosen modality tokens) for a PER-SAMPLE chosen mask. Matched compute w/ static@k."""
    model.eval()
    preds = []
    for i in range(0, len(tokens_NMd), 512):
        t = tokens_NMd[i:i + 512].to(device)
        msk = mask_NM[i:i + 512].to(device).unsqueeze(-1).float()
        agg = (t * msk).sum(1) / msk.sum(1).clamp(min=1.0)
        preds.append(model.exit_head(agg).argmax(-1).cpu())
    return torch.cat(preds).numpy()


# ==================== ADMN adaptive ENCODER-LAYER experiment ====================
if args.admn:
    import sys
    from multimodal_model.admn_baseline import ADMNLayerController
    assert args.fusion_input == 'token' and args.fusion_distill, \
        'ADMN experiment requires --fusion_input token --fusion_distill (the token+distill backbone)'
    yte = teY.numpy()
    n_drop = args.num_enc_layers - 1                                  # droppable encoder layers per modality
    assert n_drop >= 1, 'ADMN needs >=2 encoder layers to allocate (set --num_enc_layers 3)'
    B_budget = args.admn_budget if args.admn_budget > 0 else int(np.ceil(M * n_drop / 2))
    # 1) train backbone with RANDOM 2nd-encoder-layer LayerDrop (robust to any layer config)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
    best_val, best_state = -1, None
    for ep in range(args.num_epochs):
        model.train()
        for xb, yb in batches(trX, trY, args.batch_size, True):
            xi = _inp(xb)
            dens = torch.rand(len(xb), 1, 1, device=device)              # per-sample random keep density
            keep = (torch.rand(len(xb), M, n_drop, device=device) < dens).float()
            loss = F.cross_entropy(model.forward_layermask(xi, keep), yb.to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp, vy = [], []
            for xb, yb in batches(valX, valY, EVAL_BS, False):
                full = torch.ones(len(xb), M, n_drop, device=device)
                vp.append(model.forward_layermask(_inp(xb), full).argmax(-1).cpu()); vy.append(yb.cpu())
            va = (torch.cat(vp).numpy() == torch.cat(vy).numpy()).mean()
        if va > best_val:
            best_val = va; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # 2) train the ADMN controller (frozen backbone) at budget B
    for p in model.parameters():
        p.requires_grad_(False)
    ctrl = ADMNLayerController(model.channel_map, MODS, VARIATES, budget=B_budget, n_drop=n_drop).to(device)
    copt = torch.optim.AdamW(ctrl.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.ctrl_epochs):
        ctrl.train(); model.eval()
        temp = max(0.5, 2.0 * (1 - ep / args.ctrl_epochs))
        for xb, yb in batches(trX, trY, args.batch_size, True):
            mask, _ = ctrl(xb.float().to(device), temp=temp)
            loss = F.cross_entropy(model.forward_layermask(_inp(xb), mask), yb.to(device))
            copt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0); copt.step()
    # 3) eval: per-sample controller vs static-frozen(best-B) at MATCHED compute (exactly B 2nd-layers)
    ctrl.eval(); model.eval()

    @torch.no_grad()
    def _eval_mask(maskfn):
        preds = []
        for xb, _y in batches(teX, teY, EVAL_BS, False):
            preds.append(model.forward_layermask(_inp(xb), maskfn(xb)).argmax(-1).cpu())
        return torch.cat(preds).numpy()

    with torch.no_grad():                                                # controller's val freq -> static top-B
        freq = torch.zeros(M, n_drop, device=device)
        for xb, _y in batches(valX, valY, EVAL_BS, False):
            freq += ctrl(xb.float().to(device))[0].sum(0)                 # (M, n_drop)
    flat_idx = torch.topk(freq.flatten(), B_budget).indices
    static_vec = torch.zeros(M * n_drop, device=device); static_vec[flat_idx] = 1.0
    static_vec = static_vec.view(M, n_drop)
    ctrl_pred = _eval_mask(lambda xb: ctrl(xb.float().to(device))[0])
    static_pred = _eval_mask(lambda xb: static_vec.unsqueeze(0).expand(len(xb), -1, -1))
    full_pred = _eval_mask(lambda xb: torch.ones(len(xb), M, n_drop, device=device))
    none_pred = _eval_mask(lambda xb: torch.zeros(len(xb), M, n_drop, device=device))
    ca, sa = float((ctrl_pred == yte).mean()), float((static_pred == yte).mean())
    res = {'dataset': args.dataset, 'alpha': args.alpha, 'fold': args.fold, 'M': M, 'admn_budget': B_budget,
           'admn': {'controller_acc': ca, 'controller_f1': float(f1_score(yte, ctrl_pred, average='macro')),
                    'static_acc': sa, 'static_f1': float(f1_score(yte, static_pred, average='macro')),
                    'adv_over_static': ca - sa,
                    'full_acc': float((full_pred == yte).mean()), 'none_acc': float((none_pred == yte).mean()),
                    'static_layers': [f'{MODS[i // n_drop]}:L{int(i % n_drop) + 1}' for i in flat_idx.cpu().tolist()],
                    'n_drop': n_drop, 'best_val': float(best_val)},
           'n_train': len(trX), 'n_test': len(teX), 'test_subjects': test_subs}
    os.makedirs(args.out_dir, exist_ok=True)
    fp = os.path.join(args.out_dir, f'{args.dataset}_a{args.alpha}_f{args.fold}.json')
    json.dump(res, open(fp, 'w'), indent=2)
    print(f"  [ADMN] L={args.num_enc_layers} n_drop={n_drop} budget={B_budget}/{M * n_drop} "
          f"controller={ca:.4f} static={sa:.4f} adv={ca - sa:+.4f} | "
          f"full={res['admn']['full_acc']:.4f} none={res['admn']['none_acc']:.4f}")
    sys.exit(0)
# ================================================================================

# ---- train (multi-exit loss + ramped modality dropout) ----
best_val, best_state = -1, None
for ep in range(args.num_epochs):
    model.train()
    p = 0.0 if ep < 0.2 * args.num_epochs else \
        min(args.max_modality_drop, args.max_modality_drop * (ep - 0.2 * args.num_epochs) / (0.3 * args.num_epochs))
    for xb, yb in batches(trX, trY, args.batch_size, True):
        out = model(_inp(xb), modality_dropout_probs=p, training=True)
        loss, _ = mmexit_losses(out, yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sched.step()
    # val acc via full-modality fusion exit
    model.eval()
    with torch.no_grad():
        vp, vy = [], []
        for xb, yb in batches(valX, valY, EVAL_BS, False):
            vp.append(model(_inp(xb), training=False)['fusion_logits'].argmax(-1).cpu()); vy.append(yb.cpu())
        va = (torch.cat(vp).numpy() == torch.cat(vy).numpy()).mean()
    if va > best_val:
        best_val = va; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)

# ---- importance order from VAL (single-modality val acc) ----
model.eval()
with torch.no_grad():
    single = {}
    for m in MODS:
        pr, yy = [], []
        for xb, yb in batches(valX, valY, EVAL_BS, False):
            pr.append(model.forward_subset(_inp(xb), {m}).argmax(-1).cpu()); yy.append(yb.cpu())
        single[m] = (torch.cat(pr).numpy() == torch.cat(yy).numpy()).mean()
order = sorted(MODS, key=lambda m: -single[m])

# ---- post-hoc on FIXED test set ----
Lte, yte = prefix_logits(teX, teY, order)                       # [N,M,C]
Lval, yval = prefix_logits(valX, valY, order)
preds_te = Lte.argmax(-1).numpy()                               # [N,M]
static_acc = [(preds_te[:, k] == yte).mean() for k in range(M)]
static_f1 = [f1_score(yte, preds_te[:, k], average='macro') for k in range(M)]
# entropy exit: calibrate tau on val to a target meanK, apply to test
Pval = F.softmax(Lval, -1); Hval = -(Pval * Pval.clamp_min(1e-9).log()).sum(-1).numpy()
Pte = F.softmax(Lte, -1); Hte = -(Pte * Pte.clamp_min(1e-9).log()).sum(-1).numpy()
def exit_eval(H, preds, y, tau):
    N = len(y); ek = np.full(N, M)
    for k in range(M):
        first = (ek == M) & (H[:, k] <= tau); ek = np.where(first, k + 1, ek)
    pick = preds[np.arange(N), ek - 1]
    return (pick == y).mean(), f1_score(y, pick, average='macro'), ek.mean()
taus = np.linspace(Hte.min(), Hte.max(), 25)
exit_curve = []
for tau in taus:
    va, vf, vk = exit_eval(Hval, Lval.argmax(-1).numpy(), yval, tau)
    ta, tf, tk = exit_eval(Hte, preds_te, yte, tau)
    exit_curve.append({'tau': float(tau), 'val_meanK': float(vk), 'test_meanK': float(tk),
                       'test_acc': float(ta), 'test_f1': float(tf)})
# oracle: per-sample smallest-correct-k (anytime upper bound)
correct = (preds_te == yte[:, None])                           # [N,M]
first_ok = np.where(correct.any(1), correct.argmax(1) + 1, M)  # smallest correct k (1-indexed), else M
oracle_acc = float(correct.any(1).mean()); oracle_meanK = float(first_ok.mean())

# learned DynMM-style gate: per-modality input summary -> per-sample k, trained on VAL
# (the LEARNED per-sample selector the thesis is about; should misroute under shift)
def _summary(Xt):
    return torch.stack([Xt[:, :, model.channel_map[m]].float().mean(dim=(1, 2)) for m in MODS], dim=1)
_vp = Lval.argmax(-1).numpy(); _vc = (_vp == yval[:, None])
val_k = torch.tensor(np.where(_vc.any(1), _vc.argmax(1), M - 1), device=device)   # 0-idx smallest-correct-k
gsum = _summary(valX).to(device)
gate = nn.Sequential(nn.LayerNorm(M), nn.Linear(M, 32), nn.ReLU(), nn.Linear(32, M)).to(device)
gopt = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)
gate.train()
for _ in range(400):
    gopt.zero_grad(); F.cross_entropy(gate(gsum), val_k).backward(); gopt.step()
gate.eval()
with torch.no_grad():
    gk = gate(_summary(teX).to(device)).argmax(-1).cpu().numpy()                  # 0-idx predicted k
gate_pred = preds_te[np.arange(len(yte)), gk]
gate_acc = float((gate_pred == yte).mean()); gate_f1 = float(f1_score(yte, gate_pred, average='macro'))
gate_meanK = float((gk + 1).mean())
static_at_gate = float(np.interp(gate_meanK, np.arange(1, M + 1), static_acc))
gate_adv = gate_acc - static_at_gate                            # learned gate vs static @ matched compute

# matched-compute summary at target meanK = ceil(M/2)
budget = int(np.ceil(M / 2))
ex_at = min(exit_curve, key=lambda p: abs(p['test_meanK'] - budget))

# BUDGET-MATCHED gate: the raw gate argmax collapses to meanK~1, so a money plot at
# `budget` would compare gate@1 vs static@budget (NOT matched compute). Here we use the
# gate's SOFT expected-k and calibrate an additive shift on VAL so mean(k)=budget (exactly
# parallel to entropy-exit's tau calibration), then apply to test -> fair matched-compute.
with torch.no_grad():
    _glv = F.softmax(gate(_summary(valX).to(device)), dim=-1).cpu().numpy()      # [Nv,M]
    _glt = F.softmax(gate(_summary(teX).to(device)), dim=-1).cpu().numpy()       # [N,M]
_ks = np.arange(1, M + 1)
_sv = (_glv * _ks[None]).sum(1); _st = (_glt * _ks[None]).sum(1)                 # soft expected #mods
_mk = lambda s, d: np.clip(np.round(s + d), 1, M).astype(int)
_deltas = np.linspace(-M, M, 400)
_dstar = _deltas[int(np.argmin([abs(_mk(_sv, d).mean() - budget) for d in _deltas]))]
k_cal = _mk(_st, _dstar)                                                          # [N] 1-indexed
gate_b_pred = preds_te[np.arange(len(yte)), k_cal - 1]
gate_b_acc = float((gate_b_pred == yte).mean()); gate_b_f1 = float(f1_score(yte, gate_b_pred, average='macro'))
gate_b_meanK = float(k_cal.mean()); gate_b_adv = gate_b_acc - static_acc[budget - 1]
# matched-compute oracle: best achievable using <= budget modalities per sample
oracle_b_acc = float(correct[:, :budget].any(1).mean())

# ---- attention-sparsity selector (parameter-free, per-sample) ----------------
# Rank modalities by the ProbSparse M-score of their token in the fusion's first
# self-attn layer; keep the per-sample top-J (J=budget); predict via the SAME cheap
# exit head as static@J (matched compute). Tests whether an attention-derived
# per-sample modality choice beats the val-frozen STATIC top-J as shift grows.
tok_te = all_tokens(teX)                                         # [N,M,d]
attn_sc = attn_sparsity_scores(tok_te)                           # [N,M]
J = budget
topJ = torch.zeros(len(attn_sc), M, dtype=torch.bool)
topJ.scatter_(1, attn_sc.argsort(dim=1, descending=True)[:, :J], True)
attn_pred = exit_head_pred(tok_te, topJ)
attn_acc = float((attn_pred == yte).mean())
attn_f1 = float(f1_score(yte, attn_pred, average='macro'))
# sanity: reproduce static@J via the same exit-head path on the static top-J set
_st_mask = torch.zeros(len(yte), M, dtype=torch.bool)
_st_mask[:, [MODS.index(m) for m in order[:J]]] = True
_st_pred = exit_head_pred(tok_te, _st_mask)
attn_static_at = float((_st_pred == yte).mean())                 # static top-J, same head
attn_adv = attn_acc - attn_static_at                             # attn-select vs static @ matched J

# ---- FastSHAP: amortized per-sample modality Shapley selector -----------------
# Jethani et al. ICLR'22, adapted to MODALITY-level Shapley: an explainer net phi(x)
# predicts per-modality Shapley in ONE pass, trained by Shapley-kernel weighted LS
# (coalition sampling) + efficient normalization. Value function v(S)=P(true class |
# exit-head on masked-mean of S's tokens) -- the SAME head used for selection, so the
# Shapley is exact w.r.t. the downstream value fn. Selection: per-sample top-J by
# predicted Shapley. Tests whether DIRECTLY predicting per-sample Shapley misroutes
# under train/test shift (the gate collapsed; does an explicit Shapley predictor too?).
tok_tr = all_tokens(trX)                                          # [Ntr,M,d]
@torch.no_grad()
def _vfun(tokens, masks, y):                                      # P(y | exit-head(masked-mean S))
    msk = masks.unsqueeze(-1)
    agg = (tokens * msk).sum(1) / msk.sum(1).clamp(min=1.0)
    return F.softmax(model.exit_head(agg), -1).gather(1, y.view(-1, 1)).squeeze(1)
def _coalitions(B):                                               # [B,M], Shapley-kernel size dist
    sizes = np.arange(1, M); w = (M - 1.0) / (sizes * (M - sizes)); w = w / w.sum()
    s = torch.tensor(np.random.choice(sizes, size=B, p=w), device=device).unsqueeze(1)
    ordr = torch.rand(B, M, device=device).argsort(1)
    keep = (torch.arange(M, device=device).unsqueeze(0) < s).float()
    m = torch.zeros(B, M, device=device); m.scatter_(1, ordr, keep); return m
expl = nn.Sequential(nn.Flatten(), nn.LayerNorm(M * model.d_model),
                     nn.Linear(M * model.d_model, 128), nn.ReLU(), nn.Linear(128, M)).to(device)
xopt = torch.optim.AdamW(expl.parameters(), lr=1e-3, weight_decay=1e-4)
_full = torch.ones(1, M, device=device)
expl.train()
for _ in range(600):
    bi = np.random.randint(0, len(tok_tr), size=min(256, len(tok_tr)))
    tb = tok_tr[bi].to(device); yb = trY[bi].to(device)
    with torch.no_grad():
        v_e = _vfun(tb, torch.zeros(len(bi), M, device=device), yb)
        v_f = _vfun(tb, _full.expand(len(bi), -1), yb)
        Sm = _coalitions(len(bi)); v_S = _vfun(tb, Sm, yb)
    phi = expl(tb)                                                # [b,M] raw Shapley
    phi_eff = phi + ((v_f - v_e).unsqueeze(1) - phi.sum(1, keepdim=True)) / M   # efficient norm
    loss = F.mse_loss((phi_eff * Sm).sum(1), v_S - v_e)
    xopt.zero_grad(); loss.backward(); xopt.step()
expl.eval()
with torch.no_grad():
    shap_te = expl(tok_te.to(device)).cpu()                       # [N,M] (ranking eff-norm-invariant)
fs_topJ = torch.zeros(len(shap_te), M, dtype=torch.bool)
fs_topJ.scatter_(1, shap_te.argsort(dim=1, descending=True)[:, :J], True)
fs_pred = exit_head_pred(tok_te, fs_topJ)
fastshap_acc = float((fs_pred == yte).mean()); fastshap_f1 = float(f1_score(yte, fs_pred, average='macro'))
fastshap_adv = fastshap_acc - attn_static_at                     # vs static top-J (same head), matched J

# ---- MMExit policy (Hou et al., EuroPar'23 — mechanism VERIFIED from the PDF) -----------------
# MMExit has NO public code, but the paper states it is a backbone-AGNOSTIC exit scheme (applied
# to LF/MIM/TF/LRTF/HKT), so reproducing its POLICY on our shared backbone + identical per-exit
# logits (the same inputs every other method here gets) is a faithful, fair isolation of whether
# the per-sample exit policy transfers under cross-subject shift. Verified mechanism:
#   * exit confidence = NORMALIZED ENTROPY  H = -(1/log C) * sum_c p_c log p_c   (paper Eq 2);
#     exit at the first prefix with H <= tau. [Same signal family as entropy_exit; the
#     MMExit-specific contribution is the UoE serialization below, not the trigger.]
#   * UoE = lambda*acc - compute (Eq 3); the serialization picks the modality order maximizing
#     the sum-utility objective (Eq 5, Thm 1). The paper uses a fast phi-rank approximation
#     (Eq 7); we instead GREEDILY maximize the Eq-5 objective on VAL (marginal acc per added
#     modality; per-modality compute ~constant here) -> a faithful (>= as good) realization of
#     the same goal that also captures modality complementarity.
#   * single tau calibrated on VAL to meanK=budget, then APPLIED to test (genuine source->shift test).
#   CAVEAT: we do NOT reproduce MMExit's double-stage joint-exit TRAINING (Eqs 15-18); it rebalances
#   exit/modality training but is orthogonal to source->target confidence transfer (and the paper
#   never evaluates under distribution shift). Our backbone is modality-dropout trained, so its
#   subset exits are already reasonable. Compared to static in the SAME UoE order @ matched compute.
def _val_prefix_acc(ordlist, kk):
    pr, yy = [], []
    for xb, yb in batches(valX, valY, EVAL_BS, False):
        pr.append(model.forward_subset(_inp(xb), set(ordlist[:kk])).argmax(-1).cpu()); yy.append(yb.cpu())
    return (torch.cat(pr).numpy() == torch.cat(yy).numpy()).mean()
_remaining = list(MODS); order_uoe = []
while _remaining:                                                # greedy UoE serialization (maximize Eq-5 utility on VAL)
    _best = max(_remaining, key=lambda m: _val_prefix_acc(order_uoe + [m], len(order_uoe) + 1))
    order_uoe.append(_best); _remaining.remove(_best)
Lte_u, _ = prefix_logits(teX, teY, order_uoe)                    # [N,M,C] per-prefix logits in UoE order
Lval_u, _ = prefix_logits(valX, valY, order_uoe)
preds_te_u = Lte_u.argmax(-1).numpy()                            # [N,M]
static_acc_u = [float((preds_te_u[:, k] == yte).mean()) for k in range(M)]   # static@k in UoE order
_logC = float(np.log(NUMC))
Hval_u = (-(F.softmax(Lval_u, -1) * F.softmax(Lval_u, -1).clamp_min(1e-9).log()).sum(-1) / _logC).numpy()  # [Nv,M] norm entropy (Eq 2)
Hte_u = (-(F.softmax(Lte_u, -1) * F.softmax(Lte_u, -1).clamp_min(1e-9).log()).sum(-1) / _logC).numpy()     # [N,M]
def mmexit_eval(Hf, preds, y, tau):                              # exit at first prefix with norm-entropy <= tau (Eq 2)
    N = len(y); ek = np.full(N, M)
    for k in range(M):
        first = (ek == M) & (Hf[:, k] <= tau); ek = np.where(first, k + 1, ek)
    pick = preds[np.arange(N), ek - 1]
    return float((pick == y).mean()), float(f1_score(y, pick, average='macro')), float(ek.mean())
mm_curve = []
for tau in np.linspace(float(Hval_u.min()), float(Hval_u.max()), 25):
    va, vf, vk = mmexit_eval(Hval_u, Lval_u.argmax(-1).numpy(), yval, tau)
    ta, tf, tk = mmexit_eval(Hte_u, preds_te_u, yte, tau)
    mm_curve.append({'tau': float(tau), 'val_meanK': vk, 'test_meanK': tk, 'test_acc': ta, 'test_f1': tf})
mm_at = min(mm_curve, key=lambda p: abs(p['val_meanK'] - budget))   # calibrate thr on VAL -> apply to test
mmexit_acc = mm_at['test_acc']; mmexit_f1 = mm_at['test_f1']; mmexit_meanK = mm_at['test_meanK']
mmexit_static_at = float(np.interp(mmexit_meanK, np.arange(1, M + 1), static_acc_u))
mmexit_adv = mmexit_acc - mmexit_static_at                       # MMExit policy vs static (UoE order) @ matched compute

res = {
    'dataset': args.dataset, 'alpha': args.alpha, 'fold': args.fold, 'M': M,
    'order': order, 'best_val_acc': float(best_val),
    'budget_meanK': budget,
    'static': {'acc_at_budget': float(static_acc[budget - 1]), 'f1_at_budget': float(static_f1[budget - 1]),
               'per_k_acc': [float(a) for a in static_acc], 'per_k_f1': [float(a) for a in static_f1]},
    'entropy_exit': {'acc_at_budget': ex_at['test_acc'], 'f1_at_budget': ex_at['test_f1'],
                     'meanK': ex_at['test_meanK'], 'curve': exit_curve},
    'oracle': {'acc': oracle_acc, 'meanK': oracle_meanK},
    'gate': {'acc': gate_acc, 'f1': gate_f1, 'meanK': gate_meanK,
             'static_at_meanK': static_at_gate, 'adv_over_static': gate_adv},
    'gate_budget': {'acc': gate_b_acc, 'f1': gate_b_f1, 'meanK': gate_b_meanK,
                    'adv_over_static': gate_b_adv},
    'oracle_at_budget': {'acc': oracle_b_acc, 'meanK': float(budget)},
    'attn_select': {'acc': attn_acc, 'f1': attn_f1, 'meanK': float(J),
                    'static_at_meanK': attn_static_at, 'adv_over_static': attn_adv},
    'fastshap': {'acc': fastshap_acc, 'f1': fastshap_f1, 'meanK': float(J),
                 'static_at_meanK': attn_static_at, 'adv_over_static': fastshap_adv},
    'mmexit': {'acc': mmexit_acc, 'f1': mmexit_f1, 'meanK': mmexit_meanK,
               'static_at_meanK': mmexit_static_at, 'adv_over_static': mmexit_adv,
               'order_uoe': order_uoe, 'static_uoe_per_k': static_acc_u, 'curve': mm_curve},
    'n_train': len(trX), 'n_test': len(teX), 'test_subjects': test_subs,
}
os.makedirs(args.out_dir, exist_ok=True)
fp = os.path.join(args.out_dir, f'{args.dataset}_a{args.alpha}_f{args.fold}.json')
json.dump(res, open(fp, 'w'), indent=2)
print(f"  static@{budget}={static_acc[budget-1]:.4f}  exit@~{budget}={ex_at['test_acc']:.4f}"
      f"(mK={ex_at['test_meanK']:.2f})  gate_raw={gate_acc:.4f}(mK={gate_meanK:.2f})"
      f"  gate@{budget}={gate_b_acc:.4f}(adv={gate_b_adv:+.3f})  attn_sel@{J}={attn_acc:.4f}(adv={attn_adv:+.3f})"
      f"  fastshap@{J}={fastshap_acc:.4f}(adv={fastshap_adv:+.3f})  oracle@{budget}={oracle_b_acc:.4f}"
      f"  mmexit@~{budget}={mmexit_acc:.4f}(mK={mmexit_meanK:.2f},adv={mmexit_adv:+.3f})")
print(f"  saved -> {fp}")
