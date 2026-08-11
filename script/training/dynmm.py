"""Faithful original ModalityDynMM / DynMM-b (Xue & Marculescu, CVPR'23) — ONE trainer for all
datasets via --dataset, running BOTH DynMM stages in sequence.

Faithful structure (default --branches '1mod,full'): TWO branches = a unimodal expert + a
full-fusion expert, exactly the shipped imdb_dyn.py / affect_dyn.py mix (text-only + all-modality
late fusion). --resource_mode 'expensive' (default) penalizes the mean gate weight routed to the
full-fusion branch (= original `weight[:, 1].mean()`). Pass --branches '1mod,partial,full' and/or
--resource_mode flop to recover the earlier 3-branch FLOP-weighted variant.

Phase I  : train the modality-subset experts (branches from common/dynmm_branches.json[dataset]);
           each an OrigDynMMExpertG, CE loss, best-val. Experts saved per-fold under
           <results_dir>/<exp_name>/fold<N>/experts/<br>/best_model.pth.
Phase II : build OrigGateG + OrigDynMMBranchG over the frozen experts; measure per-branch
           GFLOPs; train ONLY the DiffSoftmax gate (soft by default, matching official DynMM)
           with CE + reg*resource; final test eval uses hard argmax routing.

--phase2_only: skip Phase I, load existing per-fold expert checkpoints from disk.
--hard_gate: use straight-through hard routing during training too (non-default, deviates
             from official which trains soft and only evaluates hard).

NO distillation (the only auxiliary loss is the FLOP-resource reg). lr=1e-4 + grad-clip 1.0
(these FFN-2048 transformers diverge at 1e-3).

  python training/dynmm.py --dataset iemocap  --fold 0 --results_dir ./results_dynmm
  python training/dynmm.py --dataset utd_mhad --fold 0 --results_dir ./results_dynmm --phase2_only
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ee_data import build_loaders
from multimodal_model.dynmm import OrigDynMMExpertG, OrigGateG, OrigDynMMBranchG

_BRANCHES_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'common', 'dynmm_branches.json')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True,
                choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei',
                         'meld', 'cmi', 'czu_mhad', 'mmfi', 'utd_mhad', 'ch_simsv2'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='dynmm_faithful')
ap.add_argument('--branches', default='1mod,full',
                help="comma-separated branch keys from dynmm_branches.json; "
                     "faithful original ModalityDynMM (DynMM-b) = '1mod,full'")
ap.add_argument('--resource_mode', default='expensive', choices=['expensive', 'flop'],
                help="'expensive' = faithful (penalize weight on full-fusion branch); 'flop' = FLOP-weighted")
ap.add_argument('--expert_epochs', type=int, default=30)
ap.add_argument('--gate_epochs', type=int, default=40)
ap.add_argument('--reg', type=float, default=0.1)
ap.add_argument('--tau', type=float, default=1.0)
ap.add_argument('--hard_gate', action='store_true',
                help='Use straight-through hard routing during Phase II training too '
                     '(default: soft during training, hard only at final test eval)')
ap.add_argument('--phase2_only', action='store_true',
                help='Skip Phase I; load existing per-fold expert checkpoints from '
                     '<results_dir>/<exp_name>/fold<N>/experts/<branch>/best_model.pth')
ap.add_argument('--embed_dim', type=int, default=128)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--num_layers', type=int, default=6)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=1)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject')
ap.add_argument('--drop_warmup', type=int, default=0)
ap.add_argument('--drop_ramp', type=int, default=0)
ap.add_argument('--max_drop_prob', type=float, default=0.0)
args = ap.parse_args()
set_seed(args.seed)
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')

exp_dir  = os.path.join(args.results_dir, args.exp_name)
fold_dir = os.path.join(exp_dir, f'fold{args.fold}')
os.makedirs(fold_dir, exist_ok=True)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)

_mslices = {}; _off = 0
for _m in cfg.modalities:
    _n = cfg.variates[_m]; _mslices[_m] = (_off, _off + _n); _off += _n

def _drop_sched(ep, dw, dr, mp):
    if ep < dw or mp == 0.0: return 0.0
    if ep < dw + dr: return mp * (ep - dw) / dr
    return mp

def _apply_mod_dropout(x, dp):
    if dp <= 0.0: return x
    x = x.clone(); B = x.size(0); mods = list(_mslices)
    for b in range(B):
        drops = [m for m in mods if torch.rand(1).item() < dp]
        if len(drops) == len(mods): drops = drops[:-1]
        for m in drops:
            s, e = _mslices[m]; x[b, :, s:e] = 0.0
    return x

BR_MODS  = json.load(open(_BRANCHES_JSON))[args.dataset]
BRANCHES = [b.strip() for b in args.branches.split(',') if b.strip()]
assert all(b in BR_MODS for b in BRANCHES), f'branches {BRANCHES} not all in {list(BR_MODS)}'


# ============================================================================================
# Phase I — train per-modality-subset experts (or load from disk if --phase2_only)
# ============================================================================================
def evaluate_expert(model):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for b in el:
            x, y = unpack(b, dev); P.append(model(x).argmax(-1).cpu()); Y.append(y.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    return (P == Y).mean(), f1_score(Y, P, average='macro')


def val_expert(model):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for b in vl:
            x, y = unpack(b, dev); P.append(model(x).argmax(-1).cpu()); Y.append(y.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    return (P == Y).mean()


def train_expert(br):
    sub = BR_MODS[br]
    model = OrigDynMMExpertG(sub, cfg.modalities, cfg.variates, cfg.num_classes,
                             input_mode=in_mode, embed_dim=args.embed_dim,
                             nhead=args.nhead, num_layers=args.num_layers).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.expert_epochs)
    ce    = nn.CrossEntropyLoss()
    best  = {'va': -1, 'state': None, 'ta': 0, 'tf': 0}
    for ep in range(args.expert_epochs):
        model.train()
        for b in tl:
            x, y = unpack(b, dev)
            x = _apply_mod_dropout(x, _drop_sched(ep, args.drop_warmup, args.drop_ramp, args.max_drop_prob))
            opt.zero_grad(); ce(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        va = val_expert(model); ta, tf = evaluate_expert(model)
        if va > best['va']:
            best = {'va': va, 'state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    'ta': ta, 'tf': tf}
        print(f'[expert:{br}] ep{ep} val_acc={va:.4f} test_acc={ta:.4f}', flush=True)
    # save per-fold expert checkpoint
    br_dir = os.path.join(fold_dir, 'experts', br)
    os.makedirs(br_dir, exist_ok=True)
    torch.save(best['state'], os.path.join(br_dir, 'best_model.pth'))
    json.dump({'dataset': args.dataset, 'modalities': sub, 'num_classes': cfg.num_classes,
               'fold': args.fold, 'input_mode': in_mode, 'embed_dim': args.embed_dim,
               'best_val_acc': float(best['va']), 'best_test_acc': float(best['ta']),
               'best_test_f1': float(best['tf'])},
              open(os.path.join(br_dir, 'config.json'), 'w'), indent=2)
    model.load_state_dict(best['state']); model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[expert:{br}] {sub} best_val={best['va']:.4f} "
          f"test={best['ta']:.4f}/{best['tf']:.4f} -> {br_dir}", flush=True)
    return model


def load_expert(br):
    sub    = BR_MODS[br]
    br_dir = os.path.join(fold_dir, 'experts', br)
    ckpt   = os.path.join(br_dir, 'best_model.pth')
    assert os.path.exists(ckpt), \
        f'--phase2_only: expert checkpoint not found at {ckpt}. Run without --phase2_only first.'
    model = OrigDynMMExpertG(sub, cfg.modalities, cfg.variates, cfg.num_classes,
                             input_mode=in_mode, embed_dim=args.embed_dim,
                             nhead=args.nhead, num_layers=args.num_layers).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f'[expert:{br}] loaded from {ckpt}', flush=True)
    return model


if args.phase2_only:
    print(f'=== Phase I: loading experts from disk ({args.dataset} f{args.fold}) ===', flush=True)
    experts = [load_expert(br) for br in BRANCHES]
else:
    print(f'=== Phase I: training {len(BRANCHES)} experts ({args.dataset} f{args.fold}) ===', flush=True)
    experts = [train_expert(br) for br in BRANCHES]


# ============================================================================================
# Phase II — per-branch GFLOPs + DiffSoftmax gate
# Train with soft routing (hard=False, matching official DynMM).
# Final test evaluation switches to hard argmax routing.
# ============================================================================================
x0, _ = unpack(next(iter(el)), dev); x1 = x0[:1]
flop_vec = []
for e in experts:
    e.train(); fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        e(x1)
    flop_vec.append(fc.get_total_flops() / 1e9); e.eval()
print(f'[flops] {args.dataset} per-branch GFLOPs={[round(g, 4) for g in flop_vec]}', flush=True)

gate  = OrigGateG(cfg.modalities, cfg.variates, len(BRANCHES), input_mode=in_mode,
                  gate_dim=16, nhead=args.nhead, num_layers=args.num_layers)
model = OrigDynMMBranchG(experts, gate, flop_vec, tau=args.tau, hard=args.hard_gate,
                         resource_mode=args.resource_mode).to(dev)
opt   = torch.optim.AdamW(model.gate.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.gate_epochs)
ce    = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate_gate(loader, hard_override=False):
    """Evaluate gate. hard_override=True switches to argmax routing for this call only."""
    prev_hard = model.hard
    if hard_override:
        model.hard = True
    model.eval(); P, Y, W = [], [], []
    for b in loader:
        x, y = unpack(b, dev); out, w = model(x)
        P.append(out.argmax(-1).cpu()); Y.append(y.cpu())
        W.append(torch.zeros(w.shape[0], w.shape[1])
                 .scatter_(1, w.argmax(-1, keepdim=True).cpu(), 1.0))
    model.hard = prev_hard
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy(); W = torch.cat(W).numpy()
    usage = W.mean(0)
    return (P == Y).mean(), f1_score(Y, P, average='macro'), usage, float((usage * np.array(flop_vec)).sum())


print(f'=== Phase II: training DiffSoftmax gate (soft={not args.hard_gate}, reg={args.reg}) ===',
      flush=True)
best = {'va': -1, 'ta': 0, 'tf': 0, 'usage': None, 'mg': 0}
for ep in range(args.gate_epochs):
    model.train(); [e.eval() for e in model.experts]
    for b in tl:
        x, y = unpack(b, dev); opt.zero_grad()
        out, reg = model(x, additional_loss=True)
        (ce(out, y) + args.reg * reg).backward()
        torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.0); opt.step()
    sched.step()
    # val: soft routing (training mode); test printed for monitoring only
    va, vf, _, _ = evaluate_gate(vl)
    ta, tf, us, mg = evaluate_gate(el, hard_override=True)   # test: hard argmax
    if va > best['va']:
        best = {'va': va, 'ta': ta, 'tf': tf, 'usage': us.tolist(), 'mg': mg,
                'gate_state': {k: v.detach().cpu().clone()
                               for k, v in model.gate.state_dict().items()}}
    print(f'[gate] ep{ep} val_acc={va:.4f} test_acc={ta:.4f} '
          f'usage={us.round(2).tolist()} GFLOPs={mg:.3f}', flush=True)

# Persist the trained gate (the router checkpoint the bench / deploy needs) — restore the
# best-val gate first so the saved weights match the reported usage/GFLOPs.
if best.get('gate_state') is not None:
    model.gate.load_state_dict(best['gate_state'])
gate_ckpt = os.path.join(fold_dir, 'gate.pth')
torch.save(best.get('gate_state', model.gate.state_dict()), gate_ckpt)
print(f'[DynMM] saved gate -> {gate_ckpt}', flush=True)

# Final test with hard routing (authoritative numbers)
_, tf_hard, us_hard, mg_hard = evaluate_gate(el, hard_override=True)
_, tf_soft, us_soft, mg_soft = evaluate_gate(el, hard_override=False)
print(f'[DynMM] final hard-route test: f1={tf_hard:.4f} usage={us_hard.round(3).tolist()} GFLOPs={mg_hard:.3f}',
      flush=True)
print(f'[DynMM] final soft-route test: f1={tf_soft:.4f} usage={us_soft.round(3).tolist()} GFLOPs={mg_soft:.3f}',
      flush=True)

json.dump({'dataset': args.dataset,
           'method': f'dynmm_faithful_{len(BRANCHES)}branch', 'fold': args.fold,
           'branches': BRANCHES, 'resource_mode': args.resource_mode,
           'best_test_acc': float(best['ta']), 'best_test_f1': float(best['tf']),
           'best_val_acc': float(best['va']),
           'usage': best['usage'], 'usage_branches': BRANCHES, 'mean_gflops': best['mg'],
           'per_branch_gflops': flop_vec, 'reg': args.reg,
           'hard_gate_training': args.hard_gate, 'gate_ckpt': gate_ckpt,
           'gate_dim': 16, 'nhead': args.nhead, 'num_layers': args.num_layers},
          open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
print(f"[DynMM] {args.dataset} f{args.fold} test={best['ta']:.4f}/{best['tf']:.4f} "
      f"GFLOPs={best['mg']:.3f} -> {exp_dir}", flush=True)
