"""Faithful original DynMM (Xue & Marculescu, CVPR'23) — ONE trainer for all 6 datasets
via --dataset, running BOTH DynMM stages in sequence.

Phase I  : train the 3 nested modality-subset experts (branches 1mod / partial / full from
           common/dynmm_branches.json[dataset]); each an OrigDynMMExpertG, CE loss, best-val.
           Each trained expert is kept frozen in memory AND saved under
           <results_dir>/<exp_name>/experts/<br>/.
Phase II : build OrigGateG + OrigDynMM3BranchG over the 3 frozen experts; measure per-branch
           GFLOPs with FlopCounterMode in TRAIN mode (eval mode mis-counts
           nn.TransformerEncoderLayer); train ONLY the DiffSoftmax gate with CE + reg*expected_FLOP;
           eval HARD argmax routing -> acc/f1 + branch usage + mean GFLOPs.

NO distillation (the only auxiliary loss is the FLOP-resource reg). lr=1e-4 + grad-clip 1.0
(these FFN-2048 transformers diverge at 1e-3).

  python training/dynmm.py --dataset iemocap  --fold 0 --results_dir ./results_dynmm
  python training/dynmm.py --dataset daliahar --fold 0 --results_dir ./results_dynmm
  python training/dynmm.py --dataset mosei    --fold 0 --results_dir ./results_dynmm
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
from models.dynmm import OrigDynMMExpertG, OrigGateG, OrigDynMM3BranchG

BRANCHES = ['1mod', 'partial', 'full']
_BRANCHES_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'common', 'dynmm_branches.json')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True,
                choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='dynmm')
ap.add_argument('--expert_epochs', type=int, default=30)
ap.add_argument('--gate_epochs', type=int, default=40)
ap.add_argument('--reg', type=float, default=0.1)
ap.add_argument('--tau', type=float, default=1.0)
ap.add_argument('--embed_dim', type=int, default=60)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=2)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject')
args = ap.parse_args()
set_seed(args.seed)
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')

exp_dir = os.path.join(args.results_dir, args.exp_name)
os.makedirs(exp_dir, exist_ok=True)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)
BR_MODS = json.load(open(_BRANCHES_JSON))[args.dataset]


# ============================================================================================
# Phase I — train the 3 nested modality-subset experts
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
                             input_mode=in_mode, embed_dim=args.embed_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.expert_epochs)
    ce = nn.CrossEntropyLoss()
    best = {'va': -1, 'state': None, 'ta': 0, 'tf': 0}
    for ep in range(args.expert_epochs):
        model.train()
        for b in tl:
            x, y = unpack(b, dev); opt.zero_grad(); ce(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        va = val_expert(model); ta, tf = evaluate_expert(model)
        if va > best['va']:
            best = {'va': va, 'state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    'ta': ta, 'tf': tf}
        print(f'[expert:{br}] ep{ep} val_acc={va:.4f} test_acc={ta:.4f}', flush=True)
    # save frozen expert
    br_dir = os.path.join(exp_dir, 'experts', br); os.makedirs(br_dir, exist_ok=True)
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


print(f'=== Phase I: training {len(BRANCHES)} experts ({args.dataset} f{args.fold}) ===', flush=True)
experts = [train_expert(br) for br in BRANCHES]


# ============================================================================================
# Phase II — per-branch GFLOPs + DiffSoftmax gate (CE + reg*expected_FLOP), hard-routing eval
# ============================================================================================
# per-branch GFLOPs in TRAIN mode (FlopCounterMode in eval mode mis-counts TransformerEncoderLayer)
x0, _ = unpack(next(iter(el)), dev); x1 = x0[:1]
flop_vec = []
for e in experts:
    e.train(); fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        e(x1)
    flop_vec.append(fc.get_total_flops() / 1e9); e.eval()
print(f'[flops] {args.dataset} per-branch GFLOPs={[round(g, 4) for g in flop_vec]}', flush=True)

gate = OrigGateG(cfg.modalities, cfg.variates, len(BRANCHES), input_mode=in_mode, gate_dim=10)
model = OrigDynMM3BranchG(experts, gate, flop_vec, tau=args.tau, hard=True).to(dev)
opt = torch.optim.AdamW(model.gate.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.gate_epochs)
ce = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate_gate(loader):
    model.eval(); P, Y, W = [], [], []
    for b in loader:
        x, y = unpack(b, dev); out, w = model(x)
        P.append(out.argmax(-1).cpu()); Y.append(y.cpu()); W.append(w.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy(); W = torch.cat(W).numpy()
    usage = W.mean(0)
    return (P == Y).mean(), f1_score(Y, P, average='macro'), usage, float((usage * np.array(flop_vec)).sum())


print(f'=== Phase II: training DiffSoftmax gate (reg={args.reg}) ===', flush=True)
best = {'va': -1, 'ta': 0, 'tf': 0, 'usage': None, 'mg': 0}
for ep in range(args.gate_epochs):
    model.train(); [e.eval() for e in model.experts]
    for b in tl:
        x, y = unpack(b, dev); opt.zero_grad()
        out, reg = model(x, additional_loss=True)
        (ce(out, y) + args.reg * reg).backward()
        torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.0); opt.step()
    sched.step()
    va, vf, _, _ = evaluate_gate(vl); ta, tf, us, mg = evaluate_gate(el)
    if va > best['va']:
        best = {'va': va, 'ta': ta, 'tf': tf, 'usage': us.tolist(), 'mg': mg}
    print(f'[gate] ep{ep} val_acc={va:.4f} test_acc={ta:.4f} '
          f'usage={us.round(2).tolist()} GFLOPs={mg:.3f}', flush=True)

json.dump({'dataset': args.dataset, 'method': 'dynmm_orig_3branch', 'fold': args.fold,
           'branches': BRANCHES,
           'best_test_acc': float(best['ta']), 'best_test_f1': float(best['tf']),
           'best_val_acc': float(best['va']),
           'usage_1mod_partial_full': best['usage'], 'mean_gflops': best['mg'],
           'per_branch_gflops': flop_vec, 'reg': args.reg},
          open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
print(f"[DynMM] {args.dataset} f{args.fold} test={best['ta']:.4f}/{best['tf']:.4f} "
      f"usage={best['usage']} GFLOPs={best['mg']:.3f} -> {exp_dir}", flush=True)
