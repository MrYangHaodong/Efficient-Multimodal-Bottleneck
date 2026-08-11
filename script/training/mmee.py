"""MMEE (Multimodal Anytime Early-Exit) — DEPTH-axis early exit. Off-ramp heads on
a multimodal fusion transformer, weighted multi-exit training, confidence-threshold
anytime exit at eval. ONE trainer for all 6 datasets via --dataset.

  python training/mmee.py --dataset iemocap  --fold 0 --results_dir ./results_mmee
  python training/mmee.py --dataset daliahar --fold 0 --results_dir ./results_mmee
  python training/mmee.py --dataset eav      --fold 0 --results_dir ./results_mmee
  python training/mmee.py --dataset dsads    --fold 0 --results_dir ./results_mmee  # fold = scenario
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ee_data import build_loaders
from multimodal_model.mmee import MMEEClassifier

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei', 'meld', 'cmi', 'czu_mhad', 'mmfi', 'utd_mhad', 'ch_simsv2'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--weight_decay', type=float, default=1e-4)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--word_length', type=int, default=1)
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject', choices=['random', 'cross_subject'])
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--num_layers', type=int, default=6)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='mmee')
ap.add_argument('--drop_warmup',   type=int,   default=0)
ap.add_argument('--drop_ramp',     type=int,   default=0)
ap.add_argument('--max_drop_prob', type=float, default=0.0)
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed); np.random.seed(args.seed)
TAUS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)

# modality dropout helpers
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

model = MMEEClassifier(cfg, in_len, input_mode=in_mode, d_model=args.d_model, nhead=args.nhead,
                       num_layers=args.num_layers).to(dev).float()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
ce = nn.CrossEntropyLoss()
EL = model.exit_layers
W = torch.tensor([e for e in EL], dtype=torch.float32, device=dev); W = W / W.sum()  # deeper exits weigh more
print(f'[MMEE] {args.dataset} M={cfg.modalities} layers={args.num_layers} exits={EL} '
      f'params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')


@torch.no_grad()
def eval_full(loader):
    model.eval(); P, Y = [], []
    for batch in loader:
        x, y = unpack(batch, dev)
        out = model(x)['exit_logits'][-1]            # final head
        P.append(out.argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


@torch.no_grad()
def exit_curve(loader):
    model.eval(); rows = []
    for tau in TAUS:
        P, Y, K = [], [], []
        for batch in loader:
            x, y = unpack(batch, dev)
            logit, ex = model.exit_inference(x, tau)
            P.append(logit.argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy()); K.append(ex.cpu().numpy())
        p, y, k = np.concatenate(P), np.concatenate(Y), np.concatenate(K)
        rows.append((float(k.mean()), float((p == y).mean()), float(f1_score(y, p, average='macro'))))
    return sorted(rows)


best, best_state = -1, None
for ep in range(args.num_epochs):
    model.train()
    for batch in tl:
        x, y = unpack(batch, dev)
        x = _apply_mod_dropout(x, _drop_sched(ep, args.drop_warmup, args.drop_ramp, args.max_drop_prob))
        logits = model(x)['exit_logits']
        loss = sum(w * ce(lg, y) for w, lg in zip(W, logits))   # weighted multi-exit loss
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sched.step()
    va, vf = eval_full(vl)
    if ep % 5 == 0 or ep == args.num_epochs - 1:
        print(f'  ep{ep:03d} val acc={va:.4f} f1={vf:.4f}')
    if va > best:
        best = va; best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
ta, tf = eval_full(el)
curve = exit_curve(el)
exp_dir = os.path.join(args.results_dir, args.exp_name)
os.makedirs(exp_dir, exist_ok=True)
torch.save(best_state, os.path.join(exp_dir, f'best_model_fold{args.fold}.pth'))
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'mmee',
       'best_val_acc': best, 'best_test_acc': ta, 'best_test_f1': tf,
       'num_layers': args.num_layers, 'exit_layers': EL,
       'exit_curve_meanLayer_acc_f1': curve, 'taus': TAUS}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
print(f'[MMEE] {args.dataset} TEST acc={ta:.4f} f1={tf:.4f} | '
      f'exit-curve acc {curve[0][1]:.3f}->{curve[-1][1]:.3f} @ meanLayer {curve[0][0]:.2f}->{curve[-1][0]:.2f} '
      f'(of {args.num_layers}) -> {exp_dir}')
