"""Learning-to-Route (L2R) baseline on IEMOCAP / daliahar / pamap2 / dsads.
Per-sample modality ROUTER over (M unimodal + 1 fused) paths + per-path experts,
soft-combined (L2RRoutingClassifier). MODALITY-axis per-sample routing — the
canonical "learned per-sample routing" family our paper argues collapses under
shift relative to a static val-frozen schedule.

Mirrors train/main_mmee_4ds.py (shared ee_data.build_loaders pipeline: IEMOCAP
cross-subject float; daliahar/pamap2/dsads integer SAX tokens). The router needs a
PER-MODALITY LIST, so x=(B,T,sum(variates)) is split back into a list by cfg.variates.

  python main_l2r_4ds.py --dataset iemocap  --fold 0 --cuda_pick cuda:0
  python main_l2r_4ds.py --dataset daliahar --fold 0
  python main_l2r_4ds.py --dataset dsads    --fold 0      # fold = scenario
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ee_data import build_loaders
from multimodal_model.l2r_baseline import L2RRoutingClassifier, l2r_loss

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei'])
ap.add_argument('--fold', type=int, default=None)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--weight_decay', type=float, default=5e-4)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--word_length', type=int, default=2)
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject', choices=['random', 'cross_subject'])
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--dropout', type=float, default=0.2)
ap.add_argument('--load_balance_weight', type=float, default=0.01)
ap.add_argument('--hard_routing', action='store_true', default=False, help='Gumbel hard routing in training')
ap.add_argument('--clip_grad', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', default='./results_l2r')
ap.add_argument('--exp_name', default='l2r')
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed); np.random.seed(args.seed)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)

# the router needs a PER-MODALITY LIST; ee_data's unpack gives a CONCATENATED x
# [B,T,sum(variates)]. Wrap it to split x back by cfg.variates order (contiguous).
_boundaries, _start = [], 0
for _m in cfg.modalities:
    _w = cfg.variates[_m]
    _boundaries.append((_start, _start + _w)); _start += _w


def unpack_list(batch, device):
    x, y = unpack(batch, device)                # x: (B, T, sum(variates))
    if in_mode == 'sax':
        feats = [x[..., a:b].long() for (a, b) in _boundaries]
    else:
        feats = [x[..., a:b].float() for (a, b) in _boundaries]
    return feats, y


model = L2RRoutingClassifier(cfg.modalities, dict(cfg.variates), cfg.num_classes, in_len,
                             input_mode=in_mode, d_model=args.d_model, nhead=args.nhead,
                             num_layers=args.num_layers, hidden=args.hidden,
                             dropout=args.dropout).to(dev).float()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
print(f'[l2r] {args.dataset} M={cfg.modalities} paths={model.n_paths} layers={args.num_layers} '
      f'params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')


@torch.no_grad()
def evaluate(loader):
    model.eval(); preds, ys, route = [], [], []
    for batch in loader:
        feats, y = unpack_list(batch, dev)
        out = model(feats)
        preds.append(out['logits'].argmax(1).cpu().numpy()); ys.append(y.cpu().numpy())
        route.append(out['route_probs'].mean(0).cpu().numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    route_mean = np.mean(route, axis=0).tolist()
    return float((p == y).mean()), float(f1_score(y, p, average='macro')), route_mean


best, best_state, best_route = -1, None, None
for ep in range(args.num_epochs):
    model.train()
    for batch in tl:
        feats, y = unpack_list(batch, dev)
        out = model(feats, hard=args.hard_routing)
        total, _, _ = l2r_loss(out, y, load_balance_weight=args.load_balance_weight)
        opt.zero_grad(); total.backward()
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
    sched.step()
    va, vf, route = evaluate(vl)
    if ep % 5 == 0 or ep == args.num_epochs - 1:
        print(f'  ep{ep:03d} val acc={va:.4f} f1={vf:.4f} route={[round(r,2) for r in route]}')
    if va > best:
        best = va; best_state = copy.deepcopy(model.state_dict()); best_route = route

model.load_state_dict(best_state)
ta, tf, test_route = evaluate(el)
exp_dir = os.path.join(args.results_dir, f'{args.exp_name}_{args.dataset}',
                       f'fold{args.fold}' if args.fold is not None else 'run')
os.makedirs(exp_dir, exist_ok=True)
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'l2r',
       'best_val_acc': best, 'best_test_acc': ta, 'best_test_f1': tf,
       'n_paths': model.n_paths, 'num_layers': args.num_layers,
       'load_balance_weight': args.load_balance_weight, 'hard_routing': args.hard_routing,
       'test_route_probs': test_route}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'
                                  if args.fold is not None else 'results.json'), 'w'), indent=2)
print(f'[l2r] {args.dataset} TEST acc={ta:.4f} f1={tf:.4f} | '
      f'route={[round(r,2) for r in test_route]} -> {exp_dir}')
