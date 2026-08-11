"""MultiModN baseline — ONE trainer for all 6 datasets via --dataset.

Faithful port of MultiModN (Swamy et al., NeurIPS 2023): per-modality temporal
encoder + a single shared state vector updated one modality at a time, with a
prediction head decoding the state after EVERY modality step.

MultiModN-specific loss: every modality step decodes the shared state, giving
per-prefix logits [B, M, C]; the loss is the MEAN cross-entropy over all M
prefixes (MultiModN's "readable after any prefix" supervision). Reported acc/F1
use the full-prefix (final-state) prediction.

  python training/multimodn.py --dataset iemocap  --fold 0 --results_dir ./out
  python training/multimodn.py --dataset daliahar --fold 0 --results_dir ./out
  python training/multimodn.py --dataset dsads    --fold 0 --results_dir ./out
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ee_data import build_loaders
from multimodal_model.multimodn import GenericMultiModNClassifier


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True,
                choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei', 'meld', 'cmi', 'czu_mhad', 'mmfi', 'utd_mhad', 'ch_simsv2'])
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
ap.add_argument('--num_layers', type=int, default=2,
                help='per-modality temporal-encoder depth (V6 encoder).')
ap.add_argument('--num_layers_fus', type=int, default=4,
                help='per-modality state-update MLP depth (V6 fusion).')
ap.add_argument('--num_layers_pred', type=int, default=2)
ap.add_argument('--dropout', type=float, default=0.1)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='multimodn')
ap.add_argument('--drop_warmup',   type=int,   default=0)
ap.add_argument('--drop_ramp',     type=int,   default=0)
ap.add_argument('--max_drop_prob', type=float, default=0.0)
ap.add_argument('--seq_random_order', action='store_true',
                help='shuffle the modality processing order every training batch (order '
                     'robustness for arbitrary-order deployment, e.g. the RL policy).')
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)

# modality-dropout schedule (fed to the model's BUILT-IN skip dropout, which masks
# the state update for a dropped modality — matching how the RL/robustness eval
# deploys the model, i.e. skip, not zero-impute).
def _drop_sched(ep, dw, dr, mp):
    if ep < dw or mp == 0.0: return 0.0
    if ep < dw + dr: return mp * (ep - dw) / dr
    return mp

model = GenericMultiModNClassifier(
    cfg, in_len, input_mode=in_mode,
    d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
    num_layers_fus=args.num_layers_fus, num_layers_pred=args.num_layers_pred,
    dropout=args.dropout, seq_random_order=args.seq_random_order).to(dev).float()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
print(f'[MultiModN] {args.dataset} M={cfg.modalities} in_mode={in_mode} '
      f'params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')


def prefix_loss(prefix_logits, y):
    """Mean CE over all M prefixes (MultiModN per-step supervision)."""
    B, M, C = prefix_logits.shape
    flat = prefix_logits.reshape(B * M, C)
    targets = y.unsqueeze(1).expand(B, M).reshape(B * M)
    return F.cross_entropy(flat, targets)


@torch.no_grad()
def eval_full(loader):
    model.eval(); P, Y = [], []
    for batch in loader:
        x, y = unpack(batch, dev)
        logits, _ = model(x, training=False)         # full-prefix (final-state) prediction
        P.append(logits.argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


best, best_state = -1, None
for ep in range(args.num_epochs):
    model.train()
    dp = _drop_sched(ep, args.drop_warmup, args.drop_ramp, args.max_drop_prob)
    for batch in tl:
        x, y = unpack(batch, dev)
        _, aux = model(x, modality_dropout_probs=dp, training=True)   # skip-based dropout
        loss = prefix_loss(aux['prefix_logits'], y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sched.step()
    va, vf = eval_full(vl)
    if ep % 5 == 0 or ep == args.num_epochs - 1:
        print(f'  ep{ep:03d} val acc={va:.4f} f1={vf:.4f}')
    if va > best:
        best = va; best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
ta, tf = eval_full(el)
exp_dir = os.path.join(args.results_dir, f'{args.exp_name}_{args.dataset}',
                       f'fold{args.fold}' if args.fold is not None else 'run')
os.makedirs(exp_dir, exist_ok=True)
torch.save(best_state, os.path.join(exp_dir, f'best_model_fold{args.fold}.pth'))
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'multimodn',
       'best_val_acc': best, 'best_test_acc': ta, 'best_test_f1': tf,
       'num_layers': args.num_layers, 'num_layers_fus': args.num_layers_fus}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'
                                 if args.fold is not None else 'results.json'), 'w'), indent=2)
print(f'[MultiModN] {args.dataset} TEST acc={ta:.4f} f1={tf:.4f} -> {exp_dir}')
