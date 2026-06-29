"""DMBF-style baseline on IEMOCAP / daliahar / pamap2 / dsads (concept port of DMBF:
per-modality MSP-confidence gating + spatial-temporal attention fusion + Correctness-
Ranking Loss). Same data pipeline (ee_data.build_loaders) as main_mmee_4ds.py /
main_crema_4ds.py; same DMBF model + CRL bookkeeping as main_dmbf_mosei.py.
See multimodal_model/dmbf_baseline.py.

  python main_dmbf_4ds.py --dataset iemocap  --fold 0 --cuda_pick cuda:0
  python main_dmbf_4ds.py --dataset daliahar --fold 0
  python main_dmbf_4ds.py --dataset dsads    --fold 0      # fold = scenario
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ee_data import build_loaders
from multimodal_model.shaspec_baseline import _build_channel_map
from multimodal_model.dmbf_baseline import DMBFClassifier, CorrectnessHistory, dmbf_loss

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei'])
ap.add_argument('--fold', type=int, default=None)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--weight_decay', type=float, default=1e-4)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--word_length', type=int, default=2)
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject', choices=['random', 'cross_subject'])
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--dropout', type=float, default=0.2)
ap.add_argument('--rank_weight', type=float, default=0.1, help='CRL weight (DMBF default 0.1)')
ap.add_argument('--clip_grad', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', default='./results_dmbf')
ap.add_argument('--exp_name', default='dmbf')
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed); np.random.seed(args.seed)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)
modalities = list(cfg.modalities)
M = len(modalities)
channel_map = _build_channel_map(modalities, cfg.variates)


# DMBF wants a PER-MODALITY LIST of tensors; build_loaders gives a concatenated
# x [B,T,sum(variates)]. Split it back by cfg.variates order (== channel_map order).
# sax -> keep tokens as long; float -> float.
def split_modalities(x):
    out = []
    for m in modalities:
        xm = x[:, :, channel_map[m]]
        out.append(xm.long() if in_mode == 'sax' else xm.float())
    return out


# ---------------- indexed train loader (stable sample idx for CRL) ----------------
class _IdxDS(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i): return (self.base[i], i)


_base_collate = tl.collate_fn if tl.collate_fn is not None else default_collate


def _idx_collate(batch):
    items = [b[0] for b in batch]
    idxs = [b[1] for b in batch]
    out = _base_collate(items)                 # same batch structure as build_loaders' loader
    return out, torch.tensor(idxs, dtype=torch.long)


train_idx_loader = DataLoader(
    _IdxDS(tl.dataset), batch_size=bs, shuffle=True, num_workers=args.num_workers,
    drop_last=True, collate_fn=_idx_collate, pin_memory=True)
n_train = len(tl.dataset)

model = DMBFClassifier(modalities, dict(cfg.variates), cfg.num_classes, in_len,
                       d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
                       dropout=args.dropout, input_mode=in_mode).to(dev).float()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(dev)
history = CorrectnessHistory(n_train)
print(f'[dmbf] {args.dataset} M={modalities} layers={args.num_layers} mode={in_mode} '
      f'in_len={in_len} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')


@torch.no_grad()
def evaluate(loader):
    model.eval(); preds, ys = [], []
    for batch in loader:
        x, y = unpack(batch, dev)
        logits = model(split_modalities(x))['logits']
        preds.append(logits.argmax(1).cpu().numpy()); ys.append(y.cpu().numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


best, best_state = -1, None
for epoch in range(args.num_epochs):
    model.train()
    for batch, idx in train_idx_loader:
        x, y = unpack(batch, dev)
        idx = idx.to(dev)
        out = model(split_modalities(x))
        total, _, _, _ = dmbf_loss(out, y, history, idx, ranking_criterion, dev,
                                   rank_weight=args.rank_weight)
        opt.zero_grad(); total.backward()
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
        with torch.no_grad():
            correct = (out['logits'].argmax(1) == y).float()
        history.correctness_update(idx, correct)
    history.max_correctness_update(epoch)
    sched.step()
    va, vf = evaluate(vl)
    if epoch % 5 == 0 or epoch == args.num_epochs - 1:
        print(f'  ep{epoch:03d} val acc={va:.4f} f1={vf:.4f}')
    if va > best:
        best = va; best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
ta, tf = evaluate(el)
exp_dir = os.path.join(args.results_dir, f'{args.exp_name}_{args.dataset}',
                       f'fold{args.fold}' if args.fold is not None else 'run')
os.makedirs(exp_dir, exist_ok=True)
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'dmbf',
       'best_val_acc': best, 'best_test_acc': ta, 'best_test_f1': tf, 'M': M,
       'd_model': args.d_model, 'num_layers': args.num_layers, 'rank_weight': args.rank_weight}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'
                                 if args.fold is not None else 'results.json'), 'w'), indent=2)
print(f'[dmbf] {args.dataset} TEST acc={ta:.4f} f1={tf:.4f} -> {exp_dir}')
