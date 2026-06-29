"""DMBF baseline — unified trainer for all 6 datasets via --dataset.

Replaces training/baselines/main_dmbf_4ds.py + main_dmbf_eav.py + main_dmbf_mosei.py.
Per-modality MSP-confidence gating + spatial-temporal attention fusion + Correctness-
Ranking Loss (CRL). Same data pipeline (common.ee_data.build_loaders) as the other
clean baseline trainers (e.g. training/baselines/main_mmee_4ds.py).

  python training/dmbf.py --dataset iemocap --fold 0 --results_dir ./results_dmbf
  python training/dmbf.py --dataset daliahar --fold 0 --results_dir ./results_dmbf
  python training/dmbf.py --dataset mosei --fold 0 --results_dir ./results_dmbf
"""
import os, sys, argparse, json, warnings; warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ee_data import build_loaders
from common.loader_utils import _build_channel_map
from models.dmbf import DMBFClassifier, CorrectnessHistory, dmbf_loss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from utils.helper_function import set_seed
except Exception:
    def set_seed(seed):
        torch.manual_seed(seed); np.random.seed(seed)


ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True,
                choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--rank_weight', type=float, default=0.1, help='CRL weight (DMBF default 0.1)')
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='dmbf')
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=2)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject', choices=['random', 'cross_subject'])
args = ap.parse_args()

dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(42)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)
modalities = list(cfg.modalities)
M = len(modalities)
channel_map = _build_channel_map(modalities, cfg.variates)


# build_loaders' unpack gives a concatenated x [B,T,sum(variates)]; DMBF wants a
# PER-MODALITY LIST of tensors. Split back by cfg.variates order (sax->long, float->float).
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
                       input_mode=in_mode).to(dev).float()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(dev)
history = CorrectnessHistory(n_train)
print(f'[dmbf] {args.dataset} M={modalities} mode={in_mode} in_len={in_len} '
      f'params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')


@torch.no_grad()
def evaluate(loader):
    model.eval(); preds, ys = [], []
    for batch in loader:
        x, y = unpack(batch, dev)
        logits = model(split_modalities(x))['logits']
        preds.append(logits.argmax(1).cpu().numpy()); ys.append(y.cpu().numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


best, best_state, best_test = -1, None, (0.0, 0.0)
for epoch in range(args.num_epochs):
    model.train()
    for batch, idx in train_idx_loader:
        x, y = unpack(batch, dev)
        idx = idx.to(dev)
        out = model(split_modalities(x))
        total, _, _, _ = dmbf_loss(out, y, history, idx, ranking_criterion, dev,
                                   rank_weight=args.rank_weight)
        opt.zero_grad(); total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        best = va
        best_test = evaluate(el)        # TEST at best-val

ta, tf = best_test
exp_dir = os.path.join(args.results_dir, args.exp_name)
os.makedirs(exp_dir, exist_ok=True)
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'dmbf',
       'best_val_acc': best, 'best_test_acc': ta, 'best_test_f1': tf,
       'rank_weight': args.rank_weight}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
print(f'[dmbf] {args.dataset} TEST acc={ta:.4f} f1={tf:.4f} -> {exp_dir}')
