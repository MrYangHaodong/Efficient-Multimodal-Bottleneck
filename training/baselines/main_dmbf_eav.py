"""DMBF-style baseline on EAV (EEG/Audio/Video, 5-class emotion), comparable to the
other EAV baselines. Per-modality MSP-confidence gating + spatial-temporal attention
fusion + CRL. Cross-subject 3-fold (split_id=fold). Same results-json schema.

  python main_dmbf_eav.py --fold 0 --cuda_pick cuda:0
  python main_dmbf_eav.py --aggregate
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
from datetime import datetime
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from utils.helper_function import set_seed, count_model_parameters          # noqa: E402
from data.EAV.get_data import (                                             # noqa: E402
    get_dataloader as eav_get_dataloader, collate_fn as eav_collate,
    FEATURE_DIMS, EAVDataset, NUM_CLASSES)
from multimodal_model.dmbf_baseline import (                                 # noqa: E402
    DMBFClassifier, CorrectnessHistory, dmbf_loss)

ap = argparse.ArgumentParser(description='DMBF-style on EAV')
ap.add_argument('--exp_name', default='dmbf_eav')
ap.add_argument('--results_dir', default='./results_eav/dmbf')
ap.add_argument('--dataset', default='EAV')
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--seed_num', type=int, default=42)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--modalities', nargs='+', default=list(EAVDataset.ALL_MODALITIES),
                choices=list(EAVDataset.ALL_MODALITIES))
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--split_mode', default='cross_subject', choices=['cross_subject', 'random'])
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--dropout', type=float, default=0.2)
ap.add_argument('--rank_weight', type=float, default=0.1, help='CRL weight (DMBF default 0.1)')
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--weight_decay', type=float, default=5e-4)
ap.add_argument('--clip_grad', type=float, default=1.0)
ap.add_argument('--fold', type=int, default=None, choices=[0, 1, 2], help='cross-subject split id')
ap.add_argument('--data_root', default='/files1/haodong/data/EAV')
ap.add_argument('--aggregate', action='store_true', default=False)
args = ap.parse_args()

device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
exp_name_full = f"{datetime.now().strftime('%Y-%m-%d')}_{args.dataset}_{args.exp_name}"
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)

# ---------------- aggregate mode ----------------
if args.aggregate:
    folds = []
    for k in range(3):
        fp = os.path.join(exp_dir, f'results_fold{k}.json')
        if os.path.exists(fp):
            folds.append(json.load(open(fp)))
    agg = {'experiment_name': exp_name_full, 'folds': folds}
    for key in ('best_val_acc', 'best_val_f1', 'best_test_acc', 'best_test_f1'):
        vals = [fr[key] for fr in folds if key in fr]
        if vals:
            agg[f'{key}_mean'] = float(np.mean(vals)); agg[f'{key}_std'] = float(np.std(vals))
    json.dump(agg, open(os.path.join(exp_dir, 'results.json'), 'w'), indent=4)
    print(f'Aggregated -> {exp_dir}/results.json')
    sys.exit(0)

set_seed(args.seed_num)
modalities = list(args.modalities)
M = len(modalities)
fold = args.fold if args.fold is not None else 0
print(f'Device {device} | {exp_name_full} | Modalities {modalities} | fold {fold} | {NUM_CLASSES}-class')

train_loader, val_loader, eval_loader = eav_get_dataloader(
    data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
    modalities=modalities, split_id=fold, train_shuffle=True, max_seq_len=args.max_seq_len,
    time_compression_ratio=1, use_batched_fusion=True, split_mode=args.split_mode)
input_length = args.max_seq_len


# ---------------- indexed train loader (stable sample idx for CRL) ----------------
class _IdxDS(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i): return self.base[i] + [i]   # features...+[label]+[idx]


def _idx_collate(batch):
    idxs = [b[-1] for b in batch]
    out = eav_collate([b[:-1] for b in batch])            # (padded..., label(B,1))
    return out + (torch.tensor(idxs, dtype=torch.long),)


train_idx_loader = DataLoader(
    _IdxDS(train_loader.dataset), batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, drop_last=True, collate_fn=_idx_collate, pin_memory=True)
n_train = len(train_loader.dataset)


def _unpack(batch, with_idx=False):
    if with_idx:
        *feats, label, idx = batch
        idx = idx.to(device)
    else:
        *feats, label = batch
        idx = None
    highs = [feats[2 * j].to(device).float() for j in range(M)]   # modality-j high feature
    label = label.long().to(device)
    label = label.squeeze(-1) if label.ndim > 1 else label
    return highs, label, idx


model = DMBFClassifier(modalities, {m: FEATURE_DIMS[m] for m in modalities}, NUM_CLASSES,
                       input_length, d_model=args.d_model, nhead=args.nhead,
                       num_layers=args.num_layers, dropout=args.dropout).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(device)
history = CorrectnessHistory(n_train)
print(f'Parameters: {count_model_parameters(model)}')


@torch.no_grad()
def evaluate(loader):
    model.eval(); preds, ys = [], []
    for batch in loader:
        highs, y, _ = _unpack(batch)
        logits = model(highs)['logits']
        preds.append(logits.argmax(1).cpu().numpy()); ys.append(y.cpu().numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


best = {'val_acc': -1, 'val_f1': 0, 'epoch': -1, 'state': None}
for epoch in range(args.num_epochs):
    model.train()
    for batch in train_idx_loader:
        highs, y, idx = _unpack(batch, with_idx=True)
        out = model(highs)
        total, _, _, _ = dmbf_loss(out, y, history, idx, ranking_criterion, device,
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
    va, vf = evaluate(val_loader)
    if epoch % 5 == 0 or epoch == args.num_epochs - 1:
        print(f'  ep{epoch:03d} val_acc={va:.4f} val_f1={vf:.4f}')
    if va > best['val_acc']:
        best = {'val_acc': va, 'val_f1': vf, 'epoch': epoch,
                'state': copy.deepcopy(model.state_dict())}

model.load_state_dict(best['state'])
test_acc, test_f1 = evaluate(eval_loader)
print(f'\nBest val (ep{best["epoch"]}) val_acc={best["val_acc"]:.4f} | '
      f'TEST acc={test_acc:.4f} f1={test_f1:.4f}')

ckpt = f'best_model_fold{args.fold}.pth' if args.fold is not None else 'best_val_model.pth'
torch.save(best['state'], os.path.join(exp_dir, ckpt))
res = {'experiment_name': exp_name_full, 'model_variant': 'dmbf_eav', 'fold': args.fold,
       'seed': args.seed_num, 'modalities': modalities, 'num_classes': NUM_CLASSES,
       'best_val_acc': best['val_acc'], 'best_val_f1': best['val_f1'],
       'best_epoch_val': best['epoch'],
       'best_test_acc': float(test_acc), 'best_test_f1': float(test_f1),
       'd_model': args.d_model, 'num_layers': args.num_layers, 'rank_weight': args.rank_weight,
       'lr': args.lr, 'weight_decay': args.weight_decay}
fn = f'results_fold{args.fold}.json' if args.fold is not None else 'results.json'
json.dump(res, open(os.path.join(exp_dir, fn), 'w'), indent=4)
json.dump(vars(args), open(os.path.join(exp_dir, 'config.json'), 'w'), indent=4, default=str)
print(f'Saved -> {exp_dir}/{fn}')
