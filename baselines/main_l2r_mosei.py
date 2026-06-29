"""Learning-to-Route (L2R) baseline on CMU-MOSEI (Acc-2), comparable to our other
MOSEI baselines. Per-sample modality router over (M unimodal + 1 fused) paths +
per-path experts, soft-combined. Same data/metrics/results-json as the others.

  python main_l2r_mosei.py --fold 0 --seed_num 42 --cuda_pick cuda:0
  python main_l2r_mosei.py --aggregate
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
from datetime import datetime
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from utils.helper_function import set_seed, count_model_parameters          # noqa: E402
from data.CMU_MOSEI.get_data import (                                        # noqa: E402
    get_dataloader as mosei_get_dataloader, FEATURE_DIMS, CMUMOSEIDataset)
from multimodal_model.l2r_baseline import L2RRoutingClassifier, l2r_loss     # noqa: E402

ap = argparse.ArgumentParser(description='Learning-to-Route on CMU-MOSEI')
ap.add_argument('--exp_name', default='l2r_mosei')
ap.add_argument('--results_dir', default='./results_l2r_mosei/')
ap.add_argument('--dataset', default='CMU_MOSEI')
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--seed_num', type=int, default=42)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--modalities', nargs='+', default=list(CMUMOSEIDataset.ALL_MODALITIES),
                choices=list(CMUMOSEIDataset.ALL_MODALITIES))
ap.add_argument('--max_seq_len', type=int, default=50)
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--dropout', type=float, default=0.2)
ap.add_argument('--load_balance_weight', type=float, default=0.01)
ap.add_argument('--hard_routing', action='store_true', default=False, help='Gumbel hard routing in training')
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--weight_decay', type=float, default=5e-4)
ap.add_argument('--clip_grad', type=float, default=1.0)
ap.add_argument('--fold', type=int, default=None, choices=[0, 1, 2], help='run index (file naming); split_id fixed 0')
ap.add_argument('--data_root', default='/files1/haodong/data/CMU-MOSI/CMU-MOSEI')
ap.add_argument('--mosei_task', type=int, default=2, choices=[2, 5, 7])
ap.add_argument('--aggregate', action='store_true', default=False)
args = ap.parse_args()

device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
exp_name_full = f"{datetime.now().strftime('%Y-%m-%d')}_{args.dataset}_{args.exp_name}"
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)
NUM_CLASSES = args.mosei_task
TASK = {2: 'classification2', 5: 'classification5', 7: 'classification7'}[args.mosei_task]

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
print(f'Device {device} | Experiment {exp_name_full} | Modalities {modalities} | task={TASK}')

train_loader, val_loader, eval_loader = mosei_get_dataloader(
    data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
    modalities=modalities, split_id=0, train_shuffle=True, max_seq_len=args.max_seq_len,
    time_compression_ratio=1, use_batched_fusion=True, task=TASK)
input_length = args.max_seq_len


def _unpack(batch):
    *feats, label = batch
    highs = [feats[2 * j].to(device).float() for j in range(M)]
    label = label.long().to(device)
    label = label.squeeze(-1) if label.ndim > 1 else label
    return highs, label


model = L2RRoutingClassifier(modalities, {m: FEATURE_DIMS[m] for m in modalities}, NUM_CLASSES,
                             input_length, d_model=args.d_model, nhead=args.nhead,
                             num_layers=args.num_layers, hidden=args.hidden,
                             dropout=args.dropout).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
print(f'Parameters: {count_model_parameters(model)} | paths={model.n_paths}')


@torch.no_grad()
def evaluate(loader):
    model.eval(); preds, ys, route = [], [], []
    for batch in loader:
        highs, y = _unpack(batch)
        out = model(highs)
        preds.append(out['logits'].argmax(1).cpu().numpy()); ys.append(y.cpu().numpy())
        route.append(out['route_probs'].mean(0).cpu().numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    route_mean = np.mean(route, axis=0).tolist()
    return float((p == y).mean()), float(f1_score(y, p, average='macro')), route_mean


best = {'val_acc': -1, 'val_f1': 0, 'epoch': -1, 'state': None}
for epoch in range(args.num_epochs):
    model.train()
    for batch in train_loader:
        highs, y = _unpack(batch)
        out = model(highs, hard=args.hard_routing)
        total, _, _ = l2r_loss(out, y, load_balance_weight=args.load_balance_weight)
        opt.zero_grad(); total.backward()
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
    sched.step()
    va, vf, route = evaluate(val_loader)
    if epoch % 5 == 0 or epoch == args.num_epochs - 1:
        print(f'  ep{epoch:03d} val_acc={va:.4f} val_f1={vf:.4f} route={[round(r,2) for r in route]}')
    if va > best['val_acc']:
        best = {'val_acc': va, 'val_f1': vf, 'epoch': epoch,
                'state': copy.deepcopy(model.state_dict())}

model.load_state_dict(best['state'])
test_acc, test_f1, test_route = evaluate(eval_loader)
print(f'\nBest val (ep{best["epoch"]}) val_acc={best["val_acc"]:.4f} | '
      f'TEST acc={test_acc:.4f} f1={test_f1:.4f} | route={[round(r,2) for r in test_route]}')

ckpt = f'best_model_fold{args.fold}.pth' if args.fold is not None else 'best_val_model.pth'
torch.save(best['state'], os.path.join(exp_dir, ckpt))
res = {'experiment_name': exp_name_full, 'model_variant': 'l2r_mosei', 'fold': args.fold,
       'seed': args.seed_num, 'modalities': modalities, 'num_classes': NUM_CLASSES,
       'best_val_acc': best['val_acc'], 'best_val_f1': best['val_f1'],
       'best_epoch_val': best['epoch'],
       'best_test_acc': float(test_acc), 'best_test_f1': float(test_f1),
       'test_route_probs': test_route, 'n_paths': model.n_paths,
       'd_model': args.d_model, 'num_layers': args.num_layers,
       'load_balance_weight': args.load_balance_weight, 'hard_routing': args.hard_routing,
       'lr': args.lr, 'weight_decay': args.weight_decay}
fn = f'results_fold{args.fold}.json' if args.fold is not None else 'results.json'
json.dump(res, open(os.path.join(exp_dir, fn), 'w'), indent=4)
json.dump(vars(args), open(os.path.join(exp_dir, 'config.json'), 'w'), indent=4, default=str)
print(f'Saved -> {exp_dir}/{fn}')
