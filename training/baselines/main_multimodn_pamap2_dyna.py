"""MultiModN baseline training on PAMAP2 with the V6 **dyna**
(grad-norm-based asymmetric modality-dropout) curriculum.

Structure aligned to our V6 (2 per-modality encoder layers + 4 "fusion"
layers): MultiModN's sequential shared-state fusion, where each modality's
state-update MLP has depth ``--num_layers_fus`` (default 4) and the
per-modality temporal encoder has depth ``--num_layers`` (default 2).

PAMAP2 specifics mirror ``main_flexmoe_pamap2_dyna.py``:
  * SAX tokenisation (``input_mode='sax'``).
  * Subject-disjoint 3-fold cross-validation (``--fold``); the PAMAP2
    folds are already cross-subject by construction.
  * Staggered + grad-norm-asymmetric modality-dropout curriculum.

MultiModN-specific: every modality step decodes the shared state, giving
per-prefix logits ``[B, M, C]``; the loss is the mean cross-entropy over all
M prefixes. Reported acc/F1 use the full-prefix (final-state) prediction.

Run:
    python main_multimodn_pamap2_dyna.py --fold=0 --cuda_pick=cuda:0 \
        --num_layers=2 --num_layers_fus=4
    python main_multimodn_pamap2_dyna.py --aggregate
"""
from __future__ import annotations

import sys
try:
    import torch._dynamo  # noqa: F401
except Exception as _e:  # pragma: no cover
    print(f'[warn] torch._dynamo prefetch failed: {_e}. Continuing.')
    for _m in [k for k in list(sys.modules) if k == 'torch._dynamo'
               or k.startswith('torch._dynamo.')]:
        sys.modules.pop(_m, None)

import argparse
import copy
import json
import os
import warnings
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _V6_ROOT)

from utils.helper_function import set_seed, count_model_parameters         # noqa: E402
from utils.dataset_cfg import PAMAP2                                     # noqa: E402
from data.pamap2_crosssubject import load_pamap2_subjects                                # noqa: E402
from utils.train_utils import (                                            # noqa: E402
    AverageMeter, ProgressMeter,
    get_modality_curriculum, ModalityGradientProfiler,
)
from multimodal_model.multimodn_baseline import GenericMultiModNClassifier  # noqa: E402


# ============================ Wrapper ============================

class MultiModNDynaWrapper(nn.Module):
    """Thin wrapper: forward the (B,T,C_total) SAX-token tensor and the
    per-modality drop dict to :class:`GenericMultiModNClassifier` (which
    splits modalities + applies the per-sample skip internally)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, modality_dropout_probs=None, training=False):
        return self.model(x, modality_dropout_probs=modality_dropout_probs,
                          training=training)


# ============================ Train / Eval ============================

def _to_label(y):
    y = y.long()
    if y.ndim > 1:
        y = y.argmax(dim=1)
    return y


def _prefix_loss(prefix_logits, y, label_smoothing=0.0):
    """Mean CE over all M prefixes (MultiModN per-step supervision)."""
    B, M, C = prefix_logits.shape
    flat = prefix_logits.reshape(B * M, C)
    targets = y.unsqueeze(1).expand(B, M).reshape(B * M)
    return F.cross_entropy(flat, targets, label_smoothing=label_smoothing)


def train_one_epoch(loader, model, optimizer, epoch, device,
                    modality_dropout_probs, clip_grad=1.0, label_smoothing=0.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(device).long()
        y = _to_label(y).to(device)
        final_logits, aux = model(x,
                                  modality_dropout_probs=modality_dropout_probs,
                                  training=True)
        total = _prefix_loss(aux['prefix_logits'], y, label_smoothing)
        _, predicted = torch.max(final_logits, 1)
        acc = predicted.eq(y).sum().item() / x.size(0)
        loss_meter.update(float(total.detach().cpu()), x.size(0))
        acc_meter.update(acc, x.size(0))
        optimizer.zero_grad()
        total.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if (i % 50 == 0) or (i == len(loader) - 1):
            progress = ProgressMeter(len(loader), [loss_meter, acc_meter],
                                     prefix=f'Epoch: [{epoch}]')
            progress.display(i)
            if i == len(loader) - 1:
                mean_p = (float(np.mean(list(modality_dropout_probs.values())))
                          if modality_dropout_probs else 0.0)
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
                      f'Acc: {acc_meter.avg:.4f}  meanModDrop: {mean_p:.3f}')
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_one_epoch(loader, model, epoch, device, label_smoothing=0.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    for i, (x, y) in enumerate(loader):
        x = x.to(device).long()
        y = _to_label(y).to(device)
        logits, _ = model(x, modality_dropout_probs=None, training=False)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / x.size(0)
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))
        all_preds.append(predicted.cpu())
        all_labels.append(y.cpu())
    print(f'End of Epoch {epoch} | Val Loss: {loss_meter.avg:.4f} '
          f'| Val Acc: {acc_meter.avg:.4f}')
    return (loss_meter.avg, acc_meter.avg,
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())


# ============================ Argparse ============================

parser = argparse.ArgumentParser(
    description='MultiModN on PAMAP2 with V6 grad-norm dyna modality-drop.')

parser.add_argument('--exp_name', default='multimodn_pamap2_dyna', type=str)
parser.add_argument('--results_dir',
                    default='./results_multimodn_pamap2_dyna/', type=str)
parser.add_argument('--dataset', default='PAMAP2', type=str)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# MultiModN on PAMAP2 uses SAX tokenisation. Structure aligned to V6:
# 2 encoder layers + 4 fusion (state-update) layers.
parser.add_argument('--alphabet_size', default=20, type=int)
# SAX granularity: word_length=1 -> T=512 (default), word_length=4 -> T=128.
parser.add_argument('--word_length', default=1, type=int)
parser.add_argument('--token_dim', default=16, type=int)
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=2, type=int,
                    help='per-modality temporal-encoder depth (V6 encoder).')
parser.add_argument('--num_layers_fus', default=4, type=int,
                    help='per-modality state-update MLP depth (V6 fusion).')
parser.add_argument('--num_layers_pred', default=2, type=int)
parser.add_argument('--state_size', default=0, type=int,
                    help='shared-state width; 0 -> d_model.')
parser.add_argument('--seq_random_order', action='store_true', default=False,
                    help='re-shuffle modality order each training batch '
                         '(order-robustness ablation; eval stays canonical). '
                         'Appends "_randord" to the exp tag.')
parser.add_argument('--dropout', default=0.1, type=float)

# Modality-drop curriculum
parser.add_argument('--max_modality_drop', default=0.4, type=float)
parser.add_argument('--profile_update_freq', default=5, type=int)
parser.add_argument('--dropout_signal', default='grad', type=str, choices=['grad', 'uniform'])

# Optimizer / schedule
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)

parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2, 3])
parser.add_argument('--aggregate', action='store_true', default=False)
parser.add_argument('--data_root',
                    default='/files1/haodong/data/PAMAP2/cross_subject',
                    type=str, help='(unused: PAMAP2 loads cfg.cross_subject_dir)')

args = parser.parse_args()

if args.seq_random_order:
    args.exp_name = f'{args.exp_name}_randord'


# ============================ Setup ============================

num_epochs = args.num_epochs
device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
current_date = datetime.now().strftime('%Y-%m-%d')
exp_name_full = f'{current_date}_{args.dataset}_{args.exp_name}'
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)


# ============================ Aggregation mode ============================

if args.aggregate:
    fold_results = []
    for fold_idx in range(3):
        fp = os.path.join(exp_dir, f'results_fold{fold_idx}.json')
        if os.path.exists(fp):
            with open(fp) as f:
                fold_results.append(json.load(f))
    aggregated = {'experiment_name': exp_name_full, 'folds': fold_results}
    for key in ('best_val_acc', 'best_val_f1', 'best_test_acc', 'best_test_f1'):
        vals = [fr[key] for fr in fold_results if key in fr]
        if vals:
            aggregated[f'{key}_mean'] = float(np.mean(vals))
            aggregated[f'{key}_std'] = float(np.std(vals))
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(aggregated, f, indent=4)
    print(f'Aggregated -> {exp_dir}/results.json')
    for key in ('best_val_acc', 'best_val_f1', 'best_test_acc', 'best_test_f1'):
        if f'{key}_mean' in aggregated:
            print(f"  {key}: {aggregated[f'{key}_mean']:.4f} +/- "
                  f"{aggregated[f'{key}_std']:.4f}")
    sys.exit(0)


# ============================ Data ============================

set_seed(args.seed_num)
print(f'Device: {device}')
print(f'Experiment: {exp_name_full}')

dataset_cfg = PAMAP2()
modalities = list(dataset_cfg.modalities)
num_m = len(modalities)

if args.fold is not None:
    fold_cfg = dataset_cfg.folds[args.fold]
    train_subjects, eval_subjects = fold_cfg['train_set'], fold_cfg['eval_set']
    print(f'Fold {args.fold}: train={train_subjects}, eval={eval_subjects}')
else:
    train_subjects, eval_subjects = dataset_cfg.train_set, dataset_cfg.eval_set
val_subjects = dataset_cfg.val_set

input_length = (dataset_cfg.duration * dataset_cfg.base_sample_rate) // args.word_length

sax_params = {'alphabet_size': args.alphabet_size, 'word_length': args.word_length}
train_ds = load_pamap2_subjects(dataset_cfg.cross_subject_dir, modalities, train_subjects,
                      dataset_cfg, 'sax', sax_params=sax_params)
val_ds = load_pamap2_subjects(dataset_cfg.cross_subject_dir, modalities, val_subjects,
                    dataset_cfg, 'sax', sax_params=sax_params)
eval_ds = load_pamap2_subjects(dataset_cfg.cross_subject_dir, modalities, eval_subjects,
                     dataset_cfg, 'sax', sax_params=sax_params)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers,
                          drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers,
                        drop_last=True)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers,
                         drop_last=True)


# ============================ Model ============================

inner = GenericMultiModNClassifier(
    cfg=dataset_cfg, input_length=input_length, input_mode='sax',
    alphabet_size=args.alphabet_size, token_dim=args.token_dim,
    d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
    num_layers_fus=args.num_layers_fus, num_layers_pred=args.num_layers_pred,
    dropout=args.dropout, state_size=(args.state_size or None),
    seq_random_order=args.seq_random_order,
)
model = MultiModNDynaWrapper(inner)
model = model.to(device)

profiler = ModalityGradientProfiler(inner, modalities)

optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)
scheduler = None
if args.lr_schedule != 'constant':
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, CosineAnnealingWarmRestarts)
    if args.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs,
                                      eta_min=1e-6)
    elif args.lr_schedule == 'cosine_warmrestart':
        T_0 = max(int(num_epochs * 0.2), 1)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=2, eta_min=1e-6)

print(f'Parameters: {count_model_parameters(model)}')
print(f'LR schedule: {args.lr_schedule} (base_lr={args.lr})')
print('Modality drop (staggered + grad-norm asymmetric):')
print(f'   epoch [   0, {int(num_epochs * 0.2):>3d})  -> 0')
print(f'   epoch [{int(num_epochs * 0.2):>3d}, {int(num_epochs * 0.5):>3d})  '
      f'-> linear 0 -> {args.max_modality_drop}')
print(f'   epoch [{int(num_epochs * 0.5):>3d}, {num_epochs:>3d}]  '
      f'-> {args.max_modality_drop} (plateau)')


# ============================ Training loop ============================

best_val_acc, best_val_f1, best_epoch_val = 0.0, 0.0, -1
best_model_state = None
cold_best_val_acc, cold_best_val_f1, cold_best_epoch = 0.0, 0.0, -1
cold_best_model_state = None

current_mod_probs = {m: 0.0 for m in modalities}
profile_log = []
mod_drop_complete_epoch = int(num_epochs * 0.5)

writer = SummaryWriter(
    comment=f'_{exp_name_full}_fold{args.fold}'
            if args.fold is not None else f'_{exp_name_full}')

for epoch in range(num_epochs):
    mod_base = get_modality_curriculum(epoch, num_epochs,
                                       args.max_modality_drop)

    if mod_base > 0 and epoch % args.profile_update_freq == 0:
        if any(len(profiler.grad_norms[m]) > 0 for m in modalities):
            current_mod_probs = ({m: mod_base for m in modalities} if args.dropout_signal == 'uniform' else profiler.get_asymmetric_dropout_probs(mod_base))
            snap = profiler.get_norm_snapshot()
            profile_log.append({
                'epoch': epoch, 'mod_base': mod_base,
                'grad_norms': snap,
                'mod_dropout_probs': {m: round(p, 4)
                                      for m, p in current_mod_probs.items()},
            })
        else:
            current_mod_probs = {m: mod_base for m in modalities}
    elif mod_base == 0:
        current_mod_probs = {m: 0.0 for m in modalities}

    train_loss, train_acc = train_one_epoch(
        train_loader, model, optimizer, epoch, device,
        modality_dropout_probs=current_mod_probs,
        clip_grad=args.clip_grad, label_smoothing=args.label_smoothing)

    val_loss, val_acc, val_y, val_p = evaluate_one_epoch(
        val_loader, model, epoch, device,
        label_smoothing=args.label_smoothing)
    val_f1 = f1_score(val_y, val_p, average='macro')

    current_lr = optimizer.param_groups[0]['lr']
    if scheduler is not None:
        scheduler.step()

    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/acc', train_acc, epoch)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/acc', val_acc, epoch)
    writer.add_scalar('val/f1', val_f1, epoch)
    writer.add_scalar('schedule/mod_base', mod_base, epoch)
    writer.add_scalar('schedule/lr', current_lr, epoch)
    for m, p in current_mod_probs.items():
        writer.add_scalar(f'mod_drop/{m}', p, epoch)
    if any(len(profiler.grad_norms[m]) > 0 for m in modalities):
        for m, gn in profiler.get_norm_snapshot().items():
            writer.add_scalar(f'grad_norm/{m}', gn, epoch)

    if val_acc > cold_best_val_acc:
        cold_best_val_acc = val_acc
        cold_best_val_f1 = val_f1
        cold_best_epoch = epoch
        cold_best_model_state = copy.deepcopy(model.state_dict())

    if epoch >= mod_drop_complete_epoch and val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_epoch_val = epoch
        best_model_state = copy.deepcopy(model.state_dict())


# ============================ Model selection ============================

selected_source = 'warm'
if best_model_state is None:
    best_model_state = cold_best_model_state
    best_val_acc, best_val_f1, best_epoch_val = (
        cold_best_val_acc, cold_best_val_f1, cold_best_epoch)
    selected_source = 'cold (no warm checkpoint found)'
elif cold_best_val_acc - best_val_acc > 0.02:
    best_model_state = cold_best_model_state
    best_val_acc, best_val_f1, best_epoch_val = (
        cold_best_val_acc, cold_best_val_f1, cold_best_epoch)
    selected_source = f'cold (cold {cold_best_val_acc:.4f} > warm by >2%)'

print(f'\nModel selection: {selected_source}')
print(f'  Best : epoch {best_epoch_val}, val_acc={best_val_acc:.4f}, '
      f'val_f1={best_val_f1:.4f}')
print(f'  Cold : epoch {cold_best_epoch}, val_acc={cold_best_val_acc:.4f}')

ckpt_name = (f'best_model_fold{args.fold}.pth'
             if args.fold is not None else 'best_val_model.pth')
torch.save(best_model_state, os.path.join(exp_dir, ckpt_name))


# ============================ Clean test eval ============================

print('\n' + '=' * 80)
print('Test-set evaluation (clean)')
print('=' * 80)
model.load_state_dict(best_model_state)
model.eval()

_, test_acc, t_y, t_p = evaluate_one_epoch(
    eval_loader, model, 0, device, label_smoothing=args.label_smoothing)
test_f1 = f1_score(t_y, t_p, average='macro')
print(f'  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}')

writer.add_scalar('test/acc', test_acc, 0)
writer.add_scalar('test/f1', test_f1, 0)
writer.add_scalar('best/val_acc', best_val_acc, 0)
writer.add_scalar('best/val_f1', best_val_f1, 0)
writer.add_scalar('best/epoch', best_epoch_val, 0)
writer.close()


# ============================ Save ============================

final_grad_norms = profiler.get_norm_snapshot() \
    if any(len(profiler.grad_norms[m]) > 0 for m in modalities) else {}

config = {
    'experiment_name': exp_name_full, 'dataset': args.dataset,
    'model_variant': 'multimodn_pamap2_dyna',
    'model_class': 'GenericMultiModNClassifier',
    'modality_drop_strategy':
        'staggered_linear_base + gradient_asymmetric_scaling',
    'fold': args.fold,
    'num_epochs': num_epochs, 'batch_size': args.batch_size,
    'seed': args.seed_num,
    'transform': 'sax',
    'alphabet_size': args.alphabet_size, 'token_dim': args.token_dim,
    'd_model': args.d_model, 'nhead': args.nhead,
    'num_layers': args.num_layers, 'num_layers_fus': args.num_layers_fus,
    'num_layers_pred': args.num_layers_pred,
    'state_size': inner.state_size,
    'seq_random_order': args.seq_random_order,
    'dropout': args.dropout,
    'max_modality_drop': args.max_modality_drop,
    'profile_update_freq': args.profile_update_freq,
    'lr': args.lr, 'lr_schedule': args.lr_schedule,
    'clip_grad': args.clip_grad, 'weight_decay': args.weight_decay,
    'label_smoothing': args.label_smoothing,
    'modalities': modalities,
    'train_subjects': train_subjects, 'val_subjects': val_subjects,
    'eval_subjects': eval_subjects,
}
with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

model_stats = {
    'experiment_name': exp_name_full, 'fold': args.fold,
    'best_val_acc': best_val_acc, 'best_val_f1': best_val_f1,
    'best_epoch_val': best_epoch_val, 'selected_source': selected_source,
    'cold_best': {'val_acc': cold_best_val_acc, 'val_f1': cold_best_val_f1,
                  'epoch': cold_best_epoch},
    'best_test_acc': float(test_acc),
    'best_test_f1': float(test_f1),
    'final_grad_norms': final_grad_norms,
    'profile_log': profile_log,
}
results_filename = (f'results_fold{args.fold}.json'
                    if args.fold is not None else 'results.json')
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(model_stats, f, indent=4)

profiler.remove_hooks()

print(f'\nBest Model (Epoch {best_epoch_val}): '
      f'val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}  '
      f'test_f1={test_f1:.4f}')
print(f'Selected: {selected_source}')
print(f'Directory: {exp_dir}')
