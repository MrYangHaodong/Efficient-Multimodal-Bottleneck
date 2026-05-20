"""V6 fusion-only training on DSADS with **dynamic / grad-norm-based**
modality-dropout curriculum.

DSADS sibling of `main_v6_daliahar_fusion_only_dyna.py`.  Same backbone
(`DualVideoBottleneckModelV6Downsample` with `no_selector=True`), same
staggered curriculum:

    epoch < 20% of num_epochs            → drop_prob = 0   (clean)
    20% <= epoch < 50%                   → linear 0 → max_drop
    epoch >= 50%                         → max_drop  (plateau)

Once drop_prob > 0, every ``--profile_update_freq`` epochs we sample
recent gradient norms (via ``ModalityGradientProfiler``) and rescale
``current_mod_probs[m] = clip( base × M × (norm_m / Σ norms), 0, 0.8 )``.

Differences vs the DaliaHAR version:
  - DSADS scenarios (idx 0..3) instead of subject folds.
  - DSADSDataset with stratified 80/20 val split from src.
  - input_length = duration * base_sample_rate (no `// 2`).

Run:
    python main_v6_dsads_fusion_only_dyna.py --fold=0 --cuda_pick=cuda:0
    python main_v6_dsads_fusion_only_dyna.py --aggregate
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed, count_model_parameters
from utils.dataset_cfg import DSADS
from data.dataset_builder import DSADSDataset
from utils.train_utils import (
    AverageMeter, ProgressMeter, FocalLoss,
    get_modality_curriculum,
    ModalityGradientProfiler,
)

from multimodal_model.v6_downsample_opt_batched import DualVideoBottleneckModelV6Downsample


# ============================ V6 cfg adaptor ============================

class _DSADSV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


# ============================ Wrapper ============================

class V6FusionOnlyDynaWrapper(nn.Module):
    """Per-modality independent Bernoulli dropout from a dict."""

    def __init__(self, v6_model, modalities, variates):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self.variates = variates
        self._mod_slices = {}
        offset = 0
        for m in modalities:
            n = variates[m]
            self._mod_slices[m] = (offset, offset + n)
            offset += n

    def _split(self, x):
        high, low = {}, {}
        for m in self.modalities:
            s, e = self._mod_slices[m]
            high[m] = x[:, :, s:e]
            low[m] = x[:, :, s:e]
        return high, low

    def forward(self, x, modality_dropout_probs=None, training=False):
        high, low = self._split(x)
        if training and modality_dropout_probs is not None:
            for _ in range(8):
                kept = [m for m in self.modalities
                        if torch.rand(1).item() >= modality_dropout_probs.get(m, 0.0)]
                if kept:
                    break
            else:
                idx = int(torch.randint(0, len(self.modalities), (1,)).item())
                kept = [self.modalities[idx]]
            high = {m: high[m] for m in kept}
            low = {m: low[m] for m in kept}
        out = self.model(high, low, training=training,
                         return_selection_info=False)
        logits = out[0] if isinstance(out, tuple) else out
        return logits


# ============================ Train / eval ============================

def train_one_epoch(loader, model, criterion, optimizer, epoch, device,
                    modality_dropout_probs, clip_grad=0.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.train()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device).float(), y.to(device)
        logits = model(x, modality_dropout_probs=modality_dropout_probs,
                       training=True)
        loss = criterion(logits, y)
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / x.size(0)
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        progress = ProgressMeter(len(loader), [loss_meter, acc_meter],
                                 prefix=f"Epoch: [{epoch}]")
        if (i % 50 == 0) or (i == len(loader) - 1):
            progress.display(i)
            if i == len(loader) - 1:
                mean_p = (float(np.mean(list(modality_dropout_probs.values())))
                          if modality_dropout_probs else 0.0)
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
                      f'Acc: {acc_meter.avg:.4f}  '
                      f'meanModDrop: {mean_p:.3f}')
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_one_epoch(loader, model, criterion, epoch, device):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device).float(), y.to(device)
        logits = model(x, modality_dropout_probs=None, training=False)
        loss = criterion(logits, y)
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

def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in ('y', 'yes', 't', 'true', '1'):
        return True
    if s in ('n', 'no', 'f', 'false', '0', ''):
        return False
    raise argparse.ArgumentTypeError(f'Boolean expected, got {v!r}')


parser = argparse.ArgumentParser(
    description='V6 fusion-only training on DSADS with grad-norm-based '
                'asymmetric modality-dropout curriculum.')

parser.add_argument('--exp_name', default='v6_fusion_only_dyna', type=str)
parser.add_argument('--results_dir', default='./results_v6_fusion_dyna_dsads/', type=str)
parser.add_argument('--dataset', default='DSADS', type=str)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)

# Modality-drop curriculum
parser.add_argument('--max_modality_drop', default=0.4, type=float,
                    help='Final per-modality base drop prob after the '
                         '20%%->50%% linear ramp.  Per-modality probs are '
                         '`base * M * (grad_norm_m / sum grad_norms)`, '
                         'clipped to [0, 0.8].')
parser.add_argument('--profile_update_freq', default=5, type=int,
                    help='Refresh the per-modality dropout dict every N epochs.')

# Optimizer / schedule
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad', default=1.0, type=float)

# 4 scenarios
parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2, 3],
                    help='DSADS scenario index (0..3).')
parser.add_argument('--aggregate', action='store_true', default=False)

parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0, type=float)
parser.add_argument('--eval_seed', default=42, type=int)
parser.add_argument('--val_split_ratio', default=0.2, type=float)

# V6 backbone knobs
parser.add_argument('--internal_dim', default=64, type=int)
parser.add_argument('--n_bottlenecks', default=8, type=int)
parser.add_argument('--fusion_layer', default=0, type=int)
parser.add_argument('--use_sparse_attn', type=_str2bool, default=True)
parser.add_argument('--sparse_attn_variant', default='opt', type=str,
                    choices=['opt', 'opt_strat', 'opt_strat_blk'])
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=4, type=int)
parser.add_argument('--per_modal_distill', action='store_true', default=False)
parser.add_argument('--per_modal_downsample_min_len', default=4, type=int)
parser.add_argument('--use_batched_fusion', action='store_true', default=True)

# DSADS data path
parser.add_argument('--data_root', default='/files1/haodong/data/DSADS/', type=str)

args = parser.parse_args()


# ============================ Setup ============================

num_epochs = args.num_epochs
device = torch.device(args.cuda_pick if torch.cuda.is_available() else "cpu")
current_date = datetime.now().strftime('%Y-%m-%d')
exp_name_full = f"{current_date}_{args.dataset}_{args.exp_name}"
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)


# ============================ Aggregation mode ============================

if args.aggregate:
    n_folds = 4
    fold_results = []
    for fold_idx in range(n_folds):
        fpath = os.path.join(exp_dir, f"results_fold{fold_idx}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                fold_results.append(json.load(f))
    aggregated = {'experiment_name': exp_name_full, 'folds': fold_results}
    for key in ('best_val_acc', 'best_val_f1', 'best_test_acc', 'best_test_f1'):
        vals = [fr[key] for fr in fold_results if key in fr]
        if vals:
            aggregated[f'{key}_mean'] = float(np.mean(vals))
            aggregated[f'{key}_std'] = float(np.std(vals))
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump(aggregated, f, indent=4)
    print(f"Aggregated -> {exp_dir}/results.json")
    for key in ('best_val_acc', 'best_val_f1', 'best_test_acc', 'best_test_f1'):
        if f'{key}_mean' in aggregated:
            print(f"  {key}: {aggregated[f'{key}_mean']:.4f} +/- "
                  f"{aggregated[f'{key}_std']:.4f}")
    sys.exit(0)


# ============================ Data ============================

set_seed(args.seed_num)
print(f"Device: {device}")
print(f"Experiment: {exp_name_full}")

dataset_cfg = DSADS()
modalities = dataset_cfg.modalities
num_m = len(modalities)
print('Modalities:', modalities)

scenario_idx = args.fold if args.fold is not None else 0
print(f"DSADS scenario index: {scenario_idx} (fold={args.fold})")
src_path = os.path.join(args.data_root, f'dsad_diversify_dict_scenario{scenario_idx}_src.pt')
trg_path = os.path.join(args.data_root, f'dsad_diversify_dict_scenario{scenario_idx}_trg.pt')

train_data = torch.load(src_path, weights_only=False)
test_data = torch.load(trg_path, weights_only=False)

tr_x, va_x, tr_y, va_y = train_test_split(
    train_data['samples'],
    train_data['labels'],
    test_size=args.val_split_ratio,
    random_state=args.seed_num,
    stratify=train_data['labels'],
)
train_split = {'samples': tr_x, 'labels': tr_y}
val_split = {'samples': va_x, 'labels': va_y}

input_length = dataset_cfg.duration * dataset_cfg.base_sample_rate

train_ds = DSADSDataset(train_split, modalities, dataset_cfg, args.transform)
val_ds = DSADSDataset(val_split, modalities, dataset_cfg, args.transform)
eval_ds = DSADSDataset(test_data, modalities, dataset_cfg, args.transform)

print(f'Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, eval={len(eval_ds)}')

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, drop_last=True)


# ============================ Model ============================

v6_cfg = _DSADSV6Cfg(dataset_cfg)

inner = DualVideoBottleneckModelV6Downsample(
    cfg=v6_cfg,
    output_dim=dataset_cfg.num_classes,
    input_length=input_length,
    d_model=args.d_model, nhead=args.nhead,
    num_layers_per_modal=args.num_layers_per_modal,
    num_layers=args.num_layers, dropout=args.dropout, verbose=True,
    video_low_dim=v6_cfg.video_low_dim, video_high_dim=v6_cfg.video_high_dim,
    use_bottleneck=True, n_bottlenecks=args.n_bottlenecks,
    fusion_layer=args.fusion_layer, use_sparse_moe=False,
    num_experts=args.num_experts, expert_k=1,
    internal_dim=args.internal_dim, bottleneck_head_pos=True,
    use_sparse_attn=args.use_sparse_attn, factor=args.base_factor,
    selector_video_source='high', encoder_video_source='high',
    no_selector=True,
    use_weighted_factor=False,
    use_triton=False,
    num_classes=dataset_cfg.num_classes,
    sparse_attn_variant=args.sparse_attn_variant,
    strat_block_size=args.strat_block_size,
    downsample_min_len=args.downsample_min_len,
    use_batched_fusion=args.use_batched_fusion,
    per_modal_distill=args.per_modal_distill,
    per_modal_downsample_min_len=args.per_modal_downsample_min_len,
)

model = V6FusionOnlyDynaWrapper(inner, modalities, dataset_cfg.variates)
model = model.to(device).float()

# Gradient-norm hook profiler.
profiler = ModalityGradientProfiler(inner, modalities)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = (FocalLoss(gamma=args.focal_gamma)
             if args.loss == 'focal' else nn.CrossEntropyLoss())
scheduler = None
if args.lr_schedule != 'constant':
    from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
    if args.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif args.lr_schedule == 'cosine_warmrestart':
        T_0 = max(int(num_epochs * 0.2), 1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2,
                                                eta_min=1e-6)

print(f'Parameters: {count_model_parameters(model)}')
print(f'LR schedule: {args.lr_schedule} (base_lr={args.lr})')
print(f'Modality drop (staggered + grad-norm asymmetric):')
print(f'   epoch [   0,  {int(num_epochs * 0.2):>3d})  -> 0')
print(f'   epoch [{int(num_epochs * 0.2):>3d}, {int(num_epochs * 0.5):>3d})  -> linear 0 -> {args.max_modality_drop}')
print(f'   epoch [{int(num_epochs * 0.5):>3d}, {num_epochs:>3d}]  -> {args.max_modality_drop} (plateau)')
print(f'   per-modality scaling: base * M * norm_m / sum(norms) (clipped to 0.8)')
print(f'Selector:      NONE (fusion-only)')


# ============================ Training loop ============================

best_val_acc, best_val_f1, best_epoch_val = 0.0, 0.0, -1
best_model_state = None
cold_best_val_acc, cold_best_val_f1, cold_best_epoch = 0.0, 0.0, -1
cold_best_model_state = None

current_mod_probs = {m: 0.0 for m in modalities}
profile_log = []

mod_drop_complete_epoch = int(num_epochs * 0.5)

writer = SummaryWriter(
    comment=f"_{exp_name_full}_fold{args.fold}"
            if args.fold is not None else f"_{exp_name_full}")

for epoch in range(num_epochs):
    mod_base = get_modality_curriculum(epoch, num_epochs, args.max_modality_drop)

    if mod_base > 0 and epoch % args.profile_update_freq == 0:
        if any(len(profiler.grad_norms[m]) > 0 for m in modalities):
            current_mod_probs = profiler.get_asymmetric_dropout_probs(mod_base)
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
        train_loader, model, criterion, optimizer, epoch, device,
        modality_dropout_probs=current_mod_probs, clip_grad=args.clip_grad)

    val_loss, val_acc, val_y, val_p = evaluate_one_epoch(
        val_loader, model, criterion, epoch, device)
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
        cold_best_model_state = model.state_dict()

    if epoch >= mod_drop_complete_epoch and val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_epoch_val = epoch
        best_model_state = model.state_dict()


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

print(f"\nModel selection: {selected_source}")
print(f"  Best : epoch {best_epoch_val}, val_acc={best_val_acc:.4f}, val_f1={best_val_f1:.4f}")
print(f"  Cold : epoch {cold_best_epoch}, val_acc={cold_best_val_acc:.4f}")

ckpt_name = (f"best_model_fold{args.fold}.pth"
             if args.fold is not None else "best_val_model.pth")
torch.save(best_model_state, os.path.join(exp_dir, ckpt_name))


# ============================ Clean test eval ============================

print("\n" + "=" * 80)
print("Test-set evaluation (clean)")
print("=" * 80)
model.load_state_dict(best_model_state)
model.eval()

_, test_acc, t_y, t_p = evaluate_one_epoch(eval_loader, model, criterion, 0, device)
test_f1 = f1_score(t_y, t_p, average='macro')
print(f"  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")

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
    'experiment_name': exp_name_full,
    'dataset': args.dataset,
    'model_variant': 'v6_fusion_only_dyna',
    'model_class': 'DualVideoBottleneckModelV6Downsample',
    'selector': 'NONE',
    'modality_drop_strategy': 'staggered_linear_base + gradient_asymmetric_scaling',
    'fold': args.fold,
    'scenario_idx': scenario_idx,
    'num_epochs': num_epochs, 'batch_size': args.batch_size,
    'seed': args.seed_num, 'eval_seed': args.eval_seed,
    'transform': args.transform,
    'val_split_ratio': args.val_split_ratio,
    'd_model': args.d_model, 'nhead': args.nhead,
    'num_layers': args.num_layers,
    'num_layers_per_modal': args.num_layers_per_modal,
    'dropout': args.dropout,
    'num_experts': args.num_experts, 'base_factor': args.base_factor,
    'internal_dim': args.internal_dim, 'n_bottlenecks': args.n_bottlenecks,
    'fusion_layer': args.fusion_layer, 'use_sparse_attn': args.use_sparse_attn,
    'sparse_attn_variant': args.sparse_attn_variant,
    'strat_block_size': args.strat_block_size,
    'downsample_min_len': args.downsample_min_len,
    'use_batched_fusion': args.use_batched_fusion,
    'per_modal_distill': args.per_modal_distill,
    'per_modal_downsample_min_len': args.per_modal_downsample_min_len,
    'max_modality_drop': args.max_modality_drop,
    'profile_update_freq': args.profile_update_freq,
    'lr': args.lr, 'lr_schedule': args.lr_schedule,
    'clip_grad': args.clip_grad,
    'modalities': modalities,
    'data_root': args.data_root,
}
with open(os.path.join(exp_dir, "config.json"), 'w') as f:
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
results_filename = (f"results_fold{args.fold}.json"
                    if args.fold is not None else "results.json")
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(model_stats, f, indent=4)

profiler.remove_hooks()

print(f"\nBest Model (Epoch {best_epoch_val}): "
      f"val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
print(f"Selected: {selected_source}")
print(f"Directory: {exp_dir}")
