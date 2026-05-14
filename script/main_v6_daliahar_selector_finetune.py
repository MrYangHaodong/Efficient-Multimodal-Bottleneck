"""V6 stage-2 training on DaliaHAR: freeze a pretrained fusion model
and train ONLY the modality selector (IMS) on top.

Workflow:
  1. ``main_v6_daliahar_fusion_only.py`` produces a fusion-only checkpoint
     ``best_model_fold{N}.pth`` — the V6Downsample backbone trained with
     ``no_selector=True``.
  2. **This script** rebuilds V6Downsample with ``no_selector=False``,
     loads the fusion-only state_dict with ``strict=False`` (selector
     keys are newly initialised because they aren't in the ckpt), then
     freezes every non-selector parameter and trains the IMS.
  3. Selector receives gradient from
        - CE loss (through its top-K subset gate → frozen fusion → logits)
        - probe_loss (per-modality probe CE on labels)
        - diversity_loss (anti-collapse on selection weights)
        - reinforce_loss (policy gradient on top-K with EMA baseline)

Run:
    python main_v6_daliahar_selector_finetune.py \\
        --fold=0 --cuda_pick=cuda:0 \\
        --frozen_ckpt=./results_v6_fusion/2026-05-13_DaliaHAR_v6_fusion_only/best_model_fold0.pth
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import random as _random
import sys
import warnings

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed, count_model_parameters
from utils.dataset_cfg import DaliaHAR
from data.dataset_builder import HARDataset
from utils.train_utils import AverageMeter, ProgressMeter, FocalLoss

from multimodal_model.v6_downsample_opt_batched import DualVideoBottleneckModelV6Downsample


# ============================ Schedules ============================

def _modality_drop_schedule(epoch, warmup, ramp_epochs, max_drop):
    if epoch < warmup:
        return 0.0
    if ramp_epochs <= 0:
        return max_drop
    progress = min(1.0, (epoch - warmup) / ramp_epochs)
    return max_drop * progress


def _k_schedule(epoch, strategy, num_modalities, target_k,
                random_k_min=None, random_k_max=None):
    """Top-K schedule (selector-only training has no warmup gate)."""
    if strategy == 'none':
        return None, num_modalities
    if strategy == 'random_k':
        lo = random_k_min if random_k_min is not None else target_k
        hi = random_k_max if random_k_max is not None else num_modalities
        k = _random.randint(lo, hi)
        return (None if k == num_modalities else k), k
    if strategy == 'fixed_k':
        return (None if target_k >= num_modalities else target_k), target_k
    return None, num_modalities


# ============================ V6 cfg adaptor ============================

class _DaliaV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


# ============================ Wrapper ============================

class V6SelectorTrainingWrapper(nn.Module):
    """Same channel-split shim as fusion-only training, but exposes the
    selector's auxiliary losses (probe / diversity / reinforce) through
    the return tuple so the train loop can add them to the main CE.

    Crucially: at eval time we still call into the selector path so the
    model uses its learned IMS to pick top-K.
    """

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

    def forward(self, x, modality_drop_prob=0.0, training=False, labels=None):
        high, low = self._split(x)
        if training and modality_drop_prob > 0:
            for _ in range(8):
                kept = [m for m in self.modalities
                        if torch.rand(1).item() >= modality_drop_prob]
                if kept:
                    break
            else:
                idx = int(torch.randint(0, len(self.modalities), (1,)).item())
                kept = [self.modalities[idx]]
            high = {m: high[m] for m in kept}
            low = {m: low[m] for m in kept}
        result = self.model(high, low, training=training,
                            return_selection_info=True, labels=labels)
        # When training with labels: (output, primary_idx, weights, sel, aux)
        # Otherwise:                 (output, primary_idx, weights, sel)
        if len(result) == 5:
            output, _, _, _, aux = result
            return output, aux
        return result[0], {}


# ============================ Train / eval ============================

def train_one_epoch(loader, model, criterion, optimizer, epoch, device,
                    modality_drop_prob, clip_grad=0.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    aux_meter = AverageMeter('Aux', ':.4f')
    model.train()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device).float(), y.to(device)
        logits, aux_losses = model(x, modality_drop_prob=modality_drop_prob,
                                   training=True, labels=y)
        loss = criterion(logits, y)
        aux_sum = 0.0
        for v in aux_losses.values():
            loss = loss + v
            aux_sum += v.item()
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / x.size(0)
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))
        if aux_sum > 0:
            aux_meter.update(aux_sum, x.size(0))
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                clip_grad,
            )
        optimizer.step()
        progress = ProgressMeter(len(loader), [loss_meter, acc_meter],
                                 prefix=f"Epoch: [{epoch}]")
        if (i % 50 == 0) or (i == len(loader) - 1):
            progress.display(i)
            if i == len(loader) - 1:
                aux_str = f'  Aux: {aux_meter.avg:.4f}' if aux_meter.count > 0 else ''
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
                      f'Acc: {acc_meter.avg:.4f}  '
                      f'ModDrop: {modality_drop_prob:.3f}{aux_str}')
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_one_epoch(loader, model, criterion, epoch, device):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device).float(), y.to(device)
        logits, _ = model(x, modality_drop_prob=0.0, training=False)
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
    description='V6 stage-2: freeze fusion, train selector only (DaliaHAR).')

parser.add_argument('--exp_name', default='v6_selector_finetune', type=str)
parser.add_argument('--results_dir', default='./results_v6_selector/', type=str)
parser.add_argument('--frozen_ckpt', required=True, type=str,
                    help='Path to a fusion-only best_model_fold*.pth produced '
                         'by main_v6_daliahar_fusion_only.py.')
parser.add_argument('--dataset', default='DaliaHAR', type=str)
parser.add_argument('--num_epochs', default=60, type=int,
                    help='Selector-only training is much faster — 50-80 ep '
                         'is typically enough.')
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

# Modality dropout (during selector training — kept light; the fusion is
# already robust because stage-1 trained with it).
parser.add_argument('--max_modality_drop', default=0.0, type=float)
parser.add_argument('--mod_drop_warmup', default=0, type=int)
parser.add_argument('--mod_drop_ramp_epochs', default=0, type=int)

# Optimizer
parser.add_argument('--lr', default=3e-4, type=float,
                    help='Selector is small — can use a higher LR than '
                         'stage-1 (default 3e-4 vs 1e-4).')
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0, type=float)

# 3-fold CV
parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2])
parser.add_argument('--aggregate', action='store_true', default=False)
parser.add_argument('--eval_seed', default=42, type=int)

# V6 backbone knobs (must match stage-1 for state_dict to load cleanly)
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

# Selector / IMS
parser.add_argument('--selector_strategy', default='fixed_k', type=str,
                    choices=['none', 'random_k', 'fixed_k'])
parser.add_argument('--target_k', default=2, type=int)
parser.add_argument('--random_k_min', default=None, type=int)
parser.add_argument('--random_k_max', default=None, type=int)
parser.add_argument('--lambda_probe', default=0.05, type=float)
parser.add_argument('--lambda_sparsity', default=0.0, type=float)
parser.add_argument('--lambda_reinforce', default=0.1, type=float)
parser.add_argument('--lambda_diversity', default=0.0, type=float)
parser.add_argument('--use_interaction_matrix', default=False, type=_str2bool)
parser.add_argument('--use_holo_bias', default=False, type=_str2bool)
parser.add_argument('--holo_scale', default=1.0, type=float)

# Exploration (only active during training; eval is always greedy)
parser.add_argument('--exploration_mode', default='gumbel', type=str,
                    choices=['none', 'epsilon', 'gumbel'],
                    help='Selector top-K exploration: none=greedy, '
                         'epsilon=ε-greedy random subset, '
                         'gumbel=Gumbel-top-K sample from softmax(scores/τ).')
parser.add_argument('--exploration_eps_start', default=0.5, type=float,
                    help='ε at epoch 0 (ε-greedy); linearly decayed to '
                         '--exploration_eps_end over --exploration_anneal_epochs.')
parser.add_argument('--exploration_eps_end', default=0.05, type=float)
parser.add_argument('--exploration_temp_start', default=1.5, type=float,
                    help='Gumbel-top-K temperature at epoch 0; decayed to '
                         '--exploration_temp_end over --exploration_anneal_epochs.')
parser.add_argument('--exploration_temp_end', default=0.3, type=float)
parser.add_argument('--exploration_anneal_epochs', default=None, type=int,
                    help='Epochs to anneal eps/temp from start to end. '
                         'Default = 80% of --num_epochs.')

args = parser.parse_args()


# ============================ Setup ============================

num_epochs = args.num_epochs
device = torch.device(args.cuda_pick if torch.cuda.is_available() else "cpu")
current_date = datetime.now().strftime('%Y-%m-%d')
exp_name_full = f"{current_date}_{args.dataset}_{args.exp_name}"
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)


# ============================ Aggregation ============================

if args.aggregate:
    fold_results = []
    for fold_idx in range(3):
        with open(os.path.join(exp_dir, f"results_fold{fold_idx}.json")) as f:
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
            print(f"  {key}: {aggregated[f'{key}_mean']:.4f} "
                  f"+/- {aggregated[f'{key}_std']:.4f}")
    sys.exit(0)


# ============================ Data ============================

set_seed(args.seed_num)
print(f"Device: {device}")
print(f"Experiment: {exp_name_full}")
print(f"Loading frozen fusion from: {args.frozen_ckpt}")

root_dir = '/files1/haodong/data/processed_dalia_activity'   # UPDATE PATH
dataset_cfg = DaliaHAR()
modalities = dataset_cfg.modalities
num_m = len(modalities)

if args.fold is not None:
    fold_cfg = dataset_cfg.folds[args.fold]
    train_subjects, eval_subjects = fold_cfg['train_set'], fold_cfg['eval_set']
    print(f'Fold {args.fold}: train={train_subjects}, eval={eval_subjects}')
else:
    train_subjects, eval_subjects = dataset_cfg.train_set, dataset_cfg.eval_set
val_subjects = dataset_cfg.val_set

input_length = ((dataset_cfg.duration * dataset_cfg.base_sample_rate) // 2
                if args.transform == 'sax'
                else dataset_cfg.duration * dataset_cfg.base_sample_rate)

train_ds = HARDataset(root_dir, modalities, train_subjects, dataset_cfg, args.transform)
val_ds = HARDataset(root_dir, modalities, val_subjects, dataset_cfg, args.transform)
eval_ds = HARDataset(root_dir, modalities, eval_subjects, dataset_cfg, args.transform)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, drop_last=True)


# ============================ Model ============================

v6_cfg = _DaliaV6Cfg(dataset_cfg)
use_selector = args.selector_strategy != 'none'

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
    no_selector=not use_selector,                  # ← selector enabled now
    use_weighted_factor=True,                      # active when selector on
    use_triton=False,
    num_classes=dataset_cfg.num_classes,
    lambda_probe=args.lambda_probe if use_selector else 0.0,
    lambda_diversity=args.lambda_diversity if use_selector else 0.0,
    lambda_reinforce=args.lambda_reinforce if use_selector else 0.0,
    lambda_sparsity=args.lambda_sparsity if use_selector else 0.0,
    sparse_attn_variant=args.sparse_attn_variant,
    strat_block_size=args.strat_block_size,
    downsample_min_len=args.downsample_min_len,
    use_batched_fusion=args.use_batched_fusion,
    per_modal_distill=args.per_modal_distill,
    per_modal_downsample_min_len=args.per_modal_downsample_min_len,
    use_interaction_matrix=args.use_interaction_matrix,
    use_holo_bias=args.use_holo_bias,
    holo_scale=args.holo_scale,
)

model = V6SelectorTrainingWrapper(inner, modalities, dataset_cfg.variates)


# ============================ Load frozen fusion ============================

ckpt = torch.load(args.frozen_ckpt, map_location='cpu', weights_only=False)
missing, unexpected = model.load_state_dict(ckpt, strict=False)

# Sanity: missing keys should be selector-only; unexpected should be empty.
sel_missing = [k for k in missing if 'modality_selector' in k]
non_sel_missing = [k for k in missing if 'modality_selector' not in k]
print(f"Loaded fusion checkpoint:")
print(f"  fusion-only keys missing in ckpt (selector-related, expected): "
      f"{len(sel_missing)}")
if non_sel_missing:
    print(f"  ⚠ non-selector keys MISSING from ckpt: {non_sel_missing[:5]}"
          f"{' ...' if len(non_sel_missing) > 5 else ''}")
if unexpected:
    print(f"  ⚠ unexpected keys in ckpt: {unexpected[:5]}"
          f"{' ...' if len(unexpected) > 5 else ''}")

model = model.to(device).float()


# ============================ Freeze everything except selector ============================

n_total, n_trainable = 0, 0
for name, p in model.named_parameters():
    n_total += p.numel()
    if 'modality_selector' in name:
        p.requires_grad = True
        n_trainable += p.numel()
    else:
        p.requires_grad = False

print(f"\nFreezing fusion: {n_total - n_trainable:,} params frozen, "
      f"{n_trainable:,} selector params trainable "
      f"({100 * n_trainable / n_total:.1f}% of total).")

# Optimizer only sees trainable params (selector).
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=args.lr)
criterion = (FocalLoss(gamma=args.focal_gamma)
             if args.loss == 'focal' else nn.CrossEntropyLoss())

scheduler = None
if args.lr_schedule != 'constant':
    from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
    if args.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif args.lr_schedule == 'cosine_warmrestart':
        T_0 = max(int(num_epochs * 0.2), 1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=1e-6)

print(f'LR schedule: {args.lr_schedule} (base_lr={args.lr})')
print(f'Selector strategy: {args.selector_strategy}, target_k={args.target_k}/{num_m}')
print(f'Lambdas: probe={args.lambda_probe} diversity={args.lambda_diversity} '
      f'reinforce={args.lambda_reinforce} sparsity={args.lambda_sparsity}')

# ---- Exploration setup + linear-anneal schedule ------------------
_anneal_T = (args.exploration_anneal_epochs
             if args.exploration_anneal_epochs is not None
             else max(1, int(num_epochs * 0.8)))

def _explore_step(epoch):
    """Linear anneal eps / temp from *_start at epoch 0 to *_end at
    epoch `_anneal_T`, then hold."""
    frac = min(1.0, epoch / max(_anneal_T, 1))
    eps = args.exploration_eps_start + (args.exploration_eps_end -
                                        args.exploration_eps_start) * frac
    tmp = args.exploration_temp_start + (args.exploration_temp_end -
                                         args.exploration_temp_start) * frac
    return eps, tmp

if use_selector and inner.modality_selector is not None:
    inner.modality_selector.exploration_mode = args.exploration_mode
    print(f'Exploration: mode={args.exploration_mode}  '
          f'anneal over {_anneal_T} epochs  '
          f'(eps {args.exploration_eps_start}→{args.exploration_eps_end}, '
          f'temp {args.exploration_temp_start}→{args.exploration_temp_end})')


# ============================ Training ============================

best_val_acc, best_val_f1, best_epoch_val = 0.0, 0.0, -1
best_model_state = None

for epoch in range(num_epochs):
    # Top-K schedule (per-epoch).  Selector-only training has no warmup —
    # the fusion is already converged so we can dial K in immediately.
    if use_selector:
        current_topk, display_k = _k_schedule(
            epoch, args.selector_strategy, num_m, args.target_k,
            args.random_k_min, args.random_k_max)
        inner.top_k = current_topk

        # Anneal exploration knobs each epoch (mutates the selector
        # in-place — no rebuild needed).
        if inner.modality_selector is not None and args.exploration_mode != 'none':
            cur_eps, cur_tmp = _explore_step(epoch)
            inner.modality_selector.exploration_eps = cur_eps
            inner.modality_selector.exploration_temp = cur_tmp

    mod_drop_prob = _modality_drop_schedule(
        epoch, args.mod_drop_warmup, args.mod_drop_ramp_epochs,
        args.max_modality_drop)

    train_loss, train_acc = train_one_epoch(
        train_loader, model, criterion, optimizer, epoch, device,
        modality_drop_prob=mod_drop_prob, clip_grad=args.clip_grad)

    val_loss, val_acc, val_y, val_p = evaluate_one_epoch(
        val_loader, model, criterion, epoch, device)
    val_f1 = f1_score(val_y, val_p, average='macro')

    if scheduler is not None:
        scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_epoch_val = epoch
        best_model_state = model.state_dict()


# ============================ Save + test eval ============================

print(f"\nBest selector model: epoch {best_epoch_val}  "
      f"val_acc={best_val_acc:.4f}  val_f1={best_val_f1:.4f}")

ckpt_name = (f"best_model_fold{args.fold}.pth"
             if args.fold is not None else "best_val_model.pth")
torch.save(best_model_state, os.path.join(exp_dir, ckpt_name))

print("\n" + "=" * 80)
print("Test-set evaluation (clean)")
print("=" * 80)

model.load_state_dict(best_model_state)
model.eval()

_, test_acc, t_y, t_p = evaluate_one_epoch(eval_loader, model, criterion, 0, device)
test_f1 = f1_score(t_y, t_p, average='macro')
print(f"  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")

config = {
    'experiment_name': exp_name_full,
    'stage': 'selector_only',
    'frozen_ckpt': args.frozen_ckpt,
    'dataset': args.dataset, 'fold': args.fold,
    'selector_strategy': args.selector_strategy,
    'target_k': args.target_k, 'random_k_min': args.random_k_min,
    'random_k_max': args.random_k_max,
    'lambda_probe': args.lambda_probe,
    'lambda_diversity': args.lambda_diversity,
    'lambda_reinforce': args.lambda_reinforce,
    'lambda_sparsity': args.lambda_sparsity,
    'use_interaction_matrix': args.use_interaction_matrix,
    'use_holo_bias': args.use_holo_bias, 'holo_scale': args.holo_scale,
    'num_epochs': num_epochs, 'batch_size': args.batch_size,
    'lr': args.lr, 'lr_schedule': args.lr_schedule,
    'seed': args.seed_num, 'transform': args.transform,
    'modalities': modalities,
    'n_trainable_params': n_trainable,
    'n_frozen_params': n_total - n_trainable,
}
with open(os.path.join(exp_dir, "config.json"), 'w') as f:
    json.dump(config, f, indent=4)

model_stats = {
    'experiment_name': exp_name_full, 'fold': args.fold,
    'best_val_acc': best_val_acc, 'best_val_f1': best_val_f1,
    'best_epoch_val': best_epoch_val,
    'best_test_acc': float(test_acc),
    'best_test_f1': float(test_f1),
    'n_trainable_params': n_trainable,
}
results_filename = (f"results_fold{args.fold}.json"
                    if args.fold is not None else "results.json")
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(model_stats, f, indent=4)

print(f"\nDirectory: {exp_dir}")
