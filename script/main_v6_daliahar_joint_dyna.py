"""V6 joint fusion + selector training on DaliaHAR with selector-driven
**dyna**-style modality dropout.

Differences vs `main_v6_daliahar_fusion_only_dyna.py`:

  - **Selector is enabled** (`no_selector=False`).  V6's modality_selector
    outputs W (B, M) sigmoid scores throughout training.
  - **Two-phase schedule**:
      * Phase 1 (warmup, epoch < warmup_epochs): all M modalities flow
        into fusion (no drop).  Selector parameters are FROZEN — only
        fusion + classifier + projectors train.  This gets the fusion
        backbone to a stable point before the selector starts influencing
        anything.
      * Phase 2 (epoch >= warmup_epochs): selector parameters UNFROZEN.
        The dyna gradient-norm signal is replaced with the selector's W:
        modalities with **higher W get dropped MORE often** (the same
        anti-overfit principle as gradnorm dyna — penalise the modality
        the model already relies on).  After the Bernoulli drop, if more
        than ``target_k`` modalities survive, a random K-subset is kept
        for the fusion forward.
  - Selector gradient comes from V6's `probe_loss` aux (per-modality
    probe heads) + the fact that the projector features change as
    fusion learns.

Run:
    python main_v6_daliahar_joint_dyna.py --fold=0 --cuda_pick=cuda:0
    python main_v6_daliahar_joint_dyna.py --aggregate
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
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed, count_model_parameters
from utils.dataset_cfg import DaliaHAR
from data.dataset_builder import HARDataset
from utils.train_utils import (
    AverageMeter, ProgressMeter, FocalLoss,
    get_modality_curriculum,
)

from multimodal_model.v6_downsample_opt_batched import DualVideoBottleneckModelV6Downsample


# ============================ V6 cfg adaptor ============================

class _DaliaV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


# ============================ Wrapper ============================

class V6JointDynaWrapper(nn.Module):
    """Joint wrapper: fusion (always trainable) + selector (frozen→unfrozen
    at warmup_epochs).

    Phase 1 (warmup): all M modalities go to fusion; selector forward runs
      to populate its features (for probe_loss) but its output is unused
      for modality selection.

    Phase 2: selector W (B, M) determines per-modality drop probability.
      Higher W → modality is dropped more often.  Surviving modalities
      then go through a random K-subset filter before reaching fusion.

    Eval: selector picks natural top-K (no random / no drop), standard
      V6 forward.
    """

    def __init__(self, v6_model, modalities, variates, target_k):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self.M = len(self.modalities)
        self.target_k = int(target_k)
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

    def _compute_drop_probs_from_W(self, W: torch.Tensor,
                                   mod_base: float) -> dict:
        """Selector-driven dropout probability per modality.

        W: (B, M) selector sigmoid scores.  Higher W → higher drop prob.
        mod_base: current curriculum drop magnitude in [0, max_drop].

        Returns:
          dict {modality_name: drop_prob in [0, 0.8]}
        """
        if mod_base <= 0:
            return {m: 0.0 for m in self.modalities}
        # Batch-mean W (no grad needed for sampling)
        with torch.no_grad():
            w_mean = W.detach().mean(dim=0)               # (M,)
            total = w_mean.sum().clamp(min=1e-8)
            ratios = w_mean / total                       # (M,) sums to 1
            # base × M × ratio — mirrors dyna's gradnorm formula
            probs = (mod_base * self.M * ratios).clamp(0.0, 0.8)
        return {self.modalities[i]: probs[i].item() for i in range(self.M)}

    def _bernoulli_drop_then_random_topk(self, drop_probs: dict) -> list:
        """Bernoulli-drop each modality, then keep top-K of surviving via
        uniform random.  Returns list of kept modality names (length <= K
        but at least 1)."""
        # Bernoulli drop
        survived = [m for m in self.modalities
                    if torch.rand(1).item() >= drop_probs.get(m, 0.0)]
        if not survived:
            # Numerical edge: force keep one
            idx = int(torch.randint(0, self.M, (1,)).item())
            survived = [self.modalities[idx]]
        # Random top-K of survivors
        if len(survived) > self.target_k:
            _random.shuffle(survived)
            return survived[:self.target_k]
        return survived

    def forward(self, x, mod_base: float = 0.0, training: bool = False,
                labels=None, phase: str = 'warmup'):
        """
        phase: 'warmup' = all modalities; 'dyna' = selector-driven drop
               + random top-K.  At eval (training=False), selector
               picks natural top-K regardless of phase.
        """
        high, low = self._split(x)

        # ---- Eval ----
        if not training:
            # Selector picks natural top-K via V6's internal logic.
            result = self.model(high, low, training=False,
                                return_selection_info=True, labels=labels)
            output = result[0]
            return output, {}

        # ---- Train: phase routing ----
        if phase == 'warmup':
            # All modalities flow into fusion; selector params frozen.
            # We still call V6 with override of selected = all_modalities
            # so the fusion sees all 5; selector aux losses still computed.
            result = self.model(high, low, training=True,
                                return_selection_info=True, labels=labels,
                                override_selected_modalities=self.modalities)
            if len(result) == 5:
                output, _, _, _, aux = result
                return output, aux
            return result[0], {}

        # ---- Train phase 2 (dyna) ----
        # 1. Run V6 once to extract W (selector forward + projector + encoder
        # for the override=all path).  We need W to compute drop_probs.
        # Trick: do a first forward with override=all to get W (this is
        # the "selector-aware" forward).
        # Note: this approach runs V6 twice per batch — once for W, once
        # with selected K.  Compute cost ≈ 2× single forward.
        with torch.no_grad():
            sel_inputs = {}
            for m in self.modalities:
                if m in low:
                    sel_inputs[m] = low[m]
                elif m in high:
                    sel_inputs[m] = high[m]
            # Same temporal downsample V6 applies to its own selector path,
            # so W here is computed on the same input the inner model uses.
            sel_inputs = self.model.downsample_selector_inputs(sel_inputs)
            _, W_detached, _ = self.model.modality_selector(
                sel_inputs, top_k=self.target_k, training=True)

        # 2. Compute drop probs from W
        drop_probs = self._compute_drop_probs_from_W(W_detached, mod_base)

        # 3. Bernoulli drop + random top-K on survivors
        kept = self._bernoulli_drop_then_random_topk(drop_probs)

        # 4. Forward V6 with kept modalities only
        result = self.model(high, low, training=True,
                            return_selection_info=True, labels=labels,
                            override_selected_modalities=kept)
        if len(result) == 5:
            output, _, _, _, aux = result
            aux['_n_kept'] = torch.tensor(float(len(kept)), device=output.device)
        else:
            output = result[0]
            aux = {}
        return output, aux


# ============================ Train / eval ============================

def train_one_epoch(loader, model, criterion, optimizer, epoch, device,
                    mod_base, phase, clip_grad=0.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    aux_meter = AverageMeter('Aux', ':.4f')
    nk_meter = AverageMeter('NK', ':.2f')
    model.train()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device).float(), y.to(device)
        logits, aux_losses = model(x, mod_base=mod_base, training=True,
                                   labels=y, phase=phase)
        loss = criterion(logits, y)
        aux_sum = 0.0
        n_kept = None
        for k, v in aux_losses.items():
            if k.startswith('_'):
                if k == '_n_kept':
                    n_kept = v.item()
                continue
            loss = loss + v
            aux_sum += v.item()
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / x.size(0)
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))
        if aux_sum > 0:
            aux_meter.update(aux_sum, x.size(0))
        if n_kept is not None:
            nk_meter.update(n_kept, 1)
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
                nk_str = f'  Nkept: {nk_meter.avg:.2f}' if nk_meter.count > 0 else ''
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
                      f'Acc: {acc_meter.avg:.4f}  '
                      f'phase={phase}  mod_base: {mod_base:.3f}{aux_str}{nk_str}')
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_one_epoch(loader, model, criterion, epoch, device, label: str = '',
                       verbose: bool = True):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device).float(), y.to(device)
        logits, _aux = model(x, mod_base=0.0, training=False)
        loss = criterion(logits, y)
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / x.size(0)
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))
        all_preds.append(predicted.cpu())
        all_labels.append(y.cpu())
    if verbose:
        suffix = f' [{label}]' if label else ''
        print(f'End of Epoch {epoch}{suffix} | Val Loss: {loss_meter.avg:.4f} '
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
    description='V6 joint fusion+selector training with selector-driven '
                'dyna dropout (DaliaHAR).')

parser.add_argument('--exp_name', default='v6_joint_dyna', type=str)
parser.add_argument('--results_dir', default='./results_v6_joint/', type=str)
parser.add_argument('--dataset', default='DaliaHAR', type=str)
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

# Curriculum
parser.add_argument('--max_modality_drop', default=0.4, type=float,
                    help='Final per-modality base drop prob.')
parser.add_argument('--warmup_epochs', default=None, type=int,
                    help='Phase 1 length (selector frozen, all modalities). '
                         'Default = int(num_epochs * 0.3).')

# Optimizer
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad', default=1.0, type=float)

# 3-fold CV
parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2])
parser.add_argument('--aggregate', action='store_true', default=False)
parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0, type=float)
parser.add_argument('--eval_seed', default=42, type=int)

# V6 knobs
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

# Selector
parser.add_argument('--target_k', default=4, type=int,
                    help='Target K for inference + random-K cap during '
                         'phase 2 training.')
parser.add_argument('--lambda_probe', default=0.05, type=float)
parser.add_argument('--lambda_diversity', default=0.0, type=float)
parser.add_argument('--lambda_sparsity', default=0.0, type=float)
parser.add_argument('--use_interaction_matrix', default=False, type=_str2bool)
parser.add_argument('--use_holo_bias', default=False, type=_str2bool)
parser.add_argument('--holo_scale', default=1.0, type=float)
parser.add_argument('--selector_downsample_factor', default=4, type=int,
                    help='Temporal downsample factor applied to selector inputs '
                         '(avg_pool1d). Default 4: selector sees T/4 of fusion T.')

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
    # Aggregate every numeric field that all folds share (catches per-K
    # metrics like best_val_acc_k1, best_test_acc_k3, etc.).
    common_keys = set(fold_results[0].keys())
    for fr in fold_results[1:]:
        common_keys &= set(fr.keys())
    for key in sorted(common_keys):
        vals = [fr[key] for fr in fold_results]
        if not all(isinstance(v, (int, float)) for v in vals):
            continue
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

root_dir = '/files1/haodong/data/processed_dalia_activity'
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
    no_selector=False, use_weighted_factor=True, use_triton=False,
    num_classes=dataset_cfg.num_classes,
    lambda_probe=args.lambda_probe,
    lambda_diversity=args.lambda_diversity,
    lambda_reinforce=0.0,
    lambda_sparsity=args.lambda_sparsity,
    sparse_attn_variant=args.sparse_attn_variant,
    strat_block_size=args.strat_block_size,
    downsample_min_len=args.downsample_min_len,
    use_batched_fusion=args.use_batched_fusion,
    per_modal_distill=args.per_modal_distill,
    per_modal_downsample_min_len=args.per_modal_downsample_min_len,
    use_interaction_matrix=args.use_interaction_matrix,
    use_holo_bias=args.use_holo_bias,
    holo_scale=args.holo_scale,
    selector_downsample_factor=args.selector_downsample_factor,
)
inner.top_k = args.target_k

model = V6JointDynaWrapper(inner, modalities, dataset_cfg.variates,
                            target_k=args.target_k).to(device).float()


# ============================ Selector freeze in warmup ============================

def _set_selector_trainable(model, trainable: bool):
    """Toggle requires_grad on modality_selector params."""
    for name, p in model.named_parameters():
        if 'modality_selector' in name:
            p.requires_grad = trainable


_set_selector_trainable(model, False)   # phase 1: selector frozen
warmup_epochs = (args.warmup_epochs if args.warmup_epochs is not None
                 else int(num_epochs * 0.3))

print(f'\nWarmup phase (selector frozen, all modalities): '
      f'epochs [0, {warmup_epochs})')
print(f'Dyna phase (selector unfrozen, drop ∝ selector W): '
      f'epochs [{warmup_epochs}, {num_epochs}]')

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
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=2, eta_min=1e-6)

print(f'Parameters: {count_model_parameters(model)}')
print(f'LR schedule: {args.lr_schedule} (base_lr={args.lr})')
print(f'target_k = {args.target_k}/{num_m}')


# ============================ Training loop ============================

# Per-K best tracker (K = 1..num_m).  Each epoch we run val at every K,
# stash the state dict (on CPU) when val_acc improves at that K.
best_per_k = {k: {'val_acc': -1.0, 'val_f1': 0.0, 'val_loss': float('inf'),
                  'epoch': -1, 'state': None}
              for k in range(1, num_m + 1)}

inner_ref = model.model  # underlying V6, where top_k lives
saved_target_k = inner_ref.top_k


def _snapshot_state(m):
    return {kk: vv.detach().cpu().clone() for kk, vv in m.state_dict().items()}


writer = SummaryWriter(
    comment=f"_{exp_name_full}_fold{args.fold}"
            if args.fold is not None else f"_{exp_name_full}")

for epoch in range(num_epochs):
    # Phase routing
    if epoch < warmup_epochs:
        phase = 'warmup'
        mod_base = 0.0
    else:
        phase = 'dyna'
        # Unfreeze selector at the boundary (once is enough)
        if epoch == warmup_epochs:
            _set_selector_trainable(model, True)
            # Rebuild optimiser to include newly-trainable params.
            optimizer = optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr)
            # Re-attach scheduler to new optimiser.
            if scheduler is not None:
                remaining = num_epochs - epoch
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=remaining, eta_min=1e-6)
            print(f'  >> Epoch {epoch}: SELECTOR UNFROZEN — '
                  f'switching to dyna phase')
        # Within dyna phase: curriculum ramp on mod_base
        local_epoch = epoch - warmup_epochs
        local_total = num_epochs - warmup_epochs
        mod_base = get_modality_curriculum(
            local_epoch, local_total, args.max_modality_drop)

    train_loss, train_acc = train_one_epoch(
        train_loader, model, criterion, optimizer, epoch, device,
        mod_base=mod_base, phase=phase, clip_grad=args.clip_grad)

    # Per-K validation (selector top_k temporarily overridden).
    epoch_k_results = {}
    for k_ in range(1, num_m + 1):
        inner_ref.top_k = k_
        v_loss, v_acc, v_y, v_p = evaluate_one_epoch(
            val_loader, model, criterion, epoch, device,
            label=f'k{k_}', verbose=False)
        v_f1 = f1_score(v_y, v_p, average='macro')
        epoch_k_results[k_] = (v_loss, v_acc, v_f1)
        if v_acc > best_per_k[k_]['val_acc']:
            best_per_k[k_] = {
                'val_acc': float(v_acc), 'val_f1': float(v_f1),
                'val_loss': float(v_loss), 'epoch': int(epoch),
                'state': _snapshot_state(model),
            }
        writer.add_scalar(f'val_k{k_}/loss', v_loss, epoch)
        writer.add_scalar(f'val_k{k_}/acc', v_acc, epoch)
        writer.add_scalar(f'val_k{k_}/f1', v_f1, epoch)
    inner_ref.top_k = saved_target_k

    # Compact one-line summary across all K.
    k_summary = '  '.join(
        f'k{k_}={epoch_k_results[k_][1]:.4f}'
        for k_ in range(1, num_m + 1))
    print(f'End of Epoch {epoch} | Val Acc per K: {k_summary}')

    if scheduler is not None:
        scheduler.step()

    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/acc', train_acc, epoch)
    writer.add_scalar('schedule/mod_base', mod_base, epoch)


# ============================ Test eval per K ============================

print('\n' + '=' * 80)
print('Test-set evaluation: best ckpt per K (selector top-K = K, no drop)')
print('=' * 80)

test_per_k = {}
for k_ in range(1, num_m + 1):
    entry = best_per_k[k_]
    if entry['state'] is None:
        print(f'  K={k_}: no best ckpt recorded')
        continue
    model.load_state_dict({kk: vv.to(device) for kk, vv in entry['state'].items()})
    inner_ref.top_k = k_
    _, t_acc, t_y, t_p = evaluate_one_epoch(
        eval_loader, model, criterion, 0, device,
        label=f'test_k{k_}', verbose=False)
    t_f1 = f1_score(t_y, t_p, average='macro')
    test_per_k[k_] = {'test_acc': float(t_acc), 'test_f1': float(t_f1)}
    print(f'  K={k_}: best_epoch={entry["epoch"]:>3d}  '
          f'val_acc={entry["val_acc"]:.4f}  val_f1={entry["val_f1"]:.4f}  '
          f'test_acc={t_acc:.4f}  test_f1={t_f1:.4f}')

    # Save per-K checkpoint to disk.
    ck = (f"best_model_fold{args.fold}_k{k_}.pth"
          if args.fold is not None else f"best_val_model_k{k_}.pth")
    torch.save(entry['state'], os.path.join(exp_dir, ck))
    writer.add_scalar(f'test_k{k_}/acc', t_acc, 0)
    writer.add_scalar(f'test_k{k_}/f1', t_f1, 0)
    writer.add_scalar(f'best_k{k_}/val_acc', entry['val_acc'], 0)
    writer.add_scalar(f'best_k{k_}/epoch', entry['epoch'], 0)

inner_ref.top_k = saved_target_k
writer.close()

# Surface the target_k row as "the" best (for back-compat printing only).
_tk = args.target_k if args.target_k in test_per_k else max(test_per_k.keys())
best_val_acc = best_per_k[_tk]['val_acc']
best_val_f1 = best_per_k[_tk]['val_f1']
best_epoch_val = best_per_k[_tk]['epoch']
test_acc = test_per_k[_tk]['test_acc']
test_f1 = test_per_k[_tk]['test_f1']
selected_source = f'per-K (reporting K={_tk})'


# ============================ Save ============================

config = {
    'experiment_name': exp_name_full, 'dataset': args.dataset,
    'fold': args.fold, 'target_k': args.target_k,
    'num_epochs': num_epochs, 'warmup_epochs': warmup_epochs,
    'batch_size': args.batch_size,
    'lr': args.lr, 'lr_schedule': args.lr_schedule,
    'max_modality_drop': args.max_modality_drop,
    'lambda_probe': args.lambda_probe,
    'lambda_diversity': args.lambda_diversity,
    'lambda_sparsity': args.lambda_sparsity,
    'selector_downsample_factor': args.selector_downsample_factor,
    'seed': args.seed_num, 'transform': args.transform,
    'modalities': modalities,
    'train_subjects': train_subjects, 'val_subjects': val_subjects,
    'eval_subjects': eval_subjects,
}
with open(os.path.join(exp_dir, "config.json"), 'w') as f:
    json.dump(config, f, indent=4)

model_stats = {
    'experiment_name': exp_name_full, 'fold': args.fold,
    'best_val_acc': best_val_acc, 'best_val_f1': best_val_f1,
    'best_epoch_val': best_epoch_val, 'selected_source': selected_source,
    'best_test_acc': float(test_acc),
    'best_test_f1': float(test_f1),
}
for k_ in range(1, num_m + 1):
    if k_ in test_per_k:
        entry = best_per_k[k_]
        model_stats[f'best_val_acc_k{k_}'] = float(entry['val_acc'])
        model_stats[f'best_val_f1_k{k_}'] = float(entry['val_f1'])
        model_stats[f'best_epoch_val_k{k_}'] = int(entry['epoch'])
        model_stats[f'best_test_acc_k{k_}'] = float(test_per_k[k_]['test_acc'])
        model_stats[f'best_test_f1_k{k_}'] = float(test_per_k[k_]['test_f1'])
results_filename = (f"results_fold{args.fold}.json"
                    if args.fold is not None else "results.json")
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(model_stats, f, indent=4)

print(f"\nBest Model (Epoch {best_epoch_val}): "
      f"val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
print(f"Selected: {selected_source}")
print(f"Directory: {exp_dir}")
