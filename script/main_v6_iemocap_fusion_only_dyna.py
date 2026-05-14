"""V6 fusion-only training on IEMOCAP with **dynamic / grad-norm-based**
modality-dropout curriculum.

Same backbone as `main_v6_iemocap_fusion_only.py` (no_selector=True,
opt + batched fusion).  The ONLY thing that differs is the modality-drop
schedule:

  - `main_v6_iemocap_fusion_only.py`      : uniform linear ramp; every
                                            modality gets the same drop
                                            probability `p(epoch)`.
  - **this script (`..._dyna.py`)**        : **staggered base ramp**
                                            (clean → linear → plateau)
                                            **scaled per-modality by the
                                            gradient norm** of that
                                            modality's `input_projectors`
                                            entry (hook-tracked).
                                            High-grad modalities get
                                            dropped more often, forcing
                                            weak modalities to learn.

Schedule (mirrors ``utils.train_utils.get_modality_curriculum``):
    epoch < 20% of num_epochs            → drop_prob = 0   (clean)
    20% <= epoch < 50%                   → linear 0 → max_drop
    epoch >= 50%                         → max_drop  (plateau)

Once drop_prob > 0, every ``--profile_update_freq`` epochs we sample
recent gradient norms (via ``ModalityGradientProfiler``) and rescale
``current_mod_probs[m] = clip( base_prob * M * (norm_m / Σnorms), 0, 0.8 )``
— modalities with higher per-step gradient flow get a heavier drop
budget.  Same modality dict is used across the whole batch (batched
fusion requires a fixed M per forward).

Run:
    python main_v6_iemocap_fusion_only_dyna.py --split_id=0 --cuda_pick=cuda:0
    python main_v6_iemocap_fusion_only_dyna.py --aggregate
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed, count_model_parameters
from utils.dataset_cfg import IEMOCAP
from data.IEMOCAP.get_data import get_dataloader, FEATURE_DIMS, NUM_CLASSES
from utils.train_utils import (
    AverageMeter, ProgressMeter, FocalLoss,
    get_modality_curriculum,
    ModalityGradientProfiler,
)

from multimodal_model.v6_downsample_opt_batched import DualVideoBottleneckModelV6Downsample


# ============================ V6 cfg adaptor ============================

class _IEMOCAPv6Cfg:
    """Drop-in cfg expected by DualVideoBottleneckModelV6Downsample."""
    def __init__(self, ds_cfg):
        self.modalities     = ds_cfg.modalities
        self.variates       = ds_cfg.variates
        self.video_high_dim = ds_cfg.video_high_dim
        self.video_low_dim  = ds_cfg.video_low_dim


# ============================ Wrapper ============================

class V6FusionOnlyDynaWrapper(nn.Module):
    """Same shim as `V6FusionOnlyWrapper` but the forward accepts a
    **per-modality** drop-probability dict instead of a single scalar.

    For each forward at training time:
      For modality m, draw Bernoulli(drop_probs[m]).
      If True → remove m from the V6 input dict entirely (fusion never
                sees it; matches the fusion-only "actually delete"
                semantics).
      Rejection-sample to guarantee at least one survivor.

    Eval time bypasses dropout — all M modalities flow through.
    """

    def __init__(self, v6_model, modalities):
        super().__init__()
        self.model      = v6_model
        self.modalities = list(modalities)

    def forward(self, high, low, modality_dropout_probs=None, training=False):
        if training and modality_dropout_probs is not None:
            # Per-modality independent Bernoulli.  Rejection-resample to
            # guarantee K >= 1 (probability of all-dropped = Π p_m, which
            # is < 1% for sane schedules but we still guard).
            for _ in range(8):
                kept = [m for m in self.modalities
                        if torch.rand(1).item() >= modality_dropout_probs.get(m, 0.0)]
                if kept:
                    break
            else:
                idx = int(torch.randint(0, len(self.modalities), (1,)).item())
                kept = [self.modalities[idx]]
            high = {m: high[m] for m in kept}
            low  = {m: low[m]  for m in kept}
        out = self.model(high, low, training=training,
                         return_selection_info=False)
        logits = out[0] if isinstance(out, tuple) else out
        return logits


# ============================ Batch helper ============================

def prepare_batch(batch, device):
    """Unpack a 13-element IEMOCAP batch into high/low dicts + labels."""
    (video, video_low, audio, audio_low, text, text_low,
     mocap_hand, mocap_hand_low, mocap_head, mocap_head_low,
     mocap_rotated, mocap_rotated_low, labels) = batch

    def _to(t): return t.float().to(device)

    high = {
        'video':         _to(video),
        'audio':         _to(audio),
        'text':          _to(text),
        'mocap_hand':    _to(mocap_hand),
        'mocap_head':    _to(mocap_head),
        'mocap_rotated': _to(mocap_rotated),
    }
    low = {
        'video':         _to(video_low),
        'audio':         _to(audio_low),
        'text':          _to(text_low),
        'mocap_hand':    _to(mocap_hand_low),
        'mocap_head':    _to(mocap_head_low),
        'mocap_rotated': _to(mocap_rotated_low),
    }
    labels = labels.view(-1).long().to(device)
    return high, low, labels


# ============================ Train / eval ============================

def train_one_epoch(loader, model, criterion, optimizer, epoch, device,
                    modality_dropout_probs, clip_grad=0.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter  = AverageMeter('Acc',  ':.4f')
    model.train()
    for i, batch in enumerate(loader):
        high, low, labels = prepare_batch(batch, device)
        logits = model(high, low, modality_dropout_probs=modality_dropout_probs,
                       training=True)
        loss = criterion(logits, labels)
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(labels).sum().item() / labels.size(0)
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(acc, labels.size(0))
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
    acc_meter  = AverageMeter('Acc',  ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    for i, batch in enumerate(loader):
        high, low, labels = prepare_batch(batch, device)
        logits = model(high, low, modality_dropout_probs=None, training=False)
        loss = criterion(logits, labels)
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(labels).sum().item() / labels.size(0)
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(acc, labels.size(0))
        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())
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
    description='V6 fusion-only training on IEMOCAP with grad-norm-based '
                'asymmetric modality-dropout curriculum.')

parser.add_argument('--exp_name',    default='v6_fusion_only_dyna', type=str)
parser.add_argument('--results_dir',
                    default='../model_chkpt/IEMOCAP/multimodal_model/', type=str)
parser.add_argument('--dataset',     default='IEMOCAP', type=str)
parser.add_argument('--data_root',
                    default='/home/group/maestro_visual/data/IEMOCAP_full_release',
                    type=str)
parser.add_argument('--num_epochs',   default=200, type=int)
parser.add_argument('--batch_size',   default=32,  type=int)
parser.add_argument('--num_workers',  default=4,   type=int)
parser.add_argument('--cuda_pick',    default='cuda:0', type=str)
parser.add_argument('--seed_num',     default=239,  type=int)
parser.add_argument('--max_seq_len',  default=100,  type=int,
                    help='Resample all modalities to this T (required for batched fusion).')
parser.add_argument('--input_length', default=100,  type=int,
                    help='input_length passed to V6 (F.interpolate target).')
parser.add_argument('--num_experts',  default=4, type=int)
parser.add_argument('--base_factor',  default=3, type=int)
parser.add_argument('--d_model',      default=64, type=int)
parser.add_argument('--nhead',        default=8,  type=int)
parser.add_argument('--num_layers',   default=4,  type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout',      default=0.1, type=float)

# Modality-drop curriculum: staggered base ramp + grad-norm scaling
parser.add_argument('--max_modality_drop', default=0.4, type=float,
                    help='Final per-modality base drop prob after the '
                         '20%%→50%% linear ramp.  Per-modality probs are '
                         '`base × M × (grad_norm_m / Σ grad_norms)`, '
                         'clipped to [0, 0.8].')
parser.add_argument('--profile_update_freq', default=5, type=int,
                    help='Refresh the per-modality dropout dict every N '
                         'epochs (uses recent grad-norm snapshot).')

# Optimizer / schedule
parser.add_argument('--lr',           default=1e-4,     type=float)
parser.add_argument('--lr_schedule',  default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad',    default=1.0, type=float)

# 3-split CV
parser.add_argument('--split_id',  default=None, type=int, choices=[0, 1, 2])
parser.add_argument('--aggregate', action='store_true', default=False)

# Loss
parser.add_argument('--loss',        default='ce',  type=str, choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0,   type=float)

# V6 backbone knobs
parser.add_argument('--internal_dim',   default=64, type=int)
parser.add_argument('--n_bottlenecks',  default=8,  type=int)
parser.add_argument('--fusion_layer',   default=0,  type=int)
parser.add_argument('--use_sparse_attn', type=_str2bool, default=True)
parser.add_argument('--sparse_attn_variant', default='opt', type=str,
                    choices=['opt', 'opt_strat', 'opt_strat_blk'])
parser.add_argument('--strat_block_size',            default=8, type=int)
parser.add_argument('--downsample_min_len',           default=4, type=int)
parser.add_argument('--per_modal_distill',            action='store_true', default=False)
parser.add_argument('--per_modal_downsample_min_len', default=4, type=int)
parser.add_argument('--use_batched_fusion',           action='store_true', default=True)

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
    split_results = []
    for split_idx in range(3):
        with open(os.path.join(exp_dir, f"results_split{split_idx}.json")) as f:
            split_results.append(json.load(f))
    aggregated = {'experiment_name': exp_name_full, 'splits': split_results}
    for key in ('best_val_acc', 'best_val_f1', 'best_test_acc', 'best_test_f1'):
        vals = [sr[key] for sr in split_results if key in sr]
        if vals:
            aggregated[f'{key}_mean'] = float(np.mean(vals))
            aggregated[f'{key}_std']  = float(np.std(vals))
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

dataset_cfg = IEMOCAP()
modalities  = dataset_cfg.modalities
num_m       = len(modalities)

split_id = args.split_id if args.split_id is not None else 0

train_loader, val_loader, test_loader = get_dataloader(
    data_root              = args.data_root,
    batch_size             = args.batch_size,
    num_workers            = args.num_workers,
    split_id               = split_id,
    train_shuffle          = True,
    max_seq_len            = args.max_seq_len,
    time_compression_ratio = 1,
    use_batched_fusion     = args.use_batched_fusion,
)
print(f'Split {split_id}: '
      f'train={len(train_loader.dataset)}  '
      f'val={len(val_loader.dataset)}  '
      f'test={len(test_loader.dataset)}')


# ============================ Model ============================

v6_cfg = _IEMOCAPv6Cfg(dataset_cfg)

inner = DualVideoBottleneckModelV6Downsample(
    cfg                          = v6_cfg,
    output_dim                   = dataset_cfg.num_classes,
    input_length                 = args.input_length,
    d_model                      = args.d_model,
    nhead                        = args.nhead,
    num_layers_per_modal         = args.num_layers_per_modal,
    num_layers                   = args.num_layers,
    dropout                      = args.dropout,
    verbose                      = True,
    video_low_dim                = v6_cfg.video_low_dim,
    video_high_dim               = v6_cfg.video_high_dim,
    use_bottleneck               = True,
    n_bottlenecks                = args.n_bottlenecks,
    fusion_layer                 = args.fusion_layer,
    use_sparse_moe               = False,
    num_experts                  = args.num_experts,
    expert_k                     = 1,
    internal_dim                 = args.internal_dim,
    bottleneck_head_pos          = True,
    use_sparse_attn              = args.use_sparse_attn,
    factor                       = args.base_factor,
    selector_video_source        = 'high',
    encoder_video_source         = 'high',
    no_selector                  = True,           # ← fusion-only
    use_weighted_factor          = False,           # inert under no_selector
    use_triton                   = False,
    num_classes                  = dataset_cfg.num_classes,
    sparse_attn_variant          = args.sparse_attn_variant,
    strat_block_size             = args.strat_block_size,
    downsample_min_len           = args.downsample_min_len,
    use_batched_fusion           = args.use_batched_fusion,
    per_modal_distill            = args.per_modal_distill,
    per_modal_downsample_min_len = args.per_modal_downsample_min_len,
)

model = V6FusionOnlyDynaWrapper(inner, modalities)
model = model.to(device).float()

# Gradient-norm hook profiler (one hook per input_projectors[m].weight).
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
print(f'   epoch [   0,  {int(num_epochs * 0.2):>3d})  → 0')
print(f'   epoch [{int(num_epochs * 0.2):>3d}, {int(num_epochs * 0.5):>3d})  '
      f'→ linear 0 → {args.max_modality_drop}')
print(f'   epoch [{int(num_epochs * 0.5):>3d}, {num_epochs:>3d}]  '
      f'→ {args.max_modality_drop} (plateau)')
print(f'   per-modality scaling: base × M × norm_m / Σ norms (clipped to 0.8)')
print(f'Selector:      NONE (fusion-only)')


# ============================ Training loop ============================

best_val_acc, best_val_f1, best_epoch_val = 0.0, 0.0, -1
best_model_state = None
cold_best_val_acc, cold_best_val_f1, cold_best_epoch = 0.0, 0.0, -1
cold_best_model_state = None

current_mod_probs = {m: 0.0 for m in modalities}    # initial: clean
profile_log = []    # per-epoch snapshot for debugging

mod_drop_complete_epoch = int(num_epochs * 0.5)

writer = SummaryWriter(
    comment=f"_{exp_name_full}_split{split_id}")

for epoch in range(num_epochs):
    # ---- 1. Get the base drop probability for this epoch ----
    mod_base = get_modality_curriculum(epoch, num_epochs, args.max_modality_drop)

    # ---- 2. Apply grad-norm-based per-modality scaling ----
    if mod_base > 0 and epoch % args.profile_update_freq == 0:
        if any(len(profiler.grad_norms[m]) > 0 for m in modalities):
            current_mod_probs = profiler.get_asymmetric_dropout_probs(mod_base)
            snap = profiler.get_norm_snapshot()
            profile_log.append({
                'epoch': epoch,
                'mod_base': mod_base,
                'grad_norms': snap,
                'mod_dropout_probs': {m: round(p, 4)
                                      for m, p in current_mod_probs.items()},
            })
        else:
            # Hook not yet populated — fall back to uniform.
            current_mod_probs = {m: mod_base for m in modalities}
    elif mod_base == 0:
        current_mod_probs = {m: 0.0 for m in modalities}
    # else: keep current_mod_probs from previous profile-update epoch

    # ---- 3. Train ----
    train_loss, train_acc = train_one_epoch(
        train_loader, model, criterion, optimizer, epoch, device,
        modality_dropout_probs=current_mod_probs, clip_grad=args.clip_grad)

    val_loss, val_acc, val_y, val_p = evaluate_one_epoch(
        val_loader, model, criterion, epoch, device)
    val_f1 = f1_score(val_y, val_p, average='macro')

    current_lr = optimizer.param_groups[0]['lr']
    if scheduler is not None:
        scheduler.step()

    # ---- tensorboard ----
    writer.add_scalar('train/loss',        train_loss, epoch)
    writer.add_scalar('train/acc',         train_acc,  epoch)
    writer.add_scalar('val/loss',          val_loss,   epoch)
    writer.add_scalar('val/acc',           val_acc,    epoch)
    writer.add_scalar('val/f1',            val_f1,     epoch)
    writer.add_scalar('schedule/mod_base', mod_base,   epoch)
    writer.add_scalar('schedule/lr',       current_lr, epoch)
    # Per-modality drop probability (current dict)
    for m, p in current_mod_probs.items():
        writer.add_scalar(f'mod_drop/{m}', p, epoch)
    # Recent grad-norm snapshot per modality
    if any(len(profiler.grad_norms[m]) > 0 for m in modalities):
        for m, gn in profiler.get_norm_snapshot().items():
            writer.add_scalar(f'grad_norm/{m}', gn, epoch)

    # Cold best — anytime
    if val_acc > cold_best_val_acc:
        cold_best_val_acc     = val_acc
        cold_best_val_f1      = val_f1
        cold_best_epoch       = epoch
        cold_best_model_state = model.state_dict()

    # Warm best — only after the staggered ramp completes
    if epoch >= mod_drop_complete_epoch and val_acc > best_val_acc:
        best_val_acc     = val_acc
        best_val_f1      = val_f1
        best_epoch_val   = epoch
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

ckpt_name = f"best_model_split{split_id}.pth"
torch.save(best_model_state, os.path.join(exp_dir, ckpt_name))


# ============================ Clean test eval ============================

print("\n" + "=" * 80)
print("Test-set evaluation (clean)")
print("=" * 80)
model.load_state_dict(best_model_state)
model.eval()

_, test_acc, t_y, t_p = evaluate_one_epoch(test_loader, model, criterion, 0, device)
test_f1 = f1_score(t_y, t_p, average='macro')
print(f"  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")

writer.add_scalar('test/acc',      test_acc,       0)
writer.add_scalar('test/f1',       test_f1,        0)
writer.add_scalar('best/val_acc',  best_val_acc,   0)
writer.add_scalar('best/val_f1',   best_val_f1,    0)
writer.add_scalar('best/epoch',    best_epoch_val, 0)
writer.close()


# ============================ Save ============================

final_grad_norms = (profiler.get_norm_snapshot()
                    if any(len(profiler.grad_norms[m]) > 0 for m in modalities)
                    else {})

config = {
    'experiment_name':              exp_name_full,
    'dataset':                      args.dataset,
    'model_variant':                'v6_fusion_only_dyna',
    'model_class':                  'DualVideoBottleneckModelV6Downsample',
    'selector':                     'NONE',
    'modality_drop_strategy':       'staggered_linear_base + gradient_asymmetric_scaling',
    'split_id':                     split_id,
    'num_epochs':                   num_epochs,
    'batch_size':                   args.batch_size,
    'seed':                         args.seed_num,
    'd_model':                      args.d_model,
    'nhead':                        args.nhead,
    'num_layers':                   args.num_layers,
    'num_layers_per_modal':         args.num_layers_per_modal,
    'dropout':                      args.dropout,
    'num_experts':                  args.num_experts,
    'base_factor':                  args.base_factor,
    'internal_dim':                 args.internal_dim,
    'n_bottlenecks':                args.n_bottlenecks,
    'fusion_layer':                 args.fusion_layer,
    'use_sparse_attn':              args.use_sparse_attn,
    'sparse_attn_variant':          args.sparse_attn_variant,
    'strat_block_size':             args.strat_block_size,
    'downsample_min_len':           args.downsample_min_len,
    'use_batched_fusion':           args.use_batched_fusion,
    'per_modal_distill':            args.per_modal_distill,
    'per_modal_downsample_min_len': args.per_modal_downsample_min_len,
    'max_modality_drop':            args.max_modality_drop,
    'profile_update_freq':          args.profile_update_freq,
    'lr':                           args.lr,
    'lr_schedule':                  args.lr_schedule,
    'clip_grad':                    args.clip_grad,
    'data_root':                    args.data_root,
    'max_seq_len':                  args.max_seq_len,
    'input_length':                 args.input_length,
    'modalities':                   modalities,
}
with open(os.path.join(exp_dir, "config.json"), 'w') as f:
    json.dump(config, f, indent=4)

model_stats = {
    'experiment_name':  exp_name_full,
    'split_id':         split_id,
    'best_val_acc':     best_val_acc,
    'best_val_f1':      best_val_f1,
    'best_epoch_val':   best_epoch_val,
    'selected_source':  selected_source,
    'cold_best': {
        'val_acc': cold_best_val_acc,
        'val_f1':  cold_best_val_f1,
        'epoch':   cold_best_epoch,
    },
    'best_test_acc':    float(test_acc),
    'best_test_f1':     float(test_f1),
    'final_grad_norms': final_grad_norms,
    'profile_log':      profile_log,
}
with open(os.path.join(exp_dir, f"results_split{split_id}.json"), 'w') as f:
    json.dump(model_stats, f, indent=4)

# Cleanup gradient hooks.
profiler.remove_hooks()

print(f"\nBest Model (Epoch {best_epoch_val}): "
      f"val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
print(f"Selected: {selected_source}")
print(f"Directory: {exp_dir}")
