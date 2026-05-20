"""V6 fusion-only training on IEMOCAP with **variable-length per-modality
inputs** + **linear (uniform) modality dropout**.

Pipeline differences vs ``main_v6_iemocap_fusion_only_dyna.py``:

  1. Backbone = orig V6 fusion (``v6_downsample_orig``) with
     per-modality ``ModuleDict`` encoders.
  2. ``IEMOCAPDataset(use_batched_fusion=False)`` — short modalities
     keep their native T (no zero-pad to max_seq_len); only T>max_seq_len
     gets interpolated down.  ``pad_sequence`` pads each modality to its
     own batch-max T at collate time.
  3. ``sparse_attn_variant='opt'`` (now dispatches to ``ProbSparseAttentionOpt``
     even in orig, via the patch in v6_downsample_orig.py).

This variant uses **uniform Bernoulli modality dropout**:
``drop_prob[m] = mod_base`` for ALL modalities (no gradnorm-based
asymmetry).  Schedule:
    epoch < 20% of num_epochs            -> drop_prob = 0   (clean)
    20% <= epoch < 50%                   -> linear 0 -> max_drop
    epoch >= 50%                         -> max_drop  (plateau)

ModalityGradientProfiler is still attached so ``final_grad_norms`` ends
up in results.json (for downstream gradnorm-topK eval), but it never
influences dropout probabilities.

Run:
    python main_v6_iemocap_fusion_only_varlen_linear.py --fold=0 --cuda_pick=cuda:0
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
from utils.train_utils import (
    AverageMeter, ProgressMeter, FocalLoss,
    get_modality_curriculum,
    ModalityGradientProfiler,
)
from data.IEMOCAP.get_data import (
    get_dataloader as iemocap_get_dataloader,
    FEATURE_DIMS, NUM_CLASSES, IEMOCAPDataset,
)
from multimodal_model.v6_downsample_orig import (
    DualVideoBottleneckModelV6Downsample,
)


# ============================ V6 cfg adaptor ============================


class _IEMOCAPV6Cfg:
    """Minimal cfg shim that exposes the fields V6 reads from its
    ``cfg`` arg: ``modalities``, ``variates``, ``video_high_dim``,
    ``video_low_dim``."""

    def __init__(self, modalities=None):
        if modalities is None:
            modalities = list(IEMOCAPDataset.ALL_MODALITIES)
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        # V6 treats one modality named 'video' specially (uses two
        # separate dims for selector vs encoder paths).  For IEMOCAP
        # both paths use the same DINOv3 1024-d features.
        v_dim = FEATURE_DIMS.get('video', 1024)
        self.video_high_dim = v_dim
        self.video_low_dim = v_dim
        self.num_classes = NUM_CLASSES


# ============================ Wrapper ============================


class V6IEMOCAPFusionDynaWrapper(nn.Module):
    """Unpack the IEMOCAPDataset batch (12 modality tensors + label)
    into V6's ``(high_dim_inputs, low_dim_inputs)`` dicts and apply
    per-modality independent Bernoulli dropout during training.

    Modalities are unpacked in canonical order
    (``video, audio, text, mocap_hand, mocap_head, mocap_rotated``).
    Modality dropout removes the entry from both dicts; if all are
    dropped we keep one at random as a safety net.
    """

    def __init__(self, v6_model, modalities):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        # IEMOCAPDataset returns modalities in the same canonical order
        # as ``ALL_MODALITIES``.  Each modality contributes 2 tensors
        # (high, low) consecutively, so index for modality m is at
        # 2*pos_in_returned_list.
        self._mod_order = list(modalities)

    def _unpack(self, feats):
        """``feats`` = list/tuple of 2*M tensors interleaved
        (high0, low0, high1, low1, ...).  Returns two dicts keyed by
        modality name."""
        assert len(feats) == 2 * len(self._mod_order), (
            f'Expected {2*len(self._mod_order)} tensors, got {len(feats)}')
        high, low = {}, {}
        for j, m in enumerate(self._mod_order):
            high[m] = feats[2 * j]
            low[m] = feats[2 * j + 1]
        return high, low

    def forward(self, feats, modality_dropout_probs=None, training=False):
        high, low = self._unpack(feats)
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
        return out[0] if isinstance(out, tuple) else out


# ============================ Train / eval ============================


def _unpack_batch(batch, device):
    """IEMOCAPDataset collate returns
        (feat_0, feat_1, ..., feat_{2M-1}, label[B,1])
    """
    *feats, label = batch
    feats = [f.to(device).float() for f in feats]
    label = label.squeeze(-1).long().to(device)
    return feats, label


def train_one_epoch(loader, model, criterion, optimizer, epoch, device,
                    modality_dropout_probs, clip_grad=0.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.train()
    for i, batch in enumerate(loader):
        feats, y = _unpack_batch(batch, device)
        logits = model(feats, modality_dropout_probs=modality_dropout_probs,
                       training=True)
        loss = criterion(logits, y)
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / y.size(0)
        loss_meter.update(loss.item(), y.size(0))
        acc_meter.update(acc, y.size(0))
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if (i % 50 == 0) or (i == len(loader) - 1):
            progress = ProgressMeter(len(loader), [loss_meter, acc_meter],
                                     prefix=f"Epoch: [{epoch}]")
            progress.display(i)
            if i == len(loader) - 1:
                mean_p = (float(np.mean(list(modality_dropout_probs.values())))
                          if modality_dropout_probs else 0.0)
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
                      f'Acc: {acc_meter.avg:.4f}  meanModDrop: {mean_p:.3f}')
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_one_epoch(loader, model, criterion, epoch, device):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    for i, batch in enumerate(loader):
        feats, y = _unpack_batch(batch, device)
        logits = model(feats, modality_dropout_probs=None, training=False)
        loss = criterion(logits, y)
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / y.size(0)
        loss_meter.update(loss.item(), y.size(0))
        acc_meter.update(acc, y.size(0))
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
    description='V6 fusion-only training on IEMOCAP with grad-norm '
                'asymmetric modality-dropout curriculum (6 modalities).')

parser.add_argument('--exp_name', default='v6_fusion_only_varlen_linear', type=str)
parser.add_argument('--results_dir',
                    default='./results_v6_fusion_dyna_iemocap/', type=str)
parser.add_argument('--dataset', default='IEMOCAP', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# Modalities (subset selectable)
parser.add_argument('--modalities', nargs='+',
                    default=list(IEMOCAPDataset.ALL_MODALITIES),
                    choices=list(IEMOCAPDataset.ALL_MODALITIES))

# Sequence length / dual-resolution
parser.add_argument('--max_seq_len', default=128, type=int,
                    help='Truncate / pad every modality to this T '
                         '(high-resolution path).  0 = no resampling.')
parser.add_argument('--time_compression_ratio', default=4, type=int,
                    help='Low-resolution T = high T // this.')

# Model
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--internal_dim', default=128, type=int)
parser.add_argument('--n_bottlenecks', default=8, type=int)
parser.add_argument('--fusion_layer', default=0, type=int)
parser.add_argument('--use_sparse_attn', type=_str2bool, default=True)
parser.add_argument('--sparse_attn_variant', default='opt', type=str,
                    choices=['opt'],
                    help='Restricted to ``opt`` — the orig backbone does '
                         'not ship the stratified bmm variants.')
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=4, type=int)
parser.add_argument('--per_modal_distill', action='store_true', default=False)
parser.add_argument('--per_modal_downsample_min_len', default=4, type=int)
parser.add_argument('--use_batched_fusion', type=_str2bool, default=False,
                    help='IGNORED in the orig-model variant — always False. '
                         'Kept for backwards-compatibility with shared '
                         'driver scripts.')

# Modality-drop curriculum
parser.add_argument('--max_modality_drop', default=0.4, type=float,
                    help='Final per-modality base drop prob after the '
                         '20%%->50%% linear ramp.  Per-modality probs are '
                         '`base * M * (grad_norm_m / sum grad_norms)`, '
                         'clipped to [0, 0.8].')
parser.add_argument('--profile_update_freq', default=5, type=int)

# Optimizer / schedule
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--loss', default='ce', choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0, type=float)

# 3 random splits (split 0 / 1 / 2)
parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2],
                    help='IEMOCAP split id (0..2).')
parser.add_argument('--aggregate', action='store_true', default=False)

# Data path
parser.add_argument('--data_root', default='/files1/haodong/data/IEMOCAP',
                    type=str)
parser.add_argument('--available_sessions', nargs='+', type=int, default=None,
                    help='Restrict to these IEMOCAP sessions (1..5).  '
                         'Default: use all sessions present in the CSV.  '
                         'Useful when some Session{N}_features.tar.gz '
                         'archives are still being extracted.')

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
    n_folds = 3
    fold_results = []
    for fold_idx in range(n_folds):
        fpath = os.path.join(exp_dir, f"results_fold{fold_idx}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                fold_results.append(json.load(f))
    if not fold_results:
        print(f'No fold results found in {exp_dir}.')
        sys.exit(1)
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

modalities = list(args.modalities)
num_m = len(modalities)
print('Modalities:', modalities)

fold_id = args.fold if args.fold is not None else 0
print(f"IEMOCAP split id: {fold_id} (fold={args.fold})")

# NOTE: use_batched_fusion=False is the whole point of this script
# — it tells IEMOCAPDataset to leave T<max_seq_len modalities at their
# native length (no zero-padding), and pad_sequence in the collate fn
# then pads each modality to its own batch-max T at runtime.
train_loader, val_loader, eval_loader = iemocap_get_dataloader(
    data_root=args.data_root,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    modalities=modalities,
    split_id=fold_id,
    train_shuffle=True,
    max_seq_len=args.max_seq_len,
    time_compression_ratio=args.time_compression_ratio,
    use_batched_fusion=False,
    available_sessions=args.available_sessions,
)
input_length = args.max_seq_len if args.max_seq_len > 0 else 600


# ============================ Model ============================


v6_cfg = _IEMOCAPV6Cfg(modalities)

inner = DualVideoBottleneckModelV6Downsample(
    cfg=v6_cfg,
    output_dim=NUM_CLASSES,
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
    num_classes=NUM_CLASSES,
    sparse_attn_variant=args.sparse_attn_variant,
    strat_block_size=args.strat_block_size,
    downsample_min_len=args.downsample_min_len,
    # orig backbone forces this to False internally; pass for clarity.
    use_batched_fusion=False,
)

model = V6IEMOCAPFusionDynaWrapper(inner, modalities)
model = model.to(device).float()

profiler = ModalityGradientProfiler(inner, modalities)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = (FocalLoss(gamma=args.focal_gamma)
             if args.loss == 'focal' else nn.CrossEntropyLoss())
scheduler = None
if args.lr_schedule != 'constant':
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, CosineAnnealingWarmRestarts,
    )
    if args.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif args.lr_schedule == 'cosine_warmrestart':
        T_0 = max(int(num_epochs * 0.2), 1)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=2, eta_min=1e-6)

print(f'Parameters: {count_model_parameters(model)}')
print(f'LR schedule: {args.lr_schedule} (base_lr={args.lr})')
print(f'Modality drop (staggered + grad-norm asymmetric):')
print(f'   epoch [   0,  {int(num_epochs * 0.2):>3d})  -> 0')
print(f'   epoch [{int(num_epochs * 0.2):>3d}, {int(num_epochs * 0.5):>3d})  '
      f'-> linear 0 -> {args.max_modality_drop}')
print(f'   epoch [{int(num_epochs * 0.5):>3d}, {num_epochs:>3d}]  '
      f'-> {args.max_modality_drop} (plateau)')
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

    # LINEAR (uniform) modality dropout: every modality uses the same
    # mod_base probability — no gradnorm-based per-modality rescaling.
    # The ModalityGradientProfiler hook below records grad norms ONLY for
    # diagnostic ``final_grad_norms`` in results.json; it never feeds back
    # into dropout probabilities.
    current_mod_probs = {m: mod_base for m in modalities}
    if mod_base > 0 and epoch % args.profile_update_freq == 0:
        if any(len(profiler.grad_norms[m]) > 0 for m in modalities):
            snap = profiler.get_norm_snapshot()
            profile_log.append({
                'epoch': epoch, 'mod_base': mod_base,
                'grad_norms': snap,
                'mod_dropout_probs': {m: round(p, 4)
                                      for m, p in current_mod_probs.items()},
            })

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
print(f"  Best : epoch {best_epoch_val}, val_acc={best_val_acc:.4f}, "
      f"val_f1={best_val_f1:.4f}")
print(f"  Cold : epoch {cold_best_epoch}, val_acc={cold_best_val_acc:.4f}")

ckpt_name = (f"best_model_fold{args.fold}.pth"
             if args.fold is not None else "best_val_model.pth")
torch.save(best_model_state, os.path.join(exp_dir, ckpt_name))


# ============================ Clean test eval ============================


print("\n" + "=" * 80)
print("Test-set evaluation (clean, all modalities)")
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
    'model_variant': 'v6_fusion_only_varlen_linear',
    'model_class': 'DualVideoBottleneckModelV6Downsample (orig)',
    'backbone_module': 'multimodal_model.v6_downsample_orig',
    'variable_length_inputs': True,
    'selector': 'NONE',
    'modality_drop_strategy': 'staggered_linear_base + uniform_per_modality',
    'fold': args.fold, 'split_id': fold_id,
    'modalities': modalities,
    'num_classes': NUM_CLASSES,
    'feature_dims': {m: FEATURE_DIMS[m] for m in modalities},
    'max_seq_len': args.max_seq_len,
    'time_compression_ratio': args.time_compression_ratio,
    'num_epochs': num_epochs, 'batch_size': args.batch_size,
    'seed': args.seed_num,
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
    'use_batched_fusion': False,
    # per_modal_* knobs are bmm-specific; orig backbone ignores them.
    'max_modality_drop': args.max_modality_drop,
    'profile_update_freq': args.profile_update_freq,
    'lr': args.lr, 'lr_schedule': args.lr_schedule,
    'clip_grad': args.clip_grad,
    'loss': args.loss,
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
