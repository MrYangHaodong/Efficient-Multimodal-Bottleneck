"""CrossAttnV6Clf fusion baseline on EAV (6-modality, 4-class).

Fair model-swap counterpart of ``main_v6_eav_fusion_only_dyna.py``:
identical data pipeline (``EAVDataset`` via ``get_dataloader``,
``--max_seq_len 128``, 3 stratified splits) and identical
hyperparameters / curriculum (staggered base ramp + grad-norm
asymmetric per-modality scaling), but the backbone is
``models.cross_attn_v6_clf.CrossAttnV6Clf`` instead of
``DualVideoBottleneckModelV6Downsample``.

CrossAttnV6Clf consumes a single concatenated ``(B, T, sum(C_m))``
tensor.  Because every modality is resampled to ``max_seq_len`` inside
``EAVDataset`` (downsampled when T>max_seq_len, zero-padded
otherwise when ``use_batched_fusion=True``), all 6 modalities share the
same T and can be concatenated along the channel dim.  We drop the
"low" (time-compressed) tensors -- CrossAttnV6Clf only needs one
resolution -- and feed the high-resolution stack.

Schedule (mirrors ``utils.train_utils.get_modality_curriculum``):
    epoch < 20% of num_epochs            -> drop_prob = 0   (clean)
    20% <= epoch < 50%                   -> linear 0 -> max_drop
    epoch >= 50%                         -> max_drop  (plateau)

Once drop_prob > 0, ``ModalityGradientProfiler`` rescales::

    current_mod_probs[m] = clip( base_prob * M * (norm_m / sum norms), 0, 0.8 )

The ``CrossAttnV6Clf`` model itself is imported from the OLD package at
``/files1/haodong/MAESTRO_ttn_robustness_old/models``; the v6_organized
tree does not duplicate it.

Run:
    python script/main_crossattn_v6_eav.py --fold=0 --cuda_pick=cuda:0
    python script/main_crossattn_v6_eav.py --aggregate
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

# v6_organized root for EAVDataset / train_utils.
_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.dirname(_HERE)
if _V6_ROOT not in sys.path:
    sys.path.insert(0, _V6_ROOT)

from utils.helper_function import set_seed, count_model_parameters       # noqa: E402
from utils.train_utils import (                                          # noqa: E402
    AverageMeter, ProgressMeter, FocalLoss,
    get_modality_curriculum,
    ModalityGradientProfiler,
)
from data.EAV.get_data import (                                      # noqa: E402
    get_dataloader as eav_get_dataloader,
    FEATURE_DIMS, NUM_CLASSES, EAVDataset,
)

# Standalone CrossAttnV6Clf — no dependency on the old models/ tree.
from multimodal_model.cross_attn_v6_clf import CrossAttnV6Clf             # noqa: E402


# ============================ Cfg adaptor ============================


class _EAVCrossAttnCfg:
    """Minimal cfg shim that exposes the fields CrossAttnV6Clf reads:
    ``modalities`` and ``variates``.  Sequence length differs per
    modality natively but EAVDataset(--max_seq_len=T) forces all to
    the same T, so concat along channel dim is well-defined."""

    def __init__(self, modalities=None):
        if modalities is None:
            modalities = list(EAVDataset.ALL_MODALITIES)
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        self.num_classes = NUM_CLASSES


# ============================ Wrapper ============================


class CrossAttnV6EAVDynaWrapper(nn.Module):
    """Unpack the EAVDataset batch (2*M modality tensors + label),
    drop the "low" tensors, concat the high tensors along channel dim,
    and feed CrossAttnV6Clf a single ``(B, T, sum(C_m))`` tensor.  Apply
    per-modality independent Bernoulli dropout by zeroing the
    corresponding channel slice (matches the
    ``main_crossattn_v6_emognition.py`` recipe but with a per-modality
    probability dict instead of a single scalar).

    Modalities are unpacked in canonical order
    (``video, audio, text, mocap_hand, mocap_head, mocap_rotated``) --
    EAVDataset emits them in this exact order.
    """

    def __init__(self, model, modalities, variates):
        super().__init__()
        self.model = model
        self.modalities = list(modalities)
        self.variates = variates
        self._mod_slices = {}
        offset = 0
        for m in modalities:
            n_ch = variates[m]
            self._mod_slices[m] = (offset, offset + n_ch)
            offset += n_ch
        self._total_channels = offset

    def _concat_high(self, feats):
        """``feats`` = list of 2*M tensors interleaved
        (high0, low0, high1, low1, ...).  Returns a single
        ``(B, T, sum(C_m))`` tensor built from the high-resolution
        tensors (one per modality), aligning all modalities along the
        time axis (they all share T because of --max_seq_len).
        """
        assert len(feats) == 2 * len(self.modalities), (
            f'Expected {2*len(self.modalities)} tensors, got {len(feats)}')
        high_tensors = [feats[2 * j] for j in range(len(self.modalities))]
        # Sanity: every high tensor should share T (max_seq_len > 0).
        T = high_tensors[0].shape[1]
        B = high_tensors[0].shape[0]
        for t in high_tensors[1:]:
            assert t.shape[1] == T and t.shape[0] == B, (
                f'modality tensors disagree on (B,T): expected ({B},{T}), '
                f'got {tuple(t.shape)}.  Did you forget --max_seq_len > 0?')
        return torch.cat(high_tensors, dim=2)  # (B, T, sum C_m)

    def forward(self, feats, modality_dropout_probs=None, training=False):
        x = self._concat_high(feats)
        device = x.device
        modality_mask = torch.ones(len(self.modalities), device=device)
        if training and modality_dropout_probs is not None:
            kept = None
            for _ in range(8):
                trial = [m for m in self.modalities
                         if torch.rand(1).item() >= modality_dropout_probs.get(m, 0.0)]
                if trial:
                    kept = trial
                    break
            if kept is None:
                idx = int(torch.randint(0, len(self.modalities), (1,)).item())
                kept = [self.modalities[idx]]
            kept_set = set(kept)
            x = x.clone()
            for idx, m in enumerate(self.modalities):
                if m not in kept_set:
                    s, e = self._mod_slices[m]
                    x[:, :, s:e] = 0.0
                    modality_mask[idx] = 0.0
        logits, _ = self.model(x, modality_dropout_prob=0.0, training=training)
        return logits, modality_mask


# ============================ Train / eval ============================


def _unpack_batch(batch, device):
    """EAVDataset collate returns
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
        logits, _ = model(feats, modality_dropout_probs=modality_dropout_probs,
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
        logits, _ = model(feats, modality_dropout_probs=None, training=False)
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
    description='CrossAttnV6Clf on EAV with grad-norm-based asymmetric '
                'modality-dropout curriculum (fusion-only-dyna recipe).')

parser.add_argument('--exp_name', default='crossattn_v6_dyna', type=str)
parser.add_argument('--results_dir',
                    default='./results_crossattn_v6_dyna_eav/', type=str)
parser.add_argument('--dataset', default='EAV', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# Modalities (subset selectable)
parser.add_argument('--modalities', nargs='+',
                    default=list(EAVDataset.ALL_MODALITIES),
                    choices=list(EAVDataset.ALL_MODALITIES))

# Sequence length (must be > 0 for CrossAttnV6Clf -- channel-concat
# requires a shared T across modalities; EAVDataset resamples
# every modality to max_seq_len when this is > 0.)
parser.add_argument('--max_seq_len', default=128, type=int,
                    help='Truncate / pad every modality to this T.  '
                         'MUST be > 0 for CrossAttnV6 (channel-concat '
                         'requires a shared time axis).')
parser.add_argument('--time_compression_ratio', default=4, type=int,
                    help='Low-resolution T = high T // this.  '
                         'CrossAttnV6Clf uses the high-resolution path only.')

# Model
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--use_sparse_attn', type=_str2bool, default=True)
parser.add_argument('--sparse_attn_variant', default='opt', type=str,
                    choices=['orig', 'opt', 'opt_strat', 'opt_strat_blk'])
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--per_modal_distill', action='store_true', default=False)
parser.add_argument('--per_modal_downsample_min_len', default=4, type=int)
parser.add_argument('--use_batched_fusion', type=_str2bool, default=True)

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
                    help='EAV split id (0..2).')
parser.add_argument('--split_mode', default='random', type=str,
                    choices=['random', 'cross_subject'],
                    help='random: utterance-level splits. cross_subject: session-disjoint.')
parser.add_argument('--aggregate', action='store_true', default=False)

# Data path
parser.add_argument('--data_root', default='/files1/haodong/data/EAV',
                    type=str)
parser.add_argument('--available_sessions', nargs='+', type=int, default=None,
                    help='Restrict to these EAV sessions (1..5).  '
                         'Default: use all sessions present in the CSV.')

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

if args.max_seq_len <= 0:
    raise ValueError(
        '--max_seq_len must be > 0 for CrossAttnV6Clf on EAV '
        '(channel-concat requires a shared time axis).')

fold_id = args.fold if args.fold is not None else 0
print(f"EAV split id: {fold_id} (fold={args.fold})  split_mode={args.split_mode}")

train_loader, val_loader, eval_loader = eav_get_dataloader(
    data_root=args.data_root,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    modalities=modalities,
    split_id=fold_id,
    train_shuffle=True,
    max_seq_len=args.max_seq_len,
    time_compression_ratio=args.time_compression_ratio,
    use_batched_fusion=args.use_batched_fusion,
    available_sessions=args.available_sessions,
    split_mode=args.split_mode,
)
input_length = args.max_seq_len


# ============================ Model ============================


cfg = _EAVCrossAttnCfg(modalities)

inner = CrossAttnV6Clf(
    cfg=cfg,
    num_classes=NUM_CLASSES,
    input_length=input_length,
    d_model=args.d_model, nhead=args.nhead,
    num_layers_per_modal=args.num_layers_per_modal,
    num_layers=args.num_layers, dropout=args.dropout, verbose=True,
    base_factor=args.base_factor, num_experts=args.num_experts,
    use_sparse_attn=args.use_sparse_attn,
    sparse_attn_variant=args.sparse_attn_variant,
    strat_block_size=args.strat_block_size,
    per_modal_distill=args.per_modal_distill,
    per_modal_downsample_min_len=args.per_modal_downsample_min_len,
)

model = CrossAttnV6EAVDynaWrapper(inner, modalities, cfg.variates)
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
    'model_variant': 'crossattn_v6_dyna',
    'model_class': 'CrossAttnV6Clf',
    'selector': 'NONE',
    'modality_drop_strategy': 'staggered_linear_base + gradient_asymmetric_scaling',
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
    'use_sparse_attn': args.use_sparse_attn,
    'sparse_attn_variant': args.sparse_attn_variant,
    'strat_block_size': args.strat_block_size,
    'use_batched_fusion': args.use_batched_fusion,
    'per_modal_distill': args.per_modal_distill,
    'per_modal_downsample_min_len': args.per_modal_downsample_min_len,
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
