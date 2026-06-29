"""DecAlign baseline training on MOSEI with the V6 **dyna**
(grad-norm-based asymmetric modality-dropout) curriculum.

MOSEI-specific notes vs the DaliaHAR sibling:
  * 6 modalities (video / audio / text / mocap_hand / mocap_head /
    mocap_rotated) with very heterogeneous channel widths (1280, 1024,
    768, 165, 18, 6).
  * Pre-extracted features (DINOv3 / WAV2VEC2 / BERT / MOCAP) — no SAX
    token transform (we use raw float, ``transform=None``).
  * 3 stratified 70/15/15 splits (``--fold = 0/1/2``).
  * 4-class emotion classification (``NUM_CLASSES = 4``).
  * Sequence length: ``--max_seq_len`` (default 128) for all modalities;
    the loader truncates / zero-pads to this T.

Schedule mirrors ``main_v6_mosei_fusion_only_dyna.py``:
    epoch < 20% -> 0   |   20% <= e < 50% -> linear 0 -> max_drop
    e >= 50%    -> max_drop (plateau)

Once drop_prob > 0, every ``--profile_update_freq`` epochs we sample
recent gradient norms via :class:`ModalityGradientProfiler` and rescale
per-modality drop probability proportional to its grad-norm share.

Run:
    python main_decalign_mosei_dyna.py --fold=0 --cuda_pick=cuda:0
    python main_decalign_mosei_dyna.py --aggregate
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _V6_ROOT)

from utils.helper_function import set_seed, count_model_parameters         # noqa: E402
from utils.train_utils import (                                            # noqa: E402
    AverageMeter, ProgressMeter,
    get_modality_curriculum, ModalityGradientProfiler,
)
from data.CMU_MOSEI.get_data import (                                        # noqa: E402
    get_dataloader as mosei_get_dataloader,
    FEATURE_DIMS, NUM_CLASSES, CMUMOSEIDataset,
)
from multimodal_model.decalign_baseline import build_decalign              # noqa: E402


# ============================ Per-dataset cfg shim ============================

class _MOSEICfg:
    """Minimal cfg consumed by DecAlign + the gradient profiler."""

    def __init__(self, modalities):
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        self.num_classes = NUM_CLASSES


# ============================ Wrapper ============================

class DecAlignMOSEIDynaWrapper(nn.Module):
    """Unpacks the CMUMOSEIDataset batch (2M interleaved tensors + label),
    constructs the per-modality dict {m: [B,T,V_m]} that DecAlign accepts,
    and applies per-modality Bernoulli dropout (zero out the modality's
    tensor when dropped).
    """

    def __init__(self, model, modalities):
        super().__init__()
        self.model = model
        self.modalities = list(modalities)

    def _unpack(self, feats):
        """feats = [high0, low0, high1, low1, ...]; we use the high (full
        resolution) entries for DecAlign, which expects [B,T,V_m]."""
        assert len(feats) == 2 * len(self.modalities), (
            f'Expected {2*len(self.modalities)} tensors, got {len(feats)}')
        return {m: feats[2 * j] for j, m in enumerate(self.modalities)}

    def forward(self, feats, modality_dropout_probs=None, training=False):
        d = self._unpack(feats)
        if training and modality_dropout_probs is not None:
            kept = None
            for _ in range(8):
                trial = [m for m in self.modalities
                         if torch.rand(1).item() >=
                         float(modality_dropout_probs.get(m, 0.0))]
                if trial:
                    kept = set(trial)
                    break
            if not kept:
                kept = {self.modalities[
                    int(torch.randint(0, len(self.modalities), (1,)).item())]}
            d = {m: (v if m in kept else torch.zeros_like(v))
                 for m, v in d.items()}
        return self.model(d)


# ============================ Train / Eval ============================

def _unpack_batch(batch, device):
    *feats, label = batch
    feats = [f.to(device).float() for f in feats]
    if isinstance(label, list):
        label = torch.as_tensor(label, dtype=torch.long, device=device)
    else:
        label = label.long().to(device)
        if label.ndim > 1:
            label = label.squeeze(-1)
    return feats, label


def train_one_epoch(loader, model, optimizer, epoch, device,
                    modality_dropout_probs, alpha1, alpha2,
                    clip_grad=1.0, label_smoothing=0.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.train()
    for i, batch in enumerate(loader):
        feats, y = _unpack_batch(batch, device)
        out = model(feats, modality_dropout_probs=modality_dropout_probs,
                    training=True)
        task_loss = F.cross_entropy(out['logits'], y,
                                    label_smoothing=label_smoothing)
        total = task_loss \
            + alpha1 * out['dec_loss'] \
            + alpha2 * (out['hete_loss'] + out['homo_loss'])
        _, predicted = torch.max(out['logits'], 1)
        bs = y.size(0)
        acc = predicted.eq(y).sum().item() / bs
        loss_meter.update(float(total.detach().cpu()), bs)
        acc_meter.update(acc, bs)
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
    for i, batch in enumerate(loader):
        feats, y = _unpack_batch(batch, device)
        out = model(feats, modality_dropout_probs=None, training=False)
        loss = F.cross_entropy(out['logits'], y,
                               label_smoothing=label_smoothing)
        _, predicted = torch.max(out['logits'], 1)
        bs = y.size(0)
        acc = predicted.eq(y).sum().item() / bs
        loss_meter.update(loss.item(), bs)
        acc_meter.update(acc, bs)
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
    description='DecAlign on MOSEI with V6 grad-norm dyna modality-drop.')

parser.add_argument('--exp_name', default='decalign_mosei_dyna', type=str)
parser.add_argument('--results_dir',
                    default='./results_decalign_mosei_dyna/', type=str)
parser.add_argument('--dataset', default='CMU_MOSEI', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# Modalities subset (default = all 6)
parser.add_argument('--modalities', nargs='+',
                    default=list(CMUMOSEIDataset.ALL_MODALITIES),
                    choices=list(CMUMOSEIDataset.ALL_MODALITIES))

# Sequence-length (MOSEI loader pads/truncates every modality to this T)
parser.add_argument('--max_seq_len', default=50, type=int)
parser.add_argument('--time_compression_ratio', default=4, type=int)

# DecAlign hyperparams (kept conservative; K=3 controls Sinkhorn K^M=3^6=729)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--num_prototypes', default=3, type=int,
                    help='K (Sinkhorn cost tensor is K^M).  K=3 for M=6 -> 729.')
parser.add_argument('--conv1d_kernel', default=3, type=int)
parser.add_argument('--ot_reg', default=0.1, type=float)
parser.add_argument('--ot_num_iters', default=50, type=int)
parser.add_argument('--alpha1', default=0.1, type=float)
parser.add_argument('--alpha2', default=0.01, type=float)
parser.add_argument('--dropout', default=0.1, type=float)

# Modality-drop curriculum
parser.add_argument('--max_modality_drop', default=0.4, type=float)
parser.add_argument('--profile_update_freq', default=5, type=int)

# Optimizer / schedule
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)

parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2])
parser.add_argument('--split_mode', default='random', type=str, choices=['random','cross_subject'], help='session-disjoint cross-subject splits')
parser.add_argument('--aggregate', action='store_true', default=False)
parser.add_argument('--data_root', default='/files1/haodong/data/CMU-MOSI/CMU-MOSEI',
                    type=str)
parser.add_argument('--available_sessions', nargs='+', type=int, default=None)
parser.add_argument('--use_batched_fusion', type=_str2bool, default=True)

parser.add_argument('--mosei_task', default=2, type=int, choices=[2, 5, 7],
                    help='MOSEI classification granularity: 2=Acc-2 (sign), 5=Acc-5, 7=Acc-7.')

args = parser.parse_args()
# CMU-MOSEI classification granularity: rebind NUM_CLASSES (all runtime
# usages - cfg instantiation, model build, config dict - run below this).
NUM_CLASSES = args.mosei_task
_MOSEI_TASK = {2: 'classification2', 5: 'classification5', 7: 'classification7'}[args.mosei_task]



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

modalities = list(args.modalities)
num_m = len(modalities)
print('Modalities:', modalities)

fold_id = args.fold if args.fold is not None else 0
print(f'MOSEI split id: {fold_id} (fold={args.fold})')

train_loader, val_loader, eval_loader = mosei_get_dataloader(
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
    task=_MOSEI_TASK,
)
input_length = args.max_seq_len if args.max_seq_len > 0 else 600


# ============================ Model ============================

cfg = _MOSEICfg(modalities)

inner = build_decalign(
    cfg=cfg, input_length=input_length,
    d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
    num_prototypes=args.num_prototypes, ot_reg=args.ot_reg,
    ot_num_iters=args.ot_num_iters, conv1d_kernel=args.conv1d_kernel,
    dropout=args.dropout, alpha1=args.alpha1, alpha2=args.alpha2,
    transform=None, verbose=True,
)
model = DecAlignMOSEIDynaWrapper(inner, modalities)
model = model.to(device).float()

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
        train_loader, model, optimizer, epoch, device,
        modality_dropout_probs=current_mod_probs,
        alpha1=args.alpha1, alpha2=args.alpha2,
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
    'model_variant': 'decalign_mosei_dyna',
    'model_class': 'DecAlignDSADS',
    'modality_drop_strategy':
        'staggered_linear_base + gradient_asymmetric_scaling',
    'fold': args.fold, 'split_id': fold_id,
    'modalities': modalities,
    'num_classes': NUM_CLASSES,
    'feature_dims': {m: FEATURE_DIMS[m] for m in modalities},
    'max_seq_len': args.max_seq_len,
    'time_compression_ratio': args.time_compression_ratio,
    'num_epochs': num_epochs, 'batch_size': args.batch_size,
    'seed': args.seed_num,
    'd_model': args.d_model, 'nhead': args.nhead,
    'num_layers': args.num_layers, 'num_prototypes': args.num_prototypes,
    'conv1d_kernel': args.conv1d_kernel,
    'ot_reg': args.ot_reg, 'ot_num_iters': args.ot_num_iters,
    'alpha1': args.alpha1, 'alpha2': args.alpha2, 'dropout': args.dropout,
    'max_modality_drop': args.max_modality_drop,
    'profile_update_freq': args.profile_update_freq,
    'lr': args.lr, 'lr_schedule': args.lr_schedule,
    'clip_grad': args.clip_grad, 'weight_decay': args.weight_decay,
    'label_smoothing': args.label_smoothing,
    'data_root': args.data_root,
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
