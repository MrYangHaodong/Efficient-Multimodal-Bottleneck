"""QMF / PDF confidence-weighted **late fusion** on MOSEI.

MOSEI sibling of `main_qmfpdf_daliahar_late_fusion.py`.  Same V6 backbone
(`DualVideoBottleneckModelV6Downsample`, `no_selector=True`) used purely as a
per-modality encoder: we pool `model._last_processed_modalities` into one
[B, H] vector per modality, attach a unimodal classifier head per modality,
and combine the unimodal logits by a dynamically predicted quality weight
(QMF energy-confidence or PDF Co-Belief).  See `qmf_pdf_late_fusion.py`.

Differences vs DaliaHAR:
  - up to 6 heterogeneous modalities; `CMUMOSEIDataset` returns a list of
    2*M feature tensors + label that the wrapper unpacks into V6's two dicts.
  - 3 splits (`--fold 0/1/2`); `--split_mode cross_subject` (session-disjoint)
    by default.
  - No SAX transform (pre-extracted features).

There is **no modality-dropout curriculum** — QMF/PDF handle low-quality
modalities at inference time via the learned per-sample weighting.

Run:
    python main_qmfpdf_mosei_late_fusion.py --fusion_method qmf --fold 0 --cuda_pick cuda:0
    python main_qmfpdf_mosei_late_fusion.py --fusion_method pdf --fold 0 --cuda_pick cuda:0
    python main_qmfpdf_mosei_late_fusion.py --fusion_method qmf --aggregate
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
from utils.train_utils import AverageMeter, ProgressMeter, FocalLoss
from data.CMU_MOSEI.get_data import (
    get_dataloader as mosei_get_dataloader,
    FEATURE_DIMS, NUM_CLASSES, CMUMOSEIDataset,
)
from multimodal_model.v6_downsample_opt_batched_late_fusion import (
    DualVideoBottleneckModelV6Downsample,
)
from multimodal_model.cross_attn_unified_clf import CrossAttnUnifiedClf

from qmf_pdf_late_fusion import LateFusionHead, compute_loss
from multimodal_model.latefusion_transformer import GenericLateFusionTransformer, IEMOCAPLateFusionWrapper


# ============================ V6 cfg adaptor ============================

class _MOSEIV6Cfg:
    def __init__(self, modalities=None):
        if modalities is None:
            modalities = list(CMUMOSEIDataset.ALL_MODALITIES)
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        v_dim = FEATURE_DIMS.get('video', 1024)
        self.video_high_dim = v_dim
        self.video_low_dim = v_dim
        self.num_classes = NUM_CLASSES


# ============================ Wrapper ============================

class QMFPDFMOSEIWrapper(nn.Module):
    """Unpack the CMUMOSEIDataset batch (2*M modality tensors) into V6's two
    dicts, run the backbone purely as a per-modality encoder, pool each
    modality to [B, H], and combine the unimodal logits with the QMF/PDF
    late-fusion head."""

    def __init__(self, v6_model, modalities, num_classes, d_model, method):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self._mod_order = list(modalities)
        self.fusion = LateFusionHead(modalities, d_model, num_classes, method)

    def _unpack(self, feats):
        assert len(feats) == 2 * len(self._mod_order), (
            f'Expected {2*len(self._mod_order)} tensors, got {len(feats)}')
        high, low = {}, {}
        for j, m in enumerate(self._mod_order):
            high[m] = feats[2 * j]
            low[m] = feats[2 * j + 1]
        return high, low

    def forward(self, feats):
        high, low = self._unpack(feats)
        _ = self.model(high, low, training=self.training,
                       return_selection_info=False)
        processed = self.model._last_processed_modalities
        z_per = {m: feat.mean(dim=1) for m, feat in processed.items()}
        return self.fusion(z_per)


# ============================ Train / eval ============================

def _unpack_batch(batch, device):
    *feats, label = batch
    feats = [f.to(device).float() for f in feats]
    label = label.squeeze(-1).long().to(device)
    return feats, label


def train_one_epoch(loader, model, criterion, optimizer, epoch, device,
                    method, lambda_reg, lambda_tcp, clip_grad=0.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    aux_meter = AverageMeter('Aux', ':.4f')
    model.train()
    for i, batch in enumerate(loader):
        feats, y = _unpack_batch(batch, device)
        out = model(feats)
        loss, aux = compute_loss(method, out, y, criterion, lambda_reg, lambda_tcp)
        logits = out['fused']
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / y.size(0)
        loss_meter.update(loss.item(), y.size(0))
        acc_meter.update(acc, y.size(0))
        aux_meter.update(float(aux), y.size(0))
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
                aux_name = 'reg' if method == 'qmf' else 'tcp'
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
                      f'Acc: {acc_meter.avg:.4f}  {aux_name}: {aux_meter.avg:.4f}')
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_one_epoch(loader, model, criterion, epoch, device):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    for i, batch in enumerate(loader):
        feats, y = _unpack_batch(batch, device)
        out = model(feats)
        logits = out['fused']
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
    description='QMF / PDF confidence-weighted late fusion on MOSEI.')

parser.add_argument('--fusion_method', default='qmf', choices=['qmf', 'pdf'],
                    help="'qmf'=energy-confidence weighting + ranking reg; "
                         "'pdf'=Co-Belief (TCP mono-conf x holo-conf) + TCP loss.")
parser.add_argument('--lambda_reg', default=0.5, type=float,
                    help='QMF confidence-correctness ranking regularization weight.')
parser.add_argument('--lambda_tcp', default=1.0, type=float,
                    help='PDF True-Class-Probability regression weight.')

parser.add_argument('--exp_name', default='qmfpdf_late_fusion_cs', type=str)
parser.add_argument('--model_arch', default='v6', choices=['v6', 'unified_ca'])
parser.add_argument('--input_length_override', default=-1, type=int)
parser.add_argument('--ca_d_ff_mult', default=1, type=int)
parser.add_argument('--results_dir', default='./results_qmfpdf_mosei/', type=str)
parser.add_argument('--dataset', default='CMU_MOSEI', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--modalities', nargs='+',
                    default=list(CMUMOSEIDataset.ALL_MODALITIES),
                    choices=list(CMUMOSEIDataset.ALL_MODALITIES))

parser.add_argument('--max_seq_len', default=50, type=int)
parser.add_argument('--time_compression_ratio', default=4, type=int)

# Model
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--base_factor', default=10, type=int)
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
                    choices=['opt', 'opt_strat', 'opt_strat_blk'])
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=4, type=int)
parser.add_argument('--per_modal_distill', action='store_true', default=False)
parser.add_argument('--per_modal_downsample_min_len', default=4, type=int)
parser.add_argument('--bottleneck_init_mode', default='random', type=str,
                    choices=['random', 'modal_mean_sinpe'])
parser.add_argument('--use_batched_fusion', type=_str2bool, default=True)
parser.add_argument('--fusion_add_pos_embeds', type=_str2bool, default=False)
parser.add_argument('--fusion_pos_embeds_max_len', type=int, default=256)
parser.add_argument('--fusion_pos_embed_mode', default='full',
                    choices=['full', 'decoupled', 'id_only', 'gated_id'])
parser.add_argument('--fusion_mlp_ratio', type=float, default=1.0)
parser.add_argument('--mlp_ratio', type=float, default=1.0)
parser.add_argument('--share_fusion_params', action='store_true', default=False)

# Modality-drop curriculum
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--loss', default='ce', choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)

parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2])
parser.add_argument('--split_mode', default='random', type=str,
                    choices=['random', 'cross_subject'])
parser.add_argument('--aggregate', action='store_true', default=False)

parser.add_argument('--data_root', default='/files1/haodong/data/CMU-MOSI/CMU-MOSEI', type=str)
parser.add_argument('--available_sessions', nargs='+', type=int, default=None)

parser.add_argument('--mosei_task', default=2, type=int, choices=[2, 5, 7],
                    help='MOSEI classification granularity: 2=Acc-2 (sign), 5=Acc-5, 7=Acc-7.')

args = parser.parse_args()
# CMU-MOSEI classification granularity: rebind NUM_CLASSES (all runtime
# usages - cfg instantiation, model build, config dict - run below this).
NUM_CLASSES = args.mosei_task
_MOSEI_TASK = {2: 'classification2', 5: 'classification5', 7: 'classification7'}[args.mosei_task]



# ============================ Setup ============================

num_epochs = args.num_epochs
device = torch.device(args.cuda_pick if torch.cuda.is_available() else "cpu")
current_date = datetime.now().strftime('%Y-%m-%d')
exp_name_full = f"{current_date}_{args.dataset}_{args.exp_name}_{args.fusion_method}"
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
print(f"Fusion method: {args.fusion_method.upper()}")

modalities = list(args.modalities)
num_m = len(modalities)
print('Modalities:', modalities)

fold_id = args.fold if args.fold is not None else 0
print(f"MOSEI split id: {fold_id} (fold={args.fold})  split_mode={args.split_mode}")

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
if args.input_length_override > 0:
    input_length = args.input_length_override
    print(f"[input_length override] using input_length={input_length}")


# ============================ Model ============================

v6_cfg = _MOSEIV6Cfg(modalities)
# Late-fusion transformer (official QMF/PDF style): 6-layer per-modality encoder,
# NO cross-modal fusion network. MOSEI feeds a feats-list, so the wrapper
# concatenates the per-modality highs to [B,T,sum(V)] then late-fuses.
inner = GenericLateFusionTransformer(
    cfg=v6_cfg, input_length=input_length, input_mode='float',
    d_model=args.d_model, nhead=args.nhead, num_layers=6,
    dropout=args.dropout, method=args.fusion_method,
)
model = IEMOCAPLateFusionWrapper(inner, modalities)
model = model.to(device).float()

optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)
criterion = (FocalLoss(gamma=args.focal_gamma)
             if args.loss == 'focal'
             else nn.CrossEntropyLoss(label_smoothing=args.label_smoothing))
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
print(f'Fusion: {args.fusion_method.upper()} late fusion over {modalities}')
print(f'Selector:      NONE (per-modality encoders + confidence fusion)')


# ============================ Training loop ============================

best_val_acc, best_val_f1, best_epoch_val = 0.0, 0.0, -1
best_model_state = None

writer = SummaryWriter(
    comment=f"_{exp_name_full}_fold{args.fold}"
            if args.fold is not None else f"_{exp_name_full}")

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        train_loader, model, criterion, optimizer, epoch, device,
        method=args.fusion_method, lambda_reg=args.lambda_reg,
        lambda_tcp=args.lambda_tcp, clip_grad=args.clip_grad)

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
    writer.add_scalar('schedule/lr', current_lr, epoch)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_epoch_val = epoch
        best_model_state = model.state_dict()


# ============================ Model selection ============================

print(f"\nBest : epoch {best_epoch_val}, val_acc={best_val_acc:.4f}, "
      f"val_f1={best_val_f1:.4f}")

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

config = {
    'experiment_name': exp_name_full,
    'dataset': args.dataset,
    'model_variant': f'qmfpdf_late_fusion_{args.fusion_method}',
    'model_class': 'DualVideoBottleneckModelV6Downsample',
    'fusion_method': args.fusion_method,
    'lambda_reg': args.lambda_reg, 'lambda_tcp': args.lambda_tcp,
    'selector': 'NONE',
    'fold': args.fold, 'split_id': fold_id, 'split_mode': args.split_mode,
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
    'use_batched_fusion': args.use_batched_fusion,
    'per_modal_distill': args.per_modal_distill,
    'per_modal_downsample_min_len': args.per_modal_downsample_min_len,
    'lr': args.lr, 'lr_schedule': args.lr_schedule,
    'clip_grad': args.clip_grad,
    'loss': args.loss,
    'data_root': args.data_root,
}
with open(os.path.join(exp_dir, "config.json"), 'w') as f:
    json.dump(config, f, indent=4)

model_stats = {
    'experiment_name': exp_name_full, 'fold': args.fold,
    'fusion_method': args.fusion_method,
    'best_val_acc': best_val_acc, 'best_val_f1': best_val_f1,
    'best_epoch_val': best_epoch_val,
    'best_test_acc': float(test_acc),
    'best_test_f1': float(test_f1),
}
results_filename = (f"results_fold{args.fold}.json"
                    if args.fold is not None else "results.json")
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(model_stats, f, indent=4)

print(f"\nBest Model (Epoch {best_epoch_val}): "
      f"val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
print(f"Directory: {exp_dir}")
