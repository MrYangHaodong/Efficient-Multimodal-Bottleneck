"""DynMM / DyMo on PAMAP2, V6 backbone (2-layer per-modality encoder +
4-layer bottleneck fusion) reused as the modality-subset encoder/fusion.

  --method dynmm : gated expert branches (modality subsets) + resource penalty
                   (Dynamic Multimodal Fusion, Xue & Marculescu, CVPR-W 2023).
  --method dymo  : mask-robust backbone + inference-time greedy per-sample
                   modality selection (Du et al., ICLR 2026).

See `dynmm_dymo_v6.py` for the exact gate / greedy-selection definitions.

Run:
    python main_dynmm_dymo_pamap2_late_fusion.py --method dynmm --fold 0 --cuda_pick cuda:0
    python main_dynmm_dymo_pamap2_late_fusion.py --method dymo  --fold 0 --cuda_pick cuda:0
    python main_dynmm_dymo_pamap2_late_fusion.py --method dynmm --aggregate
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed, count_model_parameters
from utils.dataset_cfg import PAMAP2
from data.pamap2_crosssubject import load_pamap2_subjects
from utils.train_utils import AverageMeter, ProgressMeter, FocalLoss

from multimodal_model.v6_downsample_opt_batched_late_fusion import DualVideoBottleneckModelV6Downsample
from multimodal_model.cross_attn_unified_clf import CrossAttnUnifiedClf

from dynmm_dymo_v6 import DynMMWrapper, DyMoWrapper, dymo_train_loss
from multimodal_model.crossmodal_transformer import CrossModalTransformer, CrossModalDyMo, CrossModalDynMM


# ============================ V6 cfg adaptor ============================

class _DaliaV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


# ============================ Train / eval ============================

def _anneal_temp(epoch, num_epochs, t0, t1):
    if t0 <= 0 or num_epochs <= 1:
        return t1
    return max(t1, t0 * (t1 / t0) ** (epoch / (num_epochs - 1)))


def _sample_keep(modalities, mask_prob):
    for _ in range(8):
        keep = [m for m in modalities if torch.rand(1).item() >= mask_prob]
        if keep:
            return keep
    idx = int(torch.randint(0, len(modalities), (1,)).item())
    return [modalities[idx]]


def train_dynmm(loader, model, criterion, optimizer, epoch, device,
                temp, hard, loss_ratio, clip_grad, unpack):
    loss_meter, acc_meter, res_meter = (AverageMeter('Loss', ':.4f'),
                                        AverageMeter('Acc', ':.4f'),
                                        AverageMeter('Res', ':.4f'))
    model.train()
    for i, batch in enumerate(loader):
        inp, y = unpack(batch, device)
        out = model(inp, temp=temp, hard=hard)
        res = model.resource_loss(out['weights'])
        loss = criterion(out['fused'], y) + loss_ratio * res
        _, pred = torch.max(out['fused'], 1)
        bs = y.size(0)
        acc_meter.update(pred.eq(y).sum().item() / bs, bs)
        loss_meter.update(loss.item(), bs)
        res_meter.update(res.item(), bs)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if (i % 50 == 0) or (i == len(loader) - 1):
            ProgressMeter(len(loader), [loss_meter, acc_meter],
                          prefix=f"Epoch: [{epoch}]").display(i)
            if i == len(loader) - 1:
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
                      f'Acc: {acc_meter.avg:.4f}  res: {res_meter.avg:.4f}  '
                      f'temp: {temp:.3f}  hard: {hard}')
    return loss_meter.avg, acc_meter.avg


def train_dymo(loader, model, criterion, optimizer, epoch, device,
               mask_prob, alpha_per_mod, clip_grad, unpack, modalities):
    loss_meter, acc_meter = AverageMeter('Loss', ':.4f'), AverageMeter('Acc', ':.4f')
    model.train()
    for i, batch in enumerate(loader):
        inp, y = unpack(batch, device)
        keep = _sample_keep(modalities, mask_prob)
        mask = model.keep_mask(y.size(0), device, keep)
        combined = model(inp, mask)
        loss = criterion(combined, y)
        _, pred = torch.max(combined, 1)
        bs = y.size(0)
        acc_meter.update(pred.eq(y).sum().item() / bs, bs)
        loss_meter.update(loss.item(), bs)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if (i % 50 == 0) or (i == len(loader) - 1):
            ProgressMeter(len(loader), [loss_meter, acc_meter],
                          prefix=f"Epoch: [{epoch}]").display(i)
            if i == len(loader) - 1:
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
                      f'Acc: {acc_meter.avg:.4f}  keep: {len(keep)}/{len(modalities)}')
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(loader, model, criterion, epoch, device, method, unpack,
             end_temp=0.001):
    loss_meter, acc_meter = AverageMeter('Loss', ':.4f'), AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    n_sel_sum, n_sel_cnt = 0.0, 0
    for batch in loader:
        inp, y = unpack(batch, device)
        if method == 'dynmm':
            out = model(inp, temp=end_temp, hard=True)
            logits = out['fused']
            n_sel_sum += model.expected_cost(out['weights']) * y.size(0)
        else:  # dymo
            logits, mask, _ = model.forward_select(inp)
            n_sel_sum += mask.float().sum(dim=1).sum().item()
        n_sel_cnt += y.size(0)
        loss = criterion(logits, y)
        _, pred = torch.max(logits, 1)
        acc_meter.update(pred.eq(y).sum().item() / y.size(0), y.size(0))
        loss_meter.update(loss.item(), y.size(0))
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())
    extra = (f'expected_cost={n_sel_sum / max(n_sel_cnt, 1):.3f}' if method == 'dynmm'
             else f'avg_selected={n_sel_sum / max(n_sel_cnt, 1):.2f}')
    print(f'End of Epoch {epoch} | Val Loss: {loss_meter.avg:.4f} '
          f'| Val Acc: {acc_meter.avg:.4f} | {extra}')
    return (loss_meter.avg, acc_meter.avg,
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(),
            n_sel_sum / max(n_sel_cnt, 1))


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


def _parse_branches(spec, modalities):
    """'sub1;sub2;...' where each sub is comma-sep modality names or 'ALL'."""
    if not spec:
        return None
    branches = []
    for part in spec.split(';'):
        part = part.strip()
        if part.upper() == 'ALL':
            branches.append(list(modalities))
        else:
            branches.append([m.strip() for m in part.split(',') if m.strip()])
    return branches


parser = argparse.ArgumentParser(
    description='DynMM / DyMo on PAMAP2 (V6 2-enc + 4-fusion backbone).')

parser.add_argument('--method', default='dynmm', choices=['dynmm', 'dymo'])
# DynMM knobs
parser.add_argument('--dynmm_branches', default='', type=str,
                    help="Semicolon-separated expert subsets, e.g. 'chest_ACC;ALL'. "
                         "Default: [first modality], [all modalities].")
parser.add_argument('--loss_ratio', default=1e-2, type=float,
                    help='DynMM resource (expected-cost) penalty weight.')
parser.add_argument('--temp', default=1.0, type=float, help='Gate start temperature.')
parser.add_argument('--end_temp', default=1e-3, type=float, help='Gate final temperature.')
parser.add_argument('--epoch_hard', default=50, type=int,
                    help='Epoch from which the gate hardens (straight-through).')
parser.add_argument('--gate_hidden', default=32, type=int)
# DyMo knobs
parser.add_argument('--dymo_mask_prob', default=0.5, type=float,
                    help='Per-modality random masking prob during DyMo training.')
parser.add_argument('--alpha_per_mod', default=1.0, type=float,
                    help='Weight on per-modality auxiliary CE (DyMo).')

parser.add_argument('--exp_name', default='dynmm_dymo_late_fusion', type=str)
parser.add_argument('--model_arch', default='v6', choices=['v6', 'unified_ca'])
parser.add_argument('--input_length_override', default=-1, type=int)  # -1: use dataset_cfg.sequence_length (PAMAP2 T=512); 128 was a wrong daliahar-template leftover
parser.add_argument('--ca_d_ff_mult', default=1, type=int)
parser.add_argument('--results_dir', default='./results_dynmm_dymo_pamap2/', type=str)
parser.add_argument('--dataset', default='PAMAP2', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--transform', default='sax_noisy', type=str)
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--base_factor', default=10, type=int)
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--modalities', nargs='+', default=None)

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad', default=1.0, type=float)

parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2])
parser.add_argument('--aggregate', action='store_true', default=False)

parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)
parser.add_argument('--eval_seed', default=42, type=int)

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
parser.add_argument('--fusion_add_pos_embeds', type=_str2bool, default=False)
parser.add_argument('--fusion_pos_embeds_max_len', type=int, default=256)
parser.add_argument('--fusion_pos_embed_mode', default='full',
                    choices=['full', 'decoupled', 'id_only', 'gated_id'])
parser.add_argument('--fusion_mlp_ratio', type=float, default=1.0)
parser.add_argument('--mlp_ratio', type=float, default=1.0)

args = parser.parse_args()


# ============================ Setup ============================

num_epochs = args.num_epochs
device = torch.device(args.cuda_pick if torch.cuda.is_available() else "cpu")
current_date = datetime.now().strftime('%Y-%m-%d')
exp_name_full = f"{current_date}_{args.dataset}_{args.exp_name}_{args.method}"
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)


# ============================ Aggregation mode ============================

if args.aggregate:
    fold_results = []
    for fold_idx in range(3):
        fpath = os.path.join(exp_dir, f"results_fold{fold_idx}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                fold_results.append(json.load(f))
    if not fold_results:
        print(f'No fold results found in {exp_dir}.')
        sys.exit(1)
    aggregated = {'experiment_name': exp_name_full, 'folds': fold_results}
    for key in ('best_val_acc', 'best_val_f1', 'best_test_acc', 'best_test_f1',
                'test_cost'):
        vals = [fr[key] for fr in fold_results if key in fr]
        if vals:
            aggregated[f'{key}_mean'] = float(np.mean(vals))
            aggregated[f'{key}_std'] = float(np.std(vals))
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump(aggregated, f, indent=4)
    print(f"Aggregated -> {exp_dir}/results.json")
    for key in ('best_val_acc', 'best_test_acc', 'best_test_f1', 'test_cost'):
        if f'{key}_mean' in aggregated:
            print(f"  {key}: {aggregated[f'{key}_mean']:.4f} +/- "
                  f"{aggregated[f'{key}_std']:.4f}")
    sys.exit(0)


# ============================ Data ============================

set_seed(args.seed_num)
print(f"Device: {device}")
print(f"Experiment: {exp_name_full}")
print(f"Method: {args.method.upper()}")

root_dir = '/files1/haodong/data/PAMAP2/cross_subject'   # per-subject .pt dir (unused; loader uses cfg.cross_subject_dir)
dataset_cfg = PAMAP2()
if args.modalities:
    keep = set(args.modalities)
    dataset_cfg.modalities = [m for m in dataset_cfg.modalities if m in keep]
    dataset_cfg.variates = {m: dataset_cfg.variates[m] for m in dataset_cfg.modalities}
    print(f'[subset] restricted to modalities: {dataset_cfg.modalities}')
modalities = dataset_cfg.modalities

if args.fold is not None:
    fold_cfg = dataset_cfg.folds[args.fold]
    train_subjects, eval_subjects = fold_cfg['train_set'], fold_cfg['eval_set']
    print(f'Fold {args.fold}: train={train_subjects}, eval={eval_subjects}')
else:
    train_subjects, eval_subjects = dataset_cfg.train_set, dataset_cfg.eval_set
val_subjects = dataset_cfg.val_set

input_length = dataset_cfg.sequence_length
if args.input_length_override > 0:
    input_length = args.input_length_override
    print(f"[input_length override] using input_length={input_length}")

train_ds = load_pamap2_subjects(dataset_cfg.cross_subject_dir, modalities, train_subjects, dataset_cfg, args.transform)
val_ds = load_pamap2_subjects(dataset_cfg.cross_subject_dir, modalities, val_subjects, dataset_cfg, args.transform)
eval_ds = load_pamap2_subjects(dataset_cfg.cross_subject_dir, modalities, eval_subjects, dataset_cfg, args.transform)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, drop_last=True)


# ============================ Model ============================

# Cross-modal Transformer (NO V6): 2-layer per-modality encoder + 4-layer
# cross-modal fusion transformer over [cls] + per-modality tokens.
inner = CrossModalTransformer(
    cfg=dataset_cfg, input_length=input_length, input_mode='sax',
    alphabet_size=20, token_dim=16,
    d_model=args.d_model, nhead=args.nhead,
    num_enc_layers=args.num_layers_per_modal, num_fusion_layers=args.num_layers,
    dropout=args.dropout,
)

if args.method == 'dynmm':
    branches = _parse_branches(args.dynmm_branches, modalities)
    model = CrossModalDynMM(inner, branches=branches, gate_hidden=args.gate_hidden)
    print(f'DynMM branches: {model.branches}  costs={model.costs.tolist()}')
else:
    model = CrossModalDyMo(inner)
    print(f'DyMo mask_prob={args.dymo_mask_prob} (mask-robust train + greedy select)')
model = model.to(device).float()


def unpack(batch, device):
    x, y = batch
    return x.to(device).float(), y.to(device)


optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)
criterion = (FocalLoss(gamma=args.focal_gamma)
             if args.loss == 'focal'
             else nn.CrossEntropyLoss(label_smoothing=args.label_smoothing))
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


# ============================ Training loop ============================

best_val_acc, best_val_f1, best_epoch_val = 0.0, 0.0, -1
best_model_state = None

writer = SummaryWriter(
    comment=f"_{exp_name_full}_fold{args.fold}"
            if args.fold is not None else f"_{exp_name_full}")

for epoch in range(num_epochs):
    if args.method == 'dynmm':
        temp = _anneal_temp(epoch, num_epochs, args.temp, args.end_temp)
        hard = epoch >= args.epoch_hard
        train_loss, train_acc = train_dynmm(
            train_loader, model, criterion, optimizer, epoch, device,
            temp=temp, hard=hard, loss_ratio=args.loss_ratio,
            clip_grad=args.clip_grad, unpack=unpack)
    else:
        train_loss, train_acc = train_dymo(
            train_loader, model, criterion, optimizer, epoch, device,
            mask_prob=args.dymo_mask_prob, alpha_per_mod=args.alpha_per_mod,
            clip_grad=args.clip_grad, unpack=unpack, modalities=modalities)

    val_loss, val_acc, val_y, val_p, _ = evaluate(
        val_loader, model, criterion, epoch, device, args.method, unpack,
        end_temp=args.end_temp)
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
        best_val_acc, best_val_f1, best_epoch_val = val_acc, val_f1, epoch
        best_model_state = model.state_dict()


# ============================ Model selection + test ============================

print(f"\nBest : epoch {best_epoch_val}, val_acc={best_val_acc:.4f}, val_f1={best_val_f1:.4f}")
ckpt_name = (f"best_model_fold{args.fold}.pth"
             if args.fold is not None else "best_val_model.pth")
torch.save(best_model_state, os.path.join(exp_dir, ckpt_name))

print("\n" + "=" * 80)
print(f"Test-set evaluation ({args.method})")
print("=" * 80)
model.load_state_dict(best_model_state)
model.eval()
_, test_acc, t_y, t_p, test_cost = evaluate(
    eval_loader, model, criterion, 0, device, args.method, unpack,
    end_temp=args.end_temp)
test_f1 = f1_score(t_y, t_p, average='macro')
print(f"  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  test_cost={test_cost:.3f}")

writer.add_scalar('test/acc', test_acc, 0)
writer.add_scalar('test/f1', test_f1, 0)
writer.add_scalar('test/cost', test_cost, 0)
writer.close()


# ============================ Save ============================

config = {
    'experiment_name': exp_name_full, 'dataset': args.dataset,
    'model_variant': f'{args.method}_v6', 'method': args.method,
    'model_class': 'DualVideoBottleneckModelV6Downsample',
    'fold': args.fold,
    'num_epochs': num_epochs, 'batch_size': args.batch_size,
    'seed': args.seed_num, 'transform': args.transform,
    'd_model': args.d_model, 'nhead': args.nhead,
    'num_layers': args.num_layers, 'num_layers_per_modal': args.num_layers_per_modal,
    'dropout': args.dropout, 'num_experts': args.num_experts,
    'base_factor': args.base_factor, 'internal_dim': args.internal_dim,
    'n_bottlenecks': args.n_bottlenecks, 'fusion_layer': args.fusion_layer,
    'use_sparse_attn': args.use_sparse_attn, 'sparse_attn_variant': args.sparse_attn_variant,
    'lr': args.lr, 'lr_schedule': args.lr_schedule, 'clip_grad': args.clip_grad,
    'modalities': modalities,
    'train_subjects': train_subjects, 'val_subjects': val_subjects,
    'eval_subjects': eval_subjects,
    # method-specific
    'dynmm_branches': (model.branches if args.method == 'dynmm' else None),
    'dynmm_costs': (model.costs.tolist() if args.method == 'dynmm' else None),
    'loss_ratio': args.loss_ratio, 'temp': args.temp, 'end_temp': args.end_temp,
    'epoch_hard': args.epoch_hard,
    'dymo_mask_prob': args.dymo_mask_prob, 'alpha_per_mod': args.alpha_per_mod,
}
with open(os.path.join(exp_dir, "config.json"), 'w') as f:
    json.dump(config, f, indent=4)

model_stats = {
    'experiment_name': exp_name_full, 'fold': args.fold, 'method': args.method,
    'best_val_acc': best_val_acc, 'best_val_f1': best_val_f1,
    'best_epoch_val': best_epoch_val,
    'best_test_acc': float(test_acc), 'best_test_f1': float(test_f1),
    'test_cost': float(test_cost),
}
results_filename = (f"results_fold{args.fold}.json"
                    if args.fold is not None else "results.json")
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(model_stats, f, indent=4)

print(f"\nBest Model (Epoch {best_epoch_val}): "
      f"val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
print(f"Directory: {exp_dir}")
