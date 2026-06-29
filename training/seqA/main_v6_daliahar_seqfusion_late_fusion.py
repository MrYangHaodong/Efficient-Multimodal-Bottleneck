"""V6 SEQUENTIAL-fusion training on DaliaHAR (方案 A / B-C) with the same
dynamic / grad-norm-based modality-dropout curriculum as the dyna script.

This is a copy of ``main_v6_daliahar_fusion_only_dyna_late_fusion.py`` that
imports the SEQUENTIAL-fusion backbone
(``v6_downsample_opt_batched_late_fusion_seqfusion``) and exposes three new
flags so the SAME training recipe can run all fusion variants for an
apples-to-apples comparison:

    --fusion_mode joint                       # baseline: original MBT joint fusion
    --fusion_mode sequential                  # 方案 A: per-layer gated bottleneck board,
                                              #   later modalities do NOT recur over depth
    --fusion_mode sequential --seq_depth_flow # 方案 B/C: every modality also threads its
                                              #   bottleneck across depth, blended with the
                                              #   per-layer board via a second gate
    --seq_modality_order 2 0 1                # optional: fix the modality processing order
                                              #   (positions in the active modality list)

Sequential fusion processes modalities one at a time and carries a gated
bottleneck state, so order matters and per-sample selection is disabled
(we run no_selector=True here anyway). Modality dropout still works: dropped
modalities are deleted before the model, and the sequential path handles a
dynamic number of active modalities.

Same backbone otherwise as `main_v6_daliahar_fusion_only.py` (no_selector=True,
opt + batched fusion).  The modality-drop schedule:

  - `main_v6_daliahar_fusion_only.py`     : uniform linear ramp; every
                                            modality gets the same drop
                                            probability `p(epoch)`.
  - **this script (`..._dyna.py`)**       : **staggered base ramp**
                                            (clean → linear → plateau)
                                            **scaled per-modality by the
                                            gradient norm** of that
                                            modality's `input_projectors`
                                            entry (hook-tracked).
                                            High-grad modalities get
                                            dropped more often, forcing
                                            weak modalities to learn.

Schedule (mirrors ``models.train_utils.get_modality_curriculum``):
    epoch < 20% of num_epochs            → drop_prob = 0   (clean)
    20% <= epoch < 50%                   → linear 0 → max_drop
    epoch >= 50%                         → max_drop  (plateau)

Once drop_prob > 0, every ``--profile_update_freq`` epochs we sample
recent gradient norms (via ``ModalityGradientProfiler``) and rescale
``current_mod_probs[m] = clip( base_prob * M * (norm_m / Σnorms), 0, 0.8 )``
— modalities with higher per-step gradient flow get a heavier drop
budget.  Same modality dict is used across the whole batch (the batched
fusion still requires a fixed M per forward).

Run:
    # 方案 A, fold 0
    python main_v6_daliahar_seqfusion_late_fusion.py --fusion_mode sequential --fold=0 --cuda_pick=cuda:0
    # 方案 B/C (depth flow), fold 0
    python main_v6_daliahar_seqfusion_late_fusion.py --fusion_mode sequential --seq_depth_flow --fold=0 --cuda_pick=cuda:0
    # joint baseline, fold 0
    python main_v6_daliahar_seqfusion_late_fusion.py --fusion_mode joint --fold=0 --cuda_pick=cuda:0
    # aggregate the 3 folds of a run
    python main_v6_daliahar_seqfusion_late_fusion.py --fusion_mode sequential --aggregate
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
import torch.nn.functional as F
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
    ModalityGradientProfiler,
)

from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import DualVideoBottleneckModelV6Downsample
from multimodal_model.cross_attn_unified_clf import CrossAttnUnifiedClf
from utils.dual_order_val import dual_val, eval_full_order


# ============================ V6 cfg adaptor ============================

class _DaliaV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


# ============================ Wrapper ============================

class V6FusionOnlyDynaWrapper(nn.Module):
    """Same channel-split shim as `V6FusionOnlyWrapper` but the forward
    accepts a **per-modality** drop-probability dict instead of a single
    scalar.

    For each forward at training time:
      For modality m, draw Bernoulli(drop_probs[m]).
      If True → remove m from the V6 input dict entirely (fusion never
                sees it; matches the fusion-only "actually delete"
                semantics).
      Rejection-sample to guarantee at least one survivor.

    Eval time bypasses dropout — all M modalities flow through.
    """

    def __init__(self, v6_model, modalities, variates,
                 use_singleton_heads=False, num_classes=None, d_model=None):
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
        # MRdIB Phase 1: per-modality singleton classifier heads.
        self.use_singleton_heads = use_singleton_heads
        if use_singleton_heads:
            assert num_classes is not None and d_model is not None
            self.singleton_heads = nn.ModuleDict({
                m: nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, num_classes),
                ) for m in self.modalities
            })

    def _split(self, x):
        high, low = {}, {}
        for m in self.modalities:
            s, e = self._mod_slices[m]
            high[m] = x[:, :, s:e]
            low[m] = x[:, :, s:e]
        return high, low

    def forward(self, x, modality_dropout_probs=None, training=False, return_prefix_logits=False):
        high, low = self._split(x)
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
            low = {m: low[m] for m in kept}
        out = self.model(high, low, training=training,
                         return_selection_info=False,
                         return_prefix_logits=return_prefix_logits)
        if return_prefix_logits:
            return out  # (logits, prefix_logits [B, K, C])
        logits = out[0] if isinstance(out, tuple) else out
        if self.use_singleton_heads and training:
            processed = self.model._last_processed_modalities
            singleton_logits = {
                m: self.singleton_heads[m](feat.mean(dim=1))
                for m, feat in processed.items()
            }
            return logits, singleton_logits
        return logits


# ============================ Train / eval ============================

def train_one_epoch(loader, model, criterion, optimizer, epoch, device,
                    modality_dropout_probs, clip_grad=0.0, alpha_U=0.0,
                    prefix_supervision=False, prefix_ds_weight=1.0, prefix_kd_weight=1.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.train()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device).float(), y.to(device)
        if prefix_supervision:
            logits, prefix_logits = model(x, modality_dropout_probs=modality_dropout_probs,
                                          training=True, return_prefix_logits=True)
            loss = criterion(logits, y)                       # full readout (== prefix_logits[:,-1])
            K = prefix_logits.shape[1]
            ds = sum(criterion(prefix_logits[:, j], y) for j in range(K)) / K  # per-prefix deep supervision
            teach = F.softmax(logits.detach(), dim=-1)
            kd = sum(F.kl_div(F.log_softmax(prefix_logits[:, j], dim=-1), teach, reduction='batchmean')
                     for j in range(K)) / K                   # self-distillation toward full readout
            loss = loss + prefix_ds_weight * ds + prefix_kd_weight * kd
        else:
            out = model(x, modality_dropout_probs=modality_dropout_probs,
                        training=True)
            if isinstance(out, tuple):
                logits, singleton_logits = out
            else:
                logits, singleton_logits = out, None
            loss = criterion(logits, y)
            if singleton_logits is not None and alpha_U > 0:
                loss_U = sum(criterion(sl, y) for sl in singleton_logits.values()) / len(singleton_logits)
                loss = loss + alpha_U * loss_U
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
    description='V6 fusion-only training on DaliaHAR with grad-norm-based '
                'asymmetric modality-dropout curriculum.')

parser.add_argument('--exp_name', default='v6_seqfusion_late_fusion', type=str)
parser.add_argument('--model_arch', default='v6', choices=['v6', 'unified_ca'],
                    help="'v6'=bottleneck fusion; 'unified_ca'=V6 pre-fusion + CrossAttn informer fusion")

# ---- Sequential fusion (方案 A / B-C) ----
parser.add_argument('--fusion_mode', default='sequential', type=str,
                    choices=['joint', 'sequential'],
                    help="'joint'=original MBT simultaneous fusion (baseline); "
                         "'sequential'=process modalities one at a time, carry a "
                         "gated per-layer bottleneck (方案 A / B-C). v6 arch only.")
parser.add_argument('--seq_depth_flow', action='store_true', default=False,
                    help='Sequential only. Absent=方案 A (later modalities read the '
                         'per-layer board, no cross-depth recurrence). Present=方案 B/C '
                         '(every modality also threads its bottleneck across depth, '
                         'blended with the board via a second gate).')
parser.add_argument('--seq_modality_order', nargs='+', type=int, default=None,
                    help='Sequential only. Fixed modality processing order as '
                         'positions in the active modality list (e.g. 2 0 1). '
                         'Default: available order. Use for order ablations.')
parser.add_argument('--seq_random_order', action='store_true', default=False,
                    help='Sequential only. Randomize the modality order every '
                         'TRAINING forward (order-robust / order-invariant '
                         'training). Eval stays deterministic (canonical order, '
                         'or --seq_modality_order if given). Overrides '
                         '--seq_modality_order at train time.')
parser.add_argument('--input_length_override', default=-1, type=int,
                    help='If >0, use this input_length instead of the computed one. '
                         '-1 (default) => word_length drives input_length.')
parser.add_argument('--ca_d_ff_mult', default=1, type=int)
parser.add_argument('--results_dir', default='./results_v6_seqfusion_daliahar/', type=str)
parser.add_argument('--dataset', default='DaliaHAR', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--base_factor', default=10, type=int)
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)

# Modality-drop curriculum: staggered base ramp + grad-norm scaling
parser.add_argument('--max_modality_drop', default=0.4, type=float,
                    help='Final per-modality base drop prob after the '
                         '20%%->50%% linear ramp.  Per-modality probs are '
                         'base x M x (grad_norm_m / sum grad_norms), '
                         'clipped to [0, 0.8].')
parser.add_argument('--profile_update_freq', default=5, type=int,
                    help='Refresh the per-modality dropout dict every N '
                         'epochs (uses recent grad-norm snapshot).')
parser.add_argument('--modalities', nargs='+', default=None,
                    help='Restrict to a modality subset (for subset-training).')
parser.add_argument('--dropout_signal', default='grad', type=str,
                    choices=['grad', 'importance', 'uniform'],
                    help="Drives per-modality drop probs: 'grad' (legacy), "
                         "'importance' (low-LOO dropped more -> protect strong), 'uniform'.")
parser.add_argument('--importance_csv', default='', type=str,
                    help='Comma-sep per-modality importance (in restricted-modalities order).')
parser.add_argument('--importance_floor', default=0.02, type=float)

# Optimizer / schedule
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad', default=1.0, type=float)

parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2])
parser.add_argument('--aggregate', action='store_true', default=False)
parser.add_argument('--use_singleton_heads', action='store_true', default=False,
                    help='MRdIB Phase 1: per-modality singleton classifier heads + L_U loss.')
parser.add_argument('--alpha_U', type=float, default=0.0,
                    help='Weight on per-modality singleton CE loss (L_U). 0 = no L_U.')
parser.add_argument('--prefix_supervision', action='store_true', default=False,
                    help='Per-prefix deep supervision + KL self-distillation (helps patience early-exit).')
parser.add_argument('--prefix_ds_weight', type=float, default=1.0,
                    help='Weight on per-prefix CE deep-supervision loss.')
parser.add_argument('--prefix_kd_weight', type=float, default=1.0,
                    help='Weight on per-prefix KL self-distillation toward the full readout.')

parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='AdamW weight decay (decoupled L2)')
parser.add_argument('--label_smoothing', default=0.0, type=float,
                    help='CrossEntropy label smoothing epsilon')
parser.add_argument('--eval_seed', default=42, type=int)

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
# SAX granularity: word_length=2 -> T=128 (default), word_length=1 -> T=256.
parser.add_argument('--word_length', default=2, type=int)
parser.add_argument('--use_batched_fusion', action='store_true', default=True)
parser.add_argument('--fusion_add_pos_embeds', type=_str2bool, default=False,
                    help='ViTaPEs Plan A: add per-modality learnable PE at fusion input.')
parser.add_argument('--fusion_pos_embeds_max_len', type=int, default=256,
                    help='Max sequence length for fusion-input per-mod PE.')
parser.add_argument('--fusion_pos_embed_mode', default='full',
                    choices=['full', 'decoupled', 'id_only', 'gated_id'],
                    help='PE mode when fusion_add_pos_embeds=true.')
parser.add_argument('--fusion_mlp_ratio', type=float, default=1.0)
parser.add_argument('--mlp_ratio', type=float, default=1.0)

parser.add_argument('--bottleneck_agg_mode', default='gate', choices=['gate','mean'],
                    help="bottleneck update: gate=ours (learned) / mean=vanilla-MBT average")
parser.add_argument('--n_fusion_distill', default=-1, type=int,
                    help='# fusion layers that distill; -1=all (ours), 0=none (vanilla MBT)')
parser.add_argument('--head_mode', default='gap', choices=['gap', 'entropy'],
                    help="classifier head: 'gap' = global avg pool (default); 'entropy' = per-modality entropy-weighted late fusion")
args = parser.parse_args()

if args.fusion_mode == 'sequential' and args.model_arch != 'v6':
    raise SystemExit("--fusion_mode=sequential is only supported with --model_arch=v6 "
                     "(the bottleneck fusion backbone).")
if args.seq_depth_flow and args.fusion_mode != 'sequential':
    print("[warn] --seq_depth_flow is ignored unless --fusion_mode=sequential")


# ============================ Setup ============================

num_epochs = args.num_epochs
device = torch.device(args.cuda_pick if torch.cuda.is_available() else "cpu")
current_date = datetime.now().strftime('%Y-%m-%d')
# Auto-tag the run dir by fusion variant so joint / A / B-C don't collide.
if args.fusion_mode == 'sequential':
    _variant_tag = 'seqC_depth' if args.seq_depth_flow else 'seqA'
    if args.seq_random_order:
        _variant_tag += '_randord'
else:
    _variant_tag = 'joint'
exp_name_full = f"{current_date}_{args.dataset}_{args.exp_name}_{_variant_tag}"
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)


# ============================ Aggregation mode ============================

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
            print(f"  {key}: {aggregated[f'{key}_mean']:.4f} +/- "
                  f"{aggregated[f'{key}_std']:.4f}")
    sys.exit(0)


# ============================ Data ============================

set_seed(args.seed_num)
print(f"Device: {device}")
print(f"Experiment: {exp_name_full}")

root_dir = '/files1/haodong/data/processed_dalia_activity'   # UPDATE PATH
dataset_cfg = DaliaHAR()
if args.modalities:   # subset-training: restrict cfg modalities + variates (canonical order)
    keep = set(args.modalities)
    dataset_cfg.modalities = [m for m in dataset_cfg.modalities if m in keep]
    dataset_cfg.variates = {m: dataset_cfg.variates[m] for m in dataset_cfg.modalities}
    print(f'[subset] restricted to modalities: {dataset_cfg.modalities}')
modalities = dataset_cfg.modalities
num_m = len(modalities)

# Importance-guided dropout: drop low-importance modalities more (protect strong ones),
# normalized so mean(drop_prob)=mod_base.  Drives an elastic, prune-friendly model.
_importance = None
if args.dropout_signal == 'importance':
    assert args.importance_csv, '--dropout_signal=importance requires --importance_csv'
    vals = [float(x) for x in args.importance_csv.split(',')]
    assert len(vals) == num_m, f'importance_csv has {len(vals)} vals, need {num_m}'
    _importance = {m: vals[i] for i, m in enumerate(modalities)}
    print(f'Importance-guided dropout, importance={_importance}')

def importance_dropout_probs(mod_base):
    mx = max(_importance.values())
    u = {m: (mx - _importance[m]) + args.importance_floor for m in modalities}
    su = sum(u.values())
    return {m: float(min(0.85, mod_base * num_m * u[m] / su)) for m in modalities}

if args.fold is not None:
    fold_cfg = dataset_cfg.folds[args.fold]
    train_subjects, eval_subjects = fold_cfg['train_set'], fold_cfg['eval_set']
    print(f'Fold {args.fold}: train={train_subjects}, eval={eval_subjects}')
else:
    train_subjects, eval_subjects = dataset_cfg.train_set, dataset_cfg.eval_set
val_subjects = dataset_cfg.val_set

input_length = ((dataset_cfg.duration * dataset_cfg.base_sample_rate) // args.word_length
                if args.transform == 'sax'
                else dataset_cfg.duration * dataset_cfg.base_sample_rate)
if args.input_length_override > 0:
    input_length = args.input_length_override
    print(f"[input_length override] using input_length={input_length}")

sax_params = {'alphabet_size': 20, 'word_length': args.word_length}
train_ds = HARDataset(root_dir, modalities, train_subjects, dataset_cfg, args.transform,
                      sax_params=sax_params)
val_ds = HARDataset(root_dir, modalities, val_subjects, dataset_cfg, args.transform,
                    sax_params=sax_params)
eval_ds = HARDataset(root_dir, modalities, eval_subjects, dataset_cfg, args.transform,
                     sax_params=sax_params)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, drop_last=True)


# ============================ Model ============================

v6_cfg = _DaliaV6Cfg(dataset_cfg)

_ModelClass = CrossAttnUnifiedClf if args.model_arch == 'unified_ca' else DualVideoBottleneckModelV6Downsample
if args.model_arch == 'unified_ca':
    _extra = dict(ca_d_ff_mult=args.ca_d_ff_mult, ca_nhead=args.nhead)
else:
    # Sequential-fusion knobs (v6 backbone only).
    _extra = dict(fusion_mode=args.fusion_mode,
                  seq_depth_flow=args.seq_depth_flow,
                  seq_modality_order=args.seq_modality_order,
                  seq_random_order=args.seq_random_order,
                  bottleneck_agg_mode=args.bottleneck_agg_mode,
                  n_fusion_distill=args.n_fusion_distill)
inner = _ModelClass(
    head_mode=args.head_mode,
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
    no_selector=True,                              # ← fusion-only
    use_weighted_factor=False,                     # inert under no_selector
    use_triton=False,
    num_classes=dataset_cfg.num_classes,
    sparse_attn_variant=args.sparse_attn_variant,
    strat_block_size=args.strat_block_size,
    downsample_min_len=args.downsample_min_len,
    use_batched_fusion=args.use_batched_fusion,
    per_modal_distill=args.per_modal_distill,
    per_modal_downsample_min_len=args.per_modal_downsample_min_len,
    fusion_add_pos_embeds=args.fusion_add_pos_embeds,
    fusion_pos_embeds_max_len=args.fusion_pos_embeds_max_len,
    fusion_pos_embed_mode=args.fusion_pos_embed_mode,
    fusion_mlp_ratio=args.fusion_mlp_ratio,
    **_extra,
)

model = V6FusionOnlyDynaWrapper(
    inner, modalities, dataset_cfg.variates,
    use_singleton_heads=args.use_singleton_heads,
    num_classes=dataset_cfg.num_classes,
    d_model=args.d_model,
)
model = model.to(device).float()
if args.use_singleton_heads:
    print(f"[MRdIB-Phase1] singleton heads enabled. alpha_U={args.alpha_U}")

# Gradient-norm hook profiler (one hook per input_projectors[m].weight).
profiler = ModalityGradientProfiler(inner, modalities)

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
if args.model_arch == 'v6':
    if args.fusion_mode == 'sequential':
        _variant = '方案 B/C (depth-flow)' if args.seq_depth_flow else '方案 A (per-layer board)'
        if args.seq_random_order:
            _ord = 'RANDOM per train-forward (eval=canonical)'
        else:
            _ord = args.seq_modality_order if args.seq_modality_order is not None else 'available order'
        print(f'Fusion mode:   SEQUENTIAL — {_variant}; order={_ord}')
    else:
        print(f'Fusion mode:   JOINT (baseline MBT)')
print(f'LR schedule: {args.lr_schedule} (base_lr={args.lr})')
print(f'Modality drop (staggered + grad-norm asymmetric):')
print(f'   epoch [   0,  {int(num_epochs * 0.2):>3d})  → 0')
print(f'   epoch [{int(num_epochs * 0.2):>3d}, {int(num_epochs * 0.5):>3d})  → linear 0 → {args.max_modality_drop}')
print(f'   epoch [{int(num_epochs * 0.5):>3d}, {num_epochs:>3d}]  → {args.max_modality_drop} (plateau)')
print(f'   per-modality scaling: base × M × norm_m / Σ norms (clipped to 0.8)')
print(f'Selector:      NONE (fusion-only)')


# ============================ Training loop ============================

best_val_acc, best_val_f1, best_epoch_val = 0.0, 0.0, -1
best_model_state = None
cold_best_val_acc, cold_best_val_f1, cold_best_epoch = 0.0, 0.0, -1
cold_best_model_state = None

# ---- DUAL-ORDER selection: method A = val-LOO order, method B = random-order avg ----
DUAL_K = 5
bestA = {'acc': -1.0, 'f1': 0.0, 'epoch': -1, 'state': None, 'order': None}
bestB = {'acc': -1.0, 'f1': 0.0, 'epoch': -1, 'state': None}


def _preload_val(loader):
    out = []
    for x, y in loader:
        high = {m: x[:, :, s:e].clone() for m, (s, e) in model._mod_slices.items()}
        out.append((high, dict(high), y.clone()))
    return out


val_batches = _preload_val(val_loader)
test_batches = _preload_val(eval_loader)

current_mod_probs = {m: 0.0 for m in modalities}    # initial: clean
profile_log = []    # per-epoch snapshot for debugging

mod_drop_complete_epoch = int(num_epochs * 0.5)

writer = SummaryWriter(
    comment=f"_{exp_name_full}_fold{args.fold}"
            if args.fold is not None else f"_{exp_name_full}")

for epoch in range(num_epochs):
    # ---- 1. Get the base drop probability for this epoch ----
    mod_base = get_modality_curriculum(epoch, num_epochs, args.max_modality_drop)

    # ---- 2. Apply per-modality scaling (signal = grad / importance / uniform) ----
    if mod_base == 0:
        current_mod_probs = {m: 0.0 for m in modalities}
    elif args.dropout_signal == 'importance':
        current_mod_probs = importance_dropout_probs(mod_base)
        if epoch % args.profile_update_freq == 0:
            profile_log.append({'epoch': epoch, 'mod_base': mod_base,
                                'mod_dropout_probs': {m: round(p, 4)
                                                      for m, p in current_mod_probs.items()}})
    elif args.dropout_signal == 'uniform':
        current_mod_probs = {m: mod_base for m in modalities}
    elif epoch % args.profile_update_freq == 0:   # grad (legacy)
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
    # else: keep current_mod_probs from previous profile-update epoch

    # ---- 3. Train ----
    train_loss, train_acc = train_one_epoch(
        train_loader, model, criterion, optimizer, epoch, device,
        modality_dropout_probs=current_mod_probs, clip_grad=args.clip_grad,
        alpha_U=args.alpha_U, prefix_supervision=args.prefix_supervision,
        prefix_ds_weight=args.prefix_ds_weight, prefix_kd_weight=args.prefix_kd_weight)

    val_loss, val_acc, val_y, val_p = evaluate_one_epoch(
        val_loader, model, criterion, epoch, device)
    val_f1 = f1_score(val_y, val_p, average='macro')

    current_lr = optimizer.param_groups[0]['lr']
    if scheduler is not None:
        scheduler.step()

    # ---- tensorboard ----
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/acc', train_acc, epoch)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/acc', val_acc, epoch)
    writer.add_scalar('val/f1', val_f1, epoch)
    writer.add_scalar('schedule/mod_base', mod_base, epoch)
    writer.add_scalar('schedule/lr', current_lr, epoch)
    # Per-modality drop probability (current dict)
    for m, p in current_mod_probs.items():
        writer.add_scalar(f'mod_drop/{m}', p, epoch)
    # Recent grad-norm snapshot per modality
    if any(len(profiler.grad_norms[m]) > 0 for m in modalities):
        for m, gn in profiler.get_norm_snapshot().items():
            writer.add_scalar(f'grad_norm/{m}', gn, epoch)

    # ---- DUAL-ORDER val (warm phase only, to bound cost): A=val-LOO order, B=rand-avg ----
    if epoch >= mod_drop_complete_epoch:
        dv = dual_val(model.model, val_batches, modalities, device, K=DUAL_K, seed=1234 + epoch)
        writer.add_scalar('valA_loo/acc', dv['accA'], epoch)
        writer.add_scalar('valB_rand/acc', dv['accB'], epoch)
        print(f"  [dual] A(val-LOO order) acc={dv['accA']:.4f} f1={dv['f1A']:.4f} order={dv['orderA']}"
              f"  |  B(rand-avg/{DUAL_K}) acc={dv['accB']:.4f} f1={dv['f1B']:.4f}")
        if dv['accA'] > bestA['acc']:
            bestA = {'acc': dv['accA'], 'f1': dv['f1A'], 'epoch': epoch,
                     'state': model.state_dict(), 'order': dv['orderA']}
        if dv['accB'] > bestB['acc']:
            bestB = {'acc': dv['accB'], 'f1': dv['f1B'], 'epoch': epoch, 'state': model.state_dict()}

    # Cold best — anytime (canonical-order val; fallback if no warm dual checkpoint)
    if val_acc > cold_best_val_acc:
        cold_best_val_acc = val_acc
        cold_best_val_f1 = val_f1
        cold_best_epoch = epoch
        cold_best_model_state = model.state_dict()


# ============================ Dual model selection + save ============================
# Method A = best by val-LOO-order val acc; Method B = best by random-order-avg val acc.
# Fallback to the cold (canonical) checkpoint if a method never selected (shouldn't happen).
from utils.dual_order_val import rand_avg_eval

if bestA['state'] is None:
    bestA = {'acc': cold_best_val_acc, 'f1': cold_best_val_f1, 'epoch': cold_best_epoch,
             'state': cold_best_model_state, 'order': list(modalities)}
if bestB['state'] is None:
    bestB = {'acc': cold_best_val_acc, 'f1': cold_best_val_f1, 'epoch': cold_best_epoch,
             'state': cold_best_model_state}

sfx = f"fold{args.fold}" if args.fold is not None else "val"
torch.save(bestA['state'], os.path.join(exp_dir, f"best_model_loo_{sfx}.pth"))   # method A
torch.save(bestB['state'], os.path.join(exp_dir, f"best_model_rand_{sfx}.pth"))  # method B
print(f"\n[method A val-LOO order]  best epoch {bestA['epoch']}  val_acc={bestA['acc']:.4f} f1={bestA['f1']:.4f}  order={bestA['order']}")
print(f"[method B random-avg]     best epoch {bestB['epoch']}  val_acc={bestB['acc']:.4f} f1={bestB['f1']:.4f}")


# ============================ Test eval for both checkpoints ============================
print("\n" + "=" * 80 + "\nTest-set evaluation (both methods)\n" + "=" * 80)
model.load_state_dict(bestA['state']); model.eval()
testA_acc, testA_f1 = eval_full_order(model.model, test_batches, modalities, device, order_names=bestA['order'])
print(f"  [A val-LOO order] test_acc={testA_acc:.4f} test_f1={testA_f1:.4f}")
model.load_state_dict(bestB['state']); model.eval()
testB_acc, testB_f1 = rand_avg_eval(model.model, test_batches, modalities, device, K=DUAL_K, seed=4321)
print(f"  [B random-avg]    test_acc={testB_acc:.4f} test_f1={testB_f1:.4f}")

# back-compat aliases (method A is the primary checkpoint)
best_val_acc, best_val_f1, best_epoch_val = bestA['acc'], bestA['f1'], bestA['epoch']
selected_source = 'dual(A=val-LOO,B=rand)'
test_acc, test_f1 = testA_acc, testA_f1
writer.add_scalar('testA/acc', testA_acc, 0); writer.add_scalar('testB/acc', testB_acc, 0)
writer.close()


# ============================ Save ============================

# Final grad-norm snapshot for inspection.
final_grad_norms = profiler.get_norm_snapshot() \
    if any(len(profiler.grad_norms[m]) > 0 for m in modalities) else {}

config = {
    'experiment_name': exp_name_full,
    'dataset': args.dataset,
    'model_variant': 'v6_seqfusion',
    'model_class': 'DualVideoBottleneckModelV6Downsample (seqfusion)',
    'head_mode': args.head_mode,
    'fusion_mode': args.fusion_mode,
    'seq_depth_flow': bool(args.seq_depth_flow),
    'seq_modality_order': args.seq_modality_order,
    'seq_random_order': bool(args.seq_random_order),
    'selector': 'NONE',
    'modality_drop_strategy': 'staggered_linear_base + gradient_asymmetric_scaling',
    'fold': args.fold,
    'num_epochs': num_epochs, 'batch_size': args.batch_size,
    'seed': args.seed_num, 'eval_seed': args.eval_seed,
    'transform': args.transform,
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
    'train_subjects': train_subjects, 'val_subjects': val_subjects,
    'eval_subjects': eval_subjects,
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
    'dual_order': {
        'A_valLOO': {'val_acc': bestA['acc'], 'val_f1': bestA['f1'], 'epoch': bestA['epoch'],
                     'order': bestA['order'], 'test_acc': float(testA_acc), 'test_f1': float(testA_f1),
                     'ckpt': f"best_model_loo_{sfx}.pth"},
        'B_randavg': {'val_acc': bestB['acc'], 'val_f1': bestB['f1'], 'epoch': bestB['epoch'],
                      'K': DUAL_K, 'test_acc': float(testB_acc), 'test_f1': float(testB_f1),
                      'ckpt': f"best_model_rand_{sfx}.pth"},
    },
    'final_grad_norms': final_grad_norms,
    'profile_log': profile_log,
}
results_filename = (f"results_fold{args.fold}.json"
                    if args.fold is not None else "results.json")
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(model_stats, f, indent=4)

# Cleanup gradient hooks.
profiler.remove_hooks()

print(f"\nBest Model (Epoch {best_epoch_val}): "
      f"val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
print(f"Selected: {selected_source}")
print(f"Directory: {exp_dir}")
