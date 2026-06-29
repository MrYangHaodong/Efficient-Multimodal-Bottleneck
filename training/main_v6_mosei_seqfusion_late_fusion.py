"""V6 fusion-only training on MOSEI with **dynamic / grad-norm-based**
modality-dropout curriculum.

MOSEI sibling of ``main_v6_dsads_fusion_only_dyna.py`` /
``main_v6_daliahar_fusion_only_dyna.py``.  Same backbone
(``DualVideoBottleneckModelV6Downsample`` with ``no_selector=True``)
and staggered curriculum:

    epoch < 20% of num_epochs            -> drop_prob = 0   (clean)
    20% <= epoch < 50%                   -> linear 0 -> max_drop
    epoch >= 50%                         -> max_drop  (plateau)

Once drop_prob > 0, every ``--profile_update_freq`` epochs we sample
recent gradient norms (via ``ModalityGradientProfiler``) and rescale
``current_mod_probs[m] = clip( base * M * (norm_m / sum(norms)), 0, 0.8 )``.

Differences vs DSADS / DaliaHAR:
  - **6 modalities** (video / audio / text / mocap_hand/head/rotated)
    with heterogeneous feature dims (18..1280).
  - ``CMUMOSEIDataset`` returns a list of (high, low) tensor pairs +
    label rather than a single concatenated ``(B, T, C)`` tensor.  The
    wrapper unpacks it into V6's two dicts directly.
  - Pre-extracted features (DINOv3 / WAV2VEC2 / BERT / MOCAP) — no SAX
    token transform.
  - 3 stratified 70/15/15 splits, ``--fold = 0/1/2``.
  - 4-class emotion classification.

Run:
    python main_v6_mosei_fusion_only_dyna.py --fold=0 --cuda_pick=cuda:0
    python main_v6_mosei_fusion_only_dyna.py --aggregate
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import copy
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from data.CMU_MOSEI.get_data import (
    get_dataloader as mosei_get_dataloader,
    FEATURE_DIMS, NUM_CLASSES, CMUMOSEIDataset,
)
from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import (
    DualVideoBottleneckModelV6Downsample,
)
from multimodal_model.cross_attn_unified_clf import CrossAttnUnifiedClf
from utils.dual_order_val import dual_val, eval_full_order, rand_avg_eval


# ============================ V6 cfg adaptor ============================


class _MOSEIV6Cfg:
    """Minimal cfg shim that exposes the fields V6 reads from its
    ``cfg`` arg: ``modalities``, ``variates``, ``video_high_dim``,
    ``video_low_dim``."""

    def __init__(self, modalities=None):
        if modalities is None:
            modalities = list(CMUMOSEIDataset.ALL_MODALITIES)
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        # V6 treats one modality named 'video' specially (uses two
        # separate dims for selector vs encoder paths).  For MOSEI
        # both paths use the same DINOv3 1024-d features.
        v_dim = FEATURE_DIMS.get('video', 1024)
        self.video_high_dim = v_dim
        self.video_low_dim = v_dim
        self.num_classes = NUM_CLASSES


# ============================ Wrapper ============================


import math
from torch.autograd import Function


class _GRL(Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, g):
        return -g


def grad_reverse(x):
    return _GRL.apply(x)


class V6MOSEIFusionDynaWrapper(nn.Module):
    """Unpack the CMUMOSEIDataset batch (12 modality tensors + label)
    into V6's ``(high_dim_inputs, low_dim_inputs)`` dicts and apply
    per-modality independent Bernoulli dropout during training.

    Modalities are unpacked in canonical order
    (``video, audio, text, mocap_hand, mocap_head, mocap_rotated``).
    Modality dropout removes the entry from both dicts; if all are
    dropped we keep one at random as a safety net.
    """

    def __init__(self, v6_model, modalities,
                 use_singleton_heads=False, num_classes=None, d_model=None,
                 use_R_loss=False, mine_hidden=128):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self._mod_order = list(modalities)
        # MRdIB phase 1: per-modality singleton classifier heads.
        self.use_singleton_heads = use_singleton_heads
        if use_singleton_heads:
            assert num_classes is not None and d_model is not None
            self.singleton_heads = nn.ModuleDict({
                m: nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, num_classes),
                ) for m in self.modalities
            })
        # MRdIB phase 2: vCLUB critics for pairwise redundancy minimization.
        # q(z_j | z_i) parameterized as Gaussian with learned mu and log_var.
        self.use_R_loss = use_R_loss
        if use_R_loss:
            assert d_model is not None
            M = len(self.modalities)
            class _QCritic(nn.Module):
                def __init__(self, d_in, d_out, hidden):
                    super().__init__()
                    self.trunk = nn.Sequential(
                        nn.Linear(d_in, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                    )
                    self.mu = nn.Linear(hidden, d_out)
                    self.logvar = nn.Linear(hidden, d_out)
                def forward(self, x):
                    h = self.trunk(x)
                    return self.mu(h), self.logvar(h)
            self.q_critics = nn.ModuleDict({
                f"{i}_{j}": _QCritic(d_model, d_model, mine_hidden)
                for i in range(M) for j in range(i+1, M)
            })

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

    def forward(self, feats, modality_dropout_probs=None, training=False, return_prefix_logits=False):
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
                         return_selection_info=False,
                         return_prefix_logits=return_prefix_logits)
        if return_prefix_logits:
            return out  # (main_logits, prefix_logits [B, K, C])
        main_logits = out[0] if isinstance(out, tuple) else out
        if training and (self.use_singleton_heads or self.use_R_loss):
            processed = self.model._last_processed_modalities
            z_per = {m: feat.mean(dim=1) for m, feat in processed.items()}  # {m: [B, H]}
            singleton_logits = None
            mi_estimates = None
            if self.use_singleton_heads:
                singleton_logits = {
                    m: self.singleton_heads[m](z_per[m]) for m in z_per
                }
            if self.use_R_loss:
                # vCLUB: returns (mle_term, vclub_term) per pair.
                mods_present = [m for m in self.modalities if m in z_per]
                idx_of = {m: i for i, m in enumerate(self.modalities)}
                mi_estimates = []
                for a, ma in enumerate(mods_present):
                    for mb in mods_present[a+1:]:
                        i, j = idx_of[ma], idx_of[mb]
                        key = f"{min(i,j)}_{max(i,j)}"
                        critic = self.q_critics[key]
                        z_i, z_j = z_per[ma], z_per[mb]
                        B, d = z_i.shape
                        perm = torch.randperm(B, device=z_i.device)
                        # MLE term: q learns p(z_j | z_i), detach so encoder doesn't see this gradient.
                        mu_d, lv_d = critic(z_i.detach())
                        # log Normal per-dim averaged: -0.5 * ((x-mu)^2/exp(lv) + lv + log(2*pi))
                        log_q_mle = (-0.5 * ((z_j.detach() - mu_d)**2 / torch.exp(lv_d) + lv_d
                                              + math.log(2*math.pi))).mean()
                        # vCLUB term: with z having grad. q params also see this gradient.
                        mu, lv = critic(z_i)
                        log_q_paired = (-0.5 * ((z_j - mu)**2 / torch.exp(lv) + lv
                                                + math.log(2*math.pi))).mean()
                        log_q_marg = (-0.5 * ((z_j[perm] - mu)**2 / torch.exp(lv) + lv
                                              + math.log(2*math.pi))).mean()
                        vclub = log_q_paired - log_q_marg
                        mi_estimates.append(torch.stack([log_q_mle, vclub]))
                mi_estimates = torch.stack(mi_estimates) if mi_estimates else None  # [n_pairs, 2]
            return main_logits, singleton_logits, mi_estimates
        return main_logits


# ============================ Train / eval ============================


def _unpack_batch(batch, device):
    """CMUMOSEIDataset collate returns
        (feat_0, feat_1, ..., feat_{2M-1}, label[B,1])
    """
    *feats, label = batch
    feats = [f.to(device).float() for f in feats]
    label = label.squeeze(-1).long().to(device)
    return feats, label


def train_one_epoch(loader, model, criterion, optimizer, epoch, device,
                    modality_dropout_probs, clip_grad=0.0, alpha_U=0.0, beta_R=0.0,
                    prefix_supervision=False, prefix_ds_weight=1.0, prefix_kd_weight=1.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    lU_meter = AverageMeter('L_U', ':.4f')
    lR_meter = AverageMeter('L_R', ':.4f')
    model.train()
    for i, batch in enumerate(loader):
        feats, y = _unpack_batch(batch, device)
        if prefix_supervision:
            logits, prefix_logits = model(feats, modality_dropout_probs=modality_dropout_probs,
                                          training=True, return_prefix_logits=True)
            loss = criterion(logits, y)
            K = prefix_logits.shape[1]
            ds = sum(criterion(prefix_logits[:, j], y) for j in range(K)) / K
            teach = F.softmax(logits.detach(), dim=-1)
            kd = sum(F.kl_div(F.log_softmax(prefix_logits[:, j], dim=-1), teach, reduction='batchmean')
                     for j in range(K)) / K
            loss = loss + prefix_ds_weight * ds + prefix_kd_weight * kd
        else:
            out = model(feats, modality_dropout_probs=modality_dropout_probs,
                        training=True)
            if isinstance(out, tuple):
                logits, singleton_logits, mine_estimates = out
            else:
                logits, singleton_logits, mine_estimates = out, None, None
            loss = criterion(logits, y)
            if singleton_logits is not None and alpha_U > 0:
                loss_U = sum(criterion(sl, y) for sl in singleton_logits.values()) / len(singleton_logits)
                loss = loss + alpha_U * loss_U
                lU_meter.update(loss_U.item(), y.size(0))
            if mine_estimates is not None and beta_R > 0:
                # vCLUB: mine_estimates is [n_pairs, 2] = [log_q_mle, vclub] per pair.
                log_q_mle = mine_estimates[:, 0].mean()    # q's MLE; we MAXIMIZE → -log_q_mle in loss
                vclub = mine_estimates[:, 1].mean()         # upper bound on I(z_i; z_j); we MINIMIZE
                # Combined: q trains via MLE (small fixed lambda 1.0), encoder minimizes vclub * beta_R.
                loss = loss + (-log_q_mle) + beta_R * vclub
                lR_meter.update(vclub.item(), y.size(0))
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
    description='V6 fusion-only training on MOSEI with grad-norm '
                'asymmetric modality-dropout curriculum (6 modalities).')

parser.add_argument('--exp_name', default='v6_seqfusion_late_fusion_cs', type=str)
parser.add_argument('--model_arch', default='v6', choices=['v6', 'unified_ca'],
                    help="'v6'=bottleneck fusion; 'unified_ca'=V6 pre-fusion + CrossAttn informer fusion")

# ---- Sequential fusion (方案 A / B-C) ----
parser.add_argument('--fusion_mode', default='sequential', type=str,
                    choices=['joint', 'sequential'],
                    help="'joint'=original MBT simultaneous fusion (baseline); "
                         "'sequential'=process modalities one at a time with a "
                         "gated per-layer bottleneck (方案 A / B-C). v6 arch only.")
parser.add_argument('--seq_depth_flow', action='store_true', default=False,
                    help='Sequential only. Absent=方案 A (per-layer board, no cross-depth '
                         'recurrence). Present=方案 B/C (every modality also threads its '
                         'bottleneck across depth, blended with the board via a 2nd gate).')
parser.add_argument('--seq_modality_order', nargs='+', type=int, default=None,
                    help='Sequential only. Fixed processing order as positions in the '
                         'active modality list (e.g. 5 0 1 2 3 4). Default: available order.')
parser.add_argument('--seq_random_order', action='store_true', default=False,
                    help='Sequential only. Randomize modality order every TRAINING '
                         'forward (order-robust). Eval stays deterministic.')

parser.add_argument('--input_length_override', default=-1, type=int)
parser.add_argument('--ca_d_ff_mult', default=1, type=int)
parser.add_argument('--results_dir',
                    default='./results_v6_seqfusion_mosei/', type=str)
parser.add_argument('--dataset', default='CMU_MOSEI', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# Modalities (subset selectable)
parser.add_argument('--modalities', nargs='+',
                    default=list(CMUMOSEIDataset.ALL_MODALITIES),
                    choices=list(CMUMOSEIDataset.ALL_MODALITIES))

# Sequence length / dual-resolution
parser.add_argument('--max_seq_len', default=50, type=int,
                    help='Truncate / pad every modality to this T '
                         '(high-resolution path).  0 = no resampling.')
parser.add_argument('--time_compression_ratio', default=4, type=int,
                    help='Low-resolution T = high T // this.')

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
parser.add_argument('--use_sparse_attn', type=_str2bool, default=False)
parser.add_argument('--sparse_attn_variant', default='opt', type=str,
                    choices=['opt', 'opt_strat', 'opt_strat_blk'])
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=64, type=int)
parser.add_argument('--n_fusion_distill', default=-1, type=int,
                    help='# of fusion layers (counted from the LAST) that distill '
                         '(stride-2). -1=all (default); 0=no fusion downsampling '
                         '(full-resolution attention). The real distill knob; '
                         'downsample_min_len is inert in the batched fusion path.')
parser.add_argument('--per_modal_distill', action='store_true', default=False)
parser.add_argument('--per_modal_downsample_min_len', default=4, type=int)
parser.add_argument('--bottleneck_agg_mode', default='gate', type=str,
                    choices=['gate', 'mean'],
                    help="bottleneck aggregation across modalities: 'gate'=ours "
                         "(gate-weighted sum + learned upsample); 'mean'=vanilla MBT "
                         "(uniform average).")
parser.add_argument('--bottleneck_init_mode', default='random', type=str,
                    choices=['random', 'modal_mean_sinpe'])
parser.add_argument('--use_batched_fusion', type=_str2bool, default=True)
parser.add_argument('--fusion_add_pos_embeds', type=_str2bool, default=False,
                    help='ViTaPEs Plan A: add per-modality learnable PE at fusion input.')
parser.add_argument('--fusion_pos_embeds_max_len', type=int, default=256,
                    help='Max sequence length for fusion-input per-mod PE.')
parser.add_argument('--fusion_pos_embed_mode', default='full',
                    choices=['full', 'decoupled', 'id_only', 'gated_id'],
                    help='PE mode when fusion_add_pos_embeds=true. '
                         '"full" = per-mod [L,H] PE (98k params). '
                         '"decoupled" = shared temporal [L,H] + mod_id [M,H] (17k params). '
                         '"id_only" = mod_id [M,H] only (768 params). '
                         '"gated_id" = id_only × sigmoid(scalar gate), data-driven activation (769 params).')
parser.add_argument('--fusion_mlp_ratio', type=float, default=1.0,
                    help='V6 fusion FFN hidden / d_model ratio (default 2.0).')
parser.add_argument('--mlp_ratio', type=float, default=1.0,
                    help='V6 FFN hidden / d_model ratio (default 2.0).')

# Modality-drop curriculum
parser.add_argument('--max_modality_drop', default=0.4, type=float,
                    help='Final per-modality base drop prob after the '
                         '20%%->50%% linear ramp.  Per-modality probs are '
                         '`base * M * (grad_norm_m / sum grad_norms)`, '
                         'clipped to [0, 0.8].')
parser.add_argument('--profile_update_freq', default=5, type=int)
parser.add_argument('--dropout_signal', default='grad', type=str,
                    choices=['grad', 'importance', 'uniform'],
                    help="What drives per-modality drop probs: 'grad' (legacy, "
                         "high-grad dropped more), 'importance' (low-LOO/Shapley "
                         "dropped more -> protect strong modalities), 'uniform'.")
parser.add_argument('--importance_csv', default='', type=str,
                    help='Comma-sep per-modality importance (in --modalities order) '
                         'for --dropout_signal=importance. Low importance -> dropped more.')
parser.add_argument('--importance_floor', default=0.02, type=float,
                    help='Floor added to inverse-importance so even the top modality '
                         'drops occasionally (keeps the model elastic).')

# Optimizer / schedule
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine', 'cosine_warmrestart'])
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--loss', default='ce', choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='AdamW weight decay (decoupled L2)')
parser.add_argument('--label_smoothing', default=0.0, type=float,
                    help='CrossEntropy label smoothing epsilon')

# 3 random splits (split 0 / 1 / 2)
parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2],
                    help='MOSEI split id (0..2).')
parser.add_argument('--split_mode', default='random', type=str,
                    choices=['random', 'cross_subject'],
                    help='random: utterance-level splits (default). cross_subject: session-disjoint splits.')
parser.add_argument('--share_fusion_params', action='store_true', default=False,
                    help='Share fusion transformer params across modalities (single modality-agnostic fusion block) instead of per-modality param sets.')
parser.add_argument('--use_singleton_heads', action='store_true', default=False,
                    help='MRdIB phase 1: add per-modality singleton classifier heads + L_U loss.')
parser.add_argument('--alpha_U', type=float, default=0.0,
                    help='Weight on per-modality singleton CE loss (L_U). 0 = no L_U.')
parser.add_argument('--use_R_loss', action='store_true', default=False,
                    help='MRdIB phase 2: MINE-based redundancy minimization between z_i, z_j.')
parser.add_argument('--beta_R', type=float, default=0.0,
                    help='Weight on MINE-based pairwise MI loss (L_R). 0 = no L_R.')
parser.add_argument('--prefix_supervision', action='store_true', default=False,
                    help='Per-prefix deep supervision + KL self-distillation (helps patience early-exit).')
parser.add_argument('--prefix_ds_weight', type=float, default=1.0)
parser.add_argument('--prefix_kd_weight', type=float, default=1.0)
parser.add_argument('--aggregate', action='store_true', default=False)

# Data path
parser.add_argument('--data_root', default='/files1/haodong/data/CMU-MOSI/CMU-MOSEI',
                    type=str)
parser.add_argument('--available_sessions', nargs='+', type=int, default=None,
                    help='Restrict to these MOSEI sessions (1..5).  '
                         'Default: use all sessions present in the CSV.  '
                         'Useful when some Session{N}_features.tar.gz '
                         'archives are still being extracted.')

parser.add_argument('--head_mode', default='gap', choices=['gap', 'entropy'],
                    help="classifier head: 'gap' = global avg pool (default); 'entropy' = per-modality entropy-weighted late fusion")
parser.add_argument('--mosei_task', default=2, type=int, choices=[2, 5, 7],
                    help='MOSEI classification granularity: 2=Acc-2 (sign), 5=Acc-5, 7=Acc-7.')

args = parser.parse_args()
# CMU-MOSEI classification granularity: rebind NUM_CLASSES (all runtime
# usages - cfg instantiation, model build, config dict - run below this).
NUM_CLASSES = args.mosei_task
_MOSEI_TASK = {2: 'classification2', 5: 'classification5', 7: 'classification7'}[args.mosei_task]


if args.fusion_mode == 'sequential' and args.model_arch != 'v6':
    raise SystemExit("--fusion_mode=sequential is only supported with --model_arch=v6.")
if args.seq_depth_flow and args.fusion_mode != 'sequential':
    print("[warn] --seq_depth_flow ignored unless --fusion_mode=sequential")


# ============================ Setup ============================


num_epochs = args.num_epochs
device = torch.device(args.cuda_pick if torch.cuda.is_available() else "cpu")
current_date = datetime.now().strftime('%Y-%m-%d')
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

# Importance-guided dropout: inverse-importance per-modality drop probs (drop low-LOO
# modalities more, protect strong ones), normalized so mean(drop_prob)=mod_base.
_importance = None
if args.dropout_signal == 'importance':
    if args.importance_csv:
        vals = [float(x) for x in args.importance_csv.split(',')]
        assert len(vals) == num_m, f'importance_csv has {len(vals)} vals, need {num_m}'
        _importance = {m: vals[i] for i, m in enumerate(modalities)}
    else:
        raise SystemExit('--dropout_signal=importance requires --importance_csv')
    print(f'Importance-guided dropout, importance={_importance}')

def importance_dropout_probs(mod_base):
    imp = _importance
    mx = max(imp.values())
    u = {m: (mx - imp[m]) + args.importance_floor for m in modalities}  # low imp -> high u
    su = sum(u.values())
    return {m: float(min(0.85, mod_base * num_m * u[m] / su)) for m in modalities}

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

_ModelClass = CrossAttnUnifiedClf if args.model_arch == 'unified_ca' else DualVideoBottleneckModelV6Downsample
if args.model_arch == 'unified_ca':
    _extra = dict(ca_d_ff_mult=args.ca_d_ff_mult, ca_nhead=args.nhead)
else:
    _extra = dict(fusion_mode=args.fusion_mode,
                  seq_depth_flow=args.seq_depth_flow,
                  seq_modality_order=args.seq_modality_order,
                  seq_random_order=args.seq_random_order)
inner = _ModelClass(
    head_mode=args.head_mode,
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
    n_fusion_distill=args.n_fusion_distill,
    bottleneck_agg_mode=args.bottleneck_agg_mode,
    use_batched_fusion=args.use_batched_fusion,
    per_modal_distill=args.per_modal_distill,
    per_modal_downsample_min_len=args.per_modal_downsample_min_len,
    bottleneck_init_mode=args.bottleneck_init_mode,
    fusion_add_pos_embeds=args.fusion_add_pos_embeds,
    fusion_pos_embeds_max_len=args.fusion_pos_embeds_max_len,
    fusion_pos_embed_mode=args.fusion_pos_embed_mode,
    fusion_mlp_ratio=args.fusion_mlp_ratio,
    share_fusion_params=args.share_fusion_params,
    **_extra,
)
if args.share_fusion_params:
    print("[SHARED-FUSION] fusion transformer params shared across modalities")

model = V6MOSEIFusionDynaWrapper(
    inner, modalities,
    use_singleton_heads=args.use_singleton_heads,
    num_classes=NUM_CLASSES,
    d_model=args.d_model,
    use_R_loss=args.use_R_loss,
)
model = model.to(device).float()
if args.use_singleton_heads:
    print(f"[MRdIB-Phase1] singleton heads enabled. alpha_U={args.alpha_U}")
if args.use_R_loss:
    print(f"[MRdIB-Phase2] MINE critics enabled. beta_R={args.beta_R}")

profiler = ModalityGradientProfiler(inner, modalities)

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
if args.model_arch == 'v6':
    if args.fusion_mode == 'sequential':
        _v = '方案 B/C (depth-flow)' if args.seq_depth_flow else '方案 A (per-layer board)'
        _o = 'RANDOM per train-forward' if args.seq_random_order else (
            args.seq_modality_order if args.seq_modality_order is not None else 'available order')
        print(f'Fusion mode:   SEQUENTIAL — {_v}; order={_o}')
    else:
        print('Fusion mode:   JOINT (baseline MBT)')
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

# ---- DUAL-ORDER selection: method A = val-LOO order, method B = random-order avg ----
DUAL_K = 5
bestA = {'acc': -1.0, 'f1': 0.0, 'epoch': -1, 'state': None, 'order': None}
bestB = {'acc': -1.0, 'f1': 0.0, 'epoch': -1, 'state': None}


def _preload_val(loader):
    out = []
    for batch in loader:
        *feats, label = batch
        high = {m: feats[2 * j].clone() for j, m in enumerate(modalities)}
        low = {m: feats[2 * j + 1].clone() for j, m in enumerate(modalities)}
        out.append((high, low, label.squeeze(-1).long().clone()))
    return out


val_batches = _preload_val(val_loader)
test_batches = _preload_val(eval_loader)

current_mod_probs = {m: 0.0 for m in modalities}
profile_log = []

mod_drop_complete_epoch = int(num_epochs * 0.5)

writer = SummaryWriter(
    comment=f"_{exp_name_full}_fold{args.fold}"
            if args.fold is not None else f"_{exp_name_full}")

for epoch in range(num_epochs):
    mod_base = get_modality_curriculum(epoch, num_epochs, args.max_modality_drop)

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

    train_loss, train_acc = train_one_epoch(
        train_loader, model, criterion, optimizer, epoch, device,
        modality_dropout_probs=current_mod_probs, clip_grad=args.clip_grad,
        alpha_U=args.alpha_U, beta_R=args.beta_R,
        prefix_supervision=args.prefix_supervision,
        prefix_ds_weight=args.prefix_ds_weight, prefix_kd_weight=args.prefix_kd_weight)

    val_loss, val_acc, val_y, val_p = evaluate_one_epoch(
        val_loader, model, criterion, epoch, device)
    val_f1 = f1_score(val_y, val_p, average='macro')

    if os.environ.get('DIAG_TEST') == '1':
        _dt_acc, _dt_f1 = eval_full_order(model.model, test_batches, modalities,
                                          device, order_names=list(modalities))
        print(f"DIAG ep{epoch} val={val_acc:.4f} test={_dt_acc:.4f} "
              f"gap={_dt_acc - val_acc:+.4f}")

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

    if epoch >= mod_drop_complete_epoch and len(modalities) >= 2:
        # dual-order val needs >=2 modalities (LOO of the only modality => empty
        # input set => model forward StopIteration). Single-modality experts have
        # no order; they fall back to the cold (global-best-val) checkpoint.
        dv = dual_val(model.model, val_batches, modalities, device, K=DUAL_K, seed=1234 + epoch)
        writer.add_scalar('valA_loo/acc', dv['accA'], epoch)
        writer.add_scalar('valB_rand/acc', dv['accB'], epoch)
        print(f"  [dual] A(val-LOO order) acc={dv['accA']:.4f} f1={dv['f1A']:.4f} order={dv['orderA']}"
              f"  |  B(rand-avg/{DUAL_K}) acc={dv['accB']:.4f} f1={dv['f1B']:.4f}")
        if dv['accA'] > bestA['acc']:
            bestA = {'acc': dv['accA'], 'f1': dv['f1A'], 'epoch': epoch,
                     'state': copy.deepcopy(model.state_dict()), 'order': dv['orderA']}
        if dv['accB'] > bestB['acc']:
            bestB = {'acc': dv['accB'], 'f1': dv['f1B'], 'epoch': epoch,
                     'state': copy.deepcopy(model.state_dict())}

    if val_acc > cold_best_val_acc:
        cold_best_val_acc = val_acc
        cold_best_val_f1 = val_f1
        cold_best_epoch = epoch
        # deepcopy: state_dict() returns live references; without a copy the saved
        # "best" checkpoint silently becomes the FINAL-epoch weights (overfit).
        cold_best_model_state = copy.deepcopy(model.state_dict())


# ============================ Model selection ============================


from utils.dual_order_val import rand_avg_eval as _rae

if bestA['state'] is None:
    bestA = {'acc': cold_best_val_acc, 'f1': cold_best_val_f1, 'epoch': cold_best_epoch,
             'state': cold_best_model_state, 'order': list(modalities)}
if bestB['state'] is None:
    bestB = {'acc': cold_best_val_acc, 'f1': cold_best_val_f1, 'epoch': cold_best_epoch,
             'state': cold_best_model_state}

sfx = f"fold{args.fold}" if args.fold is not None else "val"
torch.save(bestA['state'], os.path.join(exp_dir, f"best_model_loo_{sfx}.pth"))
torch.save(bestB['state'], os.path.join(exp_dir, f"best_model_rand_{sfx}.pth"))
print(f"\n[method A val-LOO order]  best epoch {bestA['epoch']}  val_acc={bestA['acc']:.4f} f1={bestA['f1']:.4f}  order={bestA['order']}")
print(f"[method B random-avg]     best epoch {bestB['epoch']}  val_acc={bestB['acc']:.4f} f1={bestB['f1']:.4f}")

print("\n" + "=" * 80 + "\nTest-set evaluation (both methods)\n" + "=" * 80)
model.load_state_dict(bestA['state']); model.eval()
testA_acc, testA_f1 = eval_full_order(model.model, test_batches, modalities, device, order_names=bestA['order'])
print(f"  [A val-LOO order] test_acc={testA_acc:.4f} test_f1={testA_f1:.4f}")
model.load_state_dict(bestB['state']); model.eval()
testB_acc, testB_f1 = _rae(model.model, test_batches, modalities, device, K=DUAL_K, seed=4321)
print(f"  [B random-avg]    test_acc={testB_acc:.4f} test_f1={testB_f1:.4f}")

# --- Cold / global-best-val checkpoint (parity with the other baselines).
# Every other dyna baseline reports its best STANDARD-val checkpoint, which the
# modality-drop curriculum pushes into an early (cold/pre-dropout) epoch -- they
# all end up selecting 'cold'. seqA's dual-order pick (A/B) only scans warm
# epochs (>= 50%) and ignores the higher cold val, unfairly penalising seqA.
# ``cold_best_model_state`` already holds the global-best-standard-val checkpoint
# (line ~804, no epoch guard); evaluate it on test with the canonical order and
# report it as the primary result so seqA is compared on the same footing.
model.load_state_dict(cold_best_model_state); model.eval()
testC_acc, testC_f1 = eval_full_order(model.model, test_batches, modalities, device,
                                      order_names=list(modalities))
print(f"  [cold/global-best-val, canonical order] val={cold_best_val_acc:.4f} "
      f"test_acc={testC_acc:.4f} test_f1={testC_f1:.4f} @ep{cold_best_epoch}")

best_val_acc, best_val_f1, best_epoch_val = cold_best_val_acc, cold_best_val_f1, cold_best_epoch
test_acc, test_f1 = testC_acc, testC_f1
selected_source = (f'cold/global-best-val @ep{cold_best_epoch} '
                   f'(val={cold_best_val_acc:.4f}; ref dual-A test={testA_acc:.4f}, '
                   f'dual-B test={testB_acc:.4f})')
torch.save(cold_best_model_state, os.path.join(exp_dir, f"best_model_cold_{sfx}.pth"))
writer.add_scalar('testA/acc', testA_acc, 0); writer.add_scalar('testB/acc', testB_acc, 0)
writer.add_scalar('testC_cold/acc', testC_acc, 0)
writer.close()


# ============================ Save ============================


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
    'n_fusion_distill': args.n_fusion_distill,
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

profiler.remove_hooks()

print(f"\nBest Model (Epoch {best_epoch_val}): "
      f"val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
print(f"Selected: {selected_source}")
print(f"Directory: {exp_dir}")
