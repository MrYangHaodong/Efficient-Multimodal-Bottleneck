"""IEMOCAP selector fine-tuning by Shapley distillation (vanilla KL).

Frozen fusion = the 2026-05-17 IEMOCAP fusion-only-dyna checkpoint.
Only the selector is trainable; its inputs are temporally avg-pooled
by ``--selector_downsample_factor`` (default 4) so the selector sees
``T/4`` of what the fusion sees.

Training target = per-sample Shapley values  phi_j(x_i)  precomputed
on the train split by ``compute_shapley_iemocap.py`` and read from
the npz file passed via ``--shapley_train_npz``.

Distillation loss:
    L_kl = - (1/B) sum_i sum_j p_ij * log W_ij,
    p = softmax( phi / tau ),   W = softmax(selector logits).

Per-K best-ckpt tracking: each epoch we evaluate at every top_k in
[1..M=6] (by temporarily overriding ``inner.top_k``) and keep the best
val_acc snapshot for each K.  Final test eval reports per-K accuracy
using each K's own best checkpoint.

Run:
    python script/main_v6_iemocap_selector_finetune_shapley.py \\
        --fold=0 --cuda_pick=cuda:0 \\
        --frozen_ckpt=./model_chkpt/IEMOCAP/multimodal_model/<exp>/best_model_fold0.pth \\
        --shapley_train_npz=./model_chkpt/IEMOCAP/multimodal_model/<exp>/shapley_train_fold0_phi.npz \\
        --target_k=4 --selector_downsample_factor=4 \\
        --num_epochs=30 --lr=3e-4 \\
        --exp_name=v6_shapley_distill \\
        --results_dir=./model_chkpt/IEMOCAP/selector/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed, count_model_parameters
from utils.train_utils import AverageMeter, ProgressMeter
from data.IEMOCAP.get_data import (
    IEMOCAPDataset, FEATURE_DIMS, NUM_CLASSES, collate_fn,
)
from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)


# --------------------------------------------------------------------- #
#                          Config + wrapper                             #
# --------------------------------------------------------------------- #


class _IEMOCAPV6Cfg:
    def __init__(self, modalities):
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        v_dim = FEATURE_DIMS.get('video', 1024)
        self.video_high_dim = v_dim
        self.video_low_dim = v_dim


class V6IEMOCAPShapleyDistillWrapper(nn.Module):
    """Frozen-fusion / trainable-selector wrapper for IEMOCAP.

    Unpacks the IEMOCAPDataset batch (12 modality tensors + label)
    into V6's ``(high_dim_inputs, low_dim_inputs)`` dicts and adds
    Shapley distillation loss.

    Forward inputs:
      - feats: list/tuple of 2*M tensors (h0, l0, h1, l1, ...)
      - phi:   [B, M] precomputed Shapley target  (training only)
      - training: bool
      - labels: [B] long tensor  (only used for probe head aux loss
                                  and the optional soft-probe CE)

    Forward returns ``(logits, W, aux)`` where ``aux`` may contain
    ``shapley_kl``, ``ce_soft``, ``probe_loss``.
    """

    def __init__(self, v6_model, modalities, target_k, tau=0.5,
                 loss_variant='sigmoid_kl', modality_cost_vec=None):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self.target_k = int(target_k)
        self.tau = float(tau)
        self.loss_variant = loss_variant
        self.M = len(self.modalities)
        # Per-modality inference-cost penalty vector [M], in canonical
        # modality order.  None or all-zero -> no cost penalty.  See
        # ``--lambda_cost`` / ``--modality_costs`` for the rationale.
        if modality_cost_vec is None:
            modality_cost_vec = [0.0] * self.M
        self.register_buffer(
            'modality_cost_vec',
            torch.tensor(modality_cost_vec, dtype=torch.float32))

    def _unpack(self, feats):
        assert len(feats) == 2 * self.M
        high, low = {}, {}
        for j, m in enumerate(self.modalities):
            high[m] = feats[2 * j]
            low[m] = feats[2 * j + 1]
        return high, low

    @staticmethod
    def shapley_kl(W, phi, tau):
        """Original loss: -sum_j p_j log W_j with W = sigmoid(logits).
        Pushes high W where phi is high but does NOT enforce a ranking
        constraint — fine for batch top-K but degenerate for per-sample."""
        log_W = torch.log(W.clamp(min=1e-8))
        p = F.softmax(phi / tau, dim=-1)
        return -(p * log_W).sum(dim=-1).mean()

    @staticmethod
    def shapley_softmax_kd(raw_logits, phi, tau):
        """Per-sample softmax KD: forces softmax(selector_logits) to match
        softmax(phi/tau).  Standard knowledge-distillation form — proper KL
        between two normalised distributions, so the *ranking* of selector
        scores per sample is supervised, not just the magnitudes."""
        log_p_pred = F.log_softmax(raw_logits, dim=-1)
        p_targ = F.softmax(phi / tau, dim=-1)
        return -(p_targ * log_p_pred).sum(dim=-1).mean()

    @staticmethod
    def shapley_hard_ce(raw_logits, phi, tau):
        """InfoNCE-style hard-label cross-entropy on the per-sample
        argmax of phi.  Treats "which single modality is most useful for
        this sample" as a classification problem on the selector logits.
        Inspired by the NeurIPS 2025 best paper "1000 Layer Networks for
        Self-Supervised RL" — they pick classification (InfoNCE) over
        regression (KD) because the loss has predictable scaling
        properties, especially under weak/sparse supervision.

        ``tau`` rescales logits before CE (analogous to softmax KD temp);
        leave at 1.0 for standard CE.
        """
        target = phi.argmax(dim=-1)            # [B]
        return F.cross_entropy(raw_logits / max(tau, 1e-6), target)

    def forward(self, feats, phi=None, training: bool = False, labels=None):
        high, low = self._unpack(feats)
        result = self.model(high, low, training=training,
                            return_selection_info=True, labels=labels)
        if len(result) == 5:
            logits, _, W, _, aux = result
        else:
            logits, _, W, _ = result
            aux = {}
        if training and phi is not None:
            sel = getattr(self.model, 'modality_selector', None)
            raw = getattr(sel, '_raw_logits', None) if sel is not None else None
            # Pick loss variant — explicit user choice overrides the legacy
            # per-sample auto-switch to softmax KD.
            variant = self.loss_variant
            if variant == 'softmax_kd' and raw is not None:
                aux['shapley_kl'] = self.shapley_softmax_kd(raw, phi, self.tau)
            elif variant == 'hard_ce' and raw is not None:
                aux['shapley_kl'] = self.shapley_hard_ce(raw, phi, self.tau)
            elif (sel is not None and getattr(sel, 'per_sample_topk', False)
                    and raw is not None):
                # Per-sample legacy path: auto-use softmax KD (sigmoid KL
                # can't rank per-sample — diagnosed earlier).
                aux['shapley_kl'] = self.shapley_softmax_kd(raw, phi, self.tau)
            else:
                # Default / batch mode / fallback: legacy sigmoid KL.
                aux['shapley_kl'] = self.shapley_kl(W, phi, self.tau)

            # Per-modality cost penalty (S1).  Adds sum_m cost[m] *
            # E_b[sigmoid(raw_logits[b, m])] to the loss — selector pays
            # for picking expensive modalities.  Only active when
            # modality_cost_vec is non-zero.
            cost_vec = self.modality_cost_vec
            if cost_vec is not None and cost_vec.abs().sum() > 0:
                if raw is not None:
                    pick_prob = torch.sigmoid(raw).mean(dim=0)   # [M]
                else:
                    pick_prob = W.mean(dim=0)                    # [M]
                aux['modality_cost'] = (cost_vec.to(pick_prob.device)
                                          * pick_prob).sum()
            else:
                aux['modality_cost'] = torch.tensor(0.0, device=W.device)
            # Soft-fusion CE through probe heads (differentiable through W).
            pl = getattr(self.model.modality_selector, '_probe_logits', {})
            if pl and labels is not None:
                stacked = torch.stack(
                    [pl[m] for m in self.modalities if m in pl], dim=1)
                w_sub = W[:, :stacked.shape[1]]
                soft_logits = (w_sub.unsqueeze(-1) * stacked).sum(dim=1)
                aux['ce_soft'] = F.cross_entropy(soft_logits, labels)
        return logits, W, aux


# --------------------------------------------------------------------- #
#                      Dataset wrapper with Phi target                 #
# --------------------------------------------------------------------- #


class IEMOCAPDatasetWithPhi(Dataset):
    """Wraps an IEMOCAPDataset to return ``(*feats, label, phi[i])``.

    Phi must be in the order produced by a ``shuffle=False``
    iteration of the underlying dataset (i.e. the order used by
    ``compute_shapley_iemocap.py --split=train --save_phi``).
    """

    def __init__(self, ds, phi):
        assert len(ds) == phi.shape[0], (
            f'Dataset len {len(ds)} != Phi rows {phi.shape[0]}')
        self.ds = ds
        self.phi = torch.from_numpy(phi).float()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]   # list of 2M tensors + int label
        return sample + [self.phi[idx]]   # append phi as last element


def _collate_with_phi(batch):
    """Same as ``collate_fn`` but the last item per sample is phi
    (a [M] FloatTensor) instead of label.  We keep both.  Returns
    ``(*feats_padded, labels_tensor, phi_tensor)``."""
    from torch.nn.utils.rnn import pad_sequence
    # Last item per sample is phi.  Second-last is the int label.
    feats_list = [[] for _ in range(len(batch[0]) - 2)]
    labels, phis = [], []
    for s in batch:
        for i in range(len(feats_list)):
            feats_list[i].append(s[i])
        labels.append(s[-2])
        phis.append(s[-1])
    padded = [pad_sequence(fl, batch_first=True) for fl in feats_list]
    label_t = torch.tensor(labels, dtype=torch.long).view(-1, 1)
    phi_t = torch.stack(phis, dim=0)
    return tuple(padded) + (label_t, phi_t)


# --------------------------------------------------------------------- #
#                            Train / eval                              #
# --------------------------------------------------------------------- #


def _unpack_train_batch(batch, modalities, device):
    """Train loader yields (h0, l0, h1, l1, ..., labels[B,1], phi[B,M])."""
    *feats, labels, phi = batch
    feats = [f.to(device).float() for f in feats]
    labels = labels.squeeze(-1).long().to(device)
    phi = phi.to(device).float()
    return feats, labels, phi


def _unpack_eval_batch(batch, modalities, device):
    """Eval loader yields (h0, l0, h1, l1, ..., labels[B,1])."""
    *feats, labels = batch
    feats = [f.to(device).float() for f in feats]
    labels = labels.squeeze(-1).long().to(device)
    return feats, labels


def train_one_epoch(loader, model, optimizer, epoch, device, criterion,
                    lambda_kl, lambda_ce, lambda_ce_soft, lambda_probe,
                    lambda_cost, modalities, clip_grad=1.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    kl_meter = AverageMeter('KL', ':.4f')
    ce_meter = AverageMeter('CE', ':.4f')
    ces_meter = AverageMeter('CEs', ':.4f')
    aux_meter = AverageMeter('Aux', ':.4f')
    cost_meter = AverageMeter('Cost', ':.4f')
    model.train()
    for i, batch in enumerate(loader):
        feats, y, phi = _unpack_train_batch(batch, modalities, device)
        logits, W, aux = model(feats, phi=phi, training=True, labels=y)
        kl = aux.get('shapley_kl', torch.tensor(0.0, device=device))
        if lambda_ce > 0:
            ce = criterion(logits, y)
        else:
            ce = torch.tensor(0.0, device=device)
        ce_soft = aux.get('ce_soft', torch.tensor(0.0, device=device))
        if not isinstance(ce_soft, torch.Tensor):
            ce_soft = torch.tensor(float(ce_soft), device=device)
        probe = aux.get('probe_loss', torch.tensor(0.0, device=device))
        if not isinstance(probe, torch.Tensor):
            probe = torch.tensor(float(probe), device=device)
        cost = aux.get('modality_cost', torch.tensor(0.0, device=device))
        if not isinstance(cost, torch.Tensor):
            cost = torch.tensor(float(cost), device=device)

        loss = (lambda_kl * kl + lambda_ce * ce
                + lambda_ce_soft * ce_soft + lambda_probe * probe
                + lambda_cost * cost)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], clip_grad)
        optimizer.step()

        loss_meter.update(loss.item(), y.size(0))
        kl_meter.update(kl.item(), y.size(0))
        ce_meter.update(ce.item(), y.size(0))
        ces_meter.update(ce_soft.item(), y.size(0))
        aux_meter.update(probe.item(), y.size(0))
        cost_meter.update(cost.item(), y.size(0))

        if (i % 50 == 0) or (i == len(loader) - 1):
            progress = ProgressMeter(
                len(loader),
                [loss_meter, kl_meter, ce_meter, ces_meter, aux_meter],
                prefix=f'Epoch: [{epoch}]')
            progress.display(i)
    print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
          f'KL: {kl_meter.avg:.4f}  CE: {ce_meter.avg:.4f}  '
          f'CEs: {ces_meter.avg:.4f}  Probe: {aux_meter.avg:.4f}')
    return loss_meter.avg, kl_meter.avg


@torch.no_grad()
def evaluate_one_epoch(loader, model, criterion, epoch, device, modalities,
                       label='', verbose=True):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        feats, y = _unpack_eval_batch(batch, modalities, device)
        logits, _, _ = model(feats, phi=None, training=False, labels=None)
        loss = criterion(logits, y)
        _, pred = torch.max(logits, 1)
        acc = pred.eq(y).sum().item() / y.size(0)
        loss_meter.update(loss.item(), y.size(0))
        acc_meter.update(acc, y.size(0))
        all_preds.append(pred.cpu()); all_labels.append(y.cpu())
    if verbose:
        suffix = f' [{label}]' if label else ''
        print(f'Eval{suffix}  Loss={loss_meter.avg:.4f}  Acc={acc_meter.avg:.4f}')
    return (loss_meter.avg, acc_meter.avg,
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())


# --------------------------------------------------------------------- #
#                                CLI                                    #
# --------------------------------------------------------------------- #


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in ('y', 'yes', 't', 'true', '1'):
        return True
    if s in ('n', 'no', 'f', 'false', '0', ''):
        return False
    raise argparse.ArgumentTypeError(f'Boolean expected, got {v!r}')


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='v6_shapley_distill', type=str)
parser.add_argument('--results_dir',
                    default='./model_chkpt/IEMOCAP/selector/', type=str)
parser.add_argument('--dataset', default='IEMOCAP', type=str)

# Required: frozen fusion + precomputed Shapley
parser.add_argument('--frozen_ckpt', required=True, type=str)
parser.add_argument('--shapley_train_npz', required=True, type=str)

parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2])
parser.add_argument('--aggregate', action='store_true', default=False)

# Data
parser.add_argument('--data_root', default='/files1/haodong/data/IEMOCAP', type=str)
parser.add_argument('--max_seq_len', default=128, type=int)
parser.add_argument('--time_compression_ratio', default=4, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--modalities', nargs='+',
                    default=list(IEMOCAPDataset.ALL_MODALITIES),
                    choices=list(IEMOCAPDataset.ALL_MODALITIES))

# Training knobs
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine'])
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)

# Distillation knobs
parser.add_argument('--tau', default=0.5, type=float)
parser.add_argument('--lambda_kl', default=1.0, type=float)
parser.add_argument('--lambda_ce', default=0.0, type=float)
parser.add_argument('--lambda_ce_soft', default=0.0, type=float)
parser.add_argument('--lambda_probe', default=0.05, type=float)
parser.add_argument('--lambda_cost', default=0.0, type=float,
                    help='Weight on per-modality inference-cost penalty. '
                         'Adds ``lambda_cost * sum_m cost[m] * '
                         'sigmoid(raw_logits[:, m]).mean()`` to the loss '
                         '— forces selector to prefer cheap modalities '
                         'when accuracy is comparable.  Use to push '
                         'selector off expensive video/DINOv3 features.')
parser.add_argument('--modality_costs', type=str, default=None,
                    help='JSON dict {modality: float_cost}.  Default = '
                         'no penalty (all costs 0).  E.g. for IEMOCAP: '
                         '\'{"video":10,"audio":1,"text":0.5,'
                         '"mocap_hand":0.1,"mocap_head":0.1,'
                         '"mocap_rotated":0.1}\' makes video 10x '
                         'cheaper-to-avoid than audio.')
parser.add_argument('--center_phi', default=False, type=_str2bool,
                    help='Subtract per-modality mean from Shapley target '
                         'before softmax.  Removes global "modality X is '
                         'on-average best" bias and exposes per-sample '
                         'deviation.  Useful when train/test Shapley '
                         'distributions differ (e.g. IEMOCAP text shift).')
parser.add_argument('--use_val_split', default=False, type=_str2bool,
                    help='Train selector on the val split rather than '
                         'the train split.  Requires the shapley npz '
                         'computed on the val split.  Useful when train '
                         'Shapley distribution diverges from test (e.g. '
                         'IEMOCAP); val is in-domain held-out and closer '
                         'to test distribution.')

# Selector
parser.add_argument('--target_k', default=4, type=int)
parser.add_argument('--use_interaction_matrix', default=False, type=_str2bool)
parser.add_argument('--use_holo_bias', default=False, type=_str2bool)
parser.add_argument('--holo_scale', default=1.0, type=float)
parser.add_argument('--selector_downsample_factor', default=4, type=int)
parser.add_argument('--selector_depth', default=0, type=int,
                    help='If >0, replace the 3-layer ReLU selector_mlp '
                         'with a stack of ``selector_depth`` residual '
                         'blocks (pre-LN + SiLU). Recipe from the '
                         'NeurIPS 2025 best paper "1000 Layer Networks '
                         'for Self-Supervised RL" (Wang et al.). '
                         'Recommended: try {8, 16, 32}.')
parser.add_argument('--loss_variant', default='sigmoid_kl', type=str,
                    choices=['sigmoid_kl', 'softmax_kd', 'hard_ce'],
                    help='sigmoid_kl: legacy -sum p log sigmoid(W). '
                         'softmax_kd: KL between softmax(raw_logits) and '
                         'softmax(phi/tau). hard_ce: per-sample CE with '
                         'phi.argmax as the positive class (InfoNCE-style '
                         'from contrastive RL — gives stronger ranking '
                         'signal than KD when targets are weak).')
parser.add_argument('--per_sample_topk', default=False, type=_str2bool,
                    help='Enable per-sample top-K modality routing.  Each '
                         'sample picks its own K modalities; the bottleneck '
                         'aggregator + final GAP are mask-aware so only the '
                         'selected K contribute per sample.  Required for the '
                         'phi-oracle per-sample gain (~+9.6pp at K=1).')

# Backbone (must match fusion dyna ckpt)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--internal_dim', default=128, type=int)
parser.add_argument('--n_bottlenecks', default=8, type=int)
parser.add_argument('--fusion_layer', default=0, type=int)
parser.add_argument('--use_sparse_attn', default=True, type=_str2bool)
parser.add_argument('--sparse_attn_variant', default='opt', type=str)
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--num_experts', default=4, type=int)

args = parser.parse_args()


# --------------------------------------------------------------------- #
#                                Setup                                  #
# --------------------------------------------------------------------- #


device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed_num)
current_date = datetime.now().strftime('%Y-%m-%d')
exp_name_full = f'{current_date}_{args.dataset}_{args.exp_name}'
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)


# --------------------------------------------------------------------- #
#                              Aggregation                              #
# --------------------------------------------------------------------- #


if args.aggregate:
    fold_results = []
    for fold_idx in range(3):
        p = os.path.join(exp_dir, f'results_fold{fold_idx}.json')
        if os.path.exists(p):
            with open(p) as f:
                fold_results.append(json.load(f))
    if not fold_results:
        print('No fold results to aggregate.')
        sys.exit(1)
    aggregated = {'experiment_name': exp_name_full, 'folds': fold_results}
    common = set(fold_results[0].keys())
    for fr in fold_results[1:]:
        common &= set(fr.keys())
    for key in sorted(common):
        vals = [fr[key] for fr in fold_results]
        if not all(isinstance(v, (int, float)) for v in vals):
            continue
        aggregated[f'{key}_mean'] = float(np.mean(vals))
        aggregated[f'{key}_std'] = float(np.std(vals))
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(aggregated, f, indent=4)
    print(f'Aggregated -> {exp_dir}/results.json')
    for key in ('best_val_acc', 'best_val_f1', 'best_test_acc', 'best_test_f1'):
        m = aggregated.get(f'{key}_mean'); s = aggregated.get(f'{key}_std')
        if m is not None:
            print(f'  {key}: {m:.4f} +/- {s:.4f}')
    for K in range(1, 7):
        for tag in ('best_val_acc', 'best_test_acc'):
            key = f'{tag}_k{K}'
            m = aggregated.get(f'{key}_mean'); s = aggregated.get(f'{key}_std')
            if m is not None:
                print(f'  {key}: {m:.4f} +/- {s:.4f}')
    sys.exit(0)


# --------------------------------------------------------------------- #
#                                Data                                  #
# --------------------------------------------------------------------- #


print(f'Device: {device}')
print(f'Experiment: {exp_name_full}')
print(f'Fold: {args.fold}')

modalities = list(args.modalities)
num_m = len(modalities)
num_classes = NUM_CLASSES

if args.fold is None:
    raise ValueError('--fold required (single-fold script).')
suffix = '' if args.fold == 0 else f'_split{args.fold}'
train_csv = os.path.join(args.data_root, f'train{suffix}.csv')
val_csv   = os.path.join(args.data_root, f'val{suffix}.csv')
test_csv  = os.path.join(args.data_root, f'test{suffix}.csv')

# Underlying datasets (no Phi for val/test).
ds_kwargs = dict(
    data_root=args.data_root, modalities=modalities,
    max_seq_len=args.max_seq_len,
    time_compression_ratio=args.time_compression_ratio,
    use_batched_fusion=True,
)
if args.use_val_split:
    # Use val samples as the SELECTOR-TRAINING set (with val Shapley).
    # val_ds is still used for monitoring (same data, but that's fine
    # since we're not doing model selection on a fresh val).  Test set
    # untouched.
    train_ds_raw = IEMOCAPDataset(csv_path=val_csv, **ds_kwargs)
    print(f'NOTE: --use_val_split=True. Selector trains on VAL split '
          f'(N={len(train_ds_raw)}) using val-Shapley target.')
else:
    train_ds_raw = IEMOCAPDataset(csv_path=train_csv, **ds_kwargs)
val_ds = IEMOCAPDataset(csv_path=val_csv, **ds_kwargs)
eval_ds = IEMOCAPDataset(csv_path=test_csv, **ds_kwargs)

# Load precomputed Shapley target on train
phi_pack = np.load(args.shapley_train_npz, allow_pickle=True)
phi_train = phi_pack['phi'].astype(np.float32)
phi_modalities = (list(phi_pack['modalities'])
                   if 'modalities' in phi_pack.files else None)
if phi_modalities is not None and phi_modalities != modalities:
    raise ValueError(
        f'modality order mismatch: shapley npz {phi_modalities} vs '
        f'cfg {modalities}')
print(f'Loaded Shapley train Phi:  shape={phi_train.shape}  '
      f'mean={phi_train.mean():.4f}  std={phi_train.std():.4f}')
if args.center_phi:
    print('Per-modality mean BEFORE centering:')
    for j, m in enumerate(modalities):
        print(f'  {m:<14}  mean={phi_train[:, j].mean():+.4f}  '
              f'std={phi_train[:, j].std():.4f}')
    phi_train = phi_train - phi_train.mean(axis=0, keepdims=True)
    print('Phi centered (per-modality mean -> 0). Selector will learn '
          'per-sample deviations rather than absolute argmax bias.')
    print(f'After centering:  mean={phi_train.mean():.6f}  '
          f'std={phi_train.std():.4f}')

train_ds = IEMOCAPDatasetWithPhi(train_ds_raw, phi_train)

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers,
                          pin_memory=True, drop_last=True,
                          collate_fn=_collate_with_phi)
val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, drop_last=False,
                        collate_fn=collate_fn)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers,
                         pin_memory=True, drop_last=False,
                         collate_fn=collate_fn)
print(f'  train={len(train_ds_raw)}, val={len(val_ds)}, eval={len(eval_ds)}')

input_length = args.max_seq_len if args.max_seq_len > 0 else 600


# --------------------------------------------------------------------- #
#                                Model                                 #
# --------------------------------------------------------------------- #


v6_cfg = _IEMOCAPV6Cfg(modalities)
inner = DualVideoBottleneckModelV6Downsample(
    cfg=v6_cfg,
    output_dim=num_classes,
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
    num_classes=num_classes,
    lambda_probe=args.lambda_probe, lambda_diversity=0.0,
    lambda_reinforce=0.0, lambda_sparsity=0.0,
    sparse_attn_variant=args.sparse_attn_variant,
    strat_block_size=args.strat_block_size,
    downsample_min_len=args.downsample_min_len,
    use_batched_fusion=True,
    per_modal_distill=False,
    per_modal_downsample_min_len=args.downsample_min_len,
    use_interaction_matrix=args.use_interaction_matrix,
    use_holo_bias=args.use_holo_bias,
    holo_scale=args.holo_scale,
    selector_downsample_factor=args.selector_downsample_factor,
    selector_depth=args.selector_depth,
)
inner.top_k = args.target_k
if args.per_sample_topk and inner.modality_selector is not None:
    inner.modality_selector.per_sample_topk = True
    print('per_sample_topk=True: selector picks K modalities per-sample; '
          'fusion runs all M with mask-aware aggregator + GAP.')

# Load frozen fusion (saved from wrapper -> strip "model." prefix)
state = torch.load(args.frozen_ckpt, map_location='cpu', weights_only=False)
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k, v in state.items()
             if k.startswith('model.')}
missing, unexpected = inner.load_state_dict(state, strict=False)
print(f'Loaded {args.frozen_ckpt}  '
      f'(missing={len(missing)} [expected: selector keys], '
      f'unexpected={len(unexpected)})')

# Freeze everything except modality_selector
for name, p in inner.named_parameters():
    if 'modality_selector' in name:
        p.requires_grad = True
    else:
        p.requires_grad = False
n_train_p = sum(p.numel() for p in inner.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in inner.parameters())
print(f'Trainable params: {n_train_p:,}/{n_total:,} '
      f'({100*n_train_p/n_total:.2f}%)')

_mod_cost_dict = (json.loads(args.modality_costs)
                   if args.modality_costs else {})
modality_cost_vec = [float(_mod_cost_dict.get(m, 0.0)) for m in modalities]
if any(c != 0 for c in modality_cost_vec):
    print('Modality cost vector (S1):  ' + '  '.join(
        f'{m[:5]}={c:.2f}' for m, c in zip(modalities, modality_cost_vec)))
    print(f'  lambda_cost = {args.lambda_cost}')

model = V6IEMOCAPShapleyDistillWrapper(
    inner, modalities, target_k=args.target_k, tau=args.tau,
    loss_variant=args.loss_variant,
    modality_cost_vec=modality_cost_vec,
).to(device).float()

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                       lr=args.lr)
criterion = nn.CrossEntropyLoss()
scheduler = None
if args.lr_schedule == 'cosine':
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)


# --------------------------------------------------------------------- #
#                       Per-K best-ckpt training                       #
# --------------------------------------------------------------------- #


best_per_k = {k: {'val_acc': -1.0, 'val_f1': 0.0, 'val_loss': float('inf'),
                  'epoch': -1, 'state': None}
              for k in range(1, num_m + 1)}
inner_ref = model.model
saved_target_k = inner_ref.top_k


def _snapshot_state(m):
    return {kk: vv.detach().cpu().clone() for kk, vv in m.state_dict().items()}


writer = SummaryWriter(comment=f'_{exp_name_full}_fold{args.fold}')

for epoch in range(args.num_epochs):
    t0 = time.time()
    train_loss, train_kl = train_one_epoch(
        train_loader, model, optimizer, epoch, device, criterion,
        lambda_kl=args.lambda_kl, lambda_ce=args.lambda_ce,
        lambda_ce_soft=args.lambda_ce_soft, lambda_probe=args.lambda_probe,
        lambda_cost=args.lambda_cost,
        modalities=modalities, clip_grad=args.clip_grad)

    # Per-K val
    epoch_k = {}
    for k_ in range(1, num_m + 1):
        inner_ref.top_k = k_
        v_loss, v_acc, v_y, v_p = evaluate_one_epoch(
            val_loader, model, criterion, epoch, device, modalities,
            label=f'k{k_}', verbose=False)
        v_f1 = f1_score(v_y, v_p, average='macro')
        epoch_k[k_] = (v_loss, v_acc, v_f1)
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

    if scheduler is not None:
        scheduler.step()

    k_summary = '  '.join(
        f'k{k_}={epoch_k[k_][1]:.4f}' for k_ in range(1, num_m + 1))
    print(f'End of Epoch {epoch} | Val Acc per K: {k_summary}  '
          f'({time.time()-t0:.1f}s)')
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/kl', train_kl, epoch)


# --------------------------------------------------------------------- #
#                             Test eval per K                          #
# --------------------------------------------------------------------- #


print('\n' + '=' * 80)
print('Test-set eval: best ckpt per K')
print('=' * 80)

test_per_k = {}
for k_ in range(1, num_m + 1):
    entry = best_per_k[k_]
    if entry['state'] is None:
        continue
    model.load_state_dict({kk: vv.to(device) for kk, vv in entry['state'].items()})
    inner_ref.top_k = k_
    _, t_acc, t_y, t_p = evaluate_one_epoch(
        eval_loader, model, criterion, 0, device, modalities,
        label=f'test_k{k_}', verbose=False)
    t_f1 = f1_score(t_y, t_p, average='macro')
    test_per_k[k_] = {'test_acc': float(t_acc), 'test_f1': float(t_f1)}
    print(f'  K={k_}: best_epoch={entry["epoch"]:>3d}  '
          f'val_acc={entry["val_acc"]:.4f}  val_f1={entry["val_f1"]:.4f}  '
          f'test_acc={t_acc:.4f}  test_f1={t_f1:.4f}')
    ck = (f'best_model_fold{args.fold}_k{k_}.pth'
          if args.fold is not None else f'best_val_model_k{k_}.pth')
    torch.save(entry['state'], os.path.join(exp_dir, ck))
    writer.add_scalar(f'test_k{k_}/acc', t_acc, 0)
    writer.add_scalar(f'test_k{k_}/f1', t_f1, 0)
inner_ref.top_k = saved_target_k
writer.close()

_tk = args.target_k if args.target_k in test_per_k else max(test_per_k.keys())
best_val_acc = best_per_k[_tk]['val_acc']
best_val_f1 = best_per_k[_tk]['val_f1']
best_epoch_val = best_per_k[_tk]['epoch']
test_acc = test_per_k[_tk]['test_acc']
test_f1 = test_per_k[_tk]['test_f1']


# --------------------------------------------------------------------- #
#                                Save                                  #
# --------------------------------------------------------------------- #


config = {
    'experiment_name': exp_name_full, 'dataset': args.dataset,
    'fold': args.fold, 'target_k': args.target_k,
    'modalities': modalities,
    'num_epochs': args.num_epochs, 'batch_size': args.batch_size,
    'lr': args.lr, 'lr_schedule': args.lr_schedule,
    'tau': args.tau, 'lambda_kl': args.lambda_kl,
    'lambda_ce': args.lambda_ce, 'lambda_ce_soft': args.lambda_ce_soft,
    'lambda_probe': args.lambda_probe,
    'selector_downsample_factor': args.selector_downsample_factor,
    'selector_depth': args.selector_depth,
    'loss_variant': args.loss_variant,
    'lambda_cost': args.lambda_cost,
    'modality_costs': _mod_cost_dict,
    'per_sample_topk': args.per_sample_topk,
    'max_seq_len': args.max_seq_len,
    'time_compression_ratio': args.time_compression_ratio,
    'frozen_ckpt': args.frozen_ckpt,
    'shapley_train_npz': args.shapley_train_npz,
    'seed': args.seed_num,
}
with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

model_stats = {
    'experiment_name': exp_name_full, 'fold': args.fold,
    'best_val_acc': best_val_acc, 'best_val_f1': best_val_f1,
    'best_epoch_val': best_epoch_val,
    'best_test_acc': float(test_acc), 'best_test_f1': float(test_f1),
}
for k_ in range(1, num_m + 1):
    if k_ in test_per_k:
        entry = best_per_k[k_]
        model_stats[f'best_val_acc_k{k_}'] = float(entry['val_acc'])
        model_stats[f'best_val_f1_k{k_}'] = float(entry['val_f1'])
        model_stats[f'best_epoch_val_k{k_}'] = int(entry['epoch'])
        model_stats[f'best_test_acc_k{k_}'] = float(test_per_k[k_]['test_acc'])
        model_stats[f'best_test_f1_k{k_}'] = float(test_per_k[k_]['test_f1'])

results_filename = (f'results_fold{args.fold}.json'
                    if args.fold is not None else 'results.json')
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(model_stats, f, indent=4)

print(f'\nBest model (K={_tk}, epoch={best_epoch_val}): '
      f'val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}')
print(f'Directory: {exp_dir}')
