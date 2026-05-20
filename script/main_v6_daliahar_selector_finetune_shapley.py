"""DaliaHAR selector fine-tuning by Shapley distillation.

Frozen fusion = 2026-05-14 fusion-only-dyna checkpoint.  Only the
selector is trainable; its inputs are temporally avg-pooled by
``--selector_downsample_factor`` (default 4), so the selector sees
``T/4`` of what the fusion sees.

Training target = per-sample Shapley values  phi_j(x_i)  precomputed
on the train split by ``compute_shapley_daliahar.py`` and read from
the npz file passed via ``--shapley_train_npz``.

Distillation loss (cross-entropy / KL form):
    L_kl = - (1/B) sum_i sum_j p_ij * log W_ij,
    p = softmax( phi / tau ),   W = softmax(selector logits).

Per-K best-ckpt tracking: each epoch we evaluate at every top_k in
[1..M] (by temporarily overriding ``inner.top_k``) and keep the best
val_acc snapshot for each K.  Final test eval reports per-K accuracy
using each K's own best checkpoint.

Run:
    python script/main_v6_daliahar_selector_finetune_shapley.py \
        --fold=0 --cuda_pick=cuda:0 \
        --frozen_ckpt=./model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna/best_model_fold0.pth \
        --shapley_train_npz=./model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna/shapley_train_fold0_phi.npz \
        --target_k=4 --selector_downsample_factor=4 \
        --num_epochs=30 --lr=3e-4 \
        --exp_name=v6_shapley_distill_ds4 \
        --results_dir=./model_chkpt/DaliaHAR/selector/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random as _random
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
from utils.dataset_cfg import DaliaHAR
from data.dataset_builder import HARDataset
from utils.train_utils import AverageMeter, ProgressMeter
from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)


# --------------------------------------------------------------------- #
#                          Config + wrapper                             #
# --------------------------------------------------------------------- #


class _DaliaV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


class V6ShapleyDistillWrapper(nn.Module):
    """Frozen-fusion / trainable-selector wrapper with Shapley
    distillation loss.

    Forward returns ``(logits, W, aux)`` where:

      * ``logits``     - fusion output at the selector's natural top-K.
      * ``W``          - selector softmax weights ``[B, M]`` (used as
                         the student distribution in KL).
      * ``aux``        - dict with keys:
                          - ``shapley_kl`` (training only, if phi given)
                          - ``probe_loss`` from the V6 selector aux head
                          - ``diversity`` / ``sparsity`` (forwarded from
                            V6 if present)
    """

    def __init__(self, v6_model, modalities, variates, target_k, tau=0.5):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self.variates = variates
        self.target_k = int(target_k)
        self.tau = float(tau)
        self.M = len(self.modalities)
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

    @staticmethod
    def shapley_kl(log_q: torch.Tensor, phi: torch.Tensor, tau: float) -> torch.Tensor:
        """Cross-entropy form of KL(softmax(phi/tau) || softmax(raw_logits)).

        log_q: [B, M] already-log-softmax-normalised student logits.
        phi  : [B, M] per-sample Shapley targets.

        Why: the original implementation used ``log W`` with ``W =
        sigmoid(logits)`` (independent per-modality gates). That degrades
        to ``W_j ≈ 1 for all j`` (loss → 0 with zero ranking info). Using
        softmax over raw logits enforces a proper distribution → real
        relative-importance gradient.
        """
        p = F.softmax(phi / tau, dim=-1)
        return -(p * log_q).sum(dim=-1).mean()

    @staticmethod
    def shapley_listmle(raw_logits: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Plackett-Luce log-likelihood of the gold ranking induced by phi.

        For each sample, the 'gold ranking' is phi sorted descending.
        Negative-log-likelihood under Plackett-Luce:
            -log P(π | s) = Σ_k [log Σ_{l>=k} exp(s_{π(l)}) - s_{π(k)}]

        Stricter than KL: directly penalises violations of the full
        ranking, not just probability-mass mismatch. Useful when the
        downstream metric is top-K selection.

        raw_logits: [B, M] selector raw logits (pre-softmax).
        phi:        [B, M] per-sample Shapley targets.
        """
        _, perm = phi.sort(dim=-1, descending=True)        # [B, M]
        s_sorted = torch.gather(raw_logits, dim=-1, index=perm)
        # log-cumsum-exp from k to M-1 (suffix). Compute via flip+cumsum+flip.
        log_cumsum = torch.logcumsumexp(
            s_sorted.flip(dims=[-1]), dim=-1
        ).flip(dims=[-1])
        return (log_cumsum - s_sorted).sum(dim=-1).mean()

    def forward(self, x, phi=None, training: bool = False, labels=None):
        high, low = self._split(x)
        result = self.model(high, low, training=training,
                            return_selection_info=True, labels=labels)
        if len(result) == 5:
            logits, _, W, _, aux = result
        else:
            logits, _, W, _ = result
            aux = {}
        if training and phi is not None:
            # Use selector raw pre-sigmoid logits → softmax → KL with phi.
            # This is the FIX-A path. The selector exposes ``_raw_logits``
            # at v6_selector.py:341; it is masked to -inf for unavailable
            # modalities, which softmax handles correctly (0 mass).
            raw_logits = getattr(self.model.modality_selector,
                                 '_raw_logits', None)
            if raw_logits is None:
                # Fallback: log(W) as before (sigmoid path).
                log_q = torch.log(W.clamp(min=1e-8))
            else:
                log_q = F.log_softmax(raw_logits, dim=-1)
            aux['shapley_kl'] = self.shapley_kl(log_q, phi, self.tau)
            # ListMLE ranking loss (uses raw logits directly).
            if raw_logits is not None:
                aux['shapley_listmle'] = self.shapley_listmle(raw_logits, phi)
            # Differentiable soft-fusion CE through per-modality probe heads.
            # The selector's _probe_logits[m] is the K-class output of
            # the m-th probe head — fully diff w.r.t. selector params.
            # soft logits = sum_j  W_ij * probe_logits_j   gives a path
            # where CE backprops to W (the hard top-K fusion path's
            # gradient is broken by .item() at v6_downsample:1248).
            pl = getattr(self.model.modality_selector, '_probe_logits', {})
            if pl and labels is not None:
                stacked = torch.stack(
                    [pl[m] for m in self.modalities if m in pl], dim=1)  # [B, M, C]
                # Re-align W indices to the subset of modalities present
                # in pl (should be all under normal training).
                w_sub = W[:, :stacked.shape[1]]                       # [B, M]
                soft_logits = (w_sub.unsqueeze(-1) * stacked).sum(dim=1)  # [B, C]
                aux['ce_soft'] = F.cross_entropy(soft_logits, labels)
        return logits, W, aux


# --------------------------------------------------------------------- #
#                       Dataset with Shapley target                    #
# --------------------------------------------------------------------- #


class HARDatasetWithPhi(Dataset):
    """Wraps an HARDataset to return ``(x, y, phi[i])``.

    The Phi tensor must have one row per sample in the underlying
    dataset's ``__getitem__`` order (i.e. the order produced by a
    ``shuffle=False`` DataLoader).  Pass the matching
    ``shapley_train_*_phi.npz`` produced by
    ``compute_shapley_daliahar.py --split=train --save_phi``.
    """

    def __init__(self, ds: HARDataset, phi: np.ndarray):
        assert len(ds) == phi.shape[0], (
            f'Dataset length {len(ds)} != Phi rows {phi.shape[0]}.  '
            f'Make sure Phi was computed for the same fold and split.')
        self.ds = ds
        self.phi = torch.from_numpy(phi).float()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, y, self.phi[idx]


# --------------------------------------------------------------------- #
#                              Train / eval                            #
# --------------------------------------------------------------------- #


def train_one_epoch(loader, model, optimizer, epoch, device,
                    lambda_kl, lambda_ce, lambda_ce_soft, lambda_probe,
                    lambda_diversity, lambda_sparsity, criterion,
                    clip_grad=1.0, lambda_listmle=0.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    kl_meter = AverageMeter('KL', ':.4f')
    ce_meter = AverageMeter('CE', ':.4f')
    ces_meter = AverageMeter('CEs', ':.4f')
    aux_meter = AverageMeter('Aux', ':.4f')
    mle_meter = AverageMeter('MLE', ':.4f')
    model.train()
    for i, (x, y, phi) in enumerate(loader):
        x = x.to(device).float()
        y = y.to(device)
        phi = phi.to(device).float()

        logits, W, aux = model(x, phi=phi, training=True, labels=y)

        # KL distillation (main signal)
        kl = aux.get('shapley_kl', torch.tensor(0.0, device=device))

        # Task CE.  Two variants:
        #   - ``lambda_ce``        applied to CE on the fusion's top-K
        #                          output.  This has ZERO gradient w.r.t.
        #                          the selector because hard top-K +
        #                          ``.item()`` in V6's use_weighted_factor
        #                          path detach the gradient — useful only
        #                          as a forward-only monitoring number.
        #   - ``lambda_ce_soft``   applied to CE on the soft-fusion of
        #                          per-modality probe head logits weighted
        #                          by ``W``.  Fully differentiable through
        #                          the selector → real gradient path.
        if lambda_ce > 0:
            ce = criterion(logits, y)
        else:
            ce = torch.tensor(0.0, device=device)
        ce_soft = aux.get('ce_soft', torch.tensor(0.0, device=device))
        if not isinstance(ce_soft, torch.Tensor):
            ce_soft = torch.tensor(float(ce_soft), device=device)

        # Auxiliary probe / diversity / sparsity from V6 selector head
        probe = aux.get('probe_loss', torch.tensor(0.0, device=device))
        if not isinstance(probe, torch.Tensor):
            probe = torch.tensor(float(probe), device=device)
        diversity = aux.get('diversity', torch.tensor(0.0, device=device))
        if not isinstance(diversity, torch.Tensor):
            diversity = torch.tensor(float(diversity), device=device)
        sparsity = aux.get('sparsity', torch.tensor(0.0, device=device))
        if not isinstance(sparsity, torch.Tensor):
            sparsity = torch.tensor(float(sparsity), device=device)

        listmle = aux.get('shapley_listmle', torch.tensor(0.0, device=device))
        if not isinstance(listmle, torch.Tensor):
            listmle = torch.tensor(float(listmle), device=device)

        loss = (lambda_kl * kl
                + lambda_listmle * listmle
                + lambda_ce * ce
                + lambda_ce_soft * ce_soft
                + lambda_probe * probe
                + lambda_diversity * diversity
                + lambda_sparsity * sparsity)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], clip_grad)
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        kl_meter.update(kl.item(), x.size(0))
        ce_meter.update(ce.item(), x.size(0))
        ces_meter.update(ce_soft.item(), x.size(0))
        aux_meter.update(probe.item(), x.size(0))
        mle_meter.update(listmle.item(), x.size(0))

        if (i % 50 == 0) or (i == len(loader) - 1):
            progress = ProgressMeter(
                len(loader),
                [loss_meter, kl_meter, mle_meter, ce_meter, ces_meter, aux_meter],
                prefix=f'Epoch: [{epoch}]')
            progress.display(i)
    print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
          f'KL: {kl_meter.avg:.4f}  MLE: {mle_meter.avg:.4f}  '
          f'CE: {ce_meter.avg:.4f}  CEs: {ces_meter.avg:.4f}  '
          f'Probe: {aux_meter.avg:.4f}')
    return loss_meter.avg, kl_meter.avg


@torch.no_grad()
def evaluate_one_epoch(loader, model, criterion, epoch, device,
                       label: str = '', verbose: bool = True):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device)
        logits, _, _ = model(x, phi=None, training=False, labels=None)
        loss = criterion(logits, y)
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / x.size(0)
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))
        all_preds.append(predicted.cpu())
        all_labels.append(y.cpu())
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
parser.add_argument('--results_dir', default='./model_chkpt/DaliaHAR/selector/',
                    type=str)
parser.add_argument('--dataset', default='DaliaHAR', type=str)

# Required: frozen fusion + precomputed Shapley
parser.add_argument('--frozen_ckpt', required=True, type=str)
parser.add_argument('--shapley_train_npz', required=True, type=str,
                    help='Path to shapley_train_fold{F}_phi.npz from '
                         'compute_shapley_daliahar.py.  When '
                         '--use_val_split is set, pass the val-Shapley '
                         'npz instead.')
parser.add_argument('--use_val_split', default=False, type=_str2bool,
                    help='Train selector on the val subjects rather than '
                         'the train subjects.  Pair with a val-Shapley '
                         'npz (computed via compute_shapley_daliahar.py '
                         '--split=val --save_phi) for the val-Shap recipe '
                         'that on IEMOCAP gave +8.7pp at K=1 over '
                         'train-phi ShapDistill.')
parser.add_argument('--combine_train_val', default=False, type=_str2bool,
                    help='Concat train+val data (and Phi) for selector '
                         'training. Auto-infers val Phi path from train '
                         'Phi path. val_loader still used for per-K ckpt '
                         'selection but is now contaminated; test is the '
                         'meaningful metric.')

parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2])
parser.add_argument('--aggregate', action='store_true', default=False)

# Training knobs
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine'])
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--transform', default='sax', type=str)

# Distillation knobs
parser.add_argument('--tau', default=0.5, type=float,
                    help='Temperature for softmax(phi/tau) target.  Lower '
                         '= sharper / more concentrated on argmax.')
parser.add_argument('--lambda_kl', default=1.0, type=float)
parser.add_argument('--lambda_ce', default=0.0, type=float,
                    help='Weight for cross-entropy loss on the fusion '
                         'output at the selector top-K.  NOTE: V6 '
                         'fusion uses hard top-K + ``.item()`` in the '
                         'use_weighted_factor path, so the gradient '
                         'from this CE to the selector is ZERO.  Kept '
                         'as a forward-only monitoring number.  For '
                         'actual CE supervision use --lambda_ce_soft.')
parser.add_argument('--lambda_ce_soft', default=0.0, type=float,
                    help='Weight for CE on a soft fusion of the '
                         'per-modality probe-head logits weighted by '
                         '``W`` (the selector softmax). Fully '
                         'differentiable through the selector.')
parser.add_argument('--lambda_probe', default=0.05, type=float)
parser.add_argument('--lambda_diversity', default=0.0, type=float)
parser.add_argument('--lambda_sparsity', default=0.0, type=float)

# Selector head
parser.add_argument('--target_k', default=4, type=int)
parser.add_argument('--use_interaction_matrix', default=False, type=_str2bool)
parser.add_argument('--use_holo_bias', default=False, type=_str2bool)
parser.add_argument('--holo_scale', default=1.0, type=float)
parser.add_argument('--selector_downsample_factor', default=4, type=int)
parser.add_argument('--selector_downsample_mode', default='avg_pool',
                    choices=['avg_pool', 'stride'],
                    help='How to compress selector input along time: '
                         '"avg_pool" (default, legacy avg over factor tokens) '
                         'or "stride" (keep every factor-th token, preserves '
                         'sharp transients).')
parser.add_argument('--selector_input_source', default='raw',
                    choices=['raw', 'enc_layer1', 'enc_layer2'],
                    help='Where the selector reads its per-modality input '
                         'from. "raw" = downsampled raw modality input '
                         '(legacy). "enc_layer1"/"enc_layer2" = use the '
                         'frozen-fusion encoder output after the 1st / 2nd '
                         'transformer layer. Attacks bottleneck #2 (input '
                         'compression).')

# Backbone (must match dyna training)
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--internal_dim', default=64, type=int)
parser.add_argument('--n_bottlenecks', default=8, type=int)
parser.add_argument('--fusion_layer', default=0, type=int)
parser.add_argument('--use_sparse_attn', default=True, type=_str2bool)
parser.add_argument('--sparse_attn_variant', default='opt', type=str)
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--per_sample_topk', default=False, type=_str2bool,
                    help='Enable per-sample top-K in the selector. When True, '
                         'each sample picks its own K modalities via '
                         '(weights*confs).topk(K, dim=1) and the fusion '
                         'aggregator + GAP gate on the resulting [B, M] mask. '
                         'Default False keeps the original batch-mean top-K.')
parser.add_argument('--use_multi_pool', default=False, type=_str2bool,
                    help='Selector multi-pool: concat [attn, mean, max] '
                         'per-modality summaries before projector. Triples '
                         'projector input width.')
parser.add_argument('--use_tx_summarizer', default=False, type=_str2bool,
                    help='Selector Transformer summarizer: per-modality '
                         'Linear embed -> learned pos embed -> 1-layer self-'
                         'attention TransformerEncoder over tokens, then '
                         'pool. Attacks input-compression bottleneck.')
parser.add_argument('--tx_d_model', default=64, type=int)
parser.add_argument('--tx_nhead', default=4, type=int)
parser.add_argument('--tx_layers', default=1, type=int)
parser.add_argument('--tx_dim_ff', default=128, type=int)
parser.add_argument('--tx_dropout', default=0.1, type=float)
parser.add_argument('--lambda_listmle', default=0.0, type=float,
                    help='Weight for Plackett-Luce ListMLE loss on the '
                         'per-sample Shapley ranking (uses selector raw logits).')
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='AdamW weight decay. >0 switches optimizer from '
                         'Adam to AdamW.')

args = parser.parse_args()


# --------------------------------------------------------------------- #
#                              Setup                                   #
# --------------------------------------------------------------------- #


device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed_num)
current_date = datetime.now().strftime('%Y-%m-%d')
exp_name_full = f'{current_date}_{args.dataset}_{args.exp_name}'
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)


# --------------------------------------------------------------------- #
#                            Aggregation                               #
# --------------------------------------------------------------------- #


if args.aggregate:
    fold_results = []
    for fold_idx in range(3):
        p = os.path.join(exp_dir, f'results_fold{fold_idx}.json')
        if not os.path.exists(p):
            print(f'Missing {p}, skipping.')
            continue
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
        m = aggregated.get(f'{key}_mean')
        s = aggregated.get(f'{key}_std')
        if m is not None:
            print(f'  {key}: {m:.4f} +/- {s:.4f}')
    for K in range(1, 6):
        for tag in ('best_val_acc', 'best_test_acc'):
            key = f'{tag}_k{K}'
            m = aggregated.get(f'{key}_mean')
            s = aggregated.get(f'{key}_std')
            if m is not None:
                print(f'  {key}: {m:.4f} +/- {s:.4f}')
    sys.exit(0)


# --------------------------------------------------------------------- #
#                                Data                                  #
# --------------------------------------------------------------------- #


print(f'Device: {device}')
print(f'Experiment: {exp_name_full}')
print(f'Fold: {args.fold}')

root_dir = '/files1/haodong/data/processed_dalia_activity'
dataset_cfg = DaliaHAR()
modalities = dataset_cfg.modalities
num_m = len(modalities)
num_classes = dataset_cfg.num_classes

if args.fold is None:
    raise ValueError('--fold required (single-fold training script).')
fold_cfg = dataset_cfg.folds[args.fold]
train_subjects, eval_subjects = fold_cfg['train_set'], fold_cfg['eval_set']
val_subjects = dataset_cfg.val_set

input_length = ((dataset_cfg.duration * dataset_cfg.base_sample_rate) // 2
                if args.transform == 'sax'
                else dataset_cfg.duration * dataset_cfg.base_sample_rate)

if args.use_val_split:
    train_ds_raw = HARDataset(root_dir, modalities, val_subjects,
                              dataset_cfg, args.transform)
    print(f'NOTE: --use_val_split=True. Selector trains on VAL subjects '
          f'(N={len(train_ds_raw)}) using val-Shapley target.')
else:
    train_ds_raw = HARDataset(root_dir, modalities, train_subjects,
                              dataset_cfg, args.transform)
val_ds = HARDataset(root_dir, modalities, val_subjects,
                    dataset_cfg, args.transform)
eval_ds = HARDataset(root_dir, modalities, eval_subjects,
                     dataset_cfg, args.transform)

# Load precomputed Shapley targets
phi_pack = np.load(args.shapley_train_npz, allow_pickle=True)
phi_train = phi_pack['phi'].astype(np.float32)              # [N, M]
phi_modalities = list(phi_pack['modalities']) if 'modalities' in phi_pack.files else None
if phi_modalities is not None and phi_modalities != modalities:
    raise ValueError(
        f'modality order mismatch: shapley npz {phi_modalities} vs '
        f'dataset cfg {modalities}')
print(f'Loaded Shapley train Phi:  shape={phi_train.shape}  '
      f'mean={phi_train.mean():.4f}  std={phi_train.std():.4f}')

train_ds = HARDatasetWithPhi(train_ds_raw, phi_train)

# Optionally augment train set with val subjects + val-Phi (more subject
# diversity in selector training, tests "cross-subject generalization" angle).
# Note: val_ds is still used for per-K ckpt selection, but now overlaps with
# the training data — best_val_acc becomes optimistic. Test eval is the
# meaningful number here.
if args.combine_train_val:
    val_phi_path = args.shapley_train_npz.replace(
        'shapley_train_', 'shapley_val_'
    ).replace('shapley_val_val', 'shapley_val_')
    if not os.path.exists(val_phi_path):
        raise FileNotFoundError(
            f'--combine_train_val requires val Phi npz at {val_phi_path}')
    val_phi_pack = np.load(val_phi_path, allow_pickle=True)
    val_phi = val_phi_pack['phi'].astype(np.float32)
    val_ds_with_phi = HARDatasetWithPhi(val_ds, val_phi)
    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset([train_ds, val_ds_with_phi])
    print(f'NOTE: --combine_train_val=True. Selector trains on train+val '
          f'subjects (N={len(train_ds)}) using concat(train_Phi, val_Phi).')

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, drop_last=False)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, drop_last=False)

print(f'  train={len(train_ds)}, val={len(val_ds)}, eval={len(eval_ds)}')


# --------------------------------------------------------------------- #
#                                Model                                 #
# --------------------------------------------------------------------- #


v6_cfg = _DaliaV6Cfg(dataset_cfg)
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
    lambda_probe=args.lambda_probe, lambda_diversity=args.lambda_diversity,
    lambda_reinforce=0.0, lambda_sparsity=args.lambda_sparsity,
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
    selector_downsample_mode=args.selector_downsample_mode,
    selector_input_source=args.selector_input_source,
    use_multi_pool=args.use_multi_pool,
    use_tx_summarizer=args.use_tx_summarizer,
    tx_d_model=args.tx_d_model,
    tx_nhead=args.tx_nhead,
    tx_layers=args.tx_layers,
    tx_dim_ff=args.tx_dim_ff,
    tx_dropout=args.tx_dropout,
)
inner.top_k = args.target_k

# Enable per-sample top-K if requested.  The selector's KL distillation
# loss already trains per-sample W against per-sample softmax(phi/tau),
# so this only changes (a) which mask the fusion sees during forward
# and (b) the val_acc-per-K used to pick the best per-K checkpoint.
if args.per_sample_topk and inner.modality_selector is not None:
    inner.modality_selector.per_sample_topk = True
    print('[selector] per_sample_topk=True (per-sample top-K mask)')

# Load frozen fusion (state was saved from FusionOnlyDynaWrapper so strip
# the leading "model." prefix).
state = torch.load(args.frozen_ckpt, map_location='cpu', weights_only=False)
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k, v in state.items()
             if k.startswith('model.')}
missing, unexpected = inner.load_state_dict(state, strict=False)
fusion_loaded = len([k for k in inner.state_dict().keys() if k not in missing])
print(f'Loaded {args.frozen_ckpt}')
print(f'  state keys missing  : {len(missing)} '
      f'(expected: selector keys not in dyna ckpt)')
print(f'  state keys unexpect.: {len(unexpected)}')
print(f'  fusion params loaded: {fusion_loaded} keys')

# Freeze everything except modality_selector
for name, p in inner.named_parameters():
    if 'modality_selector' in name:
        p.requires_grad = True
    else:
        p.requires_grad = False
n_train = sum(p.numel() for p in inner.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in inner.parameters())
print(f'Trainable params: {n_train:,} / {n_total:,} '
      f'({100.0 * n_train / n_total:.2f}%)')

model = V6ShapleyDistillWrapper(inner, modalities, dataset_cfg.variates,
                                target_k=args.target_k, tau=args.tau).to(device).float()

trainable = [p for p in model.parameters() if p.requires_grad]
if args.weight_decay > 0:
    optimizer = optim.AdamW(trainable, lr=args.lr,
                            weight_decay=args.weight_decay)
    print(f'[opt] AdamW lr={args.lr} wd={args.weight_decay}')
else:
    optimizer = optim.Adam(trainable, lr=args.lr)
    print(f'[opt] Adam lr={args.lr} (wd=0)')
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


writer = SummaryWriter(
    comment=f'_{exp_name_full}_fold{args.fold}')

for epoch in range(args.num_epochs):
    t0 = time.time()
    train_loss, train_kl = train_one_epoch(
        train_loader, model, optimizer, epoch, device,
        lambda_kl=args.lambda_kl, lambda_ce=args.lambda_ce,
        lambda_ce_soft=args.lambda_ce_soft,
        lambda_probe=args.lambda_probe,
        lambda_diversity=args.lambda_diversity,
        lambda_sparsity=args.lambda_sparsity, criterion=criterion,
        clip_grad=args.clip_grad,
        lambda_listmle=args.lambda_listmle)

    # Per-K val
    epoch_k = {}
    for k_ in range(1, num_m + 1):
        inner_ref.top_k = k_
        v_loss, v_acc, v_y, v_p = evaluate_one_epoch(
            val_loader, model, criterion, epoch, device,
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
        eval_loader, model, criterion, 0, device,
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

# Back-compat top-line metric: report K = target_k
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
    'num_epochs': args.num_epochs, 'batch_size': args.batch_size,
    'lr': args.lr, 'lr_schedule': args.lr_schedule,
    'tau': args.tau, 'lambda_kl': args.lambda_kl,
    'lambda_ce': args.lambda_ce, 'lambda_ce_soft': args.lambda_ce_soft,
    'lambda_probe': args.lambda_probe,
    'selector_downsample_factor': args.selector_downsample_factor,
    'frozen_ckpt': args.frozen_ckpt,
    'shapley_train_npz': args.shapley_train_npz,
    'seed': args.seed_num, 'transform': args.transform,
    'per_sample_topk': args.per_sample_topk,
    'modalities': modalities,
    'train_subjects': train_subjects, 'val_subjects': val_subjects,
    'eval_subjects': eval_subjects,
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
