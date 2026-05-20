"""DSADS selector fine-tuning with **C3: Conformal Adaptive-K Routing
via Shapley Quantile Distillation**.  DSADS sibling of the DaliaHAR
script — same wrapper / losses / per-K bookkeeping, only the data
loading differs (DSADS scenarios via ``.pt`` files instead of
subject-fold HARDataset).
"""
from __future__ import annotations

import argparse
import json
import math
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed, count_model_parameters
from utils.dataset_cfg import DSADS
from data.dataset_builder import DSADSDataset
from utils.train_utils import AverageMeter, ProgressMeter
from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)


class _DSADSV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


def pinball_loss(pred, target, tau):
    diff = target - pred
    return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()


class V6ShapleyC3Wrapper(nn.Module):
    def __init__(self, v6_model, modalities, variates, target_k,
                 tau_kl=0.5, tau_lo=0.1, tau_hi=0.9,
                 q_hidden_dim=64, uniform_dim=256):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self.variates = variates
        self.target_k = int(target_k)
        self.tau_kl = float(tau_kl)
        self.tau_lo = float(tau_lo)
        self.tau_hi = float(tau_hi)
        self.M = len(self.modalities)
        self._mod_slices = {}
        offset = 0
        for m in modalities:
            n = variates[m]
            self._mod_slices[m] = (offset, offset + n)
            offset += n
        self.q_heads_lo = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(uniform_dim, q_hidden_dim), nn.ReLU(),
                              nn.Linear(q_hidden_dim, 1))
            for m in modalities})
        self.q_heads_hi = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(uniform_dim, q_hidden_dim), nn.ReLU(),
                              nn.Linear(q_hidden_dim, 1))
            for m in modalities})

    def _split(self, x):
        high, low = {}, {}
        for m in self.modalities:
            s, e = self._mod_slices[m]
            high[m] = x[:, :, s:e]
            low[m] = x[:, :, s:e]
        return high, low

    @staticmethod
    def shapley_kl(W, phi, tau):
        log_W = torch.log(W.clamp(min=1e-8))
        p = F.softmax(phi / tau, dim=-1)
        return -(p * log_W).sum(dim=-1).mean()

    def _quantile_heads(self, projected, training):
        B = list(projected.values())[0].shape[0]
        device = list(projected.values())[0].device
        q_lo = torch.zeros(B, self.M, device=device)
        q_hi = torch.zeros(B, self.M, device=device)
        for j, m in enumerate(self.modalities):
            if m not in projected:
                continue
            feat = projected[m]
            q_lo[:, j] = self.q_heads_lo[m](feat).squeeze(-1)
            q_hi[:, j] = self.q_heads_hi[m](feat).squeeze(-1)
        return q_lo, q_hi

    def forward(self, x, phi=None, training=False, labels=None):
        high, low = self._split(x)
        result = self.model(high, low, training=training,
                            return_selection_info=True, labels=labels)
        if len(result) == 5:
            logits, _, W, _, aux = result
        else:
            logits, _, W, _ = result
            aux = {}
        projected = getattr(self.model.modality_selector, '_projected', {})
        q_lo, q_hi = self._quantile_heads(projected, training)
        aux['q_lo'] = q_lo
        aux['q_hi'] = q_hi
        if training and phi is not None:
            aux['shapley_kl'] = self.shapley_kl(W, phi, self.tau_kl)
            aux['pinball_lo'] = pinball_loss(q_lo, phi, self.tau_lo)
            aux['pinball_hi'] = pinball_loss(q_hi, phi, self.tau_hi)
            pl = getattr(self.model.modality_selector, '_probe_logits', {})
            if pl and labels is not None:
                stacked = torch.stack(
                    [pl[m] for m in self.modalities if m in pl], dim=1)
                w_sub = W[:, :stacked.shape[1]]
                soft_logits = (w_sub.unsqueeze(-1) * stacked).sum(dim=1)
                aux['ce_soft'] = F.cross_entropy(soft_logits, labels)
        return logits, W, aux

    @torch.no_grad()
    def adaptive_subset(self, x, rule='upper', threshold=0.0,
                        min_k=1, max_k=None):
        max_k = max_k if max_k is not None else self.M
        high, low = self._split(x)
        _ = self.model(high, low, training=False, return_selection_info=True, labels=None)
        projected = self.model.modality_selector._projected
        q_lo, q_hi = self._quantile_heads(projected, training=False)
        scores = q_hi if rule == 'upper' else q_lo
        B = x.shape[0]
        subsets = []
        for i in range(B):
            sel = (scores[i] > threshold).nonzero(as_tuple=True)[0].tolist()
            if len(sel) < min_k:
                topm = torch.topk(q_hi[i], min_k).indices.tolist()
                sel = sorted(topm)
            if len(sel) > max_k:
                upper_vals = q_hi[i, sel]
                keep_local = torch.topk(upper_vals, max_k).indices.tolist()
                sel = sorted(sel[k] for k in keep_local)
            subsets.append(sel)
        return subsets


class DSADSDatasetWithPhi(Dataset):
    def __init__(self, ds, phi):
        assert len(ds) == phi.shape[0]
        self.ds = ds
        self.phi = torch.from_numpy(phi).float()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, y, self.phi[idx]


def train_one_epoch(loader, model, optimizer, epoch, device, criterion,
                    lambda_kl, lambda_q, lambda_ce_soft, lambda_probe,
                    clip_grad=1.0):
    loss_meter = AverageMeter('Loss', ':.4f')
    kl_meter = AverageMeter('KL', ':.4f')
    pin_meter = AverageMeter('Pin', ':.4f')
    ces_meter = AverageMeter('CEs', ':.4f')
    probe_meter = AverageMeter('Aux', ':.4f')
    model.train()
    for i, (x, y, phi) in enumerate(loader):
        x = x.to(device).float()
        y = y.to(device)
        phi = phi.to(device).float()
        _, W, aux = model(x, phi=phi, training=True, labels=y)
        kl = aux.get('shapley_kl', torch.tensor(0.0, device=device))
        pin_lo = aux.get('pinball_lo', torch.tensor(0.0, device=device))
        pin_hi = aux.get('pinball_hi', torch.tensor(0.0, device=device))
        pinball = pin_lo + pin_hi
        ce_soft = aux.get('ce_soft', torch.tensor(0.0, device=device))
        if not isinstance(ce_soft, torch.Tensor):
            ce_soft = torch.tensor(float(ce_soft), device=device)
        probe = aux.get('probe_loss', torch.tensor(0.0, device=device))
        if not isinstance(probe, torch.Tensor):
            probe = torch.tensor(float(probe), device=device)
        loss = (lambda_kl * kl + lambda_q * pinball
                + lambda_ce_soft * ce_soft + lambda_probe * probe)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], clip_grad)
        optimizer.step()
        loss_meter.update(loss.item(), x.size(0))
        kl_meter.update(kl.item(), x.size(0))
        pin_meter.update(pinball.item(), x.size(0))
        ces_meter.update(ce_soft.item(), x.size(0))
        probe_meter.update(probe.item(), x.size(0))
        if (i % 50 == 0) or (i == len(loader) - 1):
            progress = ProgressMeter(
                len(loader),
                [loss_meter, kl_meter, pin_meter, ces_meter, probe_meter],
                prefix=f'Epoch: [{epoch}]')
            progress.display(i)
    print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
          f'KL: {kl_meter.avg:.4f}  Pin: {pin_meter.avg:.4f}  '
          f'CEs: {ces_meter.avg:.4f}  Probe: {probe_meter.avg:.4f}')
    return loss_meter.avg, pin_meter.avg


@torch.no_grad()
def evaluate_fixed_k(loader, model, criterion, device, label='', verbose=True):
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


@torch.no_grad()
def evaluate_adaptive(loader, model, criterion, device, modalities,
                      rule='upper', threshold=0.0, min_k=1, max_k=None,
                      label='adaptive', verbose=True):
    model.eval()
    all_preds, all_labels = [], []
    ks_used = []
    inner = model.model
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device)
        subsets = model.adaptive_subset(x, rule=rule, threshold=threshold,
                                        min_k=min_k, max_k=max_k)
        groups = {}
        for i, S in enumerate(subsets):
            key = tuple(sorted(S))
            groups.setdefault(key, []).append(i)
        logits_buf = torch.zeros(x.shape[0], inner.regressor[-1].out_features,
                                 device=device)
        high, low = model._split(x)
        for S, idxs in groups.items():
            keep_modals = [modalities[j] for j in S] if S else [modalities[0]]
            sub_high = {m: high[m][idxs] for m in keep_modals}
            sub_low = {m: low[m][idxs] for m in keep_modals}
            out = inner(sub_high, sub_low, training=False,
                        return_selection_info=False)
            sub_logits = out[0] if isinstance(out, tuple) else out
            logits_buf[idxs] = sub_logits
            ks_used.extend([len(S)] * len(idxs))
        _, predicted = torch.max(logits_buf, 1)
        all_preds.append(predicted.cpu())
        all_labels.append(y.cpu())
    y_arr = torch.cat(all_labels).numpy()
    p_arr = torch.cat(all_preds).numpy()
    acc = float((p_arr == y_arr).mean())
    f1 = float(f1_score(y_arr, p_arr, average='macro'))
    mean_k = float(np.mean(ks_used))
    if verbose:
        print(f'Adaptive[{label}]  Acc={acc:.4f}  F1={f1:.4f}  meanK={mean_k:.2f}')
    return acc, f1, mean_k, y_arr, p_arr


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
parser.add_argument('--exp_name', default='v6_shap_c3', type=str)
parser.add_argument('--results_dir',
                    default='./model_chkpt/DSADS/selector/', type=str)
parser.add_argument('--dataset', default='DSADS', type=str)
parser.add_argument('--data_root', default='/files1/haodong/data/DSADS/', type=str)

parser.add_argument('--frozen_ckpt', required=True, type=str)
parser.add_argument('--shapley_train_npz', required=True, type=str)
parser.add_argument('--use_val_split', default=False, type=_str2bool,
                    help='Train selector on val split (with val-Shapley npz '
                         'in --shapley_train_npz).  Same val-Shap recipe as '
                         'IEMOCAP.')

parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2, 3])
parser.add_argument('--aggregate', action='store_true', default=False)
parser.add_argument('--val_split_ratio', default=0.2, type=float)

parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine'])
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--transform', default='sax', type=str)

parser.add_argument('--tau_lo', default=0.1, type=float)
parser.add_argument('--tau_hi', default=0.9, type=float)
parser.add_argument('--lambda_q', default=1.0, type=float)
parser.add_argument('--c3_select_rule', default='upper',
                    choices=['upper', 'lower'])
parser.add_argument('--c3_threshold', default=0.0, type=float)
parser.add_argument('--min_k', default=1, type=int)
parser.add_argument('--max_k', default=5, type=int)
parser.add_argument('--q_hidden_dim', default=64, type=int)

parser.add_argument('--tau', default=0.5, type=float)
parser.add_argument('--lambda_kl', default=0.0, type=float)
parser.add_argument('--lambda_ce_soft', default=0.0, type=float)
parser.add_argument('--lambda_probe', default=0.05, type=float)

parser.add_argument('--target_k', default=4, type=int)
parser.add_argument('--use_interaction_matrix', default=False, type=_str2bool)
parser.add_argument('--use_holo_bias', default=False, type=_str2bool)
parser.add_argument('--holo_scale', default=1.0, type=float)
parser.add_argument('--selector_downsample_factor', default=2, type=int,
                    help='DSADS default 2 (T=125->63); set 1 to disable.')

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

args = parser.parse_args()

device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed_num)
current_date = datetime.now().strftime('%Y-%m-%d')
exp_name_full = f'{current_date}_{args.dataset}_{args.exp_name}'
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)


# --- Aggregation ---
if args.aggregate:
    fold_results = []
    for fold_idx in range(4):
        p = os.path.join(exp_dir, f'results_fold{fold_idx}.json')
        if not os.path.exists(p):
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
    for key in ('adaptive_test_acc','adaptive_test_f1','adaptive_mean_k',
                'best_test_acc','best_test_f1'):
        m = aggregated.get(f'{key}_mean')
        s = aggregated.get(f'{key}_std')
        if m is not None:
            print(f'  {key}: {m:.4f} +/- {s:.4f}')
    for K in range(1, 6):
        for tag in ('best_val_acc', 'best_test_acc'):
            key = f'{tag}_k{K}'
            m = aggregated.get(f'{key}_mean')
            if m is not None:
                s = aggregated.get(f'{key}_std', 0.0)
                print(f'  {key}: {m:.4f} +/- {s:.4f}')
    sys.exit(0)


# --- Data ---
dataset_cfg = DSADS()
modalities = dataset_cfg.modalities
num_m = len(modalities)
num_classes = dataset_cfg.num_classes

if args.fold is None:
    raise ValueError('--fold required.')
scenario_idx = args.fold
src_path = os.path.join(args.data_root,
                        f'dsad_diversify_dict_scenario{scenario_idx}_src.pt')
trg_path = os.path.join(args.data_root,
                        f'dsad_diversify_dict_scenario{scenario_idx}_trg.pt')
train_data = torch.load(src_path, weights_only=False)
test_data = torch.load(trg_path, weights_only=False)
tr_x, va_x, tr_y, va_y = train_test_split(
    train_data['samples'], train_data['labels'],
    test_size=args.val_split_ratio, random_state=args.seed_num,
    stratify=train_data['labels'])
train_split = {'samples': tr_x, 'labels': tr_y}
val_split = {'samples': va_x, 'labels': va_y}
input_length = dataset_cfg.duration * dataset_cfg.base_sample_rate

if args.use_val_split:
    train_ds_raw = DSADSDataset(val_split, modalities, dataset_cfg, args.transform)
    print(f'NOTE: --use_val_split=True. Selector trains on VAL split '
          f'(N={len(train_ds_raw)}) using val-Shapley target.')
else:
    train_ds_raw = DSADSDataset(train_split, modalities, dataset_cfg, args.transform)
val_ds = DSADSDataset(val_split, modalities, dataset_cfg, args.transform)
eval_ds = DSADSDataset(test_data, modalities, dataset_cfg, args.transform)

phi_pack = np.load(args.shapley_train_npz, allow_pickle=True)
phi_train = phi_pack['phi'].astype(np.float32)
print(f'Loaded Shapley train Phi:  shape={phi_train.shape}  '
      f'mean={phi_train.mean():.4f}  std={phi_train.std():.4f}')

train_ds = DSADSDatasetWithPhi(train_ds_raw, phi_train)
train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, drop_last=False)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, drop_last=False)
print(f'Device: {device}  Fold={args.fold}  '
      f'train={len(train_ds_raw)}, val={len(val_ds)}, eval={len(eval_ds)}')


# --- Model ---
v6_cfg = _DSADSV6Cfg(dataset_cfg)
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
    use_batched_fusion=True, per_modal_distill=False,
    per_modal_downsample_min_len=args.downsample_min_len,
    use_interaction_matrix=args.use_interaction_matrix,
    use_holo_bias=args.use_holo_bias, holo_scale=args.holo_scale,
    selector_downsample_factor=args.selector_downsample_factor,
)
inner.top_k = args.target_k
state = torch.load(args.frozen_ckpt, map_location='cpu', weights_only=False)
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k, v in state.items()
             if k.startswith('model.')}
missing, unexpected = inner.load_state_dict(state, strict=False)
print(f'Loaded {args.frozen_ckpt}  '
      f'(missing={len(missing)}, unexpected={len(unexpected)})')

for name, p in inner.named_parameters():
    p.requires_grad = ('modality_selector' in name)

model = V6ShapleyC3Wrapper(inner, modalities, dataset_cfg.variates,
                            target_k=args.target_k,
                            tau_kl=args.tau, tau_lo=args.tau_lo, tau_hi=args.tau_hi,
                            q_hidden_dim=args.q_hidden_dim).to(device).float()
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f'Trainable: {n_train:,}/{n_total:,} ({100*n_train/n_total:.2f}%)')

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                       lr=args.lr)
criterion = nn.CrossEntropyLoss()
if args.lr_schedule == 'cosine':
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
else:
    scheduler = None


# --- Per-K best ckpt + adaptive ---
best_per_k = {k: {'val_acc': -1.0, 'val_f1': 0.0, 'val_loss': float('inf'),
                  'epoch': -1, 'state': None}
              for k in range(1, num_m + 1)}
best_adaptive = {'val_acc': -1.0, 'val_f1': 0.0, 'mean_k': 0.0,
                 'epoch': -1, 'state': None}
inner_ref = model.model
saved_target_k = inner_ref.top_k


def _snapshot_state(m):
    return {kk: vv.detach().cpu().clone() for kk, vv in m.state_dict().items()}


writer = SummaryWriter(comment=f'_{exp_name_full}_fold{args.fold}')
for epoch in range(args.num_epochs):
    t0 = time.time()
    train_loss, train_pin = train_one_epoch(
        train_loader, model, optimizer, epoch, device, criterion,
        lambda_kl=args.lambda_kl, lambda_q=args.lambda_q,
        lambda_ce_soft=args.lambda_ce_soft, lambda_probe=args.lambda_probe,
        clip_grad=args.clip_grad)

    epoch_k = {}
    for k_ in range(1, num_m + 1):
        inner_ref.top_k = k_
        v_loss, v_acc, v_y, v_p = evaluate_fixed_k(
            val_loader, model, criterion, device,
            label=f'k{k_}', verbose=False)
        v_f1 = f1_score(v_y, v_p, average='macro')
        epoch_k[k_] = (v_loss, v_acc, v_f1)
        if v_acc > best_per_k[k_]['val_acc']:
            best_per_k[k_] = {
                'val_acc': float(v_acc), 'val_f1': float(v_f1),
                'val_loss': float(v_loss), 'epoch': int(epoch),
                'state': _snapshot_state(model),
            }
        writer.add_scalar(f'val_k{k_}/acc', v_acc, epoch)
    inner_ref.top_k = saved_target_k

    a_acc, a_f1, a_mean_k, _, _ = evaluate_adaptive(
        val_loader, model, criterion, device, modalities,
        rule=args.c3_select_rule, threshold=args.c3_threshold,
        min_k=args.min_k, max_k=args.max_k, label='adaptive', verbose=False)
    if a_acc > best_adaptive['val_acc']:
        best_adaptive = {
            'val_acc': float(a_acc), 'val_f1': float(a_f1),
            'mean_k': float(a_mean_k), 'epoch': int(epoch),
            'state': _snapshot_state(model),
        }
    writer.add_scalar('val_adaptive/acc', a_acc, epoch)
    writer.add_scalar('val_adaptive/mean_k', a_mean_k, epoch)

    if scheduler is not None:
        scheduler.step()

    k_summary = '  '.join(f'k{k_}={epoch_k[k_][1]:.4f}'
                          for k_ in range(1, num_m + 1))
    print(f'End of Epoch {epoch} | Val per K: {k_summary} | '
          f'adapt={a_acc:.4f} K̄={a_mean_k:.2f}  ({time.time()-t0:.1f}s)')


# --- Test eval ---
print('\n' + '=' * 80)
print('Test eval — per-K + adaptive-K')
print('=' * 80)

test_per_k = {}
for k_ in range(1, num_m + 1):
    entry = best_per_k[k_]
    if entry['state'] is None:
        continue
    model.load_state_dict({kk: vv.to(device) for kk, vv in entry['state'].items()})
    inner_ref.top_k = k_
    _, t_acc, t_y, t_p = evaluate_fixed_k(
        eval_loader, model, criterion, device,
        label=f'test_k{k_}', verbose=False)
    t_f1 = f1_score(t_y, t_p, average='macro')
    test_per_k[k_] = {'test_acc': float(t_acc), 'test_f1': float(t_f1)}
    print(f'  K={k_}: best_ep={entry["epoch"]:>3d}  '
          f'val={entry["val_acc"]:.4f}  test={t_acc:.4f}/{t_f1:.4f}')

entry = best_adaptive
if entry['state'] is not None:
    model.load_state_dict({kk: vv.to(device) for kk, vv in entry['state'].items()})
    inner_ref.top_k = saved_target_k
    a_acc, a_f1, a_mean_k, _, _ = evaluate_adaptive(
        eval_loader, model, criterion, device, modalities,
        rule=args.c3_select_rule, threshold=args.c3_threshold,
        min_k=args.min_k, max_k=args.max_k, label='test_adaptive', verbose=False)
    print(f'  ADAPT: best_ep={entry["epoch"]:>3d}  '
          f'val_acc={entry["val_acc"]:.4f}  '
          f'test_acc={a_acc:.4f}/{a_f1:.4f}  meanK={a_mean_k:.2f}')
    adaptive_test = {'test_acc': float(a_acc), 'test_f1': float(a_f1),
                     'mean_k': float(a_mean_k)}
else:
    adaptive_test = None
writer.close()


# --- Save ---
config = vars(args)
with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4, default=str)

_tk = args.target_k if args.target_k in test_per_k else max(test_per_k.keys())
model_stats = {
    'experiment_name': exp_name_full, 'fold': args.fold,
    'best_val_acc': best_per_k[_tk]['val_acc'],
    'best_val_f1': best_per_k[_tk]['val_f1'],
    'best_epoch_val': best_per_k[_tk]['epoch'],
    'best_test_acc': float(test_per_k[_tk]['test_acc']),
    'best_test_f1': float(test_per_k[_tk]['test_f1']),
}
for k_ in range(1, num_m + 1):
    if k_ in test_per_k:
        entry = best_per_k[k_]
        model_stats[f'best_val_acc_k{k_}'] = float(entry['val_acc'])
        model_stats[f'best_val_f1_k{k_}'] = float(entry['val_f1'])
        model_stats[f'best_epoch_val_k{k_}'] = int(entry['epoch'])
        model_stats[f'best_test_acc_k{k_}'] = float(test_per_k[k_]['test_acc'])
        model_stats[f'best_test_f1_k{k_}'] = float(test_per_k[k_]['test_f1'])
if adaptive_test is not None:
    model_stats['adaptive_val_acc'] = float(best_adaptive['val_acc'])
    model_stats['adaptive_val_f1'] = float(best_adaptive['val_f1'])
    model_stats['adaptive_val_mean_k'] = float(best_adaptive['mean_k'])
    model_stats['adaptive_test_acc'] = float(adaptive_test['test_acc'])
    model_stats['adaptive_test_f1'] = float(adaptive_test['test_f1'])
    model_stats['adaptive_mean_k'] = float(adaptive_test['mean_k'])

results_filename = (f'results_fold{args.fold}.json'
                    if args.fold is not None else 'results.json')
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(model_stats, f, indent=4)
print(f'\nSaved -> {os.path.join(exp_dir, results_filename)}')
