"""seqA — sequential gated-bottleneck fusion + modality early-exit, unified trainer.

ONE trainer for all 6 datasets via ``--dataset`` (iemocap / daliahar / pamap2 /
dsads / eav / mosei).  Replaces the six per-dataset scripts under
``training/seqA/main_v6_{...}_seqfusion_late_fusion.py`` (wesad and the
``main_v6_mosei_mmsa_bert.py`` BERT variant are intentionally dropped).

The CORE model (``models/seqA.DualVideoBottleneckModelV6Downsample``) does NOT
consume the concatenated-x ``ee_data`` loader.  It takes per-modality dicts
``high = {m: (B, T, C_m)}`` and ``low = {m: ...}`` (two resolution views) and is
called ``model(high, low, training=..., ...)``.  Each of the six original
trainers built its own dataset loader and converted every batch into
``(high, low, y)``.  ``build_data(args)`` reproduces THAT per-dataset
loader + ``(high, low, y)`` conversion behind one dispatch, then a single
shared train/eval loop runs the IEMOCAP recipe on the preloaded batches:

  * sequential fusion (``fusion_mode='sequential'`` -> 方案 A: process modalities
    one at a time, carry a gated per-layer bottleneck);
  * per-layer distillation backbone (``n_fusion_distill``, ``downsample_min_len``);
  * val-LOO modality ordering for dual-order model selection (method A = val-LOO
    order, method B = random-order average), plus a "cold" canonical-val fallback;
  * best-VAL model selection, saved ``best_model_{loo,rand,cold}_*`` checkpoints,
    and a ``results_fold{fold}.json`` with best_val_acc / best_test_acc /
    best_test_f1.

Two batch families are unified inside ``build_data``:
  * float / interleaved (iemocap, eav, mosei): the loader yields an interleaved
    list ``[high0, low0, ..., high_{M-1}, low_{M-1}, label]``; high[m]=feats[2j],
    low[m]=feats[2j+1] (a genuine time-compressed low view), y=label.long().
  * HAR / SAX (daliahar, pamap2, dsads): the loader yields a single concatenated
    ``(B, T, sum_C)`` tensor split per-modality via a channel map; low == high
    (single resolution), y direct.

Run:
    python training/seqA.py --dataset iemocap --fold 0 --results_dir ./results_seqA
    python training/seqA.py --dataset mosei   --fold 0 --results_dir ./results_seqA
    python training/seqA.py --dataset daliahar --fold 0 --word_length 2 --results_dir ./results_seqA
"""
from __future__ import annotations

import argparse
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

warnings.filterwarnings("ignore")
import copy

# Repo root on sys.path so absolute imports resolve regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from multimodal_model.seqA import DualVideoBottleneckModelV6Downsample
from utils.helper_function import set_seed, count_model_parameters
from utils.train_utils import (
    AverageMeter, ProgressMeter, FocalLoss,
    get_modality_curriculum, ModalityGradientProfiler,
)
from utils.dual_order_val import dual_val, eval_full_order, rand_avg_eval


# ============================================================================
# Shared cfg shim + wrapper
# ============================================================================

class _Cfg:
    """Minimal cfg object exposing the fields the seqA model reads from ``cfg``:
    ``modalities`` and ``variates``.  ``video_high_dim`` / ``video_low_dim`` are
    passed separately to the constructor."""

    def __init__(self, modalities, variates):
        self.modalities = list(modalities)
        self.variates = dict(variates)


class SeqAWrapper(nn.Module):
    """Apply per-modality independent Bernoulli modality-dropout during training,
    then call the inner seqA model on the per-modality (high, low) dicts.

    Faithful merge of the per-dataset ``V6*FusionDynaWrapper.forward`` dropout
    logic.  The per-dataset batch -> (high, low, y) conversion is done up-front in
    ``build_data`` (so this wrapper is dataset-agnostic and consumes dicts
    directly).  Optional MRdIB Phase-1 singleton heads (``--use_singleton_heads`` +
    ``--alpha_U``) are preserved.
    """

    def __init__(self, inner, modalities, use_singleton_heads=False,
                 num_classes=None, d_model=None):
        super().__init__()
        self.model = inner
        self.modalities = list(modalities)
        self.use_singleton_heads = use_singleton_heads
        if use_singleton_heads:
            assert num_classes is not None and d_model is not None
            self.singleton_heads = nn.ModuleDict({
                m: nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
                for m in self.modalities
            })

    def forward(self, high, low, modality_dropout_probs=None, training=False,
                return_prefix_logits=False):
        if training and modality_dropout_probs is not None:
            # Per-modality independent Bernoulli, rejection-resampled to guarantee
            # at least one survivor (matches fusion-only "actually delete" semantics).
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
        logits = out[0] if isinstance(out, tuple) else out
        if training and self.use_singleton_heads:
            processed = self.model._last_processed_modalities
            singleton_logits = {
                m: self.singleton_heads[m](feat.mean(dim=1))
                for m, feat in processed.items()
            }
            return logits, singleton_logits
        return logits


# ============================================================================
# Per-dataset data + model build
# ============================================================================

def _make_model(args, modalities, variates, num_classes, input_length,
                video_high_dim, video_low_dim, device):
    """Build the inner seqA model + wrapper with the shared (IEMOCAP) recipe."""
    cfg = _Cfg(modalities, variates)
    inner = DualVideoBottleneckModelV6Downsample(
        head_mode=args.head_mode,
        cfg=cfg,
        output_dim=num_classes,
        input_length=input_length,
        d_model=args.d_model, nhead=args.nhead,
        num_layers_per_modal=args.num_layers_per_modal,
        num_layers=args.num_layers, dropout=args.dropout, verbose=True,
        video_low_dim=video_low_dim, video_high_dim=video_high_dim,
        use_bottleneck=True, n_bottlenecks=args.n_bottlenecks,
        fusion_layer=args.fusion_layer, use_sparse_moe=False,
        num_experts=args.num_experts, expert_k=1,
        internal_dim=args.internal_dim, bottleneck_head_pos=True,
        use_sparse_attn=args.use_sparse_attn, factor=args.base_factor,
        selector_video_source='high', encoder_video_source='high',
        no_selector=True,                       # fusion-only (seqA), no selector
        use_weighted_factor=False, use_triton=False,
        num_classes=num_classes,
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
        # sequential-fusion knobs (this is the seqA variant)
        fusion_mode=args.fusion_mode,
        seq_depth_flow=args.seq_depth_flow,
        seq_modality_order=args.seq_modality_order,
        seq_random_order=args.seq_random_order,
        seq_agg_mode=args.seq_agg_mode,
    )
    model = SeqAWrapper(
        inner, modalities,
        use_singleton_heads=args.use_singleton_heads,
        num_classes=num_classes, d_model=args.d_model,
    )
    return model.to(device).float()


def _preload_interleaved(loader, modalities):
    """Float datasets: loader batch = [high0, low0, ..., label]. Build per-modality
    high/low dicts; low is the genuine time-compressed view (feats[2j+1])."""
    out = []
    for batch in loader:
        *feats, label = batch
        high = {m: feats[2 * j].clone() for j, m in enumerate(modalities)}
        low = {m: feats[2 * j + 1].clone() for j, m in enumerate(modalities)}
        out.append((high, low, label.squeeze(-1).long().clone()))
    return out


def _preload_concat(loader, mod_slices):
    """HAR datasets: loader batch = (x [B,T,sum_C], y). Channel-split into per-mod
    high dict; low aliases high (single resolution, no time compression)."""
    out = []
    for x, y in loader:
        high = {m: x[:, :, s:e].clone() for m, (s, e) in mod_slices.items()}
        out.append((high, dict(high), y.clone()))
    return out


def build_data(args, device, build_model=True):
    """Load train/val/test batches via the SHARED baseline dataloader
    (``common.ee_data.build_loaders``) — the exact pipeline every baseline in
    ``runs/baselines_d128_fair`` (and MBT) uses — then reformat the concatenated
    ``(B, T, sum_variates)`` tensor into the per-modality ``(high, low, y)`` dicts
    the seqA training loop expects.

    This makes seqA / seqA_branched consume identical, length-normalized inputs to
    the baselines: every modality is resampled/padded to ``in_len = max_seq_len``
    (done inside the dataset for CMI/CZU/IEMOCAP; in the unpack for MMFi). ``low``
    aliases ``high`` — the seqA runs here are ``no_selector`` / fusion-only and
    never read the low (time-compressed) view.

    Returns:
        (train_batches, val_batches, test_batches, mods, num_classes, model)
        each *_batches a list of (high_dict, low_dict, y) on CPU.
    """
    from common.ee_data import build_loaders

    tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)
    # 'float' = raw features (iemocap/cmi/czu/mmfi/...). 'sax' = HAR datasets
    # (daliahar/dsads/pamap2): build_loaders' sax_unpack already casts the SAX
    # integer tokens to float, so seqA/seqA_branched consume them through the same
    # Linear projectors (tokens-as-float, exactly how the original seqA trained on
    # these datasets). Anything else (true integer-embedding input) is unsupported.
    if in_mode not in ('float', 'sax'):
        raise ValueError(
            f"seqA expects float or sax(->float) features; build_loaders returned "
            f"in_mode={in_mode!r} for dataset {args.dataset!r}.")
    mods = list(cfg.modalities)
    variates = dict(cfg.variates)
    num_classes = cfg.num_classes

    # Channel boundaries (modality order) to split the concat tensor back to dicts.
    bounds, off = {}, 0
    for m in mods:
        w = variates[m]
        bounds[m] = (off, off + w)
        off += w

    def _preload(loader):
        out = []
        for batch in loader:
            x, y = unpack(batch, 'cpu')            # x: (B, T, sum_variates), length=in_len
            high = {m: x[..., a:b].contiguous() for m, (a, b) in bounds.items()}
            out.append((high, {m: high[m] for m in mods}, y.view(-1).long()))
        return out

    # The batched V6 model is only needed by the batched seqA trainer (and the
    # selector trainer). Callers that drive a different model — the branched trainer
    # and every eval/analysis script — pass build_model=False to skip constructing
    # (and printing the banner for) a model they immediately discard.
    if build_model:
        v_dim = variates.get('video', 1024)
        model = _make_model(args, mods, variates, num_classes, in_len, v_dim, v_dim, device)
    else:
        model = None
    print(f'[build_data] shared ee_data loader | dataset={args.dataset} in_mode={in_mode} '
          f'in_len={in_len} mods={mods} num_classes={num_classes}', flush=True)
    return _preload(tl), _preload(vl), _preload(el), mods, num_classes, model


def _to_dev(high, low, y, mods, device):
    h = {m: high[m].to(device, non_blocking=True).float() for m in mods}
    l = {m: low[m].to(device, non_blocking=True).float() for m in mods}
    return h, l, y.to(device)


def train_one_epoch(batches, model, criterion, optimizer, epoch, device, mods,
                    modality_dropout_probs, clip_grad=0.0, alpha_U=0.0,
                    prefix_supervision=False, prefix_ds_weight=1.0, prefix_kd_weight=1.0,
                    prefix_kd_weight_mode='const',
                    ema_model=None, ema_decay=0.999, fixed_order=None):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.train()
    for i, (high, low, y) in enumerate(batches):
        high, low, y = _to_dev(high, low, y, mods, device)
        if prefix_supervision:
            # Reorder dicts to fixed LOO order (phase-2); pass None dropout so
            # the SeqAWrapper doesn't undo the ordering by re-filtering to canonical.
            if fixed_order is not None:
                h_in = {m: high[m] for m in fixed_order if m in high}
                l_in = {m: low[m] for m in fixed_order if m in low}
                drop_p = None
            else:
                h_in, l_in, drop_p = high, low, modality_dropout_probs
            logits, prefix_logits = model(
                h_in, l_in, modality_dropout_probs=drop_p,
                training=True, return_prefix_logits=True)
            loss = criterion(logits, y)
            K = prefix_logits.shape[1]
            # k-proportional weights: later prefixes (more modalities) weighted higher
            if prefix_kd_weight_mode == 'k_prop':
                raw_w = [(j + 1) / K for j in range(K)]
                w_sum = sum(raw_w)
                ws = [w / w_sum for w in raw_w]
            else:
                ws = [1.0 / K] * K
            ds = sum(ws[j] * criterion(prefix_logits[:, j], y) for j in range(K))
            # EMA teacher: stable KD target from a frozen EMA copy of the model
            if ema_model is not None:
                with torch.no_grad():
                    ema_out = ema_model(h_in, l_in, modality_dropout_probs=None, training=False)
                    ema_logits = ema_out[0] if isinstance(ema_out, tuple) else ema_out
                teach = F.softmax(ema_logits, dim=-1)
            else:
                teach = F.softmax(logits.detach(), dim=-1)
            kd = sum(ws[j] * F.kl_div(F.log_softmax(prefix_logits[:, j], dim=-1), teach,
                                       reduction='batchmean') for j in range(K))
            loss = loss + prefix_ds_weight * ds + prefix_kd_weight * kd
        else:
            out = model(high, low, modality_dropout_probs=modality_dropout_probs,
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
        acc = predicted.eq(y).sum().item() / y.size(0)
        loss_meter.update(loss.item(), y.size(0))
        acc_meter.update(acc, y.size(0))
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        # EMA update per step (after optimizer.step so we track the updated weights)
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
        if (i % 50 == 0) or (i == len(batches) - 1):
            ProgressMeter(len(batches), [loss_meter, acc_meter],
                          prefix=f"Epoch: [{epoch}]").display(i)
            if i == len(batches) - 1:
                mean_p = (float(np.mean(list(modality_dropout_probs.values())))
                          if modality_dropout_probs else 0.0)
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
                      f'Acc: {acc_meter.avg:.4f}  meanModDrop: {mean_p:.3f}')
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_one_epoch(batches, model, criterion, epoch, device, mods):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    model.eval()
    all_preds, all_labels = [], []
    for high, low, y in batches:
        high, low, y = _to_dev(high, low, y, mods, device)
        logits = model(high, low, modality_dropout_probs=None, training=False)
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


# ============================================================================
# Argparse
# ============================================================================

def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in ('y', 'yes', 't', 'true', '1'):
        return True
    if s in ('n', 'no', 'f', 'false', '0', ''):
        return False
    raise argparse.ArgumentTypeError(f'Boolean expected, got {v!r}')


def build_parser():
    p = argparse.ArgumentParser(
        description='seqA — sequential gated-bottleneck fusion + modality early-exit; '
                    'unified trainer for 6 datasets.')

    # ---- core ----
    p.add_argument('--dataset', required=True,
                   choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei', 'meld',
                            'meccano', 'czu_mhad', 'utd_mhad', 'mmfi', 'cmi', 'ch_simsv2'])
    p.add_argument('--fold', type=int, default=None)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--cuda_pick', default='cuda:0', type=str)
    p.add_argument('--results_dir', required=True, type=str)
    p.add_argument('--exp_name', default='seqA', type=str)
    p.add_argument('--seed_num', default=239, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--aggregate', action='store_true', default=False)

    # ---- dual-resolution / sequence length (float datasets) ----
    p.add_argument('--max_seq_len', default=128, type=int,
                   help='Truncate / pad every modality to this T (high-res path). 0 = no resample.')
    p.add_argument('--time_compression_ratio', default=4, type=int,
                   help='Low-resolution T = high T // this.')
    p.add_argument('--input_length_override', default=-1, type=int)

    # ---- SAX / HAR ----
    p.add_argument('--word_length', default=1, type=int,
                   help='SAX granularity (HAR datasets). alphabet_size is fixed at 20.')
    p.add_argument('--transform', default='sax', type=str,
                   help="HAR token transform: 'sax' (daliahar) / 'sax_noisy' (pamap2/dsads).")

    # ---- split / data path ----
    p.add_argument('--split_mode', default='cross_subject', type=str,
                   choices=['random', 'cross_subject'])
    p.add_argument('--data_root', default=None, type=str,
                   help='Override the per-dataset default data root.')
    p.add_argument('--available_sessions', nargs='+', type=int, default=None,
                   help='IEMOCAP/EAV: restrict to these sessions (ignored elsewhere).')
    p.add_argument('--modalities', nargs='+', default=None,
                   help='Optional modality subset (canonical order preserved).')

    # ---- MOSEI ----
    p.add_argument('--mosei_task', default=2, type=int, choices=[2, 5, 7],
                   help='MOSEI sentiment granularity: Acc-2/5/7 -> 2/5/7 classes.')

    # ---- model ----
    p.add_argument('--d_model', default=128, type=int)
    p.add_argument('--nhead', default=8, type=int)
    p.add_argument('--num_layers', default=4, type=int)
    p.add_argument('--num_layers_per_modal', default=2, type=int)
    p.add_argument('--dropout', default=0.1, type=float)
    p.add_argument('--internal_dim', default=128, type=int)
    p.add_argument('--n_bottlenecks', default=8, type=int)
    p.add_argument('--fusion_layer', default=0, type=int)
    p.add_argument('--num_experts', default=4, type=int)
    p.add_argument('--base_factor', default=10, type=int)
    p.add_argument('--use_sparse_attn', type=_str2bool, default=True)
    p.add_argument('--sparse_attn_variant', default='opt', type=str,
                   choices=['opt', 'opt_strat', 'opt_strat_blk'])
    p.add_argument('--strat_block_size', default=8, type=int)
    p.add_argument('--downsample_min_len', default=64, type=int)
    p.add_argument('--n_fusion_distill', default=2, type=int,
                   help='# of fusion layers (from the LAST) that distill (stride-2). '
                        '-1 = all; 0 = no fusion downsampling.')
    p.add_argument('--per_modal_distill', action='store_true', default=False)
    p.add_argument('--per_modal_downsample_min_len', default=4, type=int)
    p.add_argument('--bottleneck_agg_mode', default='gate', type=str, choices=['gate', 'mean'])
    p.add_argument('--bottleneck_init_mode', default='random', type=str,
                   choices=['random', 'modal_mean_sinpe'])
    p.add_argument('--fusion_add_pos_embeds', type=_str2bool, default=False)
    p.add_argument('--fusion_pos_embeds_max_len', type=int, default=256)
    p.add_argument('--fusion_pos_embed_mode', default='full',
                   choices=['full', 'decoupled', 'id_only', 'gated_id'])
    p.add_argument('--fusion_mlp_ratio', type=float, default=1.0)
    p.add_argument('--mlp_ratio', type=float, default=1.0)
    p.add_argument('--share_fusion_params', action='store_true', default=False)
    p.add_argument('--use_batched_fusion', type=_str2bool, default=True,
                   help='Use bmm-batched (parallel) per-modality fusion blocks. '
                        'True (default) = all M modalities in one grouped-GEMM forward; '
                        'False = loop modalities sequentially (slower, math-identical).')
    p.add_argument('--head_mode', default='gap', choices=['gap', 'entropy'])

    # ---- sequential fusion (方案 A / B-C) — the seqA core ----
    p.add_argument('--fusion_mode', default='sequential', type=str,
                   choices=['joint', 'sequential'])
    p.add_argument('--seq_depth_flow', action='store_true', default=False)
    p.add_argument('--seq_modality_order', nargs='+', type=int, default=None)
    p.add_argument('--seq_random_order', action='store_true', default=False)
    p.add_argument('--seq_agg_mode', default='gate', type=str,
                   choices=['gate', 'mean', 'mlp'],
                   help="Sequential bottleneck update: 'gate'=GRU (seqA), 'mean'=running avg (mbtAvg), 'mlp'=2-layer MLP (seqMLP)")

    # ---- modality-drop curriculum ----
    p.add_argument('--max_modality_drop', default=0.4, type=float)
    p.add_argument('--warmup_epochs', type=int, default=0,
                   help='Absolute warmup epochs with zero modality dropout; '
                        'overrides the percentage-based get_modality_curriculum when > 0.')
    p.add_argument('--ramp_epochs', type=int, default=0,
                   help='Absolute ramp epochs (0→max_modality_drop) after warmup.')
    p.add_argument('--profile_update_freq', default=5, type=int)
    p.add_argument('--dropout_signal', default='grad', type=str,
                   choices=['grad', 'importance', 'uniform'])
    p.add_argument('--importance_csv', default='', type=str)
    p.add_argument('--importance_floor', default=0.02, type=float)

    # ---- optimizer / schedule / loss ----
    p.add_argument('--lr_schedule', default='cosine', type=str,
                   choices=['constant', 'cosine', 'cosine_warmrestart'])
    p.add_argument('--clip_grad', default=1.0, type=float)
    p.add_argument('--loss', default='ce', choices=['ce', 'focal', 'weighted_ce'])
    p.add_argument('--focal_gamma', default=2.0, type=float)
    p.add_argument('--weight_decay', default=0.0, type=float)
    p.add_argument('--label_smoothing', default=0.0, type=float)

    # ---- MRdIB phase-1 singleton heads ----
    p.add_argument('--use_singleton_heads', action='store_true', default=False)
    p.add_argument('--alpha_U', type=float, default=0.0)

    # ---- prefix deep-supervision (early-exit aid) ----
    p.add_argument('--prefix_supervision', action='store_true', default=False)
    p.add_argument('--prefix_ds_weight', type=float, default=1.0)
    p.add_argument('--prefix_kd_weight', type=float, default=1.0)
    p.add_argument('--prefix_kd_weight_mode', default='const', choices=['const', 'k_prop'],
                   help="'const': equal per-prefix weight (original); "
                        "'k_prop': weight prefix j by (j+1)/K so later prefixes dominate.")

    # ---- phase-2 fine-tuning (improved prefixup: fixed LOO order + EMA teacher) ----
    p.add_argument('--phase2_init_from', default=None, type=str,
                   help='Dir with best_model_loo_fold{fold}.pth from phase-1 training. '
                        'Loads these weights before the epoch loop starts.')
    p.add_argument('--use_val_loo_order', action='store_true', default=False,
                   help='Compute (or load) val-LOO modality order once at startup and '
                        'use it as the fixed training order (eliminates train/inference mismatch).')
    p.add_argument('--phase2_ema_teacher', action='store_true', default=False,
                   help='Maintain an EMA copy of the model as the KD teacher instead of '
                        'using the live model output (stabilises the distillation target).')
    p.add_argument('--ema_decay', type=float, default=0.999,
                   help='EMA decay rate for the teacher model (default 0.999).')
    return p


# ============================================================================
# Main
# ============================================================================

def _compute_class_weights(batches, num_classes, device):
    """Inverse-frequency class weights from preloaded (high, low, y) batches."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, _, y in batches:
        for lbl in y.view(-1).tolist():
            counts[int(lbl)] += 1
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def main():
    args = build_parser().parse_args()

    if args.seq_depth_flow and args.fusion_mode != 'sequential':
        print('[warn] --seq_depth_flow ignored unless --fusion_mode=sequential')

    device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
    num_epochs = args.num_epochs
    current_date = datetime.now().strftime('%Y-%m-%d')
    if args.fusion_mode == 'sequential':
        if args.seq_agg_mode == 'mean':
            variant_tag = 'mbtAvg'           # sequential fusion, running-average update
        elif args.seq_agg_mode == 'mlp':
            variant_tag = 'seqMLP'           # sequential fusion, 2-layer MLP update
        elif args.seq_depth_flow:
            variant_tag = 'seqC_depth'
        else:
            variant_tag = 'seqA'
        if args.prefix_supervision:
            variant_tag += '_prefixup'
        if args.seq_random_order:
            variant_tag += '_randord'
        if args.phase2_init_from is not None:
            variant_tag += '_v2'
    else:
        variant_tag = 'joint'
    exp_name_full = f"{current_date}_{args.dataset}_{args.exp_name}_{variant_tag}"
    exp_dir = os.path.join(args.results_dir, exp_name_full)
    os.makedirs(exp_dir, exist_ok=True)

    # ---------------------------- aggregate mode ----------------------------
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
                print(f"  {key}: {aggregated[f'{key}_mean']:.4f} +/- {aggregated[f'{key}_std']:.4f}")
        sys.exit(0)

    # ------------------------------- setup ----------------------------------
    set_seed(args.seed_num)
    print(f"Device: {device}")
    print(f"Experiment: {exp_name_full}")

    train_batches, val_batches, test_batches, modalities, num_classes, model = build_data(args, device)
    num_m = len(modalities)
    print('Modalities:', modalities, 'num_classes:', num_classes)

    inner = model.model

    # ---- phase-2 fine-tuning setup -----------------------------------------
    # Phase-2 init: load weights from a phase-1 checkpoint before training.
    if args.phase2_init_from is not None:
        sfx_ = f"fold{args.fold}" if args.fold is not None else "val"
        ckpt_path = os.path.join(args.phase2_init_from, f"best_model_loo_{sfx_}.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.phase2_init_from, f"best_model_cold_{sfx_}.pth")
        print(f'[phase2] Loading phase-1 weights: {ckpt_path}')
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=True)
        print('[phase2] Weights loaded OK.')

    # EMA teacher: frozen copy of the model; updated per step after optimizer.step().
    ema_model = None
    if args.phase2_ema_teacher:
        ema_model = copy.deepcopy(model)
        for p_ema in ema_model.parameters():
            p_ema.requires_grad_(False)
        ema_model.eval()
        print(f'[phase2] EMA teacher initialised (decay={args.ema_decay}).')

    # Fixed val-LOO order: load from phase-1 results (fast) or recompute (slow).
    fixed_loo_order = None
    if args.use_val_loo_order:
        loaded_order = None
        if args.phase2_init_from is not None:
            sfx_ = f"fold{args.fold}" if args.fold is not None else "val"
            rpath = os.path.join(args.phase2_init_from, f"results_{sfx_}.json")
            if os.path.exists(rpath):
                with open(rpath) as _f:
                    _saved = json.load(_f)
                loaded_order = (_saved.get('dual_order', {})
                                .get('A_valLOO', {}).get('order'))
                if loaded_order:
                    print(f'[phase2] LOO order from phase-1 results: {loaded_order}')
        if loaded_order is None:
            print('[phase2] Computing val-LOO order on val set...')
            dv0 = dual_val(inner, val_batches, modalities, device, K=1, seed=42)
            loaded_order = dv0['orderA']
            print(f'[phase2] Computed LOO order: {loaded_order}')
        fixed_loo_order = loaded_order
    # -------------------------------------------------------------------------

    # Importance-guided dropout helper (drop low-importance modalities more).
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

    if args.use_singleton_heads:
        print(f"[MRdIB-Phase1] singleton heads enabled. alpha_U={args.alpha_U}")

    profiler = ModalityGradientProfiler(inner, modalities)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.loss == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma)
    elif args.loss == 'weighted_ce':
        cw = _compute_class_weights(train_batches, num_classes, device)
        print(f'[weighted_ce] class weights: min={cw.min():.4f} max={cw.max():.4f} ratio={cw.max()/cw.min():.1f}x')
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = None
    if args.lr_schedule != 'constant':
        from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
        if args.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        else:
            T_0 = max(int(num_epochs * 0.2), 1)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=1e-6)

    print(f'Parameters: {count_model_parameters(model)}')
    if args.fusion_mode == 'sequential':
        _v = '方案 B/C (depth-flow)' if args.seq_depth_flow else '方案 A (per-layer board)'
        _o = ('RANDOM per train-forward' if args.seq_random_order
              else (args.seq_modality_order if args.seq_modality_order is not None
                    else 'available order'))
        print(f'Fusion mode:   SEQUENTIAL — {_v}; order={_o}')
    else:
        print('Fusion mode:   JOINT (baseline MBT)')
    print(f'LR schedule: {args.lr_schedule} (base_lr={args.lr})')
    print(f'Selector:      NONE (fusion-only)')

    # ----------------------------- training loop ----------------------------
    cold_best_val_acc, cold_best_val_f1, cold_best_epoch = 0.0, 0.0, -1
    cold_best_model_state = None

    DUAL_K = 5
    bestA = {'acc': -1.0, 'f1': 0.0, 'epoch': -1, 'state': None, 'order': None}
    bestB = {'acc': -1.0, 'f1': 0.0, 'epoch': -1, 'state': None}

    current_mod_probs = {m: 0.0 for m in modalities}
    profile_log = []
    mod_drop_complete_epoch = int(num_epochs * 0.5)

    for epoch in range(num_epochs):
        if args.warmup_epochs > 0 or args.ramp_epochs > 0:
            _w, _r = args.warmup_epochs, args.ramp_epochs
            if epoch < _w:
                mod_base = 0.0
            elif epoch < _w + _r:
                mod_base = args.max_modality_drop * (epoch - _w) / _r
            else:
                mod_base = args.max_modality_drop
        else:
            mod_base = get_modality_curriculum(epoch, num_epochs, args.max_modality_drop)
        if mod_base == 0:
            current_mod_probs = {m: 0.0 for m in modalities}
        elif args.dropout_signal == 'importance':
            current_mod_probs = importance_dropout_probs(mod_base)
            if epoch % args.profile_update_freq == 0:
                profile_log.append({'epoch': epoch, 'mod_base': mod_base,
                                    'mod_dropout_probs': {m: round(p, 4) for m, p in current_mod_probs.items()}})
        elif args.dropout_signal == 'uniform':
            current_mod_probs = {m: mod_base for m in modalities}
        elif epoch % args.profile_update_freq == 0:   # grad (legacy)
            if any(len(profiler.grad_norms[m]) > 0 for m in modalities):
                current_mod_probs = profiler.get_asymmetric_dropout_probs(mod_base)
                profile_log.append({
                    'epoch': epoch, 'mod_base': mod_base,
                    'grad_norms': profiler.get_norm_snapshot(),
                    'mod_dropout_probs': {m: round(p, 4) for m, p in current_mod_probs.items()},
                })
            else:
                current_mod_probs = {m: mod_base for m in modalities}

        train_loss, train_acc = train_one_epoch(
            train_batches, model, criterion, optimizer, epoch, device, modalities,
            modality_dropout_probs=current_mod_probs, clip_grad=args.clip_grad,
            alpha_U=args.alpha_U, prefix_supervision=args.prefix_supervision,
            prefix_ds_weight=args.prefix_ds_weight, prefix_kd_weight=args.prefix_kd_weight,
            prefix_kd_weight_mode=args.prefix_kd_weight_mode,
            ema_model=ema_model, ema_decay=args.ema_decay,
            fixed_order=fixed_loo_order)

        val_loss, val_acc, val_y, val_p = evaluate_one_epoch(
            val_batches, model, criterion, epoch, device, modalities)
        val_f1 = f1_score(val_y, val_p, average='macro')

        if scheduler is not None:
            scheduler.step()

        # Dual-order val (warm phase only): A = val-LOO order, B = random-order avg.
        if epoch >= mod_drop_complete_epoch and num_m >= 2:
            dv = dual_val(inner, val_batches, modalities, device, K=DUAL_K, seed=1234 + epoch)
            print(f"  [dual] A(val-LOO order) acc={dv['accA']:.4f} f1={dv['f1A']:.4f} order={dv['orderA']}"
                  f"  |  B(rand-avg/{DUAL_K}) acc={dv['accB']:.4f} f1={dv['f1B']:.4f}")
            if dv['accA'] > bestA['acc']:
                bestA = {'acc': dv['accA'], 'f1': dv['f1A'], 'epoch': epoch,
                         'state': model.state_dict(), 'order': dv['orderA']}
            if dv['accB'] > bestB['acc']:
                bestB = {'acc': dv['accB'], 'f1': dv['f1B'], 'epoch': epoch, 'state': model.state_dict()}

        if val_acc > cold_best_val_acc:
            cold_best_val_acc, cold_best_val_f1, cold_best_epoch = val_acc, val_f1, epoch
            cold_best_model_state = model.state_dict()

    # ---------------------------- model selection ---------------------------
    if bestA['state'] is None:
        bestA = {'acc': cold_best_val_acc, 'f1': cold_best_val_f1, 'epoch': cold_best_epoch,
                 'state': cold_best_model_state, 'order': list(modalities)}
    if bestB['state'] is None:
        bestB = {'acc': cold_best_val_acc, 'f1': cold_best_val_f1, 'epoch': cold_best_epoch,
                 'state': cold_best_model_state}

    sfx = f"fold{args.fold}" if args.fold is not None else "val"
    torch.save(bestA['state'], os.path.join(exp_dir, f"best_model_loo_{sfx}.pth"))
    torch.save(bestB['state'], os.path.join(exp_dir, f"best_model_rand_{sfx}.pth"))
    if cold_best_model_state is not None:
        torch.save(cold_best_model_state, os.path.join(exp_dir, f"best_model_cold_{sfx}.pth"))
    print(f"\n[method A val-LOO order]  best epoch {bestA['epoch']}  "
          f"val_acc={bestA['acc']:.4f} f1={bestA['f1']:.4f}  order={bestA['order']}")
    print(f"[method B random-avg]     best epoch {bestB['epoch']}  "
          f"val_acc={bestB['acc']:.4f} f1={bestB['f1']:.4f}")

    print("\n" + "=" * 80 + "\nTest-set evaluation\n" + "=" * 80)
    model.load_state_dict(bestA['state']); model.eval()
    testA_acc, testA_f1 = eval_full_order(inner, test_batches, modalities, device,
                                          order_names=bestA['order'])
    print(f"  [A val-LOO order] test_acc={testA_acc:.4f} test_f1={testA_f1:.4f}")
    model.load_state_dict(bestB['state']); model.eval()
    testB_acc, testB_f1 = rand_avg_eval(inner, test_batches, modalities, device,
                                        K=DUAL_K, seed=4321)
    print(f"  [B random-avg]    test_acc={testB_acc:.4f} test_f1={testB_f1:.4f}")

    # Method A (val-LOO order) is the primary reported result (IEMOCAP recipe).
    best_val_acc, best_val_f1, best_epoch_val = bestA['acc'], bestA['f1'], bestA['epoch']
    test_acc, test_f1 = testA_acc, testA_f1

    # ------------------------------- save -----------------------------------
    final_grad_norms = (profiler.get_norm_snapshot()
                        if any(len(profiler.grad_norms[m]) > 0 for m in modalities) else {})
    config = {
        'experiment_name': exp_name_full,
        'dataset': args.dataset,
        'model_variant': 'seqA',
        'model_class': 'DualVideoBottleneckModelV6Downsample (seqfusion)',
        'head_mode': args.head_mode,
        'fusion_mode': args.fusion_mode,
        'seq_depth_flow': bool(args.seq_depth_flow),
        'seq_modality_order': args.seq_modality_order,
        'seq_random_order': bool(args.seq_random_order),
        'selector': 'NONE',
        'fold': args.fold,
        'modalities': modalities,
        'num_classes': num_classes,
        'max_seq_len': args.max_seq_len,
        'time_compression_ratio': args.time_compression_ratio,
        'word_length': args.word_length, 'transform': args.transform,
        'num_epochs': num_epochs, 'batch_size': args.batch_size, 'seed': args.seed_num,
        'd_model': args.d_model, 'nhead': args.nhead, 'num_layers': args.num_layers,
        'num_layers_per_modal': args.num_layers_per_modal, 'dropout': args.dropout,
        'num_experts': args.num_experts, 'base_factor': args.base_factor,
        'internal_dim': args.internal_dim, 'n_bottlenecks': args.n_bottlenecks,
        'fusion_layer': args.fusion_layer, 'use_sparse_attn': args.use_sparse_attn,
        'sparse_attn_variant': args.sparse_attn_variant,
        'strat_block_size': args.strat_block_size,
        'downsample_min_len': args.downsample_min_len,
        'n_fusion_distill': args.n_fusion_distill,
        'bottleneck_agg_mode': args.bottleneck_agg_mode,
        'lr': args.lr, 'lr_schedule': args.lr_schedule, 'clip_grad': args.clip_grad,
        'loss': args.loss, 'data_root': args.data_root, 'split_mode': args.split_mode,
    }
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    model_stats = {
        'experiment_name': exp_name_full, 'fold': args.fold,
        'best_val_acc': best_val_acc, 'best_val_f1': best_val_f1,
        'best_epoch_val': best_epoch_val, 'selected_source': 'dual(A=val-LOO,B=rand)',
        'cold_best': {'val_acc': cold_best_val_acc, 'val_f1': cold_best_val_f1,
                      'epoch': cold_best_epoch},
        'best_test_acc': float(test_acc), 'best_test_f1': float(test_f1),
        'dual_order': {
            'A_valLOO': {'val_acc': bestA['acc'], 'val_f1': bestA['f1'], 'epoch': bestA['epoch'],
                         'order': bestA['order'], 'test_acc': float(testA_acc),
                         'test_f1': float(testA_f1), 'ckpt': f"best_model_loo_{sfx}.pth"},
            'B_randavg': {'val_acc': bestB['acc'], 'val_f1': bestB['f1'], 'epoch': bestB['epoch'],
                          'K': DUAL_K, 'test_acc': float(testB_acc), 'test_f1': float(testB_f1),
                          'ckpt': f"best_model_rand_{sfx}.pth"},
        },
        'final_grad_norms': final_grad_norms,
        'profile_log': profile_log,
    }
    results_filename = (f"results_fold{args.fold}.json" if args.fold is not None
                        else "results.json")
    with open(os.path.join(exp_dir, results_filename), 'w') as f:
        json.dump(model_stats, f, indent=4)

    profiler.remove_hooks()
    print(f"\nBest Model (Epoch {best_epoch_val}): val_acc={best_val_acc:.4f}  "
          f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
    print(f"Directory: {exp_dir}")


if __name__ == '__main__':
    main()
