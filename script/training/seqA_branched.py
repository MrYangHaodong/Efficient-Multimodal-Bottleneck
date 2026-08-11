#!/usr/bin/env python
"""Trainer for the branched, interleaved seqA (``models/seqA_branched.py``).

Each modality is a fully-independent encoder+fusion branch; the forward
interleaves encode->fuse per modality and supports true early exit. This script
reuses ``training/seqA.py``'s data pipeline (``build_data`` / ``build_parser``)
so every flag the existing seqA .sh files pass is still accepted, and reproduces
the dropaug recipe (modality-dropout curriculum + prefix deep-supervision + EMA
self-distillation teacher) — but drives the branched model instead of the batched
``DualVideoBottleneckModelV6Downsample``.

Differences vs training/seqA.py:
  * Model is ``SeqABranched`` (per-modality branches, interleaved enc->fuse).
  * Modality dropout is applied at BATCH level (a modality subset is chosen per
    batch and only those branches run) — per-sample drop can't share the batched
    branch call. Eval always uses all modalities in canonical order.
  * Simpler model selection: best-val macro-F1 -> single ``best_model_fold{N}.pth``
    + ``results_fold{N}.json`` (no dual-order/LOO machinery).

Layer counts: ``--num_enc_layers`` / ``--num_fusion_layers`` (branch-specific);
if omitted they fall back to ``--num_layers_per_modal`` / ``--num_layers`` so the
existing seqA .sh flags map over unchanged.

Example:
  python training/seqA_branched.py --dataset cmi --fold 0 \
      --results_dir runs/seqA_branched --exp_name branch \
      --d_model 128 --nhead 8 --num_enc_layers 3 --num_fusion_layers 3 \
      --n_bottlenecks 8 --seq_agg_mode mean --seq_random_order \
      --prefix_supervision --prefix_kd_weight_mode k_prop --phase2_ema_teacher \
      --max_modality_drop 0.4 --warmup_epochs 10 --ramp_epochs 20 --num_epochs 40 \
      --max_seq_len 128 --batch_size 32 --time_compression_ratio 4
"""
import os
import sys
import json
import copy
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from script.training.seqA import build_parser as _seqa_parser, build_data, _compute_class_weights
from multimodal_model.seqA_branched import SeqABranched


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_parser():
    """Reuse the seqA parser (accepts every flag the existing .sh files pass) and
    add branch-specific layer counts."""
    p = _seqa_parser()
    p.add_argument('--num_enc_layers', type=int, default=None,
                   help='Per-modal encoder layers (branch). Default: --num_layers_per_modal.')
    p.add_argument('--num_fusion_layers', type=int, default=None,
                   help='Per-modal fusion layers (branch). Default: --num_layers.')
    # --mlp_ratio already provided by the seqA parser.
    return p


def make_model(args, mods, feat_dims, num_classes, device):
    enc_layers = args.num_enc_layers if args.num_enc_layers is not None else args.num_layers_per_modal
    fus_layers = args.num_fusion_layers if args.num_fusion_layers is not None else args.num_layers
    # Fusion distillation: reuse --n_fusion_distill (the same flag the batched seqA uses).
    # -1 = every fusion layer distills, 0 = none (full resolution), N = last N layers.
    n_fus_distill = getattr(args, 'n_fusion_distill', 0)
    model = SeqABranched(
        modalities=mods, feat_dims=feat_dims, num_classes=num_classes,
        d_model=args.d_model, nhead=args.nhead,
        num_enc_layers=enc_layers, num_fusion_layers=fus_layers,
        n_bottlenecks=args.n_bottlenecks, mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        use_sparse_attn=args.use_sparse_attn, sparse_attn_variant=args.sparse_attn_variant,
        factor=args.base_factor, strat_block_size=args.strat_block_size,
        seq_agg_mode=args.seq_agg_mode, seq_random_order=args.seq_random_order,
        num_fusion_distill=n_fus_distill,
    ).to(device)
    print(f'[branched] enc_layers={enc_layers} fusion_layers={fus_layers} '
          f'n_fusion_distill={n_fus_distill} '
          f'd_model={args.d_model} K={args.n_bottlenecks} agg={args.seq_agg_mode} '
          f'sparse_attn={args.use_sparse_attn} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    return model


def _curriculum_drop(epoch, args):
    """Batch-level modality-drop probability for this epoch (warmup/ramp/max)."""
    w, r = args.warmup_epochs, args.ramp_epochs
    if w > 0 or r > 0:
        if epoch < w:
            return 0.0
        if epoch < w + r:
            return args.max_modality_drop * (epoch - w) / r
        return args.max_modality_drop
    return args.max_modality_drop


def _sample_kept(mods, drop_p):
    """Choose which modalities survive this batch (keep >= 1)."""
    if drop_p <= 0.0:
        return list(mods)
    kept = [m for m in mods if random.random() > drop_p]
    if not kept:
        kept = [random.choice(list(mods))]
    return kept


def _prefix_weights(K, mode):
    if mode == 'k_prop':
        raw = [(j + 1) / K for j in range(K)]
        s = sum(raw)
        return [w / s for w in raw]
    return [1.0 / K] * K


def train_one_epoch(batches, model, criterion, optimizer, epoch, device, mods,
                    drop_p, args, ema_model=None):
    model.train()
    loss_sum = acc_sum = n = 0
    for high, _low, y in batches:
        kept = _sample_kept(mods, drop_p)
        inputs = {m: high[m].to(device, non_blocking=True) for m in kept}
        y = y.to(device, non_blocking=True)

        if args.prefix_supervision:
            final, prefix, _ = model(inputs, return_prefix=True)     # prefix [B, S, C]
            loss = criterion(final, y)
            K = prefix.shape[1]
            ws = _prefix_weights(K, args.prefix_kd_weight_mode)
            ds = sum(ws[j] * criterion(prefix[:, j], y) for j in range(K))
            if ema_model is not None:
                with torch.no_grad():
                    ema_logits = ema_model(inputs)
                teach = F.softmax(ema_logits, dim=-1)
            else:
                teach = F.softmax(final.detach(), dim=-1)
            kd = sum(ws[j] * F.kl_div(F.log_softmax(prefix[:, j], dim=-1), teach,
                                      reduction='batchmean') for j in range(K))
            loss = loss + args.prefix_ds_weight * ds + args.prefix_kd_weight * kd
        else:
            final = model(inputs)
            loss = criterion(final, y)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        if ema_model is not None:
            with torch.no_grad():
                for ep, p in zip(ema_model.parameters(), model.parameters()):
                    ep.data.mul_(args.ema_decay).add_(p.data, alpha=1.0 - args.ema_decay)

        bs = y.size(0)
        loss_sum += loss.item() * bs
        acc_sum += (final.argmax(1) == y).sum().item()
        n += bs
    return loss_sum / max(n, 1), acc_sum / max(n, 1)


@torch.no_grad()
def evaluate(batches, model, device, mods):
    """Full forward (all modalities, canonical order)."""
    model.eval()
    ys, ps = [], []
    for high, _low, y in batches:
        inputs = {m: high[m].to(device, non_blocking=True) for m in mods}
        logits = model(inputs)
        ps.append(logits.argmax(1).cpu())
        ys.append(y.view(-1))
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average='macro'))
    return acc, f1


def main():
    args = build_parser().parse_args()
    device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed_num)

    date = datetime.now().strftime('%Y-%m-%d')
    exp_name_full = f'{date}_{args.dataset}_{args.exp_name}_seqAbranch'
    exp_dir = os.path.join(args.results_dir, exp_name_full)
    os.makedirs(exp_dir, exist_ok=True)
    print(f'Device: {device}\nExperiment: {exp_name_full}')

    # Data pipeline reused from training/seqA.py (discard its batched model).
    # build_model=False: don't construct the batched V6 model we'd only discard
    # (avoids the misleading DualVideoBottleneckModelV6Downsample banner + wasted build).
    train_b, val_b, test_b, mods, num_classes, _seqa_model = build_data(args, device, build_model=False)
    del _seqa_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Feature dims from the first batch (per-modality [B, T, feat]).
    first_high = train_b[0][0]
    feat_dims = {m: first_high[m].shape[-1] for m in mods}
    print('Modalities:', mods, 'num_classes:', num_classes, 'feat_dims:', feat_dims)

    model = make_model(args, mods, feat_dims, num_classes, device)

    # Loss
    if args.loss == 'weighted_ce':
        cw = _compute_class_weights(train_b, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
        print(f'[weighted_ce] class weights min={cw.min():.3f} max={cw.max():.3f}')
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_schedule == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    ema_model = None
    if args.phase2_ema_teacher:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()
        print(f'[ema] teacher initialised (decay={args.ema_decay})')

    best = {'val_f1': -1.0, 'val_acc': 0.0, 'epoch': -1, 'state': None}
    for epoch in range(args.num_epochs):
        drop_p = _curriculum_drop(epoch, args) if args.dropout_signal in ('uniform', 'grad', 'importance') else 0.0
        tr_loss, tr_acc = train_one_epoch(
            train_b, model, criterion, optimizer, epoch, device, mods, drop_p, args, ema_model)
        if scheduler is not None:
            scheduler.step()
        val_acc, val_f1 = evaluate(val_b, model, device, mods)
        if epoch % 5 == 0 or epoch == args.num_epochs - 1:
            print(f'  ep{epoch:03d} drop={drop_p:.3f} train_loss={tr_loss:.4f} '
                  f'train_acc={tr_acc:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}', flush=True)
        if val_f1 > best['val_f1']:
            best = {'val_f1': val_f1, 'val_acc': val_acc, 'epoch': epoch,
                    'state': copy.deepcopy(model.state_dict())}

    # Test with best-val model
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    test_acc, test_f1 = evaluate(test_b, model, device, mods)
    ckpt = os.path.join(exp_dir, f'best_model_fold{args.fold}.pth')
    torch.save(best['state'], ckpt)

    result = {
        'dataset': args.dataset, 'fold': args.fold,
        'method': 'seqA_branched', 'model_class': 'SeqABranched',
        'modalities': mods, 'num_classes': num_classes,
        'd_model': args.d_model, 'nhead': args.nhead,
        'num_enc_layers': args.num_enc_layers if args.num_enc_layers is not None else args.num_layers_per_modal,
        'num_fusion_layers': args.num_fusion_layers if args.num_fusion_layers is not None else args.num_layers,
        'num_fusion_distill': getattr(args, 'n_fusion_distill', 0),
        'n_bottlenecks': args.n_bottlenecks, 'seq_agg_mode': args.seq_agg_mode,
        'seq_random_order': bool(args.seq_random_order),
        'use_sparse_attn': bool(args.use_sparse_attn),
        'sparse_attn_variant': args.sparse_attn_variant, 'base_factor': args.base_factor,
        'prefix_supervision': bool(args.prefix_supervision),
        'ema_teacher': bool(args.phase2_ema_teacher),
        'max_modality_drop': args.max_modality_drop,
        'num_epochs': args.num_epochs, 'batch_size': args.batch_size,
        'max_seq_len': args.max_seq_len, 'lr': args.lr, 'seed': args.seed_num,
        'best_epoch': best['epoch'],
        'best_val_acc': best['val_acc'], 'best_val_f1': best['val_f1'],
        'best_test_acc': test_acc, 'best_test_f1': test_f1,
    }
    with open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n[seqA_branched] {args.dataset} fold{args.fold} '
          f'val(acc={best["val_acc"]:.4f} f1={best["val_f1"]:.4f}) '
          f'TEST(acc={test_acc:.4f} f1={test_f1:.4f}) -> {exp_dir}', flush=True)


if __name__ == '__main__':
    main()
