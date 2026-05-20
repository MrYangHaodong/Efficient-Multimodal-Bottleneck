"""V6 fusion training on IEMOCAP with **Classifier-Guided Gradient
Modulation** (CGGM, Guo et al. NeurIPS 2024) replacing the gradnorm-based
asymmetric modality dropout from ``main_v6_iemocap_fusion_only_dyna.py``.

What changes vs the dyna script (Plan B from the design discussion):

  1. **Per-modality aux classifiers** sit on the post-encoder pooled
     features (``model._last_processed_modalities[m].mean(dim=1)``).
     They share NO weights with the fusion path and have their own
     optimizer.

  2. **Drop probabilities come from aux classifier accuracy, not
     gradient norms.**  ``ModalityAccProfiler.get_asymmetric_dropout_probs``
     scales ``base * M * (acc_m / sum acc)``.  Same shape as the dyna
     formula but the signal source is more direct (modality-standalone
     classification accuracy → CGGM's coefficient).

  3. **Gradient modulation** at each step:
       - Compute per-modality accuracy ``acc[i]`` from aux classifiers
       - ``diff[i] = acc[i] - acc_prev[i]`` (improvement this step)
       - ``coeff[i] = (sum(diff) - diff[i]) / sum(diff)``  (slow ↑, fast ↓)
       - Multiply per-modality slice of fusion grads by ``coeff[i] * rou``:
           * ``input_projectors[m]`` weight/bias
           * each ``batched_modality_encoder.layers[L]`` param's ``[M, ...]``
             leading-dim slice (all params in BatchedTransformerBlock are
             ``[M_total, ...]`` shaped — slicing the leading dim works
             cleanly).

  4. **l_gm direction term** (added to next step's fusion loss):
     compute cosine of aux-out_layer grad vs fusion regressor-out_layer
     grad → ``llist[i]``, then ``l_gm = sum|coeff| - <coeff, llist> / M``.

  5. Original gradnorm hook is kept as a **diagnostic** so
     ``final_grad_norms`` stays in ``results.json``.

Run:
    python main_v6_iemocap_fusion_only_cggm.py --fold=0 --cuda_pick=cuda:0
    python main_v6_iemocap_fusion_only_cggm.py --aggregate
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed, count_model_parameters
from utils.train_utils import (
    AverageMeter, ProgressMeter, FocalLoss,
    get_modality_curriculum,
    ModalityGradientProfiler, ModalityAccProfiler,
)
from data.IEMOCAP.get_data import (
    get_dataloader as iemocap_get_dataloader,
    FEATURE_DIMS, NUM_CLASSES, IEMOCAPDataset,
)
from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)


# ============================ V6 cfg adaptor ============================


class _IEMOCAPV6Cfg:
    def __init__(self, modalities=None):
        if modalities is None:
            modalities = list(IEMOCAPDataset.ALL_MODALITIES)
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        v_dim = FEATURE_DIMS.get('video', 1024)
        self.video_high_dim = v_dim
        self.video_low_dim = v_dim
        self.num_classes = NUM_CLASSES


# ============================ Wrapper ============================


class V6IEMOCAPFusionCGGMWrapper(nn.Module):
    """Same modality unpacking + dropout as the dyna wrapper.  No changes
    needed for CGGM — gradient modulation happens in the train loop after
    backward."""

    def __init__(self, v6_model, modalities):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self._mod_order = list(modalities)

    def _unpack(self, feats):
        assert len(feats) == 2 * len(self._mod_order)
        high, low = {}, {}
        for j, m in enumerate(self._mod_order):
            high[m] = feats[2 * j]
            low[m] = feats[2 * j + 1]
        return high, low

    def forward(self, feats, modality_dropout_probs=None, training=False):
        high, low = self._unpack(feats)
        kept = list(self.modalities)
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
                         return_selection_info=False)
        logits = out[0] if isinstance(out, tuple) else out
        return logits, kept


# ============================ Aux classifiers ============================


class AuxModalityClassifiers(nn.Module):
    """Per-modality standalone classifier.  Input: post-encoder pooled
    feature ``[B, d_model]``.  Output: ``[B, num_classes]``.  All heads
    independent — no cross-modal weight sharing."""

    def __init__(self, modalities, d_model, num_classes, dropout=0.1):
        super().__init__()
        self.modalities = list(modalities)
        self.heads = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, num_classes),
            )
            for m in modalities
        })

    def forward(self, processed_modalities):
        """processed_modalities: dict {m: [B, T, d_model]}.  Returns dict
        {m: [B, num_classes]}.  Skips missing modalities."""
        out = {}
        for m in self.modalities:
            if m in processed_modalities:
                pooled = processed_modalities[m].mean(dim=1)  # [B, d_model]
                out[m] = self.heads[m](pooled)
        return out

    def out_layer_weights(self):
        """Return per-modality final-Linear weight Parameters for CGGM's
        cosine alignment.  Order matches ``self.modalities``."""
        ws = []
        for m in self.modalities:
            ws.append(self.heads[m][-1].weight)
        return ws


# ============================ Train / eval ============================


def _unpack_batch(batch, device):
    *feats, label = batch
    feats = [f.to(device).float() for f in feats]
    label = label.squeeze(-1).long().to(device)
    return feats, label


def _scale_per_mod_grads(inner_model, modalities, coeff, rou):
    """Multiply each modality's slice of fusion-side grads by coeff[i]*rou.

    * input_projectors[m].weight/bias.grad   (whole tensor, per-mod module)
    * batched_modality_encoder.layers[L].<param>.grad[i]  (leading-dim slice
      of [M_total, ...] tensors)

    No-op if a grad is None (e.g., modality was dropped this step).
    """
    n = inner_model
    for i, m in enumerate(modalities):
        s = float(coeff[i] * rou)
        # 1) per-modality input projector
        proj = n.input_projectors[m]
        for p in proj.parameters():
            if p.grad is not None:
                p.grad.mul_(s)
        # 2) batched encoder weight slot i
        enc = n.batched_modality_encoder
        for layer in enc.layers:
            for p in layer.parameters():
                if p.grad is None:
                    continue
                if p.grad.dim() == 0:
                    continue
                # All BatchedTransformerBlock params are [M_total, ...]
                if p.grad.shape[0] == len(modalities):
                    p.grad[i].mul_(s)


def _cos_per_mod(aux_grads, fusion_grad_flat):
    """aux_grads: list of [num_classes, d_model] tensors (one per mod).
    Returns list of cosine sim with fusion_grad_flat."""
    sims = []
    for g in aux_grads:
        gf = g.detach().reshape(-1)
        s = F.cosine_similarity(gf, fusion_grad_flat, dim=0).item()
        sims.append(s)
    return sims


def _per_mod_acc(aux_logits, y):
    """aux_logits: dict {m: [B, C]}.  Returns dict {m: float in [0,1]}."""
    out = {}
    y_cpu = y.detach().cpu().numpy()
    for m, lg in aux_logits.items():
        pred = lg.argmax(dim=-1).detach().cpu().numpy()
        out[m] = float(accuracy_score(y_cpu, pred))
    return out


def train_one_epoch_cggm(loader, wrapper, aux_clfs, optimizer, aux_optimizer,
                         criterion, epoch, device, modality_dropout_probs,
                         modalities, acc_profiler, prev_acc,
                         lambda_lgm, rou, clip_grad=0.0,
                         lgm_prev=None):
    loss_meter = AverageMeter('Loss', ':.4f')
    cls_meter = AverageMeter('AuxCE', ':.4f')
    lgm_meter = AverageMeter('Lgm', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    wrapper.train(); aux_clfs.train()
    inner = wrapper.model
    fusion_out_layer = inner.regressor[-1] if isinstance(
        inner.regressor, nn.Sequential) else inner.regressor

    for i, batch in enumerate(loader):
        feats, y = _unpack_batch(batch, device)

        # ----------- Main fusion forward + backward -----------
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        logits, kept = wrapper(feats,
                                modality_dropout_probs=modality_dropout_probs,
                                training=True)
        main_loss = criterion(logits, y)
        if lgm_prev is not None:
            total_loss = main_loss + lambda_lgm * lgm_prev
        else:
            total_loss = main_loss
        total_loss.backward(retain_graph=True)

        # Grab fusion out-layer grad before further backward overwrites it
        fusion_out_grad = (fusion_out_layer.weight.grad.detach().clone()
                            .reshape(-1) if fusion_out_layer.weight.grad is not None
                            else None)

        # ----------- Aux classifier forward + backward -----------
        # Use post-encoder processed_modalities cached on inner model
        proc = getattr(inner, '_last_processed_modalities', None)
        if proc is None:
            raise RuntimeError(
                '_last_processed_modalities not set — V6 forward did not '
                'populate it.  Did you patch the V6 forward to expose it?')
        # Detach to keep the aux branch independent from fusion grad flow.
        proc_det = {m: t.detach() for m, t in proc.items()}
        # Re-grad through aux: rebuild graph by enabling grad on detached tensors.
        for t in proc_det.values():
            t.requires_grad_(True)
        aux_logits = aux_clfs(proc_det)
        if not aux_logits:
            # All modalities dropped — skip aux step entirely.
            optimizer.step()
            loss_meter.update(main_loss.item(), y.size(0))
            continue
        aux_loss = sum(criterion(aux_logits[m], y) for m in aux_logits)
        aux_loss.backward()

        # Per-modality accuracy.  We use a *rolling-window-smoothed* accuracy
        # as the dominance signal instead of CGGM's raw step-by-step
        # ``diff = acc_now - acc_prev`` because per-batch acc is too noisy
        # (32 samples × 4 classes) — diffs can be near-zero or oscillate
        # sign, blowing up the ``(diff_sum - d) / diff_sum`` coefficient
        # to ±10^5.  Using accuracy directly keeps coeff bounded in (0,1):
        # ``coeff[i] = (sum_acc - acc[i]) / sum_acc`` — low-acc modality
        # gets a coeff close to 1 (full grad), high-acc one gets smaller
        # coeff (suppressed).
        acc_now = _per_mod_acc(aux_logits, y)
        # Update profiler first to include current step in the rolling avg.
        acc_profiler.update(acc_now)
        smoothed = acc_profiler.get_acc_snapshot(window=20)
        acc_vec = [smoothed.get(m, 0.0) for m in modalities]
        sum_acc = sum(acc_vec) + 1e-8
        coeff = [(sum_acc - a) / sum_acc for a in acc_vec]

        # Direction term: cosine of aux out-layer grad vs fusion out-layer grad
        if fusion_out_grad is not None and aux_loss.requires_grad:
            aux_outs = aux_clfs.out_layer_weights()
            aux_grads = []
            for w in aux_outs:
                if w.grad is None:
                    aux_grads.append(torch.zeros_like(w).reshape(-1))
                else:
                    aux_grads.append(w.grad.detach().reshape(-1))
            llist = []
            for g in aux_grads:
                if g.norm() < 1e-8 or fusion_out_grad.norm() < 1e-8:
                    llist.append(0.0)
                else:
                    llist.append(F.cosine_similarity(
                        g.unsqueeze(0), fusion_out_grad.unsqueeze(0)).item())
        else:
            llist = [0.0] * len(modalities)

        # l_gm for *next* step (scalar python float since used as
        # additive constant via lgm_prev — non-differentiable proxy in the
        # CGGM paper too: they just track the magnitude penalty).
        coeff_arr = np.array(coeff)
        llist_arr = np.array(llist)
        lgm_now = float(np.abs(coeff_arr).sum()
                         - (coeff_arr * llist_arr).sum() / max(len(modalities), 1))

        # ----------- Modulate per-mod fusion grads BEFORE optimizer step --------
        _scale_per_mod_grads(inner, modalities, coeff, rou)

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), clip_grad)
        optimizer.step()
        aux_optimizer.step()

        # Meters (profiler already updated before coeff computation).
        prev_acc = {m: acc_now.get(m, prev_acc.get(m, 0.0)) for m in modalities}
        lgm_prev = torch.tensor(lgm_now, device=device)

        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / y.size(0)
        loss_meter.update(main_loss.item(), y.size(0))
        cls_meter.update(float(aux_loss.item()), y.size(0))
        lgm_meter.update(lgm_now, y.size(0))
        acc_meter.update(acc, y.size(0))

        if (i % 50 == 0) or (i == len(loader) - 1):
            progress = ProgressMeter(len(loader),
                                     [loss_meter, cls_meter, lgm_meter, acc_meter],
                                     prefix=f"Epoch: [{epoch}]")
            progress.display(i)
            if i == len(loader) - 1:
                mean_p = (float(np.mean(list(modality_dropout_probs.values())))
                          if modality_dropout_probs else 0.0)
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  '
                      f'AuxCE: {cls_meter.avg:.4f}  Lgm: {lgm_meter.avg:.4f}  '
                      f'Acc: {acc_meter.avg:.4f}  meanModDrop: {mean_p:.3f}')

    return loss_meter.avg, acc_meter.avg, prev_acc, lgm_prev


@torch.no_grad()
def evaluate_one_epoch(loader, wrapper, criterion, epoch, device):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')
    wrapper.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        feats, y = _unpack_batch(batch, device)
        logits, _ = wrapper(feats, modality_dropout_probs=None, training=False)
        loss = criterion(logits, y)
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / y.size(0)
        loss_meter.update(loss.item(), y.size(0))
        acc_meter.update(acc, y.size(0))
        all_preds.append(predicted.cpu()); all_labels.append(y.cpu())
    print(f'End of Epoch {epoch} | Val Loss: {loss_meter.avg:.4f} '
          f'| Val Acc: {acc_meter.avg:.4f}')
    return (loss_meter.avg, acc_meter.avg,
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())


# ============================ CLI ============================


def _str2bool(v):
    if isinstance(v, bool): return v
    s = v.strip().lower()
    if s in ('y', 'yes', 't', 'true', '1'): return True
    if s in ('n', 'no', 'f', 'false', '0', ''): return False
    raise argparse.ArgumentTypeError(f'Boolean expected, got {v!r}')


parser = argparse.ArgumentParser(description='V6 fusion training with CGGM on IEMOCAP.')
parser.add_argument('--exp_name', default='v6_fusion_only_cggm', type=str)
parser.add_argument('--results_dir',
                    default='./results_v6_fusion_cggm_iemocap/', type=str)
parser.add_argument('--dataset', default='IEMOCAP', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--modalities', nargs='+',
                    default=list(IEMOCAPDataset.ALL_MODALITIES),
                    choices=list(IEMOCAPDataset.ALL_MODALITIES))
parser.add_argument('--max_seq_len', default=128, type=int)
parser.add_argument('--time_compression_ratio', default=4, type=int)
# Model
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--internal_dim', default=128, type=int)
parser.add_argument('--n_bottlenecks', default=8, type=int)
parser.add_argument('--fusion_layer', default=0, type=int)
parser.add_argument('--use_sparse_attn', type=_str2bool, default=True)
parser.add_argument('--sparse_attn_variant', default='opt', type=str)
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=4, type=int)
parser.add_argument('--per_modal_distill', action='store_true', default=False)
parser.add_argument('--per_modal_downsample_min_len', default=4, type=int)
parser.add_argument('--use_batched_fusion', type=_str2bool, default=True)
# Modality-drop curriculum (now driven by aux acc, not gradnorm)
parser.add_argument('--max_modality_drop', default=0.4, type=float)
parser.add_argument('--profile_update_freq', default=5, type=int)
# CGGM knobs
parser.add_argument('--cggm_rou', default=1.3, type=float,
                    help='CGGM scaling factor (Guo et al. NeurIPS 2024 default 1.3).')
parser.add_argument('--cggm_lambda', default=0.2, type=float,
                    help='Weight on l_gm direction-alignment term in main loss '
                         '(Guo et al. default 0.2).')
parser.add_argument('--cls_lr', default=1e-4, type=float,
                    help='LR for aux classifier optimizer (separate from fusion).')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='Epochs of plain training before CGGM kicks in (set >0 '
                         'if aux classifiers need to converge first).')
# Optimizer
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['constant', 'cosine'])
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--loss', default='ce', choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0, type=float)
# Fold
parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2])
parser.add_argument('--aggregate', action='store_true', default=False)
parser.add_argument('--data_root', default='/files1/haodong/data/IEMOCAP', type=str)
parser.add_argument('--available_sessions', nargs='+', type=int, default=None)

args = parser.parse_args()


# ============================ Setup ============================

num_epochs = args.num_epochs
device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
current_date = datetime.now().strftime('%Y-%m-%d')
exp_name_full = f"{current_date}_{args.dataset}_{args.exp_name}"
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
        print(f'No fold results found in {exp_dir}.'); sys.exit(1)
    agg = {'experiment_name': exp_name_full, 'folds': fold_results}
    for key in ('best_val_acc', 'best_val_f1', 'best_test_acc', 'best_test_f1'):
        vals = [fr[key] for fr in fold_results if key in fr]
        if vals:
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump(agg, f, indent=4)
    print(f"Aggregated -> {exp_dir}/results.json")
    for key in ('best_val_acc', 'best_val_f1', 'best_test_acc', 'best_test_f1'):
        if f'{key}_mean' in agg:
            print(f"  {key}: {agg[f'{key}_mean']:.4f} +/- {agg[f'{key}_std']:.4f}")
    sys.exit(0)


# ============================ Data ============================


set_seed(args.seed_num)
print(f"Device: {device}\nExperiment: {exp_name_full}")
modalities = list(args.modalities)
num_m = len(modalities)
fold_id = args.fold if args.fold is not None else 0
print(f"IEMOCAP split id: {fold_id} (fold={args.fold})")

train_loader, val_loader, eval_loader = iemocap_get_dataloader(
    data_root=args.data_root, batch_size=args.batch_size,
    num_workers=args.num_workers, modalities=modalities,
    split_id=fold_id, train_shuffle=True,
    max_seq_len=args.max_seq_len,
    time_compression_ratio=args.time_compression_ratio,
    use_batched_fusion=args.use_batched_fusion,
    available_sessions=args.available_sessions,
)
input_length = args.max_seq_len if args.max_seq_len > 0 else 600


# ============================ Model ============================


v6_cfg = _IEMOCAPV6Cfg(modalities)
inner = DualVideoBottleneckModelV6Downsample(
    cfg=v6_cfg, output_dim=NUM_CLASSES, input_length=input_length,
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
    no_selector=True, use_weighted_factor=False, use_triton=False,
    num_classes=NUM_CLASSES,
    sparse_attn_variant=args.sparse_attn_variant,
    strat_block_size=args.strat_block_size,
    downsample_min_len=args.downsample_min_len,
    use_batched_fusion=args.use_batched_fusion,
    per_modal_distill=args.per_modal_distill,
    per_modal_downsample_min_len=args.per_modal_downsample_min_len,
)
wrapper = V6IEMOCAPFusionCGGMWrapper(inner, modalities).to(device).float()

aux_clfs = AuxModalityClassifiers(
    modalities=modalities, d_model=args.d_model,
    num_classes=NUM_CLASSES, dropout=args.dropout).to(device).float()

profiler_gn = ModalityGradientProfiler(inner, modalities)        # diagnostic
acc_profiler = ModalityAccProfiler(modalities)                    # drives drop

optimizer = optim.Adam(wrapper.parameters(), lr=args.lr)
aux_optimizer = optim.Adam(aux_clfs.parameters(), lr=args.cls_lr)
criterion = (FocalLoss(gamma=args.focal_gamma)
             if args.loss == 'focal' else nn.CrossEntropyLoss())
scheduler = None
if args.lr_schedule == 'cosine':
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

print(f'Parameters fusion: {count_model_parameters(wrapper)}')
print(f'Parameters aux clfs: {count_model_parameters(aux_clfs)}')
print(f'CGGM: rou={args.cggm_rou}  lambda={args.cggm_lambda}')
print(f'Modality drop (CGGM-acc asymmetric):')
print(f'   epoch [   0, {int(num_epochs*0.2):>3d})  -> 0')
print(f'   epoch [{int(num_epochs*0.2):>3d}, {int(num_epochs*0.5):>3d})  '
      f'-> linear 0 -> {args.max_modality_drop}')
print(f'   epoch [{int(num_epochs*0.5):>3d}, {num_epochs:>3d}]  '
      f'-> {args.max_modality_drop}  (per-mod scales by aux acc ratio)')


# ============================ Training loop ============================


best_val_acc, best_val_f1, best_epoch_val = 0.0, 0.0, -1
best_model_state = None
cold_best_val_acc, cold_best_val_f1, cold_best_epoch = 0.0, 0.0, -1
cold_best_model_state = None

current_mod_probs = {m: 0.0 for m in modalities}
profile_log = []
prev_acc = {m: 0.0 for m in modalities}
lgm_prev = None
mod_drop_complete_epoch = int(num_epochs * 0.5)

writer = SummaryWriter(
    comment=f"_{exp_name_full}_fold{args.fold}"
            if args.fold is not None else f"_{exp_name_full}")

for epoch in range(num_epochs):
    mod_base = get_modality_curriculum(epoch, num_epochs, args.max_modality_drop)

    if mod_base > 0 and epoch % args.profile_update_freq == 0:
        if any(len(acc_profiler.accs[m]) > 0 for m in modalities):
            current_mod_probs = acc_profiler.get_asymmetric_dropout_probs(mod_base)
            snap = acc_profiler.get_acc_snapshot()
            profile_log.append({
                'epoch': epoch, 'mod_base': mod_base,
                'per_mod_acc': snap,
                'mod_dropout_probs': {m: round(p, 4)
                                      for m, p in current_mod_probs.items()},
            })
        else:
            current_mod_probs = {m: mod_base for m in modalities}
    elif mod_base == 0:
        current_mod_probs = {m: 0.0 for m in modalities}

    train_loss, train_acc, prev_acc, lgm_prev = train_one_epoch_cggm(
        train_loader, wrapper, aux_clfs, optimizer, aux_optimizer,
        criterion, epoch, device, current_mod_probs,
        modalities, acc_profiler, prev_acc,
        lambda_lgm=args.cggm_lambda, rou=args.cggm_rou,
        clip_grad=args.clip_grad, lgm_prev=lgm_prev)

    val_loss, val_acc, val_y, val_p = evaluate_one_epoch(
        val_loader, wrapper, criterion, epoch, device)
    val_f1 = f1_score(val_y, val_p, average='macro')

    if scheduler is not None:
        scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/acc', train_acc, epoch)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/acc', val_acc, epoch)
    writer.add_scalar('val/f1', val_f1, epoch)
    writer.add_scalar('schedule/mod_base', mod_base, epoch)
    writer.add_scalar('schedule/lr', current_lr, epoch)
    for m, p in current_mod_probs.items():
        writer.add_scalar(f'mod_drop/{m}', p, epoch)
    for m, a in acc_profiler.get_acc_snapshot().items():
        writer.add_scalar(f'aux_acc/{m}', a, epoch)
    if any(len(profiler_gn.grad_norms[m]) > 0 for m in modalities):
        for m, gn in profiler_gn.get_norm_snapshot().items():
            writer.add_scalar(f'grad_norm/{m}', gn, epoch)

    if val_acc > cold_best_val_acc:
        cold_best_val_acc = val_acc
        cold_best_val_f1 = val_f1
        cold_best_epoch = epoch
        cold_best_model_state = wrapper.state_dict()

    if epoch >= mod_drop_complete_epoch and val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_epoch_val = epoch
        best_model_state = wrapper.state_dict()


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

print(f"\nModel selection: {selected_source}")
print(f"  Best : epoch {best_epoch_val}, val_acc={best_val_acc:.4f}")
print(f"  Cold : epoch {cold_best_epoch}, val_acc={cold_best_val_acc:.4f}")

ckpt_name = (f"best_model_fold{args.fold}.pth"
             if args.fold is not None else "best_val_model.pth")
torch.save(best_model_state, os.path.join(exp_dir, ckpt_name))


# ============================ Clean test eval ============================


print("\n" + "=" * 80)
print("Test-set evaluation (clean, all modalities)")
print("=" * 80)
wrapper.load_state_dict(best_model_state)
wrapper.eval()
_, test_acc, t_y, t_p = evaluate_one_epoch(eval_loader, wrapper, criterion, 0, device)
test_f1 = f1_score(t_y, t_p, average='macro')
print(f"  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
writer.add_scalar('test/acc', test_acc, 0)
writer.add_scalar('test/f1', test_f1, 0)
writer.close()


# ============================ Save ============================


final_grad_norms = profiler_gn.get_norm_snapshot() \
    if any(len(profiler_gn.grad_norms[m]) > 0 for m in modalities) else {}
final_aux_accs = acc_profiler.get_acc_snapshot() \
    if any(len(acc_profiler.accs[m]) > 0 for m in modalities) else {}

config = {
    'experiment_name': exp_name_full, 'dataset': args.dataset,
    'model_variant': 'v6_fusion_only_cggm',
    'modality_drop_strategy': 'staggered_linear_base + cggm_acc_asymmetric_scaling',
    'cggm_rou': args.cggm_rou, 'cggm_lambda': args.cggm_lambda,
    'cls_lr': args.cls_lr,
    'fold': args.fold, 'split_id': fold_id,
    'modalities': modalities, 'num_classes': NUM_CLASSES,
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
    'max_modality_drop': args.max_modality_drop,
    'profile_update_freq': args.profile_update_freq,
    'lr': args.lr, 'lr_schedule': args.lr_schedule,
    'clip_grad': args.clip_grad, 'loss': args.loss,
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
    'final_grad_norms': final_grad_norms,
    'final_aux_accs': final_aux_accs,
    'profile_log': profile_log,
}
results_filename = (f"results_fold{args.fold}.json"
                    if args.fold is not None else "results.json")
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(model_stats, f, indent=4)

profiler_gn.remove_hooks()

print(f"\nBest Model (Epoch {best_epoch_val}): val_acc={best_val_acc:.4f}  "
      f"test_acc={test_acc:.4f}")
print(f"Saved to: {exp_dir}/{ckpt_name}")
