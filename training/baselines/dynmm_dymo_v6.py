"""Shared DynMM / DyMo logic on top of the V6 backbone (2-layer per-modality
self-attn encoder + 4-layer bottleneck fusion), reused purely as a
modality-subset-capable encoder/fusion.

----------------------------------------------------------------------------
DynMM  — Dynamic Multimodal Fusion (Xue & Marculescu, CVPR-W 2023).
  *Modality-level* variant: a lightweight GATE picks, per sample, among K
  EXPERT branches.  Each branch is a different modality subset run through the
  SHARED V6 backbone (a genuine restricted forward -> real FLOP difference).
  Branch outputs are combined by the (soft / straight-through-hard) gate
  weights, and a resource regularizer penalizes expected branch cost.

    w        = DiffSoftmax(gate(summary), tau, hard)        # [B, K]
    fused    = Σ_k w_k · expert_k(x_subset_k)               # [B, C]
    L        = CE(fused) + loss_ratio · Σ_k mean_b(w_k) · cost_k
  cost_k = |subset_k| / M  (proxy for the relative encoder FLOP of branch k).
  Temperature anneals temp -> end_temp; gate hardens (Gumbel straight-through)
  after `epoch_hard`.  At inference the hard gate would run a single branch.

----------------------------------------------------------------------------
DyMo  — Inference-Time Dynamic Modality Selection (Du et al., ICLR 2026).
  Backbone is trained mask-robust (random per-batch modality masking, ModDrop
  style) with per-modality classifier heads.  At inference we GREEDILY build a
  per-sample modality subset: start from the single strongest modality, repeat-
  edly ADD the candidate modality with the largest positive quality gain
  (reward = quality(S∪u) - quality(S)); stop when no positive gain remains
  (info-gain stop).  Quality = prediction confidence (max-softmax) of the
  subset-combined logits — a self-contained surrogate for DyMo's offline
  prototype / ICS Gaussian score.  No imputer is used (all modalities present;
  this is the complete-data / efficiency adaptation of DyMo's selection loop).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================ gate / utils ============================

def diff_softmax(logits, tau=1.0, hard=False, dim=-1):
    """Softmax with optional straight-through hard one-hot (DynMM gate)."""
    y_soft = (logits / tau).softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


def _to_dicts_dalia(x, modalities, mod_slices):
    high, low = {}, {}
    for m in modalities:
        s, e = mod_slices[m]
        high[m] = x[:, :, s:e]
        low[m] = x[:, :, s:e]
    return high, low


def _to_dicts_iemocap(feats, modalities):
    assert len(feats) == 2 * len(modalities), (
        f'Expected {2*len(modalities)} tensors, got {len(feats)}')
    high, low = {}, {}
    for j, m in enumerate(modalities):
        high[m] = feats[2 * j]
        low[m] = feats[2 * j + 1]
    return high, low


class _InputMixin:
    """Splits the dataset batch into V6's (high, low) modality dicts."""

    def _to_dicts(self, inp):
        if self.input_kind == 'dalia':
            return _to_dicts_dalia(inp, self.modalities, self.mod_slices)
        return _to_dicts_iemocap(inp, self.modalities)

    def _run_backbone(self, high, low, training):
        out = self.model(high, low, training=training,
                         return_selection_info=False)
        return out[0] if isinstance(out, tuple) else out


# ============================ DynMM ============================

class DynMMGate(nn.Module):
    """Cheap per-modality summary -> K branch logits."""

    def __init__(self, num_modalities, n_branches, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_modalities, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_branches))

    def forward(self, summary, temp=1.0, hard=False):
        return diff_softmax(self.net(summary), tau=temp, hard=hard, dim=-1)


class DynMMWrapper(_InputMixin, nn.Module):
    def __init__(self, v6_model, modalities, num_classes, input_kind,
                 variates=None, branches=None, gate_hidden=32):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self.input_kind = input_kind
        self.num_classes = num_classes
        if input_kind == 'dalia':
            assert variates is not None
            self.mod_slices, off = {}, 0
            for m in modalities:
                n = variates[m]
                self.mod_slices[m] = (off, off + n)
                off += n
        # Branches = list of modality subsets.  Default: [strongest-single, all].
        if branches is None:
            branches = [[self.modalities[0]], list(self.modalities)]
        self.branches = [[m for m in self.modalities if m in set(b)] for b in branches]
        M = len(self.modalities)
        self.register_buffer(
            'costs', torch.tensor([len(b) / M for b in self.branches], dtype=torch.float32))
        self.gate = DynMMGate(M, len(self.branches), hidden=gate_hidden)

    def _summary(self, high):
        # Per-modality mean activation over (time, feature) -> [B, M].
        cols = [high[m].mean(dim=(1, 2)) for m in self.modalities]
        return torch.stack(cols, dim=1)

    def forward(self, inp, temp=1.0, hard=False):
        high, low = self._to_dicts(inp)
        w = self.gate(self._summary(high), temp=temp, hard=hard)   # [B, K]
        branch_logits = []
        for subset in self.branches:
            h = {m: high[m] for m in subset}
            l = {m: low[m] for m in subset}
            branch_logits.append(self._run_backbone(h, l, self.training))
        stack = torch.stack(branch_logits, dim=1)                  # [B, K, C]
        fused = (w.unsqueeze(-1) * stack).sum(dim=1)               # [B, C]
        return {'fused': fused, 'weights': w}

    def resource_loss(self, weights):
        return (weights.mean(dim=0) * self.costs).sum()

    def expected_cost(self, weights):
        return float((weights.mean(dim=0) * self.costs).sum().item())


# ============================ DyMo ============================

class DyMoWrapper(_InputMixin, nn.Module):
    def __init__(self, v6_model, modalities, num_classes, d_model, input_kind,
                 variates=None):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self.input_kind = input_kind
        self.num_classes = num_classes
        if input_kind == 'dalia':
            assert variates is not None
            self.mod_slices, off = {}, 0
            for m in modalities:
                n = variates[m]
                self.mod_slices[m] = (off, off + n)
                off += n
        self.heads = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
            for m in self.modalities
        })

    def _per_mod_logits(self, high, low, keep):
        h = {m: high[m] for m in keep}
        l = {m: low[m] for m in keep}
        _ = self._run_backbone(h, l, self.training)
        processed = self.model._last_processed_modalities
        return {m: self.heads[m](feat.mean(dim=1)) for m, feat in processed.items()}

    def forward(self, inp, keep=None):
        """Training/raw forward: returns {modality: [B, C]} for the kept set."""
        high, low = self._to_dicts(inp)
        keep = keep if keep is not None else list(self.modalities)
        return self._per_mod_logits(high, low, keep)

    @torch.no_grad()
    def forward_select(self, inp):
        """Inference: greedy per-sample modality selection over per-modality
        logits.  Returns (fused [B,C], final_mask [B,M] bool)."""
        high, low = self._to_dicts(inp)
        logit_map = self._per_mod_logits(high, low, list(self.modalities))
        present = [m for m in self.modalities if m in logit_map]
        stack = torch.stack([logit_map[m] for m in present], dim=1)  # [B, m, C]
        fused, mask = dymo_greedy(stack)
        return fused, mask, present


def _quality(logits):
    """Prediction confidence = max softmax probability.  [B, C] -> [B]."""
    return logits.softmax(dim=-1).max(dim=-1).values


def _combine(stack, mask):
    """Mean of selected per-modality logits.  stack [B,M,C], mask [B,M] -> [B,C]."""
    w = mask.float()
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (w.unsqueeze(-1) * stack).sum(dim=1)


@torch.no_grad()
def dymo_greedy(stack):
    """Greedy add-strongest-first with info-gain stop.  stack [B,M,C]."""
    B, M, C = stack.shape
    dev = stack.device
    # quality of each modality alone -> start from the single strongest.
    qual_each = stack.softmax(dim=-1).max(dim=-1).values            # [B, M]
    mask = torch.zeros(B, M, dtype=torch.bool, device=dev)
    first = qual_each.argmax(dim=1)
    mask[torch.arange(B, device=dev), first] = True
    finished = torch.zeros(B, dtype=torch.bool, device=dev)

    for _ in range(M - 1):
        q_cur = _quality(_combine(stack, mask))                    # [B]
        best_gain = torch.full((B,), -1e9, device=dev)
        best_u = torch.full((B,), -1, dtype=torch.long, device=dev)
        for u in range(M):
            addable = ~mask[:, u]                                  # u not yet selected
            cand = mask.clone()
            cand[:, u] = True
            gain = _quality(_combine(stack, cand)) - q_cur          # [B]
            gain = torch.where(addable, gain, torch.full_like(gain, -1e9))
            better = gain > best_gain
            best_gain = torch.where(better, gain, best_gain)
            best_u = torch.where(better, torch.full_like(best_u, u), best_u)
        step_finish = best_gain <= 0                               # no positive gain
        finished = finished | step_finish
        do = (~finished) & (best_u >= 0)
        if do.any():
            mask[do, best_u[do]] = True
        if finished.all():
            break

    return _combine(stack, mask), mask


def dymo_train_loss(logit_map, y, criterion, alpha_per_mod=1.0):
    """CE on the mean-combined logits + per-modality aux CE."""
    present = list(logit_map.keys())
    stack = torch.stack([logit_map[m] for m in present], dim=1)    # [B, m, C]
    combined = stack.mean(dim=1)
    loss = criterion(combined, y)
    if alpha_per_mod > 0:
        aux = sum(criterion(logit_map[m], y) for m in present) / len(present)
        loss = loss + alpha_per_mod * aux
    return loss, combined
