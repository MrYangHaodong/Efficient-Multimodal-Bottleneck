"""Shared QMF / PDF confidence-weighted **late-fusion** head + losses.

Both methods replace joint (bottleneck) fusion with a per-modality unimodal
classifier whose logits are combined by a *dynamically predicted quality
weight*.  Used by ``main_qmfpdf_daliahar_late_fusion.py`` and
``main_qmfpdf_iemocap_late_fusion.py``; the V6 backbone is reused only as a
per-modality encoder (its bottleneck-fusion output is ignored — we pool
``model._last_processed_modalities`` instead).

----------------------------------------------------------------------------
QMF  — Provable Dynamic Fusion for Low-Quality Multimodal Data (Zhang et al.,
       ICML 2023).  Energy-based per-modality confidence drives the fusion
       weight, plus a confidence-vs-correctness calibration regularizer.

  logit_m   = head_m( pool(feat_m) )                       # [B, C]
  conf_m    = logsumexp(logit_m)        (= -energy; higher = more confident)
  w_m       = softmax_m( conf_m ).detach()                 # quality weight
  fused     = Σ_m w_m · logit_m
  L = CE(fused) + Σ_m CE(logit_m) + λ_reg · Σ_m L_rank(conf_m, CE_m)
      where L_rank is a pairwise hinge enforcing  CE_i > CE_j ⇒ conf_i < conf_j
      (the unimodal confidence must rank correctness — QMF's robustness reg).

----------------------------------------------------------------------------
PDF  — Predictive Dynamic Fusion (Cao et al., ICML 2024).  Co-Belief =
       mono-confidence (TCP) × holo-confidence (relative across modalities).

  logit_m   = head_m( pool(feat_m) )                       # [B, C]
  mono_m    = sigmoid( conf_head_m( pool(feat_m) ) )       # [B], TCP predictor
  holo_m    = softmax_m( mono_m )                          # relative confidence
  w_m       = (mono_m · holo_m) / Σ_k (mono_k · holo_k)    # Co-Belief weight
  fused     = Σ_m w_m · logit_m
  L = CE(fused) + Σ_m CE(logit_m) + λ_tcp · Σ_m MSE(mono_m, TCP_m)
      TCP_m = softmax(logit_m)[true class]  (detached) — trains mono_m to
      predict the True Class Probability (ConfidNet/TCP-style mono-confidence).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionHead(nn.Module):
    """Per-modality unimodal heads + QMF/PDF confidence weighting.

    forward(z_per) where ``z_per = {modality: [B, H]}`` (pooled features).
    Returns a dict with at least ``logits`` (per-modality), ``fused`` (combined
    [B, C]), ``weights`` ([B, M]) and ``present`` (modalities seen this batch).
    """

    def __init__(self, modalities, d_model, num_classes, method='qmf'):
        super().__init__()
        assert method in ('qmf', 'pdf'), method
        self.modalities = list(modalities)
        self.method = method
        self.num_classes = num_classes
        self.clf_heads = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
            for m in self.modalities
        })
        if method == 'pdf':
            # Mono-confidence (TCP) predictor: per-modality scalar in (0, 1).
            self.conf_heads = nn.ModuleDict({
                m: nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
                for m in self.modalities
            })

    def forward(self, z_per):
        present = [m for m in self.modalities if m in z_per]
        logits = {m: self.clf_heads[m](z_per[m]) for m in present}
        logit_stack = torch.stack([logits[m] for m in present], dim=1)  # [B, M, C]

        if self.method == 'qmf':
            # Energy-based confidence: c_m = logsumexp(logit_m) = -energy_m.
            conf = torch.stack(
                [torch.logsumexp(logits[m], dim=1) for m in present], dim=1)  # [B, M]
            w = torch.softmax(conf.detach(), dim=1)                            # [B, M]
            fused = (w.unsqueeze(-1) * logit_stack).sum(dim=1)                 # [B, C]
            return {'present': present, 'logits': logits, 'fused': fused,
                    'conf': conf, 'weights': w}

        # pdf
        mono = torch.cat(
            [torch.sigmoid(self.conf_heads[m](z_per[m])) for m in present], dim=1)  # [B, M]
        holo = torch.softmax(mono, dim=1)                  # relative (holo) confidence
        cobelief = mono * holo                             # Co-Belief (unnormalized)
        w = cobelief / (cobelief.sum(dim=1, keepdim=True) + 1e-8)              # [B, M]
        fused = (w.unsqueeze(-1) * logit_stack).sum(dim=1)                     # [B, C]
        return {'present': present, 'logits': logits, 'fused': fused,
                'mono': mono, 'holo': holo, 'weights': w}


def qmf_loss(out, y, criterion, lambda_reg=0.5):
    """CE(fused) + Σ CE(unimodal) + λ · confidence-correctness ranking reg."""
    present, logits, fused, conf = (
        out['present'], out['logits'], out['fused'], out['conf'])
    loss = criterion(fused, y)
    for m in present:
        loss = loss + criterion(logits[m], y)
    reg = fused.new_zeros(())
    for i, m in enumerate(present):
        ce = F.cross_entropy(logits[m], y, reduction='none')   # [B] higher = worse
        c = conf[:, i]                                          # [B] higher = confident
        ci_cj = c.unsqueeze(1) - c.unsqueeze(0)                 # [B, B]
        ce_ij = ce.unsqueeze(1) - ce.unsqueeze(0)              # [B, B]
        worse = (ce_ij > 0).float()        # sample i strictly worse than sample j
        # If i is worse than j we want conf_i <= conf_j -> penalize relu(conf_i - conf_j).
        reg = reg + (F.relu(ci_cj) * worse).sum() / (worse.sum() + 1e-8)
    loss = loss + lambda_reg * reg / max(len(present), 1)
    return loss, reg.detach()


def pdf_loss(out, y, criterion, lambda_tcp=1.0):
    """CE(fused) + Σ CE(unimodal) + λ · TCP regression for mono-confidence."""
    present, logits, fused, mono = (
        out['present'], out['logits'], out['fused'], out['mono'])
    loss = criterion(fused, y)
    for m in present:
        loss = loss + criterion(logits[m], y)
    l_tcp = fused.new_zeros(())
    for i, m in enumerate(present):
        with torch.no_grad():
            tcp = torch.softmax(logits[m], dim=1).gather(
                1, y.unsqueeze(1)).squeeze(1)                  # [B] true-class prob
        l_tcp = l_tcp + F.mse_loss(mono[:, i], tcp)
    loss = loss + lambda_tcp * l_tcp / max(len(present), 1)
    return loss, l_tcp.detach()


def compute_loss(method, out, y, criterion, lambda_reg, lambda_tcp):
    if method == 'qmf':
        return qmf_loss(out, y, criterion, lambda_reg)
    return pdf_loss(out, y, criterion, lambda_tcp)
