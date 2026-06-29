"""ShaSpec — Shared-Specific multimodal baseline (generic, multi-dataset).

Self-contained port of ``GenericShaSpecClassifier`` from
``multimodal_model/shaspec_baseline.py``.  ShaSpec splits each modality into a
SHARED encoder (one common encoder, weights tied across modalities) and a
per-modality SPECIFIC encoder; the two streams are fused by a strict
compositional layer (shared + projected residual).  Missing modalities reuse
the first-available shared feature as a surrogate.  A domain classifier on the
specific features encourages modality-discriminative specifics.

Supports both raw multivariate (``input_mode='float'``) and SAX-token inputs
(``input_mode='sax'``).  ``modality_dropout_probs`` (scalar or per-modality
dict) is applied directly inside ``forward`` so the trainer wrapper stays
trivial.

forward(x) with x=(B,T,sum(variates)) [float, or long-castable sax]:
  returns dict(logits, shared_features, specific_features, fused_features,
               domain_logits, available_mask).

Companion loss: ``shaspec_losses`` =
  task CE + alpha * shared-consistency (L1 alignment across modalities'
  shared features) + beta * specific-domain CE (modality classification).
"""
from __future__ import annotations

import random
from typing import Dict, List

import torch
import torch.nn as nn

from common.loader_utils import _build_channel_map

__all__ = ['GenericShaSpecClassifier', 'shaspec_losses']


class _TemporalEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.encoder(x))


class _StrictCompositionalLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, shared_feat, specific_feat):
        residual = self.proj(torch.cat((shared_feat, specific_feat), dim=-1))
        return shared_feat + residual


class GenericShaSpecClassifier(nn.Module):
    """ShaSpec adapted from ``DaliaShaSpecClassifier``, accepting any
    cfg-style object (needs ``.modalities``, ``.variates``, ``.num_classes``)."""

    def __init__(self, cfg, input_length: int, *,
                 input_mode: str = 'float',
                 alphabet_size: int = 20, token_dim: int = 16,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        assert input_mode in ('float', 'sax')
        self.modalities: List[str] = list(cfg.modalities)
        self.variates: Dict[str, int] = dict(cfg.variates)
        self.num_modalities = len(self.modalities)
        self.channel_map = _build_channel_map(self.modalities, self.variates)
        self.input_length = input_length
        self.input_mode = input_mode
        self.alphabet_size = alphabet_size
        self.token_dim = token_dim
        self.d_model = d_model

        if input_mode == 'sax':
            self.token_embedding = nn.Embedding(
                alphabet_size + 1, token_dim, padding_idx=0)
            in_dim = lambda m: self.variates[m] * token_dim  # noqa: E731
        else:
            self.token_embedding = None
            in_dim = lambda m: self.variates[m]              # noqa: E731

        self.shared_input_projections = nn.ModuleDict({
            m: nn.Linear(in_dim(m), d_model) for m in self.modalities
        })
        self.input_projections = nn.ModuleDict({
            m: nn.Linear(in_dim(m), d_model) for m in self.modalities
        })
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, input_length, d_model))
        self.shared_enc = _TemporalEncoder(d_model, nhead, num_layers, dropout)
        self.specific_encoders = nn.ModuleDict({
            m: _TemporalEncoder(d_model, nhead, num_layers, dropout)
            for m in self.modalities
        })
        self.compos_layer = _StrictCompositionalLayer(d_model)
        self.domain_classifier = nn.Linear(d_model, self.num_modalities)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_modalities * d_model),
            nn.Linear(self.num_modalities * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, cfg.num_classes),
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _split_modalities(self, x):
        return [x[:, :, self.channel_map[mod]] for mod in self.modalities]

    def _to_sequence(self, mod_input, projection):
        """mod_input: (B, T, V_m) float OR (B, T, V_m) long (sax)."""
        if self.input_mode == 'sax':
            emb = self.token_embedding(mod_input.long())              # (B,T,V_m,tok)
            B, T, V_m, _ = emb.shape
            flat = emb.reshape(B, T, V_m * self.token_dim)
            present = (mod_input != 0).any(dim=2).float().view(B, T, 1)
        else:
            B, T, V_m = mod_input.shape
            flat = mod_input
            present = torch.ones(B, T, 1, device=mod_input.device,
                                 dtype=mod_input.dtype)
        x = projection(flat) + self.pos_embedding[:, :T, :]
        return x * present

    def _encode_shared(self, mod_inputs_list):
        pooled = []
        for mod, mod_input in zip(self.modalities, mod_inputs_list):
            seq = self._to_sequence(mod_input, self.shared_input_projections[mod])
            pooled.append(self.shared_enc(seq).mean(dim=1))
        return torch.stack(pooled, dim=1)

    def _encode_specific(self, mod_inputs_list):
        pooled = []
        for mod, mod_input in zip(self.modalities, mod_inputs_list):
            seq = self._to_sequence(mod_input, self.input_projections[mod])
            pooled.append(self.specific_encoders[mod](seq).mean(dim=1))
        return torch.stack(pooled, dim=1)

    def _first_available_shared(self, shared_feats, available_mask):
        bsz = shared_feats.shape[0]
        if not available_mask.any(dim=1).all():
            available_mask = available_mask.clone()
            empty = ~available_mask.any(dim=1)
            available_mask[empty, 0] = True
        first_idx = available_mask.float().argmax(dim=1)
        return shared_feats[torch.arange(bsz, device=shared_feats.device),
                            first_idx]

    @staticmethod
    def _sample_modality_keep(batch_size, num_modalities,
                              modality_dropout_probs, modalities, device):
        if modality_dropout_probs is None:
            return torch.ones(batch_size, num_modalities,
                              dtype=torch.bool, device=device)
        # Build per-modality drop prob (scalar or per-mod dict).
        if isinstance(modality_dropout_probs, dict):
            probs = [float(modality_dropout_probs.get(m, 0.0))
                     for m in modalities]
        else:
            p = float(modality_dropout_probs)
            probs = [p for _ in modalities]
        if all(p <= 0 for p in probs):
            return torch.ones(batch_size, num_modalities,
                              dtype=torch.bool, device=device)
        keep_list = [0 if torch.rand(1).item() < p else 1 for p in probs]
        keep_row = torch.tensor(keep_list, dtype=torch.bool, device=device)
        if not keep_row.any():
            keep_row[random.randrange(num_modalities)] = True
        return keep_row.unsqueeze(0).expand(batch_size, -1).contiguous()

    def forward(self, x, available_mask=None, modality_dropout_probs=None,
                training=False):
        """x: (B, T, sum(variates))  -- float or long depending on input_mode.

        If ``available_mask`` is None and ``training`` is True, sample a
        modality-keep mask from ``modality_dropout_probs`` and zero the
        corresponding channel slices.
        """
        if available_mask is None and training and modality_dropout_probs is not None:
            available_mask = self._sample_modality_keep(
                x.shape[0], self.num_modalities, modality_dropout_probs,
                self.modalities, x.device)
            x = x.clone()
            for idx, mod in enumerate(self.modalities):
                chs = self.channel_map[mod]
                if available_mask is not None:
                    keep_b = available_mask[:, idx].view(-1, 1, 1)
                    x[:, :, chs[0]:chs[-1] + 1] = (
                        x[:, :, chs[0]:chs[-1] + 1] * keep_b.to(x.dtype))

        mod_inputs_list = self._split_modalities(x)

        if available_mask is None:
            if self.input_mode == 'sax':
                parts_present = [(t != 0).any(dim=2).any(dim=1)
                                 for t in mod_inputs_list]
            else:
                parts_present = [(t.abs().sum(dim=2) > 0).any(dim=1)
                                 for t in mod_inputs_list]
            available_mask = torch.stack(parts_present, dim=1)
        available_mask = available_mask.bool()
        if not available_mask.any(dim=1).all():
            available_mask = available_mask.clone()
            available_mask[~available_mask.any(dim=1), 0] = True

        masked_list = []
        for idx, mod_input in enumerate(mod_inputs_list):
            keep_bt = available_mask[:, idx].view(-1, 1, 1).expand_as(mod_input)
            if self.input_mode == 'sax':
                masked_list.append(mod_input * keep_bt.long())
            else:
                masked_list.append(mod_input * keep_bt.to(mod_input.dtype))

        shared_feats = self._encode_shared(masked_list)
        specific_feats = self._encode_specific(masked_list)

        fused_available = self.compos_layer(shared_feats, specific_feats)
        general_shared = self._first_available_shared(
            shared_feats, available_mask)
        fused_feats = torch.where(
            available_mask.unsqueeze(-1), fused_available,
            general_shared.unsqueeze(1).expand_as(fused_available),
        )

        logits = self.classifier(
            fused_feats.reshape(fused_feats.shape[0], -1))
        domain_logits = self.domain_classifier(specific_feats)
        return {'logits': logits, 'shared_features': shared_feats,
                'specific_features': specific_feats,
                'fused_features': fused_feats,
                'domain_logits': domain_logits,
                'available_mask': available_mask}


def shaspec_losses(outputs, labels, alpha: float = 0.1, beta: float = 0.02):
    """Total = task + alpha * shared_consistency + beta * specific_domain.

    Returns ``(total_loss, parts_dict)``.
    """
    import torch.nn.functional as F
    logits = outputs['logits']
    shared = outputs['shared_features']
    domain_logits = outputs['domain_logits']
    device = logits.device
    task_loss = F.cross_entropy(logits, labels)
    pair_losses = []
    M = shared.shape[1]
    for i in range(M):
        pair_losses.append(F.l1_loss(shared[:, i], shared[:, (i + 1) % M]))
    shared_loss = torch.stack(pair_losses).sum()
    domain_targets = torch.arange(M, device=device).view(1, -1).expand(
        labels.shape[0], -1)
    specific_loss = F.cross_entropy(
        domain_logits.reshape(-1, M), domain_targets.reshape(-1))
    total = task_loss + alpha * shared_loss + beta * specific_loss
    return total, {
        'total': float(total.detach().cpu()),
        'task': float(task_loss.detach().cpu()),
        'shared': float(shared_loss.detach().cpu()),
        'specific': float(specific_loss.detach().cpu()),
    }
