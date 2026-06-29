"""QMF / PDF on a pure late-fusion transformer backbone (NO V6, NO fusion net).

Faithful to the official QMF/PDF design (two *independent* unimodal networks
combined only at the logit level): here each modality gets its OWN deep
Transformer encoder (MultiModN-style, default 6 layers), is pooled and
classified independently, and the per-modality logits are combined by the
QMF energy weighting / PDF Co-Belief head (reused from ``qmf_pdf_late_fusion``).
There is **no cross-modal fusion network**.

Input handling mirrors GenericMultiModNClassifier:
  * ``input_mode='float'`` (IEMOCAP): Linear V_m -> d.
  * ``input_mode='sax'``  (HAR SAX tokens): Embedding + Linear V_m*tok -> d.
A single concatenated ``[B, T, sum(variates)]`` tensor is split per modality
by channel slices.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn

# Reuse the per-modality temporal encoder + channel-map helper from MultiModN.
from multimodal_model.multimodn_baseline import _TemporalEncoder, _build_channel_map

# Reuse the official-style QMF/PDF combine head + losses (lives under script/).
_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'script')
if _SCRIPT not in sys.path:
    sys.path.insert(0, _SCRIPT)
from qmf_pdf_late_fusion import LateFusionHead  # noqa: E402

__all__ = ['GenericLateFusionTransformer', 'IEMOCAPLateFusionWrapper']


class IEMOCAPLateFusionWrapper(nn.Module):
    """Concat the IEMOCAP feats-list highs -> [B,T,sum(V)] -> late-fusion model."""

    def __init__(self, model, modalities):
        super().__init__()
        self.model = model
        self.M = len(modalities)

    def _concat(self, feats):
        return torch.cat([feats[2 * j] for j in range(self.M)], dim=-1)

    def forward(self, feats, **kw):
        return self.model(self._concat(feats), **kw)


class GenericLateFusionTransformer(nn.Module):
    """Per-modality deep Transformer encoders + QMF/PDF logit-level late fusion."""

    def __init__(self, cfg, input_length: int, *,
                 input_mode: str = 'float',
                 alphabet_size: int = 20, token_dim: int = 16,
                 d_model: int = 64, nhead: int = 8, num_layers: int = 6,
                 dropout: float = 0.1, method: str = 'qmf'):
        super().__init__()
        assert input_mode in ('float', 'sax')
        assert method in ('qmf', 'pdf')
        self.modalities: List[str] = list(cfg.modalities)
        self.variates: Dict[str, int] = dict(cfg.variates)
        self.num_modalities = len(self.modalities)
        self.channel_map = _build_channel_map(self.modalities, self.variates)
        self.input_length = input_length
        self.input_mode = input_mode
        self.token_dim = token_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.method = method

        if input_mode == 'sax':
            self.token_embedding = nn.Embedding(alphabet_size + 1, token_dim, padding_idx=0)
            in_dim = lambda m: self.variates[m] * token_dim  # noqa: E731
        else:
            self.token_embedding = None
            in_dim = lambda m: self.variates[m]              # noqa: E731

        # per-modality input projection (hooked by ModalityGradientProfiler)
        self.input_projections = nn.ModuleDict({
            m: nn.Linear(in_dim(m), d_model) for m in self.modalities
        })
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_length, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        # deep per-modality temporal encoder (default 6 layers) -- the unimodal net
        self.encoders = nn.ModuleDict({
            m: _TemporalEncoder(d_model, nhead, num_layers, dropout)
            for m in self.modalities
        })
        # logit-level QMF/PDF combine (per-modality clf heads + confidence weighting)
        self.fusion = LateFusionHead(self.modalities, d_model, cfg.num_classes, method=method)

    def _encode_modality(self, mod_input, m):
        if self.input_mode == 'sax':
            emb = self.token_embedding(mod_input.long())             # (B,T,V_m,tok)
            B, T, V_m, _ = emb.shape
            x = emb.reshape(B, T, V_m * self.token_dim)
            present = (mod_input != 0).any(dim=2).float().view(B, T, 1)
        else:
            B, T, V_m = mod_input.shape
            x = mod_input
            present = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
        x = self.input_projections[m](x) + self.pos_embedding[:, :T, :]
        x = x * present
        x = self.encoders[m](x)                                      # (B,T,d)
        denom = present.sum(dim=1).clamp(min=1.0)
        return (x * present).sum(dim=1) / denom                      # (B,d)

    def forward(self, x, modality_dropout_probs=None, training=False, keep=None):
        """``x``: ``[B, T, sum(variates)]``. Returns the LateFusionHead dict
        (``fused``, ``logits``, ``conf``/``mono``, ``weights``, ``present``).

        Modalities can be dropped per-sample (curriculum) -- here we drop at the
        BATCH level (whole modality excluded from this forward) for simplicity,
        which is what a per-modality late-fusion baseline needs; ``keep`` forces
        a fixed subset (for top-k evaluation)."""
        B = x.shape[0]
        device = x.device

        if keep is not None:
            present_mods = [m for m in self.modalities if m in set(keep)]
        elif training and modality_dropout_probs is not None:
            present_mods = []
            for m in self.modalities:
                p = (float(modality_dropout_probs.get(m, 0.0))
                     if isinstance(modality_dropout_probs, dict) else float(modality_dropout_probs))
                if torch.rand(1).item() >= p:
                    present_mods.append(m)
            if not present_mods:                       # never drop everything
                present_mods = [self.modalities[torch.randint(self.num_modalities, (1,)).item()]]
        else:
            present_mods = list(self.modalities)

        z_per = {m: self._encode_modality(x[:, :, self.channel_map[m]], m) for m in present_mods}
        return self.fusion(z_per)
