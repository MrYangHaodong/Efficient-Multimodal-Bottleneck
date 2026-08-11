"""MultiModN baseline classifier (generic, multi-dataset).

Faithful port of MultiModN (Swamy et al., NeurIPS 2023,
https://github.com/EPFLiGHT/MultiModN) to the MAESTRO multivariate
time-series setting, with the structure deliberately aligned to our V6
baselines:

  * Per-modality **temporal encoder** = a ``num_layers``-deep Transformer
    encoder (matches the V6 / FlexMoE per-modality encoder, default 2
    layers) followed by a masked mean-pool over time -> one feature vector
    per modality.
  * **Sequential fusion** = MultiModN's defining mechanism: a single shared
    ``state`` vector is updated one modality at a time.  Each modality's
    update is the original MultiModN ``MLPEncoder`` rule -- *concatenate*
    the modality feature with the running state, then pass through an MLP
    that *overwrites* the state (un-gated).  The MLP depth is
    ``num_layers_fus`` (default 4) so the "fusion depth" matches our V6
    2-encoder + 4-fusion config.
  * **Decoder after every step**: a shared prediction head reads the state
    after each modality, giving per-prefix logits ``[B, M, C]`` -- the
    "readable after any prefix" property MultiModN is built around.  The
    training script supervises *all* prefixes (mean CE over steps).
  * **Missing / dropped modality** = MultiModN simply skips that modality's
    state update (originally ``if nan: continue``).  We implement the
    batched, per-sample analogue used by the V6 dyna curriculum: for a
    dropped sample the candidate state is masked out and the previous state
    is carried forward unchanged.

Input handling mirrors :class:`GenericFlexMoEClassifier`:
  * ``input_mode='float'`` (IEMOCAP raw features): one Linear ``V_m -> d``.
  * ``input_mode='sax'``  (DaliaHAR SAX tokens): Embedding + Linear
    ``V_m*token_dim -> d``.

Exposes ``input_projections`` (a ``ModuleDict`` of per-modality Linears) so
``ModalityGradientProfiler`` can hook per-modality gradient norms exactly as
it does for the FlexMoE/ShaSpec baselines.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from common.loader_utils import _build_channel_map, _concat_iemocap

__all__ = ['GenericMultiModNClassifier']


def _mlp(in_dim, hidden, out_dim, n_layers, dropout):
    """An ``n_layers``-deep MLP. n_layers<=1 -> a single Linear."""
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class _TemporalEncoder(nn.Module):
    """Per-modality temporal Transformer encoder (matches FlexMoE/V6)."""

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


class GenericMultiModNClassifier(nn.Module):
    """MultiModN over a generic multivariate input.

    Args mirror :class:`GenericFlexMoEClassifier` so the dyna training
    scripts are symmetric.

    Args:
        cfg:            object exposing ``modalities``, ``variates``,
                        ``num_classes``.
        input_length:   T (sequence length after the data-loader resampler).
        input_mode:     ``'float'`` or ``'sax'``.
        d_model:        per-modality encoder width.
        num_layers:     per-modality temporal-encoder depth (V6 encoder).
        num_layers_fus: depth of the per-modality state-update MLP
                        (MultiModN "encoder"; V6 fusion depth).
        num_layers_pred: decoder/prediction-head depth.
        state_size:     shared-state width (defaults to ``d_model``).
    """

    def __init__(self, cfg, input_length: int, *,
                 input_mode: str = 'float',
                 alphabet_size: int = 20, token_dim: int = 16,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 2,
                 num_layers_fus: int = 4, num_layers_pred: int = 2,
                 dropout: float = 0.1, state_size: int = None,
                 seq_random_order: bool = False):
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
        self.state_size = int(state_size or d_model)
        self.num_layers = num_layers
        self.num_layers_fus = num_layers_fus
        self.num_layers_pred = num_layers_pred
        # If True, the modality processing order is re-shuffled every training
        # forward (per batch). Eval always uses the canonical cfg order. This is
        # an order-robustness ablation -- the original MultiModN uses a fixed
        # order, so leave it off for the faithful baseline.
        self.seq_random_order = seq_random_order

        if input_mode == 'sax':
            self.token_embedding = nn.Embedding(
                alphabet_size + 1, token_dim, padding_idx=0)
            in_dim = lambda m: self.variates[m] * token_dim  # noqa: E731
        else:
            self.token_embedding = None
            in_dim = lambda m: self.variates[m]              # noqa: E731

        # Per-modality input projection (hooked by ModalityGradientProfiler).
        self.input_projections = nn.ModuleDict({
            m: nn.Linear(in_dim(m), d_model) for m in self.modalities
        })
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, input_length + 64, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.encoders = nn.ModuleDict({
            m: _TemporalEncoder(d_model, nhead, num_layers, dropout)
            for m in self.modalities
        })

        # MultiModN modality-specific state updaters: concat([feat, state])
        # -> MLP -> new state (un-gated overwrite).
        self.state_updaters = nn.ModuleDict({
            m: _mlp(d_model + self.state_size, self.state_size,
                    self.state_size, num_layers_fus, dropout)
            for m in self.modalities
        })
        self.state_norm = nn.LayerNorm(self.state_size)

        # Learnable initial state (MultiModN TrainableInitState).
        self.init_state = nn.Parameter(torch.zeros(1, self.state_size))
        nn.init.trunc_normal_(self.init_state, std=0.02)

        # Shared decoder / prediction head, applied after every step.
        self.decoder = _mlp(self.state_size, self.state_size,
                            cfg.num_classes, num_layers_pred, dropout)

    # ------------------------------------------------------------------
    def _encode_modality(self, mod_input, m):
        """Encode one modality to a pooled feature vector ``[B, d_model]``.

        ``mod_input``: ``(B, T, V_m)`` -- long tokens (sax) or float.
        """
        if self.input_mode == 'sax':
            emb = self.token_embedding(mod_input.long())     # (B,T,V_m,tok)
            B, T, V_m, _ = emb.shape
            x = emb.reshape(B, T, V_m * self.token_dim)
            present = (mod_input != 0).any(dim=2).float().view(B, T, 1)
        else:
            B, T, V_m = mod_input.shape
            x = mod_input
            present = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)

        x = self.input_projections[m](x) + self.pos_embedding[:, :T, :]
        x = x * present
        x = self.encoders[m](x)                              # (B,T,d_model)
        denom = present.sum(dim=1).clamp(min=1.0)            # (B,1)
        feat = (x * present).sum(dim=1) / denom              # (B,d_model)
        return feat

    def forward(self, x, modality_dropout_probs=None, training=False, keep=None):
        """``x``: ``(B, T, sum(variates))`` -- float or long per ``input_mode``.

        ``keep``: optional iterable of modality names to USE; all others are
        skipped (their state update is masked out for every sample). Overrides
        ``modality_dropout_probs`` -- used for fixed-subset evaluation.

        Returns ``(final_logits, aux)`` where
        ``aux['prefix_logits']`` is ``[B, M, C]`` (one decode per modality
        step) and ``aux['observed']`` is the ``[B, M]`` keep-mask.
        """
        B = x.shape[0]
        M = self.num_modalities
        device = x.device

        observed = torch.ones(B, M, dtype=torch.bool, device=device)
        if keep is not None:
            keep_set = set(keep)
            for i, m in enumerate(self.modalities):
                observed[:, i] = (m in keep_set)
        elif training and modality_dropout_probs is not None:
            for i, m in enumerate(self.modalities):
                if isinstance(modality_dropout_probs, dict):
                    prob = float(modality_dropout_probs.get(m, 0.0))
                else:
                    prob = float(modality_dropout_probs)
                if prob > 0:
                    drop = torch.rand(B, device=device) < prob
                    observed[drop, i] = False
            all_dropped = ~observed.any(dim=1)
            if all_dropped.any():
                rnd = torch.randint(0, M, (int(all_dropped.sum().item()),),
                                    device=device)
                observed[all_dropped, rnd] = True

        modality_inputs = {m: x[:, :, self.channel_map[m]]
                           for m in self.modalities}

        # Processing order: canonical (cfg) order, re-shuffled per batch when
        # order-randomization is on and we are training.
        order = list(range(M))
        if training and self.seq_random_order:
            order = torch.randperm(M, device=device).tolist()

        state = self.init_state.expand(B, -1).contiguous()
        prefix_logits = []
        for i in order:                                           # i = modality index
            m = self.modalities[i]
            feat = self._encode_modality(modality_inputs[m], m)   # (B,d)
            cand = self.state_updaters[m](torch.cat([feat, state], dim=-1))
            cand = self.state_norm(cand)
            keep_m = observed[:, i].to(cand.dtype).unsqueeze(-1)  # (B,1)
            state = keep_m * cand + (1.0 - keep_m) * state        # skip if dropped
            prefix_logits.append(self.decoder(state))

        prefix_logits = torch.stack(prefix_logits, dim=1)         # (B,M,C) -- in processing order
        final_logits = self.decoder(state)                        # (B,C)
        aux = {'prefix_logits': prefix_logits, 'observed': observed, 'order': order}
        return final_logits, aux
