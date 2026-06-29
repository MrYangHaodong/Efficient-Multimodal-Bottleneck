"""
CrossAttnV6Clf — fully standalone module.

Self-contained reimplementation of
``models/cross_attn_v6_clf.py`` (old root) with all external
dependencies inlined.  No imports from ``models/`` or any other old-root
path.  Importable directly as::

    from multimodal_model.cross_attn_v6_clf import CrossAttnV6Clf

Inlined classes:

  From the original ``cross_attn_v6_clf.py``
    ProbSparseAttention                 (informer-baseline, per-query sampling)
    SparseMoEFeedForward                (ReLU experts, no softmax on gate)
    InformerEncoderLayerWithMoE         (fusion encoder layer)
    ModalityPositionalEncoder, TemporalPositionalEncoder
    modality_dropout
    CrossAttnV6Clf                       (the top-level model)

  From ``dual_video_bottleneck_model_topk_v6_standalone.py`` (per-modal encoder)
    _BNProbSparseAttention              (alias for the bottleneck-aware
                                         ProbSparseAttention used by
                                         TransformerBlock when sparse attn
                                         is enabled)
    _BNProbSparseAttentionOpt           (optimised + stratified variants)
    _BNSparseMoEFeedForward             (GELU experts, softmax-on-topk gate;
                                         used by TransformerBlock when
                                         use_sparse_moe=True)
    TransformerBlock                    (per-modal encoder, no distill)
    _DualPathDownsample                 (Conv+MaxPool sum + GELU + LN)
    TransformerBlockDS                  (per-modal encoder, with distill)

Only standard imports: torch, numpy, math, typing.
"""
from __future__ import annotations

from math import ceil, log, sqrt
import math as _math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Per-modal encoder dependencies (inlined from
# dual_video_bottleneck_model_topk_v6_standalone.py).  Renamed with a
# leading underscore and a ``_BN`` prefix ("bottleneck-aware") to avoid
# colliding with the simpler informer-baseline ProbSparseAttention /
# SparseMoEFeedForward defined further below.
# ===========================================================================


class _BNProbSparseAttention(nn.Module):
    """Bottleneck-aware ProbSparse attention used inside TransformerBlock.

    Mirrors the original implementation in
    ``dual_video_bottleneck_model_topk_v6_standalone.py``.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, factor=5, scale=None,
                 n_bottleneck=16, bottleneck_head=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.scale = scale or 1. / sqrt(self.d_head)

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.n_bottleneck = n_bottleneck
        self.bottleneck_head = bottleneck_head

        self._record_attention = False
        self._attention_weights = None

    def forward(self, x, factor):
        B, L, _ = x.size()

        Q = self.q_linear(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K, V = self.kv_linear(x).chunk(2, dim=-1)
        K = K.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        U_part = int(min(factor * ceil(log(max(L, 2))), max(L - self.n_bottleneck, 1)))
        U_part = max(U_part, 1)
        u = max(int(min(factor * ceil(log(max(L, 2))), L)), 1)

        if self.n_bottleneck != 0 and L > self.n_bottleneck:
            if self.bottleneck_head:
                non_bottleneck_samples = torch.randint(
                    self.n_bottleneck, L, (L, U_part), device=x.device)
                bottleneck_indices = (torch.arange(self.n_bottleneck, device=x.device)
                                      .unsqueeze(0).expand(L, -1))
            else:
                non_bottleneck_samples = torch.randint(
                    0, L - self.n_bottleneck, (L, U_part), device=x.device)
                bottleneck_indices = (torch.arange(L - self.n_bottleneck, L, device=x.device)
                                      .unsqueeze(0).expand(L, -1))
            index_sample = torch.cat([bottleneck_indices, non_bottleneck_samples], dim=1)
        else:
            index_sample = torch.randint(L, (L, U_part), device=x.device)

        n_s = index_sample.shape[1]
        flat_idx = index_sample.reshape(-1)
        flat_idx = flat_idx[None, None, :, None].expand(B, self.n_heads, -1, self.d_head)
        K_sample = K.gather(2, flat_idx).view(B, self.n_heads, L, n_s, self.d_head)
        QK_sampled = torch.matmul(Q.unsqueeze(3), K_sample.transpose(-2, -1)).squeeze(3)
        M = QK_sampled.max(-1)[0] - QK_sampled.mean(-1)
        _, top_idx = torch.topk(M, u, dim=-1)

        top_queries = Q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head))

        if self._record_attention:
            scores = torch.matmul(top_queries, K.transpose(-2, -1)) * self.scale
            attn = F.softmax(scores, dim=-1)
            with torch.no_grad():
                sparse_full = torch.zeros(B, self.n_heads, L, L, device=x.device)
                b_idx = torch.arange(B, device=x.device)[:, None, None]
                h_idx = torch.arange(self.n_heads, device=x.device)[None, :, None]
                sparse_full[b_idx, h_idx, top_idx, :] = attn
                self._attention_weights = sparse_full.mean(dim=1).detach()
            context_update = torch.matmul(attn, V)
        else:
            context_update = F.scaled_dot_product_attention(
                top_queries, K, V, scale=self.scale)

        context = V.mean(dim=2, keepdim=True).expand(-1, -1, L, -1).clone()
        batch_idx = torch.arange(B, device=x.device)[:, None, None]
        head_idx = torch.arange(self.n_heads, device=x.device)[None, :, None]
        context[batch_idx, head_idx, top_idx, :] = context_update

        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(context)


class _BNProbSparseAttentionOpt(nn.Module):
    """Optimised bottleneck-aware ProbSparse (shared key sampling +
    SDPA + scatter_), plus stratified / per-block stratified variants.
    Mirrors ``ProbSparseAttentionOpt`` from the standalone file.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, factor=5, scale=None,
                 n_bottleneck=16, bottleneck_head=True,
                 sampling_strategy='global', strat_block_size=8):
        super().__init__()
        assert sampling_strategy in ('global', 'stratified', 'stratified_block'), \
            f"sampling_strategy must be 'global'|'stratified'|'stratified_block', got {sampling_strategy!r}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.scale = scale or 1. / sqrt(self.d_head)

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.n_bottleneck = n_bottleneck
        self.bottleneck_head = bottleneck_head
        self.sampling_strategy = sampling_strategy
        self.strat_block_size = max(int(strat_block_size), 1)

        self._record_attention = False
        self._attention_weights = None

    def _sample_non_bottleneck(self, lo, hi, U_part, device):
        n = hi - lo
        if self.sampling_strategy == 'stratified':
            edges = torch.linspace(0, n, U_part + 1, device=device)
            bin_starts = edges[:-1].long()
            bin_widths = (edges[1:].long() - bin_starts).clamp(min=1)
            offsets = (torch.rand(U_part, device=device) * bin_widths.float()).long()
            idx_local = (bin_starts + offsets).clamp(max=n - 1)
            return lo + idx_local
        return torch.randint(lo, hi, (U_part,), device=device)

    def _sample_non_bottleneck_blocked(self, lo, hi, n_blocks, U_part, device):
        n = hi - lo
        edges = torch.linspace(0, n, U_part + 1, device=device)
        bin_starts = edges[:-1].long()
        bin_widths = (edges[1:].long() - bin_starts).clamp(min=1)
        offsets = (torch.rand(n_blocks, U_part, device=device)
                   * bin_widths.float()).long()
        idx_local = (bin_starts.unsqueeze(0) + offsets).clamp(max=n - 1)
        return lo + idx_local

    def forward(self, x, factor):
        B, L, _ = x.size()
        H, d_h = self.n_heads, self.d_head

        Q = self.q_linear(x).view(B, L, H, d_h).transpose(1, 2)
        K, V = self.kv_linear(x).chunk(2, dim=-1)
        K = K.view(B, L, H, d_h).transpose(1, 2)
        V = V.view(B, L, H, d_h).transpose(1, 2)

        U_part = int(min(factor * ceil(log(L)), L - self.n_bottleneck))
        U_part = max(U_part, 1)
        u = int(min(factor * ceil(log(L)), L))

        if self.sampling_strategy == 'stratified_block':
            S = self.strat_block_size
            n_blocks = (L + S - 1) // S
            if self.n_bottleneck != 0:
                if self.bottleneck_head:
                    non_bot = self._sample_non_bottleneck_blocked(
                        self.n_bottleneck, L, n_blocks, U_part, x.device)
                    bot = (torch.arange(self.n_bottleneck, device=x.device)
                           .unsqueeze(0).expand(n_blocks, -1))
                else:
                    non_bot = self._sample_non_bottleneck_blocked(
                        0, L - self.n_bottleneck, n_blocks, U_part, x.device)
                    bot = (torch.arange(L - self.n_bottleneck, L, device=x.device)
                           .unsqueeze(0).expand(n_blocks, -1))
                idx_s = torch.cat([bot, non_bot], dim=1)
            else:
                idx_s = self._sample_non_bottleneck_blocked(0, L, n_blocks, U_part, x.device)
            n_s = idx_s.shape[1]

            K_sample = K.index_select(2, idx_s.reshape(-1)).view(B, H, n_blocks, n_s, d_h)

            L_pad = n_blocks * S
            Q_pad = F.pad(Q, (0, 0, 0, L_pad - L)) if L_pad > L else Q
            Q_blk = Q_pad.view(B, H, n_blocks, S, d_h)
            QK_samp = torch.matmul(Q_blk, K_sample.transpose(-2, -1))

            M = QK_samp.max(-1).values - QK_samp.mean(-1)
            M = M.view(B, H, L_pad)[:, :, :L]
            _, top_idx = torch.topk(M, u, dim=-1)
        else:
            if self.n_bottleneck != 0:
                if self.bottleneck_head:
                    non_bot = self._sample_non_bottleneck(self.n_bottleneck, L, U_part, x.device)
                    bot = torch.arange(self.n_bottleneck, device=x.device)
                else:
                    non_bot = self._sample_non_bottleneck(0, L - self.n_bottleneck, U_part, x.device)
                    bot = torch.arange(L - self.n_bottleneck, L, device=x.device)
                idx_s = torch.cat([bot, non_bot], dim=0)
            else:
                idx_s = self._sample_non_bottleneck(0, L, U_part, x.device)

            K_sample = K[:, :, idx_s, :]
            QK_samp = torch.matmul(Q, K_sample.transpose(-2, -1))

            M = QK_samp.max(-1).values - QK_samp.mean(-1)
            _, top_idx = torch.topk(M, u, dim=-1)

        top_q = Q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, d_h))

        ctx_up = F.scaled_dot_product_attention(top_q, K, V, scale=self.scale)

        if self._record_attention:
            with torch.no_grad():
                scores = torch.matmul(top_q, K.transpose(-2, -1)) * self.scale
                attn = F.softmax(scores, dim=-1)
                sparse_full = torch.zeros(B, H, L, L, device=x.device)
                b_idx = torch.arange(B, device=x.device)[:, None, None]
                h_idx = torch.arange(H, device=x.device)[None, :, None]
                sparse_full[b_idx, h_idx, top_idx, :] = attn
                self._attention_weights = sparse_full.mean(dim=1).detach()

        context = V.mean(dim=2).unsqueeze(2).expand(-1, -1, L, -1).contiguous()
        context.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, d_h), ctx_up)

        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(context)


class _BNSparseMoEFeedForward(nn.Module):
    """GELU + softmax-on-topk Sparse MoE used by TransformerBlock when
    ``use_sparse_moe=True``.  Renamed to avoid collision with the
    informer-baseline SparseMoEFeedForward defined below.
    """

    def __init__(self, d_model, expert_dim=256, num_experts=4, k=1, log_activations=False):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.log_activations = log_activations
        self.logged_expert_ids = []

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        gate_scores = self.gate(x_flat)
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)
        topk_scores = F.softmax(topk_scores, dim=-1)

        if self.log_activations:
            self.logged_expert_ids.append(topk_indices.detach().cpu())

        all_out = torch.stack([e(x_flat) for e in self.experts], dim=1)
        idx = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        chosen = all_out.gather(1, idx)
        output = (topk_scores.unsqueeze(-1) * chosen).sum(dim=1)

        return output.reshape(B, T, D)

    def get_activation_logs(self):
        return torch.cat(self.logged_expert_ids, dim=0).numpy() if self.logged_expert_ids else None


class TransformerBlock(nn.Module):
    """Transformer encoder block with self-attention and MLP (dense or sparse MoE).

    Mirrors the standalone ``TransformerBlock``.  Triton paths are
    omitted (the standalone file gracefully falls back when triton is
    unavailable, so this implementation matches that fallback path).
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_dim: int,
                 dropout_rate: float = 0.1,
                 use_sparse_moe: bool = False,
                 num_experts: int = 4,
                 expert_k: int = 1,
                 use_sparse_attn: bool = False,
                 n_bottleneck: int = 16,
                 bottleneck_head: bool = True,
                 factor: int = 5,
                 use_triton: bool = False,
                 sparse_attn_variant: str = 'orig',
                 strat_block_size: int = 8):
        super().__init__()

        self.use_sparse_moe = use_sparse_moe
        self.use_sparse_attn = use_sparse_attn
        self.factor = factor
        # Triton path is unavailable in this standalone module — always
        # falls back to PyTorch ProbSparse implementations.
        self.use_triton = False
        self.sparse_attn_variant = sparse_attn_variant

        self._record_attention = False
        self._attention_weights = None

        self.ln1 = nn.LayerNorm(hidden_size)
        if not use_sparse_attn:
            self.attn = nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
        elif sparse_attn_variant in ('opt', 'opt_strat', 'opt_strat_blk'):
            _strategy_map = {
                'opt':           'global',
                'opt_strat':     'stratified',
                'opt_strat_blk': 'stratified_block',
            }
            self.attn = _BNProbSparseAttentionOpt(
                hidden_size, num_heads, dropout=dropout_rate,
                factor=factor, n_bottleneck=n_bottleneck,
                bottleneck_head=bottleneck_head,
                sampling_strategy=_strategy_map[sparse_attn_variant],
                strat_block_size=strat_block_size,
            )
        else:
            self.attn = _BNProbSparseAttention(
                hidden_size,
                num_heads,
                dropout=dropout_rate,
                factor=factor,
                n_bottleneck=n_bottleneck,
                bottleneck_head=bottleneck_head,
            )
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.self_attn = self.attn

        if use_sparse_moe:
            self.mlp = _BNSparseMoEFeedForward(
                d_model=hidden_size,
                expert_dim=mlp_dim,
                num_experts=num_experts,
                k=expert_k,
                log_activations=False
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(mlp_dim, hidden_size),
                nn.Dropout(dropout_rate)
            )

    def forward(self, x: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False, factor=None) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        if not self.use_sparse_attn:
            x, attn_w = self.attn(x, x, x, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask,
                                  need_weights=self._record_attention)
            if self._record_attention and attn_w is not None:
                self._attention_weights = attn_w.detach()
        else:
            self.attn._record_attention = self._record_attention
            x = self.attn(x, factor or self.factor)
            if self._record_attention and self.attn._attention_weights is not None:
                self._attention_weights = self.attn._attention_weights
        x = self.dropout1(x)

        x = x + residual
        residual = x
        x = self.ln2(x)

        x = self.mlp(x)
        x = x + residual
        return x


class _DualPathDownsample(nn.Module):
    """Conv1d(stride=2) + MaxPool1d(stride=2) → element-wise sum → GELU → LayerNorm.

    Input/output: [B, L, d] → [B, L//2, d]  (L truncated to even if needed).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 == 1:
            x = x[:, :-1, :]
        xT  = x.transpose(1, 2)
        out = self.act(self.conv(xT)) + self.pool(xT)
        return self.norm(out.transpose(1, 2))


class TransformerBlockDS(TransformerBlock):
    """TransformerBlock with dual-path downsampling between attention and FFN.

    Halves the sequence length per call unless ``skip_ds=True``.
    Mirrors the standalone ``TransformerBlockDS``.
    """

    def __init__(self, *args, downsample_d: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds_module = _DualPathDownsample(downsample_d)

    def forward(self, x: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False,
                factor=None,
                skip_ds: bool = False,
                n_bottleneck: int = 0) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        if not self.use_sparse_attn:
            _is_cabsa = hasattr(self.attn, 'per_sample')
            if _is_cabsa:
                x, attn_w = self.attn(x, x, x, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask,
                                      need_weights=self._record_attention,
                                      n_bottleneck=n_bottleneck)
            else:
                x, attn_w = self.attn(x, x, x, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask,
                                      need_weights=self._record_attention)
            if self._record_attention and attn_w is not None:
                self._attention_weights = attn_w.detach()
        else:
            self.attn._record_attention = self._record_attention
            x = self.attn(x, factor or self.factor)
            if self._record_attention and self.attn._attention_weights is not None:
                self._attention_weights = self.attn._attention_weights
        x = self.dropout1(x)

        # Post-attention residual add
        x = x + residual

        # Dual-path downsample between attention and FFN
        if not skip_ds:
            x = self.ds_module(x)

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual
        return x


# ===========================================================================
# Informer baseline classes — copied verbatim from the original
# cross_attn_v6_clf.py (which itself copied them from
# /files1/haodong/MAESTRO/models/our_models.py).
# ===========================================================================


class ProbSparseAttention(nn.Module):
    """Original ProbSparse (per-query sampling, manual softmax)."""
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5, scale=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.scale = scale or 1. / sqrt(self.d_head)

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, factor):
        B, L, _ = x.size()
        Q = self.q_linear(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K, V = self.kv_linear(x).chunk(2, dim=-1)
        K = K.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        U_part = min(factor * ceil(np.log(L)), L)
        u = min(factor * ceil(np.log(L)), L)

        index_sample = torch.randint(L, (L, U_part), device=x.device)
        K_sample = K.unsqueeze(2).expand(-1, -1, L, -1, -1)
        K_sample = K_sample[:, :, torch.arange(L).unsqueeze(1), index_sample, :]
        QK_sampled = torch.matmul(Q.unsqueeze(3), K_sample.transpose(-2, -1)).squeeze(3)
        M = QK_sampled.max(-1)[0] - QK_sampled.mean(-1)
        _, top_idx = torch.topk(M, u, dim=-1)

        top_queries = Q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head))
        scores = torch.matmul(top_queries, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        context_update = torch.matmul(attn, V)

        context = V.mean(dim=2, keepdim=True).expand(-1, -1, L, -1).clone()
        batch_idx = torch.arange(B)[:, None, None]
        head_idx = torch.arange(self.n_heads)[None, :, None]
        context[batch_idx, head_idx, top_idx, :] = context_update

        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(context)


class SparseMoEFeedForward(nn.Module):
    def __init__(self, d_model, expert_dim=256, num_experts=4, k=1, log_activations=False):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.log_activations = log_activations
        self.logged_expert_ids = []
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, d_model),
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        gate_scores = self.gate(x_flat)
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)
        if self.log_activations:
            self.logged_expert_ids.append(topk_indices.detach().cpu())

        output = torch.zeros_like(x_flat)
        for i in range(self.k):
            expert_ids = topk_indices[:, i]
            one_hot_mask = F.one_hot(expert_ids, self.num_experts).bool()
            for expert_idx in range(self.num_experts):
                expert_mask = one_hot_mask[:, expert_idx]
                if expert_mask.sum() == 0:
                    continue
                selected = x_flat[expert_mask]
                result = self.experts[expert_idx](selected)
                score = topk_scores[expert_mask, i].unsqueeze(1)
                output[expert_mask] += score * result
        return output.reshape(B, T, D)


class InformerEncoderLayerWithMoE(nn.Module):
    """Fusion layer — kept unchanged from /files1/haodong/MAESTRO/."""
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, factor=5,
                 num_experts=4, k=1):
        super().__init__()
        d_ff = d_ff or 1 * d_model
        self.attention = ProbSparseAttention(d_model, n_heads, dropout, factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.moe_ff = SparseMoEFeedForward(d_model, d_ff, num_experts=num_experts, k=k)

    def forward(self, x, factor):
        attn_out = self.attention(x, factor)
        x = self.norm1(x + attn_out)
        x = x.permute(0, 2, 1)
        x = self.conv(x) + self.pool(x)
        x = x.permute(0, 2, 1)
        ff_out = self.moe_ff(x)
        return self.norm2(x + ff_out)


class ModalityPositionalEncoder(nn.Module):
    """Learned temporal_pe + modality embedding. Lazily extends temporal_pe
    if input T exceeds the registered buffer length."""
    def __init__(self, d_model, max_len, num_modalities):
        super().__init__()
        self.d_model = d_model
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.modality_pe = nn.Embedding(num_modalities, d_model)
        nn.init.normal_(self.temporal_pe, mean=0, std=0.02)

    def forward(self, x, modality_id):
        B, T, D = x.shape
        if T > self.temporal_pe.shape[1]:
            extra = T - self.temporal_pe.shape[1]
            new_pad = torch.zeros(1, extra, D, device=self.temporal_pe.device,
                                  dtype=self.temporal_pe.dtype)
            with torch.no_grad():
                extended = torch.cat([self.temporal_pe.data, new_pad], dim=1)
            self.temporal_pe = nn.Parameter(extended)
        pe = self.temporal_pe[:, :T, :]
        me = self.modality_pe(torch.tensor(modality_id, device=x.device))
        return x + pe + me.view(1, 1, D).expand(B, T, D)


class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal temporal position encoder. Lazily extends the buffer if
    input T exceeds the registered max_len."""
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._build_pe(max_len, d_model))

    @staticmethod
    def _build_pe(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-_math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        T = x.shape[1]
        if T > self.pe.shape[1]:
            new_pe = self._build_pe(T, self.d_model).to(self.pe.device, self.pe.dtype)
            self.pe = new_pe
        return x + self.pe[:, :T, :]


def modality_dropout(x, modalities, variates, dropout_prob=0.2, training=True):
    B, T, _ = x.shape
    mask = []
    start = 0
    if training and dropout_prob > 0:
        x = x.clone()
    for m in modalities:
        n = variates[m]
        if training and torch.rand(1).item() < dropout_prob:
            x[:, :, start:start + n] = 0
            mask.append(0.0)
        else:
            mask.append(1.0)
        start += n
    return x, torch.tensor(mask, device=x.device, dtype=torch.float32)


# ===========================================================================
# CrossAttnV6Clf — per-modal encoder replaced with our TransformerBlock(DS).
# ===========================================================================
class CrossAttnV6Clf(nn.Module):
    """Drop-in replacement for CrossAttnTransformerClf with per-modal encoder
    swapped to TransformerBlockDS (our seq encoder).  Fusion stays as
    InformerEncoderLayerWithMoE.

    Args (mirror CrossAttnTransformerClf, plus our seq-encoder switches):
        cfg, num_classes, input_length, d_model, nhead,
        num_layers_per_modal, num_layers, dropout, verbose,
        base_factor, num_experts                 — same as original
        use_sparse_attn:     bool=True           — sparse vs dense in our per-modal
        sparse_attn_variant: 'opt' | 'opt_strat' | 'opt_strat_blk'
        strat_block_size:    int=8
    """

    def __init__(self, cfg, num_classes,
                 input_length=256, d_model=64, nhead=8,
                 num_layers_per_modal=2, num_layers=2,
                 dropout=0.1, verbose=True,
                 base_factor=5, num_experts=4,
                 # Our seq-encoder switches (only affect per-modal)
                 use_sparse_attn=True,
                 sparse_attn_variant='opt',
                 strat_block_size=8,
                 # Distill (matches main_robust_v6_*.py defaults)
                 per_modal_distill=False,
                 per_modal_downsample_min_len=4):
        super().__init__()
        assert sparse_attn_variant in ('orig', 'opt', 'opt_strat', 'opt_strat_blk'), \
            f"sparse_attn_variant must be orig/opt/opt_strat/opt_strat_blk"

        self.modalities = cfg.modalities
        self.variates = cfg.variates
        self.num_modalities = len(self.modalities)
        self.input_length = input_length
        self.d_model = d_model
        self.base_factor = base_factor
        self.num_experts = num_experts
        self.use_sparse_attn = use_sparse_attn
        self.sparse_attn_variant = sparse_attn_variant
        self.per_modal_distill = per_modal_distill
        self.per_modal_downsample_min_len = per_modal_downsample_min_len

        # ── Input projections (kept) ─
        self.input_projections = nn.ModuleDict({
            m: nn.Linear(self.variates[m], d_model) for m in self.modalities
        })

        # ── Position encoders (kept) ─
        self.pos_encoder = ModalityPositionalEncoder(
            d_model=d_model, max_len=input_length, num_modalities=self.num_modalities)
        self.temporal_pos_encoder = TemporalPositionalEncoder(
            d_model=d_model, max_len=input_length)

        # ── Per-modality encoder REPLACED: our seq encoder ──
        if per_modal_distill:
            block_cls = TransformerBlockDS
            block_kwargs = dict(downsample_d=d_model)
        else:
            block_cls = TransformerBlock
            block_kwargs = {}
        self.per_modal_informers = nn.ModuleDict({
            m: nn.ModuleList([
                block_cls(
                    hidden_size=d_model,
                    num_heads=nhead,
                    mlp_dim=d_model * 2,
                    dropout_rate=dropout,
                    use_sparse_moe=False,
                    use_sparse_attn=False,  # use_sparse_attn,
                    n_bottleneck=0,
                    bottleneck_head=False,
                    factor=base_factor,
                    sparse_attn_variant=sparse_attn_variant,
                    strat_block_size=strat_block_size,
                    **block_kwargs,
                ) for _ in range(num_layers_per_modal)
            ]) for m in self.modalities
        })

        # ── Fusion encoder UNCHANGED: original InformerEncoderLayerWithMoE ──
        self.informer_encoder = nn.ModuleList([
            InformerEncoderLayerWithMoE(
                d_model=d_model,
                n_heads=nhead,
                d_ff=d_model * 1,
                dropout=dropout,
                factor=self.base_factor,
                num_experts=num_experts,
                k=1,
            ) for _ in range(num_layers)
        ])

        # ── Classifier (kept) ─
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ── factor_gate (kept) ─
        self.factor_gate = nn.Sequential(
            nn.Linear(self.num_modalities, self.num_modalities),
            nn.ReLU(),
            nn.Linear(self.num_modalities, self.num_modalities),
            nn.Sigmoid(),
        )

        if verbose:
            n_params = sum(p.numel() for p in self.parameters())
            distill_str = (f"TransformerBlockDS (distill, T//2 per layer, "
                           f"min_len={per_modal_downsample_min_len})"
                           if per_modal_distill
                           else "TransformerBlock (no distill, T unchanged per layer)")
            print(f"Initializing CrossAttnV6Clf  (params={n_params:,})")
            print(f"  Modalities         : {self.modalities}")
            print(f"  Per-modal encoder  : {distill_str} × {num_layers_per_modal}")
            print(f"  Fusion encoder     : InformerEncoderLayerWithMoE × {num_layers} (unchanged)")
            print(f"  Sparse attention   : {use_sparse_attn} (variant={sparse_attn_variant})")
            print(f"  base_factor        : {base_factor}  num_experts={num_experts}")

    def forward(self, x, modality_dropout_prob=0.2, training=True):
        """
        x: [B, T, total_features] — modalities concatenated along feature dim
        Returns: (logits, dynamic_factor)
        """
        x, modality_mask = modality_dropout(
            x, self.modalities, self.variates,
            dropout_prob=modality_dropout_prob, training=training)
        dynamic_factor = self.factor_gate(modality_mask) * self.base_factor

        projected_modalities = []
        start_idx = 0
        for idx, modality in enumerate(self.modalities):
            n = self.variates[modality]
            x_m = x[:, :, start_idx:start_idx + n]
            start_idx += n
            x_m = self.input_projections[modality](x_m)
            df = float(dynamic_factor[idx].detach().item())
            mm = float(modality_mask[idx].detach().item())
            factor_m = max(int(ceil(df * mm + 1e-3 * (1.0 - mm))), 1)

            x_m = self.temporal_pos_encoder(x_m)
            for layer in self.per_modal_informers[modality]:
                if self.per_modal_distill:
                    skip_ds = x_m.shape[1] <= self.per_modal_downsample_min_len
                    x_m = layer(x_m, factor=factor_m, skip_ds=skip_ds)
                else:
                    x_m = layer(x_m, factor=factor_m)
            x_m = self.pos_encoder(x_m, modality_id=idx)
            projected_modalities.append(x_m)

        x_cat = torch.cat(projected_modalities, dim=1)

        for layer in self.informer_encoder:
            x_cat = layer(x_cat, self.base_factor)

        x_pooled = torch.mean(x_cat, dim=1)
        return self.classifier(x_pooled), dynamic_factor
