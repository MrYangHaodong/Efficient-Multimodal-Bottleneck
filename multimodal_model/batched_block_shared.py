"""
batched_block_shared.py — Shared-weight transformer block for V6 fusion stage.
=============================================================================

Companion to ``fusion_bmm_parallel.BatchedTransformerBlock`` but with **a
single, modality-shared set of attention QKV / output / LayerNorm / MLP /
downsample weights**, applied identically to every modality slice.

Why a separate class?  ``BatchedTransformerBlock`` stores every parameter
tensor with a leading ``M_total`` dim — one independent set per modality —
which is the right design for per-modality encoders (each modality wants
its own representation space).  In the **fusion** stage that redundancy is
wasted: the modalities are meant to *communicate* through the bottleneck,
and CrossAttn (``models/cross_attn_v6_clf.py``) already demonstrates that
sharing one attention+MLP across modalities in fusion delivers higher
accuracy at roughly half the parameter count.

This block keeps the same external ``forward(x, mod_indices=None)`` API as
``BatchedTransformerBlock`` so it is a drop-in replacement inside the V6
fusion forward loop.  Internally:

  * Weights are regular ``nn.Linear`` / ``nn.Parameter`` *without* an M
    dimension — exactly the layout you would get from a standalone
    transformer block.  Param count is ~ 1 / M of the batched variant.
  * Forward folds ``[M_active, B, L, D]`` into ``[M_active * B, L, D]``,
    runs a standard transformer block once, and unfolds back to
    ``[M_active, B, L_out, D]``.  The bottleneck-token mechanism (where M
    modalities concatenate the same bottleneck slice along ``L``) is
    invariant to this fold — each modality still sees its own ``B`` rows
    plus the bottleneck slice within its own row, and the shared weights
    apply identically across all M slices.
  * ``use_sparse_attn=True`` is supported via a per-call instantiation of
    ``ProbSparseAttention`` (the original sparse attention from
    ``models/cross_attn_v6_clf.py``) — only one set of QKV weights is
    needed because the same projection is shared across modalities.
  * ``use_distill=True`` adds a single Conv1d(k=3,s=2) + MaxPool1d(k=2,s=2)
    dual path along ``L``, again shared across all modalities.

``mod_indices`` is accepted only for API compatibility — since there are
no per-modality weights to gather, this argument is ignored and shared
weights are used for every input slice.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ProbSparse attention (single-modality, shared weights).
# ---------------------------------------------------------------------------
# We use the same ProbSparse mechanic as the V6 ``ProbSparseAttentionOpt``
# (shared key sampling, SDPA top-u, scatter-write) but with the QKV / out
# projections inherited from this block's own weights (so they truly are
# shared across modalities in the M*B-folded forward).  Kept inline here to
# avoid a circular import with ``v6_downsample_opt_batched``.


class _ProbSparseCore(nn.Module):
    """ProbSparse top-k attention core — operates on already-projected QKV.

    Mirrors ``ProbSparseAttentionOpt.forward`` from
    ``v6_downsample_opt_batched.py`` but factored so the caller owns the
    Q/K/V projections (so this block can share them across modalities).

    Inputs:
        Q, K, V: [B, H, L, d_h]  (already split into heads)
    Returns:
        context: [B, L, H * d_h]
    """

    def __init__(self, factor: int, n_bottleneck: int, bottleneck_head: bool,
                 sampling_strategy: str, strat_block_size: int,
                 scale: Optional[float] = None):
        super().__init__()
        assert sampling_strategy in ('global', 'stratified', 'stratified_block'), \
            f"sampling_strategy must be 'global'|'stratified'|'stratified_block'"
        self.factor = factor
        self.n_bottleneck = n_bottleneck
        self.bottleneck_head = bottleneck_head
        self.sampling_strategy = sampling_strategy
        self.strat_block_size = max(int(strat_block_size), 1)
        self.scale = scale

    @staticmethod
    def _sample_non_bottleneck(lo, hi, U_part, device, strategy):
        n = hi - lo
        if strategy == 'stratified':
            edges = torch.linspace(0, n, U_part + 1, device=device)
            bin_starts = edges[:-1].long()
            bin_widths = (edges[1:].long() - bin_starts).clamp(min=1)
            offsets = (torch.rand(U_part, device=device) * bin_widths.float()).long()
            idx_local = (bin_starts + offsets).clamp(max=n - 1)
            return lo + idx_local
        return torch.randint(lo, hi, (U_part,), device=device)

    @staticmethod
    def _sample_non_bottleneck_blocked(lo, hi, n_blocks, U_part, device):
        n = hi - lo
        edges = torch.linspace(0, n, U_part + 1, device=device)
        bin_starts = edges[:-1].long()
        bin_widths = (edges[1:].long() - bin_starts).clamp(min=1)
        offsets = (torch.rand(n_blocks, U_part, device=device)
                   * bin_widths.float()).long()
        idx_local = (bin_starts.unsqueeze(0) + offsets).clamp(max=n - 1)
        return lo + idx_local

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, L, d_h = Q.shape
        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(d_h)

        U_part = int(min(self.factor * math.ceil(math.log(max(L, 2))),
                         L - self.n_bottleneck))
        U_part = max(U_part, 1)
        u = int(min(self.factor * math.ceil(math.log(max(L, 2))), L))

        if self.sampling_strategy == 'stratified_block':
            S = self.strat_block_size
            n_blocks = (L + S - 1) // S
            if self.n_bottleneck != 0:
                if self.bottleneck_head:
                    non_bot = self._sample_non_bottleneck_blocked(
                        self.n_bottleneck, L, n_blocks, U_part, Q.device)
                    bot = (torch.arange(self.n_bottleneck, device=Q.device)
                           .unsqueeze(0).expand(n_blocks, -1))
                else:
                    non_bot = self._sample_non_bottleneck_blocked(
                        0, L - self.n_bottleneck, n_blocks, U_part, Q.device)
                    bot = (torch.arange(L - self.n_bottleneck, L, device=Q.device)
                           .unsqueeze(0).expand(n_blocks, -1))
                idx_s = torch.cat([bot, non_bot], dim=1)
            else:
                idx_s = self._sample_non_bottleneck_blocked(
                    0, L, n_blocks, U_part, Q.device)
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
                    non_bot = self._sample_non_bottleneck(
                        self.n_bottleneck, L, U_part, Q.device, self.sampling_strategy)
                    bot = torch.arange(self.n_bottleneck, device=Q.device)
                else:
                    non_bot = self._sample_non_bottleneck(
                        0, L - self.n_bottleneck, U_part, Q.device, self.sampling_strategy)
                    bot = torch.arange(L - self.n_bottleneck, L, device=Q.device)
                idx_s = torch.cat([bot, non_bot], dim=0)
            else:
                idx_s = self._sample_non_bottleneck(
                    0, L, U_part, Q.device, self.sampling_strategy)
            K_sample = K[:, :, idx_s, :]
            QK_samp = torch.matmul(Q, K_sample.transpose(-2, -1))
            M = QK_samp.max(-1).values - QK_samp.mean(-1)
            _, top_idx = torch.topk(M, u, dim=-1)

        top_q = Q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, d_h))
        ctx_up = F.scaled_dot_product_attention(top_q, K, V, scale=scale)

        context = V.mean(dim=2).unsqueeze(2).expand(-1, -1, L, -1).contiguous()
        context.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, d_h), ctx_up)
        return context.transpose(1, 2).contiguous().view(B, L, H * d_h)


class BatchedTransformerBlockShared(nn.Module):
    """Shared-weight transformer block for V6 fusion.

    Same external ``forward(x, mod_indices=None)`` API as
    ``BatchedTransformerBlock`` but stores **one** set of attention / MLP /
    LN parameters (no leading M dim).  Forward folds the M dim into the
    batch dim so the same weights are applied to every modality.

    Args:
        M_total:     accepted for API compatibility; ignored (no per-modality
                     weights).
        hidden_size: feature dim D.
        num_heads:   attention heads (must divide D).
        mlp_dim:     FFN hidden dim.
        dropout_rate: attention + MLP dropout.
        use_distill: if True, add a Conv1d(k=3,s=2) + MaxPool1d(k=2,s=2) dual
                     path along L (shared across modalities).
        use_sparse_attn: if True, use ProbSparse top-k attention.
        factor:      ProbSparse sampling budget control.
        n_bottleneck: bottleneck token count for sparse attention.
        bottleneck_head: True = bottleneck tokens at the head of the sequence.
        sparse_attn_variant: 'opt' | 'opt_strat' | 'opt_strat_blk'.
        strat_block_size: stratified-block size when variant=='opt_strat_blk'.
    """

    def __init__(self, M_total: int, hidden_size: int, num_heads: int,
                 mlp_dim: int, dropout_rate: float = 0.1,
                 use_distill: bool = False,
                 use_sparse_attn: bool = False,
                 factor: int = 5,
                 n_bottleneck: int = 0,
                 bottleneck_head: bool = True,
                 sampling_strategy: str = 'global',
                 strat_block_size: int = 8):
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f'hidden_size {hidden_size} not divisible by num_heads {num_heads}'
        assert sampling_strategy in ('global', 'stratified', 'stratified_block'), \
            f"sampling_strategy must be 'global'|'stratified'|'stratified_block'"

        self.M_total = M_total            # kept for API parity (ignored)
        self.H = hidden_size
        self.nh = num_heads
        self.dh = hidden_size // num_heads
        self.mlp_dim = mlp_dim
        self.dropout_p = dropout_rate
        self.use_distill = use_distill
        self.use_sparse_attn = use_sparse_attn
        self.factor = factor
        self.n_bottleneck = n_bottleneck
        self.bottleneck_head = bottleneck_head
        self.sampling_strategy = sampling_strategy
        self.strat_block_size = max(int(strat_block_size), 1)
        self.scale = 1.0 / math.sqrt(self.dh)
        self.ln_eps = 1e-5

        # ---- Attention projections (single, shared across all M modalities)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_out = nn.Linear(hidden_size, hidden_size)

        # ---- LayerNorm (Pre-LN), single shared instance
        self.ln1 = nn.LayerNorm(hidden_size, eps=self.ln_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=self.ln_eps)

        # ---- MLP, shared
        self.mlp_fc1 = nn.Linear(hidden_size, mlp_dim)
        self.mlp_fc2 = nn.Linear(mlp_dim, hidden_size)

        # ---- Optional dual-path downsample (mirror of seq's _DualPathDownsample
        #      but with a single Conv1d kernel shared across modalities).
        if use_distill:
            self.ds_conv = nn.Conv1d(hidden_size, hidden_size,
                                     kernel_size=3, stride=2, padding=1)
            self.ds_norm = nn.LayerNorm(hidden_size, eps=self.ln_eps)

        # ---- Optional sparse attention core
        if use_sparse_attn:
            self._sparse_core = _ProbSparseCore(
                factor=factor, n_bottleneck=n_bottleneck,
                bottleneck_head=bottleneck_head,
                sampling_strategy=sampling_strategy,
                strat_block_size=self.strat_block_size,
                scale=self.scale,
            )

    # ------------------------------------------------------------------
    # Attention sub-paths (operate on already-folded [M*B, L, D] input).
    # ------------------------------------------------------------------

    def _attention_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Standard MHA via SDPA. x: [N, L, D] -> [N, L, D] (N = M*B)."""
        N, L, D = x.shape
        qkv = self.qkv(x)                                        # [N, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.reshape(N, L, self.nh, self.dh).transpose(1, 2).contiguous()

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                                        # [N, nh, L, dh]
        attn = attn.transpose(1, 2).contiguous().reshape(N, L, D)
        return self.attn_out(attn)

    def _attention_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """ProbSparse top-k MHA. x: [N, L, D] -> [N, L, D]."""
        N, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # [N, L, nh, dh] -> [N, nh, L, dh]
        q = q.reshape(N, L, self.nh, self.dh).transpose(1, 2)
        k = k.reshape(N, L, self.nh, self.dh).transpose(1, 2)
        v = v.reshape(N, L, self.nh, self.dh).transpose(1, 2)
        context = self._sparse_core(q, k, v)                     # [N, L, D]
        return self.attn_out(context)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Linear-GELU-Dropout-Linear-Dropout. x: [N, L, D] -> [N, L, D]."""
        h = F.gelu(self.mlp_fc1(x))
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        h = self.mlp_fc2(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Shared dual-path downsample (Conv1d k=3 s=2 + MaxPool k=2 s=2).

        Mirrors the per-modality ``_batched_ds`` in BatchedTransformerBlock
        but uses a single Conv1d weight shared across modalities.  Trims an
        odd L beforehand (drops the last token) to keep the integer-division
        output length consistent with the batched variant.

        x: [N, L, D] -> [N, L_new, D] where L_new = (L if even else L-1) // 2.
        """
        if x.shape[1] % 2 == 1:
            x = x[:, :-1, :].contiguous()
        # [N, L, D] -> [N, D, L]
        z = x.transpose(1, 2)
        z_conv = F.gelu(self.ds_conv(z))                         # [N, D, L/2]
        z_pool = F.max_pool1d(z, kernel_size=2, stride=2)        # [N, D, L/2]
        z_out = z_conv + z_pool
        # [N, D, L_new] -> [N, L_new, D]
        out = z_out.transpose(1, 2).contiguous()
        return self.ds_norm(out)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                mod_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [M_active, B, L, D]
            mod_indices: accepted for API parity with BatchedTransformerBlock;
                         IGNORED (no per-modality weights to gather).

        Returns:
            [M_active, B, L_out, D] (L_out = L if not use_distill else L//2).
        """
        del mod_indices  # No per-modality weights — same params for all slices.

        M, B, L, D = x.shape
        # Fold (M, B) -> N so the standard transformer block sees one batched
        # axis.  Shared weights apply identically to every modality slice.
        x_flat = x.reshape(M * B, L, D)

        # ---- Pre-LN attention
        residual = x_flat
        h = self.ln1(x_flat)
        if self.use_sparse_attn:
            h = self._attention_sparse(h)
        else:
            h = self._attention_dense(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        x_flat = h + residual

        # ---- Optional downsample (between attention and FFN, mirrors seq path)
        if self.use_distill:
            x_flat = self._downsample(x_flat)

        # ---- Pre-LN FFN
        residual = x_flat
        h = self.ln2(x_flat)
        h = self._mlp(h)
        x_flat = h + residual

        L_out = x_flat.shape[1]
        return x_flat.reshape(M, B, L_out, D)
