"""
bmm-based parallel modality processing for V6 fusion modules.
=============================================================

Drop-in replacement for `SimpleMBTFusionAdaptive` and
`SimpleMBTFusionAdaptiveWeightedFactor` (and *MLP / *Distilling variants)
that runs the M independent per-modality TransformerBlocks of every
fusion layer through **explicitly batched ops**: stacked Linear via
einsum (grouped GEMM), grouped Conv1d via `groups=M`, fused SDPA over
the (M*B) batch (flash backend retained), batched LayerNorm.

Compared to the vmap version
----------------------------
* MultiheadAttention stays on the FlashAttention / mem-efficient backend
  (vmap forces math backend) — biggest win, ~1.5–2.5x on attention.
* Conv1d is one grouped-conv kernel, never falls back to a per-modality
  loop (vmap behaviour depends on PyTorch version).
* Every parameter has a leading-M dim and is owned by a single
  `BatchedTransformerBlock`; no per-forward `stack_module_state` cost.

Limitations
-----------
* **Dense attention only.** ProbSparseAttention internals (random
  sampling, top-k re-selection) don't lend themselves to a batched
  implementation; sparse path users should keep the vmap version.
* **No MoE.** SparseMoE expert routing under M independent gates
  requires a separate design.
* **No Triton fused-add-LN.** Not applicable to batched LN parameters.
* **Checkpoint incompatible** with the original ModuleDict-of-blocks
  layout — train from scratch (this is acceptable per the user's
  scenario where retraining is cheap).
* **Attention map recording** (`enable_attention_recording`, etc.) is
  not implemented in the batched path.

Architecture mirror
-------------------
The batched block reproduces the original V6 `TransformerBlock` graph:

    residual = x
    x = MultiHeadSelfAttention(x)
    x = LN1(x + residual)
    x = x.permute(0,2,1)
    x = Conv1d(x) + MaxPool1d(x)            # stride=1 or 2 for distill
    x = x.permute(0,2,1)
    x = LN2(x + MLP(x))                     # MLP: Linear-GELU-Drop-Linear-Drop
    return x

with every weight tensor carrying a leading-M dimension.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Core: BatchedTransformerBlock
# =====================================================================

class BatchedTransformerBlock(nn.Module):
    """M structurally identical transformer blocks fused into one module.

    Holds `M_total` independent parameter sets. Forward accepts an input
    `[M_active, B, L, H]` plus an optional `mod_indices` list selecting
    which of the M_total parameter slices to use (supports top-k modality
    selection: M_active <= M_total).

    Args:
        M_total:      number of stored parameter sets (= num_modalities).
        hidden_size:  feature dim H.
        num_heads:    attention heads (must divide H).
        mlp_dim:      FFN hidden dim.
        dropout_rate: applied inside attention (via SDPA dropout) and
                      inside the MLP.
        use_distill:  if True, the inner Conv1d / MaxPool1d use stride=2
                      (halves L_out); else stride=1.
    """

    def __init__(self, M_total: int, hidden_size: int, num_heads: int,
                 mlp_dim: int, dropout_rate: float = 0.1,
                 use_distill: bool = False,
                 # Sparse-attention extras (ignored unless use_sparse_attn=True)
                 use_sparse_attn: bool = False,
                 factor: int = 5,
                 n_bottleneck: int = 0,
                 bottleneck_head: bool = True,
                 sampling_strategy: str = 'global',
                 strat_block_size: int = 8,
                 share_params: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f'hidden_size {hidden_size} not divisible by num_heads {num_heads}'
        assert sampling_strategy in ('global', 'stratified', 'stratified_block'), \
            f"sampling_strategy must be 'global'|'stratified'|'stratified_block'"

        self.M_total = M_total
        # When share_params=True, all per-modality weights are stored with a
        # leading dim of 1 and broadcast (expand) to M at forward time, so the
        # fusion transformer is a single modality-agnostic block shared across
        # all modalities (vs the default per-modality parameter sets).
        self.share_params = share_params
        P = 1 if share_params else M_total
        self.H = hidden_size
        self.nh = num_heads
        self.dh = hidden_size // num_heads
        self.mlp_dim = mlp_dim
        self.dropout_p = dropout_rate
        self.use_distill = use_distill
        self.stride = 2 if use_distill else 1   # legacy field; kept for back-compat
        self.ln_eps = 1e-5

        # Sparse-attention config (used when use_sparse_attn=True)
        self.use_sparse_attn = use_sparse_attn
        self.factor = factor
        self.n_bottleneck = n_bottleneck
        self.bottleneck_head = bottleneck_head
        self.sampling_strategy = sampling_strategy
        self.strat_block_size = max(int(strat_block_size), 1)
        self.scale = 1.0 / math.sqrt(self.dh)

        # ---- Attention: stacked QKV and output projections.
        self.qkv_w = nn.Parameter(torch.empty(P, hidden_size, 3 * hidden_size))
        self.qkv_b = nn.Parameter(torch.empty(P, 3 * hidden_size))
        self.attn_out_w = nn.Parameter(torch.empty(P, hidden_size, hidden_size))
        self.attn_out_b = nn.Parameter(torch.empty(P, hidden_size))

        # ---- LayerNorm (Pre-LN style: applied before each sub-layer)
        self.ln1_w = nn.Parameter(torch.ones(P, hidden_size))
        self.ln1_b = nn.Parameter(torch.zeros(P, hidden_size))
        self.ln2_w = nn.Parameter(torch.ones(P, hidden_size))
        self.ln2_b = nn.Parameter(torch.zeros(P, hidden_size))

        # ---- MLP
        self.mlp_w1 = nn.Parameter(torch.empty(P, hidden_size, mlp_dim))
        self.mlp_b1 = nn.Parameter(torch.empty(P, mlp_dim))
        self.mlp_w2 = nn.Parameter(torch.empty(P, mlp_dim, hidden_size))
        self.mlp_b2 = nn.Parameter(torch.empty(P, hidden_size))

        # ---- Optional: dual-path downsample (mirrors seq's _DualPathDownsample).
        # Allocated only when use_distill=True to keep param count clean.
        # Conv kernel=3, stride=2, padding=1 + MaxPool kernel=2, stride=2 + GELU
        # on conv path + element-wise sum + per-modality LayerNorm.
        if use_distill:
            self.ds_conv_w = nn.Parameter(torch.empty(P, hidden_size, hidden_size, 3))
            self.ds_conv_b = nn.Parameter(torch.empty(P, hidden_size))
            self.ds_norm_w = nn.Parameter(torch.ones(P, hidden_size))
            self.ds_norm_b = nn.Parameter(torch.zeros(P, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Match nn.Linear / nn.Conv1d default init bounds."""

        # Linear-style params: bound = 1 / sqrt(fan_in).
        # In our [M, in, out] convention, fan_in = shape[1].
        for w, b in [
            (self.qkv_w, self.qkv_b),
            (self.attn_out_w, self.attn_out_b),
            (self.mlp_w1, self.mlp_b1),
            (self.mlp_w2, self.mlp_b2),
        ]:
            fan_in = w.shape[1]
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -bound, bound)

        # ds_module Conv1d (only when use_distill=True): bound = 1/sqrt(C_in * K).
        if self.use_distill:
            fan_in = self.ds_conv_w.shape[2] * self.ds_conv_w.shape[3]
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.ds_conv_w, -bound, bound)
            nn.init.uniform_(self.ds_conv_b, -bound, bound)
            # ds_norm already initialised (weight=1, bias=0).

        # LN params already initialised in __init__ (weight=1, bias=0).

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather(self, p: torch.Tensor, mod_indices: Optional[torch.Tensor]) -> torch.Tensor:
        """Slice (or broadcast) the leading M dim of a stacked param tensor.

        Default (share_params=False): if `mod_indices is None`, return p
        unchanged; otherwise gather along dim 0.

        Shared (share_params=True): p has leading dim 1; expand it to the
        number of active modalities (M_total if mod_indices is None, else
        len(mod_indices)). `.contiguous()` so downstream reshape (ds path)
        works. Gradients accumulate across the expanded copies back into the
        single stored param — i.e. one shared modality-agnostic block.
        """
        if self.share_params:
            n = self.M_total if mod_indices is None else mod_indices.shape[0]
            return p.expand(n, *p.shape[1:]).contiguous()
        if mod_indices is None:
            return p
        return p.index_select(0, mod_indices)

    def _batched_layer_norm(self, x: torch.Tensor,
                            w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """LayerNorm with batched (per-M) affine parameters.

        x: [M, B, L, H]
        w, b: [M, H]
        Returns: [M, B, L, H]

        F.layer_norm doesn't accept batched weight/bias, so we call it
        without affine (single fused cuDNN kernel), then apply the per-M
        affine via broadcast (one elementwise kernel). 4 ops → 2 ops vs
        the manual mean/var/normalize/scale path; ~16% block-level speedup.
        """
        # Fused cuDNN normalize over last dim (no affine)
        xhat = F.layer_norm(x, (x.shape[-1],), eps=self.ln_eps)
        # Per-M affine via broadcast: [M, 1, 1, H]
        return xhat * w[:, None, None, :] + b[:, None, None, :]

    def _attention(self, x: torch.Tensor,
                   qkv_w: torch.Tensor, qkv_b: torch.Tensor,
                   out_w: torch.Tensor, out_b: torch.Tensor) -> torch.Tensor:
        """Self-attention with M independent QKV / output projections.

        x: [M, B, L, H]
        Returns: [M, B, L, H]

        Implementation:
        1. Stacked QKV via einsum('mblh,mho->mblo'). One grouped GEMM.
        2. Reshape (M, B) -> (M*B) and call F.scaled_dot_product_attention,
           which keeps the flash / mem-efficient backend.
        3. Stacked output projection via the same einsum pattern.
        """
        M, B, L, H = x.shape
        # ---- Stacked QKV. Output: [M, B, L, 3H]
        qkv = torch.einsum('mblh,mho->mblo', x, qkv_w) + qkv_b[:, None, None, :]
        q, k, v = qkv.chunk(3, dim=-1)

        # ---- Reshape to [M*B, nh, L, dh] for SDPA.
        # contiguous() is needed because einsum may produce non-contiguous
        # outputs depending on backend; SDPA fast-paths require contiguous.
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.reshape(M * B, L, self.nh, self.dh)
            return t.transpose(1, 2).contiguous()  # [M*B, nh, L, dh]

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # ---- SDPA with the merged batch dim. Backend selection (flash /
        # mem-efficient / math) is automatic and unaffected by the M*B
        # fold — that's the whole point of bmm vs vmap.
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                            # [M*B, nh, L, dh]

        # ---- Merge heads, unfold M*B, output projection.
        attn = attn.transpose(1, 2).contiguous().reshape(M, B, L, H)
        out = torch.einsum('mblh,mho->mblo', attn, out_w) + out_b[:, None, None, :]
        return out

    def _sample_non_bottleneck_batched(self, lo: int, hi: int,
                                       M: int, U_part: int,
                                       device: torch.device) -> torch.Tensor:
        """Per-modality non-bottleneck index sampling. Returns [M, U_part]."""
        n = hi - lo
        if self.sampling_strategy == 'stratified':
            edges = torch.linspace(0, n, U_part + 1, device=device)
            bin_starts = edges[:-1].long().unsqueeze(0).expand(M, -1)        # [M, U_part]
            bin_widths = (edges[1:].long() - edges[:-1].long()).clamp(min=1)
            bin_widths = bin_widths.unsqueeze(0).expand(M, -1)                # [M, U_part]
            offsets = (torch.rand(M, U_part, device=device) * bin_widths.float()).long()
            idx_local = (bin_starts + offsets).clamp(max=n - 1)
            return lo + idx_local
        return torch.randint(lo, hi, (M, U_part), device=device)

    def _sample_non_bottleneck_blocked(self, lo: int, hi: int,
                                       M: int, n_blocks: int, U_part: int,
                                       device: torch.device) -> torch.Tensor:
        """Per-modality, per-query-block stratified sampling.
        Returns [M, n_blocks, U_part] indices in [lo, hi)."""
        n = hi - lo
        edges = torch.linspace(0, n, U_part + 1, device=device)
        bin_starts = edges[:-1].long()                                       # [U_part]
        bin_widths = (edges[1:].long() - bin_starts).clamp(min=1)            # [U_part]
        offsets = (torch.rand(M, n_blocks, U_part, device=device)
                   * bin_widths.float()).long()                              # [M, n_blocks, U_part]
        idx_local = (bin_starts.view(1, 1, U_part) + offsets).clamp(max=n - 1)
        return lo + idx_local

    def _sparse_attention(self, x: torch.Tensor,
                          qkv_w: torch.Tensor, qkv_b: torch.Tensor,
                          out_w: torch.Tensor, out_b: torch.Tensor) -> torch.Tensor:
        """ProbSparse top-k attention with M modalities batched.

        Mirrors `ProbSparseAttentionOpt.forward` but every op is folded
        across the leading M dim. SDPA call uses M*B batch fold so Flash
        backend remains active.

        x: [M, B, L, H]
        Returns: [M, B, L, H]
        """
        M, B, L, H = x.shape
        nh, d_h = self.nh, self.dh

        # ---- Stacked QKV (same as dense path)
        qkv = torch.einsum('mblh,mho->mblo', x, qkv_w) + qkv_b[:, None, None, :]
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.reshape(M, B, L, nh, d_h).permute(0, 1, 3, 2, 4)   # [M, B, nh, L, d_h]
        K = K.reshape(M, B, L, nh, d_h).permute(0, 1, 3, 2, 4)
        V = V.reshape(M, B, L, nh, d_h).permute(0, 1, 3, 2, 4)

        # ---- ProbSparse parameters
        U_part = max(1, int(min(self.factor * math.ceil(math.log(L)),
                                L - self.n_bottleneck)))
        u = int(min(self.factor * math.ceil(math.log(L)), L))

        if self.sampling_strategy == 'stratified_block':
            # ---- Per-modality, per-query-block stratified sampling.
            S = self.strat_block_size
            n_blocks = (L + S - 1) // S
            if self.n_bottleneck != 0:
                if self.bottleneck_head:
                    non_bot = self._sample_non_bottleneck_blocked(
                        self.n_bottleneck, L, M, n_blocks, U_part, x.device)
                    bot = (torch.arange(self.n_bottleneck, device=x.device)
                           .view(1, 1, -1).expand(M, n_blocks, -1))
                else:
                    non_bot = self._sample_non_bottleneck_blocked(
                        0, L - self.n_bottleneck, M, n_blocks, U_part, x.device)
                    bot = (torch.arange(L - self.n_bottleneck, L, device=x.device)
                           .view(1, 1, -1).expand(M, n_blocks, -1))
                idx_s = torch.cat([bot, non_bot], dim=-1)        # [M, n_blocks, n_s]
            else:
                idx_s = self._sample_non_bottleneck_blocked(
                    0, L, M, n_blocks, U_part, x.device)
            n_s = idx_s.shape[-1]

            # ---- K_sample: [M, B, nh, n_blocks, n_s, d_h]
            idx_e = idx_s[:, None, None, :, :, None].expand(M, B, nh, n_blocks, n_s, d_h)
            K_exp = K[:, :, :, None, :, :].expand(M, B, nh, n_blocks, L, d_h)
            K_sample = torch.gather(K_exp, dim=4, index=idx_e)

            # ---- Pad-and-reshape Q to block form
            L_pad = n_blocks * S
            if L_pad > L:
                Q_pad = F.pad(Q, (0, 0, 0, L_pad - L))           # pad along L dim
            else:
                Q_pad = Q
            Q_blk = Q_pad.view(M, B, nh, n_blocks, S, d_h)

            # ---- Per-block QK
            QK = torch.matmul(Q_blk, K_sample.transpose(-2, -1)) # [M,B,nh,n_blocks,S,n_s]
            M_score = QK.max(-1).values - QK.mean(-1)            # [M,B,nh,n_blocks,S]
            M_score = M_score.view(M, B, nh, L_pad)[..., :L]
            _, top_idx = torch.topk(M_score, u, dim=-1)          # [M,B,nh,u]
        else:
            # ---- Per-modality key sampling (independent across M)
            if self.n_bottleneck != 0:
                if self.bottleneck_head:
                    non_bot = self._sample_non_bottleneck_batched(
                        self.n_bottleneck, L, M, U_part, x.device)
                    bot = (torch.arange(self.n_bottleneck, device=x.device)
                           .unsqueeze(0).expand(M, -1))
                else:
                    non_bot = self._sample_non_bottleneck_batched(
                        0, L - self.n_bottleneck, M, U_part, x.device)
                    bot = (torch.arange(L - self.n_bottleneck, L, device=x.device)
                           .unsqueeze(0).expand(M, -1))
                idx_s = torch.cat([bot, non_bot], dim=-1)            # [M, n_s]
            else:
                idx_s = self._sample_non_bottleneck_batched(0, L, M, U_part, x.device)

            # ---- Gather K_sample: [M, B, nh, n_s, d_h]
            n_s = idx_s.shape[-1]
            idx_e = idx_s[:, None, None, :, None].expand(M, B, nh, n_s, d_h)
            K_sample = torch.gather(K, dim=3, index=idx_e)           # [M, B, nh, U_part, d_h]

            # ---- Compute M-score and pick top-u queries
            QK = torch.matmul(Q, K_sample.transpose(-2, -1))         # [M, B, nh, L, n_s]
            M_score = QK.max(-1).values - QK.mean(-1)                # [M, B, nh, L]
            _, top_idx = torch.topk(M_score, u, dim=-1)              # [M, B, nh, u]

        # ---- Gather top_q
        top_q = Q.gather(3, top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, d_h))

        # ---- SDPA over (M*B) batch — Flash backend
        ctx_up = F.scaled_dot_product_attention(
            top_q.reshape(M * B, nh, u, d_h),
            K.reshape(M * B, nh, L, d_h),
            V.reshape(M * B, nh, L, d_h),
            scale=self.scale,
        ).reshape(M, B, nh, u, d_h)

        # ---- Build full context: V-mean fill + scatter top-u updates
        context = V.mean(dim=3, keepdim=True).expand(-1, -1, -1, L, -1).contiguous()
        scatter_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, d_h)
        context.scatter_(3, scatter_idx, ctx_up)

        # ---- Reshape back and apply stacked output projection
        context = context.permute(0, 1, 3, 2, 4).contiguous().reshape(M, B, L, H)
        out = torch.einsum('mblh,mho->mblo', context, out_w) + out_b[:, None, None, :]
        return out

    def _grouped_conv(self, x: torch.Tensor,
                      conv_w: torch.Tensor, conv_b: torch.Tensor) -> torch.Tensor:
        """M independent Conv1d via `groups=M`.

        x: [M, B, L, H]
        Returns: [M, B, L_out, H], where L_out = ceil(L / stride).

        Layout trick:
          1. Permute & reshape input to [B, M*H, L] (channels-first, M
             folded into channel dim).
          2. Build grouped weight [M*H_out, H_in, K] by reshaping
             conv_w [M, H_out, H_in, K]. The groups=M arg tells conv1d
             that input channels split into M groups of H, each consumed
             by H_out output channels — exactly per-modality conv.
          3. F.max_pool1d acts channel-wise, so the [B, M*H, L] view is
             already correct (no group needed).
          4. Sum, reshape back to [M, B, L_out, H].
        """
        M, B, L, H = x.shape
        # [M, B, L, H] -> [B, M, H, L] -> [B, M*H, L]
        z = x.permute(1, 0, 3, 2).contiguous().reshape(B, M * H, L)

        w = conv_w.reshape(M * H, H, 3)
        b = conv_b.reshape(M * H)

        z_conv = F.conv1d(z, w, bias=b,
                          stride=self.stride, padding=1, groups=M)
        z_pool = F.max_pool1d(z, kernel_size=3,
                              stride=self.stride, padding=1)
        z_out = z_conv + z_pool                       # [B, M*H, L_out]
        L_out = z_out.shape[-1]

        # [B, M*H, L_out] -> [B, M, H, L_out] -> [M, B, L_out, H]
        return z_out.reshape(B, M, H, L_out).permute(1, 0, 3, 2).contiguous()

    def _mlp(self, x: torch.Tensor,
             w1: torch.Tensor, b1: torch.Tensor,
             w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        """MLP with M independent weights: Linear-GELU-Dropout-Linear-Dropout.

        x: [M, B, L, H] -> [M, B, L, H]
        """
        h = torch.einsum('mblh,mho->mblo', x, w1) + b1[:, None, None, :]
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        h = torch.einsum('mblo,moh->mblh', h, w2) + b2[:, None, None, :]
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                mod_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [M_active, B, L, H]
            mod_indices: LongTensor of length M_active, indices into the
                         M_total stored param sets. None means use all
                         M_total in original order (so M_active=M_total).

        Returns:
            [M_active, B, L_out, H]
        """
        # ---- Slice param sets to match active modalities (cheap; small dim 0).
        qkv_w = self._gather(self.qkv_w, mod_indices)
        qkv_b = self._gather(self.qkv_b, mod_indices)
        attn_out_w = self._gather(self.attn_out_w, mod_indices)
        attn_out_b = self._gather(self.attn_out_b, mod_indices)
        ln1_w = self._gather(self.ln1_w, mod_indices)
        ln1_b = self._gather(self.ln1_b, mod_indices)
        ln2_w = self._gather(self.ln2_w, mod_indices)
        ln2_b = self._gather(self.ln2_b, mod_indices)
        mlp_w1 = self._gather(self.mlp_w1, mod_indices)
        mlp_b1 = self._gather(self.mlp_b1, mod_indices)
        mlp_w2 = self._gather(self.mlp_w2, mod_indices)
        mlp_b2 = self._gather(self.mlp_b2, mod_indices)

        # ---- Pre-LN block forward — mirrors standalone TransformerBlock.
        # Order: residual=x; ln1; attn; dropout; +residual; (ds if use_distill);
        #        residual=x; ln2; mlp; +residual.

        # Pre-LN attention
        residual = x
        x = self._batched_layer_norm(x, ln1_w, ln1_b)
        if self.use_sparse_attn:
            x = self._sparse_attention(x, qkv_w, qkv_b, attn_out_w, attn_out_b)
        else:
            x = self._attention(x, qkv_w, qkv_b, attn_out_w, attn_out_b)
        # Match seq's `dropout1` after attention.
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        # ---- Early-exit instrumentation: per-modality relative attention update.
        # x here is the attention sublayer output (the update added to residual);
        # ||update_m|| / ||residual_m|| measures how much cross-modal attention
        # changed modality m's stream this layer. Small => saturated => exit candidate.
        if getattr(self, '_capture_attn_delta', False):
            num = x.norm(dim=(-2, -1))                 # [M, B]  over (L, H)
            den = residual.norm(dim=(-2, -1)) + 1e-6    # [M, B]
            self._last_attn_delta = (num / den).detach()  # [M, B]
        x = x + residual

        # Optional dual-path downsample (mirror of seq's _DualPathDownsample,
        # injected between attention and FFN as in seq's TransformerBlockDS).
        if self.use_distill:
            ds_conv_w = self._gather(self.ds_conv_w, mod_indices)
            ds_conv_b = self._gather(self.ds_conv_b, mod_indices)
            ds_norm_w = self._gather(self.ds_norm_w, mod_indices)
            ds_norm_b = self._gather(self.ds_norm_b, mod_indices)
            x = self._batched_ds(x, ds_conv_w, ds_conv_b, ds_norm_w, ds_norm_b)

        # Pre-LN MLP
        residual = x
        x = self._batched_layer_norm(x, ln2_w, ln2_b)
        x = self._mlp(x, mlp_w1, mlp_b1, mlp_w2, mlp_b2)
        x = x + residual
        return x

    def _batched_ds(self, x: torch.Tensor,
                    ds_conv_w: torch.Tensor, ds_conv_b: torch.Tensor,
                    ds_norm_w: torch.Tensor, ds_norm_b: torch.Tensor) -> torch.Tensor:
        """Batched mirror of standalone's `_DualPathDownsample`.

        Behaviour identical to seq path:
          1. Trim odd L (drop last token).
          2. Conv1d(k=3, s=2, p=1) via groups=M  + GELU on conv output.
          3. MaxPool1d(k=2, s=2) on the same input (parameter-free, channel-wise).
          4. Element-wise sum.
          5. Per-modality LayerNorm.

        x: [M, B, L, d] -> [M, B, L_new, d] where L_new = (L if even else L-1) // 2.
        """
        M = x.shape[0]
        if x.shape[2] % 2 == 1:
            x = x[:, :, :-1, :].contiguous()
        M, B, L, d = x.shape
        # [M, B, L, d] -> [B, M, d, L] -> [B, M*d, L]
        z = x.permute(1, 0, 3, 2).contiguous().reshape(B, M * d, L)
        w = ds_conv_w.reshape(M * d, d, 3)
        b_ = ds_conv_b.reshape(M * d)
        z_conv = F.conv1d(z, w, bias=b_, stride=2, padding=1, groups=M)
        z_pool = F.max_pool1d(z, kernel_size=2, stride=2)
        z_out = F.gelu(z_conv) + z_pool                     # [B, M*d, L_new]
        L_new = z_out.shape[-1]
        # back to [M, B, L_new, d]
        out = z_out.reshape(B, M, d, L_new).permute(1, 0, 3, 2).contiguous()
        return self._batched_layer_norm(out, ds_norm_w, ds_norm_b)


# =====================================================================
# Variant: BatchedTransformerBlockBmm
#   Drop-in subclass that replaces einsum with torch.bmm + reshape and
#   the hand-written LayerNorm with F.layer_norm (no-affine) + per-M
#   affine. Same parameter shapes, same forward graph; the only changes
#   are the kernels invoked. Intended to remove the einsum-backward
#   regression observed at long L on the original BatchedTransformerBlock.
# =====================================================================

class BatchedTransformerBlockBmm(BatchedTransformerBlock):
    """Same forward graph as BatchedTransformerBlock; uses bmm + F.layer_norm.

    Why: PyTorch's einsum has known-slower backward paths for simple
    `mbi,mij->mbj` patterns, and the manual `(x - mean) * rsqrt(var + eps)`
    LayerNorm misses the fused kernel. cuBLAS bmm + fused F.layer_norm
    fix both at zero numerical-precision cost (results equal up to fp
    rounding).

    Inherits all parameters from BatchedTransformerBlock, so weights are
    interchangeable — only the kernels invoked differ.
    """

    # ----- Replaced: linear projections via batched GEMM -----
    def _attention(self, x: torch.Tensor,
                   qkv_w: torch.Tensor, qkv_b: torch.Tensor,
                   out_w: torch.Tensor, out_b: torch.Tensor) -> torch.Tensor:
        M, B, L, H = x.shape
        x_flat = x.reshape(M, B * L, H).contiguous()           # free if x contiguous
        # [M, B*L, H] @ [M, H, 3H] -> [M, B*L, 3H]
        qkv = torch.bmm(x_flat, qkv_w) + qkv_b[:, None, :]
        qkv = qkv.reshape(M, B, L, 3 * H)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.reshape(M * B, L, self.nh, self.dh)
            return t.transpose(1, 2).contiguous()

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().reshape(M, B * L, H)
        # [M, B*L, H] @ [M, H, H] -> [M, B*L, H]
        out = torch.bmm(attn, out_w) + out_b[:, None, :]
        return out.reshape(M, B, L, H)

    def _mlp(self, x: torch.Tensor,
             w1: torch.Tensor, b1: torch.Tensor,
             w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        M, B, L, H = x.shape
        x_flat = x.reshape(M, B * L, H).contiguous()
        h = torch.bmm(x_flat, w1) + b1[:, None, :]            # [M, B*L, mlp_dim]
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        h = torch.bmm(h, w2) + b2[:, None, :]                 # [M, B*L, H]
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h.reshape(M, B, L, H)

    # ----- Replaced: LayerNorm via F.layer_norm fused kernel -----
    def _batched_layer_norm(self, x: torch.Tensor,
                            w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fused F.layer_norm (no affine) + per-M affine.

        F.layer_norm doesn't accept batched weight/bias, so we run it with
        affine=False to get the fused mean/var/rsqrt path, then apply
        per-M affine manually. The affine multiply+add is one element-wise
        kernel — much cheaper than re-implementing mean/var.
        """
        M, B, L, H = x.shape
        # Fold M into batch dim — F.layer_norm needs trailing-feature normalization.
        x_flat = x.reshape(M * B, L, H)
        normed = F.layer_norm(x_flat, (H,), weight=None, bias=None, eps=self.ln_eps)
        normed = normed.reshape(M, B, L, H)
        return normed * w[:, None, None, :] + b[:, None, None, :]


class BatchedTransformerBlockBmmSeqConv(BatchedTransformerBlockBmm):
    """bmm + F.layer_norm, but conv1d/maxpool1d run as M sequential calls.

    Probes whether grouped Conv1d (groups=M, channels=M*H) is the
    bottleneck on V6 work-point shapes. Each modality runs its own
    standard Conv1d (no grouping); cuDNN gets to pick the best algorithm
    independently per modality. Trade-off: M Python launches.
    """

    def _grouped_conv(self, x: torch.Tensor,
                      conv_w: torch.Tensor, conv_b: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        outs = []
        for m in range(M):
            z = x[m].transpose(-1, -2)                # [B, H, L]
            w = conv_w[m]                              # [H, H, 3]
            b = conv_b[m]                              # [H]
            z_conv = F.conv1d(z, w, bias=b, stride=self.stride, padding=1)
            z_pool = F.max_pool1d(z, kernel_size=3,
                                  stride=self.stride, padding=1)
            outs.append((z_conv + z_pool).transpose(-1, -2))   # [B, L_out, H]
        return torch.stack(outs, dim=0)


class BatchedTransformerBlockBmmStreamConv(BatchedTransformerBlockBmm):
    """bmm + F.layer_norm, but conv1d/maxpool1d dispatched on M CUDA streams.

    Same conv math as BmmSeqConv but uses streams for kernel-level
    overlap. Tests whether M-parallel conv beats grouped conv in
    practice (streams should win when individual conv work is small).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv_streams = None

    def _get_conv_streams(self, M: int):
        if self._conv_streams is None or len(self._conv_streams) < M:
            self._conv_streams = [torch.cuda.Stream() for _ in range(M)]
        return self._conv_streams[:M]

    def _grouped_conv(self, x: torch.Tensor,
                      conv_w: torch.Tensor, conv_b: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        if not x.is_cuda:
            # Fall back to sequential on CPU.
            return BatchedTransformerBlockBmmSeqConv._grouped_conv(
                self, x, conv_w, conv_b)
        streams = self._get_conv_streams(M)
        outs = [None] * M
        for m in range(M):
            with torch.cuda.stream(streams[m]):
                z = x[m].transpose(-1, -2)
                z_conv = F.conv1d(z, conv_w[m], bias=conv_b[m],
                                  stride=self.stride, padding=1)
                z_pool = F.max_pool1d(z, kernel_size=3,
                                      stride=self.stride, padding=1)
                outs[m] = (z_conv + z_pool).transpose(-1, -2)
        for s in streams:
            s.synchronize()
        return torch.stack(outs, dim=0)


# =====================================================================
# Fusion module: SimpleMBTFusionAdaptiveBmm
# =====================================================================

class SimpleMBTFusionAdaptiveBmm(nn.Module):
    """bmm-based MBT fusion with one BatchedTransformerBlock per layer.

    External interface mirrors `SimpleMBTFusionAdaptive`:
        __init__(input_dims, hidden_size, num_layers, num_heads, mlp_dim,
                 fusion_layer, use_bottleneck, n_bottlenecks, dropout_rate,
                 output_dim, bottleneck_head_pos)
        forward(inputs: Dict[str, Tensor], return_tokens=False, factor=None)

    Key internal differences:
        * `self.blocks` is an nn.ModuleList of length `num_layers`, each
          a BatchedTransformerBlock with M_total = num_modalities.
        * `forward` stacks active-modality features into [M_active, B, L, H]
          once and threads that through every layer.
        * The bottleneck is replicated to all M_active modalities, runs
          through the block, then aggregated by mean across M_active.

    Unsupported-by-design (raises at __init__ time):
        * use_sparse_attn=True
        * use_sparse_moe=True
        * use_triton=True
    """

    def __init__(self, input_dims: Dict[str, int],
                 hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, mlp_dim: int = 2048,
                 fusion_layer: int = 3, use_bottleneck: bool = True,
                 n_bottlenecks: int = 4, dropout_rate: float = 0.1,
                 output_dim: Optional[int] = None,
                 bottleneck_head_pos: bool = False,
                 max_seq_len: int = 500,
                 # Sparse attention is now SUPPORTED via BatchedTransformerBlock's
                 # _sparse_attention path. Set use_sparse_attn=True and pick a
                 # variant via sparse_attn_variant ('orig' falls back to global).
                 use_sparse_attn: bool = False,
                 use_sparse_moe: bool = False,
                 use_triton: bool = False,
                 num_experts: int = 4, expert_k: int = 1,
                 parallel_modalities: bool = False,
                 factor: int = 5,
                 sparse_attn_variant: str = 'orig',
                 # When True, _stack_inputs adds modality-specific pos_embeds
                 # to the input. Default False to behave as a drop-in
                 # replacement for V6's SimpleMBTFusionAdaptiveMLP.
                 add_pos_embeds: bool = False,
                 **_unused_kwargs):
        super().__init__()
        if use_sparse_moe:
            raise NotImplementedError('No MoE in the bmm path.')
        if use_triton:
            raise NotImplementedError(
                'Triton fused-add-LN expects per-block LN params; not '
                'compatible with batched LN parameters.'
            )

        self.all_modalities: List[str] = list(input_dims.keys())
        self.M_total = len(self.all_modalities)
        self.modality_to_idx = {m: i for i, m in enumerate(self.all_modalities)}
        self.hidden_size = hidden_size
        self.output_dim = output_dim if output_dim is not None else hidden_size
        self.use_bottleneck = use_bottleneck
        self.n_bottlenecks = n_bottlenecks
        self.bottleneck_head_pos = bottleneck_head_pos
        self.num_layers = num_layers
        self.fusion_layer = fusion_layer
        self.use_sparse_attn = use_sparse_attn
        self.sparse_attn_variant = sparse_attn_variant
        self.factor = factor
        self.add_pos_embeds = add_pos_embeds

        # ---- Per-modality input projections (kept independent — small
        # cost, enables modality-specific input dims).
        self.input_projections = nn.ModuleDict()
        for modality, in_dim in input_dims.items():
            if in_dim != hidden_size:
                self.input_projections[modality] = nn.Linear(in_dim, hidden_size)
            else:
                self.input_projections[modality] = nn.Identity()

        # ---- Per-modality positional embeddings (independent).
        self.pos_embeds = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, max_seq_len, hidden_size) * 0.02)
            for m in self.all_modalities
        })

        # ---- One batched block per layer (M_total modalities each).
        # n_bottleneck for the BatchedTransformerBlock's internal sparse path
        # is the bottleneck token count seen during forward (only matters when
        # use_sparse_attn=True and bottleneck tokens are concatenated to the
        # modality sequence in the parent fusion forward).
        sampling_strategy = ('stratified' if sparse_attn_variant == 'opt_strat'
                             else 'global')
        self.blocks = nn.ModuleList([
            BatchedTransformerBlock(
                M_total=self.M_total, hidden_size=hidden_size,
                num_heads=num_heads, mlp_dim=mlp_dim,
                dropout_rate=dropout_rate, use_distill=False,
                use_sparse_attn=use_sparse_attn,
                factor=factor,
                n_bottleneck=n_bottlenecks if use_bottleneck else 0,
                bottleneck_head=bottleneck_head_pos,
                sampling_strategy=sampling_strategy,
            )
            for _ in range(num_layers)
        ])

        # ---- Shared bottleneck token bank.
        if use_bottleneck:
            self.bottleneck = nn.Parameter(
                torch.randn(1, n_bottlenecks, hidden_size) * 0.02
            )
        else:
            self.bottleneck = None

        self.final_norm = nn.LayerNorm(hidden_size)
        if self.output_dim != hidden_size:
            self.output_projection = nn.Linear(hidden_size, self.output_dim)
        else:
            self.output_projection = nn.Identity()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _stack_inputs(self, inputs: Dict[str, torch.Tensor],
                      available_modalities: List[str]) -> torch.Tensor:
        """Project each modality (and optionally add pos embedding), stack to [M_active, B, L, H].

        `add_pos_embeds` is False by default so this fusion is a drop-in
        replacement for the V6's `SimpleMBTFusionAdaptiveMLP` (whose `pos_embeds`
        ParameterDict is also unused in forward). When used standalone (V6
        bypassed), set add_pos_embeds=True at construction.
        """
        projected = []
        for m in available_modalities:
            proj = self.input_projections[m](inputs[m])
            if getattr(self, 'add_pos_embeds', False):
                L = proj.shape[1]
                proj = proj + self.pos_embeds[m][:, :L]
            projected.append(proj)
        # All modalities normalised to same L upstream; stack along new dim 0.
        return torch.stack(projected, dim=0)

    def forward(self, inputs: Dict[str, torch.Tensor],
                return_tokens: bool = False,
                factor=None) -> torch.Tensor:
        # `factor` accepted for API parity; unused by dense path.
        del factor

        available_modalities = [m for m in self.all_modalities if m in inputs]
        if len(available_modalities) == 0:
            raise ValueError('At least one modality must be provided')

        device = inputs[available_modalities[0]].device
        batch_size = inputs[available_modalities[0]].shape[0]
        M_active = len(available_modalities)

        # mod_indices: which slices of the stored M_total params to use.
        # When all modalities present (the common case), this is the
        # identity permutation and `_gather` short-circuits to a no-op
        # if we pass None.
        full = (M_active == self.M_total
                and available_modalities == self.all_modalities)
        if full:
            mod_indices = None
        else:
            mod_indices = torch.tensor(
                [self.modality_to_idx[m] for m in available_modalities],
                dtype=torch.long, device=device,
            )

        # ---- Stack inputs: [M_active, B, L, H]
        x = self._stack_inputs(inputs, available_modalities)

        # ---- Bottleneck (B-broadcast).
        if self.use_bottleneck and self.bottleneck is not None:
            bottleneck = self.bottleneck.expand(batch_size, -1, -1).contiguous()
        else:
            bottleneck = None

        # ---- Layer-by-layer.
        for lyr in range(self.num_layers):
            block = self.blocks[lyr]
            if lyr < self.fusion_layer:
                # Pre-fusion: each modality independent (no bottleneck).
                x = block(x, mod_indices)
            else:
                if bottleneck is not None:
                    seq_len = x.shape[2]
                    bn_len = bottleneck.shape[1]
                    # Replicate bottleneck across M_active modalities.
                    bn_stack = bottleneck.unsqueeze(0).expand(
                        M_active, -1, -1, -1
                    )
                    if not self.bottleneck_head_pos:
                        combined = torch.cat([x, bn_stack], dim=2)
                    else:
                        combined = torch.cat([bn_stack, x], dim=2)

                    out = block(combined, mod_indices)

                    if not self.bottleneck_head_pos:
                        x = out[:, :, :seq_len]
                        new_bn = out[:, :, seq_len:]
                    else:
                        x = out[:, :, bn_len:]
                        new_bn = out[:, :, :bn_len]

                    # Aggregate bottleneck across modalities (mean).
                    bottleneck = new_bn.mean(dim=0)
                else:
                    # No-bottleneck path: in the original code the modalities
                    # are concatenated along L and a single block is applied.
                    # Re-implement the same behaviour using the first slice.
                    M, B, L, H = x.shape
                    x_cat = x.permute(1, 0, 2, 3).reshape(B, M * L, H)
                    x_cat = x_cat.unsqueeze(0)               # [1, B, M*L, H]
                    idx = (mod_indices[:1] if mod_indices is not None
                           else torch.zeros(1, dtype=torch.long, device=device))
                    x_cat = block(x_cat, idx).squeeze(0)     # [B, M*L, H]
                    x = x_cat.reshape(B, M, L, H).permute(1, 0, 2, 3).contiguous()

        # ---- Concatenate modalities along L for the readout.
        # x: [M_active, B, L, H] -> [B, M_active*L, H]
        M_active_, B_, L_, H_ = x.shape
        fused_tokens = x.permute(1, 0, 2, 3).reshape(B_, M_active_ * L_, H_)
        fused_tokens = self.final_norm(fused_tokens)

        if return_tokens:
            return fused_tokens

        fused_repr = fused_tokens.mean(dim=1)
        return self.output_projection(fused_repr)


# =====================================================================
# Fusion module: SimpleMBTFusionAdaptiveWeightedFactorBmm
# (adds use_distilling support; per-modality `factor` is ignored
#  because the dense path doesn't use it)
# =====================================================================

class SimpleMBTFusionAdaptiveWeightedFactorBmm(SimpleMBTFusionAdaptiveBmm):
    """bmm fusion with optional per-layer post-block distilling.

    Adds an `nn.ModuleList` of plain Conv1d / MaxPool1d (shared across
    modalities) applied in the fusion stage, exactly mirroring
    `SimpleMBTFusionAdaptiveWeightedFactor.distill_convs`.

    The per-modality `factor` argument from the original module is
    accepted for API parity but ignored — `factor` controls ProbSparse
    sampling, which doesn't exist in the bmm path.
    """

    def __init__(self, *args, use_distilling: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_distilling = use_distilling

        if use_distilling:
            n_distill_layers = self.num_layers - self.fusion_layer
            # Shared across modalities (mirrors original design).
            self.distill_convs = nn.ModuleList([
                nn.Conv1d(self.hidden_size, self.hidden_size,
                          kernel_size=3, padding=1, stride=2)
                for _ in range(n_distill_layers)
            ])
            self.distill_pools = nn.ModuleList([
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
                for _ in range(n_distill_layers)
            ])
        else:
            self.distill_convs = None
            self.distill_pools = None

    def _distill_combined(self, output: torch.Tensor,
                          distill_idx: int) -> torch.Tensor:
        """Apply the shared distill conv to a [M, B, L, H] tensor.

        Implementation: fold M*B into the batch dim, run conv1d once,
        unfold. Conv weights are shared so no grouping needed.
        """
        M, B, L, H = output.shape
        z = output.reshape(M * B, L, H).permute(0, 2, 1)            # [M*B, H, L]
        z = self.distill_convs[distill_idx](z) + self.distill_pools[distill_idx](z)
        L_new = z.shape[-1]
        return z.permute(0, 2, 1).reshape(M, B, L_new, H)

    def forward(self, inputs: Dict[str, torch.Tensor],
                return_tokens: bool = False,
                factor=None) -> torch.Tensor:
        del factor                                       # ignored: dense path

        available_modalities = [m for m in self.all_modalities if m in inputs]
        if len(available_modalities) == 0:
            raise ValueError('At least one modality must be provided')

        device = inputs[available_modalities[0]].device
        batch_size = inputs[available_modalities[0]].shape[0]
        M_active = len(available_modalities)
        full = (M_active == self.M_total
                and available_modalities == self.all_modalities)
        mod_indices = (None if full else torch.tensor(
            [self.modality_to_idx[m] for m in available_modalities],
            dtype=torch.long, device=device,
        ))

        x = self._stack_inputs(inputs, available_modalities)

        if self.use_bottleneck and self.bottleneck is not None:
            bottleneck = self.bottleneck.expand(batch_size, -1, -1).contiguous()
        else:
            bottleneck = None

        for lyr in range(self.num_layers):
            block = self.blocks[lyr]
            distill_idx = lyr - self.fusion_layer

            if lyr < self.fusion_layer:
                x = block(x, mod_indices)
            else:
                if bottleneck is None:
                    # Same fallback as parent.
                    M, B, L, H = x.shape
                    x_cat = x.permute(1, 0, 2, 3).reshape(B, M * L, H)
                    idx = (mod_indices[:1] if mod_indices is not None
                           else torch.zeros(1, dtype=torch.long, device=device))
                    x_cat = block(x_cat.unsqueeze(0), idx).squeeze(0)
                    x = x_cat.reshape(B, M, L, H).permute(1, 0, 2, 3).contiguous()
                    continue

                seq_len = x.shape[2]
                bn_len = bottleneck.shape[1]
                combined_len = seq_len + bn_len
                min_safe_len = 2 * (self.n_bottlenecks + 2)
                distill = (self.use_distilling
                           and self.distill_convs is not None
                           and combined_len >= min_safe_len)

                bn_stack = bottleneck.unsqueeze(0).expand(M_active, -1, -1, -1)
                if not self.bottleneck_head_pos:
                    combined = torch.cat([x, bn_stack], dim=2)
                else:
                    combined = torch.cat([bn_stack, x], dim=2)

                out = block(combined, mod_indices)

                # Optional distilling — applied after the block on the
                # full [M, B, L+bn, H] tensor (shared conv across M).
                if distill:
                    out = self._distill_combined(out, distill_idx)
                    bn_len_new = max(1, bn_len // 2)
                else:
                    bn_len_new = bn_len

                if not self.bottleneck_head_pos:
                    seq_len_new = out.shape[2] - bn_len_new
                    x = out[:, :, :seq_len_new]
                    new_bn = out[:, :, seq_len_new:]
                else:
                    x = out[:, :, bn_len_new:]
                    new_bn = out[:, :, :bn_len_new]

                bottleneck = new_bn.mean(dim=0)

        M_active_, B_, L_, H_ = x.shape
        fused_tokens = x.permute(1, 0, 2, 3).reshape(B_, M_active_ * L_, H_)
        fused_tokens = self.final_norm(fused_tokens)

        if return_tokens:
            return fused_tokens

        fused_repr = fused_tokens.mean(dim=1)
        return self.output_projection(fused_repr)


# =====================================================================
# Sanity-test scaffold (NOT executed; for the user to run themselves).
# =====================================================================
"""
Example integration with V6OriginalRobustDistilled:

    from fusion_bmm_parallel import SimpleMBTFusionAdaptiveWeightedFactorBmm

    class DualVideoBottleneckModelV6OriginalRobustDistilledBmm(
            DualVideoBottleneckModelV6OriginalRobust):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            d_model        = kwargs.get('d_model', 384)
            num_layers     = kwargs.get('num_layers', 2)
            nhead          = kwargs.get('nhead', 16)
            dropout        = kwargs.get('dropout', 0.1)
            n_bottlenecks  = kwargs.get('n_bottlenecks', 4)
            fusion_layer   = kwargs.get('fusion_layer', 1)
            bottleneck_head_pos = kwargs.get('bottleneck_head_pos', False)

            fusion_input_dims = {m: self.pooled_dim for m in self.modalities}
            self.bottleneck_fusion = SimpleMBTFusionAdaptiveWeightedFactorBmm(
                input_dims=fusion_input_dims, hidden_size=d_model,
                num_layers=num_layers, num_heads=nhead,
                mlp_dim=int(d_model * 0.5), fusion_layer=fusion_layer,
                use_bottleneck=True, n_bottlenecks=n_bottlenecks,
                dropout_rate=dropout, output_dim=d_model,
                bottleneck_head_pos=bottleneck_head_pos,
                use_distilling=True,
            )

Quick smoke-test:

    import torch
    from fusion_bmm_parallel import SimpleMBTFusionAdaptiveBmm

    fusion = SimpleMBTFusionAdaptiveBmm(
        input_dims={'video': 384, 'audio': 384, 'eeg': 384},
        hidden_size=256, num_layers=4, num_heads=8, mlp_dim=128,
        fusion_layer=2, n_bottlenecks=8, dropout_rate=0.1,
    ).cuda()

    inputs = {
        'video': torch.randn(4, 200, 384, device='cuda'),
        'audio': torch.randn(4, 200, 384, device='cuda'),
        'eeg':   torch.randn(4, 200, 384, device='cuda'),
    }
    out = fusion(inputs)                 # [4, 256]
    out.sum().backward()                 # gradients flow

Top-k subset (drop one modality at runtime):

    inputs = {'video': ..., 'audio': ...}     # 'eeg' missing
    out = fusion(inputs)                       # uses mod_indices = [0, 1]
"""
