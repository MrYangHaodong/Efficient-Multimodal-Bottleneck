"""
V6 Downsample (opt + batched) — single-variant standalone.

Trimmed-down variant of dual_video_bottleneck_model_topk_v6_standalone.py
that only supports the configuration:

    sparse_attn_variant = 'opt'    (ProbSparseAttentionOpt, shared key sampling)
    use_batched_fusion  = True     (BatchedModalityEncoder per-modal encoder +
                                    SimpleMBTFusionAdaptiveMLPDownsampleBmm fusion)

All other sparse-attn variants (orig, BSA, CABSA, CAFlex) and the non-batched
fusion path have been deleted. The constructor of
``DualVideoBottleneckModelV6Downsample`` still accepts ``use_batched_fusion``
for back-compat but ignores its value (always batched).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt, ceil, log
from typing import Dict, Optional, Tuple, Union, List

try:
    from MAESTRO.models.triton_ops import (
        TritonProbSparseAttention,
        TRITON_AVAILABLE,
        triton_fused_add_layernorm,
    )
except Exception:
    TRITON_AVAILABLE = False
    TritonProbSparseAttention = None
    triton_fused_add_layernorm = None

# Compile flex_attention once at import time so every call gets the fused kernel.
# Falls back to the uncompiled version if FlexAttention is unavailable (PyTorch < 2.5).
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attn_raw, create_block_mask
    _flex_attn = torch.compile(_flex_attn_raw, dynamic=False)
    _FLEX_ATTN_OK = True
except Exception:
    _flex_attn = None
    create_block_mask = None
    _FLEX_ATTN_OK = False


# ===================================================================
# From bottleneck_fusion_adaptive_optimized.py
# ===================================================================

class ProbSparseAttentionOpt(nn.Module):
    """Drop-in optimised replacement for ProbSparseAttention.

    Three optimisations vs the original (per-query sampling + manual softmax + clone):
      1. Shared key sampling — `K_sample` shape goes from
         [B, H, L, n_bottleneck+U_part, d_h] to [B, H, n_bottleneck+U_part, d_h].
      2. SDPA (Flash backend) for the dense attention over the top-`u` queries —
         scores/softmax stay in SRAM, no [B, H, u, L] tensor materialised.
      3. `scatter_` in place of `clone() + fancy-index assignment` for placing
         updated rows back into the [B, H, L, d_h] context tensor.

    Bottleneck-aware sampling is preserved: `n_bottleneck` indices are always
    included in the key sample (head or tail of the sequence depending on
    `bottleneck_head`); only `U_part` non-bottleneck keys are drawn per
    forward call (shared across queries).

    sampling_strategy controls how those `U_part` non-bottleneck indices are drawn:
      'global'             — uniform random over the non-bottleneck range, single
                             shared set across all queries.
      'stratified'         — partition the non-bottleneck range into `U_part` equal
                             bins, draw one index per bin. Single shared set.
      'stratified_block'   — like 'stratified' but draw `n_blocks` independent sets
                             (`n_blocks = ceil(L / strat_block_size)`); queries are
                             partitioned into contiguous blocks of `strat_block_size`
                             and each block uses its own stratified key set.
                             strat_block_size=1   → per-query stratified.
                             strat_block_size>=L  → equivalent to 'stratified'.
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
        """Draw U_part indices in [lo, hi) per `self.sampling_strategy`.

        Returns:
            shared modes ('global', 'stratified'): [U_part]
            'stratified_block':                    [n_blocks, U_part] (caller handles)
        """
        n = hi - lo
        if self.sampling_strategy == 'stratified':
            edges = torch.linspace(0, n, U_part + 1, device=device)
            bin_starts = edges[:-1].long()                        # [U_part]
            bin_widths = (edges[1:].long() - bin_starts).clamp(min=1)
            offsets = (torch.rand(U_part, device=device) * bin_widths.float()).long()
            idx_local = (bin_starts + offsets).clamp(max=n - 1)
            return lo + idx_local
        return torch.randint(lo, hi, (U_part,), device=device)

    def _sample_non_bottleneck_blocked(self, lo, hi, n_blocks, U_part, device):
        """Per-block stratified sampling. Returns [n_blocks, U_part] indices in [lo, hi)."""
        n = hi - lo
        edges = torch.linspace(0, n, U_part + 1, device=device)
        bin_starts = edges[:-1].long()                                            # [U_part]
        bin_widths = (edges[1:].long() - bin_starts).clamp(min=1)                 # [U_part]
        offsets = (torch.rand(n_blocks, U_part, device=device)
                   * bin_widths.float()).long()                                    # [n_blocks, U_part]
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
            # ─── Per-query-block stratified key sampling ───
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
                idx_s = torch.cat([bot, non_bot], dim=1)            # [n_blocks, n_s]
            else:
                idx_s = self._sample_non_bottleneck_blocked(0, L, n_blocks, U_part, x.device)
            n_s = idx_s.shape[1]

            # K_sample: [B,H,n_blocks,n_s,d_h]
            K_sample = K.index_select(2, idx_s.reshape(-1)).view(B, H, n_blocks, n_s, d_h)

            # Pad-and-reshape Q to block form, then per-block matmul
            L_pad = n_blocks * S
            Q_pad = F.pad(Q, (0, 0, 0, L_pad - L)) if L_pad > L else Q
            Q_blk = Q_pad.view(B, H, n_blocks, S, d_h)
            QK_samp = torch.matmul(Q_blk, K_sample.transpose(-2, -1))   # [B,H,n_blocks,S,n_s]

            M = QK_samp.max(-1).values - QK_samp.mean(-1)               # [B,H,n_blocks,S]
            M = M.view(B, H, L_pad)[:, :, :L]
            _, top_idx = torch.topk(M, u, dim=-1)                       # [B,H,u]
        else:
            # ─── Opt 1: shared key sampling (global or stratified) ───
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

            K_sample = K[:, :, idx_s, :]                                # [B,H,n_s,d_h]
            QK_samp = torch.matmul(Q, K_sample.transpose(-2, -1))       # [B,H,L,n_s]

            M = QK_samp.max(-1).values - QK_samp.mean(-1)               # [B,H,L]
            _, top_idx = torch.topk(M, u, dim=-1)                       # [B,H,u]

        top_q = Q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, d_h))

        # ─── Opt 2: SDPA for the top-u dense attention ───
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

        # ─── Opt 3: scatter_ (no clone + no batch/head index tensors) ───
        context = V.mean(dim=2).unsqueeze(2).expand(-1, -1, L, -1).contiguous()
        context.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, d_h), ctx_up)

        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(context)


class SparseMoEFeedForward(nn.Module):
    """Sparse Mixture of Experts Feed-Forward Layer."""

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
    """Transformer encoder block with self-attention and MLP (dense or sparse MoE)."""

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
        self.use_triton = use_triton and TRITON_AVAILABLE and TritonProbSparseAttention is not None
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
            self.attn = ProbSparseAttentionOpt(
                hidden_size, num_heads, dropout=dropout_rate,
                factor=factor, n_bottleneck=n_bottleneck,
                bottleneck_head=bottleneck_head,
                sampling_strategy=_strategy_map[sparse_attn_variant],
                strat_block_size=strat_block_size,
            )
        else:
            raise ValueError(
                f"sparse_attn_variant={sparse_attn_variant!r} not supported in "
                f"v6_downsample_opt_batched (only 'opt'/'opt_strat'/'opt_strat_blk' "
                f"or use_triton=True with TRITON available)."
            )
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.self_attn = self.attn

        if use_sparse_moe:
            self.mlp = SparseMoEFeedForward(
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

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
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

        if (self.use_triton and triton_fused_add_layernorm is not None
                and x.is_cuda and x.is_contiguous()):
            residual, x = triton_fused_add_layernorm(
                x, residual, self.ln2.weight, self.ln2.bias, self.ln2.eps
            )
        else:
            x = x + residual
            residual = x
            x = self.ln2(x)

        x = self.mlp(x)
        x = x + residual
        return x


# ===================================================================
# From improved_selector_v2.py  (now in models/v6_selector.py)
# ===================================================================
# The selector classes live in their own module so they can evolve
# independently of the V6 backbone.  Re-export here for back-compat
# with any code that imports from this file.
from selector.v6_selector import (
    ModalityProbeHead,
    ImprovedModalitySelector,
    CurriculumScheduler,
)



class RoPEPositionalEncoding(nn.Module):
    """Standalone RoPE positional encoding (batch-first: [B, T, D])."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.0,
                 base: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.dropout = nn.Dropout(p=dropout)

        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos().unsqueeze(0))
        self.register_buffer('sin_cache', emb.sin().unsqueeze(0))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return torch.cat([-x[..., d:], x[..., :d]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        cos = self.cos_cache[:, :T, :]
        sin = self.sin_cache[:, :T, :]
        return self.dropout(x * cos + self._rotate_half(x) * sin)


class BatchedModalityEncoder(nn.Module):
    """Per-modality encoder that processes M modalities in one batched call.

    Drop-in replacement for the ModuleDict[ModalityEncoder] structure in V5+
    models when bmm-batched per-modal processing is desired. Holds num_layers
    `BatchedTransformerBlock`s, each with M_total stacked weight sets — so M
    modalities run through one batched forward instead of M sequential calls.

    NOTE: when `use_distill=True`, every layer's stride is fixed at construction.
    The per-layer `skip_ds` short-input safety from `ModalityEncoder` doesn't
    apply here because BatchedTransformerBlock's conv stride can't be toggled
    at runtime. Choose `num_layers` such that the input length stays > 1 after
    halving num_layers times.

    Forward expects a stacked tensor of shape [M_active, B, L, D] and an
    optional `mod_indices` LongTensor of length M_active selecting which of
    the M_total stored param sets to use (for top-k modality selection).

    Args:
        M_total:             total parameter sets stored (= num_modalities)
        d_model:             feature dim
        num_layers:          transformer layers (= num_layers_per_modal)
        nhead:               attention heads
        dropout:             dropout rate
        use_sparse_attn:     enable ProbSparse attention (uses BatchedTransformerBlock's
                             _sparse_attention)
        n_bottleneck:        bottleneck token count for sparse path (typically 0
                             for per-modal)
        factor:              ProbSparse factor
        sparse_attn_variant: 'orig' / 'opt' / 'opt_strat'
        use_distill:         per-layer stride-2 conv+pool downsample
    """

    def __init__(self, M_total: int, d_model: int, num_layers: int = 2,
                 nhead: int = 8, dropout: float = 0.1,
                 use_sparse_attn: bool = False, n_bottleneck: int = 0,
                 factor: int = 5, sparse_attn_variant: str = 'orig',
                 use_distill: bool = False,
                 strat_block_size: int = 8,
                 mlp_ratio: float = 1.0):
        super().__init__()
        # Lazy import to avoid circular dependency
        from models.fusion_bmm_parallel import BatchedTransformerBlock
        self.use_distill = use_distill
        self.M_total = M_total
        _strategy_map = {
            'opt_strat':     'stratified',
            'opt_strat_blk': 'stratified_block',
        }
        sampling_strategy = _strategy_map.get(sparse_attn_variant, 'global')
        self.layers = nn.ModuleList([
            BatchedTransformerBlock(
                M_total=M_total, hidden_size=d_model, num_heads=nhead,
                mlp_dim=int(d_model * mlp_ratio), dropout_rate=dropout,
                use_distill=use_distill,
                use_sparse_attn=use_sparse_attn,
                factor=factor,
                n_bottleneck=n_bottleneck,
                bottleneck_head=False,
                sampling_strategy=sampling_strategy,
                strat_block_size=strat_block_size,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x_stacked: torch.Tensor,
                mod_indices: Optional[torch.Tensor] = None,
                factor=None,
                capture_layer: Optional[int] = None,
                layer_keep: Optional[torch.Tensor] = None):
        """
        Args:
            x_stacked: [M_active, B, L, D]
            mod_indices: optional LongTensor [M_active], indices into M_total stored
                         param sets. None means use all M_total in original order.
            capture_layer: optional int. If set, also returns the layer output
                         AFTER index ``capture_layer`` (0-indexed). Returns
                         (final_output, captured_output). When None, returns
                         only the final output.
            layer_keep: optional float tensor [M_active, B, num_layers-1] for
                         ADMN-style per-(modality, sample) LayerDrop. Layer 0 is
                         always on; layer i>=1 uses column i-1: keep=1 runs the
                         layer, keep=0 is a residual passthrough (skip). Requires
                         ``use_distill=False`` (shapes must be preserved across
                         layers for the blend to be well-defined).
        Returns:
            [M_active, B, L_out, D] (or tuple if capture_layer set)
        """
        del factor  # BatchedTransformerBlock doesn't take a factor arg in fwd
        if layer_keep is not None:
            assert not self.use_distill, \
                'layer_keep (ADMN LayerDrop) needs use_distill=False (length-preserving layers)'
        captured = None
        for i, layer in enumerate(self.layers):
            x_new = layer(x_stacked, mod_indices)
            if layer_keep is not None and i >= 1:
                k = layer_keep[:, :, i - 1].view(x_new.shape[0], x_new.shape[1], 1, 1).to(x_new.dtype)
                x_stacked = x_new * k + x_stacked * (1.0 - k)
            else:
                x_stacked = x_new
            if capture_layer is not None and i == capture_layer:
                captured = x_stacked
        if capture_layer is not None:
            return x_stacked, captured
        return x_stacked


def _save_attention_heatmap_grid(attention_maps: Dict[str, torch.Tensor],
                                 save_path: str,
                                 modalities: List[str],
                                 num_layers: int,
                                 title: str = "MBT Fusion Attention Heatmaps"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(modalities), num_layers
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(3.5 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    for row, modality in enumerate(modalities):
        for col in range(n_cols):
            ax = axes[row][col]
            key = f'{modality}_layer_{col}'
            if key in attention_maps:
                attn = attention_maps[key]
                if attn.dim() == 4:
                    attn = attn.mean(dim=(0, 1))
                elif attn.dim() == 3:
                    attn = attn.mean(dim=0)
                im = ax.imshow(attn.cpu().float().numpy(), aspect='auto',
                               cmap='viridis', interpolation='nearest')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='gray')
            ax.set_title(f'{modality} / Layer {col}', fontsize=9)
            if col == 0:
                ax.set_ylabel('Query', fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel('Key', fontsize=8)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Attention heatmaps saved to {save_path}")


class BottleneckMLPAggregatorDownsample(nn.Module):
    """
    M × [B, K//2, d]
      → gate-weighted sum over M  → [B, K//2, d]
      → learned upsample K//2 → K → [B, K,   d]

    The upsample (nn.Linear on the sequence dim) restores the bottleneck to
    full K tokens so the next fusion layer always receives K bottleneck inputs.
    """

    def __init__(self, d_model: int, k_bottleneck: int, dropout: float = 0.1,
                 agg_mode: str = 'gate'):
        super().__init__()
        assert k_bottleneck % 2 == 0, "n_bottlenecks must be even for K//2 split"
        assert agg_mode in ('gate', 'mean'), agg_mode
        self.agg_mode = agg_mode                  # 'gate'=ours (learned) / 'mean'=vanilla-MBT
        self.k_bottleneck = k_bottleneck
        self.k_half = k_bottleneck // 2
        self.gate     = nn.Linear(d_model, 1)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)
        self.upsample = nn.Linear(self.k_half, k_bottleneck)

    def forward(self, bottleneck_list: List[torch.Tensor],
                selected_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            bottleneck_list: M tensors of shape [B, K_in, d]. Typically K_in == K//2
                (after a fusion-distill layer halves the sequence). When the caller
                hits a non-halving path (block stride=1, or short-T skip), K_in == K
                and the upsample is bypassed.
            selected_mask: optional [B, M] bool tensor. When given, gates for
                non-selected (sample, modality) entries are forced to softmax
                weight 0 by setting their logits to -inf, so only selected
                modalities contribute to the fused bottleneck per sample.
        Returns:
            [B, K, d]
        """
        tokens = torch.stack(bottleneck_list, dim=1)           # [B, M, K_in, d]
        if self.agg_mode == 'mean':
            # vanilla-MBT: uniform (masked) mean over modalities — no learned gate.
            if selected_mask is not None:
                w = selected_mask.float().unsqueeze(-1).unsqueeze(-1)   # [B,M,1,1]
                fused = (tokens * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
            else:
                fused = tokens.mean(dim=1)                     # [B, K_in, d]
        else:
            gate_logits = self.gate(tokens)                    # [B, M, K_in, 1]
            if selected_mask is not None:
                # mask: [B, M] -> [B, M, 1, 1] broadcast
                m = selected_mask.bool().unsqueeze(-1).unsqueeze(-1)
                gate_logits = gate_logits.masked_fill(~m, float('-inf'))
            gates  = F.softmax(gate_logits, dim=1)             # [B, M, K_in, 1]
            fused  = (gates * tokens).sum(dim=1)               # [B, K_in, d]
        fused  = self.dropout(fused)
        # Apply upsample only when input came from a halved layer (K_in == K//2)
        if fused.shape[1] == self.k_half:
            fused = self.upsample(fused.transpose(1, 2)).transpose(1, 2)
        return self.norm(fused)


class SimpleMBTFusionAdaptiveMLPDownsampleBmm(nn.Module):
    """V6Downsample fusion with bmm-batched parallelism across modalities.

    Mirrors `SimpleMBTFusionAdaptiveMLPDownsample` (per-modality split +
    `BottleneckMLPAggregatorDownsample` upsample) but the per-modality
    `TransformerBlockDS` is replaced with a single `BatchedTransformerBlock`
    per fusion layer that processes all M modalities in one batched forward.

    Per fusion layer:
      1. Stack [M, B, T+K, d]
      2. Run BatchedTransformerBlock(use_distill=True) → halves seq → [M, B, (T+K)//2, d]
      3. Split last k_half as compressed bottleneck per modality → list of M × [B, K//2, d]
      4. BottleneckMLPAggregatorDownsample: gate-aggregate + Linear upsample → [B, K, d]

    Supports both dense and sparse attention (`use_sparse_attn` +
    `sparse_attn_variant`) by delegating to BatchedTransformerBlock's
    `_attention` / `_sparse_attention` paths.
    """

    def __init__(self, input_dims: Dict[str, int],
                 hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, mlp_dim: int = 2048,
                 fusion_layer: int = 3, use_bottleneck: bool = True,
                 n_bottlenecks: int = 4, dropout_rate: float = 0.1,
                 output_dim: Optional[int] = None,
                 bottleneck_head_pos: bool = False,
                 max_seq_len: int = 500,
                 use_sparse_attn: bool = False,
                 use_sparse_moe: bool = False,
                 use_triton: bool = False,
                 num_experts: int = 4, expert_k: int = 1,
                 parallel_modalities: bool = False,
                 factor: int = 5,
                 sparse_attn_variant: str = 'orig',
                 downsample_min_len: int = 4,
                 n_fusion_distill: int = -1,
                 strat_block_size: int = 8,
                 bottleneck_init_mode: str = 'random',
                 add_pos_embeds: bool = False,
                 pos_embeds_max_len: int = 256,
                 pos_embed_mode: str = 'full',
                 share_fusion_params: bool = False,
                 fusion_mode: str = 'joint',
                 seq_modality_order: Optional[List[int]] = None,
                 seq_depth_flow: bool = False,
                 seq_random_order: bool = False,
                 bottleneck_agg_mode: str = 'gate',
                 **_unused_kwargs):
        super().__init__()
        if use_sparse_moe:
            raise NotImplementedError('No MoE in the bmm path.')
        assert fusion_mode in ('joint', 'sequential'), \
            f"fusion_mode must be 'joint' or 'sequential', got '{fusion_mode}'"
        if use_triton:
            raise NotImplementedError('Triton not compatible with batched LN.')
        assert n_bottlenecks % 2 == 0, 'n_bottlenecks must be even for K//2 split'
        assert bottleneck_init_mode in ('random', 'modal_mean_sinpe'), \
            f"bottleneck_init_mode must be 'random' or 'modal_mean_sinpe', got '{bottleneck_init_mode}'"
        self.bottleneck_init_mode = bottleneck_init_mode

        # Lazy import to avoid circular dependency
        from models.fusion_bmm_parallel import BatchedTransformerBlock

        self.all_modalities: List[str] = list(input_dims.keys())
        self.M_total = len(self.all_modalities)
        self.modality_to_idx = {m: i for i, m in enumerate(self.all_modalities)}
        self.hidden_size = hidden_size
        self.output_dim = output_dim if output_dim is not None else hidden_size
        self.use_bottleneck = use_bottleneck
        self.n_bottlenecks = n_bottlenecks
        self.k_half = n_bottlenecks // 2
        self.bottleneck_head_pos = bottleneck_head_pos
        self.num_layers = num_layers
        self.fusion_layer = fusion_layer
        self.downsample_min_len = downsample_min_len
        self.use_sparse_attn = use_sparse_attn
        self.sparse_attn_variant = sparse_attn_variant

        # Per-modality input projections (independent, small cost)
        self.input_projections = nn.ModuleDict()
        for modality, in_dim in input_dims.items():
            if in_dim != hidden_size:
                self.input_projections[modality] = nn.Linear(in_dim, hidden_size)
            else:
                self.input_projections[modality] = nn.Identity()

        # Per-modality additive PE at fusion input.
        # Modes:
        #   'full'      — pos_embeds[m] shape [1, L, H] per modality (Plan A original).
        #                 Cost: M × L × H (e.g., 6 × 128 × 128 ≈ 98k for IEMOCAP).
        #   'decoupled' — shared temporal_pe [1, L, H] + mod_id_pe [M, H] (our_models.py style).
        #                 Cost: L × H + M × H (e.g., 128×128 + 6×128 ≈ 17k for IEMOCAP).
        #   'id_only'   — only mod_id_pe [M, H], skip temporal_pe.
        #                 Cost: M × H (e.g., 6 × 128 = 768 for IEMOCAP).
        #   'gated_id'  — id_only modulated by learnable σ(pe_gate). gate init=0 → σ=0.5.
        #                 Gradient drives gate→1 if PE helps (IEMOCAP), →0 if harmful (DaliaHAR).
        #                 Cost: M × H + 1.
        self.add_pos_embeds = add_pos_embeds
        self.pos_embed_mode = pos_embed_mode
        if add_pos_embeds:
            assert pos_embed_mode in ('full', 'decoupled', 'id_only', 'gated_id'), \
                f"pos_embed_mode must be 'full'/'decoupled'/'id_only'/'gated_id', got {pos_embed_mode!r}"
            if pos_embed_mode == 'full':
                self.pos_embeds = nn.ParameterDict({
                    m: nn.Parameter(torch.randn(1, pos_embeds_max_len, hidden_size) * 0.02)
                    for m in self.all_modalities
                })
            elif pos_embed_mode == 'decoupled':
                self.temporal_pe = nn.Parameter(
                    torch.randn(1, pos_embeds_max_len, hidden_size) * 0.02)
                self.mod_id_pe = nn.Embedding(self.M_total, hidden_size)
                nn.init.normal_(self.mod_id_pe.weight, mean=0.0, std=0.02)
            elif pos_embed_mode == 'id_only':
                self.mod_id_pe = nn.Embedding(self.M_total, hidden_size)
                nn.init.normal_(self.mod_id_pe.weight, mean=0.0, std=0.02)
            else:  # 'gated_id'
                self.mod_id_pe = nn.Embedding(self.M_total, hidden_size)
                nn.init.normal_(self.mod_id_pe.weight, mean=0.0, std=0.02)
                self.pe_gate = nn.Parameter(torch.zeros(1))  # σ(0) = 0.5

        # Pre-fusion blocks (lyr < fusion_layer): no downsample, no bottleneck.
        # Fusion blocks (lyr >= fusion_layer): distill (stride-2 conv+pool) ONLY on
        # the last ``n_fusion_distill`` fusion layers. n_fusion_distill = -1 -> all
        # fusion layers distill (legacy default); 0 -> no fusion downsampling at all
        # (full-resolution attention through the whole stack). This is the REAL knob
        # for the distill ablation — ``downsample_min_len`` is inert here because the
        # block's conv stride is fixed at construction.
        _n_fuse_total = num_layers - fusion_layer
        _ndist = (_n_fuse_total if n_fusion_distill < 0
                  else min(n_fusion_distill, _n_fuse_total))
        self.n_fusion_distill = _ndist
        # A fusion layer distills iff it is among the LAST _ndist fusion layers.
        self._fusion_distill_flags = [
            (lyr >= fusion_layer
             and (lyr - fusion_layer) >= (_n_fuse_total - _ndist))
            for lyr in range(num_layers)
        ]
        _strategy_map = {
            'opt_strat':     'stratified',
            'opt_strat_blk': 'stratified_block',
        }
        sampling_strategy = _strategy_map.get(sparse_attn_variant, 'global')
        self.blocks = nn.ModuleList([
            BatchedTransformerBlock(
                M_total=self.M_total, hidden_size=hidden_size,
                num_heads=num_heads, mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                use_distill=self._fusion_distill_flags[lyr],
                use_sparse_attn=use_sparse_attn,
                factor=factor,
                n_bottleneck=n_bottlenecks if use_bottleneck else 0,
                bottleneck_head=bottleneck_head_pos,
                sampling_strategy=sampling_strategy,
                strat_block_size=strat_block_size,
                share_params=share_fusion_params,
            )
            for lyr in range(num_layers)
        ])

        if use_bottleneck:
            if bottleneck_init_mode == 'modal_mean_sinpe':
                # Data-driven bottleneck: at forward time, compute per-sample
                # modal-mean over (M, T) → [B, H], replicate to k positions, and
                # add a deterministic sinusoidal positional encoding to break
                # symmetry between the k tokens. No learnable bottleneck
                # parameter; zero seed-dependence on bottleneck init.
                pe_pos = torch.arange(n_bottlenecks).float().unsqueeze(1)
                pe_div = torch.exp(
                    torch.arange(0, hidden_size, 2).float()
                    * -(math.log(10000.0) / hidden_size)
                )
                pe = torch.zeros(n_bottlenecks, hidden_size)
                pe[:, 0::2] = torch.sin(pe_pos * pe_div)
                pe[:, 1::2] = torch.cos(pe_pos * pe_div)
                self.register_buffer(
                    'bottleneck_pe',
                    (pe * 0.1).unsqueeze(0),       # [1, k, H], scale 0.1 perturbation
                )
                self.bottleneck = None             # no learnable bottleneck
            else:
                self.bottleneck = nn.Parameter(
                    torch.randn(1, n_bottlenecks, hidden_size) * 0.02
                )
            # K//2 → K via Linear upsample
            self.bottleneck_aggregator = BottleneckMLPAggregatorDownsample(
                d_model=hidden_size,
                k_bottleneck=n_bottlenecks,
                dropout=dropout_rate,
                agg_mode=bottleneck_agg_mode,
            )
        else:
            self.bottleneck = None
            self.bottleneck_aggregator = None

        # ---- Sequential (recursive) fusion: 方案 A ------------------------
        # When fusion_mode == 'sequential', modalities are NOT fused jointly
        # each layer. Instead they are processed ONE AT A TIME in a fixed
        # order; a per-fusion-layer bottleneck state is carried forward and
        # updated by each incoming modality through a GRU-style gate (à la
        # Gated Recursive Fusion, arXiv:2507.02985). This trades MBT's
        # order-independent symmetric average for an ordered O(n) recurrence.
        self.fusion_mode = fusion_mode
        self.seq_modality_order = (list(seq_modality_order)
                                   if seq_modality_order is not None else None)
        # Order-randomized training: draw a fresh modality order each TRAINING
        # forward so the gates/blocks become order-robust (eval stays
        # deterministic — canonical order, or seq_modality_order if set).
        self.seq_random_order = seq_random_order
        # seq_depth_flow=False (variant A): later modalities read this layer's
        #   cross-modal "board" fresh each layer; bottleneck does NOT recur over
        #   depth for them (only modality 0 does). Matches "save each layer's
        #   tokens, reuse for the next modality" literally.
        # seq_depth_flow=True  (variant C): EVERY modality also carries its
        #   bottleneck across depth (like the original joint V6), blending that
        #   own carry with the per-layer board through a second gate before each
        #   block. Richer, closer to V6/GRF depth semantics.
        self.seq_depth_flow = seq_depth_flow
        if fusion_mode == 'sequential' and use_bottleneck:
            n_fuse = num_layers - fusion_layer
            # One cross-MODALITY update gate per fusion layer: z = sigmoid(W[h_old;
            # h_new]). Default (small-random) weight + zero bias -> z is
            # content-dependent but centred near 0.5 at init (balanced blend), so
            # the gate is immediately expressive yet starts roughly even between
            # the carried bottleneck and the incoming modality, then learns how
            # much of the carried state to keep vs. overwrite per modality step.
            self.seq_gates = nn.ModuleList(
                [nn.Linear(2 * hidden_size, hidden_size) for _ in range(n_fuse)]
            )
            for g in self.seq_gates:
                nn.init.zeros_(g.bias)
            # Cross-DEPTH blend gates (only when seq_depth_flow): mix a modality's
            # own depth-carried bottleneck with the per-layer board.
            if seq_depth_flow:
                self.seq_blend_gates = nn.ModuleList(
                    [nn.Linear(2 * hidden_size, hidden_size) for _ in range(n_fuse)]
                )
                for g in self.seq_blend_gates:
                    nn.init.zeros_(g.bias)
            else:
                self.seq_blend_gates = None
        else:
            self.seq_gates = None
            self.seq_blend_gates = None

        self.final_norm = nn.LayerNorm(hidden_size)
        if self.output_dim != hidden_size:
            self.output_projection = nn.Linear(hidden_size, self.output_dim)
        else:
            self.output_projection = nn.Identity()

    def _stack_inputs(self, inputs, available_modalities):
        # Project each modality and stack to [M_active, B, L, H]
        projected = []
        for m in available_modalities:
            proj = self.input_projections[m](inputs[m])    # [B, L, H]
            if self.add_pos_embeds:
                L = proj.shape[1]
                if self.pos_embed_mode == 'full':
                    proj = proj + self.pos_embeds[m][:, :L]
                elif self.pos_embed_mode == 'decoupled':
                    m_idx = self.modality_to_idx[m]
                    mid = self.mod_id_pe.weight[m_idx].view(1, 1, -1)
                    proj = proj + self.temporal_pe[:, :L] + mid
                elif self.pos_embed_mode == 'id_only':
                    m_idx = self.modality_to_idx[m]
                    mid = self.mod_id_pe.weight[m_idx].view(1, 1, -1)
                    proj = proj + mid
                else:  # 'gated_id'
                    m_idx = self.modality_to_idx[m]
                    mid = self.mod_id_pe.weight[m_idx].view(1, 1, -1)
                    gate = torch.sigmoid(self.pe_gate)
                    proj = proj + gate * mid
            projected.append(proj)
        return torch.stack(projected, dim=0)

    def forward(self, inputs: Dict[str, torch.Tensor],
                return_tokens: bool = False, factor=None,
                selected_mask: Optional[torch.Tensor] = None):
        """If ``selected_mask`` is a [B, M_total] bool tensor, the fusion
        runs on all available modalities but only the selected-per-sample
        modalities contribute to the bottleneck aggregator and final GAP.
        Indexed by ``self.all_modalities`` order — ``self._stack_inputs``
        preserves that order, so column j in mask = stream j here.
        """
        del factor
        available_modalities = [m for m in self.all_modalities if m in inputs]
        if len(available_modalities) == 0:
            raise ValueError('At least one modality must be provided')

        # Align selected_mask to available_modalities order (subset of canonical)
        active_mask = None
        if selected_mask is not None:
            avail_canon_idx = [self.modality_to_idx[m] for m in available_modalities]
            avail_canon = torch.tensor(avail_canon_idx, device=selected_mask.device,
                                        dtype=torch.long)
            active_mask = selected_mask.index_select(1, avail_canon)  # [B, M_active] bool

        device = inputs[available_modalities[0]].device
        batch_size = inputs[available_modalities[0]].shape[0]
        M_active = len(available_modalities)
        full = (M_active == self.M_total
                and available_modalities == self.all_modalities)
        mod_indices = (None if full else torch.tensor(
            [self.modality_to_idx[m] for m in available_modalities],
            dtype=torch.long, device=device,
        ))

        x = self._stack_inputs(inputs, available_modalities)        # [M, B, L, H]

        if self.use_bottleneck and self.bottleneck_init_mode == 'modal_mean_sinpe':
            # Per-sample modal mean over (M_active, L) → [B, H]; respect active_mask
            # (only-selected modalities contribute) if provided.
            if active_mask is not None:
                # active_mask: [B, M_active] → broadcast to [M_active, B, 1, 1]
                m_mask = active_mask.t().unsqueeze(-1).unsqueeze(-1).float()
                weighted = (x * m_mask).sum(dim=(0, 2))                        # [B, H]
                denom = (m_mask.sum(dim=0) * x.shape[2]).clamp(min=1.0)       # [B, 1]
                modal_mean = weighted / denom                                  # [B, H]
            else:
                modal_mean = x.mean(dim=(0, 2))                               # [B, H]
            bottleneck = (modal_mean.unsqueeze(1)
                          .expand(-1, self.n_bottlenecks, -1)
                          + self.bottleneck_pe)                                # [B, k, H]
            bottleneck = bottleneck.contiguous()
        elif self.use_bottleneck and self.bottleneck is not None:
            bottleneck = self.bottleneck.expand(batch_size, -1, -1).contiguous()
        else:
            bottleneck = None

        # ---- Pre-fusion layers (always batched, independent per modality) ----
        for lyr in range(self.fusion_layer):
            x = self.blocks[lyr](x, mod_indices)

        if self.fusion_mode == 'sequential' and bottleneck is not None:
            # 方案 A: process modalities one at a time, carry a gated bottleneck.
            x = self._fuse_sequential(x, bottleneck, mod_indices, M_active,
                                      device, active_mask)
        else:
            # ---- Joint MBT fusion (original path, unchanged) ----
            for lyr in range(self.fusion_layer, self.num_layers):
                block = self.blocks[lyr]
                if bottleneck is None:
                    # Fallback: cat-along-seq path (no bottleneck)
                    M, B, L, H = x.shape
                    x_cat = x.permute(1, 0, 2, 3).reshape(B, M * L, H).unsqueeze(0)
                    idx = (mod_indices[:1] if mod_indices is not None
                           else torch.zeros(1, dtype=torch.long, device=device))
                    x_cat = block(x_cat, idx).squeeze(0)
                    x = x_cat.reshape(B, M, -1, H).permute(1, 0, 2, 3).contiguous()
                    continue

                T = x.shape[2]
                bn_len = bottleneck.shape[1]

                # Stack bottleneck across modalities then concat along seq dim
                bn_stack = bottleneck.unsqueeze(0).expand(M_active, -1, -1, -1)
                if not self.bottleneck_head_pos:
                    combined = torch.cat([x, bn_stack], dim=2)          # [M, B, T+K, H]
                else:
                    combined = torch.cat([bn_stack, x], dim=2)

                # Run batched block. Note: the block's stride is FIXED at construction
                # to use_distill=(lyr >= fusion_layer). The downsample_min_len threshold
                # cannot toggle stride at runtime; instead we detect halving from L_out.
                out = block(combined, mod_indices)                      # [M, B, L_out, H]
                L_out = out.shape[2]
                block_halved = self._fusion_distill_flags[lyr]          # this layer's actual stride

                # ds_module trims odd L before halving (mirrors seq's _DualPathDownsample),
                # so expected halved length is integer-division: (T+bn_len)//2.
                if block_halved and L_out == (T + bn_len) // 2:
                    # Halved path: block produced the expected half-length output.
                    # Split: k_half tokens at the bottleneck end → aggregator upsamples
                    # back to K so the next layer receives K-token bottleneck.
                    k_half = self.k_half
                    if not self.bottleneck_head_pos:
                        new_bn_list = [out[i, :, -k_half:, :] for i in range(M_active)]
                        x = out[:, :, :-k_half, :]
                    else:
                        new_bn_list = [out[i, :, :k_half, :] for i in range(M_active)]
                        x = out[:, :, k_half:, :]
                    bottleneck = self.bottleneck_aggregator(
                        new_bn_list, selected_mask=active_mask)  # [B, K, H] via upsample
                else:
                    # Non-halved path (block stride=1, or unexpected L_out). Slice the
                    # full original bn_len; aggregator detects width != k_half and
                    # passes through without upsample.
                    if not self.bottleneck_head_pos:
                        new_bn_list = [out[i, :, T:T + bn_len, :] for i in range(M_active)]
                        x = out[:, :, :T, :]
                    else:
                        new_bn_list = [out[i, :, :bn_len, :] for i in range(M_active)]
                        x = out[:, :, bn_len:, :]
                    bottleneck = self.bottleneck_aggregator(
                        new_bn_list, selected_mask=active_mask)

        # Concat modalities along sequence dim, GAP for readout
        M_active_, B_, L_, H_ = x.shape
        fused_tokens = x.permute(1, 0, 2, 3).reshape(B_, M_active_ * L_, H_)
        fused_tokens = self.final_norm(fused_tokens)
        if return_tokens:
            return fused_tokens.reshape(B_, M_active_, L_, H_) #.permute(1, 0, 2, 3).contiguous()
        if active_mask is not None:
            # Per-sample mask-aware GAP: only average tokens belonging to
            # selected modalities. Expand [B, M_active] -> [B, M_active*L_].
            tok_mask = active_mask.float().unsqueeze(-1).expand(B_, M_active_, L_)
            tok_mask = tok_mask.reshape(B_, M_active_ * L_).unsqueeze(-1)
            denom = tok_mask.sum(dim=1).clamp_min(1.0)
            fused_repr = (fused_tokens * tok_mask).sum(dim=1) / denom
        else:
            fused_repr = fused_tokens.mean(dim=1)
        return self.output_projection(fused_repr)

    # ------------------------------------------------------------------
    # 方案 A — sequential (recursive) gated bottleneck fusion
    # ------------------------------------------------------------------
    @staticmethod
    def _gru_blend(gate, h_keep, h_in):
        """GRU-style gate (GRF GFU): z = sigmoid(W [h_keep ; h_in]);
        returns (1 - z) * h_keep + z * h_in. All [B, K, H]. Note blend(h, h) == h
        exactly (so a self-blend is a no-op regardless of z)."""
        z = torch.sigmoid(gate(torch.cat([h_keep, h_in], dim=-1)))   # [B, K, H]
        return (1.0 - z) * h_keep + z * h_in

    def _seq_gate_update(self, fuse_idx, h_old, h_new):
        """Cross-MODALITY update gate: blend the carried per-layer bottleneck
        ``h_old`` with an incoming modality's bottleneck ``h_new``. Zero-bias
        init → z ≈ 0.5 at start, then learns how much of the carried state to
        keep vs. overwrite per modality step."""
        return self._gru_blend(self.seq_gates[fuse_idx], h_old, h_new)

    def _seq_depth_blend(self, fuse_idx, h_carry, h_board):
        """Cross-DEPTH blend (only used when ``seq_depth_flow=True``): mix a
        modality's own depth-carried bottleneck ``h_carry`` with this layer's
        cross-modal board ``h_board`` before feeding the block. At the entry
        layer h_carry == h_board so this is an exact no-op."""
        return self._gru_blend(self.seq_blend_gates[fuse_idx], h_carry, h_board)

    def _fuse_sequential(self, x, bottleneck_init, mod_indices, M_active,
                         device, active_mask):
        """Sequential gated bottleneck fusion (方案 A).

        ``x`` is [M_active, B, L, H] after the pre-fusion layers. Modalities are
        processed ONE AT A TIME. The order is: a fresh random permutation each
        TRAINING forward if ``seq_random_order`` (eval falls back to the fixed
        order); else ``self.seq_modality_order``; else the available order. The
        first modality lays down a per-fusion-layer
        bottleneck trajectory (its bottleneck flows across depth, as in a
        single-stream bottleneck transformer). Every later modality runs its own
        block per layer and GATE-updates that layer's saved cross-modal state.

        The bottleneck a later modality READS at layer l depends on
        ``seq_depth_flow``:
          * False (A): bn_in = bn_state[l]  — read the per-layer board fresh; no
            cross-depth recurrence for later modalities.
          * True  (C): bn_in = blend(own depth-carry, bn_state[l]) — the modality
            also threads its bottleneck across depth (seeded from the board's
            entry layer) and blends it with each layer's board via a second gate.

        Modality tokens are collected back in the original modality order so the
        readout layout matches the joint path ([M_active, B, L_out, H]).

        Order matters by construction (this is the price of dropping MBT's
        symmetric average); see ``seq_modality_order`` to control it.
        """
        if active_mask is not None:
            raise NotImplementedError(
                "fusion_mode='sequential' does not support per-sample selector "
                "masks (per_sample_topk=True). Use batch-mode top-k or "
                "no_selector so unselected modalities never enter fusion.")

        fusion_layers = list(range(self.fusion_layer, self.num_layers))
        # Per-fusion-layer carried bottleneck state, accumulated across modalities.
        bn_state = {lyr: None for lyr in fusion_layers}

        if self.seq_random_order and self.training:
            # Order-randomized training: one fresh permutation per forward (shared
            # across the batch — per-sample orders can't share the batched block).
            order = torch.randperm(M_active, device=device).tolist()
        elif self.seq_modality_order is None:
            order = list(range(M_active))
        else:
            # Configured order is interpreted as positions in the ACTIVE list;
            # keep only valid positions and append any missing ones (robust to
            # dynamic M_active under modality selection).
            order = [i for i in self.seq_modality_order if 0 <= i < M_active]
            order += [i for i in range(M_active) if i not in order]

        out_tokens = [None] * M_active
        for step, mi in enumerate(order):
            xm = x[mi:mi + 1]                                   # [1, B, L, H]
            if mod_indices is None:
                mod_idx_m = torch.tensor([mi], dtype=torch.long, device=device)
            else:
                mod_idx_m = mod_indices[mi:mi + 1]

            # within-modality depth carry. mod 0 seeds from the init bottleneck;
            # later modalities (only relevant when seq_depth_flow) seed from the
            # board's entry layer so the entry-layer blend is an exact no-op.
            if step == 0:
                bn_depth = bottleneck_init
            elif self.seq_depth_flow:
                bn_depth = bn_state[self.fusion_layer]
            for lyr in fusion_layers:
                block = self.blocks[lyr]
                fuse_idx = lyr - self.fusion_layer
                # First modality flows the bottleneck across depth; later
                # modalities read this layer's accumulated cross-modal board
                # (optionally blended with their own depth-carry).
                if step == 0:
                    bn_in = bn_depth
                elif self.seq_depth_flow:
                    bn_in = self._seq_depth_blend(fuse_idx, bn_depth, bn_state[lyr])
                else:
                    bn_in = bn_state[lyr]                        # [B, K, H]
                T = xm.shape[2]
                bn_len = bn_in.shape[1]
                bn_stack = bn_in.unsqueeze(0)                   # [1, B, K, H]
                if not self.bottleneck_head_pos:
                    combined = torch.cat([xm, bn_stack], dim=2)     # [1, B, T+K, H]
                else:
                    combined = torch.cat([bn_stack, xm], dim=2)

                out = block(combined, mod_idx_m)                # [1, B, L_out, H]
                L_out = out.shape[2]

                if L_out == (T + bn_len) // 2:
                    k_half = self.k_half
                    if not self.bottleneck_head_pos:
                        bn_piece = out[:, :, -k_half:, :]       # [1, B, k_half, H]
                        xm = out[:, :, :-k_half, :]
                    else:
                        bn_piece = out[:, :, :k_half, :]
                        xm = out[:, :, k_half:, :]
                else:
                    if not self.bottleneck_head_pos:
                        bn_piece = out[:, :, T:T + bn_len, :]
                        xm = out[:, :, :T, :]
                    else:
                        bn_piece = out[:, :, :bn_len, :]
                        xm = out[:, :, bn_len:, :]
                # Single-modality aggregation reuses the learned k_half->K upsample.
                bn_new = self.bottleneck_aggregator([bn_piece[0]])  # [B, K, H]

                if step == 0:
                    bn_state[lyr] = bn_new
                    bn_depth = bn_new                           # carry across depth (mod 0)
                else:
                    bn_state[lyr] = self._seq_gate_update(
                        fuse_idx, bn_state[lyr], bn_new)
                    if self.seq_depth_flow:
                        bn_depth = bn_new                       # carry across depth (later mods)
            out_tokens[mi] = xm[0]                              # [B, L_out, H]

        # Introspection only (read-only; does not affect the forward output):
        # stash the final carried bottleneck board per fusion layer for
        # bottleneck-saturation early-exit analysis.
        self._last_bn_state = {lyr: bn_state[lyr].detach()
                               for lyr in fusion_layers if bn_state[lyr] is not None}
        self._last_seq_order = list(order)                     # processing order (active indices)
        return torch.stack(out_tokens, dim=0)                  # [M_active, B, L_out, H]


class DualVideoBottleneckModelV6Downsample(nn.Module):
    """
    V6 model with Conv1d+MaxPool downsampling at each MBT fusion layer.

    Flat standalone class (no parent class beyond nn.Module). The bottleneck
    fusion module is SimpleMBTFusionAdaptiveMLPDownsampleBmm:

      - Each fusion layer runs a dedicated Conv1d(k=3)+GELU+MaxPool(stride=2)
        on the full concatenated [modality|bottleneck] output sequence.
      - Modality tokens halve in length each fusion layer.
      - Bottleneck tokens are compressed to K//2 per modality, fed into
        BottleneckMLPAggregatorDownsample, then upsampled back to K — so the
        bottleneck input to every layer is always the full K tokens.

    Per-modality encoding uses BatchedModalityEncoder (bmm-batched across the
    M modalities).  Improved selector (`ImprovedModalitySelector`) is built
    when `no_selector=False`; otherwise modalities are weighted uniformly.

    Forward returns:
        labels=None:  (output, primary_idx, modality_weights, selected_modalities)
        labels given: (output, primary_idx, modality_weights, selected_modalities, aux_losses)

    Notes:
        downsample_min_len (int, default 4): skip modality downsampling when the
            modality sequence length is at or below this threshold.
        n_bottlenecks must be even.
        use_batched_fusion is accepted for back-compat but ignored — this
            variant always uses the batched path.
    """

    def __init__(self, cfg, output_dim=1, input_length=45, d_model=384, nhead=16,
                 num_layers_per_modal=2, num_layers=2, dropout=0.1, verbose=True,
                 video_low_dim=384, video_high_dim=1024,
                 use_bottleneck=True, use_sparse_attn=False, n_bottlenecks=4, fusion_layer=1,
                 use_sparse_moe=False, num_experts=4, expert_k=1,
                 factor=5, internal_dim=768, bottleneck_head_pos=False,
                 mselector_mlp_hidden_dim=128,
                 top_k=None,
                 use_weighted_factor=False,
                 selector_video_source='low',
                 encoder_video_source='high',
                 no_selector=False,
                 use_triton=False,
                 num_classes=5,
                 lambda_probe=0.1,
                 lambda_diversity=0.01,
                 lambda_reinforce=0.01,
                 lambda_sparsity=0.1,
                 downsample_min_len=4,
                 n_fusion_distill=-1,
                 bottleneck_agg_mode: str = 'gate',
                 sparse_attn_variant='orig',
                 strat_block_size: int = 8,
                 use_batched_fusion=False,
                 per_modal_distill: bool = False,
                 per_modal_downsample_min_len: int = 4,
                 use_interaction_matrix: bool = True,
                 use_holo_bias: bool = False,
                 holo_scale: float = 1.0,
                 selector_downsample_factor: int = 1,
                 selector_downsample_mode: str = 'avg_pool',
                 selector_input_source: str = 'raw',
                 selector_depth: int = 0,
                 use_multi_pool: bool = False,
                 use_tx_summarizer: bool = False,
                 tx_d_model: int = 64,
                 tx_nhead: int = 4,
                 tx_layers: int = 1,
                 tx_dim_ff: int = 128,
                 tx_dropout: float = 0.1,
                 use_concat_probe_head: bool = False,
                 bottleneck_init_mode: str = 'random',
                 fusion_add_pos_embeds: bool = False,
                 fusion_pos_embeds_max_len: int = 256,
                 fusion_pos_embed_mode: str = 'full',
                 fusion_mlp_ratio: float = 1.0,
                 mlp_ratio: float = 1.0,
                 share_fusion_params: bool = False,
                 fusion_mode: str = 'joint',
                 seq_modality_order=None,
                 seq_depth_flow: bool = False,
                 seq_random_order: bool = False,
                 head_mode: str = 'entropy',
                 ):
        super().__init__()
        assert head_mode in ('entropy', 'gap'), \
            f"head_mode must be 'entropy' or 'gap', got '{head_mode}'"

        assert selector_video_source in ('low', 'high'), \
            f"selector_video_source must be 'low' or 'high', got '{selector_video_source}'"
        assert encoder_video_source in ('low', 'high'), \
            f"encoder_video_source must be 'low' or 'high', got '{encoder_video_source}'"

        n_mods = len(cfg.modalities) if hasattr(cfg, 'modalities') else 3
        assert top_k is None or (isinstance(top_k, int) and 1 <= top_k <= n_mods), \
            f"top_k must be None or an int in [1, {n_mods}], got {top_k}"

        self.top_k = top_k
        self.use_weighted_factor = use_weighted_factor
        self.no_selector = no_selector
        # Classifier head: 'entropy' = per-modality late fusion combined by
        # per-prediction entropy; 'gap' = global average pool over modalities and
        # tokens then a single regressor pass (classic v6 head).
        self.head_mode = head_mode
        self.video_low_dim = video_low_dim
        self.video_high_dim = video_high_dim
        self.selector_video_source = selector_video_source
        self.encoder_video_source = encoder_video_source
        self.lambda_probe = lambda_probe
        self.lambda_diversity = lambda_diversity
        self.lambda_reinforce = lambda_reinforce
        self.lambda_sparsity = lambda_sparsity
        self.sparse_attn_variant = sparse_attn_variant
        self.strat_block_size = strat_block_size

        self.selector_video_dim = video_low_dim if selector_video_source == 'low' else video_high_dim
        self.encoder_video_dim = video_low_dim if encoder_video_source == 'low' else video_high_dim
        self.selector_downsample_factor = int(selector_downsample_factor)
        assert self.selector_downsample_factor >= 1, \
            f"selector_downsample_factor must be >=1, got {self.selector_downsample_factor}"
        assert selector_downsample_mode in ('avg_pool', 'stride'), \
            f"selector_downsample_mode must be 'avg_pool' or 'stride', got '{selector_downsample_mode}'"
        self.selector_downsample_mode = selector_downsample_mode
        assert selector_input_source in ('raw', 'enc_layer1', 'enc_layer2'), \
            f"selector_input_source must be 'raw', 'enc_layer1', or 'enc_layer2', got '{selector_input_source}'"
        self.selector_input_source = selector_input_source
        # Map: 'enc_layer1' -> capture after layer 0, 'enc_layer2' -> after layer 1.
        self._enc_capture_idx = {
            'raw': None, 'enc_layer1': 0, 'enc_layer2': 1,
        }[selector_input_source]

        self.modalities = cfg.modalities if hasattr(cfg, 'modalities') else ['video', 'audio', 'eeg']
        self.num_modalities = len(self.modalities)
        self.variates = cfg.variates if hasattr(cfg, 'variates') else {
            'audio': 768, 'video': video_high_dim, 'eeg': 30
        }
        self.variates['video'] = self.encoder_video_dim

        self.input_length = input_length
        self.verbose = verbose
        self.d_model = d_model
        self.fc_dim = d_model
        self.bottleneck_head_pos = bottleneck_head_pos
        self.factor = factor
        self.internal_dim = internal_dim
        self.pooled_dim = internal_dim
        self.per_modal_distill = per_modal_distill
        self.per_modal_downsample_min_len = per_modal_downsample_min_len

        # `use_batched_fusion` is accepted for back-compat but ignored: this
        # variant is hard-coded to the batched path.
        self.use_batched_fusion = True

        # ---- Per-modality input projection
        self.temporal_pos_encoder = RoPEPositionalEncoding(
            d_model=self.pooled_dim, max_len=input_length)

        self.input_projectors = nn.ModuleDict()
        for modality in self.modalities:
            input_dim = self.variates.get(modality, 768)
            self.input_projectors[modality] = nn.Linear(input_dim, self.internal_dim)

        # ---- Improved modality selector
        # selector input dim depends on selector_input_source:
        #   'raw'           -> per-modality variates (small, 1-3 for HAR)
        #   'enc_layer{N}'  -> encoder output dim (= self.pooled_dim/internal_dim)
        self.selector_dim_dict = {}
        if self.selector_input_source == 'raw':
            for m in self.modalities:
                self.selector_dim_dict[m] = (
                    self.selector_video_dim if m == 'video' else self.variates.get(m, 256)
                )
        else:
            for m in self.modalities:
                self.selector_dim_dict[m] = self.pooled_dim

        if not no_selector:
            self.modality_selector = ImprovedModalitySelector(
                modalities=self.modalities,
                low_dim_dict=self.selector_dim_dict,
                mlp_hidden_dim=mselector_mlp_hidden_dim,
                num_classes=num_classes,
                uniform_dim=256,
                use_interaction_matrix=use_interaction_matrix,
                use_holo_bias=use_holo_bias,
                holo_scale=holo_scale,
                selector_depth=selector_depth,
                use_multi_pool=use_multi_pool,
                use_tx_summarizer=use_tx_summarizer,
                tx_d_model=tx_d_model,
                tx_nhead=tx_nhead,
                tx_layers=tx_layers,
                tx_dim_ff=tx_dim_ff,
                tx_dropout=tx_dropout,
                use_concat_probe_head=use_concat_probe_head,
            )
        else:
            self.modality_selector = None

        # ---- Per-modality pre/post processing modules
        self.temporal_summarization = nn.ModuleDict()
        for modality in self.modalities:
            self.temporal_summarization[modality] = nn.Sequential(
                nn.Conv1d(self.internal_dim, self.internal_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )

        self.layer_norms = nn.ModuleDict({
            m: nn.LayerNorm(self.internal_dim) for m in self.modalities
        })
        self.output_norms = nn.ModuleDict({
            m: nn.LayerNorm(self.pooled_dim) for m in self.modalities
        })
        self.encoder_output_dropout = nn.Dropout(p=min(dropout * 2, 0.3))

        # Kept as an empty ModuleDict for state_dict / iteration back-compat.
        # The per-modality ModalityEncoder of the original parent has been
        # replaced by the batched encoder below.
        self.modality_encoders = nn.ModuleDict()

        # ---- Batched per-modality encoder (replaces parent's per-modal dict)
        self.batched_modality_encoder = BatchedModalityEncoder(
            M_total=self.num_modalities,
            d_model=self.pooled_dim,
            num_layers=num_layers_per_modal,
            nhead=nhead,
            dropout=dropout,
            use_sparse_attn=False,  # per-modal encoder kept dense in this variant
            n_bottleneck=0,
            factor=factor,
            sparse_attn_variant=sparse_attn_variant,
            strat_block_size=strat_block_size,
            use_distill=per_modal_distill,
            mlp_ratio=mlp_ratio,
        )

        # ---- Bottleneck fusion (V6Downsample bmm fusion)
        fusion_input_dims = {m: self.pooled_dim for m in self.modalities}
        self.bottleneck_fusion = SimpleMBTFusionAdaptiveMLPDownsampleBmm(
            input_dims=fusion_input_dims,
            hidden_size=d_model,
            num_layers=num_layers,
            num_heads=nhead,
            mlp_dim=int(d_model * fusion_mlp_ratio),
            fusion_layer=fusion_layer,
            use_bottleneck=use_bottleneck,
            n_bottlenecks=n_bottlenecks,
            dropout_rate=dropout,
            output_dim=d_model,
            use_sparse_moe=use_sparse_moe,
            use_sparse_attn=use_sparse_attn,
            num_experts=num_experts,
            expert_k=expert_k,
            bottleneck_head_pos=bottleneck_head_pos,
            factor=factor,
            use_triton=use_triton,
            downsample_min_len=downsample_min_len,
            n_fusion_distill=n_fusion_distill,
            bottleneck_agg_mode=bottleneck_agg_mode,
            sparse_attn_variant=sparse_attn_variant,
            strat_block_size=strat_block_size,
            bottleneck_init_mode=bottleneck_init_mode,
            add_pos_embeds=fusion_add_pos_embeds,
            pos_embeds_max_len=fusion_pos_embeds_max_len,
            pos_embed_mode=fusion_pos_embed_mode,
            share_fusion_params=share_fusion_params,
            fusion_mode=fusion_mode,
            seq_modality_order=seq_modality_order,
            seq_depth_flow=seq_depth_flow,
            seq_random_order=seq_random_order,
        )

        # ---- Classifier head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        if verbose:
            n_fusion = num_layers - fusion_layer
            print(f"Initializing DualVideoBottleneckModelV6Downsample")
            print(f"  Modalities:    {self.modalities}")
            print(f"  Video low-dim: {video_low_dim}, Video high-dim: {video_high_dim}")
            print(f"  Selector video source: {selector_video_source} (dim={self.selector_video_dim})")
            print(f"  Encoder video source:  {encoder_video_source} (dim={self.encoder_video_dim})")
            print(f"  Internal dim: {self.internal_dim}, Pooled dim: {self.pooled_dim}")
            print(f"  n_bottlenecks: {n_bottlenecks}  k_half: {n_bottlenecks // 2}")
            print(f"  Fusion layers: {n_fusion}  (each has independent Conv1d+MaxPool)")
            print(f"  Modality tokens halve each fusion layer  "
                  f"(min_len={downsample_min_len})")
            print(f"  Bottleneck: K//2 per modality → aggregator → upsample → K={n_bottlenecks}")
            print(f"  Per-modal distill: {per_modal_distill}")
            print(f"  Sparse attn variant: {sparse_attn_variant}  "
                  f"use_weighted_factor: {use_weighted_factor}")
            print(f"  use_triton: {use_triton}")
            print(f"  Batched fusion: True  (per-modal encoder also batched: True)")
            if self.modality_selector is not None:
                print(f"  ImprovedModalitySelector: dim_dict={self.selector_dim_dict}")
                print(f"    num_classes={num_classes}, mlp_hidden={mselector_mlp_hidden_dim}")
                print(f"    selector_downsample_factor={self.selector_downsample_factor}")
                print(f"    use_interaction_matrix={use_interaction_matrix}  "
                      f"use_holo_bias={use_holo_bias}  holo_scale={holo_scale}")
                if top_k is not None:
                    print(f"    Hard top-K: top_k={top_k}")
                else:
                    print(f"    Selection disabled (all modalities kept)")
            else:
                print(f"  No modality selector: all modalities used, uniform weights")
            print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    # ------------------------------------------------------------------
    # Selector temporal downsample helper
    # ------------------------------------------------------------------
    def downsample_selector_inputs(self, sel_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress each modality along time by ``selector_downsample_factor``.

        Input/output shape: ``[B, T, C]``.  Factor==1 returns input unchanged;
        for sequences shorter than the factor, the modality is left at full
        length (avoid collapsing too aggressively for short windows).

        Mode controlled by ``selector_downsample_mode``:
          - ``avg_pool`` (default, legacy): ``F.avg_pool1d`` over time, smooth.
          - ``stride``: keep every f-th token (no smoothing). Preserves sharp
                        local peaks / bursts that avg-pool would wash out —
                        useful when per-sample Shapley heterogeneity comes
                        from short transients (e.g. activity transitions).
        """
        f = self.selector_downsample_factor
        if f <= 1:
            return sel_inputs
        mode = getattr(self, 'selector_downsample_mode', 'avg_pool')
        out = {}
        for m, x in sel_inputs.items():
            if x.dim() != 3:
                out[m] = x
                continue
            T = x.shape[1]
            if T < f:
                out[m] = x
                continue
            if mode == 'stride':
                out[m] = x[:, ::f, :].contiguous()
            else:
                xt = x.transpose(1, 2)
                xt = F.avg_pool1d(xt, kernel_size=f, stride=f, ceil_mode=False)
                out[m] = xt.transpose(1, 2).contiguous()
        return out

    # ------------------------------------------------------------------
    # Selector freeze/unfreeze utilities
    # ------------------------------------------------------------------
    def freeze_selector(self, keep_probes_training: bool = True):
        if self.modality_selector is None:
            return
        for name, p in self.modality_selector.named_parameters():
            if keep_probes_training and 'probe_head' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def unfreeze_selector(self):
        if self.modality_selector is None:
            return
        for p in self.modality_selector.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Attention recording / heatmap utilities
    # ------------------------------------------------------------------
    def enable_attention_recording(self, enable: bool = True,
                                   layers: Optional[List[int]] = None):
        # SimpleMBTFusionAdaptiveMLPDownsampleBmm uses BatchedTransformerBlock
        # which does not expose the same enable_attention_recording API, so
        # this is a no-op for the bmm path. Kept for API back-compat.
        fusion = self.bottleneck_fusion
        if hasattr(fusion, 'enable_attention_recording'):
            fusion.enable_attention_recording(enable, layers)

    def save_attention_heatmaps(self, save_path: str,
                                title: str = "MBT Fusion Attention Heatmaps"):
        fusion = self.bottleneck_fusion
        if not hasattr(fusion, 'get_attention_maps'):
            print("Warning: bottleneck_fusion does not support attention recording.")
            return
        attention_maps = fusion.get_attention_maps()
        if not attention_maps:
            print("Warning: No attention maps recorded.")
            return
        available_modalities, recorded_layers = [], set()
        for key in attention_maps:
            parts = key.split('_layer_')
            modality = parts[0]
            if modality not in available_modalities:
                available_modalities.append(modality)
            recorded_layers.add(int(parts[1]))
        num_layers = max(recorded_layers) + 1 if recorded_layers else 0
        _save_attention_heatmap_grid(
            attention_maps, save_path, available_modalities, num_layers, title
        )
        if hasattr(fusion, 'clear_attention_maps'):
            fusion.clear_attention_maps()

    # ------------------------------------------------------------------
    # Batched per-modality encoding (single batched call across modalities)
    # ------------------------------------------------------------------
    def _encode_modalities(self, projected_features, selected_modalities, base_factor,
                            capture_layer: Optional[int] = None,
                            layer_keep: Optional[torch.Tensor] = None):
        """Encode all modalities in one batched forward via BatchedModalityEncoder.

        Per-modality preprocessing (interpolate / layer_norm / temporal_pos_encoder)
        and post-processing (output_norm / encoder_output_dropout) still run in
        a Python loop because they are cheap pointwise ops with per-modality
        ModuleDict weights — only the heavy transformer-encoder body is batched.

        When ``capture_layer`` is set, also returns a dict of intermediate
        encoder outputs (after layer ``capture_layer``, 0-indexed), pre-output-
        norm and pre-dropout. Used by selector_input_source='enc_layer*' to
        feed mid-encoder features into the modality selector.
        """
        # Determine the active modality order (same iteration order as parent)
        active_mods = []
        for modality in self.modalities:
            if modality not in projected_features:
                continue
            if selected_modalities is not None and modality not in selected_modalities:
                continue
            active_mods.append(modality)
        if not active_mods:
            return ({}, {}) if capture_layer is not None else {}

        # ---- Per-modality preprocessing (interpolate, layer_norm, temporal_pos)
        pre_outputs = []
        for modality in active_mods:
            x = projected_features[modality]
            x = x.permute(0, 2, 1)
            if x.shape[2] > self.input_length:
                x = F.interpolate(x, size=self.input_length, mode='linear',
                                  align_corners=False)
            x = x.permute(0, 2, 1)
            x = self.layer_norms[modality](x)
            x = self.temporal_pos_encoder(x)
            pre_outputs.append(x)

        # ---- Stack to [M_active, B, T, D] and run batched encoder
        x_stacked = torch.stack(pre_outputs, dim=0)
        # mod_indices selects which of M_total stored weight sets to use
        if len(active_mods) == self.num_modalities and active_mods == self.modalities:
            mod_indices = None
        else:
            mod_indices = torch.tensor(
                [self.modalities.index(m) for m in active_mods],
                dtype=torch.long, device=x_stacked.device,
            )
        if capture_layer is not None:
            x_stacked, captured = self.batched_modality_encoder(
                x_stacked, mod_indices=mod_indices, capture_layer=capture_layer)
            captured_dict = {m: captured[i] for i, m in enumerate(active_mods)}
        else:
            # ADMN LayerDrop: layer_keep is [M, B, L-1] in self.modalities order; reorder
            # to active_mods if a strict subset is present (ADMN keeps all -> no reorder).
            lk = layer_keep
            if lk is not None and active_mods != self.modalities:
                sel = [self.modalities.index(m) for m in active_mods]
                lk = lk[sel]
            x_stacked = self.batched_modality_encoder(
                x_stacked, mod_indices=mod_indices, layer_keep=lk)

        # ---- Per-modality post-processing (output_norm, dropout)
        processed = {}
        for i, modality in enumerate(active_mods):
            y = x_stacked[i]
            y = self.output_norms[modality](y)
            y = self.encoder_output_dropout(y)
            processed[modality] = y
        if capture_layer is not None:
            return processed, captured_dict
        return processed

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, high_dim_inputs, low_dim_inputs, training=True,
                factor=None, return_selection_info=True, labels=None,
                override_selected_modalities=None, layer_keep=None,
                return_prefix_logits=False):
        selector_video_dict = (low_dim_inputs if self.selector_video_source == 'low'
                               else high_dim_inputs)
        encoder_video_dict = (low_dim_inputs if self.encoder_video_source == 'low'
                              else high_dim_inputs)

        # Two execution paths:
        #   A) selector_input_source == 'raw' (legacy)
        #      selector first (on raw downsampled input) → encoder (on selected) → fusion
        #   B) selector_input_source == 'enc_layer{N}' (new)
        #      project → encoder on ALL (capture layer N) → selector(captured)
        #      → fusion (with per-sample mask or filtered modalities)
        base_factor = factor if factor is not None else self.factor
        use_enc_features = (self._enc_capture_idx is not None
                            and not self.no_selector)

        # ---- Per-modality input projection (used in both paths)
        projected_features = {}
        for modality in self.modalities:
            if modality == 'video':
                if 'video' in encoder_video_dict:
                    projected_features['video'] = self.input_projectors['video'](
                        encoder_video_dict['video']
                    )
            else:
                if modality in high_dim_inputs:
                    projected_features[modality] = self.input_projectors[modality](
                        high_dim_inputs[modality]
                    )

        # ---- Modality selection
        if self.no_selector:
            batch_size = next(iter(high_dim_inputs.values())).shape[0]
            device = next(iter(high_dim_inputs.values())).device
            modality_weights = torch.full(
                (batch_size, self.num_modalities), 1.0 / self.num_modalities, device=device
            )
            primary_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
            selected_modalities = None
            processed_modalities = self._encode_modalities(
                projected_features, selected_modalities, base_factor, layer_keep=layer_keep)
        elif use_enc_features:
            # PATH B: encoder first (on ALL modalities), capture layer N,
            # use that as selector input. Then fusion uses full encoder output.
            processed_modalities, captured = self._encode_modalities(
                projected_features, None, base_factor,
                capture_layer=self._enc_capture_idx)
            # Optional temporal downsample of selector input (same factor).
            selector_inputs = self.downsample_selector_inputs(captured)
            primary_idx, modality_weights, selected_modalities = self.modality_selector(
                selector_inputs, top_k=self.top_k, training=training)
            # Note: with per_sample_topk=True, selected_modalities is None and
            # the per-sample mask gates fusion. With per_sample_topk=False and
            # selected_modalities != None, we filter processed for batch-mode.
            if (selected_modalities is not None
                    and not getattr(self.modality_selector, 'per_sample_topk', False)):
                processed_modalities = {
                    m: processed_modalities[m] for m in selected_modalities
                    if m in processed_modalities
                }
        else:
            # PATH A: legacy — selector first on raw, encoder filters by selected
            selector_inputs = {}
            for modality in self.modalities:
                if modality == 'video':
                    if 'video' in selector_video_dict:
                        selector_inputs['video'] = selector_video_dict['video']
                else:
                    if modality in low_dim_inputs:
                        selector_inputs[modality] = low_dim_inputs[modality]
                    elif modality in high_dim_inputs:
                        selector_inputs[modality] = high_dim_inputs[modality]
            selector_inputs = self.downsample_selector_inputs(selector_inputs)
            primary_idx, modality_weights, selected_modalities = self.modality_selector(
                selector_inputs, top_k=self.top_k, training=training)
            if override_selected_modalities is not None:
                selected_modalities = override_selected_modalities
            processed_modalities = self._encode_modalities(
                projected_features, selected_modalities, base_factor)

        # Expose for CGGM aux classifiers (per-modality post-encoder features).
        self._last_processed_modalities = processed_modalities

        # ---- Fusion factor (optional per-modality weighting)
        if self.use_weighted_factor and not self.no_selector:
            fusion_factor = {
                m: modality_weights[:, i].mean().item() * base_factor * self.num_modalities
                for i, m in enumerate(self.modalities)
                if m in processed_modalities
            }
        else:
            fusion_factor = base_factor

        # ---- Bottleneck fusion + classifier
        # If selector ran in per-sample mode it exposes a [B, M_total] mask;
        # forward it to fusion so the aggregator + GAP only count selected
        # modalities per sample.
        selector_mask = None
        if (self.modality_selector is not None
                and getattr(self.modality_selector, 'per_sample_topk', False)):
            selector_mask = getattr(self.modality_selector, '_selected_mask', None)
        fused = self.bottleneck_fusion(
            processed_modalities, return_tokens=True, factor=fusion_factor,
            selected_mask=selector_mask,
        )  # [B, M, L, H]

        prefix_logits = None
        if self.head_mode == 'gap':
            # Global average pooling: pool the fused [B, M, L, H] tokens over BOTH
            # the modality and token axes -> [B, H], then one regressor pass.
            gap_repr = fused.mean(dim=(1, 2))                 # [B, H]
            output = self.regressor(gap_repr)                # [B, C]
            # Per-prefix readouts (for deep supervision / patience-aware training).
            # In sequential fusion the first-k PROCESSED modalities' tokens do not
            # depend on later ones, so GAP over fused[:, order[:k]] == a k-only
            # forward's readout. Cumulative-mean over the processing order gives all
            # M prefix readouts in one pass; prefix_logits[:, -1] == output.
            if return_prefix_logits and getattr(self.bottleneck_fusion, '_last_seq_order', None) is not None:
                order = self.bottleneck_fusion._last_seq_order            # active indices, processing order
                idx = torch.as_tensor(order, dtype=torch.long, device=fused.device)
                tok_proc = fused.mean(dim=2).index_select(1, idx)         # [B, M_active, H] in processing order
                csum = torch.cumsum(tok_proc, dim=1)
                denom = torch.arange(1, tok_proc.shape[1] + 1, dtype=csum.dtype,
                                     device=csum.device).view(1, -1, 1)
                prefix_logits = self.regressor(csum / denom)             # [B, M_active, C]
        else:
            # Per-modality late fusion: split [B, M, L, H] into M streams, average
            # each over the L tokens -> [B, H], classify each independently, then
            # combine the M logits using per-prediction entropy as the weights.
            mod_repr = fused.mean(dim=2)                          # [B, M, H]
            mod_logits = self.regressor(mod_repr)                 # [B, M, C] (Linear on last dim)
            log_probs = F.log_softmax(mod_logits, dim=-1)         # [B, M, C]
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # [B, M]
            weights = F.softmax(-entropy, dim=1)                  # [B, M] (lower entropy -> higher weight)
            output = (weights.unsqueeze(-1) * mod_logits).sum(dim=1)  # [B, C]

        if return_selection_info:
            if labels is not None and training and not self.no_selector:
                aux_losses = self._compute_all_aux_losses(
                    labels, modality_weights, selected_modalities, output
                )
                if prefix_logits is not None:
                    return output, primary_idx, modality_weights, selected_modalities, aux_losses, prefix_logits
                return output, primary_idx, modality_weights, selected_modalities, aux_losses
            if prefix_logits is not None:
                return output, primary_idx, modality_weights, selected_modalities, prefix_logits
            return output, primary_idx, modality_weights, selected_modalities

        if prefix_logits is not None:
            return output, prefix_logits
        return output

    # ------------------------------------------------------------------
    # Auxiliary losses (probe / sparsity / reinforce)
    # ------------------------------------------------------------------
    def _compute_all_aux_losses(self, labels, modality_weights, selected_modalities, output):
        aux_losses = {}

        is_classification = labels.dtype in (torch.long, torch.int)
        if is_classification:
            raw_aux = self.modality_selector.compute_auxiliary_losses(labels=labels)
            if 'probe_loss' in raw_aux and self.lambda_probe > 0:
                aux_losses['probe_loss'] = self.lambda_probe * raw_aux['probe_loss']

        if self.lambda_sparsity > 0 and selected_modalities is not None:
            selected_indices = set(self.modalities.index(m) for m in selected_modalities
                                   if m in self.modalities)
            non_selected_weights = []
            for i in range(self.num_modalities):
                if i not in selected_indices:
                    non_selected_weights.append(modality_weights[:, i])
            if non_selected_weights:
                sparsity_loss = torch.stack(non_selected_weights, dim=0).mean()
                aux_losses['sparsity_loss'] = self.lambda_sparsity * sparsity_loss

        if selected_modalities is not None and self.top_k is not None:
            with torch.no_grad():
                if output.shape[-1] > 1 and labels.dtype in (torch.long, torch.int):
                    task_loss = F.cross_entropy(output.detach(), labels, reduction='none')
                else:
                    task_loss = F.mse_loss(output.detach().squeeze(-1), labels.float(), reduction='none')

                if not hasattr(self, '_reward_baseline'):
                    self._reward_baseline = task_loss.mean().item()
                else:
                    self._reward_baseline = (0.9 * self._reward_baseline
                                             + 0.1 * task_loss.mean().item())
                advantage = -(task_loss - self._reward_baseline)

            selected_indices = [self.modalities.index(m) for m in selected_modalities
                                 if m in self.modalities]
            if selected_indices:
                log_probs = torch.log(modality_weights[:, selected_indices] + 1e-8)
                log_prob_action = log_probs.sum(dim=1)
                reinforce_loss = -(advantage * log_prob_action).mean()
                aux_losses['reinforce_loss'] = self.lambda_reinforce * reinforce_loss

        return aux_losses
