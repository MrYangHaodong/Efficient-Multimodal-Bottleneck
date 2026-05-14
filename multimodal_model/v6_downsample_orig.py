"""
Dual Video Bottleneck Model — V6 Downsample (Flattened Standalone).

Self-contained module containing only the classes required to construct and
run ``DualVideoBottleneckModelV6Downsample``. The model class has no parent
beyond ``nn.Module`` — all V5 base init/forward logic plus V6Downsample
overrides have been inlined.

Retained classes (in dependency order):
    ProbSparseAttention, SparseMoEFeedForward, TransformerBlock,
    _DualPathDownsample, TransformerBlockDS,
    RoPEPositionalEncoding, ModalityEncoder,
    BottleneckMLPAggregatorDownsample, SimpleMBTFusionAdaptiveMLPDownsample,
    DualVideoBottleneckModelV6Downsample

Selector classes (``ModalityProbeHead``, ``ImprovedModalitySelector``,
``CurriculumScheduler``) are re-exported from ``.v6_selector``.
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
# ProbSparse attention (used inside TransformerBlock / TransformerBlockDS)
# ===================================================================

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5, scale=None, n_bottleneck=16, bottleneck_head=True):
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
                non_bottleneck_samples = torch.randint(self.n_bottleneck, L, (L, U_part), device=x.device)
                bottleneck_indices = torch.arange(self.n_bottleneck, device=x.device).unsqueeze(0).expand(L, -1)
            else:
                non_bottleneck_samples = torch.randint(0, L - self.n_bottleneck, (L, U_part), device=x.device)
                bottleneck_indices = torch.arange(L - self.n_bottleneck, L, device=x.device).unsqueeze(0).expand(L, -1)
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
            # Manual path retained so the attention matrix can be exposed.
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


# ===================================================================
# Sparse MoE FFN (used by TransformerBlock when use_sparse_moe=True)
# ===================================================================

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


# ===================================================================
# Transformer blocks (vanilla + downsample-aware)
# ===================================================================

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
        elif use_triton and TRITON_AVAILABLE and TritonProbSparseAttention is not None:
            self.attn = TritonProbSparseAttention(
                hidden_size, num_heads, dropout=dropout_rate,
                factor=factor, n_bottleneck=n_bottleneck,
                bottleneck_head=bottleneck_head,
            )
        else:
            # ProbSparseAttention is the only supported variant in this
            # standalone file. The `sparse_attn_variant` / `strat_block_size`
            # kwargs are accepted for back-compat but ignored.
            AttnClass = ProbSparseAttention
            self.attn = AttnClass(
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


class _DualPathDownsample(nn.Module):
    """Conv1d(stride=2) + MaxPool1d(stride=2) → element-wise sum → GELU → LayerNorm.

    Both paths operate on the input x directly (not sequentially), giving
    a learned strided representation and a peak-activation representation that
    complement each other.  The sum is a parameter-free residual-like shortcut
    that stabilises gradients across many fusion layers.

    Input/output: [B, L, d] → [B, L//2, d]  (L must be even; caller ensures this).
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
        xT  = x.transpose(1, 2)                         # [B, d, L]
        out = self.act(self.conv(xT)) + self.pool(xT)   # [B, d, L//2]
        return self.norm(out.transpose(1, 2))            # [B, L//2, d]


class TransformerBlockDS(TransformerBlock):
    """TransformerBlock with dual-path downsampling injected between attention and FFN.

    After the post-attention residual add, _DualPathDownsample halves the sequence
    length ([B, T+K, d] → [B, (T+K)//2, d]).  The FFN then runs on the shorter
    sequence, which is both faster and forces the network to work with more
    compressed representations before the next fusion layer.

    Args:
        downsample_d: channel dimension for _DualPathDownsample (= hidden_size).
        All other args forwarded to TransformerBlock.

    The forward method accepts an extra keyword-only arg ``skip_ds`` (default False).
    When True, the downsampling is skipped (used for sequences that are already too
    short, controlled by the calling fusion module).
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
            x = self.ds_module(x)   # [B, L//2, d]

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual
        return x


# ===================================================================
# Selector re-export (selector lives in v6_selector.py)
# ===================================================================
from selector.v6_selector import (
    ModalityProbeHead,
    ImprovedModalitySelector,
    CurriculumScheduler,
)


# ===================================================================
# Positional encoding + per-modality encoder
# ===================================================================

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


class ModalityEncoder(nn.Module):
    """Per-modality encoder.

    By default, stacks `num_layers` plain TransformerBlocks (no downsample).
    When `use_distill=True`, stacks TransformerBlockDS instead — each layer
    halves the sequence length via Conv1d(stride=2) + MaxPool1d(stride=2)
    + GELU + LayerNorm injected between attention and FFN.
    """

    def __init__(self, d_model, nhead=8, num_layers=2, dropout=0.1, max_len=500,
                 use_sparse_attn=False, n_bottlenecks=0, factor=5, use_triton=False,
                 sparse_attn_variant='orig',
                 use_distill: bool = False,
                 downsample_min_len: int = 4,
                 strat_block_size: int = 8):
        super().__init__()
        self.use_distill = use_distill
        self.downsample_min_len = downsample_min_len

        if use_distill:
            self.transformer_encoder = nn.ModuleList([
                TransformerBlockDS(
                    hidden_size=d_model,
                    num_heads=nhead,
                    mlp_dim=int(d_model * 2.0),
                    dropout_rate=dropout,
                    use_sparse_moe=False,
                    num_experts=4,
                    expert_k=1,
                    use_sparse_attn=use_sparse_attn,
                    n_bottleneck=n_bottlenecks,
                    factor=factor,
                    use_triton=use_triton,
                    sparse_attn_variant=sparse_attn_variant,
                    strat_block_size=strat_block_size,
                    downsample_d=d_model,
                ) for _ in range(num_layers)
            ])
        else:
            self.transformer_encoder = nn.ModuleList([
                TransformerBlock(
                    hidden_size=d_model,
                    num_heads=nhead,
                    mlp_dim=int(d_model * 2.0),
                    dropout_rate=dropout,
                    use_sparse_moe=False,
                    num_experts=4,
                    expert_k=1,
                    use_sparse_attn=use_sparse_attn,
                    n_bottleneck=n_bottlenecks,
                    factor=factor,
                    use_triton=use_triton,
                    sparse_attn_variant=sparse_attn_variant,
                    strat_block_size=strat_block_size,
                ) for _ in range(num_layers)
            ])

    def forward(self, x, factor=None):
        for layer in self.transformer_encoder:
            if self.use_distill:
                # Per-layer skip when L is already short to avoid degenerate sizes.
                skip_ds = x.shape[1] <= self.downsample_min_len
                x = layer(x, factor=factor, skip_ds=skip_ds)
            else:
                x = layer(x, factor=factor)
        return x


# ===================================================================
# Bottleneck aggregator (downsample-aware, with learned upsample)
# ===================================================================

class BottleneckMLPAggregatorDownsample(nn.Module):
    """
    M × [B, K//2, d]
      → gate-weighted sum over M  → [B, K//2, d]
      → learned upsample K//2 → K → [B, K,   d]

    The upsample (nn.Linear on the sequence dim) restores the bottleneck to
    full K tokens so the next fusion layer always receives K bottleneck inputs.
    """

    def __init__(self, d_model: int, k_bottleneck: int, dropout: float = 0.1):
        super().__init__()
        assert k_bottleneck % 2 == 0, "n_bottlenecks must be even for K//2 split"
        self.k_bottleneck = k_bottleneck
        self.k_half = k_bottleneck // 2
        self.gate     = nn.Linear(d_model, 1)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)
        self.upsample = nn.Linear(self.k_half, k_bottleneck)

    def forward(self, bottleneck_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            bottleneck_list: M tensors of shape [B, K_in, d]. Typically K_in == K//2
                (after a fusion-distill layer halves the sequence). When the caller
                hits a non-halving path (block stride=1, or short-T skip), K_in == K
                and the upsample is bypassed.
        Returns:
            [B, K, d]
        """
        tokens = torch.stack(bottleneck_list, dim=1)           # [B, M, K_in, d]
        gates  = F.softmax(self.gate(tokens), dim=1)           # [B, M, K_in, 1]
        fused  = (gates * tokens).sum(dim=1)                   # [B, K_in, d]
        fused  = self.dropout(fused)
        # Apply upsample only when input came from a halved layer (K_in == K//2)
        if fused.shape[1] == self.k_half:
            fused = self.upsample(fused.transpose(1, 2)).transpose(1, 2)
        return self.norm(fused)


# ===================================================================
# Fusion module: SimpleMBTFusionAdaptiveMLPDownsample
# (flattened — base SimpleMBTFusionAdaptive and SimpleMBTFusionAdaptiveMLP
# have been inlined; no other fusion variants are kept.)
# ===================================================================

class SimpleMBTFusionAdaptiveMLPDownsample(nn.Module):
    """MBT fusion where downsampling happens inside each fusion TransformerBlock.

    Each fusion layer (per modality):
      1. Concat [modality | bottleneck]                          [B, T+K, d]
      2. Inside TransformerBlockDS.forward():
           a. ln1 → attention → dropout → residual add          [B, T+K, d]
           b. _DualPathDownsample (when T > downsample_min_len)  [B, (T+K)//2, d]
           c. ln2 → FFN → residual add                          [B, (T+K)//2, d]
      3. Split: last k_half tokens → bn_compressed [B, K//2, d]
                remainder          → modality_next [B, ~T//2, d]
      4. BottleneckMLPAggregatorDownsample: K//2 × M → gate → upsample → K [B, K, d]

    Modality tokens halve each fusion layer; bottleneck always returns to K tokens.
    Each (modality, fusion_layer) block owns its own _DualPathDownsample weights.
    """

    def __init__(self,
                 input_dims: Dict[str, int],
                 hidden_size: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 mlp_dim: int = 2048,
                 fusion_layer: int = 3,
                 use_bottleneck: bool = True,
                 n_bottlenecks: int = 4,
                 dropout_rate: float = 0.1,
                 output_dim: Optional[int] = None,
                 use_sparse_moe: bool = False,
                 use_sparse_attn: bool = False,
                 num_experts: int = 4,
                 expert_k: int = 1,
                 parallel_modalities: bool = False,
                 bottleneck_head_pos: bool = False,
                 factor: int = 5,
                 use_triton: bool = False,
                 sparse_attn_variant: str = 'orig',
                 strat_block_size: int = 8,
                 downsample_min_len: int = 4):
        super().__init__()

        self.all_modalities = list(input_dims.keys())
        self.parallel_modalities = parallel_modalities
        if self.parallel_modalities:
            print("Parallel!!!")
        self._cuda_streams: Optional[List[torch.cuda.Stream]] = None
        self.hidden_size = hidden_size
        self.output_dim = output_dim if output_dim is not None else hidden_size
        self.use_bottleneck = use_bottleneck
        self.use_sparse_attn = use_sparse_attn
        self.factor = factor
        self.n_bottlenecks = n_bottlenecks
        self.use_sparse_moe = use_sparse_moe
        self.bottleneck_head_pos = bottleneck_head_pos
        self.sparse_attn_variant = sparse_attn_variant
        self.strat_block_size = strat_block_size

        assert n_bottlenecks % 2 == 0, "n_bottlenecks must be even"
        self.k_half = n_bottlenecks // 2
        self.downsample_min_len = downsample_min_len

        self.input_projections = nn.ModuleDict()
        for modality, input_dim in input_dims.items():
            if input_dim != hidden_size:
                self.input_projections[modality] = nn.Linear(input_dim, hidden_size)
            else:
                self.input_projections[modality] = nn.Identity()

        max_seq_len = 500
        self.pos_embeds = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, max_seq_len, hidden_size) * 0.02)
            for modality in self.all_modalities
        })

        self.num_layers = num_layers
        self.fusion_layer = fusion_layer
        self.blocks = nn.ModuleDict()

        # Pre-fusion layers: plain TransformerBlocks per modality.
        # Fusion layers: TransformerBlockDS per modality (each owns its own
        # _DualPathDownsample). Build them in one pass.
        for lyr in range(num_layers):
            is_fusion = lyr >= fusion_layer
            for modality in self.all_modalities:
                block_name = f'{modality}_layer_{lyr}'
                if is_fusion:
                    self.blocks[block_name] = TransformerBlockDS(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        dropout_rate=dropout_rate,
                        use_sparse_moe=use_sparse_moe,
                        num_experts=num_experts,
                        expert_k=expert_k,
                        use_sparse_attn=self.use_sparse_attn,
                        n_bottleneck=self.n_bottlenecks,
                        factor=self.factor,
                        use_triton=use_triton,
                        sparse_attn_variant=sparse_attn_variant,
                        strat_block_size=strat_block_size,
                        downsample_d=hidden_size,
                    )
                else:
                    self.blocks[block_name] = TransformerBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        dropout_rate=dropout_rate,
                        use_sparse_moe=use_sparse_moe,
                        num_experts=num_experts,
                        expert_k=expert_k,
                        use_sparse_attn=self.use_sparse_attn,
                        n_bottleneck=self.n_bottlenecks,
                        factor=self.factor,
                        use_triton=use_triton,
                        sparse_attn_variant=sparse_attn_variant,
                        strat_block_size=strat_block_size,
                    )

        if use_bottleneck:
            self.bottleneck = nn.Parameter(
                torch.randn(1, n_bottlenecks, hidden_size) * 0.02
            )
        else:
            self.bottleneck = None

        self.final_norm = nn.LayerNorm(hidden_size)

        if output_dim != hidden_size:
            self.output_projection = nn.Linear(hidden_size, output_dim)
        else:
            self.output_projection = nn.Identity()

        # Downsample-aware bottleneck aggregator (replaces simple averaging
        # used by SimpleMBTFusionAdaptive base).
        self.bottleneck_aggregator = BottleneckMLPAggregatorDownsample(
            d_model=hidden_size,
            k_bottleneck=n_bottlenecks,
            dropout=dropout_rate,
        )

    # ------------------------------------------------------------------
    # Attention-recording helpers (unchanged from base)
    # ------------------------------------------------------------------
    def enable_attention_recording(self, enable: bool = True, layers: Optional[List[int]] = None):
        for name, block in self.blocks.items():
            if layers is None:
                block._record_attention = enable
            else:
                lyr = int(name.split('_layer_')[-1])
                block._record_attention = enable and (lyr in layers)
            if hasattr(block.attn, '_record_attention'):
                block.attn._record_attention = block._record_attention
            if not enable:
                block._attention_weights = None
                if hasattr(block.attn, '_attention_weights'):
                    block.attn._attention_weights = None

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        maps = {}
        for name, block in self.blocks.items():
            if block._attention_weights is not None:
                maps[name] = block._attention_weights
        return maps

    def clear_attention_maps(self):
        for block in self.blocks.values():
            block._attention_weights = None
            if hasattr(block.attn, '_attention_weights'):
                block.attn._attention_weights = None

    def _get_cuda_streams(self, num_streams: int) -> List[torch.cuda.Stream]:
        if self._cuda_streams is None or len(self._cuda_streams) < num_streams:
            self._cuda_streams = [torch.cuda.Stream() for _ in range(num_streams)]
        return self._cuda_streams[:num_streams]

    # ------------------------------------------------------------------
    # Sequential / parallel fusion-layer processing (downsample-aware)
    # ------------------------------------------------------------------
    def _process_modalities_sequential(
            self,
            x: Dict[str, torch.Tensor],
            bottleneck: torch.Tensor,
            available_modalities: List[str],
            lyr: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        new_bn = []
        k_half = self.k_half

        for modality in available_modalities:
            T = x[modality].shape[1]
            combined = (
                torch.cat([x[modality], bottleneck], dim=1)
                if not self.bottleneck_head_pos else
                torch.cat([bottleneck, x[modality]], dim=1)
            )
            block   = self.blocks[f'{modality}_layer_{lyr}']
            skip_ds = T <= self.downsample_min_len
            output  = block(combined, skip_ds=skip_ds)

            if not skip_ds:
                # Block downsampled: [B, (T+K)//2, d] — split last k_half as bottleneck
                if not self.bottleneck_head_pos:
                    new_bn.append(output[:, -k_half:, :])
                    x[modality] = output[:, :-k_half, :]
                else:
                    new_bn.append(output[:, :k_half, :])
                    x[modality] = output[:, k_half:, :]
            else:
                # T too short; block returned [B, T+K, d] — slice original positions
                if not self.bottleneck_head_pos:
                    x[modality] = output[:, :T]
                    bn_full     = output[:, T:]
                else:
                    bn_full     = output[:, :self.n_bottlenecks]
                    x[modality] = output[:, self.n_bottlenecks:]
                new_bn.append(F.avg_pool1d(
                    bn_full.transpose(1, 2), kernel_size=2, stride=2
                ).transpose(1, 2))

        bottleneck = self.bottleneck_aggregator(new_bn)
        return x, bottleneck

    def _process_modalities_parallel_cuda(
            self,
            x: Dict[str, torch.Tensor],
            bottleneck: torch.Tensor,
            available_modalities: List[str],
            lyr: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        streams = self._get_cuda_streams(len(available_modalities))
        results = []
        for i, modality in enumerate(available_modalities):
            with torch.cuda.stream(streams[i]):
                T       = x[modality].shape[1]
                combined = (
                    torch.cat([x[modality], bottleneck], dim=1)
                    if not self.bottleneck_head_pos else
                    torch.cat([bottleneck, x[modality]], dim=1)
                )
                block   = self.blocks[f'{modality}_layer_{lyr}']
                skip_ds = T <= self.downsample_min_len
                output  = block(combined, skip_ds=skip_ds)
                results.append((modality, output, T, skip_ds, streams[i]))

        k_half = self.k_half
        new_bn = []
        for modality, output, T, skip_ds, stream in results:
            stream.synchronize()
            if not skip_ds:
                if not self.bottleneck_head_pos:
                    new_bn.append(output[:, -k_half:, :])
                    x[modality] = output[:, :-k_half, :]
                else:
                    new_bn.append(output[:, :k_half, :])
                    x[modality] = output[:, k_half:, :]
            else:
                if not self.bottleneck_head_pos:
                    x[modality] = output[:, :T]
                    bn_full     = output[:, T:]
                else:
                    bn_full     = output[:, :self.n_bottlenecks]
                    x[modality] = output[:, self.n_bottlenecks:]
                new_bn.append(F.avg_pool1d(
                    bn_full.transpose(1, 2), kernel_size=2, stride=2
                ).transpose(1, 2))

        bottleneck = self.bottleneck_aggregator(new_bn)
        return x, bottleneck

    def _add_positional_embedding(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        pos_embed = self.pos_embeds[modality]
        if seq_len <= pos_embed.shape[1]:
            return x + pos_embed[:, :seq_len, :]
        else:
            pos_embed_interp = torch.nn.functional.interpolate(
                pos_embed.permute(0, 2, 1),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
            return x + pos_embed_interp

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                return_tokens: bool = False,
                factor=None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        available_modalities = [m for m in self.all_modalities if m in inputs]

        if len(available_modalities) == 0:
            raise ValueError("At least one modality must be provided")

        batch_size = list(inputs.values())[0].shape[0]

        x = {}
        for modality in available_modalities:
            x[modality] = self.input_projections[modality](inputs[modality])

        bottleneck = None
        if self.use_bottleneck and self.bottleneck is not None:
            bottleneck = self.bottleneck.expand(batch_size, -1, -1)

        for lyr in range(self.num_layers):
            if lyr < self.fusion_layer:
                for modality in available_modalities:
                    block = self.blocks[f'{modality}_layer_{lyr}']
                    x[modality] = block(x[modality], factor=factor)
            else:
                if self.use_bottleneck and bottleneck is not None:
                    if self.parallel_modalities and torch.cuda.is_available() and len(available_modalities) > 1:
                        x, bottleneck = self._process_modalities_parallel_cuda(
                            x, bottleneck, available_modalities, lyr
                        )
                    else:
                        x, bottleneck = self._process_modalities_sequential(
                            x, bottleneck, available_modalities, lyr
                        )
                else:
                    all_tokens = []
                    seq_lengths = []
                    for modality in available_modalities:
                        all_tokens.append(x[modality])
                        seq_lengths.append(x[modality].shape[1])

                    combined = torch.cat(all_tokens, dim=1)
                    block = self.blocks[f'{available_modalities[0]}_layer_{lyr}']
                    combined = block(combined, factor)

                    start_idx = 0
                    for i, modality in enumerate(available_modalities):
                        end_idx = start_idx + seq_lengths[i]
                        x[modality] = combined[:, start_idx:end_idx]
                        start_idx = end_idx

        fused_tokens = torch.cat([x[modality] for modality in available_modalities], dim=1)
        output_tokens = self.final_norm(fused_tokens)

        if return_tokens:
            return output_tokens

        fused_representation = output_tokens.mean(dim=1)
        output = self.output_projection(fused_representation)
        return output


# ===================================================================
# DualVideoBottleneckModelV6Downsample — SOLE model class.
# All V5 base init/forward logic and V6Downsample overrides inlined.
# ===================================================================

class DualVideoBottleneckModelV6Downsample(nn.Module):
    """
    V6 Downsample model: improved selector + hard top-K + per-fusion-layer
    Conv1d+MaxPool downsampling.

    Forward returns:
        labels=None:  (output, primary_idx, modality_weights, selected_modalities)
        labels given: (output, primary_idx, modality_weights, selected_modalities, aux_losses)

    n_bottlenecks must be even.

    Args of note:
        downsample_min_len (int, default 4): skip modality downsampling when the
            modality sequence length is at or below this threshold.
        use_batched_fusion (bool): accepted for back-compat but always disabled
            in this standalone file.
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
                 sparse_attn_variant='orig',
                 strat_block_size: int = 8,
                 use_batched_fusion=False,
                 per_modal_distill: bool = False,
                 per_modal_downsample_min_len: int = 4,
                 use_interaction_matrix: bool = True,
                 use_holo_bias: bool = False,
                 holo_scale: float = 1.0,
                 ):
        super().__init__()

        assert selector_video_source in ('low', 'high'), \
            f"selector_video_source must be 'low' or 'high', got '{selector_video_source}'"
        assert encoder_video_source in ('low', 'high'), \
            f"encoder_video_source must be 'low' or 'high', got '{encoder_video_source}'"

        n_mods = len(cfg.modalities) if hasattr(cfg, 'modalities') else 3
        assert top_k is None or (isinstance(top_k, int) and 1 <= top_k <= n_mods), \
            f"top_k must be None or an int in [1, {n_mods}], got {top_k}"

        # ----- store core config -----
        self.top_k = top_k
        self.use_weighted_factor = use_weighted_factor
        self.no_selector = no_selector
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

        # `use_batched_fusion` is retained as a kwarg for back-compat, but
        # this standalone file only supports the non-batched configuration.
        self.use_batched_fusion = False
        self.batched_modality_encoder = None

        # ----- temporal pos enc + input projectors -----
        self.temporal_pos_encoder = RoPEPositionalEncoding(
            d_model=self.pooled_dim, max_len=input_length
        )

        self.input_projectors = nn.ModuleDict()
        for modality in self.modalities:
            input_dim = self.variates.get(modality, 768)
            self.input_projectors[modality] = nn.Linear(input_dim, self.internal_dim)

        # ----- modality selector (optional) -----
        self.selector_dim_dict = {}
        for m in self.modalities:
            self.selector_dim_dict[m] = (
                self.selector_video_dim if m == 'video' else self.variates.get(m, 256)
            )

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
            )
        else:
            self.modality_selector = None

        # ----- per-modality layer norm + encoders -----
        self.layer_norms = nn.ModuleDict({
            m: nn.LayerNorm(self.internal_dim) for m in self.modalities
        })

        self.per_modal_distill = per_modal_distill
        self.per_modal_downsample_min_len = per_modal_downsample_min_len
        self.modality_encoders = nn.ModuleDict()
        for modality in self.modalities:
            self.modality_encoders[modality] = ModalityEncoder(
                d_model=self.pooled_dim, nhead=nhead, num_layers=num_layers_per_modal,
                dropout=dropout, max_len=input_length, use_sparse_attn=use_sparse_attn,
                n_bottlenecks=0, factor=factor, use_triton=use_triton,
                sparse_attn_variant=sparse_attn_variant,
                strat_block_size=strat_block_size,
                use_distill=per_modal_distill,
                downsample_min_len=per_modal_downsample_min_len,
            )

        self.output_norms = nn.ModuleDict({
            m: nn.LayerNorm(self.pooled_dim) for m in self.modalities
        })

        self.encoder_output_dropout = nn.Dropout(p=min(dropout * 2, 0.3))

        # ----- fusion module (V6 Downsample variant; sole option) -----
        fusion_input_dims = {m: self.pooled_dim for m in self.modalities}
        self.bottleneck_fusion = SimpleMBTFusionAdaptiveMLPDownsample(
            input_dims=fusion_input_dims,
            hidden_size=d_model,
            num_layers=num_layers,
            num_heads=nhead,
            mlp_dim=int(d_model * 2.0),
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
            sparse_attn_variant=sparse_attn_variant,
            strat_block_size=strat_block_size,
        )

        # ----- classifier head -----
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
            print(f"  use_triton: {use_triton}")
            if not no_selector:
                print(f"  ImprovedModalitySelector: dim_dict={self.selector_dim_dict}")
                print(f"    num_classes={num_classes}, mlp_hidden={mselector_mlp_hidden_dim}")
                print(f"    lambda_probe={lambda_probe}, lambda_diversity={lambda_diversity}")
                print(f"    use_interaction_matrix={use_interaction_matrix}  "
                      f"use_holo_bias={use_holo_bias}  holo_scale={holo_scale}")
                if top_k is not None:
                    print(f"    Hard top-K: top_k={top_k}")
                else:
                    print(f"    Selection disabled (all modalities kept)")
            else:
                print(f"  No modality selector: all modalities used, uniform weights")
            print(f"  Weighted factor fusion: {use_weighted_factor}")
            print(f"  Sparse attn variant: {sparse_attn_variant}")
            print(f"  n_bottlenecks: {n_bottlenecks}  k_half: {n_bottlenecks // 2}")
            print(f"  Fusion layers: {n_fusion}  (each has independent Conv1d+MaxPool)")
            print(f"  Modality tokens halve each fusion layer  "
                  f"(min_len={downsample_min_len})")
            print(f"  Bottleneck: K//2 per modality → aggregator → upsample → K={n_bottlenecks}")
            print(f"  Per-modal distill: {per_modal_distill}")
            print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    # ------------------------------------------------------------------
    # Selector freeze/unfreeze helpers
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

    def enable_attention_recording(self, enable: bool = True,
                                   layers: Optional[List[int]] = None):
        self.bottleneck_fusion.enable_attention_recording(enable, layers)

    # ------------------------------------------------------------------
    # Per-modality encoding helpers
    # ------------------------------------------------------------------
    def _process_modality_no_weight(self, x, modality, modality_idx, factor=None):
        x = x.permute(0, 2, 1)
        if x.shape[2] > self.input_length:
            x = F.interpolate(x, size=self.input_length, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)
        x = self.layer_norms[modality](x)
        x = self.temporal_pos_encoder(x)
        x = self.modality_encoders[modality](x, factor)
        x = self.output_norms[modality](x)
        x = self.encoder_output_dropout(x)
        return x

    def _encode_modalities(self, projected_features, selected_modalities, base_factor):
        """Encode all available modalities sequentially (one ModuleDict call per modality).

        The V6Downsample-specific batched encoder path (use_batched_fusion=True)
        is disabled in this standalone file, so this is the only code path.
        """
        processed = {}
        for modality_idx, modality in enumerate(self.modalities):
            if modality not in projected_features:
                continue
            if selected_modalities is not None and modality not in selected_modalities:
                continue
            processed[modality] = self._process_modality_no_weight(
                projected_features[modality], modality, modality_idx, base_factor
            )
        return processed

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, high_dim_inputs, low_dim_inputs, training=True,
                factor=None, return_selection_info=True, labels=None):
        selector_video_dict = (low_dim_inputs if self.selector_video_source == 'low'
                               else high_dim_inputs)
        encoder_video_dict = (low_dim_inputs if self.encoder_video_source == 'low'
                              else high_dim_inputs)

        if self.no_selector:
            batch_size = next(iter(high_dim_inputs.values())).shape[0]
            device = next(iter(high_dim_inputs.values())).device
            modality_weights = torch.full(
                (batch_size, self.num_modalities), 1.0 / self.num_modalities, device=device
            )
            primary_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
            selected_modalities = None
        else:
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

            primary_idx, modality_weights, selected_modalities = self.modality_selector(
                selector_inputs,
                top_k=self.top_k,
                training=training,
            )

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

        base_factor = factor if factor is not None else self.factor
        processed_modalities = self._encode_modalities(
            projected_features, selected_modalities, base_factor)

        if self.use_weighted_factor and not self.no_selector:
            fusion_factor = {
                m: modality_weights[:, i].mean().item() * base_factor * self.num_modalities
                for i, m in enumerate(self.modalities)
                if m in processed_modalities
            }
        else:
            fusion_factor = base_factor

        fused = self.bottleneck_fusion(
            processed_modalities, return_tokens=False, factor=fusion_factor
        )

        output = self.regressor(fused)

        if return_selection_info:
            if labels is not None and training and not self.no_selector:
                aux_losses = self._compute_all_aux_losses(
                    labels, modality_weights, selected_modalities, output
                )
                return output, primary_idx, modality_weights, selected_modalities, aux_losses
            return output, primary_idx, modality_weights, selected_modalities

        return output

    # ------------------------------------------------------------------
    # Auxiliary loss computation
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
