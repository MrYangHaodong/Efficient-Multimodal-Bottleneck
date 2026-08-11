"""Branched sequential seqA — one fully-independent encoder+fusion branch per modality.

Unlike the batched ``DualVideoBottleneckModelV6Downsample`` in ``models/seqA.py``
(which stores M stacked weight sets in one tensor and runs all modalities through
a grouped-GEMM, encoding *all* modalities before any fusion), this model gives
each modality a **completely separate branch**:

    branch[m] = projector[m] + encoder[m] (per-modal transformer)
                             + fusion[m]  (bottleneck transformer)

and the forward **interleaves** them, one modality at a time:

    enc(order[0]) -> fuse(order[0]) -> [exit?]
    enc(order[1]) -> fuse(order[1]) -> [exit?]
    enc(order[2]) -> fuse(order[2]) -> [exit?]
    ...

The only thing shared across modalities is the **bottleneck** (the K-token
communication channel) plus the cross-modality aggregator, the final norm, and
the classifier head. Because a modality's encoder runs only when its step is
reached, an early exit at step k genuinely skips the encoder+fusion FLOPs of
every modality after k — which the batched model cannot do (it encodes all M up
front and only derives prefix readouts post-hoc).

Design notes
------------
* Each branch is a plain ``nn.Module`` with its own parameters — nothing is
  indexed by a modality axis. The state_dict keys are ``branches.<m>.*``.
* Fusion reads the running bottleneck, jointly attends over ``[bottleneck ; tokens]``,
  writes an updated bottleneck, and the parent folds it into the running state via
  ``seq_agg_mode`` (mean / gate / mlp), matching the sequential-seqA recurrence.
* ``forward(..., exit_fn=...)`` lets any policy (fixed-k prefix, entropy threshold,
  or an external RL policy) stop the loop; ``return_prefix=True`` returns the
  per-step readouts (for prefix supervision / analysis).
* Token length is preserved through fusion (no distill/downsample) to keep each
  branch simple and self-contained; add per-branch distill later if needed.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

# Reuse the proven building blocks from the batched model.
from models.seqA import TransformerBlock


class _DualPathDownsample(nn.Module):
    """Conv1d(k=3,s=2)+GELU  ⊕  MaxPool1d(k=2,s=2)  → sum → LayerNorm.

    ``[B, L, d] -> [B, L//2, d]`` (L truncated to even first). This is the exact
    standalone fusion-distill primitive from the batched ``DualVideoBottleneckModel
    V6Downsample`` (models/crossattn.py ``_DualPathDownsample`` / the batched
    ``BatchedTransformerBlock._batched_ds``), copied here so ``SeqABranched`` stays
    self-contained.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 == 1:                      # keep conv/pool lengths aligned
            x = x[:, :-1, :]
        xT = x.transpose(1, 2)                       # [B, d, L]
        out = self.act(self.conv(xT)) + self.pool(xT)
        return self.norm(out.transpose(1, 2))        # [B, L//2, d]


class ModalityBranch(nn.Module):
    """A single modality's complete, independent branch: projector + per-modal
    encoder + bottleneck-fusion transformer. Holds all its own parameters."""

    def __init__(self, feat_dim: int, d_model: int, nhead: int,
                 num_enc_layers: int, num_fusion_layers: int,
                 n_bottlenecks: int, mlp_ratio: float = 1.0, dropout: float = 0.1,
                 use_sparse_attn: bool = True, sparse_attn_variant: str = 'opt',
                 factor: int = 5, strat_block_size: int = 8, max_len: int = 512,
                 num_fusion_distill: int = 0):
        super().__init__()
        mlp_dim = int(d_model * mlp_ratio)

        # ---- input projection + per-modal preprocessing ----
        self.projector = nn.Linear(feat_dim, d_model)
        self.in_norm = nn.LayerNorm(d_model)

        # ---- per-modal encoder (dense self-attention over this modality only) ----
        self.encoder = nn.ModuleList([
            TransformerBlock(hidden_size=d_model, num_heads=nhead, mlp_dim=mlp_dim,
                             dropout_rate=dropout, use_sparse_attn=False)
            for _ in range(num_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # ---- bottleneck-fusion transformer (attends over [bottleneck ; tokens]) ----
        # bottleneck_head=True marks the first K positions as bottleneck tokens for
        # the sparse-attention path (they always attend/are attended densely).
        self.fusion = nn.ModuleList([
            TransformerBlock(hidden_size=d_model, num_heads=nhead, mlp_dim=mlp_dim,
                             dropout_rate=dropout, use_sparse_attn=use_sparse_attn,
                             sparse_attn_variant=sparse_attn_variant, factor=factor,
                             n_bottleneck=n_bottlenecks, bottleneck_head=True,
                             strat_block_size=strat_block_size)
            for _ in range(num_fusion_layers)
        ])
        self.fusion_norm = nn.LayerNorm(d_model)

        # ---- fusion distillation (ported from the batched seqA) --------------
        # The LAST ``num_fusion_distill`` fusion layers halve the [bottleneck ; tokens]
        # sequence via a dual-path conv+pool, exactly like the batched
        # DualVideoBottleneckModelV6Downsample (n_fusion_distill = -1 -> all fusion
        # layers distill; 0 -> none, i.e. full-resolution / original behaviour). The
        # bottleneck (first ``k_half`` positions after a halving) is upsampled
        # k_half -> K so the running bottleneck keeps width K across layers & modalities.
        # NOTE vs the reference: the halving is applied AFTER each fusion block (between
        # blocks, Informer-style) rather than mid-block between attention and MLP — this
        # keeps the change local to the branch and gives the same progressive halving.
        nfd = (num_fusion_layers if num_fusion_distill < 0
               else min(num_fusion_distill, num_fusion_layers))
        self.num_fusion_distill = nfd
        self._distill_flags = [i >= (num_fusion_layers - nfd) for i in range(num_fusion_layers)]
        self.fusion_ds = nn.ModuleDict({
            str(i): _DualPathDownsample(d_model)
            for i, flag in enumerate(self._distill_flags) if flag
        })
        if nfd > 0:
            assert n_bottlenecks % 2 == 0, \
                "n_bottlenecks must be even for fusion distillation (K//2 split)"
            self.k_bottleneck = n_bottlenecks
            self.k_half = n_bottlenecks // 2
            self.bn_upsample = nn.Linear(self.k_half, n_bottlenecks)   # k_half -> K on seq dim
        else:
            self.bn_upsample = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, feat_dim] -> encoded tokens [B, T, d_model]. Runs ONLY this
        modality's encoder. No sequence-length interpolation: the raw T is kept
        (matching mbt_vanilla / SimpleMBTFusion, which feeds full-length inputs)."""
        x = self.projector(x)                                  # [B, T, D]
        x = self.in_norm(x)
        for blk in self.encoder:
            x = blk(x)
        return self.enc_norm(x)

    def fuse(self, tokens: torch.Tensor, bottleneck: torch.Tensor):
        """Fold this modality into the shared bottleneck.

        tokens:     [B, T, D]   this modality's encoded tokens
        bottleneck: [B, K, D]   running shared bottleneck (read)
        returns (updated_bottleneck [B, K, D], out_tokens [B, T', D]) where
        ``T' = T // 2**num_fusion_distill`` (T' == T when distillation is off).

        The bottleneck is prepended (first K positions). On a distilling fusion
        layer the whole [bottleneck ; tokens] sequence is halved; the leading
        ``k_half`` positions are taken as the (halved) bottleneck and upsampled back
        to width K, so the running bottleneck stays width K for the next layer /
        modality (mirrors the batched model's K -> K//2 -> K aggregator).
        """
        K = bottleneck.shape[1]
        bn, tok = bottleneck, tokens
        for i, blk in enumerate(self.fusion):
            h = torch.cat([bn, tok], dim=1)                    # [B, K + T_cur, D]
            h = blk(h)                                         # attention + MLP (length preserved)
            ds = self.fusion_ds[str(i)] if str(i) in self.fusion_ds else None
            if ds is not None:
                h = ds(h)                                      # halve -> [B, (K+T)//2, D]
                bn_piece = h[:, :self.k_half, :]               # first k_half = bottleneck
                tok = h[:, self.k_half:, :]                    # remainder = tokens (halved)
                bn = self.bn_upsample(bn_piece.transpose(1, 2)).transpose(1, 2)   # k_half -> K
            else:
                bn = h[:, :K, :]
                tok = h[:, K:, :]
        return self.fusion_norm(bn), self.fusion_norm(tok)


class SeqABranched(nn.Module):
    """Interleaved, per-modality-branched sequential seqA with early exit."""

    def __init__(self, modalities: List[str], feat_dims: Dict[str, int],
                 num_classes: int, d_model: int = 128, nhead: int = 8,
                 num_enc_layers: int = 3, num_fusion_layers: int = 3,
                 n_bottlenecks: int = 8, mlp_ratio: float = 1.0, dropout: float = 0.1,
                 use_sparse_attn: bool = True, sparse_attn_variant: str = 'opt',
                 factor: int = 5, strat_block_size: int = 8,
                 max_len: int = 512,
                 seq_agg_mode: str = 'mean', seq_random_order: bool = False,
                 head_hidden: Optional[int] = None,
                 num_fusion_distill: int = 0):
        super().__init__()
        assert seq_agg_mode in ('mean', 'gate', 'mlp'), seq_agg_mode
        self.modalities = list(modalities)
        self.num_modalities = len(self.modalities)
        self.d_model = d_model
        self.n_bottlenecks = n_bottlenecks
        self.seq_agg_mode = seq_agg_mode
        self.seq_random_order = seq_random_order
        self.num_fusion_distill = num_fusion_distill

        # ---- one completely independent branch per modality ----
        self.branches = nn.ModuleDict({
            m: ModalityBranch(
                feat_dim=feat_dims[m], d_model=d_model, nhead=nhead,
                num_enc_layers=num_enc_layers, num_fusion_layers=num_fusion_layers,
                n_bottlenecks=n_bottlenecks, mlp_ratio=mlp_ratio, dropout=dropout,
                use_sparse_attn=use_sparse_attn, sparse_attn_variant=sparse_attn_variant,
                factor=factor, strat_block_size=strat_block_size, max_len=max_len,
                num_fusion_distill=num_fusion_distill)
            for m in self.modalities
        })

        # ---- shared bottleneck bank (the cross-modality channel) ----
        self.bottleneck = nn.Parameter(torch.randn(1, n_bottlenecks, d_model) * 0.02)

        # ---- cross-modality bottleneck aggregation across steps ----
        if seq_agg_mode == 'gate':
            self.seq_gate = nn.Linear(2 * d_model, d_model)
            nn.init.zeros_(self.seq_gate.bias)
        elif seq_agg_mode == 'mlp':
            self.seq_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # ---- readout head ----
        self.final_norm = nn.LayerNorm(d_model)
        hh = head_hidden or d_model
        self.regressor = nn.Sequential(
            nn.Linear(d_model, hh), nn.GELU(), nn.Dropout(dropout), nn.Linear(hh, num_classes))

    # ------------------------------------------------------------------
    def _agg_bottleneck(self, step: int, running: torch.Tensor,
                        new_bn: torch.Tensor) -> torch.Tensor:
        """Fold a modality's updated bottleneck into the running shared state."""
        if step == 0:
            return new_bn
        if self.seq_agg_mode == 'mean':
            return (step * running + new_bn) / (step + 1)
        if self.seq_agg_mode == 'mlp':
            return self.seq_mlp(torch.cat([running, new_bn], dim=-1))
        # gate (GRU-style): z = sigmoid(W[running ; new]); (1-z)running + z new
        z = torch.sigmoid(self.seq_gate(torch.cat([running, new_bn], dim=-1)))
        return (1.0 - z) * running + z * new_bn

    def _readout(self, tok_sum: torch.Tensor, tok_count: int) -> torch.Tensor:
        """GAP over all tokens processed so far -> logits [B, C]."""
        rep = self.final_norm(tok_sum / max(tok_count, 1))
        return self.regressor(rep)

    def _resolve_order(self, active: List[str], device) -> List[str]:
        if self.seq_random_order and self.training:
            perm = torch.randperm(len(active), device=device).tolist()
            return [active[i] for i in perm]
        return active

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Drop dead ``branches.*.pos.*`` buffers left by checkpoints trained before
        the (never-applied) RoPE module was removed, so strict loading still works."""
        state_dict = {k: v for k, v in state_dict.items() if '.pos.' not in k}
        return super().load_state_dict(state_dict, *args, **kwargs)

    # ------------------------------------------------------------------
    def forward(self, inputs: Dict[str, torch.Tensor],
                order: Optional[List[str]] = None,
                exit_fn: Optional[Callable[[int, torch.Tensor, torch.Tensor], bool]] = None,
                return_prefix: bool = False):
        """Interleaved encode->fuse per modality, one branch at a time.

        Args:
            inputs:  {modality: [B, T, feat_dim]}. Only present modalities are used.
            order:   explicit processing order (list of modality names). Defaults to
                     ``self.modalities`` (or a random permutation if seq_random_order
                     and training).
            exit_fn: optional ``(step, logits_k, bottleneck) -> bool``. Return True to
                     STOP after step ``step`` — remaining modalities are never encoded
                     or fused (true early-exit FLOP savings). Applied at batch level.
            return_prefix: also return per-step logits and the count of modalities used.

        Returns:
            logits [B, C]                              (return_prefix=False)
            (logits, prefix_logits [B, S, C], n_used)  (return_prefix=True)
        """
        active = [m for m in (order or self.modalities) if m in inputs]
        if not active:
            raise ValueError('no active modalities present in inputs')
        any_x = inputs[active[0]]
        B, device = any_x.shape[0], any_x.device
        active = self._resolve_order(active, device)

        bottleneck = self.bottleneck.expand(B, -1, -1).contiguous()   # [B, K, D]
        tok_sum = torch.zeros(B, self.d_model, device=device, dtype=self.bottleneck.dtype)
        tok_count = 0
        prefix_logits: List[torch.Tensor] = []
        used_order: List[str] = []

        for step, m in enumerate(active):
            branch = self.branches[m]
            tokens = branch.encode(inputs[m])                                  # ENCODER (this mod only)
            new_bn, out_tok = branch.fuse(tokens, bottleneck)                  # FUSION  (this mod only)
            bottleneck = self._agg_bottleneck(step, bottleneck, new_bn)

            # accumulate GAP statistics so the readout uses all tokens seen so far
            tok_sum = tok_sum + out_tok.sum(dim=1)
            tok_count += out_tok.shape[1]

            logits_k = self._readout(tok_sum, tok_count)
            prefix_logits.append(logits_k)
            used_order.append(m)

            if exit_fn is not None and bool(exit_fn(step, logits_k, bottleneck)):
                break

        self._last_used_order = used_order          # introspection
        final = prefix_logits[-1]
        if return_prefix:
            return final, torch.stack(prefix_logits, dim=1), len(prefix_logits)
        return final


def build_seqa_branched(modalities, feat_dims, num_classes, **kw) -> SeqABranched:
    """Convenience constructor mirroring the seqA arg names."""
    return SeqABranched(modalities, feat_dims, num_classes, **kw)
