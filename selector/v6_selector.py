"""Improved Modality Selector (IMS) — single-import standalone.

Extracted from `dual_video_bottleneck_model_topk_v6_standalone.py` so the
selector can be evolved / replaced independently of the V6 backbone.
``ImprovedModalitySelector`` is the only class you need to import — the
probe head is bundled as a nested class so the full select pipeline
(probe → confidence → MLP → top-K) is self-contained.

Public surface:
  - ``ImprovedModalitySelector`` — batch-level top-K selector that combines
                                   per-modality features, internal probe-head
                                   confidences, pairwise cosine agreements,
                                   and (optionally) a Holo-Confidence bias /
                                   pairwise interaction matrix.
  - ``CurriculumScheduler``      — K-annealing helper (initial_k → final_k
                                   over training). Kept separate because
                                   it has no learnable params and runs once
                                   per epoch.
  - ``ModalityProbeHead``        — back-compat alias for
                                   ``ImprovedModalitySelector._ProbeHead``.
                                   Old checkpoints reference state-dict
                                   keys like ``probe_heads.<m>.head.weight``;
                                   the alias keeps that exact path valid.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    """Pre-LN residual block with Swish/SiLU activation, as used in the
    1000-Layer Self-Supervised RL paper (Wang et al., NeurIPS 2025 best
    paper) — the combination they identify as the minimal stable recipe
    for scaling depth: LayerNorm → Linear → SiLU → Linear → residual."""

    def __init__(self, h: int):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(self.ln(x))))


class _DeepSelectorMLP(nn.Module):
    """Replacement for the default 3-layer selector MLP.  Uses ``depth``
    residual blocks (pre-LN + SiLU).  Scaling tip from the NeurIPS 2025
    best paper: depth > width — same accuracy at ~50× fewer params via
    depth on contrastive RL.  Default hidden size matches the legacy
    selector_mlp so existing checkpoints stay roughly comparable.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128,
                 depth: int = 16):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([_ResBlock(hidden_dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm_out(x)
        return self.proj_out(x)


__all__ = [
    'ImprovedModalitySelector',
    'CurriculumScheduler',
    'ModalityProbeHead',   # back-compat alias, see bottom of file
]


class ImprovedModalitySelector(nn.Module):
    """
    Improved Modality Selector — batch-level selection, per-sample weighting.

    Architecture:
        For each modality:
            [B, T, C] → attention aggregate → [B, C] → project → [B, 256]
                                                                → probe head → confidence scalar

        Selector input = [M*256 features] + [M confidences] + [M*(M-1)/2 agreements]
                        → MLP → [M logits] + interaction_bonus → softmax → weights [B, M]

        Batch selection: mean(confidence-weighted weights) across batch → top-K
    """

    # ------------------------------------------------------------------
    # Nested probe head
    # ------------------------------------------------------------------
    class _ProbeHead(nn.Module):
        """Lightweight per-modality classifier for confidence estimation.

        Internal to the selector — exposed as ``ImprovedModalitySelector._ProbeHead``
        for completeness, and aliased as ``ModalityProbeHead`` at module level
        so older state-dict keys (``probe_heads.<m>.head.weight``) still load.
        """
        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
            self.head = nn.Linear(input_dim, num_classes)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            logits = self.head(x)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)
            max_entropy = torch.log(torch.tensor(
                float(logits.shape[-1]), dtype=x.dtype, device=x.device
            ))
            confidence = 1.0 - entropy / max_entropy
            return confidence, logits

    def __init__(
        self,
        modalities: List[str],
        low_dim_dict: Dict[str, int],
        mlp_hidden_dim: int = 128,
        num_classes: int = 5,
        uniform_dim: int = 256,
        use_interaction_matrix: bool = True,
        use_holo_bias: bool = False,
        holo_scale: float = 1.0,
        holo_eps: float = 1e-3,
        selector_depth: int = 0,
        use_multi_pool: bool = False,
        use_tx_summarizer: bool = False,
        tx_d_model: int = 64,
        tx_nhead: int = 4,
        tx_layers: int = 1,
        tx_dim_ff: int = 128,
        tx_dropout: float = 0.1,
        tx_max_seq_len: int = 512,
    ):
        super().__init__()
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.low_dim_dict = low_dim_dict
        self.uniform_dim = uniform_dim
        # When False, skip the interaction_matrix term in forward (it's
        # still kept as a parameter so existing checkpoints load cleanly,
        # but stays at zero — autograd doesn't reach it).
        self.use_interaction_matrix = use_interaction_matrix
        # Holo-Confidence bias (PDF Eq. 5) computed from per-modality
        # confidences. Adds a closed-form, zero-param cross-modal bias on
        # top of selector_mlp logits. Anti-collapse by construction: a
        # modality more confident than its peers gets a larger holo boost.
        self.use_holo_bias = use_holo_bias
        self.holo_scale = float(holo_scale)
        self.holo_eps = float(holo_eps)
        # Multi-pool aggregation: concat [attn_pool, mean_pool, max_pool]
        # along the channel dim before per-modality projector. Triples
        # projector input width but keeps projector output at uniform_dim
        # so selector_mlp sees the same input shape. Helps when the
        # learned-attention pool alone misses local burst / peak signals
        # that drive per-sample Shapley heterogeneity.
        self.use_multi_pool = bool(use_multi_pool)
        # Transformer summarizer: per-modality embed → 1-layer (or N) self-
        # attention TransformerEncoder over tokens → then pool. Attacks the
        # 'input compression' bottleneck — lets tokens see each other before
        # we squash to a single per-modality vector. Independent of
        # ``use_multi_pool``; combined is the most expressive setup.
        self.use_tx_summarizer = bool(use_tx_summarizer)
        self.tx_d_model = int(tx_d_model)
        self.tx_max_seq_len = int(tx_max_seq_len)

        if self.use_tx_summarizer:
            # Per-modality embed V_m -> tx_d_model, position embed, and
            # 1-layer TransformerEncoder. Then attention_vectors live at
            # tx_d_model, projectors take tx_d_model input.
            self.tx_embeds = nn.ModuleDict({
                m: nn.Linear(low_dim_dict.get(m, 256), self.tx_d_model)
                for m in modalities
            })
            self.tx_pos_embeds = nn.ParameterDict({
                m: nn.Parameter(
                    torch.randn(self.tx_max_seq_len, self.tx_d_model) * 0.02)
                for m in modalities
            })
            self.tx_summarizers = nn.ModuleDict({
                m: nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.tx_d_model, nhead=tx_nhead,
                        dim_feedforward=tx_dim_ff, dropout=tx_dropout,
                        batch_first=True, activation='gelu'),
                    num_layers=tx_layers)
                for m in modalities
            })
            self.attention_vectors = nn.ParameterDict({
                m: nn.Parameter(torch.randn(self.tx_d_model) * 0.02)
                for m in modalities
            })
            proj_in_mult = 3 if self.use_multi_pool else 1
            self.projectors = nn.ModuleDict({
                m: nn.Linear(proj_in_mult * self.tx_d_model, uniform_dim)
                for m in modalities
            })
        else:
            self.attention_vectors = nn.ParameterDict({
                m: nn.Parameter(torch.randn(low_dim_dict.get(m, 256)) * 0.02)
                for m in modalities
            })
            proj_in_mult = 3 if self.use_multi_pool else 1
            self.projectors = nn.ModuleDict({
                m: nn.Linear(proj_in_mult * low_dim_dict.get(m, 256), uniform_dim)
                for m in modalities
            })

        self.probe_heads = nn.ModuleDict({
            m: self._ProbeHead(uniform_dim, num_classes)
            for m in modalities
        })

        # Agreement column was removed (see `_compute_pairwise_agreement`
        # commented out below), so selector_input is just
        # [M*uniform_dim features] + [M confidences].
        selector_input_dim = (
            self.num_modalities * uniform_dim
            + self.num_modalities
        )

        # Selector MLP: legacy 3-layer ReLU vs deep residual (NeurIPS 2025
        # best paper recipe — scaling depth with residual+LN+SiLU).  When
        # ``selector_depth > 0`` we replace the legacy MLP with a stack of
        # ``selector_depth`` residual blocks; the hidden width stays at
        # ``mlp_hidden_dim`` so total params scale linearly with depth.
        self.selector_depth = int(selector_depth)
        if self.selector_depth > 0:
            self.selector_mlp = _DeepSelectorMLP(
                in_dim=selector_input_dim,
                out_dim=self.num_modalities,
                hidden_dim=mlp_hidden_dim,
                depth=self.selector_depth,
            )
        else:
            self.selector_mlp = nn.Sequential(
                nn.Linear(selector_input_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, self.num_modalities),
            )

        self.interaction_matrix = nn.Parameter(
            torch.zeros(self.num_modalities, self.num_modalities)
        )

        # ------------- Exploration knobs (mutable per epoch) -----------
        # Default: deterministic top-K (the original behaviour).  Set
        # `selector.exploration_mode` to 'epsilon' or 'gumbel' to enable
        # stochastic sampling for REINFORCE-style training; the
        # training loop can anneal `exploration_eps` / `exploration_temp`
        # over epochs (high early, low late) without rebuilding the model.
        self.exploration_mode: str = 'none'   # 'none' | 'epsilon' | 'gumbel'
        self.exploration_eps: float = 0.0     # used when mode == 'epsilon'
        self.exploration_temp: float = 1.0    # used when mode == 'gumbel'

        # When True, top-K is computed PER-SAMPLE: each batch sample selects
        # its own K modalities via (weights * confs).topk(K, dim=1).  The
        # forward returns selected_modalities=None and exposes a binary mask
        # `self._selected_mask[B, M_total]` for downstream fusion masking.
        # When False (default), the original batch-aggregated top-K runs.
        self.per_sample_topk: bool = False
        self._selected_mask: Optional[torch.Tensor] = None

        self._probe_logits: Dict[str, torch.Tensor] = {}
        self._batch_selection_entropy: Optional[torch.Tensor] = None

    def _aggregate(self, H: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        d = H.shape[-1]
        scores = torch.matmul(H, W) / (d ** 0.5)
        attn = F.softmax(scores, dim=1)
        return torch.sum(attn.unsqueeze(-1) * H, dim=1)

    # def _compute_pairwise_agreement(
    #     self, projected: Dict[str, torch.Tensor], batch_size: int, device: torch.device
    # ) -> torch.Tensor:
    #     agreements = []
    #     for i, m_i in enumerate(self.modalities):
    #         for j, m_j in enumerate(self.modalities):
    #             if j <= i:
    #                 continue
    #             if m_i in projected and m_j in projected:
    #                 sim = F.cosine_similarity(projected[m_i], projected[m_j], dim=-1)
    #             else:
    #                 sim = torch.zeros(batch_size, device=device)
    #             agreements.append(sim)
    #     return torch.stack(agreements, dim=-1)

    def forward(
        self,
        low_dim_features: Dict[str, torch.Tensor],
        top_k: Optional[int] = None,
        training: bool = True,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[str]]]:
        available_modalities = list(low_dim_features.keys())
        batch_size = list(low_dim_features.values())[0].shape[0]
        device = list(low_dim_features.values())[0].device

        self._probe_logits = {}
        self._batch_selection_entropy = None

        if len(available_modalities) == 1:
            modality = available_modalities[0]
            weights = torch.zeros(batch_size, self.num_modalities, device=device)
            mod_idx = self.modalities.index(modality)
            weights[:, mod_idx] = 1.0
            primary_idx = torch.full((batch_size,), mod_idx, dtype=torch.long, device=device)
            return primary_idx, weights, None

        aggregated = {}
        projected = {}
        for m in available_modalities:
            H = low_dim_features[m]                        # [B, L, d_low]
            if self.use_tx_summarizer:
                # Embed V_m -> tx_d_model, add positional, run self-attn.
                # Tokens interact before we collapse to per-modality vector.
                H = self.tx_embeds[m](H)                    # [B, L, tx_d_model]
                L = H.shape[1]
                pe = self.tx_pos_embeds[m][:L].unsqueeze(0)  # [1, L, tx_d_model]
                H = H + pe
                H = self.tx_summarizers[m](H)               # [B, L, tx_d_model]
            v_attn = self._aggregate(H, self.attention_vectors[m])   # [B, d_sum]
            if self.use_multi_pool:
                v_mean = H.mean(dim=1)
                v_max  = H.max(dim=1).values
                agg = torch.cat([v_attn, v_mean, v_max], dim=-1)
            else:
                agg = v_attn
            aggregated[m] = agg
            projected[m] = self.projectors[m](agg)
        # Expose projected features so outer wrappers (e.g. the C3 conformal
        # quantile head) can re-use them without duplicating the per-modality
        # projector forward.
        self._projected = projected

        confidences = {}
        self._probe_logits = {}
        for m in available_modalities:
            conf, logits = self.probe_heads[m](projected[m])
            confidences[m] = conf.squeeze(-1)
            self._probe_logits[m] = logits

        #agreement = self._compute_pairwise_agreement(projected, batch_size, device)

        feature_list = []
        confidence_list = []
        for m in self.modalities:
            if m in projected:
                feature_list.append(projected[m])
                confidence_list.append(confidences[m])
            else:
                feature_list.append(torch.zeros(batch_size, self.uniform_dim, device=device))
                confidence_list.append(torch.zeros(batch_size, device=device))

        concat_features = torch.cat(feature_list, dim=1)
        concat_confidence = torch.stack(confidence_list, dim=1)
        selector_input = torch.cat([concat_features, concat_confidence], dim=1)   #FIXME agreement?

        logits = self.selector_mlp(selector_input)

        # Test-time calibration hook: external code can set
        # ``self._tt_logit_bias`` to a [B, M] tensor (or [M]) which gets
        # added to selector logits before sigmoid + top-K. Unused if attr
        # is missing or None.
        tt_bias = getattr(self, '_tt_logit_bias', None)
        if tt_bias is not None:
            logits = logits + tt_bias.to(device=logits.device, dtype=logits.dtype)

        avail_mask = torch.zeros(batch_size, self.num_modalities, device=device)
        for m in available_modalities:
            avail_mask[:, self.modalities.index(m)] = 1.0

        if self.use_interaction_matrix:
            W_sym = (self.interaction_matrix + self.interaction_matrix.T) / 2
            interaction_bonus = torch.matmul(avail_mask, W_sym)
            logits = logits + interaction_bonus * avail_mask
        # else: skip — selector_mlp output is the only logit source.

        if self.use_holo_bias:
            # Holo-Confidence (PDF Eq. 5): closed-form pairwise modality bias
            # computed from existing per-modality confidences (no extra params).
            # Treat unavailable modalities as fully confident (c=1) so they
            # don't contribute to the joint log-product (log 1 = 0).
            c_for_holo = torch.where(
                avail_mask.bool(),
                concat_confidence,
                torch.ones_like(concat_confidence),
            )
            log_c = torch.log(c_for_holo.clamp(self.holo_eps,
                                               1.0 - self.holo_eps))    # [B, M]
            S = log_c.sum(dim=1, keepdim=True) - 1e-12                    # [B, 1] (negative)
            holo = (S - log_c) / S                                        # [B, M] in (0, 1)
            holo = holo * avail_mask                                      # zero out unavailable
            logits = logits + self.holo_scale * holo

        unavail_mask = (avail_mask == 0)
        logits = logits.masked_fill(unavail_mask, float('-inf'))

        # Expose raw selector logits for KL-style distillation losses that
        # need pre-sigmoid logits (e.g., softmax KD vs ``softmax(phi/tau)``).
        # Per-sample top-K mode in particular requires a normalised ranking
        # signal — sigmoid alone can saturate all gates simultaneously.
        self._raw_logits = logits

        weights = torch.sigmoid(logits)     # per-modality independent gating
        primary_idx = torch.argmax(weights, dim=1)

        # Anti-collapse: compute batch-level selection-rate entropy ALWAYS,
        # not only when top_k is set.
        #
        # With sigmoid weights each `weights[b, m]` is an INDEPENDENT
        # per-modality gating probability — `weights.sum(dim=1)` is NOT
        # 1, so the previous code (treating row-mean as a probability
        # distribution) was incorrect.  We instead:
        #   1. Compute each modality's batch-averaged gating value
        #        p_m = mean_b(weights[b, m])   ∈ [0, 1]
        #      — i.e. "how often does the selector turn modality m on?"
        #   2. Normalise across modalities so it acts as a categorical
        #      distribution of "which modality dominates the batch":
        #        q_m = p_m / Σ p_m
        #   3. Take Shannon entropy of q.  Entropy is max when q is
        #      uniform (selector uses all modalities equally) and 0 when
        #      a single modality dominates — same anti-collapse semantics
        #      as the original softmax version, so the downstream
        #      `diversity_loss = F.relu(1.0 - entropy)` still works.
        avail_idx_list = [self.modalities.index(m) for m in available_modalities]
        p_per_mod = weights[:, avail_idx_list].mean(dim=0)              # [n_avail], in [0,1]
        q = p_per_mod / (p_per_mod.sum() + 1e-8)                        # normalise to a distribution
        self._batch_selection_entropy = -(
            q * (q + 1e-8).log()
        ).sum()

        selected_modalities = None
        self._selected_mask = None
        n_available = len(available_modalities)

        if top_k is not None and top_k < n_available:
            available_indices = [self.modalities.index(m) for m in available_modalities]
            available_weights = weights[:, available_indices]
            available_confs = concat_confidence[:, available_indices]
            k = max(1, min(top_k, n_available))

            if self.per_sample_topk:
                # ---- Per-sample top-K branch ----
                # Each sample picks its own K modalities. Output a binary
                # mask [B, M_total]; selected_modalities stays None so the
                # outer V6 forward processes all M modalities and routes
                # the mask into the fusion module.
                per_sample_scores = available_weights * available_confs   # [B, n_avail]
                topk_positions = per_sample_scores.topk(k, dim=1).indices  # [B, k]
                selected_mask = torch.zeros(batch_size, self.num_modalities,
                                             device=device, dtype=torch.bool)
                avail_idx_tensor = torch.tensor(available_indices, device=device,
                                                 dtype=torch.long)
                canonical_topk = avail_idx_tensor[topk_positions]          # [B, k]
                selected_mask.scatter_(1, canonical_topk, True)
                self._selected_mask = selected_mask
                # Return selected_modalities=None so V6 doesn't filter
                # encoders by a batch-level subset; the mask drives fusion.
                return primary_idx, weights, None

            # ---- Original batch-aggregated top-K ----
            conf_weighted_scores = (available_weights * available_confs).mean(dim=0)

            # Dispatch on exploration mode.  Only takes effect during
            # training; eval always uses the deterministic argmax-top-K so
            # the test-set numbers are reproducible.
            mode = self.exploration_mode if training else 'none'

            if mode == 'epsilon' and float(torch.rand(1).item()) < self.exploration_eps:
                # ε-greedy: with probability ε draw a uniformly-random
                # K-subset of available modalities, otherwise greedy top-K.
                perm = torch.randperm(n_available, device=weights.device)
                topk_positions = perm[:k].tolist()
            elif mode == 'gumbel':
                # Gumbel-top-K — equivalent to sampling K items without
                # replacement from softmax(scores/τ).  τ→0 reproduces
                # greedy; τ large mixes uniformly.
                tau = max(self.exploration_temp, 1e-6)
                log_scores = (conf_weighted_scores + 1e-12).log() / tau
                noise = torch.rand_like(log_scores)
                gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
                perturbed = log_scores + gumbel
                topk_positions = perturbed.topk(k).indices.tolist()
            else:
                # Greedy / 'none' / eval path.
                topk_positions = conf_weighted_scores.topk(k).indices.tolist()

            selected_modalities = [available_modalities[pos] for pos in topk_positions]
            # Note: previously a separate `score_dist` entropy was overwritten
            # here. We now keep the cleaner weight-distribution entropy from
            # above so diversity_loss is consistent across all epochs.

        return primary_idx, weights, selected_modalities

    def compute_auxiliary_losses(
        self,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        if self._probe_logits:
            probe_losses = []
            for m, logits in self._probe_logits.items():
                probe_losses.append(F.cross_entropy(logits, labels))
            losses['probe_loss'] = torch.stack(probe_losses).mean()

        if self._batch_selection_entropy is not None:
            losses['diversity_loss'] = F.relu(
                1.0 - self._batch_selection_entropy
            )

        return losses


class CurriculumScheduler:
    """Anneal temperature and budget over training."""

    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = 10,
        initial_k: Optional[int] = None,
        final_k: int = 2,
        num_modalities: int = 5,
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.initial_k = initial_k or num_modalities
        self.final_k = final_k

    def step(self, epoch: int) -> Optional[int]:
        if epoch < self.warmup_epochs:
            return None

        progress = (epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        progress = min(progress, 1.0)

        k_range = self.initial_k - self.final_k
        current_k = self.initial_k - int(progress * k_range)
        return max(current_k, self.final_k)


# Back-compat alias — the probe head is now nested inside
# ImprovedModalitySelector but old code (and any future external users
# who only want the probe head) can still reference it by the original
# name.  State-dict keys remain `probe_heads.<m>.head.weight`.
ModalityProbeHead = ImprovedModalitySelector._ProbeHead
