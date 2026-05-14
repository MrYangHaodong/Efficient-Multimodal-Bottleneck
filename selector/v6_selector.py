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

        self.attention_vectors = nn.ParameterDict({
            m: nn.Parameter(torch.randn(low_dim_dict.get(m, 256)) * 0.02)
            for m in modalities
        })

        self.projectors = nn.ModuleDict({
            m: nn.Linear(low_dim_dict.get(m, 256), uniform_dim)
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
            aggregated[m] = self._aggregate(low_dim_features[m], self.attention_vectors[m])
            projected[m] = self.projectors[m](aggregated[m])

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
        n_available = len(available_modalities)

        if top_k is not None and top_k < n_available:
            available_indices = [self.modalities.index(m) for m in available_modalities]
            available_weights = weights[:, available_indices]
            available_confs = concat_confidence[:, available_indices]

            conf_weighted_scores = (available_weights * available_confs).mean(dim=0)
            k = max(1, min(top_k, n_available))

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
