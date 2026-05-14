"""V6 wrapper that replaces REINFORCE with Group Relative Policy Optimization
(GRPO, à la DeepSeek-R1) for the modality selector.

Compared to vanilla REINFORCE:
  - REINFORCE uses an EMA reward baseline (state-independent).
  - GRPO samples G actions for the same state, normalises rewards within
    the group, and uses the standardised value as advantage.

This implementation adds the three DeepSeek-R1 ingredients that vanilla
policy gradient lacks:

  1. **PPO-style importance ratio + clipping**.  log π_new is computed
     under the live selector params (`modality_weights_0`, with grad).
     log π_old is the detached snapshot from the same forward.  In
     single-update RL (one optimiser step per batch) the ratio is
     identically 1 and contributes no gradient — exactly the REINFORCE
     limit.  Keeping the ratio in place makes it easy to switch to
     multi-step GRPO updates by re-running the loss block against a
     stored snapshot.

  2. **KL penalty against a reference policy**.  The reference is an EMA
     of the per-batch mean modality_weights, refreshed each step under
     `no_grad`.  Penalises drift from a slow-moving anchor (DDPG-target
     style).  This is the term that turns vanilla policy gradient into a
     trust-region method.

  3. **Entropy bonus**.  H(π) = -Σ π log π averaged over the batch.
     Added to the loss with a negative sign so the optimiser maximises
     entropy and the selector keeps exploring (does not collapse to a
     deterministic top-K early).

The IMS top-K is deterministic; we sample group members externally via
Gumbel-top-K and inject the chosen subset into V6 forward via
`override_selected_modalities`.

Sketch (per training batch):
  0. Apply uniform modality dropout (shared across the G group).
  1. g=0: full V6 forward with grad.  Captures modality_weights_0 (the
     policy distribution over M modalities) and the selector's natural
     top-K.  Compute reward_0 = -CE(output_0, y).
  2. g=1..G-1: sample top-K_g via Gumbel-top-K from
     modality_weights_0.mean(0).  Run V6 forward under no_grad with
     override_selected_modalities=top-K_g and compute reward_g.
  3. advantages = (rewards - mean_g) / (std_g + eps), detached.
  4. log_pi_new[g, b] = Σ_{m ∈ sel_g} log π_new(m | b)    (with grad)
     log_pi_old        = log_pi_new.detach()
     ratio             = exp(log_pi_new - log_pi_old)
     clipped_ratio     = ratio.clamp(1-ε, 1+ε)
     pg_term           = min(adv * ratio, adv * clipped_ratio)
     grpo_loss         = -mean(pg_term)
  5. kl_loss   = mean( KL(π_new || π_ref_ema) )
     ent_loss  = -mean( H(π_new) )
  6. total = λ_grpo · grpo_loss + λ_kl · kl_loss + λ_entropy · ent_loss

The encoder/fusion/regressor only see gradient from g=0's forward + the
main task loss.  The selector params see gradient from log_pi_new of all
G samples plus the KL anchor and entropy bonus.
"""
from __future__ import annotations
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['V6GRPOWrapper']


class V6GRPOWrapper(nn.Module):
    """Wraps a V6 model with GRPO-style policy gradient for the IMS selector.

    Args:
        v6_model: the inner DualVideoBottleneckModelV6Downsample (must have
                  a non-None modality_selector and no_selector=False).
        modalities: list of modality name strings, in the canonical order.
        variates: dict mapping modality name -> input channel count.
        n_grpo_samples: G, number of action samples per batch (>=2 for
                        GRPO; 1 falls back to standard non-GRPO training).
        lambda_grpo: scalar weight on the GRPO policy-gradient term.
        lambda_kl: scalar weight on KL(π_new || π_ref_ema).  Set 0 to
                   disable.
        lambda_entropy: scalar weight on the entropy bonus (negative-sign
                        added to loss).  Set 0 to disable.
        kl_ema_alpha: EMA decay for the reference policy (closer to 1 =
                      slower-moving anchor).  0.99 is a sensible default.
        clip_eps: PPO-style ratio clip.  0.2 = DeepSeek-R1 default.  Set 0
                  to disable clipping (pure policy gradient).
        grpo_eps: numerical floor for the within-group reward std.
    """

    def __init__(self, v6_model, modalities, variates,
                 n_grpo_samples: int = 2,
                 lambda_grpo: float = 0.1,
                 lambda_kl: float = 0.02,
                 lambda_entropy: float = 0.01,
                 kl_ema_alpha: float = 0.99,
                 clip_eps: float = 0.2,
                 grpo_eps: float = 1e-8):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self.variates = variates
        self.n_grpo_samples = int(n_grpo_samples)
        self.lambda_grpo = float(lambda_grpo)
        self.lambda_kl = float(lambda_kl)
        self.lambda_entropy = float(lambda_entropy)
        self.kl_ema_alpha = float(kl_ema_alpha)
        self.clip_eps = float(clip_eps)
        self.grpo_eps = float(grpo_eps)

        # EMA reference policy: shape [M], lazily initialised on first
        # forward so the device matches the model's device.
        self.register_buffer(
            '_ref_weights_ema',
            torch.full((len(self.modalities),), 1.0 / max(1, len(self.modalities))),
            persistent=False,
        )
        self._ref_initialised = False

        self._mod_slices: Dict[str, tuple] = {}
        offset = 0
        for m in self.modalities:
            n_ch = variates[m]
            self._mod_slices[m] = (offset, offset + n_ch)
            offset += n_ch
        self._total_channels = offset

    # ---------------------------------------------------------------- helpers
    def _split_to_dicts(self, x):
        high_dim, low_dim = {}, {}
        for m in self.modalities:
            s, e = self._mod_slices[m]
            high_dim[m] = x[:, :, s:e]
            low_dim[m] = x[:, :, s:e]
        return high_dim, low_dim

    def _gumbel_topk_sample(self, modality_weights: torch.Tensor, k: int) -> List[str]:
        """Sample a length-K subset of modality names via Gumbel-top-K
        from the per-batch averaged modality_weights distribution.

        log_scores are divided by ``self.gumbel_temperature`` (default 1)
        before adding Gumbel noise — τ > 1 flattens the distribution so
        Gumbel noise can actually flip rankings.  Without this, a
        confident selector (sigmoid pushed to extremes) locks the group
        into one identical K-subset and GRPO gradients collapse to zero.
        """
        scores = modality_weights.detach().mean(dim=0)  # [M]
        log_scores = (scores + 1e-12).log()
        gumbel = -torch.log(-torch.log(torch.rand_like(log_scores) + 1e-9) + 1e-9)
        tau = max(getattr(self, 'gumbel_temperature', 1.0), 1e-6)
        perturbed = log_scores / tau + gumbel
        topk_idx = perturbed.topk(min(k, len(self.modalities))).indices.tolist()
        return [self.modalities[i] for i in topk_idx]

    def _selected_log_prob(self, modality_weights: torch.Tensor,
                           selected_modalities: List[str]) -> Optional[torch.Tensor]:
        """sum_{m in selected} log P(m | sample) under modality_weights.
        Returns [B] tensor (per-sample log-prob) or None if list empty."""
        if not selected_modalities:
            return None
        idx = [self.modalities.index(m) for m in selected_modalities
               if m in self.modalities]
        if not idx:
            return None
        log_w = torch.log(modality_weights[:, idx] + 1e-8)
        return log_w.sum(dim=1)  # [B]

    def _apply_modality_dropout(self, x: torch.Tensor, drop_prob: float):
        """Uniform-prob per-modality dropout (identical to V6DSADSWrapper)."""
        device = x.device
        modality_mask = torch.ones(len(self.modalities), device=device)
        if drop_prob > 0:
            x = x.clone()
            for idx, m in enumerate(self.modalities):
                if torch.rand(1).item() < drop_prob:
                    s, e = self._mod_slices[m]
                    x[:, :, s:e] = 0.0
                    modality_mask[idx] = 0.0
        return x, modality_mask

    def _update_reference_ema(self, modality_weights_0: torch.Tensor) -> None:
        """Update the EMA reference policy from the current batch.  No grad."""
        with torch.no_grad():
            batch_mean = modality_weights_0.detach().mean(dim=0)  # [M]
            if not self._ref_initialised:
                self._ref_weights_ema.copy_(batch_mean)
                self._ref_initialised = True
            else:
                self._ref_weights_ema.mul_(self.kl_ema_alpha).add_(
                    batch_mean, alpha=1.0 - self.kl_ema_alpha)

    # ---------------------------------------------------------------- forward
    def forward(self, x, modality_drop_prob: float = 0.0,
                training: bool = False, labels=None):
        # Apply modality dropout (shared across the G group when training).
        if training and modality_drop_prob > 0:
            x, modality_mask = self._apply_modality_dropout(x, modality_drop_prob)
        else:
            modality_mask = torch.ones(len(self.modalities), device=x.device)

        high_dim, low_dim = self._split_to_dicts(x)

        has_selector = (not self.model.no_selector
                        and self.model.modality_selector is not None)

        # Eval / no-GRPO / no-selector / no-labels path: standard single forward
        if (not training) or (self.n_grpo_samples <= 1) or (labels is None) \
                or (not has_selector):
            if has_selector:
                result = self.model(high_dim, low_dim, training=training,
                                    return_selection_info=True, labels=labels)
                if len(result) == 5:
                    output, _, _, _, aux_losses = result
                    return output, modality_mask, aux_losses
                output = result[0]
                return output, modality_mask
            output = self.model(high_dim, low_dim, training=training,
                                return_selection_info=False)
            return output, modality_mask

        # ---------------- GRPO path: G forwards ----------------
        # g=0: with grad, use selector's natural top-K
        result_0 = self.model(high_dim, low_dim, training=training,
                              return_selection_info=True, labels=labels)
        output_0, _, modality_weights_0, selected_0, aux_losses = result_0

        if selected_0 is None:
            # Selector did not produce a top-K (e.g., warmup epoch). Fall back.
            return output_0, modality_mask, aux_losses

        with torch.no_grad():
            ce_0 = F.cross_entropy(output_0.detach(), labels, reduction='none')
        reward_0 = -ce_0  # [B]
        log_p_0 = self._selected_log_prob(modality_weights_0, selected_0)
        if log_p_0 is None:
            return output_0, modality_mask, aux_losses

        rewards = [reward_0]
        log_probs_new = [log_p_0]

        K = len(selected_0)

        for g in range(1, self.n_grpo_samples):
            sel_g = self._gumbel_topk_sample(modality_weights_0, K)
            # Avoid the trivial case sel_g == sel_0 by perturbing once more.
            if K < len(self.modalities) and set(sel_g) == set(selected_0):
                sel_g = self._gumbel_topk_sample(modality_weights_0, K)

            with torch.no_grad():
                result_g = self.model(high_dim, low_dim, training=training,
                                      return_selection_info=True, labels=labels,
                                      override_selected_modalities=sel_g)
                output_g = result_g[0] if isinstance(result_g, tuple) else result_g
                ce_g = F.cross_entropy(output_g, labels, reduction='none')
            rewards.append(-ce_g)
            log_p_g = self._selected_log_prob(modality_weights_0, sel_g)
            log_probs_new.append(log_p_g if log_p_g is not None
                                 else torch.zeros_like(log_p_0))

        # Group-relative advantage: standardise rewards within the G samples
        rewards_stack = torch.stack(rewards, dim=0)            # [G, B]
        log_p_new_stack = torch.stack(log_probs_new, dim=0)    # [G, B]
        log_p_old_stack = log_p_new_stack.detach()             # snapshot

        adv_mean = rewards_stack.mean(dim=0, keepdim=True)
        adv_std = rewards_stack.std(dim=0, keepdim=True) + self.grpo_eps
        advantages = (rewards_stack - adv_mean) / adv_std      # [G, B]
        advantages = advantages.detach()

        # Policy-gradient loss.
        #
        # In single-step RL (one optimiser step per sampled batch) the PPO
        # ratio `exp(log_p_new - log_p_old.detach())` is identically 1 in
        # VALUE, so `-(advantage * ratio).mean()` collapses to 0 even
        # though the gradient is non-zero.  That makes monitoring nearly
        # useless ("GRPO=0" looks like the loss is broken when it isn't).
        #
        # We use the equivalent REINFORCE form
        #     loss = -mean( advantage * log π_new(a|s) )
        # which has the same gradient w.r.t. log_p_new but a non-zero
        # value, so the meter reflects what's actually being optimised.
        # The PPO ratio is only needed for multi-step updates with a
        # frozen old policy, which we don't do here.
        if self.clip_eps > 0:
            # Keep optional clipping for stability — but on log-prob
            # difference rather than the exponential ratio.
            log_ratio = log_p_new_stack - log_p_old_stack
            clipped_log_ratio = log_ratio.clamp(-self.clip_eps, self.clip_eps)
            pg_per_sample = torch.minimum(
                advantages * (log_p_new_stack),
                advantages * (log_p_old_stack + clipped_log_ratio),
            )
        else:
            pg_per_sample = advantages * log_p_new_stack
        grpo_loss = -pg_per_sample.mean()

        # ---------------- KL penalty against EMA reference ----------------
        # π_new = modality_weights_0 (with grad).  π_ref = EMA buffer
        # (detached).  Exact analytical KL since the distribution is a
        # small categorical (size M).
        self._update_reference_ema(modality_weights_0)
        ref = self._ref_weights_ema.detach().unsqueeze(0)            # [1, M]
        log_pi = torch.log(modality_weights_0 + 1e-8)                # [B, M]
        log_ref = torch.log(ref + 1e-8)                              # [1, M]
        kl_per_sample = (modality_weights_0 * (log_pi - log_ref)).sum(dim=1)
        kl_loss = kl_per_sample.mean()

        # ---------------- Entropy bonus ----------------
        # H(π) = -Σ π log π, mean over batch.  Add with negative sign so
        # the optimiser maximises entropy ⇒ keeps exploration alive.
        entropy_per_sample = -(modality_weights_0 * log_pi).sum(dim=1)
        entropy = entropy_per_sample.mean()
        entropy_loss = -entropy

        # ---------------- Bundle into aux_losses ----------------
        aux_losses.pop('reinforce_loss', None)
        aux_losses['grpo_loss'] = self.lambda_grpo * grpo_loss
        if self.lambda_kl > 0:
            aux_losses['grpo_kl'] = self.lambda_kl * kl_loss
        if self.lambda_entropy > 0:
            aux_losses['grpo_entropy'] = self.lambda_entropy * entropy_loss

        return output_0, modality_mask, aux_losses
