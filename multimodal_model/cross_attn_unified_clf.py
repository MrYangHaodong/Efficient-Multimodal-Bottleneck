"""CrossAttn with UNIFIED V6 pre-fusion.

Goal: an apples-to-apples CrossAttn-vs-V6 comparison where ONLY the fusion differs.
This model inherits the ENTIRE pre-fusion stack from V6
(DualVideoBottleneckModelV6Downsample):
    input_projectors (Linear per modality)
    → F.interpolate to input_length (no-op when input_length == data length)
    → layer_norms
    → temporal_pos_encoder (RoPE)
    → batched_modality_encoder (BatchedTransformerBlock × 2, dense attn)
    → output_norms

and replaces ONLY the fusion:
    V6:        bottleneck_fusion (SimpleMBTFusionAdaptiveMLPDownsampleBmm) + regressor
    this CA:   concat per-modality tokens → InformerEncoderLayerWithMoE × N
               (ProbSparse + sparse-MoE, with Conv1d downsample) → mean-pool → classifier

So pre-fusion is byte-identical to V6 (inherited `_encode_modalities`); the only
architectural difference is the fusion block.
"""
import torch
import torch.nn as nn

from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)
from multimodal_model.cross_attn_v6_clf import InformerEncoderLayerWithMoE


class CrossAttnUnifiedClf(DualVideoBottleneckModelV6Downsample):
    def __init__(self, *args, ca_d_ff_mult: int = 1, ca_num_layers: int = 4,
                 ca_num_experts: int = 4, ca_nhead: int = 8, **kwargs):
        # capture for our own fusion before building V6 (which ignores these)
        self._ca_num_classes = kwargs.get('num_classes', kwargs.get('output_dim', 5))
        # capture PE flags BEFORE super().__init__ (which would consume them for the
        # now-discarded bottleneck_fusion — we re-implement them at the informer level).
        self._ca_add_pos_embeds = kwargs.get('fusion_add_pos_embeds', False)
        self._ca_pos_embed_mode = kwargs.get('fusion_pos_embed_mode', 'full')
        self._ca_pos_embeds_max_len = kwargs.get('fusion_pos_embeds_max_len', 256)
        super().__init__(*args, **kwargs)

        d = self.pooled_dim
        # Drop V6's fusion + regressor (we keep their pre-fusion only).
        self.bottleneck_fusion = None
        self.regressor = None

        # ---- CrossAttn fusion (concat → informer MoE × N → pool → classifier) ----
        self.informer_encoder = nn.ModuleList([
            InformerEncoderLayerWithMoE(
                d_model=d, n_heads=ca_nhead, d_ff=d * ca_d_ff_mult,
                dropout=0.1, factor=self.factor, num_experts=ca_num_experts, k=1)
            for _ in range(ca_num_layers)
        ])
        self.ca_classifier = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d, self._ca_num_classes),
        )

        # ---- PE injection at CrossAttn fusion input (mirrors V6's options) ----
        if self._ca_add_pos_embeds:
            assert self._ca_pos_embed_mode in ('full', 'decoupled', 'id_only'), \
                f"unknown pos_embed_mode {self._ca_pos_embed_mode!r}"
            M = len(self.modalities)
            if self._ca_pos_embed_mode == 'full':
                self.ca_pos_embeds = nn.ParameterDict({
                    m: nn.Parameter(torch.randn(1, self._ca_pos_embeds_max_len, d) * 0.02)
                    for m in self.modalities
                })
            elif self._ca_pos_embed_mode == 'decoupled':
                self.ca_temporal_pe = nn.Parameter(
                    torch.randn(1, self._ca_pos_embeds_max_len, d) * 0.02)
                self.ca_mod_id_pe = nn.Embedding(M, d)
                nn.init.normal_(self.ca_mod_id_pe.weight, mean=0.0, std=0.02)
            else:  # 'id_only'
                self.ca_mod_id_pe = nn.Embedding(M, d)
                nn.init.normal_(self.ca_mod_id_pe.weight, mean=0.0, std=0.02)
        self._ca_mod_to_idx = {m: i for i, m in enumerate(self.modalities)}

    def forward(self, high_dim_inputs, low_dim_inputs, training=True,
                factor=None, return_selection_info=True, labels=None,
                override_selected_modalities=None):
        base_factor = factor if factor is not None else self.factor
        encoder_video_dict = (low_dim_inputs if self.encoder_video_source == 'low'
                              else high_dim_inputs)

        # ---- Per-modality input projection (identical to V6) ----
        projected_features = {}
        for modality in self.modalities:
            if modality == 'video':
                if 'video' in encoder_video_dict:
                    projected_features['video'] = self.input_projectors['video'](
                        encoder_video_dict['video'])
            else:
                if modality in high_dim_inputs:
                    projected_features[modality] = self.input_projectors[modality](
                        high_dim_inputs[modality])

        # ---- V6 pre-fusion (interpolate + LN + RoPE + batched encoder + out-norm) ----
        processed = self._encode_modalities(projected_features, None, base_factor)

        # ---- CrossAttn fusion: concat per-modality tokens along time ----
        active = [m for m in self.modalities if m in processed]
        # Inject per-modality PE (ViTaPEs-style) at fusion input before concat.
        if getattr(self, '_ca_add_pos_embeds', False):
            for m in active:
                x_m = processed[m]
                L = x_m.shape[1]
                if self._ca_pos_embed_mode == 'full':
                    x_m = x_m + self.ca_pos_embeds[m][:, :L]
                elif self._ca_pos_embed_mode == 'decoupled':
                    m_idx = self._ca_mod_to_idx[m]
                    mid = self.ca_mod_id_pe.weight[m_idx].view(1, 1, -1)
                    x_m = x_m + self.ca_temporal_pe[:, :L] + mid
                else:  # 'id_only'
                    m_idx = self._ca_mod_to_idx[m]
                    mid = self.ca_mod_id_pe.weight[m_idx].view(1, 1, -1)
                    x_m = x_m + mid
                processed[m] = x_m
        x_cat = torch.cat([processed[m] for m in active], dim=1)   # (B, sum_T, D)
        for layer in self.informer_encoder:
            x_cat = layer(x_cat, base_factor)
        pooled = x_cat.mean(dim=1)
        output = self.ca_classifier(pooled)

        # ---- return in V6's tuple format (so V6 training wrapper works) ----
        B = output.shape[0]
        device = output.device
        modality_weights = torch.full(
            (B, self.num_modalities), 1.0 / self.num_modalities, device=device)
        primary_idx = torch.zeros(B, dtype=torch.long, device=device)
        if return_selection_info:
            if labels is not None and training and not self.no_selector:
                return output, primary_idx, modality_weights, None, {}
            return output, primary_idx, modality_weights, None
        return output
