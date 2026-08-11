"""Vanilla (non-adaptive) MBT classifier — a plain Multimodal Bottleneck Transformer.

VENDORED into the energy_adaptive bundle.  Byte-identical to
``Efficient-Multimodal-Bottleneck/models/mbt_vanilla.py`` except that
``SimpleMBTFusion`` is imported from the BUNDLED ``models/bottleneck_fusion.py``
(a verbatim copy of ``MAESTRO/models/bottleneck_fusion.py``) instead of via a
sys.path hack into the sibling MAESTRO repo.  No repo imports at run time.

Wraps the genuine shared-bottleneck-token fusion `SimpleMBTFusion` with a
per-modality input split (from the concatenated (B, T, sum_variates) tensor the
ee_data harness produces) and a linear classifier head.  NO early-exit / gating /
threshold / modality selection — a single forward pass always using ALL
modalities.  This is the non-adaptive counterpart of the seqA bottleneck fusion,
at the same structure params (d_model / num_layers / nhead / fusion_layer /
n_bottlenecks).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from common.loader_utils import _build_channel_map
from models.bottleneck_fusion import SimpleMBTFusion


class VanillaMBTClf(nn.Module):
    """Plain MBT classifier over a concatenated (B, T, sum_variates) input.

    Splits the concat tensor per modality via the channel map, feeds a
    {modality: (B, T, dim)} dict to SimpleMBTFusion (bottleneck-token fusion,
    always all modalities), then classifies the pooled fused vector.
    """
    def __init__(self, cfg, num_classes, d_model=128, num_layers=6, num_heads=8,
                 mlp_dim=256, fusion_layer=3, n_bottlenecks=4, dropout=0.1):
        super().__init__()
        self.modalities = list(cfg.modalities)
        self.variates = dict(cfg.variates)
        self.cmap = _build_channel_map(self.modalities, self.variates)
        self.fusion = SimpleMBTFusion(
            input_dims={m: self.variates[m] for m in self.modalities},
            hidden_size=d_model, num_layers=num_layers, num_heads=num_heads,
            mlp_dim=mlp_dim, fusion_layer=fusion_layer, use_bottleneck=True,
            n_bottlenecks=n_bottlenecks, dropout_rate=dropout, output_dim=d_model,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):                                   # x: (B, T, sum_variates)
        inp = {m: x[:, :, self.cmap[m]] for m in self.modalities}
        fused = self.fusion(inp)                            # (B, d_model)
        return self.classifier(fused)                       # (B, num_classes)
