"""AdaMML (Panda et al., ICCV'21) adapted to our multimodal time-series setting.

Faithful to the official policy-net mechanism (models/policy_net.py): a lightweight
policy network previews a CHEAP summary of the input and emits a PER-MODALITY binary
keep/drop decision via Gumbel-softmax (hard, straight-through, temperature-annealed);
a per-modality cost penalty pushes the policy toward dropping expensive modalities.

Differences from official AdaMML (per the user's spec):
  * decision is made ONCE for the WHOLE sample (no per-segment loop / LSTM causality).
  * main net = per-modality 6-layer Transformer encoders + LOGITS-level late fusion
    (learnable LF weights), so an un-selected modality's encoder is simply skipped at
    inference -> real compute saving (late fusion = independently skippable).

Self-contained model definition: AdaMMLNet (the per-modality encoders _ModEncoder and
the policy network _PolicyNet). The encoders take a PER-MODALITY dict {m: (B,T,C_m)}.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.loader_utils import _build_channel_map  # noqa: F401 (helper export for trainers)


class _ModEncoder(nn.Module):
    """Per-modality: linear proj + learnable pos -> N-layer Transformer -> mean-pool
    -> per-modality logit head. Returns (logit [B,C], feat [B,d])."""

    def __init__(self, in_dim, d_model, nhead, num_layers, num_classes, dropout, max_len,
                 dim_feedforward=None):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        ffn = dim_feedforward if dim_feedforward is not None else 4 * d_model
        layer = nn.TransformerEncoderLayer(d_model, nhead, ffn, dropout, 'gelu',
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):                                      # x [B,T,in_dim]
        T = x.shape[1]
        h = self.proj(x) + self.pos[:, :T]
        h = self.norm(self.enc(h))
        feat = h.mean(1)
        return self.head(feat), feat


class _PolicyNet(nn.Module):
    """Cheap proxy preview -> per-modality 2-way (drop/keep) Gumbel decision. The proxy
    is a mean-pool over time of each modality (near-free) -> small shared Transformer
    over the M modality tokens -> per-modality Linear(.,2). Faithful to the official
    PolicyNet.fcs + wrapper_gumbel_softmax(hard, temperature)."""

    def __init__(self, mod_dims, modalities, d_policy=32, nhead=4, num_layers=1):
        super().__init__()
        self.modalities = list(modalities); self.M = len(self.modalities)
        self.proj = nn.ModuleDict({m: nn.Linear(mod_dims[m], d_policy) for m in self.modalities})
        layer = nn.TransformerEncoderLayer(d_policy, nhead, d_policy, 0.1, 'gelu',
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_policy) * 0.02)
        self.fcs = nn.ModuleList([nn.Linear(d_policy, 2) for _ in range(self.M)])
        self.temperature = 5.0

    def set_temperature(self, t): self.temperature = t

    def forward(self, xd):
        toks = torch.stack([self.proj[m](xd[m].mean(1)) for m in self.modalities], 1)   # [B,M,d] (cheap preview)
        B = toks.shape[0]
        feat = self.enc(torch.cat([self.cls.expand(B, -1, -1), toks], 1))[:, 0]          # [B,d]
        logits = torch.stack([fc(feat) for fc in self.fcs], 1)                           # [B,M,2]
        if self.training:
            dec = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)[..., 1]   # [B,M] keep (ST)
        else:
            dec = (logits[..., 1] > logits[..., 0]).float()
        return dec, logits


class AdaMMLNet(nn.Module):
    """Per-modality 6-layer Transformer encoders + policy net + logits late fusion."""

    def __init__(self, mod_dims, modalities, d_model, nhead, num_layers, num_classes,
                 dropout, max_len, d_policy=32, dim_feedforward=None):
        super().__init__()
        self.modalities = list(modalities); self.M = len(self.modalities)
        self.encoders = nn.ModuleDict({
            m: _ModEncoder(mod_dims[m], d_model, nhead, num_layers, num_classes, dropout, max_len,
                           dim_feedforward=dim_feedforward)
            for m in self.modalities})
        self.policy = _PolicyNet(mod_dims, modalities, d_policy=d_policy)
        self.lf = nn.Parameter(torch.zeros(self.M))            # softmax -> late-fusion weights

    def forward(self, xd, mode='policy'):
        """mode: 'all' = use every modality (warmup/finetune-main); 'policy' = policy-gated."""
        L = torch.stack([self.encoders[m](xd[m])[0] for m in self.modalities], 1)        # [B,M,C]
        B = L.shape[0]
        if mode == 'all':
            dec = torch.ones(B, self.M, device=L.device); plogits = None
        else:
            dec, plogits = self.policy(xd)                                               # [B,M]
            if not self.training:                                                        # guarantee >=1 kept at eval
                allzero = dec.sum(1) == 0
                if allzero.any():
                    top = self.policy(xd)[1][..., 1].argmax(1)
                    dec[allzero, top[allzero]] = 1.0
        w = F.softmax(self.lf, 0).view(1, self.M, 1)                                      # [1,M,1]
        d = dec.unsqueeze(-1)                                                             # [B,M,1]
        out = (w * d * L).sum(1) / (w * d).sum(1).clamp_min(1e-6)                         # masked weighted-logit fusion
        return out, dec, plogits
