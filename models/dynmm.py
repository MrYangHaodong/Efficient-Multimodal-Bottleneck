"""Faithful original DynMM (Xue & Marculescu, "Dynamic Multimodal Fusion", CVPR'23 /
MultiBench affect code) — generalized to a concatenated-x input over arbitrary modalities.

Three nested modality-subset experts ("branches": 1mod / partial / full) plus a per-sample
DiffSoftmax gate trained with a FLOP-resource regularizer (the ONLY auxiliary loss; NO
distillation). Mirrors the official ModalityDynMM/affect backbone:
  - per-modality encoder = MultiBench `Transformer`: Conv1d(1x1) projection + 5-layer
    nn.TransformerEncoder (nhead=5, default ffn=2048, seq-first), output = LAST timestep.
  - expert (`OrigDynMMExpertG`) = per-modality Transformer encoders + `Concat` late fusion + MLP.
  - gate (`OrigGateG`) = MBTransformer(., gate_dim) + Linear(gate_dim, n_branch), DiffSoftmax
    (straight-through hard / soft).
  - resource loss = sum_b gate_w_b * branch_FLOP_b.

Input x = (B, T, sum_variates) concatenated per cfg.modalities order (ee_data unpack); each
branch splits x via the channel map and encodes its own modality subset. `float` or `sax`
(token-embedded) inputs are both supported. Self-contained; used by training/dynmm.py.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from common.loader_utils import _build_channel_map


class MBTransformer(nn.Module):
    """MultiBench `Transformer`: 1x1 conv projection + 5-layer TransformerEncoder, last-token pooled."""
    def __init__(self, n_features: int, dim: int, nhead: int = 5, num_layers: int = 5):
        super().__init__()
        assert dim % nhead == 0, f'dim {dim} not divisible by nhead {nhead}'
        self.conv = nn.Conv1d(n_features, dim, kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead)   # ffn=2048, relu, NOT batch_first (defaults)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):                       # x: (B, T, n_features)
        x = self.conv(x.permute(0, 2, 1))       # (B, dim, T)
        x = x.permute(2, 0, 1)                  # (T, B, dim)  seq-first
        return self.transformer(x)[-1]          # (B, dim)  last timestep


def diff_softmax(logits, tau: float = 1.0, hard: bool = False, dim: int = -1):
    """Official DynMM DiffSoftmax: soft by default; straight-through hard if hard=True."""
    y = (logits / tau).softmax(dim)
    if hard:
        idx = y.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, idx, 1.0)
        return y_hard - y.detach() + y
    return y


# ============================================================================================
# GENERALIZED (6 datasets): concatenated-x input + optional SAX token-embedding + arbitrary
# modalities. Same MultiBench Transformer + Concat + MLP, NO distillation.
# ============================================================================================
class OrigDynMMExpertG(nn.Module):
    """One branch over a modality SUBSET; takes concatenated x. float or SAX (token-embed)."""
    def __init__(self, sub_modalities, full_modalities, variates, num_classes, input_mode='float',
                 embed_dim=60, alphabet_size=20, token_dim=16, hidden=128, dropout=0.1):
        super().__init__()
        self.sub = list(sub_modalities); self.cmap = _build_channel_map(full_modalities, variates)
        self.input_mode = input_mode; self.token_dim = token_dim
        if input_mode == 'sax':
            self.token_embedding = nn.Embedding(alphabet_size + 1, token_dim, padding_idx=0)
            in_dim = lambda m: variates[m] * token_dim                          # noqa: E731
        else:
            self.token_embedding = None
            in_dim = lambda m: variates[m]                                      # noqa: E731
        self.encoders = nn.ModuleDict({m: MBTransformer(in_dim(m), embed_dim) for m in self.sub})
        self.head = nn.Sequential(nn.Linear(embed_dim * len(self.sub), hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def _enc(self, x, m):
        xm = x[:, :, self.cmap[m]]
        if self.input_mode == 'sax':
            e = self.token_embedding(xm.long()); B, T, V, _ = e.shape; xm = e.reshape(B, T, V * self.token_dim)
        return self.encoders[m](xm)

    def forward(self, x):                                                       # x: (B,T,sum_var)
        return self.head(torch.cat([self._enc(x, m) for m in self.sub], dim=-1))


class OrigGateG(nn.Module):
    """Gate: (token-embed if SAX) full concat -> MBTransformer(.,gate_dim) -> Linear(.,n_branch)."""
    def __init__(self, full_modalities, variates, n_branch, input_mode='float',
                 gate_dim=10, alphabet_size=20, token_dim=16):
        super().__init__()
        self.mods = list(full_modalities); self.cmap = _build_channel_map(full_modalities, variates)
        self.input_mode = input_mode; self.token_dim = token_dim
        if input_mode == 'sax':
            self.token_embedding = nn.Embedding(alphabet_size + 1, token_dim, padding_idx=0)
            gin = sum(variates[m] * token_dim for m in full_modalities)
        else:
            self.token_embedding = None
            gin = sum(variates[m] for m in full_modalities)
        self.tr = MBTransformer(gin, gate_dim); self.fc = nn.Linear(gate_dim, n_branch)

    def forward(self, x):
        if self.input_mode == 'sax':
            parts = []
            for m in self.mods:
                xm = x[:, :, self.cmap[m]]; e = self.token_embedding(xm.long())
                B, T, V, _ = e.shape; parts.append(e.reshape(B, T, V * self.token_dim))
            x = torch.cat(parts, dim=-1)
        return self.fc(self.tr(x))


class OrigDynMM3BranchG(nn.Module):
    """3 frozen experts + a per-sample gate with FLOP-resource reg (NO distillation)."""
    def __init__(self, experts, gate, flop_vec, tau=1.0, hard=True):
        super().__init__()
        self.experts = nn.ModuleList(experts); self.gate = gate
        self.register_buffer('flop', torch.tensor(flop_vec, dtype=torch.float32))
        self.tau, self.hard = tau, hard

    def forward(self, x, additional_loss=False):
        w = diff_softmax(self.gate(x), tau=self.tau, hard=self.hard)            # (B, n_branch)
        preds = torch.stack([e(x) for e in self.experts], dim=1)               # (B, n_branch, C)
        out = (w.unsqueeze(-1) * preds).sum(1)                                  # gate-weighted (soft/ST-hard) ensemble
        if additional_loss:
            return out, (w * self.flop).sum(1).mean()                          # expected FLOPs (resource reg)
        return out, w
