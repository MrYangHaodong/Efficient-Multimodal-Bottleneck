"""FAITHFUL original DynMM (Xue & Marculescu, MULA/CVPR'23) port for CMU-MOSEI, 3 nested branches.
Mirrors the official ModalityDynMM/affect code (MultiBench backbone), adapted to our Acc-2
classification pipeline:
  - per-modality encoder = MultiBench `Transformer`: Conv1d(1x1) proj + 5-layer nn.TransformerEncoder
    (nhead=5, default ffn=2048, seq-first), output = LAST timestep.
  - expert = per-modality Transformer encoders + `Concat` late fusion + MLP head.
  - gate = Transformer(409, 10) + Linear(10, n_branch), DiffSoftmax (straight-through hard/soft).
  - resource loss = sum_b gate_w_b * branch_FLOP_b (the only auxiliary loss; NO distillation).
Branches (dynmm_branches.json["mosei"]): 1mod=[text], partial=[text,vision], full=[vision,audio,text].
"""
from __future__ import annotations
import torch
import torch.nn as nn

ENC_DIM = {'vision': 60, 'audio': 120, 'text': 120}   # MultiBench DynMM-MOSEI per-modality embed dims


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


class OrigDynMMExpert(nn.Module):
    """One branch: per-modality MBTransformer encoders -> Concat -> MLP head (classification)."""
    def __init__(self, modalities, feat_dims, num_classes, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.modalities = list(modalities)
        self.encoders = nn.ModuleDict({m: MBTransformer(feat_dims[m], ENC_DIM[m]) for m in self.modalities})
        cat = sum(ENC_DIM[m] for m in self.modalities)
        self.head = nn.Sequential(nn.Linear(cat, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, feats):                   # feats: dict m -> (B, T, C_m)  (may contain extra modalities)
        z = torch.cat([self.encoders[m](feats[m]) for m in self.modalities], dim=-1)
        return self.head(z)


def diff_softmax(logits, tau: float = 1.0, hard: bool = False, dim: int = -1):
    """Official DynMM DiffSoftmax: soft by default; straight-through hard if hard=True."""
    y = (logits / tau).softmax(dim)
    if hard:
        idx = y.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, idx, 1.0)
        return y_hard - y.detach() + y
    return y


class OrigDynMM3Branch(nn.Module):
    """3 frozen experts + a per-sample gate (Transformer(409,10)+Linear) with FLOP-resource reg."""
    def __init__(self, experts, gate_in_dim: int, flop_vec, gate_dim: int = 10,
                 tau: float = 1.0, hard: bool = True):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = nn.Sequential(MBTransformer(gate_in_dim, gate_dim), nn.Linear(gate_dim, len(experts)))
        self.register_buffer('flop', torch.tensor(flop_vec, dtype=torch.float32))
        self.tau, self.hard = tau, hard

    def forward(self, feats, gate_in, additional_loss: bool = False):
        w = diff_softmax(self.gate(gate_in), tau=self.tau, hard=self.hard)        # (B, n_branch)
        preds = torch.stack([e(feats) for e in self.experts], dim=1)             # (B, n_branch, C)
        out = (w.unsqueeze(-1) * preds).sum(1)                                   # gate-weighted (soft/ST-hard) ensemble
        if additional_loss:
            return out, (w * self.flop).sum(1).mean()                            # expected FLOPs (resource reg)
        return out, w


# ============================================================================================
# GENERALIZED (5 datasets): concatenated-x input + SAX token-embedding + arbitrary modalities.
# Input x = (B, T, sum_variates) concatenated per cfg.modalities order (ee_data unpack); each
# branch splits x via channel_map and encodes its subset. Same MultiBench Transformer + Concat
# + MLP, NO distillation. Used by main_dynmm_orig_{expert,gate}_5ds.py.
# ============================================================================================
def _channel_map(modalities, variates):
    mp, s = {}, 0
    for m in modalities:
        mp[m] = list(range(s, s + variates[m])); s += variates[m]
    return mp


class OrigDynMMExpertG(nn.Module):
    """One branch over a modality SUBSET; takes concatenated x. float or SAX (token-embed)."""
    def __init__(self, sub_modalities, full_modalities, variates, num_classes, input_mode='float',
                 embed_dim=60, alphabet_size=20, token_dim=16, hidden=128, dropout=0.1):
        super().__init__()
        self.sub = list(sub_modalities); self.cmap = _channel_map(full_modalities, variates)
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
        self.mods = list(full_modalities); self.cmap = _channel_map(full_modalities, variates)
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
    def __init__(self, experts, gate, flop_vec, tau=1.0, hard=True):
        super().__init__()
        self.experts = nn.ModuleList(experts); self.gate = gate
        self.register_buffer('flop', torch.tensor(flop_vec, dtype=torch.float32))
        self.tau, self.hard = tau, hard

    def forward(self, x, additional_loss=False):
        w = diff_softmax(self.gate(x), tau=self.tau, hard=self.hard)
        preds = torch.stack([e(x) for e in self.experts], dim=1)
        out = (w.unsqueeze(-1) * preds).sum(1)
        if additional_loss:
            return out, (w * self.flop).sum(1).mean()
        return out, w
