"""DyMo / DynMM on a cross-modal Transformer backbone (NO V6).

Architecture (closer to the official DyMo DynamicTransformer):
  * per-modality encoder = ``num_enc_layers``-deep Transformer (default 2),
    mean-pooled to ONE token per modality;
  * cross-modal fusion = ``num_fusion_layers``-deep Transformer (default 4)
    over ``[cls] + per-modality tokens`` (with modality embeddings), masked to
    the present subset; the ``cls`` token is classified -> fused logits.

Wrappers:
  * ``CrossModalDyMo``  -- mask-robust training + native greedy per-sample
    modality selection (the reward = max-softmax quality of the FUSED output,
    so each candidate RE-RUNS the fusion transformer: the "multiple inference"
    cost is intrinsic, as in the official DyMo).
  * ``CrossModalDynMM`` -- learned gate over expert branches (modality subsets);
    each branch re-runs the fusion transformer; resource penalty on branch cost.

Input handling matches the other baselines: a concatenated ``[B,T,sum(V)]``
tensor split by channel slices; ``input_mode`` float (IEMOCAP) or sax (HAR).
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import torch
import torch.nn as nn

from multimodal_model.multimodn_baseline import _TemporalEncoder, _build_channel_map

__all__ = ['CrossModalTransformer', 'CrossModalDyMo', 'CrossModalDynMM',
           'IEMOCAPDyMoWrapper', 'IEMOCAPDynMMWrapper']


def _concat_iemocap(feats, M):
    return torch.cat([feats[2 * j] for j in range(M)], dim=-1)


class CrossModalTransformer(nn.Module):
    """2-layer per-modality encoders + 4-layer cross-modal Transformer fusion."""

    def __init__(self, cfg, input_length: int, *,
                 input_mode: str = 'float', alphabet_size: int = 20, token_dim: int = 16,
                 d_model: int = 64, nhead: int = 8, num_enc_layers: int = 2,
                 num_fusion_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        assert input_mode in ('float', 'sax')
        self.modalities: List[str] = list(cfg.modalities)
        self.variates: Dict[str, int] = dict(cfg.variates)
        self.M = len(self.modalities)
        self.channel_map = _build_channel_map(self.modalities, self.variates)
        self.input_mode = input_mode
        self.token_dim = token_dim
        self.d_model = d_model
        self.num_classes = cfg.num_classes

        if input_mode == 'sax':
            self.token_embedding = nn.Embedding(alphabet_size + 1, token_dim, padding_idx=0)
            in_dim = lambda m: self.variates[m] * token_dim  # noqa: E731
        else:
            self.token_embedding = None
            in_dim = lambda m: self.variates[m]              # noqa: E731

        self.input_projections = nn.ModuleDict({
            m: nn.Linear(in_dim(m), d_model) for m in self.modalities
        })
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_length, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.encoders = nn.ModuleDict({
            m: _TemporalEncoder(d_model, nhead, num_enc_layers, dropout)
            for m in self.modalities
        })
        # cross-modal fusion transformer over [cls] + M modality tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.modality_emb = nn.Embedding(self.M, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=num_fusion_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.num_classes))

    def _mod_token(self, mod_input, m):
        if self.input_mode == 'sax':
            emb = self.token_embedding(mod_input.long())
            B, T, V_m, _ = emb.shape
            x = emb.reshape(B, T, V_m * self.token_dim)
            present = (mod_input != 0).any(dim=2).float().view(B, T, 1)
        else:
            B, T, V_m = mod_input.shape
            x = mod_input
            present = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
        x = self.input_projections[m](x) + self.pos_embedding[:, :T, :]
        x = x * present
        x = self.encoders[m](x)
        denom = present.sum(dim=1).clamp(min=1.0)
        return (x * present).sum(dim=1) / denom              # (B, d) one token

    def encode_tokens(self, x):
        """[B,T,sum(V)] -> per-modality tokens (B,M,d) with modality_emb added.
        Encode ONCE so multi-mask routing (DynMM) can reuse it across branches."""
        toks = [self._mod_token(x[:, :, self.channel_map[m]], m) + self.modality_emb.weight[i]
                for i, m in enumerate(self.modalities)]
        return torch.stack(toks, dim=1)                                     # (B,M,d)

    def fuse_feat(self, mod_tok, mask):
        """mod_tok (B,M,d), mask [B,M] bool -> (fused logits (B,C), cls feature (B,d)).
        cls feature = pooled fused representation fed to the classifier; the
        DyMo-proto projection/prototype head reads exactly this feature."""
        B = mod_tok.shape[0]; device = mod_tok.device
        cls = self.cls_token.expand(B, -1, -1)                             # (B,1,d)
        seq = torch.cat([cls, mod_tok], dim=1)                            # (B,1+M,d)
        kpm = torch.zeros(B, 1 + self.M, dtype=torch.bool, device=device)  # True => ignore
        kpm[:, 1:] = ~mask
        out = self.fusion(seq, src_key_padding_mask=kpm)
        cls_feat = self.norm(out[:, 0])                                    # (B,d)
        return self.classifier(cls_feat), cls_feat

    def fuse(self, mod_tok, mask):
        """mod_tok (B,M,d) from encode_tokens, mask [B,M] bool -> fused logits (B,C)."""
        return self.fuse_feat(mod_tok, mask)[0]

    def forward(self, x, mask):
        """``x``: [B,T,sum(V)].  ``mask``: [B,M] bool, True = modality present.
        Returns fused logits [B, C]. (eval-identical to the old single-pass forward.)"""
        return self.fuse(self.encode_tokens(x), mask)


def _quality(logits):
    return logits.softmax(dim=-1).max(dim=-1).values


class CrossModalDyMo(nn.Module):
    """Mask-robust training + native greedy per-sample modality selection."""

    def __init__(self, model: CrossModalTransformer):
        super().__init__()
        self.model = model
        self.M = model.M

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], self.M, dtype=torch.bool, device=x.device)
        return self.model(x, mask)

    def random_mask(self, B, device, drop_prob):
        mask = torch.rand(B, self.M, device=device) >= drop_prob
        empty = ~mask.any(dim=1)
        if empty.any():
            mask[empty, torch.randint(self.M, (int(empty.sum()),), device=device)] = True
        return mask

    def keep_mask(self, B, device, keep):
        keep = set(keep)
        row = torch.tensor([m in keep for m in self.model.modalities], device=device)
        return row.unsqueeze(0).expand(B, -1).contiguous()

    @torch.no_grad()
    def forward_select(self, x):
        """Greedy add-strongest-first; reward = max-softmax of FUSED output.
        Each candidate RE-RUNS the fusion transformer (multiple inference)."""
        B = x.shape[0]; device = x.device; M = self.M
        eye = torch.eye(M, dtype=torch.bool, device=device)
        # quality of each single modality alone
        q_each = torch.stack([_quality(self.model(x, eye[i].unsqueeze(0).expand(B, -1)))
                              for i in range(M)], dim=1)                    # [B,M]
        mask = torch.zeros(B, M, dtype=torch.bool, device=device)
        first = q_each.argmax(dim=1)
        mask[torch.arange(B, device=device), first] = True
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(M - 1):
            q_cur = _quality(self.model(x, mask))
            best_gain = torch.full((B,), -1e9, device=device)
            best_u = torch.full((B,), -1, dtype=torch.long, device=device)
            for u in range(M):
                addable = ~mask[:, u]
                cand = mask.clone(); cand[:, u] = True
                gain = _quality(self.model(x, cand)) - q_cur
                gain = torch.where(addable, gain, torch.full_like(gain, -1e9))
                better = gain > best_gain
                best_gain = torch.where(better, gain, best_gain)
                best_u = torch.where(better, torch.full_like(best_u, u), best_u)
            finished = finished | (best_gain <= 0)
            do = (~finished) & (best_u >= 0)
            if do.any():
                mask[do, best_u[do]] = True
            if finished.all():
                break
        present = [m for m in self.model.modalities]
        return self.model(x, mask), mask, present


class CrossModalDynMM(nn.Module):
    """Learned gate over expert branches (modality subsets), resource-penalized."""

    def __init__(self, model: CrossModalTransformer, branches=None, gate_hidden=32):
        super().__init__()
        self.model = model
        self.modalities = list(model.modalities)
        M = model.M
        if branches is None:
            branches = [[self.modalities[0]], list(self.modalities)]
        self.branches = [[m for m in self.modalities if m in set(b)] for b in branches]
        self.register_buffer('costs', torch.tensor(
            [len(b) / M for b in self.branches], dtype=torch.float32))
        self._masks = [torch.tensor([m in set(b) for m in self.modalities]) for b in self.branches]
        self.gate = nn.Sequential(nn.Linear(M, gate_hidden), nn.ReLU(),
                                  nn.Linear(gate_hidden, len(self.branches)))

    def _summary(self, x):
        # per-modality mean activation over time+feature -> [B, M]
        cols = [x[:, :, self.model.channel_map[m]].float().mean(dim=(1, 2)) for m in self.modalities]
        return torch.stack(cols, dim=1)

    def forward(self, x, temp=1.0, hard=False):
        B = x.shape[0]; device = x.device
        logits = self.gate(self._summary(x)) / temp
        w = logits.softmax(-1)
        if hard:
            idx = w.argmax(-1, keepdim=True)
            wh = torch.zeros_like(w).scatter_(-1, idx, 1.0)
            w = wh - w.detach() + w
        tokens = self.model.encode_tokens(x)        # encode ONCE, reuse across branches
        branch_logits = []
        for bm in self._masks:
            mask = bm.to(device).unsqueeze(0).expand(B, -1)
            branch_logits.append(self.model.fuse(tokens, mask))
        stack = torch.stack(branch_logits, dim=1)                          # [B,K,C]
        fused = (w.unsqueeze(-1) * stack).sum(dim=1)
        return {'fused': fused, 'weights': w}

    def resource_loss(self, weights):
        return (weights.mean(dim=0) * self.costs).sum()

    def expected_cost(self, weights):
        return float((weights.mean(dim=0) * self.costs).sum().item())


class IEMOCAPDyMoWrapper(nn.Module):
    """Concat the IEMOCAP feats-list -> [B,T,sum(V)] -> CrossModalDyMo."""

    def __init__(self, dymo: CrossModalDyMo, modalities):
        super().__init__()
        self.dymo = dymo
        self.M = len(modalities)

    def forward(self, feats, mask=None):
        return self.dymo(_concat_iemocap(feats, self.M), mask)

    def keep_mask(self, B, device, keep):
        return self.dymo.keep_mask(B, device, keep)

    def random_mask(self, B, device, drop_prob):
        return self.dymo.random_mask(B, device, drop_prob)

    @torch.no_grad()
    def forward_select(self, feats):
        return self.dymo.forward_select(_concat_iemocap(feats, self.M))


class IEMOCAPDynMMWrapper(nn.Module):
    """Concat the IEMOCAP feats-list -> [B,T,sum(V)] -> CrossModalDynMM."""

    def __init__(self, dynmm: CrossModalDynMM, modalities):
        super().__init__()
        self.dynmm = dynmm
        self.M = len(modalities)
        self.branches = dynmm.branches
        self.costs = dynmm.costs

    def forward(self, feats, temp=1.0, hard=False):
        return self.dynmm(_concat_iemocap(feats, self.M), temp=temp, hard=hard)

    def resource_loss(self, weights):
        return self.dynmm.resource_loss(weights)

    def expected_cost(self, weights):
        return self.dynmm.expected_cost(weights)
