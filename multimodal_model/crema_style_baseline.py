"""CREMA-style baseline (concept port of Yui010206/CREMA, ICLR'25 — "Generalizable
and Efficient Video-Language Reasoning via Multimodal Modular Fusion").

The ORIGINAL CREMA is a frozen BLIP-2 video-LLM (per-modality Q-Formers + BEATs /
3D encoders + an LLM decoder for video QA) and is NOT portable to SAX-HAR / IEMOCAP
time-series classification. This is a faithful port of its three CONCEPTUAL ideas
onto a generic multimodal backbone, clearly labelled CREMA-STYLE (not official CREMA):

  1. Multimodal MODULAR fusion : a SHARED encoder backbone + a per-modality bottleneck
     ADAPTER (parameter-efficient, like CREMA's per-modality modules), and a small
     learned QUERY that pools each modality to a summary token (a Q-Former-lite MMQA).
  2. Modality-SEQUENTIAL training : modalities are added one at a time; every prefix
     (after k modalities) is supervised (mean CE over prefixes).
  3. Modality-ADAPTIVE early exit : after fusing k modalities, an exit head predicts;
     at inference the model exits once confident -> MODALITY-axis early exit.

forward(x, order) -> dict(prefix_logits=[M tensors (B,C)], order=[...]).
exit_inference(x, threshold, order) -> (logits (B,C), n_mods_used (B,)).
"""
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_model.shaspec_baseline import _build_channel_map


class _Adapter(nn.Module):
    """Bottleneck residual adapter d -> r -> d."""
    def __init__(self, d, r, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, r), nn.GELU(), nn.Dropout(dropout), nn.Linear(r, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x + self.net(x))


class CREMAStyleClassifier(nn.Module):
    def __init__(self, cfg, input_length: int, *, input_mode: str = 'float',
                 alphabet_size: int = 20, token_dim: int = 16,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 3,
                 adapter_dim: int = 32, num_query: int = 4, dropout: float = 0.1):
        super().__init__()
        assert input_mode in ('float', 'sax')
        self.modalities: List[str] = list(cfg.modalities)
        self.variates: Dict[str, int] = dict(cfg.variates)
        self.M = len(self.modalities)
        self.channel_map = _build_channel_map(self.modalities, self.variates)
        self.input_mode = input_mode
        self.token_dim = token_dim
        self.d_model = d_model

        if input_mode == 'sax':
            self.token_embedding = nn.Embedding(alphabet_size + 1, token_dim, padding_idx=0)
            in_dim = lambda m: self.variates[m] * token_dim                 # noqa: E731
        else:
            self.token_embedding = None
            in_dim = lambda m: self.variates[m]                            # noqa: E731
        self.input_projections = nn.ModuleDict({
            m: nn.Linear(in_dim(m), d_model) for m in self.modalities})
        # +64 margin: SAX tokenization can yield T = ceil(dur/word_len) (off-by-one vs input_length).
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_length + 64, d_model))

        # SHARED backbone (weights shared across modalities = modular fusion)
        self.shared_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2,
                                       dropout=dropout, batch_first=True, activation='gelu'),
            num_layers=num_layers)
        # per-modality bottleneck adapters (the modular, param-efficient part)
        self.adapters = nn.ModuleDict({m: _Adapter(d_model, adapter_dim, dropout) for m in self.modalities})
        # Q-Former-lite: learned queries pool each modality to a summary token
        self.num_query = num_query
        self.query = nn.Parameter(torch.zeros(1, num_query, d_model))
        self.query_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.modality_embedding = nn.Parameter(torch.zeros(1, self.M, d_model))

        # sequential fusion over accumulated modality-summary tokens
        self.fusion = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2,
                                                 dropout=dropout, batch_first=True, activation='gelu')
        # per-prefix (modality-count) exit heads: 1..M modalities fused
        self.exit_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, cfg.num_classes))
            for _ in range(self.M)])
        for p in (self.pos_embedding, self.query, self.modality_embedding):
            nn.init.trunc_normal_(p, std=0.02)

    def _summary(self, x, j, m):
        """Embed modality m (index j) -> shared enc -> adapter -> query-pool -> (B, d)."""
        xm = x[:, :, self.channel_map[m]]
        if self.input_mode == 'sax':
            e = self.token_embedding(xm.long())
            B, T, V, _ = e.shape
            xm = e.reshape(B, T, V * self.token_dim)
        B, T = xm.shape[0], xm.shape[1]
        h = self.input_projections[m](xm) + self.pos_embedding[:, :T, :]
        h = self.shared_enc(h)                                    # shared backbone
        h = self.adapters[m](h)                                  # per-modality adapter
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.query_attn(q, h, h)                     # (B, num_query, d)
        return pooled.mean(dim=1) + self.modality_embedding[:, j]   # (B, d)

    def forward(self, x, order: Optional[List[int]] = None):
        if order is None:
            order = list(range(self.M))
        acc = []                       # accumulated modality summary tokens
        prefix_logits = []
        for k, j in enumerate(order):
            acc.append(self._summary(x, j, self.modalities[j]))
            stack = torch.stack(acc, dim=1)                      # (B, k+1, d)
            fused = self.fusion(stack).mean(dim=1)              # fuse accumulated -> (B,d)
            prefix_logits.append(self.exit_heads[k](fused))
        return {'prefix_logits': prefix_logits, 'order': order}

    @torch.no_grad()
    def exit_inference(self, x, threshold: float, order: Optional[List[int]] = None):
        if order is None:
            order = list(range(self.M))
        B = x.shape[0]
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        out = None
        n_used = torch.full((B,), self.M, dtype=torch.long, device=x.device)
        acc = []
        for k, j in enumerate(order):
            acc.append(self._summary(x, j, self.modalities[j]))
            fused = self.fusion(torch.stack(acc, dim=1)).mean(dim=1)
            logit = self.exit_heads[k](fused)
            if out is None:
                out = logit.clone()
            conf = F.softmax(logit, -1).max(-1).values
            take = (~done) & ((conf >= threshold) | (k == len(order) - 1))
            out[take] = logit[take]
            n_used[take] = k + 1
            done = done | take
            if done.all():
                break
        return out, n_used
