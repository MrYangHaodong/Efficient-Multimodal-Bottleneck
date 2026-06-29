"""MMEE — Multimodal Anytime Early-Exit baseline (DEPTH-axis early exit).

Faithful port of the early-exit MECHANISM from Jordy-VL/multi-modal-early-exit
(ICDAR'24, "Multimodal Adaptive Inference with Anytime Early Exiting") — multiple
off-ramp exit heads on intermediate transformer layers + weighted multi-exit
training (DeeBERT / PABEE / FrameExit lineage) + confidence-threshold anytime exit
— but on a generic multimodal time-series backbone (concat per-modality tokens ->
fusion transformer), so it runs on IEMOCAP / daliahar / pamap2 / dsads / eav / mosei.

Complementary to seqA: seqA exits in the MODALITY axis; MMEE exits in the DEPTH axis.

forward(x) with x=(B,T,sum(variates)) [float, or long-castable sax]:
  returns dict(exit_logits=[E tensors (B,C)], exit_layers=[ints]).
exit_inference(x, threshold): per-sample anytime exit at the first head whose
  max-softmax >= threshold; returns (logits (B,C), exit_idx (B,) layer index used).
"""
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.loader_utils import _build_channel_map


class MMEEClassifier(nn.Module):
    def __init__(self, cfg, input_length: int, *, input_mode: str = 'float',
                 alphabet_size: int = 20, token_dim: int = 16,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 6,
                 exit_layers=None, dropout: float = 0.1):
        super().__init__()
        assert input_mode in ('float', 'sax')
        self.modalities: List[str] = list(cfg.modalities)
        self.variates: Dict[str, int] = dict(cfg.variates)
        self.M = len(self.modalities)
        self.channel_map = _build_channel_map(self.modalities, self.variates)
        self.input_mode = input_mode
        self.token_dim = token_dim
        self.d_model = d_model
        self.num_layers = num_layers
        # exit after these (1-indexed) layers; always include the final layer.
        if exit_layers is None:
            exit_layers = list(range(1, num_layers + 1))
        self.exit_layers = sorted(set(int(e) for e in exit_layers) | {num_layers})

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
        self.modality_embedding = nn.Parameter(torch.zeros(1, self.M, 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2,
                                       dropout=dropout, batch_first=True, activation='gelu')
            for _ in range(num_layers)])
        # one off-ramp head per exit layer (operates on the CLS token)
        self.exit_heads = nn.ModuleDict({
            str(e): nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, cfg.num_classes))
            for e in self.exit_layers})
        for p in (self.pos_embedding, self.modality_embedding, self.cls):
            nn.init.trunc_normal_(p, std=0.02)

    def _embed(self, x):
        """x:(B,T,sumV) -> token sequence (B, 1+M*T, d) with CLS + modality+pos embeds."""
        toks = []
        for j, m in enumerate(self.modalities):
            xm = x[:, :, self.channel_map[m]]
            if self.input_mode == 'sax':
                e = self.token_embedding(xm.long())                         # (B,T,V,tok)
                B, T, V, _ = e.shape
                xm = e.reshape(B, T, V * self.token_dim)
            B, T = xm.shape[0], xm.shape[1]
            h = self.input_projections[m](xm) + self.pos_embedding[:, :T, :]
            h = h + self.modality_embedding[:, j]                           # (B,T,d)
            toks.append(h)
        seq = torch.cat(toks, dim=1)                                        # (B, M*T, d)
        cls = self.cls.expand(seq.shape[0], -1, -1)
        return torch.cat([cls, seq], dim=1)                                # (B, 1+M*T, d)

    def forward(self, x):
        h = self._embed(x)
        exit_logits, used = [], []
        for li, layer in enumerate(self.layers, start=1):
            h = layer(h)
            if li in self.exit_layers:
                exit_logits.append(self.exit_heads[str(li)](h[:, 0]))       # CLS token
                used.append(li)
        return {'exit_logits': exit_logits, 'exit_layers': used}

    @torch.no_grad()
    def exit_inference(self, x, threshold: float):
        """Per-sample anytime exit at first head with max-softmax >= threshold."""
        h = self._embed(x)
        B = h.shape[0]
        done = torch.zeros(B, dtype=torch.bool, device=h.device)
        out_logits = None
        exit_idx = torch.full((B,), self.exit_layers[-1], dtype=torch.long, device=h.device)
        for li, layer in enumerate(self.layers, start=1):
            h = layer(h)
            if li in self.exit_layers:
                logit = self.exit_heads[str(li)](h[:, 0])
                if out_logits is None:
                    out_logits = logit.clone()
                conf = F.softmax(logit, -1).max(-1).values
                take = (~done) & ((conf >= threshold) | (li == self.exit_layers[-1]))
                out_logits[take] = logit[take]
                exit_idx[take] = li
                done = done | take
                if done.all():
                    break
        return out_logits, exit_idx
