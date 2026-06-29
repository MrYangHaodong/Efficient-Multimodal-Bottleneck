"""MMExit (Hou et al., Euro-Par 2023) faithful reimplementation as a baseline,
on a backbone shared with our other methods so the comparison isolates the EXIT
STAGE, not backbone capacity.

Design (matches MMExit's "n encoder exits + 1 fusion exit"):
  * per-modality encoder = ``num_layers``-deep Transformer (default 2), mean-pooled
    to ONE token per modality -- SAME encoder the other baselines use (no distill);
  * ENCODER-STAGE early exits: after the running prefix of k modalities (in a
    static Utility-of-Exit order at eval, randomized at train), a CHEAP shared exit
    head reads the masked running-MEAN of the k pooled tokens -> logits + entropy;
  * FUSION exit (the strong final one): concat all modality tokens + a
    ``num_fusion_layers``-deep self-attention Transformer over [cls]+tokens -> logits.

Contrast with V6-seqA: MMExit's early exits are cheap pooled-feature heads, whereas
seqA fuses (gated bottleneck) at EVERY step. So at low K, seqA reads a fused
representation while MMExit reads encoder outputs -- the experiment is whether
fusion-stage exit beats encoder-stage exit at low compute.

Training: multi-exit loss (mean CE over all encoder exits + the fusion exit), with
the same dyna modality-dropout curriculum + order randomization as the other models.
Input handling mirrors the other baselines (sax for HAR, float for IEMOCAP).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_model.multimodn_baseline import _TemporalEncoder, _build_channel_map

__all__ = ['GenericMMExitClassifier', 'mmexit_losses', 'IEMOCAPMMExitWrapper']


class GenericMMExitClassifier(nn.Module):
    def __init__(self, cfg, input_length: int, *,
                 input_mode: str = 'float', alphabet_size: int = 20, token_dim: int = 16,
                 d_model: int = 64, nhead: int = 8, num_layers: int = 2,
                 num_fusion_layers: int = 2, dropout: float = 0.1,
                 fusion_input: str = 'pool', fusion_distill: bool = False):
        super().__init__()
        assert input_mode in ('float', 'sax')
        assert fusion_input in ('pool', 'token')
        self.modalities: List[str] = list(cfg.modalities)
        self.variates: Dict[str, int] = dict(cfg.variates)
        self.M = len(self.modalities)
        self.num_classes = cfg.num_classes
        self.channel_map = _build_channel_map(self.modalities, self.variates)
        self.input_mode = input_mode
        self.token_dim = token_dim
        self.d_model = d_model
        # 'pool'  = MMExit-faithful: mean-pool each modality to ONE token, fuse M tokens.
        # 'token' = fuse the RAW per-modality encoder token sequences (concat M*T tokens)
        #           through the fusion transformer (a stronger token-level fusion).
        self.fusion_input = fusion_input

        if input_mode == 'sax':
            self.token_embedding = nn.Embedding(alphabet_size + 1, token_dim, padding_idx=0)
            in_dim = lambda m: self.variates[m] * token_dim       # noqa: E731
        else:
            self.token_embedding = None
            in_dim = lambda m: self.variates[m]                   # noqa: E731

        self.input_projections = nn.ModuleDict({
            m: nn.Linear(in_dim(m), d_model) for m in self.modalities})
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_length, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.encoders = nn.ModuleDict({
            m: _TemporalEncoder(d_model, nhead, num_layers, dropout) for m in self.modalities})

        # cheap encoder-stage exit head (shared across exit positions)
        self.exit_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, self.num_classes))

        # strong final fusion exit: [cls] + modality tokens -> self-attention
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.modality_emb = nn.Embedding(self.M, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=num_fusion_layers)
        self.fusion_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.num_classes))
        # token+distill fusion: run the 4 fusion layers one-by-one, halving the modality
        # token sequence after each layer (Conv1d k3 + GELU + MaxPool/2), while a persistent
        # [cls] bottleneck reads the progressively-distilled tokens. (seqA-style distill,
        # as a regularizer for cross-subject generalization.)
        self.fusion_distill = fusion_distill
        if fusion_input == 'token' and fusion_distill:
            self.fusion_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout,
                    activation='gelu', batch_first=True, norm_first=True)
                for _ in range(num_fusion_layers)])
            self.fusion_distill_ops = nn.ModuleList([
                nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                              nn.GELU(), nn.MaxPool1d(2))
                for _ in range(num_fusion_layers)])

    def _encode_modality(self, mod_input, m):
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
        return (x * present).sum(dim=1) / denom                    # (B, d)

    def _fusion_logits(self, tokens, idxs):
        """Strong fusion exit over the given modality tokens (list of (B,d))."""
        B = tokens[0].shape[0]
        toks = torch.stack([t + self.modality_emb.weight[i] for t, i in zip(tokens, idxs)], dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        out = self.fusion(torch.cat([cls, toks], dim=1))
        return self.fusion_head(out[:, 0])

    # ----- token-level fusion path (fusion_input='token') -----------------------
    def _encode_seq(self, mod_input, m):
        """Per-modality encoder token SEQUENCE (B,T,d) + token-presence (B,T) bool
        (no mean-pool, unlike _encode_modality)."""
        if self.input_mode == 'sax':
            emb = self.token_embedding(mod_input.long())
            B, T, V_m, _ = emb.shape
            x = emb.reshape(B, T, V_m * self.token_dim)
            present = (mod_input != 0).any(dim=2)                  # (B,T)
        else:
            B, T, V_m = mod_input.shape
            x = mod_input
            present = torch.ones(B, T, dtype=torch.bool, device=x.device)
        x = self.input_projections[m](x) + self.pos_embedding[:, :T, :]
        x = self.encoders[m](x)                                    # (B,T,d)
        return x, present

    # ----- ADMN-style per-modality encoder-layer gating (encoder-only adaptive depth) -----
    def _encode_seq_gated(self, mod_input, m, keep_layers):
        """Like _encode_seq but encoder layers 1..L-1 are each gated per-sample by
        keep_layers[:, li-1] (B, L-1) via ADMN-style residual passthrough. Layer-0 always runs."""
        if self.input_mode == 'sax':
            emb = self.token_embedding(mod_input.long())
            B, T, V_m, _ = emb.shape
            x = emb.reshape(B, T, V_m * self.token_dim)
            present = (mod_input != 0).any(dim=2)
        else:
            B, T, V_m = mod_input.shape
            x = mod_input
            present = torch.ones(B, T, dtype=torch.bool, device=x.device)
        x = self.input_projections[m](x) + self.pos_embedding[:, :T, :]
        layers = self.encoders[m].encoder.layers
        x = layers[0](x)                                          # layer 0 always on
        for li in range(1, len(layers)):                         # layers 1..L-1 gated
            x_new = layers[li](x)
            k = keep_layers[:, li - 1].view(-1, 1, 1).float()
            x = x_new * k + x * (1.0 - k)
        return self.encoders[m].norm(x), present

    def forward_layermask(self, x, layer_keep):
        """ADMN forward: ALL modalities present; each modality's encoder layers 1..L-1 are gated
        by layer_keep[:, j, :] (B, M, L-1); then the token+distill fusion over all M -> logits."""
        seqs, prs = [], []
        for j, m in enumerate(self.modalities):
            s, p = self._encode_seq_gated(x[:, :, self.channel_map[m]], m, layer_keep[:, j, :])
            seqs.append(s); prs.append(p)
        return self._fuse_tokens(seqs, prs, list(range(self.M)))

    def forward_layermask_skip(self, x, layer_keep_int):
        """GFLOPs-faithful ADMN forward: layer_keep_int (M, L-1) binary, batch-shared — DROPPED
        encoder layers are truly SKIPPED (not computed-then-masked), matching ADMN's batch=1
        inference. For measuring the real per-config compute (use batch=1)."""
        seqs, prs = [], []
        for j, m in enumerate(self.modalities):
            mi = x[:, :, self.channel_map[m]]
            if self.input_mode == 'sax':
                emb = self.token_embedding(mi.long()); B, T, V_m, _ = emb.shape
                h = emb.reshape(B, T, V_m * self.token_dim); pres = (mi != 0).any(dim=2)
            else:
                B, T, V_m = mi.shape; h = mi; pres = torch.ones(B, T, dtype=torch.bool, device=h.device)
            h = self.input_projections[m](h) + self.pos_embedding[:, :T, :]
            layers = self.encoders[m].encoder.layers
            h = layers[0](h)                                       # layer-0 always
            for li in range(1, len(layers)):
                if float(layer_keep_int[j, li - 1]) > 0.5:         # SKIP dropped layers (real savings)
                    h = layers[li](h)
            seqs.append(self.encoders[m].norm(h)); prs.append(pres)
        return self._fuse_tokens(seqs, prs, list(range(self.M)))

    def _fuse_tokens(self, seqs, presents, idxs, mod_keep=None):
        """Token-level fusion: concat the per-modality token sequences (each +modality
        emb), prepend [cls], run the fusion transformer with a key-padding mask that
        ignores SAX-pad tokens and (per-sample) dropped modalities. Read [cls] -> logits.
            seqs: list of (B,T,d) ; presents: list of (B,T) bool ; idxs: modality indices
            mod_keep: optional (B, k) float/bool per-sample keep for the k modalities."""
        B, T = seqs[0].shape[0], seqs[0].shape[1]
        toks = torch.cat([s + self.modality_emb.weight[i] for s, i in zip(seqs, idxs)], dim=1)  # (B,k*T,d)
        pres = torch.cat(presents, dim=1)                          # (B,k*T) bool
        if mod_keep is not None:
            mk = mod_keep.bool().unsqueeze(-1).expand(B, len(seqs), T).reshape(B, -1)            # (B,k*T)
            pres = pres & mk
        if self.fusion_distill:
            # per-layer distill: [cls] bottleneck reads tokens, then tokens halve each layer.
            cls = self.cls_token.expand(B, -1, -1)                 # (B,1,d)
            for li, flayer in enumerate(self.fusion_layers):
                x = torch.cat([cls, toks], dim=1)                  # (B, 1+L, d)
                kpm = torch.zeros(B, x.shape[1], dtype=torch.bool, device=x.device)
                kpm[:, 1:] = ~pres
                x = flayer(x, src_key_padding_mask=kpm)
                cls, toks = x[:, :1], x[:, 1:]
                if toks.shape[1] >= 2:                             # distill (halve) the modality tokens
                    toks = self.fusion_distill_ops[li](toks.transpose(1, 2)).transpose(1, 2)
                    pres = F.max_pool1d(pres.float().unsqueeze(1), 2).squeeze(1) > 0   # downsample mask
            return self.fusion_head(cls[:, 0])
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, toks], dim=1)                        # (B, 1+k*T, d)
        kpm = torch.zeros(B, seq.shape[1], dtype=torch.bool, device=seq.device)
        kpm[:, 1:] = ~pres                                         # True == ignore (cls always kept)
        out = self.fusion(seq, src_key_padding_mask=kpm)
        return self.fusion_head(out[:, 0])

    @torch.no_grad()
    def forward_subset(self, x, chosen):
        """Inference at a fixed modality subset: encode ONLY `chosen`. |chosen|==M ->
        strong fusion exit; else cheap encoder-stage exit head on the prefix mean.
        Encodes only the chosen modalities -> cost scales with k (for GFLOPs eval)."""
        present = [m for m in self.modalities if m in set(chosen)]
        idxs = [self.modalities.index(m) for m in present]
        if self.fusion_input == 'token':
            seqs, prs = [], []
            for m in present:
                s, p = self._encode_seq(x[:, :, self.channel_map[m]], m)
                seqs.append(s); prs.append(p)
            return self._fuse_tokens(seqs, prs, idxs)              # token fusion at ANY subset size
        toks = [self._encode_modality(x[:, :, self.channel_map[m]], m) for m in present]
        if len(present) == self.M:
            return self._fusion_logits(toks, idxs)
        agg = torch.stack(toks, dim=1).mean(dim=1)
        return self.exit_head(agg)

    def forward(self, x, modality_dropout_probs=None, training=False, order=None):
        """Training/full forward: encode all M, per-sample dropout mask, random order,
        running-mean encoder-stage exits at every prefix + the fusion exit.
        Returns {'exit_logits':[B,M,C] (in `order`), 'fusion_logits':[B,C], 'order', 'present'}."""
        B = x.shape[0]; device = x.device

        present = torch.ones(B, self.M, device=device)
        if training and modality_dropout_probs is not None:
            for i, m in enumerate(self.modalities):
                p = (float(modality_dropout_probs.get(m, 0.0))
                     if isinstance(modality_dropout_probs, dict) else float(modality_dropout_probs))
                if p > 0:
                    present[torch.rand(B, device=device) < p, i] = 0.0
            empty = present.sum(dim=1) == 0
            if empty.any():
                present[empty, torch.randint(self.M, (int(empty.sum()),), device=device)] = 1.0

        if order is None:
            order = (torch.randperm(self.M, device=device).tolist() if training else list(range(self.M)))

        # ---- token-level fusion: per-prefix fusion over RAW modality token sequences ----
        if self.fusion_input == 'token':
            seqs, prs = [], []
            for m in self.modalities:
                s, p = self._encode_seq(x[:, :, self.channel_map[m]], m)
                seqs.append(s); prs.append(p)
            ex = []
            for k in range(1, self.M + 1):
                idx = order[:k]
                ex.append(self._fuse_tokens([seqs[i] for i in idx], [prs[i] for i in idx],
                                            idx, mod_keep=present[:, idx]))
            exit_logits = torch.stack(ex, dim=1)                   # (B,M,C) in `order`
            return {'exit_logits': exit_logits, 'fusion_logits': exit_logits[:, -1, :],
                    'order': order, 'present': present}

        # ---- pool fusion (MMExit-faithful): mean-pool each modality to ONE token ----
        tokens = torch.stack(
            [self._encode_modality(x[:, :, self.channel_map[m]], m) for m in self.modalities], dim=1)  # (B,M,d)
        # running masked-mean -> cheap exit head at each prefix position
        exit_logits = []
        cum_sum = torch.zeros(B, self.d_model, device=device)
        cum_cnt = torch.zeros(B, 1, device=device)
        for i in order:
            w = present[:, i:i + 1]
            cum_sum = cum_sum + w * tokens[:, i, :]
            cum_cnt = cum_cnt + w
            exit_logits.append(self.exit_head(cum_sum / cum_cnt.clamp(min=1.0)))
        exit_logits = torch.stack(exit_logits, dim=1)              # (B, M, C) in `order`

        # strong fusion exit over present tokens
        toks = torch.stack([tokens[:, i, :] + self.modality_emb.weight[i] for i in range(self.M)], dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        kpm = torch.zeros(B, 1 + self.M, dtype=torch.bool, device=device)
        kpm[:, 1:] = present == 0
        fout = self.fusion(torch.cat([cls, toks], dim=1), src_key_padding_mask=kpm)
        fusion_logits = self.fusion_head(fout[:, 0])
        return {'exit_logits': exit_logits, 'fusion_logits': fusion_logits,
                'order': order, 'present': present}


def mmexit_losses(out, y, label_smoothing: float = 0.0):
    """Multi-exit loss: mean CE over all encoder-stage exits + the fusion exit."""
    el = out['exit_logits']
    B, M, C = el.shape
    exit_loss = F.cross_entropy(el.reshape(B * M, C), y.unsqueeze(1).expand(B, M).reshape(B * M),
                                label_smoothing=label_smoothing)
    fusion_loss = F.cross_entropy(out['fusion_logits'], y, label_smoothing=label_smoothing)
    total = exit_loss + fusion_loss
    return total, {'exit': float(exit_loss.detach()), 'fusion': float(fusion_loss.detach())}


class IEMOCAPMMExitWrapper(nn.Module):
    """Concat the IEMOCAP feats-list highs -> [B,T,sum(V)] -> GenericMMExitClassifier."""

    def __init__(self, model, modalities):
        super().__init__()
        self.model = model
        self.M = len(modalities)

    def _concat(self, feats):
        return torch.cat([feats[2 * j] for j in range(self.M)], dim=-1)

    def forward(self, feats, modality_dropout_probs=None, training=False, order=None):
        return self.model(self._concat(feats), modality_dropout_probs=modality_dropout_probs,
                          training=training, order=order)

    @torch.no_grad()
    def forward_subset(self, feats, chosen):
        return self.model.forward_subset(self._concat(feats), chosen)
