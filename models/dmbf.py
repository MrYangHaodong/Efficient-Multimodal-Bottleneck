"""DMBF — Dynamic Multi-Biosignal Fusion (faithful concept port of honeyvig/DMBF,
"Dynamic Multi-biosignal Fusion for Detecting the Mental States of Drivers and
Passengers in Vehicles", IEEE JBHI 2025).

The ORIGINAL DMBF fuses physiological biosignals (EEG/ECG/PPG/RSP/GSR) with an
EEGNet4 per-modality backbone. Its three signature ideas are modality-agnostic and
ported here onto a generic multimodal time-series backbone (CMU-MOSEI: vision/audio/
text sequences), clearly labelled DMBF-STYLE:

  1. Per-modality MSP CONFIDENCE GATING : each modality gets a confidence head; its
     max-softmax confidence (sigmoid-scaled) multiplies that modality's feature map,
     so low-quality modalities are down-weighted per sample.
  2. SPATIAL-TEMPORAL ATTENTION fusion : after concatenating the confidence-weighted
     per-modality maps along the channel axis, a temporal-attention branch (FC over
     time + ReLU gate) and a spatial-attention branch (FC over channels + softmax
     gate) are averaged, then classified.
  3. Correctness-Ranking Loss (CRL) : training ranks samples by per-sample confidence
     to match their running correctness (MarginRankingLoss) — calibrates confidence.

Deviation from the official code (intentional, to make DMBF a *fair* baseline):
the original applies F.softmax to the classifier/conf logits BEFORE CrossEntropyLoss
(a double-softmax). Here heads emit RAW logits and CE is applied properly.

forward(feats_list) -> dict(logits (B,C), conf_logits [M x (B,C)], sample_conf (B,)).
See crl_utils-equivalent History below for the CRL bookkeeping.
"""
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ModalityEncoder(nn.Module):
    """Per-modality temporal encoder: Linear(C->d) + pos-emb + N transformer layers.
    Returns the token sequence (B, T, d) — same style as our other MOSEI baselines.

    Supports two input regimes (backward-compatible; default is the original float):
      * input_mode='float' : x is (B, T, variates) continuous -> Linear(variates, d).
      * input_mode='sax'   : x is (B, T, variates) integer SAX tokens; a SHARED
        nn.Embedding (passed in) maps each token to token_dim, reshape to
        (B, T, variates*token_dim) -> Linear(variates*token_dim, d).
    """
    def __init__(self, variates, d_model, nhead, num_layers, input_length, dropout,
                 input_mode='float', token_embedding=None, token_dim=16):
        super().__init__()
        self.input_mode = input_mode
        self.variates = variates
        self.token_embedding = token_embedding
        self.token_dim = token_dim
        in_dim = variates * token_dim if input_mode == 'sax' else variates
        self.proj = nn.Linear(in_dim, d_model)
        # +64 margin: SAX tokenization can yield T slightly above input_length.
        self.pos = nn.Parameter(torch.zeros(1, input_length + 64, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):                       # x: (B, T, C) float, or (B,T,V) long sax
        if self.input_mode == 'sax':
            e = self.token_embedding(x.long())              # (B, T, V, token_dim)
            B, T, V, _ = e.shape
            x = e.reshape(B, T, V * self.token_dim)         # (B, T, V*token_dim)
        T = x.shape[1]
        h = self.proj(x) + self.pos[:, :T, :]
        return self.enc(h)                      # (B, T, d)


class DMBFClassifier(nn.Module):
    def __init__(self, modalities: List[str], feat_dims: Dict[str, int],
                 num_classes: int, input_length: int, *,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 2,
                 dropout: float = 0.2, input_mode: str = 'float',
                 alphabet_size: int = 20, token_dim: int = 16):
        super().__init__()
        assert input_mode in ('float', 'sax')
        self.modalities = list(modalities)
        self.M = len(self.modalities)
        self.T = input_length
        self.d = d_model
        self.num_classes = num_classes
        self.input_mode = input_mode

        # SAX regime: one SHARED token embedding (padding_idx=0) for all modalities.
        if input_mode == 'sax':
            self.token_embedding = nn.Embedding(alphabet_size + 1, token_dim, padding_idx=0)
        else:
            self.token_embedding = None

        self.encoders = nn.ModuleDict({
            m: _ModalityEncoder(feat_dims[m], d_model, nhead, num_layers, input_length, dropout,
                                input_mode=input_mode, token_embedding=self.token_embedding,
                                token_dim=token_dim)
            for m in self.modalities})
        # per-modality confidence head on the flattened (d*T) feature map
        self.conf_heads = nn.ModuleList([
            nn.Linear(d_model * input_length, num_classes) for _ in self.modalities])
        # temporal attn over the time axis (T -> T, ReLU gate)
        self.temporal_fc = nn.Linear(input_length, input_length)
        # spatial attn over the channel axis (M*d -> M*d, softmax gate)
        self.spatial_fc = nn.Linear(self.M * d_model, self.M * d_model)
        # fused classifier on flattened (M*d*T)
        self.classifier = nn.Linear(self.M * d_model * input_length, num_classes)

    def forward(self, feats_list: List[torch.Tensor]):
        assert len(feats_list) == self.M
        B = feats_list[0].shape[0]
        feat_maps, conf_logits = [], []
        for j, m in enumerate(self.modalities):
            h = self.encoders[m](feats_list[j])          # (B, T, d)
            # The flatten-based heads are fixed to self.T; SAX tokenization can yield
            # T off-by-one vs input_length, so pad/truncate the time axis to self.T.
            if h.shape[1] != self.T:
                if h.shape[1] > self.T:
                    h = h[:, :self.T, :]
                else:
                    h = F.pad(h, (0, 0, 0, self.T - h.shape[1]))
            fm = h.transpose(1, 2)                        # (B, d, T)   channels=d, time=T
            feat_maps.append(fm)
            conf_logits.append(self.conf_heads[j](fm.reshape(B, -1)))   # (B, C)

        # MSP confidence per modality, then sigmoid scaling (faithful to DMBF)
        confs = [F.softmax(cl, dim=1).max(dim=1).values for cl in conf_logits]   # each (B,)
        conf_stack = torch.stack(confs, dim=1)            # (B, M)
        sample_conf = conf_stack.mean(dim=1)              # (B,)  -> CRL ranking signal
        scaled = torch.sigmoid(conf_stack)                # (B, M)

        weighted = [feat_maps[j] * scaled[:, j].view(B, 1, 1) for j in range(self.M)]
        cat = torch.cat(weighted, dim=1)                  # (B, M*d, T)

        # temporal attention: gate over time, ReLU (retain positive)
        t_gate = F.relu(self.temporal_fc(cat))            # (B, M*d, T)
        tmp = cat * t_gate
        # spatial attention: gate over channels, softmax
        sp_in = cat.permute(0, 2, 1)                      # (B, T, M*d)
        sp_gate = F.softmax(self.spatial_fc(sp_in), dim=-1)
        sp = (sp_in * sp_gate).permute(0, 2, 1)           # (B, M*d, T)

        fused = 0.5 * (tmp + sp)                          # average the two branches
        logits = self.classifier(fused.reshape(B, -1))    # (B, C)  -- raw logits
        return {'logits': logits, 'conf_logits': conf_logits, 'sample_conf': sample_conf}


# --------------------------------------------------------------------------------
# CRL bookkeeping (port of DMBF_Ours/Helpers/crl_utils.py History) + loss helper.
# --------------------------------------------------------------------------------

class CorrectnessHistory:
    """Running per-sample correctness/confidence, for the Correctness-Ranking Loss."""
    def __init__(self, n_data: int):
        self.correctness = np.zeros(n_data, dtype=np.float64)
        self.max_correctness = 1

    def correctness_update(self, idx: torch.Tensor, correct: torch.Tensor):
        idx = idx.cpu().numpy()
        self.correctness[idx] += correct.cpu().numpy()

    def max_correctness_update(self, epoch: int):
        if epoch > 1:
            self.max_correctness += 1

    def _normalize(self, vals):
        lo = self.correctness.min()
        hi = float(self.max_correctness)
        return (vals - lo) / (hi - lo + 1e-12)

    def target_margin(self, idx1: torch.Tensor, idx2: torch.Tensor, device):
        c1 = self._normalize(self.correctness[idx1.cpu().numpy()])
        c2 = self._normalize(self.correctness[idx2.cpu().numpy()])
        target = (np.greater(c1, c2).astype('float') - np.less(c1, c2).astype('float'))
        margin = np.abs(c1 - c2)
        return (torch.from_numpy(target).float().to(device),
                torch.from_numpy(margin).float().to(device))


def dmbf_loss(out, y, history, idx, ranking_criterion, device, rank_weight=0.1):
    """Total DMBF loss = CE(fused) + mean_m CE(conf_head_m) + rank_weight * CRL."""
    logits = out['logits']
    cls_loss = F.cross_entropy(logits, y)
    conf_loss = torch.stack([F.cross_entropy(cl, y) for cl in out['conf_logits']]).mean()

    # CRL: rank this sample's confidence vs the next sample's (roll by -1)
    conf = out['sample_conf'].squeeze()
    rank_in1 = conf
    rank_in2 = torch.roll(conf, -1)
    idx2 = torch.roll(idx, -1)
    rank_target, rank_margin = history.target_margin(idx, idx2, device)
    nz = rank_target.clone(); nz[nz == 0] = 1.0
    rank_in2 = rank_in2 + rank_margin / nz
    ranking_loss = ranking_criterion(rank_in1, rank_in2, rank_target)

    total = cls_loss + conf_loss + rank_weight * ranking_loss
    return total, cls_loss, conf_loss, ranking_loss
