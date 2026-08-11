"""DyMo (Du et al., 2026) — faithful prototype selection on a cross-modal Transformer
fusion backbone, complete-data EFFICIENCY setting (greedy KEEP-subset, no imputer).

Self-contained port: merges the official DyMo MECHANISM (prototype-distance quality,
per-subset Gaussian probability, prototype training loss, per-epoch prototype
recompute, greedy per-sample reward selection) onto the repo's shared
``models.crossmodal_transformer.CrossModalTransformer`` backbone, so it runs on
IEMOCAP / daliahar / pamap2 / dsads / eav / mosei.

The DyMo MECHANISM (verbatim from the official repo):
  * prototype-distance quality  -> ``PrototypeDistance_COS``
  * per-subset Gaussian prob     -> ``calculate_probabilities_COS``
  * prototype training loss      -> ``PrototypeLoss`` (cosine)
  * per-epoch prototype recompute as class-mean of projected cls features
  * greedy reward  R(u) = quality(S u {u}) * G - quality(S),  G = log(p)/log(p_u) if p_u<p
The subset feature is the REAL fusion-transformer forward over the subset (cls token),
projected + L2-normalised. Per-modality tokens are encoded once and reused, so this is
faithful AND tractable. Subsets are keyed by the PRESENT-mask tuple (True = present)
consistently across ``subset2id``, the model forward, and the probability calc.
"""
from __future__ import annotations

from itertools import combinations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.crossmodal_transformer import CrossModalTransformer
from common.loader_utils import _concat_iemocap, _build_channel_map  # noqa: F401

__all__ = ['CrossModalDyMoProto', 'PrototypeLoss', 'PrototypeDistance_COS',
           'calculate_probabilities_COS', 'estimate_subset_gaussian',
           'proto_greedy_select']


# ============================ Official DyMo loss / quality ============================

class PrototypeLoss(nn.Module):
    """Push samples to the positive prototype and away from negatives (label-prob loss).

    label: (B, C) one-hot, prototypes: (C, D), feat: (B, D) (L2-normalised).
    """

    def __init__(self, temperature, metric_name: str = 'cosine_similarity') -> None:
        super().__init__()
        self.temperature = temperature
        self.metric_name = metric_name
        print(f'PrototypeLoss metric: {self.metric_name}')
        assert self.metric_name in ('cosine_similarity', 'squared_euclidean'), \
            f'Unsupported metric: {self.metric_name}'

    def forward(self, label: torch.Tensor, prototypes: torch.Tensor, feat: torch.Tensor):
        if self.metric_name == 'cosine_similarity':
            out = torch.mm(feat, prototypes.t()) / self.temperature
        else:  # squared_euclidean
            feat = feat.unsqueeze(1)
            prototypes = prototypes.unsqueeze(0)
            out = -((feat - prototypes).pow(2).sum(dim=2)) / self.temperature
        log_out = F.log_softmax(out, dim=1)
        loss = -torch.sum(log_out * label, dim=1)
        return loss.mean()


class PrototypeDistance_COS(nn.Module):
    """Quality of multimodal features = max softmax over cosine sim to class prototypes."""

    def __init__(self, temperature) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, feat: torch.Tensor, prototypes: torch.Tensor):
        sim = torch.mm(feat, prototypes.t()) / self.temperature             # (B, C)
        prediction = F.softmax(sim, dim=-1)
        quality, max_id = torch.max(prediction, dim=1)
        return quality, max_id, prediction


def calculate_probabilities_COS(feat, mask, max_id, subset_gaussian):
    """Per-subset Gaussian probability of the (cosine) distance to the predicted prototype."""
    subset2id = subset_gaussian['subset2id']
    prototypes_all = subset_gaussian['prototypes'].to(feat.device)          # (S, C, D)
    dist_std_all = subset_gaussian['dist_std'].to(feat.device)              # (S, C)
    probs, dist_list = [], []
    for i in range(len(feat)):
        feat_i = feat[i]                                                    # (D,)
        mask_i = mask[i]
        label = int(max_id[i])
        subset_id = subset2id[tuple(mask_i.tolist())]
        prototypes = prototypes_all[subset_id]                             # (C, D)
        dist_std = dist_std_all[subset_id, label]                         # scalar
        distribution = torch.distributions.Normal(0, dist_std)
        dist_classes = torch.matmul(prototypes, feat_i.unsqueeze(1)).squeeze(1)
        dist_score = 1 - dist_classes[label]
        prob = 2 * (1 - distribution.cdf(dist_score))
        probs.append(prob)
        dist_list.append(dist_score)
    probs = torch.stack(probs).to(feat.device)
    dist_list = torch.stack(dist_list).to(feat.device)
    return probs, dist_list


# ============================ Subset bookkeeping ============================

def _present_subsets(M: int):
    """All non-empty present-subsets of {0..M-1} as present-mask tuples (True=present)."""
    subs = []
    for k in range(1, M + 1):
        for c in combinations(range(M), k):
            mask = [False] * M
            for j in c:
                mask[j] = True
            subs.append(tuple(mask))
    return subs


# ============================ DyMo wrapper on the cross-modal backbone ============================

class CrossModalDyMoProto(nn.Module):
    """CrossModalTransformer cls-fusion + official DyMo prototype head & selection."""

    def __init__(self, model: CrossModalTransformer, num_classes: int,
                 projection_dim: int = 16, proto_temperature: float = 0.1):
        super().__init__()
        self.model = model
        self.M = model.M
        self.num_classes = num_classes
        self.proj_dim = projection_dim
        self.temperature = proto_temperature
        self.projection = nn.Linear(model.d_model, projection_dim)
        self.register_buffer('prototypes', torch.zeros(num_classes, projection_dim))
        self.quality_metric = PrototypeDistance_COS(temperature=proto_temperature)
        # subset bookkeeping (present-mask tuple -> id), all non-empty subsets
        subs = _present_subsets(self.M)
        self.subsets = subs
        self.subset2id = {s: i for i, s in enumerate(subs)}
        self.id2subset = {i: s for i, s in enumerate(subs)}

    # ---- core forwards ----
    def _feat_from_cls(self, cls_feat):
        return F.normalize(self.projection(cls_feat), dim=-1)

    def forward_train(self, x, mask):
        """x [B,T,sumV]; mask [B,M] bool (True=present). -> (logits [B,C], feat [B,D] norm)."""
        tok = self.model.encode_tokens(x)
        logits, cls_feat = self.model.fuse_feat(tok, mask)
        return logits, self._feat_from_cls(cls_feat)

    def fuse_from_tokens(self, tok, mask):
        logits, cls_feat = self.model.fuse_feat(tok, mask)
        return logits, self._feat_from_cls(cls_feat)

    @staticmethod
    def cal_prototypes(label_onehot, feat):
        """Official: class feature sums + counts (for per-epoch prototype recompute)."""
        class_sum = label_onehot.t() @ feat                                # (C, D)
        class_count = label_onehot.sum(dim=0, keepdim=True).t()            # (C, 1)
        return class_sum, class_count


# ----------------------------------------------------------------------------
# Offline per-subset Gaussian estimation (REAL subset forwards; tokens cached).
# Returns subset_gaussian usable by calculate_probabilities_COS:
#   {'subset2id', 'prototypes' (S,C,D), 'dist_std' (S,C), 'overall_prototypes'}
# ----------------------------------------------------------------------------
@torch.no_grad()
def estimate_subset_gaussian(proto: CrossModalDyMoProto, loader, device, unpack):
    proto.eval()
    M, C, D = proto.M, proto.num_classes, proto.proj_dim
    subs = proto.subsets
    S = len(subs)
    masks = torch.tensor(subs, dtype=torch.bool)                           # (S, M)
    feat_cache = [[] for _ in range(S)]                                    # per-subset (B,D)
    labels_all = []
    for batch in loader:
        x, y = unpack(batch, device)
        labels_all.append(y.cpu())
        tok = proto.model.encode_tokens(x)
        for si in range(S):
            m = masks[si].to(device).unsqueeze(0).expand(x.shape[0], -1)
            _, feat = proto.fuse_from_tokens(tok, m)
            feat_cache[si].append(feat.cpu())
    y_all = torch.cat(labels_all)                                          # (N,)
    subset_protos = torch.zeros(S, C, D)
    dist_std = torch.ones(S, C)
    for si in range(S):
        feat = torch.cat(feat_cache[si], dim=0)                            # (N, D)
        for c in range(C):
            sel = y_all == c
            if sel.sum() < 2:
                subset_protos[si, c] = proto.prototypes[c].cpu()
                dist_std[si, c] = 1.0
                continue
            fc = feat[sel]
            p = F.normalize(fc.mean(0), dim=-1)                            # class-subset prototype
            subset_protos[si, c] = p
            d = 1.0 - (fc * p).sum(-1)                                     # cosine distances
            dist_std[si, c] = d.std(unbiased=False).clamp(min=1e-3)
    return {'subset2id': proto.subset2id, 'prototypes': subset_protos,
            'dist_std': dist_std, 'overall_prototypes': proto.prototypes.detach().cpu(),
            'M': M, 'num_classes': C}


# ----------------------------------------------------------------------------
# Inference: official greedy reward selection (verbatim formula), real forwards.
#   returns (fused_logits [B,C], keep_mask [B,M] bool)
# ----------------------------------------------------------------------------
@torch.no_grad()
def proto_greedy_select(proto: CrossModalDyMoProto, x, sg, device):
    proto.eval()
    B, M = x.shape[0], proto.M
    overall = sg['overall_prototypes'].to(device)                          # (C, D)
    tok = proto.model.encode_tokens(x)

    def feat_logits(mask):
        return proto.fuse_from_tokens(tok, mask)                           # (logits, feat)

    def quality(feat):
        q, mid, _ = proto.quality_metric(feat, overall)
        return q, mid

    def gprob(feat, mask, mid):
        return calculate_probabilities_COS(feat, mask, mid, sg)[0]

    # start from single strongest modality (by prototype quality)
    eye = torch.eye(M, dtype=torch.bool, device=device)
    q_each = []
    for i in range(M):
        _, f = feat_logits(eye[i].unsqueeze(0).expand(B, -1))
        q_each.append(quality(f)[0])
    q_each = torch.stack(q_each, dim=1)                                    # (B, M)
    mask = torch.zeros(B, M, dtype=torch.bool, device=device)
    mask[torch.arange(B, device=device), q_each.argmax(1)] = True
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(M - 1):
        _, f = feat_logits(mask)
        q, mid = quality(f)
        p = gprob(f, mask, mid)
        best_R = torch.full((B,), -1e9, device=device)
        best_u = torch.full((B,), -1, dtype=torch.long, device=device)
        for u in range(M):
            addable = ~mask[:, u]
            cand = mask.clone(); cand[:, u] = True
            _, fu = feat_logits(cand)
            qu, midu = quality(fu)
            pu = gprob(fu, cand, midu)
            # official COS calibration G = log(p)/log(p_u) when p_u<p else 1
            G = torch.ones_like(p)
            lower = pu < p
            G = torch.where(lower, torch.log(p.clamp(1e-9, 1 - 1e-9)) /
                            torch.log(pu.clamp(1e-9, 1 - 1e-9)), G)
            R = qu * G - q
            R = torch.where(addable, R, torch.full_like(R, -1e9))
            better = R > best_R
            best_R = torch.where(better, R, best_R)
            best_u = torch.where(better, torch.full_like(best_u, u), best_u)
        finished = finished | (best_R <= 0)
        do = (~finished) & (best_u >= 0)
        if do.any():
            mask[do, best_u[do]] = True
        if finished.all():
            break
    logits, _ = feat_logits(mask)
    return logits, mask
