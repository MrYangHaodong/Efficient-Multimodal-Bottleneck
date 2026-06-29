"""ADMN-style adaptive ENCODER-LAYER allocation (Liu et al., NeurIPS 2025), ported to our
2-layer-encoder + 4-layer-distill-fusion backbone — WITHOUT the noise/quality machinery.

ADMN's mechanism (faithfully reproduced): a lightweight controller emits per-(modality,layer)
logits; a Gumbel-softmax top-k under a global layer budget picks which layers run; layer-0 of
each modality is always on; the encoder runs only the selected layers (LayerDrop-pretrained so
any subset works); straight-through estimator for training. Here, with 2-layer encoders, the
controller decides which modalities get their 2nd encoder layer (budget B of M), per sample.

Used to test the N question on the DEPTH axis: does ADMN's per-sample layer allocation beat a
static (val-frozen best-B) allocation under cross-subject shift?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax_sample(logits, temperature, scale=1.0):
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + 1e-12) + 1e-12) * scale          # ADMN scale=0.1
    return F.softmax((logits + g) / temperature, dim=-1)


def get_top_k(x, k):
    """Binary mask: 1 at the top-k entries of each row (ADMN's get_top_k)."""
    k = max(0, min(int(k), x.shape[1]))
    out = torch.zeros_like(x)
    if k == 0:
        return out
    idx = torch.topk(x, k, dim=1).indices
    return out.scatter_(1, idx, 1.0)


class ADMNLayerController(nn.Module):
    """Lightweight controller (ADMN ConvLayerController analog, noise head removed).
    Input: [B,T,C] -> per-modality mean-pooled summary -> small transformer over the M
    modality tokens -> M logits -> Gumbel top-k(budget) -> per-sample mask of which
    modalities get their 2nd encoder layer."""

    def __init__(self, channel_map, modalities, in_dims, budget, n_drop=1, h=128, depth=2, nhead=4):
        super().__init__()
        self.modalities = list(modalities)
        self.M = len(self.modalities)
        self.channel_map = channel_map
        self.n_drop = n_drop                                       # droppable layers per modality (L-1)
        self.budget = budget                                       # # droppable layers to KEEP (top-k over M*n_drop)
        self.proj = nn.ModuleDict({m: nn.Linear(in_dims[m], h) for m in self.modalities})
        layer = nn.TransformerEncoderLayer(h, nhead, 4 * h, 0.1, 'gelu',
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.cls = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.head = nn.Sequential(nn.Linear(h, 200, bias=False), nn.ReLU(),
                                  nn.Linear(200, self.M * n_drop, bias=False))  # logit per droppable layer
        for p in self.head.parameters():
            nn.init.uniform_(p, -0.005, 0.005)

    def forward(self, x, temp=1.0):
        B = x.shape[0]
        toks = []
        for m in self.modalities:
            xm = x[:, :, self.channel_map[m]].float().mean(dim=1)  # [B, V_m]
            toks.append(self.proj[m](xm))
        toks = torch.stack(toks, dim=1)                            # [B, M, h]
        cls = self.cls.expand(B, -1, -1)
        feat = self.enc(torch.cat([cls, toks], dim=1))[:, 0]       # [B, h]
        logits = self.head(feat)                                   # [B, M*n_drop]
        if self.training:
            soft = gumbel_softmax_sample(logits, temp, scale=0.1)
            disc = get_top_k(soft, self.budget)
            mask = soft + (disc - soft).detach()                   # straight-through, k-hot over M*n_drop
        else:
            mask = get_top_k(logits, self.budget)                  # hard top-k at inference
        return mask.view(B, self.M, self.n_drop), logits           # mask [B, M, n_drop]
