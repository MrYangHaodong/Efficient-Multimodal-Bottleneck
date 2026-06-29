"""Learning-to-Route (L2R) — faithful concept port of Grosenick-Lab-Cornell/
learning-to-route, "Learning to Route: Per-Sample Adaptive Routing for Multimodal
Multitask Prediction" (Ajirak et al., NeurIPS 2025).

The original routes each sample through one of 4 hand-crafted paths over 2 modalities
(numeric / text): N1, T1, and two cross-fused paths, then a per-path STL/MTL task
router + experts soft-combine the prediction. Ported here to time-series multimodal
classification (M modalities, SINGLE task), so:

  * the STL/MTL TASK router is dropped (single task);
  * the modality ROUTER selects per-sample among M unimodal paths + 1 all-fused path
    (M+1 paths), generalising the original's "raw-unimodal + cross-fused" design;
  * each modality is first encoded to an embedding (transformer encoder + mean-pool),
    matching our other baselines, then routing/experts operate in embedding space.

This is the canonical "learned per-sample routing" family (cf. DynMM/AdaMML) that our
paper argues collapses/underperforms a static val-frozen schedule under shift.

forward(feats_list, hard) -> dict(logits, route_probs (B,M+1), route_logits, path_logits).
"""
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.loader_utils import _build_channel_map, _concat_iemocap  # noqa: F401


class _ModalityEncoder(nn.Module):
    """Per-modality encoder: Linear(C->d) + pos-emb + N transformer layers -> (B,T,d).

    input_mode='float' (default): x is (B,T,variates) continuous -> Linear(variates,d).
    input_mode='sax': x is (B,T,variates) integer SAX tokens; embedded via a SHARED
      token Embedding to (B,T,variates*token_dim) -> Linear(variates*token_dim,d).
    """
    def __init__(self, variates, d_model, nhead, num_layers, input_length, dropout,
                 input_mode='float', token_dim=16, token_embedding=None):
        super().__init__()
        self.input_mode = input_mode
        self.token_dim = token_dim
        self.token_embedding = token_embedding          # shared nn.Embedding (sax only)
        in_dim = variates * token_dim if input_mode == 'sax' else variates
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, input_length + 64, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):                       # (B, T, C)  float, or long sax tokens
        if self.input_mode == 'sax':
            e = self.token_embedding(x.long())  # (B, T, V, token_dim)
            B, T, V, _ = e.shape
            x = e.reshape(B, T, V * self.token_dim)
        T = x.shape[1]
        h = self.proj(x) + self.pos[:, :T, :]
        return self.enc(h)                      # (B, T, d)


class L2RRoutingClassifier(nn.Module):
    def __init__(self, modalities: List[str], feat_dims: Dict[str, int],
                 num_classes: int, input_length: int, *,
                 input_mode: str = 'float', alphabet_size: int = 20, token_dim: int = 16,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 2,
                 hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        assert input_mode in ('float', 'sax')
        self.modalities = list(modalities)
        self.M = len(self.modalities)
        self.n_paths = self.M + 1                # M unimodal paths + 1 all-fused path
        self.d = d_model
        self.input_mode = input_mode

        # sax: shared token embedding (token 0 = padding/missing) reused by all encoders
        if input_mode == 'sax':
            self.token_embedding = nn.Embedding(alphabet_size + 1, token_dim, padding_idx=0)
        else:
            self.token_embedding = None

        self.encoders = nn.ModuleDict({
            m: _ModalityEncoder(feat_dims[m], d_model, nhead, num_layers, input_length, dropout,
                                input_mode=input_mode, token_dim=token_dim,
                                token_embedding=self.token_embedding)
            for m in self.modalities})

        # modality router: MLP over concat(all embeddings) -> probs over M+1 paths
        self.router = nn.Sequential(
            nn.Linear(self.M * d_model, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_paths))
        # cross-nonlinear fusion for the all-fused path (analog of original's sin-MLP)
        self.fuse = nn.Sequential(
            nn.Linear(self.M * d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        # per-path expert (encoder) + classification head (single task -> C logits)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(hidden, hidden), nn.ReLU())
            for _ in range(self.n_paths)])
        self.heads = nn.ModuleList([nn.Linear(hidden, num_classes) for _ in range(self.n_paths)])

    def forward(self, feats_list: List[torch.Tensor], hard: bool = False, tau: float = 1.0):
        assert len(feats_list) == self.M
        embs = [self.encoders[m](feats_list[j]).mean(dim=1)         # (B, d)
                for j, m in enumerate(self.modalities)]
        cat = torch.cat(embs, dim=-1)                               # (B, M*d)

        # path representations: M unimodal embeddings + 1 cross-fused
        path_reps = list(embs) + [self.fuse(cat)]                   # (M+1) x (B, d)

        route_logits = self.router(cat)                             # (B, M+1)
        if hard:
            route_probs = F.gumbel_softmax(route_logits, tau=tau, hard=True, dim=-1)
        else:
            route_probs = F.softmax(route_logits, dim=-1)

        path_logits = [self.heads[k](self.experts[k](path_reps[k])) # each (B, C)
                       for k in range(self.n_paths)]
        stacked = torch.stack(path_logits, dim=1)                   # (B, M+1, C)
        logits = (route_probs.unsqueeze(-1) * stacked).sum(dim=1)   # (B, C)  soft-combined
        return {'logits': logits, 'route_probs': route_probs,
                'route_logits': route_logits, 'path_logits': path_logits}


def l2r_loss(out, y, load_balance_weight=0.01):
    """CE on the routed prediction + a small load-balance term (mean path usage ->
    uniform) to keep the router from collapsing onto one path early."""
    cls_loss = F.cross_entropy(out['logits'], y)
    probs = out['route_probs']                                      # (B, M+1)
    usage = probs.mean(dim=0)                                       # (M+1,)
    n = usage.shape[0]
    # KL(usage || uniform) — 0 when all paths used equally
    load = (usage * (usage * n + 1e-9).log()).sum()
    return cls_loss + load_balance_weight * load, cls_loss, load
