"""DecAlign — decoupled-alignment multimodal fusion baseline (self-contained).

Faithful port of the DecAlign architecture, generalized from the original
3-modality (text/audio/video) design to M-modality multivariate time series.
This module is fully self-contained: it bundles the :class:`DecAlignDSADS`
model definition plus a :func:`build_decalign` factory, with no cross-model
imports (only ``common.loader_utils`` helpers).

Architecture (see DecAlign, MAESTRO port):
  * Modality decoupling: per-modality ``encoder_uni_*`` (specific) + a SHARED
    ``encoder_com`` (common). Cosine push s vs c apart.
  * Heterogeneity alignment: GMM prototypes per modality + multi-marginal
    Sinkhorn OT over the M-dimensional cost tensor (K^M entries) + per-pair
    local-prototype L2.
  * Homogeneity alignment: pairwise statistical matching (mean / var / skew)
    + pairwise MMD over the common features.
  * Three-path fusion: (A) transformer on concat specific features,
    (B) M*(M-1) directed cross-attn + M self-memory transformers,
    (C) homogeneity branch (avg-pool common features). Final classifier on
    cat([hete = A + B, homo = C]).

Total loss: task_loss + alpha1 * dec_loss + alpha2 * (hete_loss + homo_loss).

Forward expects {modality: [B, T, V_m]} OR a pre-concatenated
[B, T, sum(V)] tensor (split internally via the channel map). In ``transform``
in ('sax', 'sax_noisy') mode the input is integer SAX tokens, embedded to
``token_dim`` float vectors before the Conv1d projection.
"""
from __future__ import annotations
from itertools import combinations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.loader_utils import _build_channel_map, _concat_iemocap  # noqa: F401

__all__ = ['DecAlignDSADS', 'build_decalign']


# ---------------------------------------------------------------------------
# DecAlign — generic M-modality
# ---------------------------------------------------------------------------
class DecAlignDSADS(nn.Module):
    def __init__(
        self,
        cfg,
        num_classes: int,
        input_length: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        num_prototypes: int = 5,
        ot_reg: float = 0.1,
        ot_num_iters: int = 50,
        conv1d_kernel: int = 3,
        dropout: float = 0.1,
        alpha1: float = 0.1,           # decouple loss weight
        alpha2: float = 0.01,          # alignment loss weight (hete + homo)
        transform: Optional[str] = None,  # None | 'sax' | 'sax_noisy'
        alphabet_size: int = 20,          # SAX alphabet (only used when transform != None)
        token_dim: int = 16,              # per-token embedding dim
        verbose: bool = True,
    ):
        super().__init__()
        self.modalities: List[str] = list(cfg.modalities)
        self.variates: Dict[str, int] = dict(cfg.variates)
        self.M = len(self.modalities)
        self.d = d_model
        self.K = num_prototypes
        self.input_length = input_length
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.ot_reg = ot_reg
        self.ot_num_iters = ot_num_iters
        # Per-modality column ranges in the concatenated [B, T, sum(V)] tensor.
        self.channel_map = _build_channel_map(self.modalities, self.variates)

        # SAX support: when transform is 'sax' or 'sax_noisy', x[m] is a
        # [B, T, V_m] long tensor of integer tokens. We embed each token to a
        # token_dim float vector and feed V_m * token_dim channels to Conv1d
        # instead of V_m. padding_idx=0 keeps the missing/noisy sentinel
        # mapped to a frozen zero vector. With transform=None, the path is
        # unchanged (raw float Conv1d, V_m channels).
        self.transform = transform
        self.alphabet_size = alphabet_size
        self.token_dim = token_dim
        self._sax_mode = transform in ('sax', 'sax_noisy')
        if self._sax_mode:
            self.token_embedding = nn.Embedding(
                alphabet_size + 1, token_dim, padding_idx=0)
        else:
            self.token_embedding = None

        # Padding chosen so Conv1d preserves T (kernel must be odd).
        assert conv1d_kernel % 2 == 1, "conv1d_kernel must be odd to preserve T"
        pad = conv1d_kernel // 2

        # -- Step 1: per-modality input projection (Conv1d, preserves T)
        # In SAX mode, input channels = V_m * token_dim (after embedding).
        def _in_ch(m):
            return self.variates[m] * (token_dim if self._sax_mode else 1)
        self.proj = nn.ModuleDict({
            m: nn.Conv1d(_in_ch(m), d_model,
                         kernel_size=conv1d_kernel, padding=pad, bias=False)
            for m in self.modalities
        })
        # Alias consumed by the ModalityGradientProfiler grad-norm hook.
        self.input_projections = self.proj

        # -- Step 2: Decouple — specific (per-modality) + common (SHARED conv)
        self.encoder_uni = nn.ModuleDict({
            m: nn.Conv1d(d_model, d_model, kernel_size=1, padding=0, bias=False)
            for m in self.modalities
        })
        self.encoder_com = nn.Conv1d(d_model, d_model, kernel_size=1, padding=0, bias=False)

        # -- Step 3: GMM prototypes (heterogeneity alignment)
        self.proto = nn.ParameterDict({
            m: nn.Parameter(torch.randn(num_prototypes, d_model) * 0.02)
            for m in self.modalities
        })
        self.logvar = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(num_prototypes, d_model))
            for m in self.modalities
        })

        # -- Step 4: Three fusion paths
        # Path A — Transformer on concat([s_m], feature dim) of width M*d
        layer_A = nn.TransformerEncoderLayer(
            d_model=self.M * d_model, nhead=nhead,
            dim_feedforward=4 * self.M * d_model, dropout=dropout,
            activation='gelu', batch_first=False, norm_first=True,
        )
        self.transformer_fusion = nn.TransformerEncoder(layer_A, num_layers=num_layers)

        # Path B — M*(M-1) cross-attention + M self-memory transformers
        self.cross_attn = nn.ModuleDict()
        for m1 in self.modalities:
            for m2 in self.modalities:
                if m1 == m2:
                    continue
                self.cross_attn[f'{m1}_with_{m2}'] = nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=nhead,
                    dropout=dropout, batch_first=False,
                )
        # Self-memory after cat'ing M-1 cross-attn outputs (width (M-1)*d)
        self.self_mem = nn.ModuleDict()
        for m in self.modalities:
            layer_mem = nn.TransformerEncoderLayer(
                d_model=(self.M - 1) * d_model, nhead=nhead,
                dim_feedforward=4 * (self.M - 1) * d_model, dropout=dropout,
                activation='gelu', batch_first=False, norm_first=True,
            )
            self.self_mem[m] = nn.TransformerEncoder(layer_mem, num_layers=num_layers)
        # Project Path B output [N, M*(M-1)*d] -> [N, M*d]
        self.cma_proj = nn.Linear(self.M * (self.M - 1) * d_model, self.M * d_model)

        # -- Step 5: classifier on cat([hete, homo], dim=1) of width 2*M*d
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * self.M * d_model),
            nn.Linear(2 * self.M * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        if verbose:
            n_params = sum(p.numel() for p in self.parameters())
            print(f'Initializing DecAlignDSADS  (params={n_params:,})')
            print(f'  Modalities       : {self.modalities}  (M={self.M})')
            print(f'  d_model / nhead  : {d_model} / {nhead}')
            print(f'  Prototypes K     : {num_prototypes}  (K^M tensor = {num_prototypes**self.M} entries)')
            print(f'  Cross-attn count : {self.M * (self.M - 1)} + {self.M} self-mem')
            print(f'  Path A concat dim: {self.M * d_model}')
            print(f'  Final rep dim    : {2 * self.M * d_model}')
            print(f'  alpha1={alpha1}  alpha2={alpha2}  ot_reg={ot_reg}  ot_iters={ot_num_iters}')

    # =====================================================================
    # Helpers — split, decouple-loss, prototype, OT, MMD
    # =====================================================================

    def _split(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split [B, T, sum(variates)] into per-modality dict {m: [B, T, V_m]}."""
        return {m: x[:, :, self.channel_map[m]] for m in self.modalities}

    def compute_decoupling_loss(self, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Push specific (s) and common (c) apart. Cosine >= 0 => penalize."""
        s_pool = s.mean(dim=2)
        c_pool = c.mean(dim=2)
        s_n = F.normalize(s_pool, dim=-1)
        c_n = F.normalize(c_pool, dim=-1)
        return (s_n * c_n).sum(dim=-1).abs().mean()

    def compute_prototypes(
        self, s: torch.Tensor, proto: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Soft GMM assignment, [N, d, T] -> [N, K]."""
        feat = s.mean(dim=2)                                 # [N, d]
        diff = feat.unsqueeze(1) - proto.unsqueeze(0)        # [N, K, d]
        var = logvar.exp().unsqueeze(0)                       # [1, K, d]
        dist = (diff ** 2 / var).sum(dim=2)                   # [N, K]
        return F.softmax(-dist, dim=1)

    def pairwise_cost(
        self, mu1: torch.Tensor, logvar1: torch.Tensor,
        mu2: torch.Tensor, logvar2: torch.Tensor, eps: float = 1e-9,
    ) -> torch.Tensor:
        """Cost matrix between two prototype sets: Euclidean + cov-matching."""
        diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)            # [K, K, d]
        dist_sq = (diff ** 2).sum(dim=2)                      # [K, K]
        sigma1 = logvar1.exp(); sigma2 = logvar2.exp()
        cov_term = (sigma1.unsqueeze(1) + sigma2.unsqueeze(0) -
                    2 * (sigma1.unsqueeze(1) * sigma2.unsqueeze(0) + eps).sqrt()).sum(dim=2)
        return dist_sq + cov_term                             # [K, K]

    def multi_marginal_sinkhorn(
        self,
        C: torch.Tensor,                  # shape [K]*M
        marginals: List[torch.Tensor],    # list of M tensors, each [K] (sum=1)
        reg: float,
        num_iters: int = 50,
        eps: float = 1e-9,
    ):
        """Multi-marginal Sinkhorn (M-marginal generalization of Sinkhorn).

        At convergence T = K_kernel * outer_prod(scales), where scales[d]
        normalize the marginal of T over all-dims-except-d to marginals[d].
        """
        M = len(marginals)
        K = C.size(0)
        K_kernel = torch.exp(-C / reg)                        # [K]*M
        device = C.device
        scales = [torch.ones(K, device=device) for _ in range(M)]

        for _ in range(num_iters):
            for d in range(M):
                # Build outer product of scales except scales[d], multiply with K_kernel.
                T = K_kernel
                for e in range(M):
                    if e == d:
                        continue
                    shape = [1] * M
                    shape[e] = K
                    T = T * scales[e].view(shape)
                # Sum over all dims except d -> marginal along d.
                marg = T.sum(dim=tuple(i for i in range(M) if i != d))
                scales[d] = marginals[d] / (marg + eps)

        # Final transport plan T = K_kernel * outer_prod(scales).
        T = K_kernel
        for d in range(M):
            shape = [1] * M
            shape[d] = K
            T = T * scales[d].view(shape)

        ot_loss = (T * C).sum()
        entropy = -(T * (T + eps).log()).sum()
        return T, ot_loss + 0.001 * reg * entropy

    def compute_hetero_loss(self, s_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Heterogeneity loss = multi-marginal OT + pairwise local-prototype L2."""
        # 1. Soft GMM assignments
        w = {m: self.compute_prototypes(s_dict[m], self.proto[m], self.logvar[m])
             for m in self.modalities}
        # 2. Marginal distributions over prototypes
        nu = {}
        for m in self.modalities:
            mar = w[m].mean(dim=0)
            nu[m] = mar / (mar.sum() + 1e-9)

        # 3. M-D joint cost tensor: sum of pairwise costs broadcast into [K]*M
        K, M = self.K, self.M
        device = self.proto[self.modalities[0]].device
        C = torch.zeros([K] * M, device=device)
        for ai, bi in combinations(range(M), 2):
            m_a, m_b = self.modalities[ai], self.modalities[bi]
            cost_ab = self.pairwise_cost(
                self.proto[m_a], self.logvar[m_a],
                self.proto[m_b], self.logvar[m_b],
            )                                                 # [K, K]
            shape = [1] * M
            shape[ai] = K; shape[bi] = K
            C = C + cost_ab.reshape(shape)

        # 4. Multi-marginal Sinkhorn
        marginals = [nu[m] for m in self.modalities]
        _, ot_loss = self.multi_marginal_sinkhorn(
            C, marginals, reg=self.ot_reg, num_iters=self.ot_num_iters,
        )

        # 5. Pairwise local-prototype L2 (M*(M-1) terms, weighted by source modality's w)
        feat = {m: s_dict[m].mean(dim=2) for m in self.modalities}   # [N, d] each
        local_loss = 0.0
        for m1 in self.modalities:
            for m2 in self.modalities:
                if m1 == m2:
                    continue
                # weighted L2 from samples of m1 to prototypes of m2
                d_n_k = ((feat[m1].unsqueeze(1) - self.proto[m2].unsqueeze(0)) ** 2).sum(dim=2)
                local_loss = local_loss + (w[m1] * d_n_k).mean()
        return ot_loss + local_loss

    def compute_mmd(self, x: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0):
        xx = x @ x.t()
        yy = y @ y.t()
        xy = x @ y.t()
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        K_xx = torch.exp(-(rx.t() + rx - 2 * xx) / (2 * bandwidth))
        K_yy = torch.exp(-(ry.t() + ry - 2 * yy) / (2 * bandwidth))
        K_xy = torch.exp(-(rx.t() + ry - 2 * xy) / (2 * bandwidth))
        return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

    def compute_homo_loss(self, c_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Homogeneity loss = pairwise stat matching (mean/var/skew) + pairwise MMD."""
        # 1. Pre-compute per-modality (mean, var, skew)
        stats = {}
        c_pool = {}
        for m in self.modalities:
            c = c_dict[m]                                     # [N, d, T]
            mu = c.mean(dim=(0, 2))                            # [d]
            sigma = c.var(dim=(0, 2))                          # [d]
            centered = c - mu.view(1, -1, 1)
            skew = (centered ** 3).mean(dim=(0, 2)) / (sigma + 1e-6).pow(1.5)
            stats[m] = (mu, sigma, skew)
            c_pool[m] = c.mean(dim=2)                          # [N, d]

        # 2. Pairwise stat matching (M*(M-1)/2 pairs)
        L_sem = 0.0
        for m1, m2 in combinations(self.modalities, 2):
            mu1, s1, sk1 = stats[m1]; mu2, s2, sk2 = stats[m2]
            L_sem = L_sem + (mu1 - mu2).pow(2).sum() \
                          + (s1 - s2).pow(2).sum() \
                          + (sk1 - sk2).pow(2).sum()

        # 3. Pairwise MMD
        L_mmd = 0.0
        for m1, m2 in combinations(self.modalities, 2):
            L_mmd = L_mmd + self.compute_mmd(c_pool[m1], c_pool[m2])

        return L_sem + L_mmd

    # =====================================================================
    # Forward
    # =====================================================================

    def forward(self, x):
        """Args:
            x: either dict {modality: [B, T, V_m]} or [B, T, sum(variates)]

        Returns:
            dict with 'logits', 'dec_loss', 'hete_loss', 'homo_loss',
            and the intermediate tensors fusion_rep_*.
        """
        if torch.is_tensor(x):
            x = self._split(x)

        # -- Step 1: per-modality projection (Conv1d expects [B, C, T])
        # SAX mode: input is long tokens [B, T, V_m]. Embed each token to
        # a token_dim float vector, flatten the V_m x token_dim axes, then
        # feed V_m*token_dim channels to Conv1d.
        # Float mode: keep the original Conv1d-on-raw-V_m-channels path.
        proj = {}
        for m in self.modalities:
            if self._sax_mode:
                # x[m]: [B, T, V_m] long
                xm_long = x[m].long()
                emb = self.token_embedding(xm_long)             # [B, T, V_m, token_dim]
                B, T, V_m, _ = emb.shape
                xm = emb.reshape(B, T, V_m * self.token_dim)    # [B, T, V_m*tok]
                xm = xm.transpose(1, 2).contiguous()            # [B, V_m*tok, T]
            else:
                xm = x[m].transpose(1, 2)                       # [B, V_m, T]
            proj[m] = self.proj[m](xm)                          # [B, d, T]

        # -- Step 2: Decouple specific / common
        s = {m: self.encoder_uni[m](proj[m]) for m in self.modalities}   # [B, d, T]
        c = {m: self.encoder_com(proj[m]) for m in self.modalities}      # [B, d, T] (shared conv)

        # Decouple loss (sum over modalities)
        dec_loss = 0.0
        for m in self.modalities:
            dec_loss = dec_loss + self.compute_decoupling_loss(s[m], c[m])

        # -- Step 3: alignment losses
        hete_loss = self.compute_hetero_loss(s)
        homo_loss = self.compute_homo_loss(c)

        # -- Step 4: three fusion paths
        # Path A: transformer on concat([s_m], feature dim)
        s_perm = {m: s[m].permute(2, 0, 1) for m in self.modalities}      # [T, N, d]
        T_min = min(s_perm[m].size(0) for m in self.modalities)
        s_perm = {m: s_perm[m][:T_min] for m in self.modalities}
        fused_A = torch.cat([s_perm[m] for m in self.modalities], dim=2)  # [T, N, M*d]
        trans_out = self.transformer_fusion(fused_A)                       # [T, N, M*d]
        fusion_rep_trans = trans_out[-1]                                   # [N, M*d]

        # Path B: cross-modal attention + self-memory per modality
        last_hs = []
        for m1 in self.modalities:
            others = []
            for m2 in self.modalities:
                if m1 == m2:
                    continue
                q = s_perm[m1]; k = s_perm[m2]; v = s_perm[m2]
                attn_out, _ = self.cross_attn[f'{m1}_with_{m2}'](q, k, v)
                others.append(attn_out)                                    # each [T, N, d]
            fused_m1 = torch.cat(others, dim=2)                            # [T, N, (M-1)*d]
            mem_out = self.self_mem[m1](fused_m1)                          # [T, N, (M-1)*d]
            last_hs.append(mem_out[-1])                                    # [N, (M-1)*d]
        fusion_rep_cma = torch.cat(last_hs, dim=1)                          # [N, M*(M-1)*d]
        fusion_rep_cma = self.cma_proj(fusion_rep_cma)                      # [N, M*d]

        # Path C: homogeneity branch — avg-pool c_* over time, concat
        fusion_rep_homo = torch.cat([c[m].mean(dim=2) for m in self.modalities], dim=1)  # [N, M*d]

        # -- Step 5: combine and classify
        fusion_rep_hete = fusion_rep_trans + fusion_rep_cma                 # [N, M*d]
        final_rep = torch.cat([fusion_rep_hete, fusion_rep_homo], dim=1)    # [N, 2*M*d]
        logits = self.classifier(final_rep)

        return {
            'logits': logits,
            'dec_loss': dec_loss,
            'hete_loss': hete_loss,
            'homo_loss': homo_loss,
            'fusion_rep_trans': fusion_rep_trans,
            'fusion_rep_cma': fusion_rep_cma,
            'fusion_rep_hete': fusion_rep_hete,
            'fusion_rep_homo': fusion_rep_homo,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_decalign(
    cfg,
    input_length: int,
    *,
    d_model: int = 64,
    nhead: int = 8,
    num_layers: int = 3,
    num_prototypes: int = 5,
    ot_reg: float = 0.1,
    ot_num_iters: int = 50,
    conv1d_kernel: int = 3,
    dropout: float = 0.1,
    alpha1: float = 0.1,
    alpha2: float = 0.01,
    transform: Optional[str] = None,
    alphabet_size: int = 20,
    token_dim: int = 16,
    verbose: bool = True,
) -> DecAlignDSADS:
    """Wraps :class:`DecAlignDSADS` so callers don't duplicate the long kwarg
    list. ``model.input_projections`` aliases the proj ModuleDict for the
    ModalityGradientProfiler grad-norm hook."""
    model = DecAlignDSADS(
        cfg=cfg,
        num_classes=cfg.num_classes,
        input_length=input_length,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        num_prototypes=num_prototypes, ot_reg=ot_reg,
        ot_num_iters=ot_num_iters, conv1d_kernel=conv1d_kernel,
        dropout=dropout, alpha1=alpha1, alpha2=alpha2,
        transform=transform, alphabet_size=alphabet_size,
        token_dim=token_dim, verbose=verbose,
    )
    return model
