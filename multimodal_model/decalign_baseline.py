"""DecAlign baseline (re-export wrapper) — clean_models self-contained version.

Re-exports :class:`DecAlignDSADS` (now living at
``multimodal_model.decalign_dsads`` in this package, copied verbatim from the
old MAESTRO root) and exposes a tiny helper that builds the DecAlign model from
a generic dataset-cfg object exposing ``modalities``, ``variates``,
``num_classes``.

Unlike the v6_organized version, this does NOT inject the old MAESTRO root onto
``sys.path`` — ``decalign_dsads`` is bundled here, so the package is fully
self-contained.
"""
from __future__ import annotations

from typing import Optional

from multimodal_model.decalign_dsads import DecAlignDSADS

__all__ = ['DecAlignDSADS', 'build_decalign']


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
    """Factory wrapping :class:`DecAlignDSADS` so callers don't have to
    duplicate the long kwarg list, and that exposes
    ``model.input_projections`` (= the conv-1d projection ModuleDict) for
    the :class:`ModalityGradientProfiler` hook."""
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
    # DecAlign uses ``self.proj`` as the per-modality input projection
    # ModuleDict.  ModalityGradientProfiler looks for either
    # ``input_projections`` or ``input_projectors`` — alias here so the
    # gradient-norm hook attaches to the proj.weight grads.
    model.input_projections = model.proj
    return model
