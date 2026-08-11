"""Shared loader helpers used by ee_data and by models that take a concatenated (B,T,sum_variates)
input tensor (float or SAX-token), split per modality via a channel map."""
import torch


def _concat_iemocap(feats, M):
    """Concatenate per-modality HIGH features [high0, low0, high1, low1, ...] -> (B, T, sum_C)."""
    return torch.cat([feats[2 * j] for j in range(M)], dim=-1)


def _build_channel_map(modalities, variates):
    """modality -> list of column indices in the concatenated input tensor."""
    mapping, start = {}, 0
    for mod in modalities:
        mapping[mod] = list(range(start, start + variates[mod]))
        start += variates[mod]
    return mapping
