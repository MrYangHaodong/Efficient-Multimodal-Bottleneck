"""
Minimal PointNet encoder: shared MLP + global max-pool.

No spatial transformer — keeps the code simple and handles arbitrary in_dim
(D=5 for mmwave x/y/z/Doppler/intensity, D=3 for lidar x/y/z).

Reference architecture: yanx27/Pointnet_Pointnet2_pytorch (MIT).
"""

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """
    Maps a variable-length point cloud (N, in_dim) → fixed vector (out_dim,).

    Args:
        in_dim:  number of input channels per point (3 for lidar, 5 for mmwave)
        out_dim: output feature dimension (default 128)

    Forward:
        x    : (B, in_dim, N)  — N can vary across calls; B is the frame batch
        mask : (B, N) bool     — True for real points, False for zero-padding
                                 if None, all points treated as valid

    Returns: (B, out_dim)
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64,      1), nn.BatchNorm1d(64),      nn.ReLU(),
            nn.Conv1d(64,     128,     1), nn.BatchNorm1d(128),     nn.ReLU(),
            nn.Conv1d(128,    out_dim, 1), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, in_dim, N)
        if x.shape[2] == 0:
            return torch.zeros(x.shape[0], self.out_dim, device=x.device)

        feat = self.mlp(x)                          # (B, out_dim, N)

        if mask is not None:
            # mask: (B, N) True=valid — set invalid positions to -inf before max
            inv = ~mask.unsqueeze(1)                # (B, 1, N)
            feat = feat.masked_fill(inv, float('-inf'))

        out = feat.max(dim=2)[0]                    # (B, out_dim)

        # Replace -inf (all-padding frames) with 0
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out
