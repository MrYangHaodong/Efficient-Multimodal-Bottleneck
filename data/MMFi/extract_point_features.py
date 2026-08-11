"""
Pre-extract mmwave and lidar features for MM-Fi using a PointNet encoder.

Reads raw .bin files from per-subject zips, encodes each frame's variable-length
point cloud into a fixed (out_dim,) vector, and saves (T=297, out_dim) npy arrays.

Output directory structure:
  {feat_root}/
    mmwave_d{out_dim}/
      S01_A01.npy  ...  S40_A27.npy    # (297, out_dim) float32
    lidar_d{out_dim}/
      S01_A01.npy  ...  S40_A27.npy    # (297, out_dim) float32

Usage:
  cd /home/group/maestro_visual/Efficient-Multimodal-Bottleneck
  python data/MMFi/extract_point_features.py \
      --data-root /home/group/maestro_visual/data/MM-Fi \
      --feat-root /home/group/maestro_visual/data/MM-Fi/features \
      --out-dim 128 --frame-batch 64 --device cuda:0

The encoder weights are random-initialized.  The pre-extracted features are
fixed descriptors; seqA learns a linear projection on top during training.
Run this script once per out_dim setting before training.
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import torch

from pointnet_encoder import PointNetEncoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SUBJECTS = list(range(1, 41))   # S01 … S40
ALL_ACTIONS  = list(range(1, 28))   # A01 … A27
T_FRAMES     = 297


# ---------------------------------------------------------------------------
# Raw loading helpers
# ---------------------------------------------------------------------------

def _list_frames(zf: zipfile.ZipFile, subject: int, action: int, modality: str,
                 ext: str) -> list:
    prefix = f'S{subject:02d}/A{action:02d}/{modality}/'
    files  = sorted(
        [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(ext)],
        key=lambda x: int(x.rsplit('frame', 1)[1].split('.')[0])
    )
    return files


def _read_mmwave_frame(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Returns (N, 5) float64 — x, y, z, Doppler, intensity.  N may be 0."""
    raw = zf.read(name)
    if len(raw) == 0:
        return np.zeros((0, 5), dtype=np.float64)
    pts = np.frombuffer(raw, dtype=np.float64).reshape(-1, 5)
    return pts


def _read_lidar_frame(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Returns (N, 3) float64 — x, y, z."""
    raw = zf.read(name)
    if len(raw) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = np.frombuffer(raw, dtype=np.float64).reshape(-1, 3)
    return pts


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_sequence(frames: list, encoder: PointNetEncoder,
                     device: torch.device, frame_batch: int) -> np.ndarray:
    """
    frames : list of T np.ndarray, each shape (N_t, D), N_t variable
    Returns: (T, out_dim) float32
    """
    T   = len(frames)
    D   = frames[0].shape[1] if len(frames) > 0 else encoder.mlp[0].in_channels
    out = np.zeros((T, encoder.out_dim), dtype=np.float32)

    for start in range(0, T, frame_batch):
        batch_frames = frames[start:start + frame_batch]
        B = len(batch_frames)

        # Determine max N in this sub-batch
        ns    = [f.shape[0] for f in batch_frames]
        max_n = max(ns) if max(ns) > 0 else 1  # avoid (B, D, 0) tensor

        # Build padded tensor (B, D, max_n) and boolean mask (B, max_n)
        pts_t  = torch.zeros(B, D, max_n, dtype=torch.float32, device=device)
        mask_t = torch.zeros(B, max_n, dtype=torch.bool, device=device)
        for i, (f, n) in enumerate(zip(batch_frames, ns)):
            if n > 0:
                pts_t[i, :, :n]  = torch.from_numpy(f.astype(np.float32)).T
                mask_t[i, :n]    = True
            # if n == 0: all zeros, mask stays False → output will be zeros

        with torch.no_grad():
            feat = encoder(pts_t, mask_t)           # (B, out_dim)

        out[start:start + B] = feat.cpu().numpy()

    return out


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract(data_root: Path, feat_root: Path, out_dim: int,
            frame_batch: int, device: torch.device,
            modalities: list = ('mmwave', 'lidar')):

    enc_mm = PointNetEncoder(in_dim=5, out_dim=out_dim).to(device).eval()
    enc_li = PointNetEncoder(in_dim=3, out_dim=out_dim).to(device).eval()
    torch.manual_seed(42)   # reproducible random init

    for mod in modalities:
        out_dir = feat_root / f'{mod}_d{out_dim}'
        out_dir.mkdir(parents=True, exist_ok=True)
        encoder   = enc_mm if mod == 'mmwave' else enc_li
        read_fn   = _read_mmwave_frame if mod == 'mmwave' else _read_lidar_frame
        ext       = '.bin'
        n_cols    = 5 if mod == 'mmwave' else 3

        total = len(ALL_SUBJECTS) * len(ALL_ACTIONS)
        done  = 0
        print(f'\n=== {mod} → {out_dir} ===')

        for sid in ALL_SUBJECTS:
            zip_path = data_root / 'Zipfiles' / f'S{sid:02d}.zip'
            if not zip_path.exists():
                print(f'  [SKIP] {zip_path} not found')
                continue

            with zipfile.ZipFile(str(zip_path)) as zf:
                for aid in ALL_ACTIONS:
                    out_file = out_dir / f'S{sid:02d}_A{aid:02d}.npy'
                    if out_file.exists():
                        done += 1
                        continue

                    frame_names = _list_frames(zf, sid, aid, mod, ext)
                    if not frame_names:
                        # Modality missing for this action — save zeros
                        np.save(str(out_file),
                                np.zeros((T_FRAMES, out_dim), dtype=np.float32))
                        done += 1
                        continue

                    frames = [read_fn(zf, fn) for fn in frame_names]

                    # Pad missing trailing frames with empty point clouds
                    while len(frames) < T_FRAMES:
                        frames.append(np.zeros((0, n_cols), dtype=np.float64))

                    feat = _encode_sequence(frames, encoder, device, frame_batch)
                    np.save(str(out_file), feat)

                    done += 1
                    if done % 50 == 0 or done == total:
                        print(f'  {done}/{total}  last: S{sid:02d}_A{aid:02d}  '
                              f'shape={feat.shape}  '
                              f'n_pts_sample={[f.shape[0] for f in frames[:5]]}')

        print(f'  {mod} done — {done}/{total} files saved to {out_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Pre-extract MM-Fi mmwave/lidar point cloud features with PointNet.')
    ap.add_argument('--data-root',   default='/home/group/maestro_visual/data/MM-Fi')
    ap.add_argument('--feat-root',   default='/home/group/maestro_visual/data/MM-Fi/features')
    ap.add_argument('--out-dim',     type=int, default=128)
    ap.add_argument('--frame-batch', type=int, default=64,
                    help='Frames processed per PointNet forward pass')
    ap.add_argument('--device',      default='cuda:0')
    ap.add_argument('--modalities',  nargs='+', default=['mmwave', 'lidar'],
                    choices=['mmwave', 'lidar'])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  out_dim={args.out_dim}  frame_batch={args.frame_batch}')

    extract(
        data_root   = Path(args.data_root),
        feat_root   = Path(args.feat_root),
        out_dim     = args.out_dim,
        frame_batch = args.frame_batch,
        device      = device,
        modalities  = args.modalities,
    )
    print('\nExtraction complete.')


if __name__ == '__main__':
    main()
