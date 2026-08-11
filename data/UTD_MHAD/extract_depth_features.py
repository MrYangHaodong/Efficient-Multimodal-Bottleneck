"""
Pre-extract UTD-MHAD depth features using DINOv2 (dinov3_vitl16).

Reads (240, 320, T) uint16 depth .mat files, encodes each frame as a
DINOv2 CLS token, and saves (T, feature_dim) float32 npy arrays.

Output:
  {feat_root}/depth_d{feat_dim}/
    a1_s1_t1.npy  ...    # (T, feat_dim) float32, variable T per sample

Depth preprocessing (matching MM-Fi pipeline):
  - uint16 × 0.001 → metres, clip [0.1, 6.0] m, normalize [0,1]
  - Tile grayscale to 3 channels, resize to image_size × image_size
  - ImageNet mean/std normalization
  - DINOv2 ViT-L CLS token → (feat_dim,) per frame

Usage:
  cd /home/group/maestro_visual/Efficient-Multimodal-Bottleneck
  conda run -n maestro python data/UTD_MHAD/extract_depth_features.py \
      --data-root /home/group/maestro_visual/data/UTD-MHAD \
      --feat-root /home/group/maestro_visual/data/UTD-MHAD/features \
      --device cuda:3 --batch-size 64 --image-size 224
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_DALIA = Path('/home/group/maestro_visual/MAESTRO/data_utils/dalia_preprocess')
DINOV3_REPO  = _DALIA / 'dinov3'
DINOV3_CKPT  = _DALIA / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOV3_MODEL = 'dinov3_vitl16'

UTD_ROOT  = Path('/home/group/maestro_visual/data/UTD-MHAD')
FEAT_ROOT = UTD_ROOT / 'features'

DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 6.0
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# numpy._core compatibility shim (scipy.io needs this on numpy 2.x)
# ---------------------------------------------------------------------------
for _sub in ('numeric', 'multiarray', 'umath', 'numerictypes',
             '_methods', '_dtype', '_internal'):
    fq = f'numpy._core.{_sub}'
    if fq not in sys.modules and hasattr(np.core, _sub):
        sys.modules[fq] = getattr(np.core, _sub)

import scipy.io as sio


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dinov3(device: torch.device):
    model = torch.hub.load(str(DINOV3_REPO), DINOV3_MODEL, source='local',
                           pretrained=str(DINOV3_CKPT) if DINOV3_CKPT.exists() else True)
    model.to(device).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224, device=device))
    feat_dim = (out if isinstance(out, torch.Tensor) else
                out.get('x_norm_clstoken', next(iter(out.values())))).shape[-1]
    return model, feat_dim


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_depth_frame(frame_uint16: np.ndarray, image_size: int) -> np.ndarray:
    """(H, W) uint16 → (3, image_size, image_size) float32, ImageNet-normed."""
    depth_m = frame_uint16.astype(np.float32) * 0.001
    depth_m = np.clip(depth_m, DEPTH_MIN_M, DEPTH_MAX_M)
    depth_n = (depth_m - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
    img = Image.fromarray((depth_n * 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=0)                     # (3, H, W)
    arr = (arr - IMG_MEAN[:, None, None]) / IMG_STD[:, None, None]
    return arr


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_file(mat_path: Path, model, feat_dim: int, image_size: int,
                 batch_size: int, device: torch.device) -> np.ndarray:
    mat   = sio.loadmat(str(mat_path))
    depth = np.asarray(mat['d_depth'])          # (H, W, T)
    T     = depth.shape[2]

    frames = [preprocess_depth_frame(depth[:, :, t], image_size) for t in range(T)]
    out    = np.zeros((T, feat_dim), dtype=np.float32)

    for start in range(0, T, batch_size):
        batch_np = np.stack(frames[start:start + batch_size])   # (B, 3, H, W)
        batch_t  = torch.from_numpy(batch_np).to(device)
        with torch.no_grad():
            result = model(batch_t)
        feat = (result if isinstance(result, torch.Tensor) else
                result.get('x_norm_clstoken',
                result.get('cls_token',
                result['x_norm_patchtokens'].mean(1))))
        out[start:start + len(batch_np)] = feat.cpu().float().numpy()

    return out   # (T, feat_dim)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract(data_root: Path, feat_root: Path, image_size: int,
            batch_size: int, device: torch.device):
    print(f'Loading {DINOV3_MODEL} …')
    model, feat_dim = load_dinov3(device)
    print(f'  feat_dim={feat_dim}  image_size={image_size}  batch_size={batch_size}')

    out_dir = feat_root / f'depth_d{feat_dim}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Output → {out_dir}')

    depth_files = sorted((data_root / 'Depth').glob('*_depth.mat'))
    total = len(depth_files)
    print(f'  Files: {total}')

    for i, mat_path in enumerate(depth_files):
        stem     = mat_path.stem.replace('_depth', '')
        out_file = out_dir / f'{stem}.npy'
        if out_file.exists():
            continue

        feat = extract_file(mat_path, model, feat_dim, image_size, batch_size, device)
        np.save(str(out_file), feat)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f'  {i+1}/{total}  {stem}  shape={feat.shape}')

    print(f'\nDepth extraction complete — {total} files in {out_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root',   default=str(UTD_ROOT))
    ap.add_argument('--feat-root',   default=str(FEAT_ROOT))
    ap.add_argument('--device',      default='cuda:3')
    ap.add_argument('--batch-size',  type=int, default=64)
    ap.add_argument('--image-size',  type=int, default=224)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    extract(Path(args.data_root), Path(args.feat_root),
            args.image_size, args.batch_size, device)


if __name__ == '__main__':
    main()
