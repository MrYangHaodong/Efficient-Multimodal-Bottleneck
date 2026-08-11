"""
Pre-extract UTD-MHAD RGB features using DINOv2 (dinov3_vitl16).

Reads *_color.avi files (640×480 MJPEG, 15 fps, ~40-80 frames), encodes
each frame as a DINOv2 CLS token, and saves (T, feature_dim) float32 arrays.

Output:
  {feat_root}/rgb_d{feat_dim}/
    a1_s1_t1.npy  ...    # (T, feat_dim) float32, variable T per sample

RGB preprocessing (matching emognition pipeline):
  - Read all frames from AVI via torchvision.io.read_video
  - Resize to image_size × image_size
  - ImageNet mean/std normalization
  - DINOv2 ViT-L CLS token → (feat_dim,) per frame

Usage:
  cd /home/group/maestro_visual/Efficient-Multimodal-Bottleneck
  conda run -n maestro python data/UTD_MHAD/extract_rgb_features.py \
      --data-root /home/group/maestro_visual/data/UTD-MHAD \
      --feat-root /home/group/maestro_visual/data/UTD-MHAD/features \
      --device cuda:4 --batch-size 64 --image-size 224
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video

_DALIA = Path('/home/group/maestro_visual/MAESTRO/data_utils/dalia_preprocess')
DINOV3_REPO  = _DALIA / 'dinov3'
DINOV3_CKPT  = _DALIA / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOV3_MODEL = 'dinov3_vitl16'

UTD_ROOT  = Path('/home/group/maestro_visual/data/UTD-MHAD')
FEAT_ROOT = UTD_ROOT / 'features'

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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

def preprocess_frames(frames_uint8: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    frames_uint8 : (T, H, W, 3) uint8  from torchvision read_video
    Returns      : (T, 3, image_size, image_size) float32, ImageNet-normed
    """
    # (T, H, W, 3) → (T, 3, H, W), float [0,1]
    x = frames_uint8.permute(0, 3, 1, 2).float() / 255.0
    # Resize
    x = F.interpolate(x, size=(image_size, image_size),
                      mode='bilinear', align_corners=False)
    # Normalize
    x = (x - IMG_MEAN) / IMG_STD
    return x   # (T, 3, H, W)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_file(avi_path: Path, model, feat_dim: int, image_size: int,
                 batch_size: int, device: torch.device) -> np.ndarray:
    frames, _, _ = read_video(str(avi_path), pts_unit='sec', output_format='THWC')
    # frames: (T, H, W, 3) uint8
    if frames.shape[0] == 0:
        return np.zeros((1, feat_dim), dtype=np.float32)

    frames_proc = preprocess_frames(frames, image_size)   # (T, 3, H, W) float32
    T   = frames_proc.shape[0]
    out = np.zeros((T, feat_dim), dtype=np.float32)

    for start in range(0, T, batch_size):
        batch_t = frames_proc[start:start + batch_size].to(device)
        with torch.no_grad():
            result = model(batch_t)
        feat = (result if isinstance(result, torch.Tensor) else
                result.get('x_norm_clstoken',
                result.get('cls_token',
                result['x_norm_patchtokens'].mean(1))))
        out[start:start + len(batch_t)] = feat.cpu().float().numpy()

    return out   # (T, feat_dim)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract(data_root: Path, feat_root: Path, image_size: int,
            batch_size: int, device: torch.device):
    print(f'Loading {DINOV3_MODEL} …')
    model, feat_dim = load_dinov3(device)
    print(f'  feat_dim={feat_dim}  image_size={image_size}  batch_size={batch_size}')

    out_dir = feat_root / f'rgb_d{feat_dim}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Output → {out_dir}')

    rgb_files = sorted((data_root / 'RGB').glob('*_color.avi'))
    total = len(rgb_files)
    print(f'  Files: {total}')

    for i, avi_path in enumerate(rgb_files):
        stem     = avi_path.stem.replace('_color', '')
        out_file = out_dir / f'{stem}.npy'
        if out_file.exists():
            continue

        feat = extract_file(avi_path, model, feat_dim, image_size, batch_size, device)
        np.save(str(out_file), feat)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f'  {i+1}/{total}  {stem}  shape={feat.shape}')

    print(f'\nRGB extraction complete — {total} files in {out_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root',   default=str(UTD_ROOT))
    ap.add_argument('--feat-root',   default=str(FEAT_ROOT))
    ap.add_argument('--device',      default='cuda:4')
    ap.add_argument('--batch-size',  type=int, default=64)
    ap.add_argument('--image-size',  type=int, default=224)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    extract(Path(args.data_root), Path(args.feat_root),
            args.image_size, args.batch_size, device)


if __name__ == '__main__':
    main()
