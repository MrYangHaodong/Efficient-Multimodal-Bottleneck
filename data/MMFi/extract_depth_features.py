"""
Pre-extract MM-Fi depth features using DINOv2 (dinov3_vitl16).

Reads 16-bit depth PNGs from per-subject zips, normalizes each frame,
feeds through DINOv2 ViT-L, and saves the CLS token per frame.

Output:
  {feat_root}/depth_d{feature_dim}/
    S01_A01.npy  ...  S40_A27.npy    # (297, feature_dim) float32

Depth preprocessing:
  - Load uint16 PNG × 0.001 → metres
  - Clip to [0.1, 6.0] m, normalize to [0, 1]
  - Tile to 3 channels (H, W) → (3, H, W)
  - Resize to image_size × image_size
  - ImageNet mean/std normalization (matching emognition pipeline)
  - CLS token from DINOv2 ViT-L → (feature_dim,) per frame

Usage:
  cd /home/group/maestro_visual/Efficient-Multimodal-Bottleneck
  python data/MMFi/extract_depth_features.py \
      --data-root /home/group/maestro_visual/data/MM-Fi \
      --feat-root /home/group/maestro_visual/data/MM-Fi/features \
      --device cuda:2 --batch-size 32 --image-size 224
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Paths — mirror emognition script conventions
# ---------------------------------------------------------------------------

_DALIA = Path('/home/group/maestro_visual/MAESTRO/data_utils/dalia_preprocess')
DINOV3_REPO  = _DALIA / 'dinov3'
DINOV3_CKPT  = _DALIA / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOV3_MODEL = 'dinov3_vitl16'

ALL_SUBJECTS = list(range(1, 41))
ALL_ACTIONS  = list(range(1, 28))
T_FRAMES     = 297

DEPTH_MIN_M  = 0.1    # clip below (sensor noise / invalid)
DEPTH_MAX_M  = 6.0    # clip above (room background)
IMG_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dinov3(repo_dir: Path, model_name: str, ckpt: Path,
                device: torch.device):
    if not repo_dir.exists():
        raise FileNotFoundError(f'DINOv3 repo not found: {repo_dir}')
    ckpt_arg = str(ckpt) if ckpt.exists() else True
    model = torch.hub.load(str(repo_dir), model_name,
                           source='local', pretrained=ckpt_arg)
    model.to(device).eval()

    # probe feature dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=device)
        out   = model(dummy)
    if isinstance(out, torch.Tensor):
        feat_dim = out.shape[-1]
    elif isinstance(out, dict):
        feat_dim = (out.get('x_norm_clstoken') or
                    out.get('cls_token') or
                    next(iter(out.values()))).shape[-1]
    else:
        feat_dim = 1024
    return model, feat_dim


# ---------------------------------------------------------------------------
# Depth preprocessing
# ---------------------------------------------------------------------------

def preprocess_depth(depth_uint16: np.ndarray, image_size: int) -> np.ndarray:
    """
    depth_uint16 : (H, W) uint16
    Returns      : (3, image_size, image_size) float32, ImageNet-normalised
    """
    depth_m = depth_uint16.astype(np.float32) * 0.001          # → metres
    depth_m = np.clip(depth_m, DEPTH_MIN_M, DEPTH_MAX_M)
    depth_n = (depth_m - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)  # [0,1]

    # Tile grayscale to 3 channels and resize
    img_pil = Image.fromarray((depth_n * 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.BILINEAR
    )
    arr = np.array(img_pil, dtype=np.float32) / 255.0          # [0,1]
    arr = np.stack([arr, arr, arr], axis=0)                     # (3, H, W)
    arr = (arr - IMG_MEAN[:, None, None]) / IMG_STD[:, None, None]
    return arr                                                   # (3, H, W)


# ---------------------------------------------------------------------------
# Per-action extraction
# ---------------------------------------------------------------------------

def extract_action(zf: zipfile.ZipFile, subject: int, action: int,
                   model, feat_dim: int, image_size: int,
                   batch_size: int, device: torch.device) -> np.ndarray:
    prefix = f'S{subject:02d}/A{action:02d}/depth/'
    names  = sorted(
        [n for n in zf.namelist() if n.startswith(prefix) and n.endswith('.png')],
        key=lambda x: int(x.rsplit('frame', 1)[1].split('.')[0])
    )

    if not names:
        return np.zeros((T_FRAMES, feat_dim), dtype=np.float32)

    # Preprocess all frames
    frames = []
    for nm in names:
        raw = zf.read(nm)
        img = np.array(Image.open(io.BytesIO(raw)))   # uint16 (H, W)
        frames.append(preprocess_depth(img, image_size))

    # Pad to T_FRAMES if needed
    while len(frames) < T_FRAMES:
        frames.append(np.zeros((3, image_size, image_size), dtype=np.float32))

    # Run DINOv3 in batches
    out = np.zeros((T_FRAMES, feat_dim), dtype=np.float32)
    for start in range(0, T_FRAMES, batch_size):
        batch_np = np.stack(frames[start:start + batch_size])      # (B,3,H,W)
        batch_t  = torch.from_numpy(batch_np).to(device)
        with torch.no_grad():
            result = model(batch_t)
        if isinstance(result, torch.Tensor):
            feat = result
        elif isinstance(result, dict):
            feat = (result.get('x_norm_clstoken') or
                    result.get('cls_token') or
                    result['x_norm_patchtokens'].mean(1))
        else:
            feat = torch.zeros(len(batch_np), feat_dim, device=device)
        out[start:start + len(batch_np)] = feat.cpu().float().numpy()

    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def extract(data_root: Path, feat_root: Path, image_size: int,
            batch_size: int, device: torch.device):

    print(f'Loading DINOv3 model ({DINOV3_MODEL}) …')
    model, feat_dim = load_dinov3(DINOV3_REPO, DINOV3_MODEL, DINOV3_CKPT, device)
    print(f'  feature_dim={feat_dim}  image_size={image_size}  batch_size={batch_size}')

    out_dir = feat_root / f'depth_d{feat_dim}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Output → {out_dir}')

    total = len(ALL_SUBJECTS) * len(ALL_ACTIONS)
    done  = 0

    for sid in ALL_SUBJECTS:
        zip_path = data_root / 'Zipfiles' / f'S{sid:02d}.zip'
        if not zip_path.exists():
            done += len(ALL_ACTIONS)
            continue

        with zipfile.ZipFile(str(zip_path)) as zf:
            for aid in ALL_ACTIONS:
                out_file = out_dir / f'S{sid:02d}_A{aid:02d}.npy'
                if out_file.exists():
                    done += 1
                    continue

                feat = extract_action(zf, sid, aid, model, feat_dim,
                                      image_size, batch_size, device)
                np.save(str(out_file), feat)
                done += 1

                if done % 27 == 0 or done == total:
                    print(f'  {done}/{total}  S{sid:02d}_A{aid:02d}  '
                          f'shape={feat.shape}')

    print(f'\nDepth extraction complete — {done}/{total} files in {out_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Extract DINOv2 depth features for MM-Fi')
    ap.add_argument('--data-root',   default='/home/group/maestro_visual/data/MM-Fi')
    ap.add_argument('--feat-root',   default='/home/group/maestro_visual/data/MM-Fi/features')
    ap.add_argument('--device',      default='cuda:2')
    ap.add_argument('--batch-size',  type=int, default=32)
    ap.add_argument('--image-size',  type=int, default=224)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    extract(
        data_root  = Path(args.data_root),
        feat_root  = Path(args.feat_root),
        image_size = args.image_size,
        batch_size = args.batch_size,
        device     = device,
    )


if __name__ == '__main__':
    main()
