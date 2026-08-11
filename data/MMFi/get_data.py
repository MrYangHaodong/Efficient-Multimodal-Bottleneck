"""
MM-Fi Dataset Loader — 5-Modality HAR.

MM-Fi: Multi-Modal Non-Invasive 4D Human Dataset (NeurIPS 2023).
27-class action recognition, 40 subjects, cross-subject 3-fold CV.

Modalities
----------
  infra1   : 17 body keypoints (u,v) from IR camera  →  (T, 34)
  wifi     : WiFi-CSI amplitude (3, 114, 10) per frame, official
             NaN-interp + per-frame min-max norm, flattened →  (T, 3420)
  mmwave   : pre-extracted PointNet features (extract_point_features.py)
             →  (T, POINT_OUT_DIM)
  lidar    : pre-extracted PointNet features (extract_point_features.py)
             →  (T, POINT_OUT_DIM)
  depth    : depth map 16-bit PNG × 0.001 m, flattened  →  (T, H*W)
             (expensive; only use if needed)

Point features must be extracted before training:
    cd Efficient-Multimodal-Bottleneck
    python data/MMFi/extract_point_features.py \
        --feat-root /home/group/maestro_visual/data/MM-Fi/features \
        --out-dim 128 --device cuda:0

Cross-subject splits (40 subjects, 3-fold, non-overlapping test sets)
----------------------------------------------------------------------
  fold 0 : train S14-S39  val S40  test S01-S13
  fold 1 : train S01-S13+S27-S39  val S40  test S14-S26
  fold 2 : train S01-S12+S14-S26  val S13  test S27-S40
"""

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MMFI_ROOT     = Path('/home/group/maestro_visual/data/MM-Fi')
FEAT_ROOT     = MMFI_ROOT / 'features'
POINT_OUT_DIM  = 128    # must match --out-dim used during extraction
DEPTH_OUT_DIM  = 1024   # DINOv2 ViT-L CLS token dim

NUM_CLASSES = 27

FEATURE_DIMS: Dict[str, int] = {
    'infra1':  34,            # 17 joints × 2
    'wifi':    3420,          # 3 × 114 × 10 flattened
    'mmwave':  POINT_OUT_DIM,
    'lidar':   POINT_OUT_DIM,
    'depth':   DEPTH_OUT_DIM,
}

_SPLITS: Dict[int, Dict[str, List[int]]] = {
    0: {
        'train': list(range(14, 40)),
        'val':   [40],
        'test':  list(range(1, 14)),
    },
    1: {
        'train': list(range(1, 14)) + list(range(27, 40)),
        'val':   [40],
        'test':  list(range(14, 27)),
    },
    2: {
        'train': list(range(1, 13)) + list(range(14, 27)),
        'val':   [13],
        'test':  list(range(27, 41)),
    },
}

_ALL_MODALITIES = ['infra1', 'wifi', 'mmwave', 'lidar', 'depth']


# ---------------------------------------------------------------------------
# Zip-based loaders (infra1, wifi, depth)
# ---------------------------------------------------------------------------

def _zip_path(data_root: Path, subject: int) -> Path:
    return data_root / 'Zipfiles' / f'S{subject:02d}.zip'


def _sorted_frames(zf: zipfile.ZipFile, prefix: str, ext: str) -> list:
    return sorted(
        [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(ext)],
        key=lambda x: int(x.rsplit('frame', 1)[1].split('.')[0])
    )


def _load_infra1(zf: zipfile.ZipFile, subject: int, action: int) -> np.ndarray:
    """(T, 34) float32 — 17 joint (u,v) keypoints from IR camera."""
    prefix = f'S{subject:02d}/A{action:02d}/infra1/'
    frames = []
    for fn in _sorted_frames(zf, prefix, '.npy'):
        arr = np.load(io.BytesIO(zf.read(fn)))   # (17, 2)
        frames.append(arr.flatten().astype(np.float32))
    return np.stack(frames)  # (T, 34)


def _load_wifi(zf: zipfile.ZipFile, subject: int, action: int) -> np.ndarray:
    """(T, 3420) float32 — WiFi-CSI with official NaN-interp + per-frame min-max."""
    prefix = f'S{subject:02d}/A{action:02d}/wifi-csi/'
    frames = []
    for fn in _sorted_frames(zf, prefix, '.mat'):
        amp = sio.loadmat(io.BytesIO(zf.read(fn)))['CSIamp']  # (3, 114, 10)
        amp[np.isinf(amp)] = np.nan
        # NaN interpolation per sub-step (official preprocessing)
        for i in range(10):
            col = amp[:, :, i]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col[nan_mask] = col[~nan_mask].mean()
        lo, hi = amp.min(), amp.max()
        if hi > lo:
            amp = (amp - lo) / (hi - lo)
        else:
            amp = np.zeros_like(amp)
        frames.append(amp.flatten().astype(np.float32))  # 3420
    return np.stack(frames)  # (T, 3420)


# ---------------------------------------------------------------------------
# Pre-extracted point feature loader (mmwave, lidar)
# ---------------------------------------------------------------------------

def _load_point_feat(feat_root: Path, modality: str, subject: int,
                     action: int) -> np.ndarray:
    """Load (T, POINT_OUT_DIM) float32 from pre-extracted npy."""
    path = feat_root / f'{modality}_d{POINT_OUT_DIM}' / f'S{subject:02d}_A{action:02d}.npy'
    if not path.exists():
        raise FileNotFoundError(
            f'{path}\nRun extract_point_features.py --out-dim {POINT_OUT_DIM} first.'
        )
    return np.load(str(path))  # (297, POINT_OUT_DIM)


def _load_depth_feat(feat_root: Path, subject: int, action: int) -> np.ndarray:
    """Load (T, DEPTH_OUT_DIM) float32 from pre-extracted DINOv2 npy."""
    path = feat_root / f'depth_d{DEPTH_OUT_DIM}' / f'S{subject:02d}_A{action:02d}.npy'
    if not path.exists():
        raise FileNotFoundError(
            f'{path}\nRun extract_depth_features.py first.'
        )
    return np.load(str(path))  # (297, DEPTH_OUT_DIM)


# ---------------------------------------------------------------------------
# Subject preloader
# ---------------------------------------------------------------------------

def _subject_cache_path(feat_root: Path, subject: int, modalities: List[str]) -> Path:
    """Per-subject cache keyed on the modality list (order-independent)."""
    mod_tag = '-'.join(sorted(modalities))
    cache_dir = feat_root / 'preload_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'subj{subject:02d}_{mod_tag}.pkl'


def _preload_subject(
    data_root: Path,
    feat_root: Path,
    subject: int,
    modalities: List[str],
) -> List[Tuple]:
    """Returns list of (feat_mod0, feat_mod1, ..., label) for all 27 actions.

    Decoding per-frame .mat/.npy out of the per-subject zip (especially wifi's
    ~297 scipy.io.loadmat calls per action) dominates startup, so the parsed
    per-subject sample list is pickled to <FEAT_ROOT>/preload_cache/ and reused.
    Set MMFI_NO_CACHE=1 to force a fresh decode.
    """
    import os
    import pickle

    use_cache = os.environ.get('MMFI_NO_CACHE', '0') != '1'
    cache = _subject_cache_path(feat_root, subject, modalities) if use_cache else None
    if cache is not None and cache.exists():
        with open(cache, 'rb') as f:
            return pickle.load(f)

    zip_src_mods  = [m for m in modalities if m in ('infra1', 'wifi', 'depth')]
    point_mods    = [m for m in modalities if m in ('mmwave', 'lidar')]
    samples = []

    # Open zip only if needed
    if zip_src_mods:
        zf_ctx = zipfile.ZipFile(str(_zip_path(data_root, subject)))
    else:
        zf_ctx = None

    try:
        for action in range(1, 28):
            feats = []
            for mod in modalities:
                if mod == 'infra1':
                    feats.append(_load_infra1(zf_ctx, subject, action))
                elif mod == 'wifi':
                    feats.append(_load_wifi(zf_ctx, subject, action))
                elif mod in ('mmwave', 'lidar'):
                    feats.append(_load_point_feat(feat_root, mod, subject, action))
                elif mod == 'depth':
                    feats.append(_load_depth_feat(feat_root, subject, action))
                else:
                    raise ValueError(f'Unsupported modality: {mod}')
            samples.append(tuple(feats) + (action - 1,))
    finally:
        if zf_ctx is not None:
            zf_ctx.close()

    if cache is not None:
        tmp = cache.with_suffix('.pkl.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)   # atomic; safe under concurrent folds
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MMFiDataset(Dataset):
    """MM-Fi 5-modality (infra1 / wifi / mmwave / lidar) HAR, 27 classes."""

    ALL_MODALITIES = _ALL_MODALITIES

    def __init__(
        self,
        data_root: str = str(MMFI_ROOT),
        feat_root: str = str(FEAT_ROOT),
        split: str = 'train',
        split_id: int = 0,
        modalities: List[str] = None,
        time_compression_ratio: int = 1,
        normalize: bool = True,
    ):
        self.data_root = Path(data_root)
        self.feat_root = Path(feat_root)
        self.modalities = modalities if modalities is not None else list(_ALL_MODALITIES)
        self.time_compression_ratio = max(1, int(time_compression_ratio))
        self.normalize = normalize

        fold = split_id if split_id in _SPLITS else 0
        subject_ids = _SPLITS[fold][split]

        print(f'  Preloading MMFi {split} subjects (fold {fold}): {subject_ids}')
        self.samples: List[Tuple] = []
        for sid in subject_ids:
            self.samples.extend(
                _preload_subject(self.data_root, self.feat_root, sid, self.modalities)
            )

    # ------------------------------------------------------------------
    # Processing (same pattern as other datasets in this repo)
    # ------------------------------------------------------------------

    def _normalize(self, feat: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return feat
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        return feat / (norms.max() + 1e-8)

    def _compress_time(self, feat: np.ndarray) -> np.ndarray:
        if self.time_compression_ratio <= 1:
            return feat
        return feat[::self.time_compression_ratio]

    def _process(self, raw: np.ndarray) -> np.ndarray:
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self._normalize(raw)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List:
        *feats, label = self.samples[idx]
        out = []
        for raw in feats:
            feat = self._process(raw)
            out.append(torch.FloatTensor(feat))
            out.append(torch.FloatTensor(self._compress_time(feat)))
        return out + [int(label)]

    @property
    def feature_dims(self) -> Dict[str, int]:
        return {m: FEATURE_DIMS[m] for m in self.modalities}

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    n_feats = len(batch[0]) - 1
    feat_lists = [[] for _ in range(n_feats)]
    labels = []
    for sample in batch:
        for i in range(n_feats):
            feat_lists[i].append(sample[i])
        labels.append(sample[-1])
    padded = [pad_sequence(fl, batch_first=True) for fl in feat_lists]
    return tuple(padded) + (torch.tensor(labels, dtype=torch.long).view(-1, 1),)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root: str = str(MMFI_ROOT),
    feat_root: str = str(FEAT_ROOT),
    batch_size: int = 32,
    num_workers: int = 0,
    modalities: List[str] = None,
    split_id: int = 0,
    train_shuffle: bool = True,
    time_compression_ratio: int = 1,
    normalize: bool = True,
    **kwargs,                       # absorb unused seqA kwargs gracefully
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if modalities is None:
        modalities = list(_ALL_MODALITIES)
    fold = split_id if split_id is not None else 0

    print(f'\nLoading MMFi (fold={fold}) from {data_root}')
    print(f'  Modalities  : {modalities}')
    print(f'  Feature dims: {{{", ".join(f"{m}:{FEATURE_DIMS[m]}" for m in modalities)}}}')

    ds_kwargs = dict(
        data_root=data_root, feat_root=feat_root, split_id=fold,
        modalities=modalities, time_compression_ratio=time_compression_ratio,
        normalize=normalize,
    )
    train_ds = MMFiDataset(**ds_kwargs, split='train')
    val_ds   = MMFiDataset(**ds_kwargs, split='val')
    test_ds  = MMFiDataset(**ds_kwargs, split='test')
    print(f'  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)} samples')

    ldr = dict(batch_size=batch_size, collate_fn=collate_fn,
               num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=train_shuffle, drop_last=True,  **ldr),
        DataLoader(val_ds,   shuffle=False,         drop_last=False, **ldr),
        DataLoader(test_ds,  shuffle=False,         drop_last=False, **ldr),
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=str(MMFI_ROOT))
    ap.add_argument('--feat-root', default=str(FEAT_ROOT))
    ap.add_argument('--split-id', type=int, default=0)
    ap.add_argument('--modalities', nargs='+', default=None)
    args = ap.parse_args()

    mods = args.modalities or list(_ALL_MODALITIES)
    t0 = time.time()
    tl, vl, el = get_dataloader(
        data_root=args.data_root, feat_root=args.feat_root,
        batch_size=4, modalities=mods, split_id=args.split_id,
        time_compression_ratio=4,
    )
    print(f'Preload: {time.time()-t0:.1f}s')

    batch = next(iter(tl))
    feats, label = batch[:-1], batch[-1]
    names = [f'{m}_{s}' for m in mods for s in ('high', 'low')]
    for name, t in zip(names, feats):
        print(f'  {name:24s}: {tuple(t.shape)}')
    print(f'  {"label":24s}: {tuple(label.shape)}  {label.flatten().tolist()}')
