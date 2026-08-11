"""
UTD-MHAD Dataset Loader — 4-Modality (Skeleton / Inertial / Depth / RGB) HAR.

Drop-in counterpart of data/CZU_MHAD/get_data.py:
exposes get_dataloader, FEATURE_DIMS, NUM_CLASSES, UTDMHADDataset with
ALL_MODALITIES, and get_feature_dims.

Dataset
-------
UTD Multimodal Human Action Dataset (Chen et al., 2015).
27 actions × 8 subjects × 4 trials ≈ 861 valid samples.
Filename pattern: a{action}_s{subject}_t{trial}_{modality}.{ext}

Modalities
----------
  skeleton : Skeleton/a*_skeleton.mat  → d_skel (20,3,T) → (T, 60)
  inertial : Inertial/a*_inertial.mat  → d_iner (T, 6)
  depth    : pre-extracted DINOv2 features (extract_depth_features.py)
             features/depth_d1024/a*_s*_t*.npy → (T, 1024)
  rgb      : pre-extracted DINOv2 features (extract_rgb_features.py)
             features/rgb_d1024/a*_s*_t*.npy   → (T, 1024)

Run extraction before using depth/rgb:
  conda run -n maestro python data/UTD_MHAD/extract_depth_features.py --device cuda:3
  conda run -n maestro python data/UTD_MHAD/extract_rgb_features.py   --device cuda:4

Labels
------
27-class HAR.  action number a1..a27 → label 0..26.

Splits  (cross-subject)
-----------------------
  split_id=0 : train=s3,s4,s5,s6,s8  val=s7  test=s1,s2
  split_id=1 : train=s1,s2,s5,s6,s8  val=s7  test=s3,s4
  split_id=2 : train=s1,s2,s3,s4,s6  val=s5  test=s7,s8

__getitem__ returns
-------------------
[mod0_high, mod0_low, mod1_high, mod1_low, ..., label]
(order follows self.modalities; each returned twice — full-T high and
time-compressed low; label last as int).
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

for _sub in ('numeric', 'multiarray', 'umath', 'numerictypes',
             '_methods', '_dtype', '_internal'):
    fq = f'numpy._core.{_sub}'
    if fq not in sys.modules and hasattr(np.core, _sub):
        sys.modules[fq] = getattr(np.core, _sub)

import torch
import torch.nn.functional as F
import scipy.io as sio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UTD_ROOT  = Path('/home/group/maestro_visual/data/UTD-MHAD')
FEAT_ROOT = UTD_ROOT / 'features'

DINOV2_DIM = 1024   # DINOv2 ViT-L CLS token

NUM_CLASSES = 27

FEATURE_DIMS: Dict[str, int] = {
    'skeleton': 60,        # 20 joints × 3 (x,y,z)
    'inertial': 6,         # 3-axis accel + 3-axis gyro
    'depth':    DINOV2_DIM,  # DINOv2 ViT-L, pre-extracted per frame
    'rgb':      DINOV2_DIM,  # DINOv2 ViT-L, pre-extracted per frame
}

# subject sets per split — 8 subjects, 3-fold cross-subject
# each fold: 5 train / 1 val / 2 test  (~540 / ~108 / ~215 samples)
_SPLITS: Dict[int, Dict[str, List[int]]] = {
    0: {'train': [3, 4, 5, 6, 8], 'val': [7], 'test': [1, 2]},
    1: {'train': [1, 2, 5, 6, 8], 'val': [7], 'test': [3, 4]},
    2: {'train': [1, 2, 3, 4, 6], 'val': [5], 'test': [7, 8]},
}

_STEM_RE = re.compile(r'^a(\d+)_s(\d+)_t(\d+)$')


def _build_samples(data_root: Path,
                   modalities: List[str]) -> List[Tuple[str, int, int]]:
    """Enumerate (stem, label, subject) for samples present in all requested modalities."""
    ine_stems  = {f.stem.replace('_inertial', '')
                  for f in (data_root / 'Inertial').glob('*.mat')}
    feat_root  = data_root / 'features'

    samples = []
    for f in sorted((data_root / 'Skeleton').glob('*.mat')):
        stem = f.stem.replace('_skeleton', '')
        m    = _STEM_RE.match(stem)
        if m is None:
            continue
        if 'inertial' in modalities and stem not in ine_stems:
            continue
        if 'depth' in modalities:
            if not (feat_root / f'depth_d{DINOV2_DIM}' / f'{stem}.npy').exists():
                continue
        if 'rgb' in modalities:
            if not (feat_root / f'rgb_d{DINOV2_DIM}' / f'{stem}.npy').exists():
                continue
        action  = int(m.group(1))
        subject = int(m.group(2))
        samples.append((stem, action - 1, subject))
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UTDMHADDataset(Dataset):
    """UTD-MHAD 4-modality (Skeleton/Inertial/Depth/RGB) 27-class HAR dataset."""

    ALL_MODALITIES = ['skeleton', 'inertial', 'depth', 'rgb']

    def __init__(
        self,
        data_root: str = str(UTD_ROOT),
        split: str = 'train',
        split_id: int = 0,
        modalities: List[str] = None,
        max_seq_len: int = 0,
        time_compression_ratio: int = 1,
        use_batched_fusion: bool = True,
        available_sessions: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.modalities = modalities if modalities is not None else list(self.ALL_MODALITIES)
        self.max_seq_len = max_seq_len
        self.time_compression_ratio = max(1, int(time_compression_ratio))
        self.use_batched_fusion = use_batched_fusion
        self.normalize = normalize

        fold = split_id if split_id in _SPLITS else 0
        subject_set = set(_SPLITS[fold][split])

        all_samples = _build_samples(self.data_root, self.modalities)
        self.samples = [(stem, label)
                        for stem, label, subj in all_samples
                        if subj in subject_set]

    # ------------------------------------------------------------------
    # Feature loaders
    # ------------------------------------------------------------------

    def _load_skeleton(self, stem: str) -> np.ndarray:
        path = self.data_root / 'Skeleton' / f'{stem}_skeleton.mat'
        mat  = sio.loadmat(str(path))
        d    = np.asarray(mat['d_skel'], dtype=np.float32)  # (20, 3, T)
        d    = d.transpose(2, 0, 1).reshape(d.shape[2], -1) # (T, 60)
        return d

    def _load_inertial(self, stem: str) -> np.ndarray:
        path = self.data_root / 'Inertial' / f'{stem}_inertial.mat'
        mat  = sio.loadmat(str(path))
        return np.asarray(mat['d_iner'], dtype=np.float32)  # (T, 6)

    def _load_pretrained(self, stem: str, modality: str) -> np.ndarray:
        """Load pre-extracted DINOv2 features: (T, DINOV2_DIM) float32."""
        path = self.data_root / 'features' / f'{modality}_d{DINOV2_DIM}' / f'{stem}.npy'
        if not path.exists():
            raise FileNotFoundError(
                f'{path}\nRun extract_{modality}_features.py first.'
            )
        return np.load(str(path))  # (T, DINOV2_DIM)

    # ------------------------------------------------------------------
    # Processing helpers (identical pattern to CZU-MHAD)
    # ------------------------------------------------------------------

    def _to_clean(self, feat: np.ndarray) -> np.ndarray:
        feat[np.isinf(feat)] = 0.0
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize(self, feat: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return feat
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        return feat / (norms.max() + 1e-8)

    def _resample(self, feat: np.ndarray) -> np.ndarray:
        if self.max_seq_len <= 0:
            return feat
        T = feat.shape[0]
        if T == self.max_seq_len:
            return feat
        if T > self.max_seq_len:
            x = torch.FloatTensor(feat).unsqueeze(0).permute(0, 2, 1)
            x = F.interpolate(x, size=self.max_seq_len, mode='linear', align_corners=False)
            return x.permute(0, 2, 1).squeeze(0).numpy()
        if self.use_batched_fusion:
            pad = np.zeros((self.max_seq_len - T, feat.shape[1]), dtype=np.float32)
            return np.concatenate([feat, pad], axis=0)
        return feat

    def _compress_time(self, feat: np.ndarray) -> np.ndarray:
        if self.time_compression_ratio <= 1:
            return feat
        return feat[::self.time_compression_ratio]

    def _process(self, raw: np.ndarray) -> np.ndarray:
        return self._resample(self._normalize(self._to_clean(raw)))

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List:
        stem, label = self.samples[idx]
        features = []
        for m in self.modalities:
            if m == 'skeleton':
                raw = self._load_skeleton(stem)
            elif m == 'inertial':
                raw = self._load_inertial(stem)
            else:  # depth, rgb — pre-extracted DINOv2
                raw = self._load_pretrained(stem, m)
            feat = self._process(raw)
            features.append(torch.FloatTensor(feat))
            features.append(torch.FloatTensor(self._compress_time(feat)))
        return features + [int(label)]

    @property
    def feature_dims(self) -> Dict[str, int]:
        return {m: FEATURE_DIMS[m] for m in self.modalities}

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_features(batch):
    n_features = len(batch[0]) - 1
    feature_lists = [[] for _ in range(n_features)]
    labels = []
    for sample in batch:
        for i in range(n_features):
            feature_lists[i].append(sample[i])
        labels.append(sample[-1])
    padded = [pad_sequence(fl, batch_first=True) for fl in feature_lists]
    return padded, labels


def collate_fn(batch):
    padded, labels = _collate_features(batch)
    label_tensor = torch.tensor(labels, dtype=torch.long).view(-1, 1)
    return tuple(padded) + (label_tensor,)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root: str = str(UTD_ROOT),
    batch_size: int = 32,
    num_workers: int = 4,
    modalities: List[str] = None,
    split_id: int = 0,
    train_shuffle: bool = True,
    max_seq_len: int = 0,
    time_compression_ratio: int = 1,
    use_batched_fusion: bool = True,
    available_sessions: Optional[List[int]] = None,
    split_mode: str = 'cross_subject',
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train/val/test DataLoaders for UTD-MHAD."""
    if modalities is None:
        modalities = list(UTDMHADDataset.ALL_MODALITIES)
    data_root_p = Path(data_root)
    fold = split_id if split_id is not None else 0

    print(f'\nLoading UTD-MHAD (split_id={fold}) from {data_root_p}')
    print(f'  Modalities  : {modalities}')
    print(f'  Feature dims: {{{", ".join(f"{m}:{FEATURE_DIMS[m]}" for m in modalities)}}}')
    print(f'  num_classes={NUM_CLASSES}  max_seq_len={max_seq_len}  '
          f'time_compression_ratio={time_compression_ratio}  normalize={normalize}')

    ds_kwargs = dict(
        data_root=str(data_root_p), split_id=fold, modalities=modalities,
        max_seq_len=max_seq_len, time_compression_ratio=time_compression_ratio,
        use_batched_fusion=use_batched_fusion, available_sessions=available_sessions,
        normalize=normalize)

    train_dataset = UTDMHADDataset(**ds_kwargs, split='train')
    val_dataset   = UTDMHADDataset(**ds_kwargs, split='val')
    test_dataset  = UTDMHADDataset(**ds_kwargs, split='test')

    print(f'  Train: {len(train_dataset)}  Val: {len(val_dataset)}  '
          f'Test: {len(test_dataset)} samples')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False,
                              collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def get_feature_dims(modalities: List[str] = None) -> Dict[str, int]:
    if modalities is None:
        modalities = list(UTDMHADDataset.ALL_MODALITIES)
    return {m: FEATURE_DIMS[m] for m in modalities}


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UTD-MHAD dataloader smoke test')
    parser.add_argument('--data-root', default=str(UTD_ROOT))
    parser.add_argument('--split-id', type=int, default=0)
    parser.add_argument('--max-seq-len', type=int, default=64)
    parser.add_argument('--modalities', nargs='+', default=None)
    args = parser.parse_args()

    print('=' * 60); print('UTD-MHAD dataloader smoke test'); print('=' * 60)
    mods = args.modalities or list(UTDMHADDataset.ALL_MODALITIES)
    tl, vl, el = get_dataloader(
        data_root=args.data_root, batch_size=4, num_workers=0,
        modalities=mods, split_id=args.split_id, max_seq_len=args.max_seq_len,
        time_compression_ratio=4)
    batch = next(iter(tl)); feats = batch[:-1]; labels = batch[-1]
    TENSOR_NAMES = [f'{m}_{s}' for m in mods for s in ('high', 'low')]
    for name, t in zip(TENSOR_NAMES, feats):
        print(f'   {name:20s}: {tuple(t.shape)}  dtype={t.dtype}')
    print(f'   {"label":20s}: {tuple(labels.shape)}  values={labels.flatten().tolist()}')
    print('\nDone.')
