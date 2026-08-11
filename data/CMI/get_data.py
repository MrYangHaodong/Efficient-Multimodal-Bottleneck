"""
CMI (Child Mind Institute) — Detect Behavior with Sensor Data Dataloader.

18-class gesture recognition, 81 subjects, 8,151 sequences, ~70 frames/seq.

Modalities (7 total):
  imu    : acc_x/y/z + rot_w/x/y/z → 7-d per frame
  thm    : thm_1..thm_5             → 5-d per frame
  tof_1  : tof_1_v0..v63            → 64-d per frame
  tof_2  : tof_2_v0..v63            → 64-d per frame
  tof_3  : tof_3_v0..v63            → 64-d per frame
  tof_4  : tof_4_v0..v63            → 64-d per frame
  tof_5  : tof_5_v0..v63            → 64-d per frame

Missing ToF values are encoded as -1.0 in the CSV → replaced with 0.0.

Splits (3-fold cross-subject, 81 subjects sorted alphabetically):
  Subject groups: A=[0:27], B=[27:54], C=[54:81]
  fold_id=0: test=A, val=B[0:9],  train=B[9:]+C
  fold_id=1: test=B, val=C[0:9],  train=A+C[9:]
  fold_id=2: test=C, val=A[0:9],  train=A[9:]+B

__getitem__ returns:
  [imu_high, imu_low, thm_high, thm_low, ..., tof_5_high, tof_5_low, label]
  (14 tensors + 1 label = 15 elements, interleaved high/low per modality)

Usage:
  from data.CMI.get_data import get_dataloader, FEATURE_DIMS, NUM_CLASSES, CMIDataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CMI_ROOT = Path('/home/group/maestro_visual/data/CMI')
TRAIN_CSV = CMI_ROOT / 'train.csv'

NUM_CLASSES = 18

_IMU_COLS  = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
_THM_COLS  = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
_TOF_COLS  = {f'tof_{i}': [f'tof_{i}_v{j}' for j in range(64)] for i in range(1, 6)}

_MOD_COLS = {
    'imu':   _IMU_COLS,
    'thm':   _THM_COLS,
    **_TOF_COLS,
}

FEATURE_DIMS: Dict[str, int] = {
    'imu':   7,
    'thm':   5,
    'tof_1': 64,
    'tof_2': 64,
    'tof_3': 64,
    'tof_4': 64,
    'tof_5': 64,
}

# 18 gesture classes in sorted order (alphabetically)
GESTURE_NAMES: List[str] = [
    'Above ear - pull hair',
    'Cheek - pinch skin',
    'Drink from bottle/cup',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Feel around in tray and pull out an object',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Glasses on/off',
    'Neck - pinch skin',
    'Neck - scratch',
    'Pinch knee/leg skin',
    'Pull air toward your face',
    'Scratch knee/leg skin',
    'Text on phone',
    'Wave hello',
    'Write name in air',
    'Write name on leg',
]
GESTURE_TO_IDX: Dict[str, int] = {g: i for i, g in enumerate(GESTURE_NAMES)}


# ---------------------------------------------------------------------------
# Split definitions (computed from sorted subject list at import time)
# ---------------------------------------------------------------------------

def _build_splits(subjects: List[str]) -> Dict[int, Dict[str, List[str]]]:
    """Given sorted subject list, return 3-fold cross-subject split defs."""
    S = sorted(subjects)
    A, B, C = S[0:27], S[27:54], S[54:81]
    return {
        0: {'test': A,          'val': B[0:9],  'train': B[9:] + C},
        1: {'test': B,          'val': C[0:9],  'train': A + C[9:]},
        2: {'test': C,          'val': A[0:9],  'train': A[9:] + B},
    }


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def _cache_path(csv_path: Path) -> Path:
    """Cache file keyed on the CSV's size+mtime so it auto-invalidates on change."""
    st = csv_path.stat()
    tag = f'{st.st_size}_{int(st.st_mtime)}'
    cache_dir = csv_path.parent / 'cache'
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f'seq_data_{tag}.pkl'


def _load_and_preprocess_csv(csv_path: Path) -> Dict[str, dict]:
    """Load train.csv, preprocess, and return per-sequence dict.

    The 1.1 GB CSV is parsed once and the resulting per-sequence dict is pickled
    to <CMI_ROOT>/cache/seq_data_<size>_<mtime>.pkl. Subsequent runs load the
    pickle (seconds) instead of re-parsing the CSV (~1-2 min). Set the env var
    CMI_NO_CACHE=1 to force a fresh parse.

    Returns dict:
      seq_data[seq_id] = {
          'subject': str,
          'label':   int,
          'imu':     (T, 7)  float32,
          'thm':     (T, 5)  float32,
          'tof_1':   (T, 64) float32,
          ...
      }
    """
    import os
    import pickle
    import time

    use_cache = os.environ.get('CMI_NO_CACHE', '0') != '1'
    cache = _cache_path(csv_path) if use_cache else None
    if cache is not None and cache.exists():
        t0 = time.time()
        with open(cache, 'rb') as f:
            seq_data = pickle.load(f)
        print(f'  Loaded CMI cache {cache.name} '
              f'({len(seq_data)} sequences) in {time.time()-t0:.1f}s', flush=True)
        return seq_data

    print(f'  Loading CMI CSV from {csv_path} …', flush=True)
    df = pd.read_csv(str(csv_path))
    print(f'  Loaded {len(df)} rows, {len(df.columns)} columns.', flush=True)

    # Replace ToF missing-value sentinel (-1.0) with 0.0
    tof_all = [c for c in df.columns if c.startswith('tof_')]
    df[tof_all] = df[tof_all].replace(-1.0, 0.0)

    # Fill any remaining NaN with 0
    df = df.fillna(0.0)

    # Map gesture string → label int
    df['label'] = df['gesture'].map(GESTURE_TO_IDX)
    # Drop rows with unmapped gesture (safety)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    seq_data: Dict[str, dict] = {}
    for seq_id, grp in df.groupby('sequence_id', sort=False):
        grp = grp.sort_values('sequence_counter')
        subject = grp['subject'].iloc[0]
        label   = int(grp['label'].iloc[0])
        entry = {'subject': subject, 'label': label}
        for mod, cols in _MOD_COLS.items():
            entry[mod] = grp[cols].values.astype(np.float32)
        seq_data[str(seq_id)] = entry

    print(f'  Built {len(seq_data)} sequences.', flush=True)

    if cache is not None:
        tmp = cache.with_suffix('.pkl.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(seq_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)   # atomic; safe under concurrent folds
        print(f'  Wrote CMI cache -> {cache}', flush=True)
    return seq_data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CMIDataset(Dataset):
    """CMI gesture recognition dataset — 7 modalities, 18 classes."""

    ALL_MODALITIES = ['imu', 'thm', 'tof_1', 'tof_2', 'tof_3', 'tof_4', 'tof_5']

    def __init__(
        self,
        seq_data: Dict[str, dict],
        splits:   Dict[int, Dict[str, List[str]]],
        split:    str = 'train',
        split_id: int = 0,
        modalities: List[str] = None,
        max_seq_len: int = 0,
        time_compression_ratio: int = 1,
        normalize: bool = True,
    ):
        self.seq_data = seq_data
        self.modalities = modalities if modalities is not None else list(self.ALL_MODALITIES)
        self.max_seq_len = max_seq_len
        self.time_compression_ratio = max(1, int(time_compression_ratio))
        self.normalize = normalize

        fold = split_id if split_id in splits else 0
        subject_set = set(splits[fold][split])

        self.samples = [
            (seq_id, entry['label'])
            for seq_id, entry in seq_data.items()
            if entry['subject'] in subject_set
        ]

    # ------------------------------------------------------------------
    # Processing helpers
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
        pad = np.zeros((self.max_seq_len - T, feat.shape[1]), dtype=np.float32)
        return np.concatenate([feat, pad], axis=0)

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
        seq_id, label = self.samples[idx]
        entry = self.seq_data[seq_id]
        features = []
        for m in self.modalities:
            raw  = entry[m]             # (T, D)
            feat = self._process(raw)   # (max_seq_len, D) or (T, D)
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

def collate_fn(batch):
    n_features = len(batch[0]) - 1
    feature_lists = [[] for _ in range(n_features)]
    labels = []
    for sample in batch:
        for i in range(n_features):
            feature_lists[i].append(sample[i])
        labels.append(sample[-1])
    padded = [pad_sequence(fl, batch_first=True) for fl in feature_lists]
    label_tensor = torch.tensor(labels, dtype=torch.long).view(-1, 1)
    return tuple(padded) + (label_tensor,)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root: str = str(CMI_ROOT),
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
    """Train/val/test DataLoaders for CMI gesture recognition."""
    csv_path = Path(data_root) / 'train.csv'
    if modalities is None:
        modalities = list(CMIDataset.ALL_MODALITIES)

    fold = split_id if split_id is not None else 0
    print(f'\nLoading CMI (split_id={fold}) from {data_root}')
    print(f'  Modalities  : {modalities}')
    print(f'  Feature dims: {{{", ".join(f"{m}:{FEATURE_DIMS[m]}" for m in modalities)}}}')
    print(f'  num_classes={NUM_CLASSES}  max_seq_len={max_seq_len}  '
          f'time_compression_ratio={time_compression_ratio}')

    seq_data = _load_and_preprocess_csv(csv_path)
    subjects  = list({entry['subject'] for entry in seq_data.values()})
    splits    = _build_splits(subjects)

    ds_kwargs = dict(
        seq_data=seq_data, splits=splits, split_id=fold, modalities=modalities,
        max_seq_len=max_seq_len, time_compression_ratio=time_compression_ratio,
        normalize=normalize)

    train_ds = CMIDataset(**ds_kwargs, split='train')
    val_ds   = CMIDataset(**ds_kwargs, split='val')
    test_ds  = CMIDataset(**ds_kwargs, split='test')

    print(f'  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)} sequences')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False,
                              collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def get_feature_dims(modalities: List[str] = None) -> Dict[str, int]:
    if modalities is None:
        modalities = list(CMIDataset.ALL_MODALITIES)
    return {m: FEATURE_DIMS[m] for m in modalities}


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CMI dataloader smoke test')
    parser.add_argument('--data-root', default=str(CMI_ROOT))
    parser.add_argument('--split-id', type=int, default=0)
    parser.add_argument('--max-seq-len', type=int, default=128)
    parser.add_argument('--modalities', nargs='+', default=None)
    args = parser.parse_args()

    print('=' * 60); print('CMI dataloader smoke test'); print('=' * 60)
    mods = args.modalities or list(CMIDataset.ALL_MODALITIES)
    tl, vl, el = get_dataloader(
        data_root=args.data_root, batch_size=4, num_workers=0,
        modalities=mods, split_id=args.split_id, max_seq_len=args.max_seq_len,
        time_compression_ratio=4)
    batch = next(iter(tl))
    feats = batch[:-1]; labels = batch[-1]
    TENSOR_NAMES = [f'{m}_{s}' for m in mods for s in ('high', 'low')]
    for name, t in zip(TENSOR_NAMES, feats):
        print(f'   {name:20s}: {tuple(t.shape)}  dtype={t.dtype}')
    print(f'   {"label":20s}: {tuple(labels.shape)}  values={labels.flatten().tolist()}')
    print('\nDone.')
