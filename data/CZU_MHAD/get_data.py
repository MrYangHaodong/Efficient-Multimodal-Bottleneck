"""
CZU-MHAD Dataset Loader — 7-Modality Cross-Subject HAR.

Dataset
-------
CZU-MHAD (China Zhejiang University Multimodal Human Action Dataset,
Li et al.) — 5 subjects × 22 actions × multiple trials.  Pre-extracted:

    depth_features/{subject}/{file_stem}.pkl  → dict, features (T, 1024)
    skeleton_mat/{file_stem}.mat              → {'skeleton': (T, 100)}
    sensor_mat/{file_stem}.mat               → {'sensor': (10,1) object}

Sensor: 10 LPMS-B2 IMUs, each 7-ch (accel xyz + gyro xyz + timestamp).
Grouped into 5 body-part segments (2 IMUs × 7-ch = 14-d each):
  sensor_head  : IMU 0, 1
  sensor_torso : IMU 2, 3
  sensor_rarm  : IMU 4, 5
  sensor_larm  : IMU 6, 7
  sensor_leg   : IMU 8, 9

Modalities (7 total):
  depth        : 1024-d  (DINOv2 ViT-L, pre-extracted per frame)
  skeleton     : 100-d   (joint coordinates, raw from mat)
  sensor_head  : 14-d    (head IMUs 0+1)
  sensor_torso : 14-d    (torso IMUs 2+3)
  sensor_rarm  : 14-d    (right-arm IMUs 4+5)
  sensor_larm  : 14-d    (left-arm IMUs 6+7)
  sensor_leg   : 14-d    (leg IMUs 8+9)

Labels
------
22-class HAR.

Splits (cross-subject, 3 folds, 5 subjects: cx / cyy / myj / qyh / zyh)
-------------------------------------------------------------------------
  fold_id=0 : test=zyh,  val=qyh,  train=cx+cyy+myj
  fold_id=1 : test=myj,  val=cyy,  train=cx+qyh+zyh
  fold_id=2 : test=qyh,  val=cx,   train=cyy+myj+zyh

__getitem__ returns
-------------------
[depth_high, depth_low, skeleton_high, skeleton_low,
 sensor_head_high, sensor_head_low, sensor_torso_high, sensor_torso_low,
 sensor_rarm_high, sensor_rarm_low, sensor_larm_high, sensor_larm_low,
 sensor_leg_high,  sensor_leg_low,  label]
(14 tensors + 1 label = 15 elements, interleaved high/low per modality)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

for _sub in ('numeric', 'multiarray', 'umath', 'numerictypes',
             '_methods', '_dtype', '_internal'):
    fq = f'numpy._core.{_sub}'
    if fq not in sys.modules and hasattr(np.core, _sub):
        sys.modules[fq] = getattr(np.core, _sub)

import pickle
import torch
import torch.nn.functional as F
import scipy.io as sio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CZUMHAD_ROOT = Path('/home/group/maestro_visual/data/CZU-MHAD')

NUM_CLASSES = 22

# 2 IMUs × 7 channels per body-part group
_SENSOR_GROUPS: Dict[str, Tuple[int, int]] = {
    'sensor_head':  (0, 1),
    'sensor_torso': (2, 3),
    'sensor_rarm':  (4, 5),
    'sensor_larm':  (6, 7),
    'sensor_leg':   (8, 9),
}

FEATURE_DIMS: Dict[str, int] = {
    'depth':        1024,
    'skeleton':     100,
    'sensor_head':  14,
    'sensor_torso': 14,
    'sensor_rarm':  14,
    'sensor_larm':  14,
    'sensor_leg':   14,
}

# Cross-subject 3-fold splits
# Subjects: cx (286), cyy (220), myj (220), qyh (241), zyh (198) samples
_SPLITS: Dict[int, Dict[str, List[str]]] = {
    0: {'test': ['zyh'],  'val': ['qyh'],  'train': ['cx', 'cyy', 'myj']},
    1: {'test': ['myj'],  'val': ['cyy'],  'train': ['cx', 'qyh', 'zyh']},
    2: {'test': ['qyh'],  'val': ['cx'],   'train': ['cyy', 'myj', 'zyh']},
    3: {'test': ['cx'],   'val': ['zyh'],  'train': ['cyy', 'myj', 'qyh']},
    4: {'test': ['cyy'],  'val': ['myj'],  'train': ['cx', 'qyh', 'zyh']},
}


def _load_all_samples(data_root: Path) -> pd.DataFrame:
    """Combine train/val/test CSVs into a single sample table."""
    dfs = []
    for stem in ['train', 'val', 'test']:
        p = data_root / f'{stem}.csv'
        if p.exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True).drop_duplicates('file_stem')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CZUMHADDataset(Dataset):
    """CZU-MHAD 7-modality cross-subject 22-class HAR dataset."""

    ALL_MODALITIES = ['depth', 'skeleton',
                      'sensor_head', 'sensor_torso', 'sensor_rarm',
                      'sensor_larm', 'sensor_leg']

    def __init__(
        self,
        data_root: str = str(CZUMHAD_ROOT),
        split: str = 'train',
        split_id: int = 0,
        modalities: List[str] = None,
        max_seq_len: int = 0,
        time_compression_ratio: int = 1,
        use_batched_fusion: bool = True,
        available_sessions: Optional[List[int]] = None,
        normalize: bool = True,
        _all_samples_df: pd.DataFrame = None,
    ):
        self.data_root = Path(data_root)
        self.modalities = modalities if modalities is not None else list(self.ALL_MODALITIES)
        self.max_seq_len = max_seq_len
        self.time_compression_ratio = max(1, int(time_compression_ratio))
        self.use_batched_fusion = use_batched_fusion
        self.normalize = normalize

        fold = split_id if split_id in _SPLITS else 0
        subject_set = set(_SPLITS[fold][split])

        df = _all_samples_df if _all_samples_df is not None else _load_all_samples(self.data_root)
        subset = df[df['subject'].isin(subject_set)]
        self.samples = subset[['file_stem', 'subject', 'label']].values.tolist()

    # ------------------------------------------------------------------
    # Loaders per modality
    # ------------------------------------------------------------------

    @staticmethod
    def _pkl_load(path: Path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return torch.load(path, map_location='cpu', weights_only=False)

    def _load_depth(self, file_stem: str, subject: str) -> np.ndarray:
        path = self.data_root / 'depth_features' / subject / f'{file_stem}.pkl'
        raw = self._pkl_load(path)
        feat = raw['features'] if isinstance(raw, dict) else raw
        if hasattr(feat, 'detach'):
            feat = feat.detach().cpu().numpy()
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
        return feat   # (T, 1024)

    def _load_skeleton(self, file_stem: str) -> np.ndarray:
        path = self.data_root / 'skeleton_mat' / f'{file_stem}.mat'
        mat = sio.loadmat(str(path))
        skel = np.asarray(mat['skeleton'], dtype=np.float32)  # (T, 100)
        if skel.ndim == 1:
            skel = skel.reshape(1, -1)
        return skel   # (T, 100)

    def _load_sensor_group(self, file_stem: str, imu_a: int, imu_b: int) -> np.ndarray:
        """Load 2 IMU streams, align to stream-imu_a's time length, concat → (T, 14)."""
        path = self.data_root / 'sensor_mat' / f'{file_stem}.mat'
        mat = sio.loadmat(str(path))
        raw = mat['sensor']   # (10, 1) object array; each element (T_i, 7)

        def _get_stream(i: int) -> np.ndarray:
            s = np.asarray(raw[i, 0], dtype=np.float32)
            return s.reshape(-1, 1) if s.ndim == 1 else s

        sa = _get_stream(imu_a)
        sb = _get_stream(imu_b)
        T_ref = sa.shape[0]

        if sb.shape[0] != T_ref:
            x_old = np.linspace(0, 1, sb.shape[0], dtype=np.float32)
            x_new = np.linspace(0, 1, T_ref, dtype=np.float32)
            sb = np.stack([np.interp(x_new, x_old, sb[:, c])
                           for c in range(sb.shape[1])], axis=1).astype(np.float32)

        return np.concatenate([sa, sb], axis=1)   # (T_ref, 14)

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
        file_stem, subject, label = self.samples[idx]

        # Load sensor mat once, reuse for all 5 sensor-group modalities
        _sensor_mat_cache: Dict[str, np.ndarray] = {}

        features = []
        for m in self.modalities:
            if m == 'depth':
                raw = self._load_depth(file_stem, subject)
            elif m == 'skeleton':
                raw = self._load_skeleton(file_stem)
            elif m in _SENSOR_GROUPS:
                imu_a, imu_b = _SENSOR_GROUPS[m]
                raw = self._load_sensor_group(file_stem, imu_a, imu_b)
            else:
                raise ValueError(f'Unknown modality: {m!r}')
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
    data_root: str = str(CZUMHAD_ROOT),
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
    """Train/val/test DataLoaders for CZU-MHAD (cross-subject)."""
    if modalities is None:
        modalities = list(CZUMHADDataset.ALL_MODALITIES)
    data_root_p = Path(data_root)
    fold = split_id if split_id is not None else 0

    print(f'\nLoading CZU-MHAD cross-subject (fold={fold}) from {data_root_p}')
    sp = _SPLITS[fold]
    print(f'  train={sp["train"]}  val={sp["val"]}  test={sp["test"]}')
    print(f'  Modalities  : {modalities}')
    print(f'  Feature dims: {{{", ".join(f"{m}:{FEATURE_DIMS[m]}" for m in modalities)}}}')
    print(f'  num_classes={NUM_CLASSES}  max_seq_len={max_seq_len}  '
          f'time_compression_ratio={time_compression_ratio}')

    all_df = _load_all_samples(data_root_p)

    ds_kwargs = dict(
        data_root=str(data_root_p), split_id=fold, modalities=modalities,
        max_seq_len=max_seq_len, time_compression_ratio=time_compression_ratio,
        use_batched_fusion=use_batched_fusion, available_sessions=available_sessions,
        normalize=normalize, _all_samples_df=all_df)

    train_dataset = CZUMHADDataset(**ds_kwargs, split='train')
    val_dataset   = CZUMHADDataset(**ds_kwargs, split='val')
    test_dataset  = CZUMHADDataset(**ds_kwargs, split='test')

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
        modalities = list(CZUMHADDataset.ALL_MODALITIES)
    return {m: FEATURE_DIMS[m] for m in modalities}


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CZU-MHAD cross-subject dataloader smoke test')
    parser.add_argument('--data-root', default=str(CZUMHAD_ROOT))
    parser.add_argument('--split-id', type=int, default=0)
    parser.add_argument('--max-seq-len', type=int, default=100)
    parser.add_argument('--modalities', nargs='+', default=None)
    args = parser.parse_args()

    print('=' * 60); print('CZU-MHAD cross-subject dataloader smoke test'); print('=' * 60)
    mods = args.modalities or list(CZUMHADDataset.ALL_MODALITIES)
    tl, vl, el = get_dataloader(
        data_root=args.data_root, batch_size=4, num_workers=0,
        modalities=mods, split_id=args.split_id, max_seq_len=args.max_seq_len,
        time_compression_ratio=4)
    batch = next(iter(tl)); feats = batch[:-1]; labels = batch[-1]
    TENSOR_NAMES = [f'{m}_{s}' for m in mods for s in ('high', 'low')]
    for name, t in zip(TENSOR_NAMES, feats):
        print(f'   {name:25s}: {tuple(t.shape)}  dtype={t.dtype}')
    print(f'   {"label":25s}: {tuple(labels.shape)}  values={labels.flatten().tolist()}')
    print('\nDone.')
