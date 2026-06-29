"""
CMU-MOSEI Dataset Loader — 3-Modality Sentiment Classification.

Drop-in counterpart of ``data/IEMOCAP/get_data.py``: exposes the *exact*
same public symbols (``get_dataloader``, ``FEATURE_DIMS``, ``NUM_CLASSES``,
``CMUMOSEIDataset`` with an ``ALL_MODALITIES`` attribute, ``get_feature_dims``)
and emits batches in the same layout, so every ``main_*_iemocap*.py`` training
script can be ported to MOSEI by swapping only the data import + a few CLI
defaults.

Modalities
----------
vision : FACET 4.2 facial features      (50, 35)   — fixed length
audio  : COVAREP acoustic features      (50, 74)   — fixed length
text   : GloVe 300d word embeddings     (50, 300)  — fixed length

The released ``mosei_senti_data.pkl`` (MultiBench / CMU-MultimodalSDK aligned
pack) is already word-aligned and zero-padded/truncated to T=50 for every
modality, so all three modalities share a common time axis natively.

Label
-----
Continuous sentiment in [-3, +3].  Following the standard MOSEI ``Acc-2``
convention used by MultiBench (``affect/get_data.py``), it is binarised to a
2-class label::

    sentiment >  0  -> 1  (positive)
    sentiment <= 0  -> 0  (non-positive; neutral 0.0 folded into class 0)

This makes MOSEI fit the CrossEntropy + accuracy + macro-F1 loop the
``clean_models`` scripts already use for IEMOCAP, with ``NUM_CLASSES = 2``.

Data splits
-----------
``mosei_senti_data.pkl`` ships the canonical ``train`` / ``valid`` / ``test``
partition.  To feed the 3-fold aggregation machinery the scripts inherit from
the HAR / IEMOCAP pipeline, ``split_id`` is interpreted as:

    split_id 0  -> canonical train / valid / test (comparable to literature)
    split_id 1  -> test held fixed; (train + valid) re-partitioned, seed 1235
    split_id 2  -> test held fixed; (train + valid) re-partitioned, seed 1236

The test set is *always* the canonical MOSEI test split, so the reported
``best_test_acc`` stays comparable across folds; only the train/val partition
(hence model selection) varies.  Pass ``split_id=0`` (the default) for a single
canonical run.

__getitem__ returns
-------------------
[vision, vision_low, audio, audio_low, text, text_low, label]
  vision     : FloatTensor (T, 35)       — full T (= max_seq_len)
  vision_low : FloatTensor (T//r, 35)    — time-compressed (selector path)
  audio      : FloatTensor (T, 74)
  audio_low  : FloatTensor (T//r, 74)
  text       : FloatTensor (T, 300)
  text_low   : FloatTensor (T//r, 300)
  label      : int  (0/1)  for task='classification'
               float       for task='regression'
where r = time_compression_ratio (default 1 = no compression).  Each modality
is returned twice (high + low) exactly like ``IEMOCAPDataset`` so the model
wrappers that read ``feats[2*j]`` (high) work unchanged.

Batch collation pads variable-length modalities to the longest T in the batch
via ``pad_sequence(batch_first=True)`` (a no-op here since every sample shares
T, but kept for parity with the IEMOCAP collate).
"""

import os
import pickle
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# numpy>=2.0 vs <2.0 pickle compat (mirrors IEMOCAP/get_data.py): pkl files
# saved under numpy>=2.0 reference ``numpy._core.*``; alias every submodule
# pickle might resolve so ``pickle.load`` works under numpy<2.0 too.
for _sub in ('numeric', 'multiarray', 'umath', 'numerictypes',
             '_methods', '_dtype', '_internal'):
    fq = f'numpy._core.{_sub}'
    if fq not in sys.modules and hasattr(getattr(np, 'core', None), _sub):
        sys.modules[fq] = getattr(np.core, _sub)

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default location of the aligned sentiment pack.  Either a directory holding
# ``mosei_senti_data.pkl`` or the .pkl file itself is accepted by get_dataloader.
MOSEI_ROOT = Path('/files1/haodong/data/CMU-MOSI/CMU-MOSEI')
MOSEI_PKL_NAME = 'mosei_senti_data.pkl'

NUM_CLASSES = 2   # default task = binary Acc-2 (sentiment > 0 vs <= 0)

# Supported classification granularities (CMU-MOSEI standard) -> #classes.
# 'regression' predicts the raw [-3,3] sentiment (handled separately).
TASK_NUM_CLASSES = {
    'classification':  2,   # alias of Acc-2
    'classification2': 2,   # Acc-2: sign of sentiment
    'classification5': 5,   # Acc-5: clip(round(x), -2, 2) + 2
    'classification7': 7,   # Acc-7: clip(round(x), -3, 3) + 3
}


def get_num_classes(task: str = 'classification') -> int:
    """Number of output classes for a given classification task name."""
    if task == 'regression':
        return 1
    if task not in TASK_NUM_CLASSES:
        raise ValueError(f'Unknown task {task!r}; '
                         f'choose from {list(TASK_NUM_CLASSES)} or "regression"')
    return TASK_NUM_CLASSES[task]

FEATURE_DIMS: Dict[str, int] = {
    'vision':  35,   # FACET 4.2 facial action / landmark features
    'audio':   74,   # COVAREP acoustic features
    'text':   300,   # GloVe 840B 300d word embeddings
}

# Re-split seeds for split_id 1 / 2 (split_id 0 = canonical, no re-split).
_RESPLIT_BASE_SEED = 1234


# ---------------------------------------------------------------------------
# Pickle loading + per-split preprocessing helpers
# ---------------------------------------------------------------------------

def _resolve_pkl_path(data_root: str) -> Path:
    """Accept either the .pkl file path or a directory containing it."""
    p = Path(data_root)
    if p.is_file():
        return p
    cand = p / MOSEI_PKL_NAME
    if cand.is_file():
        return cand
    # Fall back to the default location.
    default = MOSEI_ROOT / MOSEI_PKL_NAME
    if default.is_file():
        return default
    raise FileNotFoundError(
        f'Could not locate {MOSEI_PKL_NAME}. Tried: {p}, {cand}, {default}')


def _drop_empty_text(split: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Drop entries whose text is all-zero (matches MultiBench drop_entry)."""
    text = split['text']
    keep = np.where(np.abs(text).sum(axis=(1, 2)) != 0)[0]
    if len(keep) == len(text):
        return split
    return {k: v[keep] for k, v in split.items()}


def _clean_split(split: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Make a float32 copy, replace audio -inf with 0, nan->0, drop empty text."""
    out = {}
    for k in ('vision', 'audio', 'text'):
        arr = np.asarray(split[k], dtype=np.float32).copy()
        arr[np.isinf(arr)] = 0.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out[k] = arr
    out['labels'] = np.asarray(split['labels'], dtype=np.float32).reshape(-1)
    out = _drop_empty_text(out)
    return out


def _load_splits(pkl_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with open(pkl_path, 'rb') as f:
        alldata = pickle.load(f)
    return {sp: _clean_split(alldata[sp]) for sp in ('train', 'valid', 'test')}


def _resplit_trainval(train: Dict[str, np.ndarray],
                      valid: Dict[str, np.ndarray],
                      seed: int) -> Tuple[Dict, Dict]:
    """Pool train+valid and re-partition into new train/valid with the same
    validation size, keeping determinism via ``seed``.  Test is untouched by
    the caller."""
    n_val = len(valid['labels'])
    pooled = {k: np.concatenate([train[k], valid[k]], axis=0)
              for k in train.keys()}
    n = len(pooled['labels'])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    new_train = {k: v[train_idx] for k, v in pooled.items()}
    new_valid = {k: v[val_idx] for k, v in pooled.items()}
    return new_train, new_valid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CMUMOSEIDataset(Dataset):
    """CMU-MOSEI 3-modality sentiment dataset.

    Args mirror ``IEMOCAPDataset`` so the dataloader factory signatures match.

    Args:
        split_data:             Pre-loaded split dict with keys
                                ``vision`` (N,T,35), ``audio`` (N,T,74),
                                ``text`` (N,T,300), ``labels`` (N,).
        modalities:             Subset of ['vision','audio','text'] (default all).
        max_seq_len:            Resample/pad every modality to this T.
                                <=0 keeps native T (=50).
        time_compression_ratio: Uniform subsampling factor for the low path.
        use_batched_fusion:     Zero-pad short sequences to max_seq_len.
        normalize:              Per-sample max-L2 normalisation (IEMOCAP-style).
        task:                   'classification' (binary, default) or 'regression'.
    """

    ALL_MODALITIES = ['vision', 'audio', 'text']

    def __init__(
        self,
        split_data: Dict[str, np.ndarray],
        modalities: List[str] = None,
        max_seq_len: int = 50,
        time_compression_ratio: int = 1,
        use_batched_fusion: bool = True,
        normalize: bool = True,
        task: str = 'classification',
    ):
        self.data = split_data
        self.modalities = modalities if modalities is not None else list(self.ALL_MODALITIES)
        self.max_seq_len = max_seq_len
        self.time_compression_ratio = max(1, int(time_compression_ratio))
        self.use_batched_fusion = use_batched_fusion
        self.normalize = normalize
        self.task = task

    # ------------------------------------------------------------------
    # Feature transforms (mirror IEMOCAPDataset)
    # ------------------------------------------------------------------

    def _normalize(self, feat: np.ndarray) -> np.ndarray:
        """Max L2-norm normalisation across time: (T, C) -> (T, C)."""
        if not self.normalize:
            return feat
        norms = np.linalg.norm(feat, axis=1, keepdims=True)  # (T, 1)
        max_norm = norms.max()
        return feat / (max_norm + 1e-8)

    def _resample(self, feat: np.ndarray) -> np.ndarray:
        """Resample toward max_seq_len.

        T > max_seq_len: linear-interpolate down.
        T < max_seq_len: zero-pad (when use_batched_fusion) else leave as-is.
        No-op when max_seq_len <= 0 or T == max_seq_len.
        """
        if self.max_seq_len <= 0:
            return feat
        T = feat.shape[0]
        if T == self.max_seq_len:
            return feat
        if T > self.max_seq_len:
            x = torch.FloatTensor(feat).unsqueeze(0).permute(0, 2, 1)  # (1,C,T)
            x = F.interpolate(x, size=self.max_seq_len, mode='linear',
                              align_corners=False)
            return x.permute(0, 2, 1).squeeze(0).numpy()
        if self.use_batched_fusion:
            pad = np.zeros((self.max_seq_len - T, feat.shape[1]), dtype=np.float32)
            return np.concatenate([feat, pad], axis=0)
        return feat

    def _compress_time(self, feat: np.ndarray) -> np.ndarray:
        """Uniformly subsample tokens: (T, C) -> (T // ratio, C)."""
        if self.time_compression_ratio <= 1:
            return feat
        return feat[::self.time_compression_ratio]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data['labels'])

    def __getitem__(self, idx: int) -> List:
        features = []
        for m in self.modalities:
            feat = self.data[m][idx]                       # (T, C) float32
            feat = self._resample(self._normalize(feat))
            features.append(torch.FloatTensor(feat))                       # high
            features.append(torch.FloatTensor(self._compress_time(feat)))  # low

        raw = float(self.data['labels'][idx])
        if self.task == 'regression':
            label = torch.tensor(raw, dtype=torch.float32)
        elif self.task in ('classification', 'classification2'):
            label = int(raw > 0)                              # Acc-2: >0 vs <=0
        elif self.task == 'classification5':
            label = int(np.clip(np.round(raw), -2, 2)) + 2    # Acc-5: 0..4
        elif self.task == 'classification7':
            label = int(np.clip(np.round(raw), -3, 3)) + 3    # Acc-7: 0..6
        else:
            raise ValueError(f'Unknown task {self.task!r}')
        return features + [label]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_dims(self) -> Dict[str, int]:
        return {m: FEATURE_DIMS[m] for m in self.modalities}

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def _collate_features(batch):
    """Split features and labels, pad variable-length tensors (parity w/ IEMOCAP)."""
    n_features = len(batch[0]) - 1   # last element is label
    feature_lists = [[] for _ in range(n_features)]
    labels = []
    for sample in batch:
        for i in range(n_features):
            feature_lists[i].append(sample[i])
        labels.append(sample[-1])
    padded = [pad_sequence(fl, batch_first=True) for fl in feature_lists]
    return padded, labels


def collate_fn(batch):
    """Classification collate: label tensor (B,1) int64."""
    padded, labels = _collate_features(batch)
    label_tensor = torch.tensor(labels, dtype=torch.long).view(-1, 1)
    return tuple(padded) + (label_tensor,)


def collate_fn_regression(batch):
    """Regression collate: label tensor (B,1) float32."""
    padded, labels = _collate_features(batch)
    label_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return tuple(padded) + (label_tensor,)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root: str = str(MOSEI_ROOT),
    batch_size: int = 32,
    num_workers: int = 4,
    modalities: List[str] = None,
    split_id: int = 0,
    train_shuffle: bool = True,
    max_seq_len: int = 50,
    time_compression_ratio: int = 1,
    use_batched_fusion: bool = True,
    available_sessions: Optional[List[int]] = None,   # ignored (IEMOCAP-only)
    split_mode: str = 'random',                       # ignored (MOSEI single split)
    normalize: bool = True,
    task: str = 'classification',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train / val / test DataLoaders for CMU-MOSEI.

    Signature matches ``data/IEMOCAP/get_data.get_dataloader`` so model scripts
    port over by swapping only the import.  ``available_sessions`` and
    ``split_mode`` are accepted for API parity and ignored (MOSEI ships a single
    canonical split).  See module docstring for how ``split_id`` is interpreted.
    """
    if modalities is None:
        modalities = list(CMUMOSEIDataset.ALL_MODALITIES)

    pkl_path = _resolve_pkl_path(data_root)
    splits = _load_splits(pkl_path)
    train_d, valid_d, test_d = splits['train'], splits['valid'], splits['test']

    if split_id and split_id > 0:
        train_d, valid_d = _resplit_trainval(
            train_d, valid_d, seed=_RESPLIT_BASE_SEED + split_id)

    print(f'\nLoading CMU-MOSEI (split_id={split_id}) from {pkl_path}')
    print(f'  Modalities  : {modalities}')
    print(f'  Feature dims: {{{", ".join(f"{m}:{FEATURE_DIMS[m]}" for m in modalities)}}}')
    print(f'  task={task}  num_classes={get_num_classes(task)}  max_seq_len={max_seq_len}  '
          f'time_compression_ratio={time_compression_ratio}  normalize={normalize}')
    print(f'  Train: {len(train_d["labels"])}  Val: {len(valid_d["labels"])}  '
          f'Test: {len(test_d["labels"])} samples')

    ds_kwargs = dict(
        modalities=modalities,
        max_seq_len=max_seq_len,
        time_compression_ratio=time_compression_ratio,
        use_batched_fusion=use_batched_fusion,
        normalize=normalize,
        task=task,
    )
    train_dataset = CMUMOSEIDataset(train_d, **ds_kwargs)
    val_dataset   = CMUMOSEIDataset(valid_d, **ds_kwargs)
    test_dataset  = CMUMOSEIDataset(test_d, **ds_kwargs)

    collate = collate_fn_regression if task == 'regression' else collate_fn

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate)

    return train_loader, val_loader, test_loader


def get_feature_dims(modalities: List[str] = None) -> Dict[str, int]:
    """Return feature dimensions for the requested modalities."""
    if modalities is None:
        modalities = list(CMUMOSEIDataset.ALL_MODALITIES)
    return {m: FEATURE_DIMS[m] for m in modalities}


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CMU-MOSEI dataloader utility')
    parser.add_argument('--data-root', type=str, default=str(MOSEI_ROOT))
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--max-seq-len', type=int, default=50)
    parser.add_argument('--time-compression-ratio', type=int, default=4)
    args = parser.parse_args()

    print('=' * 60)
    print('CMU-MOSEI dataloader smoke test')
    print('=' * 60)
    MOD_NAMES = ['vision', 'audio', 'text']
    TENSOR_NAMES = [f'{m}_{suf}' for m in MOD_NAMES for suf in ('high', 'low')]
    for split_id in range(3):
        print(f'\n--- split_id {split_id} ---')
        try:
            train_loader, val_loader, test_loader = get_dataloader(
                data_root=args.data_root, batch_size=4, num_workers=0,
                split_id=split_id, max_seq_len=args.max_seq_len,
                time_compression_ratio=args.time_compression_ratio,
            )
            batch = next(iter(train_loader))
            feats, labels = batch[:-1], batch[-1]
            for name, t in zip(TENSOR_NAMES, feats):
                print(f'   {name:14s}: {tuple(t.shape)}  dtype={t.dtype}')
            print(f'   {"label":14s}: {tuple(labels.shape)}  dtype={labels.dtype}  '
                  f'values={labels.flatten().tolist()}')
        except Exception as exc:
            import traceback
            print(f'   ERROR: {exc}')
            traceback.print_exc()
    print('\nDone.')
