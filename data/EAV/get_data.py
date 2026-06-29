"""
EAV Dataset Loader — 3-Modality (EEG / Audio / Video) Emotion Classification.

Drop-in counterpart of ``data/IEMOCAP/get_data.py``: exposes the *exact* same
public symbols (``get_dataloader``, ``FEATURE_DIMS``, ``NUM_CLASSES``,
``EAVDataset`` with an ``ALL_MODALITIES`` attribute, ``get_feature_dims``,
``generate_splits``) and emits batches in the same layout, so every
``main_*_iemocap*.py`` training script ports to EAV by swapping only the data
import + a few CLI defaults.

Dataset (EAV: EEG-Audio-Video, Lee et al., Scientific Data 2024)
---------------------------------------------------------------
42 subjects, 5 balanced emotions, 200 trials/subject (100 *Listening* + 100
*Speaking*, interleaved). Pre-extracted features on disk per subject::

    subjectN/eeg_features/{NNN}_Trial_{TT}_{Listening|Speaking}_{Emotion}.pkl
    subjectN/video_features/{NNN}_Trial_{TT}_{Listening|Speaking}_{Emotion}.pkl
    subjectN/audio_features/{NNN}_..._{Emotion}_{Aud|aud}_audio_features.pkl

Modalities
----------
eeg   : raw 30-channel EEG               (10000, 30)  float64  — every trial
video : DINOv3 ViT-L/16 frame features   (100, 1024)  float32  — every trial
audio : WAV2VEC2_BASE frame features     (999, 768)   float32  — *Speaking only*

Audio is recorded **only while the subject speaks**, so it is present for the
100 Speaking trials and ABSENT for the 100 Listening trials. A trial is kept
only if every *requested* modality is available; hence the default trimodal
(eeg+audio+video) configuration uses the ~100 Speaking trials/subject (≈4200
total), whereas dropping audio (e.g. eeg+video) yields all 200/subject (≈8400).

Label
-----
5-class emotion classification (alphabetical-by-id fixed order)::

    0 = Neutral   1 = Anger   2 = Happiness   3 = Sadness   4 = Calmness

Data splits
-----------
``split_mode='cross_subject'`` (default — the natural EAV protocol): subjects
are partitioned 70/15/15 into disjoint train/val/test sets, three seeds give
splits 0/1/2 (CSV suffix ``_cs_split{0,1,2}``).
``split_mode='random'`` : trial-level stratified 70/15/15 (suffix ``''`` /
``_split{1,2}``), comparable to IEMOCAP's random splits.

__getitem__ returns
-------------------
[eeg, eeg_low, audio, audio_low, video, video_low, label]  (modalities in
``self.modalities`` order; each modality returned twice — high full-T and low
time-compressed; label last int 0-4), exactly like ``IEMOCAPDataset``.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# numpy>=2.0 vs <2.0 pickle compat (mirrors IEMOCAP/MOSEI get_data.py): pkl files
# saved under numpy>=2.0 reference ``numpy._core.*``; alias every submodule pickle
# might resolve so ``pickle.load`` works under numpy<2.0 too.
for _sub in ('numeric', 'multiarray', 'umath', 'numerictypes',
             '_methods', '_dtype', '_internal'):
    fq = f'numpy._core.{_sub}'
    if fq not in sys.modules and hasattr(np.core, _sub):
        sys.modules[fq] = getattr(np.core, _sub)

import pickle                                   # noqa: E402  (after numpy shim)
import torch                                    # noqa: E402
import torch.nn.functional as F                 # noqa: E402
from torch.nn.utils.rnn import pad_sequence     # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EAV_ROOT = Path('/files1/haodong/data/EAV')

# Fixed 5-class emotion -> label map.
EMOTION_LABEL: Dict[str, int] = {
    'Neutral': 0, 'Anger': 1, 'Happiness': 2, 'Sadness': 3, 'Calmness': 4,
}
EMOTION_NAMES = {v: k for k, v in EMOTION_LABEL.items()}
NUM_CLASSES = 5

FEATURE_DIMS: Dict[str, int] = {
    'eeg':   30,     # raw 30-channel EEG
    'audio': 768,    # WAV2VEC2_BASE frame features
    'video': 1024,   # DINOv3 ViT-L/16 frame features
}

_MODALITY_DIR = {'eeg': 'eeg_features', 'audio': 'audio_features', 'video': 'video_features'}

# Random seeds for the 3 splits (cross-subject and random share the seeds).
_SPLIT_SEEDS = [42, 123, 456]
_TRAIN_FRAC = 0.70
_VAL_FRAC = 0.15   # test gets the remaining 0.15

_FNAME_RE = re.compile(r'(\d+)_Trial_(\d+)_(Listening|Speaking)_(\w+)\.pkl')
_CSV_COLS = ['subject', 'idx', 'trial', 'ls', 'emotion', 'emotion_label',
             'eeg_path', 'audio_path', 'video_path']


# ---------------------------------------------------------------------------
# Manifest scan + split generation
# ---------------------------------------------------------------------------

def _scan_manifest(data_root: Path) -> pd.DataFrame:
    """Scan the EAV tree -> one row per trial with relative modality paths.
    ``audio_path`` is '' for Listening trials (no audio recorded)."""
    rows = []
    for sub_dir in sorted(data_root.glob('subject*'),
                          key=lambda p: int(p.name.replace('subject', ''))):
        sid = int(sub_dir.name.replace('subject', ''))
        # index -> audio filename (Speaking trials only)
        audio_by_idx = {}
        for ap in (sub_dir / 'audio_features').glob('*.pkl'):
            audio_by_idx[int(ap.name.split('_')[0])] = ap
        for ef in sorted((sub_dir / 'eeg_features').glob('*.pkl')):
            m = _FNAME_RE.match(ef.name)
            if not m:
                continue
            idx, trial, ls, emo = m.groups()
            if emo not in EMOTION_LABEL:
                continue
            idx = int(idx)
            vp = sub_dir / 'video_features' / ef.name           # same basename as eeg
            ap = audio_by_idx.get(idx)
            rows.append({
                'subject': sid, 'idx': idx, 'trial': int(trial), 'ls': ls,
                'emotion': emo, 'emotion_label': EMOTION_LABEL[emo],
                'eeg_path': str(ef.relative_to(data_root)),
                'audio_path': str(ap.relative_to(data_root)) if ap is not None else '',
                'video_path': str(vp.relative_to(data_root)) if vp.exists() else '',
            })
    df = pd.DataFrame(rows, columns=_CSV_COLS)
    print(f'Scanned {len(df)} EAV trials over {df["subject"].nunique()} subjects '
          f'(emotions: {dict(df["emotion"].value_counts())}; '
          f'with-audio: {int((df["audio_path"] != "").sum())})')
    return df


def _stratified_trial_split(df: pd.DataFrame, seed: int):
    """Trial-level stratified 70/15/15 by emotion label."""
    tr, va, te = [], [], []
    rng = np.random.RandomState(seed)
    for lab in sorted(df['emotion_label'].unique()):
        cdf = df[df['emotion_label'] == lab].reset_index(drop=True)
        idx = rng.permutation(len(cdf))
        ntr, nval = int(len(cdf) * _TRAIN_FRAC), int(len(cdf) * _VAL_FRAC)
        tr.append(cdf.iloc[idx[:ntr]]); va.append(cdf.iloc[idx[ntr:ntr + nval]])
        te.append(cdf.iloc[idx[ntr + nval:]])
    pack = lambda L: pd.concat(L).sample(frac=1, random_state=seed).reset_index(drop=True)
    return pack(tr), pack(va), pack(te)


def _subject_split(df: pd.DataFrame, seed: int):
    """Cross-subject: partition the 42 subjects 70/15/15 (disjoint)."""
    subs = np.array(sorted(df['subject'].unique()))
    perm = np.random.RandomState(seed).permutation(len(subs))
    n = len(subs); ntr, nval = int(n * _TRAIN_FRAC), int(n * _VAL_FRAC)
    tr_s = set(subs[perm[:ntr]]); val_s = set(subs[perm[ntr:ntr + nval]])
    te_s = set(subs[perm[ntr + nval:]])
    pick = lambda S: df[df['subject'].isin(S)].reset_index(drop=True)
    return pick(tr_s), pick(val_s), pick(te_s)


def generate_splits(data_root: Optional[Path] = None, verbose: bool = True) -> None:
    """Build both random (trial-level) and cross-subject split CSVs.

    random      split i -> train{suffix}.csv / val / test    (suffix ''/_split1/_split2)
    cross-subj  split i -> train_cs_split{i}.csv / val_cs_split{i} / test_cs_split{i}
    """
    data_root = Path(data_root) if data_root is not None else EAV_ROOT
    df = _scan_manifest(data_root)
    for split_idx, seed in enumerate(_SPLIT_SEEDS):
        # ---- random (trial-level) ----
        rtr, rva, rte = _stratified_trial_split(df, seed)
        suffix = '' if split_idx == 0 else f'_split{split_idx}'
        rtr.to_csv(data_root / f'train{suffix}.csv', index=False)
        rva.to_csv(data_root / f'val{suffix}.csv', index=False)
        rte.to_csv(data_root / f'test{suffix}.csv', index=False)
        # ---- cross-subject ----
        ctr, cva, cte = _subject_split(df, seed)
        ctr.to_csv(data_root / f'train_cs_split{split_idx}.csv', index=False)
        cva.to_csv(data_root / f'val_cs_split{split_idx}.csv', index=False)
        cte.to_csv(data_root / f'test_cs_split{split_idx}.csv', index=False)
        if verbose:
            print(f'Split {split_idx} (seed={seed}): '
                  f'random tr/va/te={len(rtr)}/{len(rva)}/{len(rte)} | '
                  f'cross-subj tr/va/te={len(ctr)}/{len(cva)}/{len(cte)} '
                  f'(subjects {ctr["subject"].nunique()}/{cva["subject"].nunique()}/'
                  f'{cte["subject"].nunique()})')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EAVDataset(Dataset):
    """EAV 3-modality (EEG/Audio/Video) emotion classification dataset.

    Args mirror ``IEMOCAPDataset``. A trial is kept only if EVERY requested
    modality has a feature file (so requesting 'audio' restricts to Speaking
    trials). ``available_sessions`` is reinterpreted as an optional subject
    whitelist (API parity with IEMOCAP).
    """

    ALL_MODALITIES = ['eeg', 'audio', 'video']

    def __init__(
        self,
        data_root: str = str(EAV_ROOT),
        csv_path: str = None,
        modalities: List[str] = None,
        max_seq_len: int = 0,
        time_compression_ratio: int = 1,
        use_batched_fusion: bool = True,
        available_sessions: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        self.data_root = Path(data_root)
        self.modalities = modalities if modalities is not None else list(self.ALL_MODALITIES)
        self.max_seq_len = max_seq_len
        self.time_compression_ratio = max(1, int(time_compression_ratio))
        self.use_batched_fusion = use_batched_fusion
        self.normalize = normalize

        if csv_path is None:
            csv_path = str(self.data_root / 'train.csv')
        df = pd.read_csv(csv_path).fillna({'audio_path': '', 'video_path': '', 'eeg_path': ''})

        # optional subject whitelist (API parity: available_sessions)
        if available_sessions is not None:
            avail = set(int(s) for s in available_sessions)
            df = df[df['subject'].isin(avail)].reset_index(drop=True)

        # keep only trials with EVERY requested modality present
        path_col = {'eeg': 'eeg_path', 'audio': 'audio_path', 'video': 'video_path'}
        before = len(df)
        for m in self.modalities:
            df = df[df[path_col[m]].astype(str) != ''].reset_index(drop=True)
        if len(df) != before:
            print(f'  Modalities {self.modalities}: kept {len(df)}/{before} trials '
                  f'with all requested modalities present')
        self.samples = df.to_dict('records')

    # ------------------------------------------------------------------
    # Feature loading / transforms (mirror IEMOCAPDataset)
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_load(path: Path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return torch.load(path, map_location='cpu', weights_only=False)

    def _load_features(self, path: Path) -> np.ndarray:
        """Load a modality .pkl -> float32 (T, C). Handles bare arrays (EEG) and
        ``{'features': ...}`` dicts (audio/video), squeezing a leading singleton."""
        data = self._raw_load(path)
        feat = data['features'] if isinstance(data, dict) else data
        if hasattr(feat, 'detach'):
            feat = feat.detach().cpu().numpy()
        feat = np.asarray(feat, dtype=np.float32)
        feat[np.isinf(feat)] = 0.0
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        if feat.ndim == 3 and feat.shape[0] == 1:      # (1, T, C) -> (T, C)
            feat = feat.squeeze(0)
        elif feat.ndim == 1:                            # (C,) -> (1, C)
            feat = feat.reshape(1, -1)
        return feat

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
            x = torch.FloatTensor(feat).unsqueeze(0).permute(0, 2, 1)   # (1,C,T)
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

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List:
        row = self.samples[idx]
        path_col = {'eeg': 'eeg_path', 'audio': 'audio_path', 'video': 'video_path'}
        features = []
        for m in self.modalities:
            feat = self._load_features(self.data_root / row[path_col[m]])
            feat = self._resample(self._normalize(feat))
            features.append(torch.FloatTensor(feat))                       # high
            features.append(torch.FloatTensor(self._compress_time(feat)))  # low
        return features + [int(row['emotion_label'])]

    @property
    def feature_dims(self) -> Dict[str, int]:
        return {m: FEATURE_DIMS[m] for m in self.modalities}

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES


# ---------------------------------------------------------------------------
# Collate functions (identical to IEMOCAP)
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
# Dataloader factory (signature mirrors IEMOCAP.get_dataloader)
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root: str = str(EAV_ROOT),
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
    """Train/val/test DataLoaders for EAV. Signature matches IEMOCAP's so model
    scripts port by swapping the import. ``split_mode`` defaults to
    ``'cross_subject'`` (the natural EAV protocol)."""
    if modalities is None:
        modalities = list(EAVDataset.ALL_MODALITIES)
    data_root = Path(data_root)

    if split_mode == 'cross_subject':
        suffix = f'_cs_split{split_id}'
    else:
        suffix = '' if split_id == 0 else f'_split{split_id}'
    train_csv = data_root / f'train{suffix}.csv'
    val_csv = data_root / f'val{suffix}.csv'
    test_csv = data_root / f'test{suffix}.csv'

    if not train_csv.exists():
        print(f'Split CSV not found ({train_csv}). Generating all splits...')
        generate_splits(data_root)

    print(f'\nLoading EAV (split_id={split_id}, split_mode={split_mode}) from {data_root}')
    print(f'  Modalities  : {modalities}')
    print(f'  Feature dims: {{{", ".join(f"{m}:{FEATURE_DIMS[m]}" for m in modalities)}}}')
    print(f'  num_classes={NUM_CLASSES}  max_seq_len={max_seq_len}  '
          f'time_compression_ratio={time_compression_ratio}  normalize={normalize}')

    ds_kwargs = dict(
        data_root=str(data_root), modalities=modalities, max_seq_len=max_seq_len,
        time_compression_ratio=time_compression_ratio, use_batched_fusion=use_batched_fusion,
        available_sessions=available_sessions, normalize=normalize)
    train_dataset = EAVDataset(**ds_kwargs, csv_path=str(train_csv))
    val_dataset = EAVDataset(**ds_kwargs, csv_path=str(val_csv))
    test_dataset = EAVDataset(**ds_kwargs, csv_path=str(test_csv))

    print(f'  Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)} samples')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False,
                             collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def get_feature_dims(modalities: List[str] = None) -> Dict[str, int]:
    if modalities is None:
        modalities = list(EAVDataset.ALL_MODALITIES)
    return {m: FEATURE_DIMS[m] for m in modalities}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EAV dataloader utility')
    parser.add_argument('--generate-splits', action='store_true')
    parser.add_argument('--data-root', type=str, default=str(EAV_ROOT))
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--max-seq-len', type=int, default=128)
    parser.add_argument('--modalities', nargs='+', default=None)
    parser.add_argument('--split-mode', default='cross_subject',
                        choices=['cross_subject', 'random'])
    args = parser.parse_args()

    if args.generate_splits:
        print('Generating EAV splits...')
        generate_splits(Path(args.data_root))
        print('Done.')
    elif args.smoke_test:
        print('=' * 60); print('EAV dataloader smoke test'); print('=' * 60)
        mods = args.modalities or list(EAVDataset.ALL_MODALITIES)
        TENSOR_NAMES = [f'{m}_{s}' for m in mods for s in ('high', 'low')]
        for split_id in range(3):
            print(f'\n--- {args.split_mode} split {split_id} ---')
            try:
                tl, vl, el = get_dataloader(
                    data_root=args.data_root, batch_size=4, num_workers=0,
                    modalities=mods, split_id=split_id, max_seq_len=args.max_seq_len,
                    time_compression_ratio=4, split_mode=args.split_mode)
                batch = next(iter(tl)); feats, labels = batch[:-1], batch[-1]
                for name, t in zip(TENSOR_NAMES, feats):
                    print(f'   {name:12s}: {tuple(t.shape)}  dtype={t.dtype}')
                print(f'   {"label":12s}: {tuple(labels.shape)}  values={labels.flatten().tolist()}')
            except Exception as exc:
                import traceback; print(f'   ERROR: {exc}'); traceback.print_exc()
        print('\nDone.')
