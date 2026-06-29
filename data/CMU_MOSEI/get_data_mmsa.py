"""CMU-MOSEI MMSA-format dataloader (aligned_50 + BERT) — for the BERT-regime
pipeline, matching the DecAlign / MMSA (THUIAR) setup.

Reads `mosei_aligned_50.pkl` (MMSA-FET): keys
    text_bert            (N, 3, 50) int64   -> [input_ids, attention_mask, token_type_ids]
    text                 (N, 50, 768) float -> precomputed BERT features (use_bert=False path)
    vision               (N, 50, 35)  float -> FACET
    audio                (N, 50, 74)  float -> COVAREP
    regression_labels    (N,)         float -> sentiment in [-3, 3]
    raw_text, id, classification_labels

Returns batches as dicts: {'text_bert','vision','audio','label'}. The finetuned
BERT lives INSIDE the model wrapper (text_bert -> 768 token features); audio/vision
are per-feature standardized with TRAIN stats + percentile clipping (MMSA default
`feature_standardize_modalities=['audio','vision']`, `clip_percentiles=[0.5,99.5]`).
"""
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

MMSA_MOSEI_PKL = '/files1/haodong/data/CMU-MOSI/CMU-MOSEI/MMSA/mosei_aligned_50.pkl'
FEATURE_DIMS = {'vision': 35, 'audio': 74, 'text': 768}   # text = BERT hidden
T = 50


def _standardize(train, others, clip_pct=(0.5, 99.5)):
    """Per-feature standardize (mean/std from TRAIN) + percentile clip. Operates on
    (N,T,C) arrays; returns standardized copies for (train, *others)."""
    flat = train.reshape(-1, train.shape[-1])
    mu = flat.mean(0, keepdims=True)
    sd = flat.std(0, keepdims=True) + 1e-8
    lo = np.percentile(flat, clip_pct[0], axis=0, keepdims=True)
    hi = np.percentile(flat, clip_pct[1], axis=0, keepdims=True)

    def _apply(x):
        x = np.clip(x, lo, hi)
        return ((x - mu) / sd).astype(np.float32)
    return _apply(train), [_apply(o) for o in others]


class MMSADataset(Dataset):
    def __init__(self, split: Dict, text_mode: str = 'bert'):
        # text_mode='bert'   -> text_bert (3,50) int tokens for a finetuned BERT
        # text_mode='frozen' -> precomputed BERT features 'text' (50,768) float (FROZEN)
        self.text_mode = text_mode
        if text_mode == 'frozen':
            self.text = np.nan_to_num(np.asarray(split['text'], np.float32), posinf=0., neginf=0.)
        else:
            self.text_bert = np.asarray(split['text_bert']).astype(np.int64)  # (N,3,50)
        self.vision = np.asarray(split['vision']).astype(np.float32)
        self.audio = np.asarray(split['audio']).astype(np.float32)
        self.label = np.asarray(split['regression_labels']).astype(np.float32).reshape(-1)
        self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)
        self.vision = np.nan_to_num(self.vision, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        txt = (torch.from_numpy(self.text[i]) if self.text_mode == 'frozen'
               else torch.from_numpy(self.text_bert[i]))    # (50,768) float | (3,50) long
        return (txt, torch.from_numpy(self.vision[i]), torch.from_numpy(self.audio[i]),
                torch.tensor(self.label[i], dtype=torch.float32))


def _collate(batch):
    tb, v, a, y = zip(*batch)
    return (torch.stack(tb), torch.stack(v), torch.stack(a), torch.stack(y))


def get_mmsa_dataloader(pkl_path: str = MMSA_MOSEI_PKL, batch_size: int = 32,
                        num_workers: int = 4, standardize: bool = True,
                        train_shuffle: bool = True, text_mode: str = 'bert'
                        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    tr, va, te = d['train'], d['valid'], d['test']

    if standardize:
        for mod in ('audio', 'vision'):
            tr_arr = np.nan_to_num(np.asarray(tr[mod], np.float32), posinf=0., neginf=0.)
            va_arr = np.nan_to_num(np.asarray(va[mod], np.float32), posinf=0., neginf=0.)
            te_arr = np.nan_to_num(np.asarray(te[mod], np.float32), posinf=0., neginf=0.)
            s_tr, (s_va, s_te) = _standardize(tr_arr, [va_arr, te_arr])
            tr[mod], va[mod], te[mod] = s_tr, s_va, s_te

    txt_desc = 'BERT768-frozen(precomputed)' if text_mode == 'frozen' else 'BERT-tokens(finetune)'
    print(f'\nLoading CMU-MOSEI MMSA aligned_50 from {os.path.basename(pkl_path)}')
    print(f'  dims: vision35 / audio74 / text={txt_desc} | T={T} | standardize={standardize}')
    print(f'  Train {len(tr["regression_labels"])}  Val {len(va["regression_labels"])}  '
          f'Test {len(te["regression_labels"])}')

    loaders = []
    for sp, shuf, drop in [(tr, train_shuffle, True), (va, False, False), (te, False, False)]:
        loaders.append(DataLoader(MMSADataset(sp, text_mode=text_mode), batch_size=batch_size,
                                  shuffle=shuf, num_workers=num_workers, pin_memory=True,
                                  drop_last=drop, collate_fn=_collate))
    return tuple(loaders)


# ---------------------------------------------------------------------------
# Standard MOSEI regression metrics (MMSA convention)
# ---------------------------------------------------------------------------

def mosei_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """y_pred, y_true: (N,) continuous in [-3,3]. Returns the standard MMSA metrics."""
    from sklearn.metrics import f1_score, accuracy_score
    y_pred = y_pred.reshape(-1); y_true = y_true.reshape(-1)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    # Acc-7: round(clip(.,-3,3)) -> 7 bins
    p7 = np.clip(np.round(y_pred), -3, 3); t7 = np.clip(np.round(y_true), -3, 3)
    acc7 = float((p7 == t7).mean())
    # Acc-5
    p5 = np.clip(np.round(y_pred), -2, 2); t5 = np.clip(np.round(y_true), -2, 2)
    acc5 = float((p5 == t5).mean())
    # Has0 Acc-2 / F1: >=0 vs <0
    h_t = (y_true >= 0); h_p = (y_pred >= 0)
    has0_acc2 = float(accuracy_score(h_t, h_p)); has0_f1 = float(f1_score(h_t, h_p, average='weighted'))
    # Non0 Acc-2 / F1: drop true==0, then >0 vs <0
    nz = y_true != 0
    n_t = (y_true[nz] > 0); n_p = (y_pred[nz] > 0)
    non0_acc2 = float(accuracy_score(n_t, n_p)); non0_f1 = float(f1_score(n_t, n_p, average='weighted'))
    return {'mae': mae, 'corr': corr, 'acc7': acc7, 'acc5': acc5,
            'has0_acc2': has0_acc2, 'has0_f1': has0_f1,
            'non0_acc2': non0_acc2, 'non0_f1': non0_f1}


if __name__ == '__main__':
    tr, va, te = get_mmsa_dataloader(batch_size=8, num_workers=0)
    tb, v, a, y = next(iter(tr))
    print('batch:', 'text_bert', tuple(tb.shape), tb.dtype, '| vision', tuple(v.shape),
          '| audio', tuple(a.shape), '| label', tuple(y.shape), y[:4].tolist())
