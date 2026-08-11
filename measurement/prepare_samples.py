#!/usr/bin/env python3
"""Write per-sample, per-modality .npy files for bench_loading.py.

The loading-inclusive benchmark needs each modality in its OWN file so that
seqA+RL can read them one at a time — reading only the modalities its policy
actually acquires. A single bundled file per sample would make that impossible
to measure: you cannot read a third of a .npz.

Output layout:
  samples/{ds}_fold{k}/sample_{i:05d}/{modality}.npy
  samples/{ds}_fold{k}/sample_{i:05d}/label.npy

Source is the batched test tensors in data/<ds>_fold<k>_test.pt.

Note on format: .npy (not .pt). PyTorch's zip-based writer fails when asked to
emit this many small files, and numpy's format is both simpler and faster to
read one array at a time — which is exactly the access pattern being measured.

Sizing: --n 50 gives ~7 GB and is plenty for latency (which is stable across
samples). Use --n 0 for full test sets when you also want trustworthy F1.

Usage:
  cd runs/measurement
  python prepare_samples.py [--n 50] [--datasets iemocap cmi]
"""
import argparse, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
OUT_DIR = os.path.join(HERE, 'samples')

DS_FOLDS = {
    'iemocap': 3, 'mmfi': 3, 'cmi': 3, 'czu_mhad': 3,
    'dsads': 4, 'eav': 3, 'utd_mhad': 3,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50,
                    help='max samples per fold (0 = all)')
    ap.add_argument('--datasets', nargs='+', default=list(DS_FOLDS),
                    choices=list(DS_FOLDS))
    a = ap.parse_args()

    print(f'Writing per-modality samples -> {OUT_DIR}')
    print(f'n per fold: {a.n or "all"}\n')
    total = 0

    for ds in a.datasets:
        for fold in range(DS_FOLDS[ds]):
            src = os.path.join(DATA_DIR, f'{ds}_fold{fold}_test.pt')
            if not os.path.exists(src):
                print(f'  MISSING: {src}')
                continue

            d = torch.load(src, map_location='cpu', weights_only=False)
            mods = list(d['mods'])

            idx, done = 0, False
            for b in d['batches']:
                if done:
                    break
                for i in range(b['y'].shape[0]):
                    if a.n and idx >= a.n:
                        done = True
                        break
                    sd = os.path.join(OUT_DIR, f'{ds}_fold{fold}', f'sample_{idx:05d}')
                    os.makedirs(sd, exist_ok=True)
                    for m in mods:
                        np.save(os.path.join(sd, f'{m}.npy'),
                                b['high'][m][i].float().numpy())
                    label = int(b['y'][i].item()) if b['y'].ndim == 1 \
                            else int(b['y'][i].argmax().item())
                    np.save(os.path.join(sd, 'label.npy'),
                            np.array(label, dtype=np.int64))
                    idx += 1

            total += idx
            print(f'  {ds} fold{fold}: {idx} samples × {len(mods)} modalities')

    print(f'\nDone — {total} samples written to {OUT_DIR}')


if __name__ == '__main__':
    main()
