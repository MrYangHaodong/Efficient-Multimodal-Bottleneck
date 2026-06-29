"""Cross-subject PAMAP2 loading.

The original PAMAP2 splits (``pamap2_split{0,1,2}_{train,test}.pt``) pool all
subjects and split randomly -> NOT subject-disjoint. ``pamap2_crosssubject_
preprocess.py`` instead writes one dict per subject (``S101.pt``..``S109.pt``).

``load_pamap2_subjects`` concatenates the dicts for a given list of subjects
into a single ``PAMAP2Dataset`` -> a subject-disjoint train/val/test split, the
same protocol DaliaHAR/IEMOCAP use.
"""
import os
import numpy as np
import torch

from data.dataset_builder import PAMAP2Dataset

__all__ = ['load_pamap2_subjects']


def load_pamap2_subjects(cs_dir, modalities, subjects, cfg, transform=None,
                         sax_params=None, noise_config=None):
    """Concatenate the per-subject PAMAP2 dicts in ``subjects`` -> PAMAP2Dataset.

    Arg order mirrors ``HARDataset(root_dir, modalities, subjects, cfg,
    transform)`` so a PAMAP2 training script is a drop-in swap of the loader.

    Args:
        cs_dir:     dir holding ``S###.pt`` (cfg.cross_subject_dir).
        modalities: modality names to keep.
        subjects:   list like ['S101', 'S102', ...].
        cfg:        PAMAP2 cfg object.
    """
    dicts = []
    for s in subjects:
        p = os.path.join(cs_dir, f'{s}.pt')
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{p} missing -- run script/pamap2_crosssubject_preprocess.py first')
        dicts.append(torch.load(p, map_location='cpu', weights_only=False))

    keys = list(modalities) + ['labels']
    merged = {}
    for k in keys:
        merged[k] = np.concatenate([np.asarray(d[k]) for d in dicts], axis=0)
    return PAMAP2Dataset(merged, modalities, cfg, transform=transform,
                         sax_params=sax_params, noise_config=noise_config)
