"""Add DSADS to seqA_vlo_vs_random_exit (val-LOO order vs 5-random-order avg exit).
Reuses helpers from compute_vlo_vs_random.py; merges a 'dsads' entry into
../results_3p3/seqA_vlo_vs_random_exit_3ds.json. Run from clean_models/train/."""
from __future__ import annotations
import glob, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from compute_vlo_vs_random import (build_har, acc_with_har, prefix_logits_har,
                                    curve_for_order, avg_curves, to_rows, dev, R, N_RAND)
from utils.dataset_cfg import DSADS
from data.dataset_builder import DSADSDataset

cfg = DSADS(); MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
T = (cfg.duration * cfg.base_sample_rate) // 1                 # 125
variates = dict(cfg.variates)
sl, off = {}, 0
for m in MODS:
    sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
seqa_dir = sorted(glob.glob('../results_3p3/dsads/seqA/*'))[0]   # gap-head seqA (same as exit_dsads)
C = json.load(open(os.path.join(seqa_dir, 'config.json')))
gpk = json.load(open(os.path.join(R, 'dsads', 'exit_dsads.json')))['dsads']['gflops_per_k']
DATA = '/files1/haodong/data/DSADS/'
SAX = {'alphabet_size': 20, 'word_length': 1}


def loaders(fold):
    _src = torch.load(DATA + f'dsad_diversify_dict_scenario{fold}_src.pt', weights_only=False)
    _trg = torch.load(DATA + f'dsad_diversify_dict_scenario{fold}_trg.pt', weights_only=False)
    _, _vax, _, _vay = train_test_split(_src['samples'], _src['labels'], test_size=0.2,
                                        random_state=2711, stratify=_src['labels'])
    vl = DataLoader(DSADSDataset({'samples': _vax, 'labels': _vay}, MODS, cfg, 'sax', sax_params=SAX),
                    batch_size=64, shuffle=False, num_workers=4)
    el = DataLoader(DSADSDataset(_trg, MODS, cfg, 'sax', sax_params=SAX),
                    batch_size=64, shuffle=False, num_workers=4)
    return vl, el


vlo_folds, rand_folds, orders = [], [], []
for fold in range(4):
    vl, el = loaders(fold)
    model = build_har(C, MODS, variates, T, NUMC, C['num_layers_per_modal'])
    sd = torch.load(os.path.join(seqa_dir, f'best_model_fold{fold}.pth'), map_location='cpu', weights_only=True)
    model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()
    full = acc_with_har(model, vl, MODS, sl)
    loo = {m: full - acc_with_har(model, vl, [x for x in MODS if x != m], sl) for m in MODS}
    vlo_order = sorted(MODS, key=lambda m: -loo[m]); orders.append(vlo_order)
    print(f'  dsads fold{fold} full_val={full:.4f} vlo_order={vlo_order}')
    labels, Hs, preds = prefix_logits_har(model, el, vlo_order, sl, M)
    vlo_folds.append(curve_for_order(labels, Hs, preds, M))
    rng = np.random.default_rng(1234 + fold); rc = []
    for _ in range(N_RAND):
        perm = list(rng.permutation(MODS))
        lb, Hr, pr = prefix_logits_har(model, el, perm, sl, M)
        rc.append(curve_for_order(lb, Hr, pr, M))
    rand_folds.append(avg_curves(rc))
    del model; torch.cuda.empty_cache()

vlo = avg_curves(vlo_folds); rnd = avg_curves(rand_folds)
rec = {'M': M, 'gflops_per_k': gpk, 'vlo': to_rows(vlo), 'random': to_rows(rnd)}
p = os.path.join(R, 'seqA_vlo_vs_random_exit_3ds.json')
d = json.load(open(p)); d['dsads'] = rec; json.dump(d, open(p, 'w'), indent=2)
print('merged dsads ->', p)
print(f"dsads: vlo full acc={np.array(rec['vlo'])[-1,1]:.4f} | cheapest vlo acc={np.array(rec['vlo'])[0,1]:.4f} "
      f"random cheapest acc={np.array(rec['random'])[0,1]:.4f}")
