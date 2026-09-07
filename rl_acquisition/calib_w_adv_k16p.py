"""Recalibrate W_ADV for the parity (k16p) backbones and patch run_design.py.

W_ADV = median(max softmax over all nodes/samples) / max(global Shapley phi), measured
on the fold-0 VAL cache. It balances the small marginal-Shapley advice against the
confidence term in the Q-prior; both quantities depend on the backbone, so the k16
values do not transfer to the parity architecture.

Run AFTER the parity caches are built:
    RLB_RUNSET=k16p RLB_CACHE_SUBDIR=botneck_16_parity python calib_w_adv_k16p.py
"""
import os
import re

os.environ.setdefault('RLB_RUNSET', 'k16p')
os.environ.setdefault('RLB_CACHE_SUBDIR', 'botneck_16_parity')

import numpy as np
import environment as env
import shapley

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['dsads', 'eav', 'pamap2', 'utd_mhad']

vals = {}
for ds in DATASETS:
    cache = env.get_cache('val', ds, 0)
    M, md = cache['M'], env.depth_of(ds)
    phi = shapley.shapley_values(cache, M, md)
    conf_med = float(np.median(cache['softmax'].max(-1).ravel()))
    w = conf_med / float(np.max(np.abs(phi)))
    vals[ds] = round(w, 2)
    print(f'{ds:10s} conf_median={conf_med:.4f}  max|phi|={np.max(np.abs(phi)):.4f}  W_ADV={w:.2f}')

p = os.path.join(HERE, 'run_design.py')
src = open(p).read()
new = 'W_ADV_MAP_K16P = ' + repr(vals)
src2 = re.sub(r'W_ADV_MAP_K16P = \{[^}]*\}', new, src, count=1)
assert src2 != src, 'W_ADV_MAP_K16P line not found in run_design.py'
open(p, 'w').write(src2)
print('\npatched run_design.py ->', new)
