#!/usr/bin/env python
"""seqA GloVe-regression early-exit curve on CMU-MOSEI: GFLOPs vs Acc-7 and vs MAE
(per-prefix, val-LOO modality order, 3-seed mean). MAE: lower is better (axis inverted)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(R, 'mosei', 'glove_reg_exit.json')))
M = D['M']; gpk = D['gflops_per_k']
cur = np.array(D['mean_curve_k_acc7_mae'])          # [M,3] = k, acc7, mae
x = np.array(gpk)                                    # GFLOPs at k=1..M (val-LOO prefix)
acc7 = cur[:, 1]; mae = cur[:, 2]
# per-seed spread
ps = D['per_seed']
acc7_all = np.array([[c[1] for c in s['curve']] for s in ps])
mae_all = np.array([[c[2] for c in s['curve']] for s in ps])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
ax = axes[0]
ax.plot(x, acc7, '-o', color='tab:blue', lw=2.6, ms=7, zorder=6)
ax.fill_between(x, acc7_all.min(0), acc7_all.max(0), color='tab:blue', alpha=0.15)
for k in range(M):
    ax.annotate(f'k={k+1}', (x[k], acc7[k]), textcoords='offset points', xytext=(5, -12), fontsize=9, color='tab:blue')
ax.set_xscale('log'); ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel('Acc-7 (higher better)')
ax.set_title('Acc-7 vs GFLOPs'); ax.grid(alpha=0.25, which='both')

ax = axes[1]
ax.plot(x, mae, '-s', color='tab:red', lw=2.6, ms=7, zorder=6)
ax.fill_between(x, mae_all.min(0), mae_all.max(0), color='tab:red', alpha=0.15)
for k in range(M):
    ax.annotate(f'k={k+1}', (x[k], mae[k]), textcoords='offset points', xytext=(5, 6), fontsize=9, color='tab:red')
ax.set_xscale('log'); ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel('MAE (lower better)')
ax.invert_yaxis()       # so "up = better" matches the Acc-7 panel
ax.set_title('MAE vs GFLOPs'); ax.grid(alpha=0.25, which='both')

fig.suptitle(f'CMU-MOSEI GloVe-regression seqA adaptive modality early-exit (M=3 vision/audio/text, {len(ps)}-seed mean, band=min/max).\n'
             f'val-LOO order {ps[0]["order"]}; GFLOPs/sample: text-only {gpk[0]:.3f} -> +1 {gpk[1]:.3f} -> all-3 {gpk[2]:.3f} '
             f'({100*gpk[0]/gpk[2]:.0f}% of full at k=1).', fontsize=10)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.85, wspace=0.20)
o = os.path.join(R, 'mosei', 'gflops_mosei_glove_reg_acc7_mae_exit.png'); fig.savefig(o, dpi=150); print('saved ->', o)
print('k / Acc-7 / MAE  (3-seed mean):')
for k in range(M):
    print(f'  k={k+1} ({gpk[k]:.3f} GFLOPs): Acc-7 {acc7[k]:.4f}   MAE {mae[k]:.4f}')
