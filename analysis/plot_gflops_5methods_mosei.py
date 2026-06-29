"""CMU-MOSEI GFLOPs vs Acc-2 / Macro-F1: seqA_ps adaptive modality-exit CURVE + MMEE depth-exit
CURVE + DMBF / DynMM / AdaMML points. GFLOPs measured with ONE FlopCounterMode (architecture-only,
no checkpoints needed) for DMBF & MMEE; seqA from arrays gpk; DynMM/AdaMML from their run logs.
Each method at its as-trained compute (note: DMBF uses tcr=1 -> longer seq -> heavier)."""
import os, sys, json, glob
from types import SimpleNamespace
import numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_HERE = os.path.dirname(os.path.abspath(__file__)); _CM = os.path.dirname(_HERE)
sys.path.insert(0, _CM); sys.path.insert(0, os.path.join(_CM, 'train'))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def measure(model, run_once):
    model.train()   # IMPORTANT: train mode disables the fused _transformer_encoder_layer_fwd
    fcm = FlopCounterMode(display=False)   # kernel, so nn.TransformerEncoderLayer flops are counted
    with torch.no_grad(), fcm:
        run_once()
    return fcm.get_total_flops() / 1e9     # batch=1 -> per-sample GFLOPs


# ---------- DMBF full GFLOPs (tcr=1, T=50) ----------
from data.CMU_MOSEI.get_data import get_dataloader as mget, FEATURE_DIMS, NUM_CLASSES
from models.dmbf import DMBFClassifier
mods = ['vision', 'audio', 'text']
_, _, el = mget(batch_size=4, num_workers=2, modalities=mods, split_id=0, train_shuffle=False,
                max_seq_len=50, time_compression_ratio=1, use_batched_fusion=True, task='classification2')
feats = list(next(iter(el)))[:-1]
highs = [feats[2 * j][:1].to(dev).float() for j in range(3)]
dmbf = DMBFClassifier(mods, {m: FEATURE_DIMS[m] for m in mods}, NUM_CLASSES, 50,
                      d_model=128, nhead=8, num_layers=2, dropout=0.2).to(dev)
G_DMBF = measure(dmbf, lambda: dmbf(highs))
print(f"DMBF full GFLOPs = {G_DMBF:.4f}")

# ---------- MMEE depth GFLOPs (tcr=4) at L=1 and L=6 -> linear per-layer ----------
from ee_data import build_loaders
from models.mmee import MMEEClassifier
a = SimpleNamespace(dataset='mosei', max_seq_len=50, time_compression_ratio=4, fold=0, batch_size=4, num_workers=2)
_, _, mel, cfg, in_len, in_mode, _, unpack = build_loaders(a)
x, _y = unpack(next(iter(mel)), dev); x1 = x[:1]
def build_mmee(L): return MMEEClassifier(cfg, in_len, input_mode=in_mode, d_model=128, nhead=8,
                                         num_layers=L, exit_layers=list(range(1, L + 1))).to(dev)
G6 = measure(build_mmee(6), lambda: None) if False else None
m6 = build_mmee(6); G6 = measure(m6, lambda: m6(x1))
m1 = build_mmee(1); G1 = measure(m1, lambda: m1(x1))
per_layer = (G6 - G1) / 5.0
mmee_gflops = lambda L: G1 + (L - 1) * per_layer
print(f"MMEE GFLOPs: L1={G1:.4f}  L6={G6:.4f}  per-fusion-layer={per_layer:.4f}")

# ---------- seqA gpk + seqA_ps adaptive curve (3-seed pooled) ----------
gpk = np.load(os.path.join(_HERE, 'lsmi_exit', 'arrays_mosei.npz'))['gflops_per_k']
ew = json.load(open(os.path.join(_HERE, 'lsmi_exit', 'exit_withR.json')))['mosei']
def seqa_curve(sch):
    c = np.array(ew[sch]); x = np.interp(c[:, 0], [1, 2, 3], gpk); o = np.argsort(x)
    return x[o], c[o, 1], c[o, 2]

# ---------- MMEE adaptive curve (avg 3 seeds) ----------
mmee_curves = [json.load(open(p))['exit_curve_meanLayer_acc_f1']
               for p in sorted(glob.glob(os.path.join(_CM, 'train/results_mmee/mmee_s*_mosei/run/results.json')))]
cv = np.array(mmee_curves).mean(0)               # [npts,3] meanLayer,acc,f1
o = np.argsort(cv[:, 0]); cv = cv[o]
mmee_x = np.array([mmee_gflops(L) for L in cv[:, 0]])

# ---------- single-point methods (GFLOPs, acc, f1) ----------
PTS = {'DMBF':   (G_DMBF, 0.7318, 0.7312, 'tab:green', 'D'),
       'DynMM':  (0.0757, 0.7269, 0.7268, 'tab:orange', 's'),
       'AdaMML': (0.1300, 0.6558, 0.6558, 'tab:red', 'X')}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
for ax, yi, ylab in [(axes[0], 1, 'Acc-2'), (axes[1], 2, 'Macro-F1')]:
    xs, ya, yf = seqa_curve('entropy')
    ax.plot(xs, ya if yi == 1 else yf, '-o', color='tab:blue', lw=2.6, ms=5, zorder=6,
            label='seqA_ps (adaptive modality-exit)')
    ax.plot(mmee_x, cv[:, yi], '-^', color='tab:purple', lw=2.2, ms=5, zorder=5,
            label='MMEE (adaptive depth-exit)')
    for nm, (g, a2, f2, c, mk) in PTS.items():
        ax.scatter([g], [a2 if yi == 1 else f2], color=c, marker=mk, s=130, zorder=8,
                   edgecolor='w', linewidth=0.8, label=nm)
    ax.set_xscale('log'); ax.set_xlabel('GFLOPs / sample (log)'); ax.set_ylabel(ylab)
    ax.set_title(f'CMU-MOSEI: {ylab} vs compute'); ax.grid(alpha=0.3, which='both')
    if yi == 1: ax.legend(fontsize=9, loc='lower right')
fig.suptitle('CMU-MOSEI accuracy–compute: adaptive seqA_ps (modality-exit) & MMEE (depth-exit) curves vs '
             'DMBF / DynMM / AdaMML points (3-seed). GFLOPs via one FlopCounterMode; each at its as-trained config.', fontsize=10.5)
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.11, top=0.89, wspace=0.18)
out = os.path.join(_HERE, 'gflops_acc_f1_5methods_mosei.png'); fig.savefig(out, dpi=140); print('saved ->', out)
json.dump({'G_DMBF': G_DMBF, 'G_MMEE_L1': G1, 'G_MMEE_L6': G6, 'gpk_seqA': gpk.tolist(),
           'DynMM_gflops': 0.0757, 'AdaMML_gflops': 0.130,
           'seqA_ps_entropy_curve': [list(map(float, t)) for t in zip(*seqa_curve('entropy'))],
           'mmee_curve_gflops_acc_f1': [[float(mmee_x[i]), float(cv[i, 1]), float(cv[i, 2])] for i in range(len(cv))]},
          open(os.path.join(_HERE, 'gflops_5methods_mosei.json'), 'w'), indent=1)
print('saved -> results_3p3/gflops_5methods_mosei.json')
