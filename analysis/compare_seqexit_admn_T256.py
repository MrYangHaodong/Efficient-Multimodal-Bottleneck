"""GFLOPs vs acc/F1 at T=256: seqA ADAPTIVE EXITING (entropy modality early-exit,
the modality axis) vs ADMN adaptive DEPTH (per-modality layer dropping). Both on
the SAME trained T=256 seqA backbone (daliahar + wesad).

seqA-exit: process modalities strong-first (val-LOO order); after k modalities use
softmax-entropy of the head as confidence -> exit if entropy <= tau (sweep tau) ->
(meanK, acc, f1). GFLOPs from per-K modality cost (FlopCounterMode).
ADMN: load the trained ADMN results (none / controller@budget=static / full acc);
GFLOPs by counting only ACTIVE per-modal layers (none=1-layer, full=3-layer cost,
linear in #active 2nd/3rd layers) since masked layers are skipped at deploy.

Out: compare_seqexit_vs_admn_T256.png + .json. Run from clean_models/train/.
"""
from __future__ import annotations
import glob, json, math, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_function import set_seed
from utils.dataset_cfg import DaliaHAR, WESAD
from data.dataset_builder import HARDataset
from models.seqA import (
    DualVideoBottleneckModelV6Downsample)

dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
set_seed(42)
TAUS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 10.0]
DS = {'daliahar': (DaliaHAR, '/files1/haodong/data/processed_dalia_activity', '../results_3p3/T256'),
      'wesad': (WESAD, '/files1/haodong/data/WESAD/processed_wesad_activity', '../results_3p3/T256_wesad')}


class _V6Cfg:
    def __init__(s, mods, variates):
        s.modalities = list(mods); s.variates = {m: variates[m] for m in mods}
        f = s.modalities[0]; s.video_high_dim = s.variates[f]; s.video_low_dim = s.variates[f]


def build(C, cfg, T, NUMC, nlpm):
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=_V6Cfg(cfg.modalities, cfg.variates), output_dim=NUMC, input_length=T,
        d_model=C['d_model'], nhead=C['nhead'], num_layers_per_modal=nlpm, num_layers=C['num_layers'],
        dropout=C['dropout'], verbose=False, video_low_dim=cfg.variates[cfg.modalities[0]],
        video_high_dim=cfg.variates[cfg.modalities[0]], use_bottleneck=True, n_bottlenecks=C['n_bottlenecks'],
        fusion_layer=C['fusion_layer'], use_sparse_moe=False, num_experts=C['num_experts'], expert_k=1,
        internal_dim=C['internal_dim'], bottleneck_head_pos=True, use_sparse_attn=C['use_sparse_attn'],
        factor=C['base_factor'], selector_video_source='high', encoder_video_source='high', no_selector=True,
        use_weighted_factor=False, use_triton=False, num_classes=NUMC, sparse_attn_variant=C['sparse_attn_variant'],
        strat_block_size=C['strat_block_size'], downsample_min_len=C['downsample_min_len'],
        use_batched_fusion=C['use_batched_fusion'], per_modal_distill=C['per_modal_distill'],
        per_modal_downsample_min_len=C['per_modal_downsample_min_len'], fusion_add_pos_embeds=False,
        fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full', fusion_mlp_ratio=1.0,
        fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'], seq_modality_order=C['seq_modality_order'],
        seq_random_order=C['seq_random_order'], bottleneck_agg_mode='gate', n_fusion_distill=-1)


def sub(x, mods, sl):
    h = {m: x[:, :, sl[m][0]:sl[m][1]] for m in mods}
    return h, dict(h)


@torch.no_grad()
def acc_with(model, loader, keep, sl):
    cor = n = 0
    for x, y in loader:
        x = x.to(dev).float(); o = model(*sub(x, keep, sl), training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        cor += int((o.argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


@torch.no_grad()
def prefix_logits(model, loader, order, sl, M):
    by_k, labels = [], None
    for k in range(1, M + 1):
        outs, ys = [], []
        for x, y in loader:
            x = x.to(dev).float(); o = model(*sub(x, order[:k], sl), training=False, return_selection_info=False)
            o = o[0] if isinstance(o, tuple) else o
            outs.append(o.cpu()); ys.append(y)
        by_k.append(torch.cat(outs)); labels = torch.cat(ys).numpy() if labels is None else labels
    L = torch.stack(by_k, 1)                                  # [N, M, C]
    P = F.softmax(L, -1); Hs = -(P * P.clamp_min(1e-9).log()).sum(-1).numpy()  # [N, M]
    preds = L.argmax(-1).numpy()                              # [N, M]
    return labels, Hs, preds


def simulate(labels, Hs, preds, M, tau):
    N = len(labels); exit_k = np.full(N, M)
    for k in range(M):
        first = (exit_k == M) & (Hs[:, k] <= tau)
        exit_k = np.where(first, k + 1, exit_k)
    pe = preds[np.arange(N), exit_k - 1]
    from sklearn.metrics import f1_score
    return float((pe == labels).mean()), float(f1_score(labels, pe, average='macro')), float(exit_k.mean())


def gflops(model, x1, mods, sl):
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        model(*sub(x1, mods, sl), training=False, return_selection_info=False)
    return fc.get_total_flops() / 1e9


OUT = {}
for name, (CFG, ROOT, RESROOT) in DS.items():
    cfg = CFG(); MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
    T = (cfg.duration * cfg.base_sample_rate); sl, off = {}, 0
    for m in MODS:
        sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
    seqa_dir = sorted(glob.glob(os.path.join(RESROOT, 'seqA', '*')))[0]
    C = json.load(open(os.path.join(seqa_dir, 'config.json')))
    val_ds = HARDataset(ROOT, MODS, cfg.val_set, cfg, 'sax', sax_params={'alphabet_size': 20, 'word_length': 1})

    # --- seqA adaptive-exit frontier (per-fold val-LOO order; entropy exit on test) ---
    perfold = {t: {'acc': [], 'f1': [], 'mk': []} for t in TAUS}
    for fold in range(3):
        fc = cfg.folds[fold]
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
        test_ds = HARDataset(ROOT, MODS, fc['eval_set'], cfg, 'sax', sax_params={'alphabet_size': 20, 'word_length': 1})
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
        model = build(C, cfg, T, NUMC, C['num_layers_per_modal']).to(dev).float()
        sd = torch.load(os.path.join(seqa_dir, f'best_model_fold{fold}.pth'), map_location='cpu', weights_only=True)
        model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()
        full = acc_with(model, val_loader, MODS, sl)
        loo = {m: full - acc_with(model, val_loader, [x for x in MODS if x != m], sl) for m in MODS}
        order = sorted(MODS, key=lambda m: -loo[m])
        labels, Hs, preds = prefix_logits(model, test_loader, order, sl, M)
        for t in TAUS:
            a, f, mk = simulate(labels, Hs, preds, M, t)
            perfold[t]['acc'].append(a); perfold[t]['f1'].append(f); perfold[t]['mk'].append(mk)
    exit_curve = sorted([(np.mean(perfold[t]['mk']), np.mean(perfold[t]['acc']), np.mean(perfold[t]['f1']))
                         for t in TAUS])

    # --- GFLOPs: modality per-K (full depth) + depth none(1-layer)/full(3-layer) ---
    x1 = next(iter(DataLoader(val_ds, batch_size=1)))[0].float().to(dev)
    m3 = build(C, cfg, T, NUMC, C['num_layers_per_modal']).to(dev).float().eval()
    gpk = [gflops(m3, x1, MODS[:k], sl) for k in range(1, M + 1)]      # modality axis, K mods, full depth
    g_full = gpk[-1]                                                    # all M, all layers
    m1 = build(C, cfg, T, NUMC, 1).to(dev).float().eval()
    g_none = gflops(m1, x1, MODS, sl)                                   # all M, only layer-0
    n_drop = C['num_layers_per_modal'] - 1
    per_layer = (g_full - g_none) / (M * n_drop)                       # cost of one droppable per-modal layer

    # --- ADMN results (acc) + its GFLOPs at none / budget / full ---
    admn = {}
    for fold in range(3):
        j = glob.glob(os.path.join(RESROOT, 'admn_seqA', '*', f'results_fold{fold}.json'))
        if j:
            admn[fold] = json.load(open(j[0]))['admn']
    Bbud = json.load(open(glob.glob(os.path.join(RESROOT, 'admn_seqA', '*', 'results_fold0.json'))[0]))['admn_budget']
    admn_acc = {k: float(np.mean([admn[f][k] for f in admn])) for k in ['none_acc', 'controller_acc', 'full_acc']}
    admn_f1 = {'controller': float(np.mean([admn[f]['controller_f1'] for f in admn]))}
    admn_pts = {  # (gflops, acc) for none / controller@budget(=static) / full
        'none':    (g_none, admn_acc['none_acc']),
        'budget':  (g_none + Bbud * per_layer, admn_acc['controller_acc']),
        'full':    (g_full, admn_acc['full_acc'])}
    OUT[name] = {'M': M, 'exit_curve': exit_curve, 'gflops_per_k': gpk,
                 'g_none': g_none, 'g_full': g_full, 'admn_budget': Bbud, 'per_layer_gflops': per_layer,
                 'admn_points': admn_pts, 'admn_f1_controller': admn_f1['controller']}
    print(f"{name}: exit meanK->acc [{exit_curve[0][1]:.3f}..{exit_curve[-1][1]:.3f}]  "
          f"GFLOPs none={g_none:.3f} full={g_full:.3f} ADMN budget={Bbud} "
          f"@{admn_pts['budget'][0]:.3f}GFLOPs acc={admn_pts['budget'][1]:.3f}")

# --- plot ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for col, name in enumerate(['daliahar', 'wesad']):
    o = OUT[name]; M = o['M']; gpk = o['gflops_per_k']
    ec = np.array(o['exit_curve'])                            # (meanK, acc, f1)
    gf_exit = np.interp(ec[:, 0], np.arange(1, M + 1), gpk)
    for row, (yi, ylab) in enumerate([(1, 'Accuracy'), (2, 'Macro-F1')]):
        ax = axes[row, col]
        ax.plot(gf_exit, ec[:, yi], '-o', color='tab:blue', lw=2.4, ms=4, label='seqA adaptive-exit (modality)', zorder=6)
        ap = o['admn_points']
        ax.plot([ap['none'][0], ap['budget'][0], ap['full'][0]],
                [ap['none'][1], ap['budget'][1], ap['full'][1]] if row == 0 else
                [ap['none'][1], o['admn_f1_controller'], ap['full'][1]],
                '-s', color='tab:red', lw=2.0, ms=8, label='ADMN adaptive-depth (none/budget/full)', zorder=7)
        ax.scatter([ap['budget'][0]], [ap['budget'][1] if row == 0 else o['admn_f1_controller']],
                   s=160, facecolors='none', edgecolors='tab:red', linewidths=2, zorder=8,
                   label='ADMN controller@budget (=static, adv=0)')
        ax.set_xlabel('GFLOPs / sample'); ax.set_ylabel(ylab)
        ax.set_title(f'{name} (M={M}, T=256)' + ('  [seqA-exit vs ADMN]' if row == 0 else ''), fontsize=10)
        ax.grid(alpha=0.3)
        if col == 0 and row == 0:
            ax.legend(fontsize=8, loc='lower right')
fig.suptitle('seqA adaptive EXITING (modality axis) vs ADMN adaptive DEPTH (layer axis) — GFLOPs vs acc/F1, cross-subject T=256\n'
             'seqA-exit traces a real accuracy–compute frontier; ADMN depth is nearly flat (none≈full) and its per-sample controller collapses to static (adv=0).',
             fontsize=10)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.9, wspace=0.2, hspace=0.3)
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_3p3')
fig.savefig(os.path.join(R, 'compare_seqexit_vs_admn_T256.png'), dpi=150)
json.dump(OUT, open(os.path.join(R, 'compare_seqexit_vs_admn_T256.json'), 'w'), indent=2)
print('saved ->', os.path.join(R, 'compare_seqexit_vs_admn_T256.png'))
