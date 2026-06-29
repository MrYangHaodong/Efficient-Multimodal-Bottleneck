"""seqA pamap2 @T=512 adaptive-exit curve + per-K modality GFLOPs (same method as
compare_seqexit_admn_T256.py for daliahar/wesad). Output keys match CMP256 so the
baseline plot can read pamap2 at T=512. Run from clean_models/train/."""
from __future__ import annotations
import glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_function import set_seed
from utils.dataset_cfg import PAMAP2
from data.pamap2_crosssubject import load_pamap2_subjects
from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import DualVideoBottleneckModelV6Downsample

dev = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'); set_seed(42)
TAUS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 10.0]
cfg = PAMAP2(); MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes; T = 512
sl, off = {}, 0
for m in MODS:
    sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
seqa_dir = sorted(glob.glob('../results_3p3/seqA_pamap2/2026-06-20*'))[0]
C = json.load(open(os.path.join(seqa_dir, 'config.json')))


class _V6Cfg:
    def __init__(s, c):
        s.modalities = c.modalities; s.variates = c.variates
        f = c.modalities[0]; s.video_high_dim = c.variates[f]; s.video_low_dim = c.variates[f]


def build():
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=_V6Cfg(cfg), output_dim=NUMC, input_length=T, d_model=C['d_model'], nhead=C['nhead'],
        num_layers_per_modal=C['num_layers_per_modal'], num_layers=C['num_layers'], dropout=C['dropout'], verbose=False,
        video_low_dim=cfg.variates[MODS[0]], video_high_dim=cfg.variates[MODS[0]], use_bottleneck=True,
        n_bottlenecks=C['n_bottlenecks'], fusion_layer=C['fusion_layer'], use_sparse_moe=False,
        num_experts=C['num_experts'], expert_k=1, internal_dim=C['internal_dim'], bottleneck_head_pos=True,
        use_sparse_attn=C['use_sparse_attn'], factor=C['base_factor'], selector_video_source='high',
        encoder_video_source='high', no_selector=True, use_weighted_factor=False, use_triton=False, num_classes=NUMC,
        sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
        downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
        per_modal_distill=C['per_modal_distill'], per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
        fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full',
        fusion_mlp_ratio=1.0, fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'],
        seq_modality_order=C['seq_modality_order'], seq_random_order=C['seq_random_order'],
        bottleneck_agg_mode='gate', n_fusion_distill=-1).to(dev).float()


def sub(x, mods):
    h = {m: x[:, :, sl[m][0]:sl[m][1]] for m in mods}; return h, dict(h)


@torch.no_grad()
def acc_with(model, loader, keep):
    cor = n = 0
    for x, y in loader:
        x = x.to(dev).float(); o = model(*sub(x, keep), training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        cor += int((o.argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


@torch.no_grad()
def prefix_logits(model, loader, order):
    by, labels = [], None
    for k in range(1, M + 1):
        outs, ys = [], []
        for x, y in loader:
            x = x.to(dev).float(); o = model(*sub(x, order[:k]), training=False, return_selection_info=False)
            o = o[0] if isinstance(o, tuple) else o
            outs.append(o.cpu()); ys.append(y)
        by.append(torch.cat(outs)); labels = torch.cat(ys).numpy() if labels is None else labels
    L = torch.stack(by, 1); P = F.softmax(L, -1)
    Hs = -(P * P.clamp_min(1e-9).log()).sum(-1).numpy(); preds = L.argmax(-1).numpy()
    return labels, Hs, preds


def simulate(labels, Hs, preds, tau):
    N = len(labels); ek = np.full(N, M)
    for k in range(M):
        ek = np.where((ek == M) & (Hs[:, k] <= tau), k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    return float((pe == labels).mean()), float(f1_score(labels, pe, average='macro')), float(ek.mean())


def loader(subs):
    return DataLoader(load_pamap2_subjects(cfg.cross_subject_dir, MODS, subs, cfg, 'sax_noisy',
                      sax_params={'alphabet_size': 20, 'word_length': 1}), batch_size=64, shuffle=False, num_workers=4)


val_loader = loader(cfg.val_set)
perfold = {t: {'acc': [], 'f1': [], 'mk': []} for t in TAUS}
borda = {m: 0.0 for m in MODS}; orders = []
print('=== seqA pamap2 @T512 val-LOO modality order (strong-first; used for early-exit prefix) ===')
for fold in range(3):
    fc = cfg.folds[fold]; test_loader = loader(fc['eval_set'])
    model = build()
    sd = torch.load(os.path.join(seqa_dir, f'best_model_fold{fold}.pth'), map_location='cpu', weights_only=True)
    model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()
    full = acc_with(model, val_loader, MODS)
    loo = {m: full - acc_with(model, val_loader, [x for x in MODS if x != m]) for m in MODS}
    order = sorted(MODS, key=lambda m: -loo[m]); orders.append(order)
    for r, m in enumerate(order):
        borda[m] += (M - r)
    print(f'  fold{fold} full_val={full:.4f}  LOO: ' + ', '.join(f'{m}={loo[m]:+.3f}' for m in order))
    print(f'         order: {order}')
    labels, Hs, preds = prefix_logits(model, test_loader, order)
    for t in TAUS:
        a, f, mk = simulate(labels, Hs, preds, t)
        perfold[t]['acc'].append(a); perfold[t]['f1'].append(f); perfold[t]['mk'].append(mk)
consensus = sorted(MODS, key=lambda m: -borda[m])
print(f'  >>> CONSENSUS val-LOO order (strong->weak): {consensus}')
exit_curve = sorted([(float(np.mean(perfold[t]['mk'])), float(np.mean(perfold[t]['acc'])), float(np.mean(perfold[t]['f1']))) for t in TAUS])

# per-K modality GFLOPs at T=512
x1 = next(iter(val_loader))[0][:1].float().to(dev)
m = build().eval(); gpk = []
for k in range(1, M + 1):
    fcm = FlopCounterMode(display=False)
    with fcm, torch.no_grad():
        m(*sub(x1, MODS[:k]), training=False, return_selection_info=False)
    gpk.append(fcm.get_total_flops() / 1e9)

out = {'pamap2': {'M': M, 'exit_curve': exit_curve, 'gflops_per_k': gpk}}
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_3p3')
json.dump(out, open(os.path.join(R, 'seqA_pamap2/exit_T512_saxnoisy.json'), 'w'), indent=2)
print(f"pamap2 @T512: exit acc {exit_curve[0][1]:.3f}->{exit_curve[-1][1]:.3f} | gflops_per_k={[round(g,3) for g in gpk]}")
print('saved -> results_3p3/seqA_pamap2_T512_exit.json')
