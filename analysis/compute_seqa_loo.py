"""Compute val leave-one-out (LOO) modality importance on the trained T=256 seqA
models (same method that determined seqA's modality order: full_acc - acc_without_m
on the VAL set, leakage-free). Prints a strong-first ranking per fold + a consensus
across folds -> used to define the ModalityDynMM branches (top-1 / top-ceil(M/2) / all).
Run from clean_models/train/:  python compute_seqa_loo.py
"""
from __future__ import annotations
import glob, json, math, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_function import set_seed
from utils.dataset_cfg import DaliaHAR, WESAD
from data.dataset_builder import HARDataset
from models.seqA import (
    DualVideoBottleneckModelV6Downsample)

dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
set_seed(42)
DS = {'daliahar': ('/files1/haodong/data/processed_dalia_activity', DaliaHAR, '../results_3p3/T256/seqA/*'),
      'wesad': ('/files1/haodong/data/WESAD/processed_wesad_activity', WESAD, '../results_3p3/T256_wesad/seqA/*')}


class _V6Cfg:
    def __init__(s, c):
        s.modalities = c.modalities; s.variates = c.variates
        f = c.modalities[0]; s.video_high_dim = c.variates[f]; s.video_low_dim = c.variates[f]


def build(C, cfg, T, NUMC):
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=_V6Cfg(cfg), output_dim=NUMC, input_length=T,
        d_model=C['d_model'], nhead=C['nhead'], num_layers_per_modal=C['num_layers_per_modal'],
        num_layers=C['num_layers'], dropout=C['dropout'], verbose=False,
        video_low_dim=cfg.variates[cfg.modalities[0]], video_high_dim=cfg.variates[cfg.modalities[0]],
        use_bottleneck=True, n_bottlenecks=C['n_bottlenecks'], fusion_layer=C['fusion_layer'],
        use_sparse_moe=False, num_experts=C['num_experts'], expert_k=1, internal_dim=C['internal_dim'],
        bottleneck_head_pos=True, use_sparse_attn=C['use_sparse_attn'], factor=C['base_factor'],
        selector_video_source='high', encoder_video_source='high', no_selector=True,
        use_weighted_factor=False, use_triton=False, num_classes=NUMC,
        sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
        downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
        per_modal_distill=C['per_modal_distill'], per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
        fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full',
        fusion_mlp_ratio=1.0, fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'],
        seq_modality_order=C['seq_modality_order'], seq_random_order=C['seq_random_order'],
        bottleneck_agg_mode='gate', n_fusion_distill=-1).to(dev).float()


def sub(x, mods, sl):
    h = {m: x[:, :, sl[m][0]:sl[m][1]] for m in mods}
    return h, dict(h)


@torch.no_grad()
def acc_with(model, loader, keep, sl):
    cor = n = 0
    for x, y in loader:
        x = x.to(dev).float(); h, l = sub(x, keep, sl)
        o = model(h, l, training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        cor += int((o.argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


for name, (ROOT, CFG, gl) in DS.items():
    cfg = CFG(); MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
    sl, off = {}, 0
    for m in MODS:
        sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
    seqa_dir = sorted(glob.glob(gl))[0]
    C = json.load(open(os.path.join(seqa_dir, 'config.json')))
    T = (cfg.duration * cfg.base_sample_rate) // 1
    val_ds = HARDataset(ROOT, MODS, cfg.val_set, cfg, 'sax', sax_params={'alphabet_size': 20, 'word_length': 1})
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, drop_last=False)
    print(f'\n===== {name} (M={M}) val={cfg.val_set} =====')
    rank_pts = {m: 0.0 for m in MODS}
    for fold in range(3):
        model = build(C, cfg, T, NUMC)
        sd = torch.load(os.path.join(seqa_dir, f'best_model_fold{fold}.pth'), map_location='cpu', weights_only=True)
        inner_sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
        model.load_state_dict(inner_sd, strict=False); model.eval()
        full = acc_with(model, val_loader, MODS, sl)
        loo = {m: full - acc_with(model, val_loader, [x for x in MODS if x != m], sl) for m in MODS}
        order = sorted(MODS, key=lambda m: -loo[m])
        for r, m in enumerate(order):
            rank_pts[m] += (M - r)                                  # Borda points
        print(f'  fold{fold} full_val={full:.4f}  LOO: ' +
              ', '.join(f'{m}={loo[m]:+.3f}' for m in order))
        print(f'         strong-first: {order}')
    consensus = sorted(MODS, key=lambda m: -rank_pts[m])
    k = math.ceil(M / 2)
    print(f'  >>> CONSENSUS strong-first: {consensus}')
    print(f'  >>> branches: 1-mod={consensus[:1]}  partial(top-{k})={consensus[:k]}  full=all({M})')
