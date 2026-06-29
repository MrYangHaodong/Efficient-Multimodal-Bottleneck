"""val-LOO modality importance on the trained PAMAP2 seqA (same method/format as
compute_seqa_loo.py for daliahar/wesad). PAMAP2 is HARDataset-like (concatenated
[B,T,C]) so the channel-slice LOO applies directly; only the loader/cfg differ.
Run from clean_models/train/:  python compute_seqa_loo_pamap2.py
"""
from __future__ import annotations
import glob, json, math, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_function import set_seed
from utils.dataset_cfg import PAMAP2
from data.pamap2_crosssubject import load_pamap2_subjects
from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import (
    DualVideoBottleneckModelV6Downsample)

dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
set_seed(42)
cfg = PAMAP2(); MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
# this seqA_pamap2 ckpt was trained at T=128 (raw 512 SAX'd with word_length=4)
WL = 4; T = cfg.sequence_length // WL                      # 512 // 4 = 128 (matches the ckpt's pos cache)
seqa_dir = sorted(glob.glob('../results_3p3/seqA_pamap2/*'))[0]
C = json.load(open(os.path.join(seqa_dir, 'config.json')))
TRANSFORM = 'sax'                                          # clean tokens (like daliahar/wesad LOO)
sl, off = {}, 0
for m in MODS:
    sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
val_loader = DataLoader(load_pamap2_subjects(cfg.cross_subject_dir, MODS, cfg.val_set, cfg, TRANSFORM,
                                             sax_params={'alphabet_size': 20, 'word_length': WL}),
                        batch_size=64, shuffle=False, num_workers=4, drop_last=False)


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


@torch.no_grad()
def acc_with(model, keep):
    cor = n = 0
    for x, y in val_loader:
        x = x.to(dev).float()
        h = {m: x[:, :, sl[m][0]:sl[m][1]] for m in keep}
        o = model(h, dict(h), training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        cor += int((o.argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


print(f'===== PAMAP2 (M={M}) val={cfg.val_set} transform={TRANSFORM} =====')
borda = {m: 0.0 for m in MODS}
for fold in range(3):
    model = build()
    sd = torch.load(os.path.join(seqa_dir, f'best_model_fold{fold}.pth'), map_location='cpu', weights_only=True)
    model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()
    full = acc_with(model, MODS)
    loo = {m: full - acc_with(model, [x for x in MODS if x != m]) for m in MODS}
    order = sorted(MODS, key=lambda m: -loo[m])
    for r, m in enumerate(order):
        borda[m] += (M - r)
    print(f'  fold{fold} full_val={full:.4f}  LOO: ' + ', '.join(f'{m}={loo[m]:+.3f}' for m in order))
    print(f'         strong-first: {order}')
consensus = sorted(MODS, key=lambda m: -borda[m]); k = math.ceil(M / 2)
print(f'  >>> CONSENSUS strong-first: {consensus}')
print(f'  >>> branches: 1mod={consensus[:1]} partial(top-{k})={consensus[:k]} full=all({M})')
