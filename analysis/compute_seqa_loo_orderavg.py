"""ORDER-AVERAGED val-LOO importance on seqA (sequential gated-bottleneck fusion).

seqA fuses modalities one-at-a-time, so a modality's contribution depends on its
POSITION in the order (canonical order happens to put chest_ACC last -> possible
position bias). This recomputes LOO importance AVERAGED over K random fusion orders:
for each random permutation pi of the M modalities we set
`model.bottleneck_fusion.seq_modality_order = pi` (eval -> deterministic, uses pi),
measure full_acc(pi) and, for each m, acc with the (M-1)-subset fused in pi's order
(pi reindexed after dropping m). loo_pi[m] = full(pi) - without_m(pi); average over pi.
Compares the order-averaged ranking to the canonical-order ranking.
Run from clean_models/train/:  python compute_seqa_loo_orderavg.py
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

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
K_ORDERS = 8
DS = {'daliahar': ('/files1/haodong/data/processed_dalia_activity', DaliaHAR, '../results_3p3/T256/seqA/*'),
      'wesad': ('/files1/haodong/data/WESAD/processed_wesad_activity', WESAD, '../results_3p3/T256_wesad/seqA/*')}


class _V6Cfg:
    def __init__(s, c):
        s.modalities = c.modalities; s.variates = c.variates
        f = c.modalities[0]; s.video_high_dim = c.variates[f]; s.video_low_dim = c.variates[f]


def build(C, cfg, T, NUMC):
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=_V6Cfg(cfg), output_dim=NUMC, input_length=T, d_model=C['d_model'], nhead=C['nhead'],
        num_layers_per_modal=C['num_layers_per_modal'], num_layers=C['num_layers'], dropout=C['dropout'], verbose=False,
        video_low_dim=cfg.variates[cfg.modalities[0]], video_high_dim=cfg.variates[cfg.modalities[0]],
        use_bottleneck=True, n_bottlenecks=C['n_bottlenecks'], fusion_layer=C['fusion_layer'], use_sparse_moe=False,
        num_experts=C['num_experts'], expert_k=1, internal_dim=C['internal_dim'], bottleneck_head_pos=True,
        use_sparse_attn=C['use_sparse_attn'], factor=C['base_factor'], selector_video_source='high',
        encoder_video_source='high', no_selector=True, use_weighted_factor=False, use_triton=False, num_classes=NUMC,
        sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
        downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
        per_modal_distill=C['per_modal_distill'], per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
        fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full',
        fusion_mlp_ratio=1.0, fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'],
        seq_modality_order=None, seq_random_order=False, bottleneck_agg_mode='gate', n_fusion_distill=-1).to(dev).float()


@torch.no_grad()
def acc(model, loader, keep_mods, sl, order):
    model.bottleneck_fusion.seq_modality_order = order        # positions in the active(=keep) list
    cor = n = 0
    for x, y in loader:
        x = x.to(dev).float()
        h = {m: x[:, :, sl[m][0]:sl[m][1]] for m in keep_mods}
        o = model(h, dict(h), training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        cor += int((o.argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


for name, (ROOT, CFG, gl) in DS.items():
    cfg = CFG(); MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
    sl, off = {}, 0
    for m in MODS:
        sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
    seqa_dir = sorted(glob.glob(gl))[0]; C = json.load(open(os.path.join(seqa_dir, 'config.json')))
    T = cfg.duration * cfg.base_sample_rate
    val_loader = DataLoader(HARDataset(ROOT, MODS, cfg.val_set, cfg, 'sax', sax_params={'alphabet_size': 20, 'word_length': 1}),
                            batch_size=64, shuffle=False, num_workers=4, drop_last=False)
    print(f'\n===== {name} (M={M}) order-averaged over K={K_ORDERS} perms =====')
    borda_oavg, borda_canon = {m: 0.0 for m in MODS}, {m: 0.0 for m in MODS}
    for fold in range(3):
        model = build(C, cfg, T, NUMC)
        sd = torch.load(os.path.join(seqa_dir, f'best_model_fold{fold}.pth'), map_location='cpu', weights_only=True)
        model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()
        # canonical-order LOO (reference)
        full_c = acc(model, val_loader, MODS, sl, list(range(M)))
        loo_c = {}
        for j, m in enumerate(MODS):
            keep = [x for x in MODS if x != m]
            loo_c[m] = full_c - acc(model, val_loader, keep, sl, list(range(M - 1)))
        # order-averaged LOO
        rng = np.random.default_rng(1234 + fold)
        loo_o = {m: [] for m in MODS}
        for _ in range(K_ORDERS):
            pi = list(rng.permutation(M))                      # perm over canonical positions
            full_o = acc(model, val_loader, MODS, sl, pi)
            for j, m in enumerate(MODS):
                keep = [x for x in MODS if x != m]             # canonical order minus m
                # reindex pi onto the (M-1) active list (drop pos j, shift >j down)
                pim = [(i if i < j else i - 1) for i in pi if i != j]
                loo_o[m].append(full_o - acc(model, val_loader, keep, sl, pim))
        loo_o = {m: float(np.mean(v)) for m, v in loo_o.items()}
        order_c = sorted(MODS, key=lambda m: -loo_c[m])
        order_o = sorted(MODS, key=lambda m: -loo_o[m])
        for r, m in enumerate(order_c): borda_canon[m] += (M - r)
        for r, m in enumerate(order_o): borda_oavg[m] += (M - r)
        print(f'  fold{fold}: canonical -> {order_c}')
        print(f'           order-avg -> {order_o}')
        print(f'           LOO(oavg): ' + ', '.join(f'{m}={loo_o[m]:+.3f}' for m in order_o))
    cons_c = sorted(MODS, key=lambda m: -borda_canon[m])
    cons_o = sorted(MODS, key=lambda m: -borda_oavg[m])
    k = math.ceil(M / 2)
    print(f'  >>> CONSENSUS canonical : {cons_c}')
    print(f'  >>> CONSENSUS order-avg : {cons_o}')
    print(f'  >>> order-avg branches  : 1mod={cons_o[:1]} partial(top-{k})={cons_o[:k]} full=all({M})')
    print(f'  >>> SAME as canonical?    {"YES" if cons_c == cons_o else "NO -- ranking changed"}')
