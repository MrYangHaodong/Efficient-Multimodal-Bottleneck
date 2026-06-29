"""Compute val leave-one-out (LOO) modality importance on the trained IEMOCAP
seqA model (sequential fusion, no_selector). Mirrors compute_seqa_loo.py for
daliahar/wesad but adapted to IEMOCAP's 6 modalities and list-based batch format.

Method (leakage-free, on the VAL set per fold):
    loo[m] = full_acc - acc_without_m
where acc_without_m drops modality m from the high/low input dicts (seqA fuses
whatever modalities are present). Higher loo[m] => more important. Strong-first
= sort by descending loo[m]. Borda-aggregate the per-fold rankings into a
cross-fold consensus.

Run from clean_models/train/:  python compute_seqa_loo_iemocap.py
"""
from __future__ import annotations
import json, math, os, sys, warnings
warnings.filterwarnings('ignore')
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_function import set_seed
from data.IEMOCAP.get_data import (
    get_dataloader as iemocap_get_dataloader,
    FEATURE_DIMS, NUM_CLASSES, IEMOCAPDataset,
)
from models.seqA import (
    DualVideoBottleneckModelV6Downsample,
)

# ---- least-loaded GPU pick (model is tiny, can share) ----
def _pick_dev():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return torch.device(f'cuda:{best}')

dev = _pick_dev()
set_seed(42)

SEQA_DIR = ('/files1/haodong/MAESTRO_ttn_robustness_old/clean_models/'
            'results_3p3/seqA_iemocap/'
            '2026-06-16_IEMOCAP_v6_seqfusion_late_fusion_cs_seqA_randord')
DATA_ROOT = '/files1/haodong/data/IEMOCAP'
SPLIT_MODE = 'cross_subject'   # exp dir is *_cs_* -> cross-subject splits


class _IEMOCAPV6Cfg:
    """Replicates the cfg shim from the IEMOCAP seqA training script."""
    def __init__(self, modalities):
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        v_dim = FEATURE_DIMS.get('video', 1024)
        self.video_high_dim = v_dim
        self.video_low_dim = v_dim
        self.num_classes = NUM_CLASSES


def build(C, modalities, input_length):
    cfg = _IEMOCAPV6Cfg(modalities)
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=cfg, output_dim=NUM_CLASSES, input_length=input_length,
        d_model=C['d_model'], nhead=C['nhead'],
        num_layers_per_modal=C['num_layers_per_modal'], num_layers=C['num_layers'],
        dropout=C['dropout'], verbose=False,
        video_low_dim=cfg.video_low_dim, video_high_dim=cfg.video_high_dim,
        use_bottleneck=True, n_bottlenecks=C['n_bottlenecks'],
        fusion_layer=C['fusion_layer'], use_sparse_moe=False,
        num_experts=C['num_experts'], expert_k=1, internal_dim=C['internal_dim'],
        bottleneck_head_pos=True, use_sparse_attn=C['use_sparse_attn'],
        factor=C['base_factor'],
        selector_video_source='high', encoder_video_source='high', no_selector=True,
        use_weighted_factor=False, use_triton=False, num_classes=NUM_CLASSES,
        sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
        downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
        per_modal_distill=C['per_modal_distill'],
        per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
        fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256,
        fusion_pos_embed_mode='full', fusion_mlp_ratio=1.0,
        fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'],
        seq_modality_order=C['seq_modality_order'], seq_random_order=C['seq_random_order'],
        bottleneck_agg_mode=C.get('bottleneck_agg_mode', 'gate'),
        n_fusion_distill=C['n_fusion_distill'],
    ).to(dev).float()


def unpack(batch, mods):
    """IEMOCAP collate -> (feat0, feat1, ..., feat_{2M-1}, label[B,1]).
    high[m]=feats[2j], low[m]=feats[2j+1]."""
    *feats, label = batch
    feats = [f.to(dev).float() for f in feats]
    label = label.squeeze(-1).long()
    high = {m: feats[2 * j] for j, m in enumerate(mods)}
    low = {m: feats[2 * j + 1] for j, m in enumerate(mods)}
    return high, low, label


@torch.no_grad()
def acc_with(model, loader, all_mods, keep):
    cor = n = 0
    for batch in loader:
        high, low, y = unpack(batch, all_mods)
        h = {m: high[m] for m in keep}
        l = {m: low[m] for m in keep}
        o = model(h, l, training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        cor += int((o.argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


C = json.load(open(os.path.join(SEQA_DIR, 'config.json')))
MODS = list(C['modalities']); M = len(MODS)
input_length = C['max_seq_len'] if C['max_seq_len'] > 0 else 600

print(f'Device: {dev}')
print(f'\n===== IEMOCAP seqA (M={M})  split_mode={SPLIT_MODE} =====')
print(f'Modalities: {MODS}')

rank_pts = {m: 0.0 for m in MODS}
per_fold_orders = []
for fold in range(3):
    # per-fold cross-subject val loader (same split the main uses)
    _, val_loader, _ = iemocap_get_dataloader(
        data_root=DATA_ROOT, batch_size=64, num_workers=4, modalities=MODS,
        split_id=fold, train_shuffle=False, max_seq_len=C['max_seq_len'],
        time_compression_ratio=C['time_compression_ratio'],
        use_batched_fusion=C['use_batched_fusion'],
        available_sessions=None, split_mode=SPLIT_MODE,
    )

    model = build(C, MODS, input_length)
    sd = torch.load(os.path.join(SEQA_DIR, f'best_model_fold{fold}.pth'),
                    map_location='cpu', weights_only=True)
    inner_sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(inner_sd, strict=False)
    model.eval()

    full = acc_with(model, val_loader, MODS, MODS)
    loo = {m: full - acc_with(model, val_loader, MODS, [x for x in MODS if x != m])
           for m in MODS}
    order = sorted(MODS, key=lambda m: -loo[m])
    per_fold_orders.append(order)
    for r, m in enumerate(order):
        rank_pts[m] += (M - r)   # Borda points
    print(f'\n  fold{fold} full_val={full:.4f}')
    print('    LOO (strong-first): ' + ', '.join(f'{m}={loo[m]:+.4f}' for m in order))
    print(f'    strong-first: {order}')

consensus = sorted(MODS, key=lambda m: -rank_pts[m])
k = math.ceil(M / 2)
print(f'\n  Borda points: ' + ', '.join(f'{m}={rank_pts[m]:.0f}' for m in consensus))
print(f'  >>> CONSENSUS strong-first: {consensus}')
print(f'  >>> branches: 1-mod={consensus[:1]}  partial(top-{k})={consensus[:k]}  full=all({M})')
