"""TRY: GloVe-regime seqA REGRESSION on CMU-MOSEI -> adaptive modality early-exit curve
in Acc-7 and MAE. The 5-seed GloVe seqA is Acc-2 (classification) so cannot give Acc-7/MAE;
here we train a regression seqA (output_dim=1, MSE) on the same GloVe features (vision35/
audio74/text300, T=50), with modality dropout so modality prefixes are usable, then read the
per-prefix (val-LOO order) Acc-7 & MAE. 3 seeds {21,365,1024}. Output -> mosei/glove_reg_exit.json"""
from __future__ import annotations
import copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
_HERE = os.path.dirname(os.path.abspath(__file__)); _CM = os.path.dirname(_HERE); sys.path.insert(0, _CM)
from utils.helper_function import set_seed
from data.CMU_MOSEI.get_data import get_dataloader, CMUMOSEIDataset, FEATURE_DIMS
from data.CMU_MOSEI.get_data_mmsa import mosei_metrics
from models.seqA import DualVideoBottleneckModelV6Downsample

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODS = list(CMUMOSEIDataset.ALL_MODALITIES); M = len(MODS); T = 50
SEEDS = [21, 365, 1024]; EPOCHS = 60; DROP_P = 0.4


def build():
    class _Cfg:
        modalities = list(MODS); variates = {m: FEATURE_DIMS[m] for m in MODS}
        video_high_dim = FEATURE_DIMS['vision']; video_low_dim = FEATURE_DIMS['vision']
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=_Cfg(), output_dim=1, input_length=T, d_model=128, nhead=8,
        num_layers_per_modal=3, num_layers=3, dropout=0.1, verbose=False,
        video_low_dim=FEATURE_DIMS['vision'], video_high_dim=FEATURE_DIMS['vision'], use_bottleneck=True,
        n_bottlenecks=4, fusion_layer=0, use_sparse_moe=False, num_experts=4, expert_k=1, internal_dim=128,
        bottleneck_head_pos=True, use_sparse_attn=False, factor=10, selector_video_source='high',
        encoder_video_source='high', no_selector=True, use_weighted_factor=False, use_triton=False, num_classes=1,
        sparse_attn_variant='opt', strat_block_size=8, downsample_min_len=64, use_batched_fusion=True,
        per_modal_distill=False, per_modal_downsample_min_len=4, fusion_add_pos_embeds=False,
        fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full', fusion_mlp_ratio=1.0,
        fusion_mode='sequential', seq_depth_flow=False, seq_modality_order=None, seq_random_order=False,
        bottleneck_agg_mode='gate', n_fusion_distill=-1).to(dev).float()


def preload(loader):
    out = []
    for batch in loader:
        *feats, label = batch
        high = {m: feats[2 * j].to(dev).float() for j, m in enumerate(MODS)}
        out.append((high, label.float().to(dev).view(-1)))
    return out


def fwd(model, high, keep):
    o = model({m: high[m] for m in keep}, {m: high[m] for m in keep}, training=False, return_selection_info=False)
    o = o[0] if isinstance(o, tuple) else o
    return o.view(-1)


@torch.no_grad()
def predict(model, batches, keep):
    P, Y = [], []
    for high, y in batches:
        P.append(fwd(model, high, keep).cpu().numpy()); Y.append(y.cpu().numpy())
    return np.concatenate(P), np.concatenate(Y)


per_seed = []   # each: {'order':..., 'curve':[(meanK,acc7,mae) per k]}
gpk = None
for seed in SEEDS:
    set_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    tl, vl, el = get_dataloader(batch_size=32, num_workers=4, modalities=MODS, split_id=0,
                                max_seq_len=T, time_compression_ratio=4, task='regression')
    vb, tb = preload(vl), preload(el)
    model = build(); opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6); mse = nn.MSELoss()
    best_mae, best_state = 1e9, None
    for ep in range(EPOCHS):
        model.train()
        for batch in tl:
            *feats, label = batch
            high = {m: feats[2 * j].to(dev).float() for j, m in enumerate(MODS)}
            y = label.float().to(dev).view(-1)
            kept = [m for m in MODS if torch.rand(1).item() >= DROP_P] or [MODS[int(torch.randint(0, M, (1,)))]]
            o = model({m: high[m] for m in kept}, {m: high[m] for m in kept}, training=True, return_selection_info=False)
            o = o[0] if isinstance(o, tuple) else o
            loss = mse(o.view(-1), y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        pv, yv = predict(model, vb, MODS); vmae = float(np.mean(np.abs(pv - yv)))
        if vmae < best_mae:
            best_mae = vmae; best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    # val-LOO order: removing a strong modality raises val MAE most
    full_mae = best_mae
    loo = {}
    for m in MODS:
        pv, yv = predict(model, vb, [x for x in MODS if x != m]); loo[m] = float(np.mean(np.abs(pv - yv))) - full_mae
    order = sorted(MODS, key=lambda m: -loo[m])
    # per-prefix test acc7 / mae
    curve = []
    for k in range(1, M + 1):
        p, y = predict(model, tb, order[:k]); mm = mosei_metrics(p, y)
        curve.append((float(k), mm['acc7'], mm['mae']))
    per_seed.append({'seed': seed, 'order': order, 'val_mae': full_mae, 'curve': curve})
    print(f'  seed{seed}: val_mae={full_mae:.4f} order={order}  '
          f'acc7 k1->k3 {curve[0][1]:.3f}->{curve[-1][1]:.3f}  mae {curve[0][2]:.3f}->{curve[-1][2]:.3f}')
    if gpk is None:
        h1 = {m: tb[0][0][m][:1] for m in MODS}; gpk = []
        for k in range(1, M + 1):
            fcm = FlopCounterMode(display=False)
            with fcm, torch.no_grad():
                fwd(model, h1, order[:k])
            gpk.append(fcm.get_total_flops() / 1e9)
    del model; torch.cuda.empty_cache()

curves = np.array([[c for c in s['curve']] for s in per_seed])    # [seeds, M, 3]
mean_curve = curves.mean(0).tolist()
out = {'M': M, 'gflops_per_k': gpk, 'mean_curve_k_acc7_mae': mean_curve,
       'per_seed': per_seed, 'seeds': SEEDS}
os.makedirs(os.path.join(_HERE, 'mosei'), exist_ok=True)
json.dump(out, open(os.path.join(_HERE, 'mosei', 'glove_reg_exit.json'), 'w'), indent=2)
print('saved -> results_3p3/mosei/glove_reg_exit.json')
print('mean curve (k, acc7, mae):', [[round(x, 3) for x in r] for r in mean_curve], 'gflops_per_k=', [round(g, 3) for g in gpk])
