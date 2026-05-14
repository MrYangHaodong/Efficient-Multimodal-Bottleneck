"""For a trained selector ckpt, run inference on a DaliaHAR fold and
report **how often each modality lands in the top-K** that the IMS picks.

Compares with the gradient-norm ranking saved by the dyna training JSON
to see whether the learned selector converges to the same modality
priority as the post-hoc gradient profile.

Run:
    python inspect_selector_choices.py \\
        --selector_ckpt=./results_v6_selector/.../best_model_fold0.pth \\
        --fold=0 --target_k=2 \\
        --gradnorm_json=./results_v6_fusion_dyna/.../results_fold0.json \\
        --output_json=/tmp/sel_choices.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed
from utils.dataset_cfg import DaliaHAR
from data.dataset_builder import HARDataset
from multimodal_model.v6_downsample_opt_batched import DualVideoBottleneckModelV6Downsample


class _Cfg:
    def __init__(self, ds):
        self.modalities = ds.modalities
        self.variates = ds.variates
        first = ds.modalities[0]
        self.video_high_dim = ds.variates[first]
        self.video_low_dim = self.video_high_dim


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return v.strip().lower() in ('y', 'yes', 't', 'true', '1')


parser = argparse.ArgumentParser()
parser.add_argument('--selector_ckpt', required=True, type=str)
parser.add_argument('--fold', default=0, type=int, choices=[0, 1, 2])
parser.add_argument('--target_k', default=2, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--gradnorm_json', default=None, type=str,
                    help='Optional: also load grad_norms for side-by-side compare')
parser.add_argument('--output_json', default=None, type=str)
parser.add_argument('--use_train_set', default=False, type=_str2bool,
                    help='If true, evaluate on the train subjects instead of test.')
# Backbone (must match training)
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--n_bottlenecks', default=8, type=int)
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--internal_dim', default=64, type=int)
args = parser.parse_args()


# ---- Setup ----
device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(239)
ds_cfg = DaliaHAR()
T = (ds_cfg.duration * ds_cfg.base_sample_rate) // 2
cfg = _Cfg(ds_cfg)

inner = DualVideoBottleneckModelV6Downsample(
    cfg=cfg, output_dim=ds_cfg.num_classes, input_length=T,
    d_model=args.d_model, nhead=args.nhead,
    num_layers_per_modal=args.num_layers_per_modal,
    num_layers=args.num_layers, dropout=0.1, verbose=False,
    video_low_dim=cfg.video_low_dim, video_high_dim=cfg.video_high_dim,
    use_bottleneck=True, n_bottlenecks=args.n_bottlenecks, fusion_layer=0,
    use_sparse_moe=False, num_experts=args.num_experts, expert_k=1,
    internal_dim=args.internal_dim, bottleneck_head_pos=True,
    use_sparse_attn=True, factor=args.base_factor,
    selector_video_source='high', encoder_video_source='high',
    no_selector=False, use_weighted_factor=True, use_triton=False,
    num_classes=ds_cfg.num_classes,
    lambda_probe=0.05, lambda_diversity=0.2, lambda_reinforce=0.0, lambda_sparsity=0.0,
    sparse_attn_variant='opt', strat_block_size=8,
    downsample_min_len=4, use_batched_fusion=True,
    per_modal_distill=False, per_modal_downsample_min_len=4,
    use_interaction_matrix=False, use_holo_bias=False, holo_scale=1.0,
).to(device).float()
inner.top_k = args.target_k

# Load ckpt (state_dict may have 'model.' prefix from wrapper-saved)
sd = torch.load(args.selector_ckpt, map_location='cpu', weights_only=False)
sd_fixed = {(k[6:] if k.startswith('model.') else k): v for k, v in sd.items()}
missing, unexpected = inner.load_state_dict(sd_fixed, strict=False)
print(f"Loaded {args.selector_ckpt}")
print(f"  missing={len(missing)}  unexpected={len(unexpected)}")

# Dataset
fold_cfg = ds_cfg.folds[args.fold]
subjects = fold_cfg['train_set'] if args.use_train_set else fold_cfg['eval_set']
ds = HARDataset('/files1/haodong/data/processed_dalia_activity',
                ds_cfg.modalities, subjects, ds_cfg, args.transform)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, drop_last=False)

# Channel slicing
mod_slices = {}
offset = 0
for m in ds_cfg.modalities:
    mod_slices[m] = (offset, offset + ds_cfg.variates[m])
    offset += ds_cfg.variates[m]


# ---- Run inference, count selected_modalities ----
counts = Counter()
n_batches = 0
inner.eval()
with torch.no_grad():
    for x, y in loader:
        x = x.to(device).float()
        high = {m: x[:, :, s:e] for m, (s, e) in mod_slices.items()}
        low = high
        # We need top-K, so call with return_selection_info=True and read sel
        result = inner(high, low, training=False, return_selection_info=True)
        # signature: (output, primary_idx, modality_weights, selected_modalities)
        sel = result[3]
        if sel is None:    # top_k == None
            sel = list(ds_cfg.modalities)
        for m in sel:
            counts[m] += 1
        n_batches += 1


# ---- Normalise into frequencies (sums to K) ----
freqs = {m: counts.get(m, 0) / n_batches for m in ds_cfg.modalities}
ranked = sorted(ds_cfg.modalities, key=lambda m: -freqs[m])

print(f"\nFold {args.fold}, K={args.target_k}, n_batches={n_batches}, "
      f"set={'train' if args.use_train_set else 'eval'}")
print(f"\n=== Selector empirical top-K frequency (sums to {args.target_k:.0f}) ===")
for i, m in enumerate(ranked, 1):
    bar = '█' * int(freqs[m] * 40)
    print(f"  {i}. {m:<14s}  {freqs[m]:>5.2%}  {bar}")

# ---- Compare with gradnorm if provided ----
gradnorm_ranked = None
if args.gradnorm_json:
    j = json.load(open(args.gradnorm_json))
    gn = j.get('final_grad_norms', {})
    if gn:
        gradnorm_ranked = sorted(ds_cfg.modalities, key=lambda m: -gn.get(m, 0))
        print(f"\n=== Gradnorm ranking (from {args.gradnorm_json}) ===")
        for i, m in enumerate(gradnorm_ranked, 1):
            print(f"  {i}. {m:<14s}  grad_norm={gn.get(m, 0):.6f}")

        # Concordance: for each modality, position in selector ranking vs gradnorm
        print(f"\n=== Rank concordance ===")
        print(f"  {'modality':<14s}  {'selector rank':>14s}  {'gradnorm rank':>14s}  {'Δ':>3s}")
        for m in ds_cfg.modalities:
            sr = ranked.index(m) + 1
            gr = gradnorm_ranked.index(m) + 1
            print(f"  {m:<14s}  {sr:>14d}  {gr:>14d}  {gr-sr:>+3d}")

        # Top-K overlap: how many of the top-K modalities selector picks match gradnorm's top-K
        K = args.target_k
        sel_top = set(ranked[:K])
        gn_top = set(gradnorm_ranked[:K])
        overlap = len(sel_top & gn_top)
        print(f"\nTop-{K} overlap: {overlap}/{K} ({100*overlap/K:.0f}%)")
        print(f"  selector top-{K}: {ranked[:K]}")
        print(f"  gradnorm top-{K}: {gradnorm_ranked[:K]}")


# ---- Save JSON ----
if args.output_json:
    out = {
        'fold': args.fold,
        'target_k': args.target_k,
        'n_batches': n_batches,
        'set': 'train' if args.use_train_set else 'eval',
        'modality_freqs': freqs,
        'selector_ranking': ranked,
        'gradnorm_ranking': gradnorm_ranked,
        'selector_ckpt': args.selector_ckpt,
        'gradnorm_json': args.gradnorm_json,
    }
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=4)
    print(f"\nWrote: {args.output_json}")
