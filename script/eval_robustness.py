"""Evaluate PL-GRPO selector vs gradnorm-static vs fusion-K=5 baselines under
modality missing / noise corruption at TEST time.

Three methods, all use the SAME frozen fusion ckpt (stage-1 fusion_only):
  1. PL-GRPO selector (K=4): learned, input-conditional top-K
  2. Gradnorm static (K=4): forced same 4 modalities every batch (from fusion_dyna)
  3. Fusion K=5: no selector, all 5 modalities

Corruption scenarios:
  * clean
  * zero one modality at a time (5 scenarios)
  * Gaussian noise (sigma=2.0) on one modality (5 scenarios)

Outputs per (method, fold, scenario): test_acc, test_f1.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

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


def build_v6(ds_cfg, T, K, no_selector, device):
    cfg = _Cfg(ds_cfg)
    inner = DualVideoBottleneckModelV6Downsample(
        cfg=cfg, output_dim=ds_cfg.num_classes, input_length=T,
        d_model=64, nhead=8, num_layers_per_modal=2, num_layers=4, dropout=0.1, verbose=False,
        video_low_dim=cfg.video_low_dim, video_high_dim=cfg.video_high_dim,
        use_bottleneck=True, n_bottlenecks=8, fusion_layer=0, use_sparse_moe=False,
        num_experts=4, expert_k=1, internal_dim=64, bottleneck_head_pos=True,
        use_sparse_attn=True, factor=3, selector_video_source='high', encoder_video_source='high',
        no_selector=no_selector, use_weighted_factor=True, use_triton=False,
        num_classes=ds_cfg.num_classes,
        lambda_probe=0.05, lambda_diversity=0.2,
        lambda_reinforce=0.0, lambda_sparsity=0.0,
        sparse_attn_variant='opt', strat_block_size=8,
        downsample_min_len=4, use_batched_fusion=True,
        per_modal_distill=False, per_modal_downsample_min_len=4,
        use_interaction_matrix=False, use_holo_bias=False, holo_scale=1.0,
    ).to(device).float()
    inner.top_k = K
    return inner


def load_ckpt(model, path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    # strip 'model.' prefix from grpo-wrapper-saved ckpts
    sd_fixed = {(k[6:] if k.startswith('model.') else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd_fixed, strict=False)
    return miss, unexp


def apply_corruption(x, scenario, mod_slices, rng):
    """Apply test-time corruption. x: (B, T, total_C)."""
    if scenario == 'clean':
        return x
    parts = scenario.split('_', 1)
    op, mod = parts[0], parts[1]
    if mod not in mod_slices:
        return x
    s, e = mod_slices[mod]
    x = x.clone()
    if op == 'zero':
        x[:, :, s:e] = 0.0
    elif op == 'noise':
        # σ=2.0 Gaussian additive noise (input is normalized; this is large)
        noise = torch.randn(x.shape[0], x.shape[1], e - s, device=x.device, generator=rng) * 2.0
        x[:, :, s:e] = x[:, :, s:e] + noise
    return x


@torch.no_grad()
def eval_method(inner, loader, device, mod_slices, modalities,
                scenario, forced_topk=None, rng_seed=42):
    """Run inference, return acc, f1, selected_freq dict (if learnable selector)."""
    inner.eval()
    g = torch.Generator(device=device).manual_seed(rng_seed)
    preds, labels = [], []
    sel_counter = {m: 0 for m in modalities}
    n_batches = 0
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device)
        x = apply_corruption(x, scenario, mod_slices, g)
        high = {m: x[:, :, s:e] for m, (s, e) in mod_slices.items()}
        low = high
        kw = {}
        if forced_topk is not None:
            kw['override_selected_modalities'] = list(forced_topk)
        out = inner(high, low, training=False, return_selection_info=True, **kw)
        logits = out[0]
        # selected info if available
        if isinstance(out, tuple) and len(out) >= 4 and out[3] is not None:
            for m in out[3]:
                if m in sel_counter:
                    sel_counter[m] += 1
        _, p = torch.max(logits, 1)
        preds.append(p.cpu())
        labels.append(y.cpu())
        n_batches += 1
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, average='macro'))
    freq = {m: sel_counter[m] / max(n_batches, 1) for m in modalities}
    return acc, f1, freq


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cuda_pick', default='cuda:0', type=str)
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--fusion_dir', required=True,
                   help='dir containing best_model_fold{0,1,2}.pth (fusion-only, no selector)')
    p.add_argument('--selector_dir', required=True,
                   help='dir containing best_model_fold{0,1,2}.pth (PL-GRPO selector at K=K)')
    p.add_argument('--dyna_dir', required=True,
                   help='dir containing results_fold{0,1,2}.json with final_grad_norms')
    p.add_argument('--K', default=4, type=int,
                   help='top-K for selector and gradnorm-static methods')
    p.add_argument('--output_json', required=True)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
    ds_cfg = DaliaHAR()
    T = (ds_cfg.duration * ds_cfg.base_sample_rate) // 2
    modalities = ds_cfg.modalities

    mod_slices = {}
    offset = 0
    for m in modalities:
        mod_slices[m] = (offset, offset + ds_cfg.variates[m])
        offset += ds_cfg.variates[m]

    scenarios = ['clean']
    for m in modalities:
        scenarios.append(f'zero_{m}')
    for m in modalities:
        scenarios.append(f'noise_{m}')

    methods = ['fusion_k5', 'gradnorm_static_kK', 'plgrpo_selector_kK']

    results = {}  # results[method][fold][scenario] = {acc, f1, freq}

    for fold in [0, 1, 2]:
        print(f"\n========== fold {fold} ==========")
        # Build test loader once per fold
        fold_cfg = ds_cfg.folds[fold]
        ds = HARDataset('/files1/haodong/data/processed_dalia_activity',
                        modalities, fold_cfg['eval_set'], ds_cfg, 'sax')
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, drop_last=False)

        # Load gradnorm ranking
        with open(os.path.join(args.dyna_dir, f'results_fold{fold}.json')) as f:
            gn = json.load(f)['final_grad_norms']
        gradnorm_topK = sorted(modalities, key=lambda m: -gn.get(m, 0.0))[:args.K]
        print(f"  gradnorm top-{args.K}: {gradnorm_topK}")

        # ---- fusion K=5 ----
        print(f"  [fusion K=5] loading {args.fusion_dir}/best_model_fold{fold}.pth")
        inner = build_v6(ds_cfg, T, K=5, no_selector=True, device=device)
        load_ckpt(inner, os.path.join(args.fusion_dir, f'best_model_fold{fold}.pth'))

        for scenario in scenarios:
            acc, f1, _ = eval_method(inner, loader, device, mod_slices, modalities, scenario)
            results.setdefault('fusion_k5', {}).setdefault(str(fold), {})[scenario] = {
                'acc': acc, 'f1': f1}
            print(f"    fusion_k5   {scenario:<20s}  acc={acc:.4f}  f1={f1:.4f}")

        # ---- gradnorm static K=K ----
        # Use fusion ckpt with selector head fresh (but bypass via override) — same inner
        # Reload (state may be unchanged but safe to re-build with selector enabled)
        print(f"  [gradnorm static K={args.K}] (selector head not used, force topK via override)")
        inner = build_v6(ds_cfg, T, K=args.K, no_selector=False, device=device)
        load_ckpt(inner, os.path.join(args.fusion_dir, f'best_model_fold{fold}.pth'))
        for scenario in scenarios:
            acc, f1, freq = eval_method(inner, loader, device, mod_slices, modalities,
                                        scenario, forced_topk=gradnorm_topK)
            results.setdefault('gradnorm_static', {}).setdefault(str(fold), {})[scenario] = {
                'acc': acc, 'f1': f1}
            print(f"    gradnorm    {scenario:<20s}  acc={acc:.4f}  f1={f1:.4f}")

        # ---- PL-GRPO selector K=K ----
        print(f"  [PL-GRPO selector K={args.K}] loading {args.selector_dir}/best_model_fold{fold}.pth")
        inner = build_v6(ds_cfg, T, K=args.K, no_selector=False, device=device)
        load_ckpt(inner, os.path.join(args.selector_dir, f'best_model_fold{fold}.pth'))
        for scenario in scenarios:
            acc, f1, freq = eval_method(inner, loader, device, mod_slices, modalities, scenario)
            results.setdefault('plgrpo_selector', {}).setdefault(str(fold), {})[scenario] = {
                'acc': acc, 'f1': f1, 'sel_freq': freq}
            top_sel = sorted(freq.items(), key=lambda x: -x[1])[:args.K]
            top_sel_str = ' '.join(f"{m}:{v:.2f}" for m, v in top_sel)
            print(f"    plgrpo      {scenario:<20s}  acc={acc:.4f}  f1={f1:.4f}  sel: {top_sel_str}")

    # Save
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {args.output_json}")


if __name__ == '__main__':
    main()
