"""Ablation: learned-selector top-K  vs  random-K  on DaliaHAR.

Given:
  - a fusion-only checkpoint (stage-1, ``no_selector=True`` training)
  - a selector-finetune checkpoint (stage-2, fusion frozen + IMS trained)

Both share the SAME fusion weights (stage 2 froze fusion).  The only
difference is which K modalities reach the fusion at inference time:

  * **Scenario A — learned selector**: rebuild V6Downsample with
    ``no_selector=False``, load the selector-finetune ckpt, let the IMS
    pick top-K confidence-weighted modalities per batch.

  * **Scenario B — random K**: rebuild V6Downsample with
    ``no_selector=True``, load the fusion-only ckpt, and randomly drop
    (M − K) modalities per batch.  Repeat over ``--n_random_seeds`` and
    report mean ± std so the comparison isn't seed-noise.

Output: a side-by-side table of clean test accuracy + macro-F1 for
each K value in ``--ks`` (default = [1, 2, 3, 4, M]).

Run:
    cd /files1/haodong/MAESTRO_ttn_robustness_old/models_export
    python eval_selector_vs_random_topk.py \\
        --fold=0 --cuda_pick=cuda:0 \\
        --fusion_ckpt=./results_v6_fusion/2026-05-13_DaliaHAR_v6_fusion_only/best_model_fold0.pth \\
        --selector_ckpt=./results_v6_selector/<date>_DaliaHAR_v6_selector_finetune/best_model_fold0.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed
from utils.dataset_cfg import DaliaHAR
from data.dataset_builder import HARDataset

from multimodal_model.v6_downsample_opt_batched import DualVideoBottleneckModelV6Downsample


# ============================ cfg adaptor ============================

class _DaliaV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


# ============================ Wrappers ============================

class _ConcatWrapper(torch.nn.Module):
    """Channel-split wrapper — same shape as both training scripts."""
    def __init__(self, v6_model, modalities, variates):
        super().__init__()
        self.model = v6_model
        self.modalities = list(modalities)
        self.variates = variates
        self._mod_slices = {}
        offset = 0
        for m in modalities:
            n = variates[m]
            self._mod_slices[m] = (offset, offset + n)
            offset += n

    def _split(self, x):
        high, low = {}, {}
        for m in self.modalities:
            s, e = self._mod_slices[m]
            high[m] = x[:, :, s:e]
            low[m] = x[:, :, s:e]
        return high, low

    def forward_selector(self, x):
        """Scenario A: full M modalities in, model's IMS picks K internally."""
        high, low = self._split(x)
        result = self.model(high, low, training=False,
                            return_selection_info=False)
        return result[0] if isinstance(result, tuple) else result

    def forward_random_k(self, x, k, rng):
        """Scenario B: random K-subset of modalities reaches the fusion."""
        high, low = self._split(x)
        if k < len(self.modalities):
            # rng is a torch.Generator for reproducibility across batches.
            perm = torch.randperm(len(self.modalities), generator=rng).tolist()
            keep = [self.modalities[i] for i in perm[:k]]
            high = {m: high[m] for m in keep}
            low = {m: low[m] for m in keep}
        result = self.model(high, low, training=False,
                            return_selection_info=False)
        return result[0] if isinstance(result, tuple) else result


# ============================ Model builders ============================

def _build_model(no_selector, dataset_cfg, args, device):
    v6_cfg = _DaliaV6Cfg(dataset_cfg)
    input_length = ((dataset_cfg.duration * dataset_cfg.base_sample_rate) // 2
                    if args.transform == 'sax'
                    else dataset_cfg.duration * dataset_cfg.base_sample_rate)

    inner = DualVideoBottleneckModelV6Downsample(
        cfg=v6_cfg,
        output_dim=dataset_cfg.num_classes,
        input_length=input_length,
        d_model=args.d_model, nhead=args.nhead,
        num_layers_per_modal=args.num_layers_per_modal,
        num_layers=args.num_layers, dropout=args.dropout, verbose=False,
        video_low_dim=v6_cfg.video_low_dim, video_high_dim=v6_cfg.video_high_dim,
        use_bottleneck=True, n_bottlenecks=args.n_bottlenecks,
        fusion_layer=args.fusion_layer, use_sparse_moe=False,
        num_experts=args.num_experts, expert_k=1,
        internal_dim=args.internal_dim, bottleneck_head_pos=True,
        use_sparse_attn=args.use_sparse_attn, factor=args.base_factor,
        selector_video_source='high', encoder_video_source='high',
        no_selector=no_selector,
        use_weighted_factor=not no_selector,
        use_triton=False,
        num_classes=dataset_cfg.num_classes,
        sparse_attn_variant=args.sparse_attn_variant,
        strat_block_size=args.strat_block_size,
        downsample_min_len=args.downsample_min_len,
        use_batched_fusion=True,
        per_modal_distill=False,
        per_modal_downsample_min_len=4,
        # Selector kwargs (ignored when no_selector=True)
        use_interaction_matrix=args.use_interaction_matrix,
        use_holo_bias=args.use_holo_bias, holo_scale=args.holo_scale,
        lambda_probe=args.lambda_probe,
        lambda_diversity=args.lambda_diversity,
        lambda_reinforce=args.lambda_reinforce,
        lambda_sparsity=0.0,
    )
    return _ConcatWrapper(inner, dataset_cfg.modalities, dataset_cfg.variates).to(device).float()


# ============================ Eval routines ============================

@torch.no_grad()
def eval_selector(model, loader, device, k):
    """Run scenario A: model's IMS picks top-K."""
    model.eval()
    inner = model.model
    inner.top_k = (k if k < len(model.modalities) else None)
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device).float(), y.to(device)
        logits = model.forward_selector(x)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y.cpu())
    p = torch.cat(all_preds).numpy()
    l = torch.cat(all_labels).numpy()
    return accuracy_score(l, p), f1_score(l, p, average='macro')


@torch.no_grad()
def eval_random_k(model, loader, device, k, seed):
    """Run scenario B: random K-subset per batch."""
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(seed)
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device).float(), y.to(device)
        logits = model.forward_random_k(x, k, rng)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y.cpu())
    p = torch.cat(all_preds).numpy()
    l = torch.cat(all_labels).numpy()
    return accuracy_score(l, p), f1_score(l, p, average='macro')


# ============================ Argparse ============================

def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in ('y', 'yes', 't', 'true', '1'):
        return True
    if s in ('n', 'no', 'f', 'false', '0', ''):
        return False
    raise argparse.ArgumentTypeError(f'Boolean expected, got {v!r}')


parser = argparse.ArgumentParser(
    description='Compare learned-selector top-K vs random K (DaliaHAR).')

parser.add_argument('--fusion_ckpt', required=True, type=str)
parser.add_argument('--selector_ckpt', required=True, type=str)
parser.add_argument('--fold', default=0, type=int, choices=[0, 1, 2])
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--ks', nargs='+', type=int, default=None,
                    help='List of K values to evaluate; default = [1, 2, 3, 4, M].')
parser.add_argument('--n_random_seeds', default=5, type=int,
                    help='Number of seeds for random-K eval (averaged).')
parser.add_argument('--output_json', default=None, type=str,
                    help='Optional JSON path to dump the results table.')

# V6 backbone knobs — must match training-time config.
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--internal_dim', default=64, type=int)
parser.add_argument('--n_bottlenecks', default=8, type=int)
parser.add_argument('--fusion_layer', default=0, type=int)
parser.add_argument('--use_sparse_attn', type=_str2bool, default=True)
parser.add_argument('--sparse_attn_variant', default='opt', type=str,
                    choices=['opt', 'opt_strat', 'opt_strat_blk'])
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=4, type=int)
parser.add_argument('--use_interaction_matrix', default=False, type=_str2bool)
parser.add_argument('--use_holo_bias', default=False, type=_str2bool)
parser.add_argument('--holo_scale', default=1.0, type=float)
parser.add_argument('--lambda_probe', default=0.05, type=float)
parser.add_argument('--lambda_diversity', default=0.2, type=float)
parser.add_argument('--lambda_reinforce', default=0.1, type=float)

args = parser.parse_args()


# ============================ Setup ============================

device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed_num)

print(f"Device: {device}")
print(f"Fold {args.fold}")
print(f"Fusion ckpt    : {args.fusion_ckpt}")
print(f"Selector ckpt  : {args.selector_ckpt}")

root_dir = '/files1/haodong/data/processed_dalia_activity'   # UPDATE PATH
dataset_cfg = DaliaHAR()
modalities = dataset_cfg.modalities
M = len(modalities)

fold_cfg = dataset_cfg.folds[args.fold]
eval_subjects = fold_cfg['eval_set']
print(f"Eval subjects: {eval_subjects}")

eval_ds = HARDataset(root_dir, modalities, eval_subjects, dataset_cfg, args.transform)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, drop_last=True)

ks = args.ks if args.ks is not None else [1, 2, 3, 4, M]
ks = sorted(set(k for k in ks if 1 <= k <= M))


# ============================ Build + load: Scenario A (selector) ============================

print("\n" + "=" * 78)
print(f"Building Scenario A — V6Downsample WITH selector (no_selector=False)")
print("=" * 78)
model_A = _build_model(no_selector=False, dataset_cfg=dataset_cfg,
                       args=args, device=device)
ckpt_A = torch.load(args.selector_ckpt, map_location='cpu', weights_only=False)
missing, unexpected = model_A.load_state_dict(ckpt_A, strict=False)
sel_missing = [k for k in missing if 'modality_selector' in k]
fus_missing = [k for k in missing if 'modality_selector' not in k]
print(f"  loaded {len(ckpt_A)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
if fus_missing:
    print(f"  ⚠ fusion-side missing in selector ckpt: {fus_missing[:5]}")
if unexpected:
    print(f"  ⚠ unexpected keys: {unexpected[:5]}")


# ============================ Build + load: Scenario B (fusion-only) ============================

print("\n" + "=" * 78)
print(f"Building Scenario B — V6Downsample WITHOUT selector (no_selector=True)")
print("=" * 78)
model_B = _build_model(no_selector=True, dataset_cfg=dataset_cfg,
                       args=args, device=device)
ckpt_B = torch.load(args.fusion_ckpt, map_location='cpu', weights_only=False)
missing, unexpected = model_B.load_state_dict(ckpt_B, strict=False)
print(f"  loaded {len(ckpt_B)} tensors; missing={len(missing)} unexpected={len(unexpected)}")


# ============================ Run the sweep ============================

print("\n" + "=" * 78)
print(f"Evaluation sweep — fold {args.fold}, M={M}, "
      f"random-K seeds={args.n_random_seeds}")
print("=" * 78)

results = {}
header = f"  {'K':>3}  {'sel_acc':>9} {'sel_f1':>9}    " \
         f"{'rnd_acc(mean±std)':>22} {'rnd_f1(mean±std)':>22}    {'Δacc':>7}"
print(header)
print("  " + "-" * (len(header) - 2))

for k in ks:
    # Scenario A: deterministic learned top-K (run once)
    a_acc, a_f1 = eval_selector(model_A, eval_loader, device, k)

    # Scenario B: random K, average over n seeds
    rnd_accs, rnd_f1s = [], []
    for s in range(args.n_random_seeds):
        rb_acc, rb_f1 = eval_random_k(model_B, eval_loader, device, k,
                                      seed=args.seed_num + s)
        rnd_accs.append(rb_acc); rnd_f1s.append(rb_f1)
    rnd_acc_mean, rnd_acc_std = float(np.mean(rnd_accs)), float(np.std(rnd_accs))
    rnd_f1_mean, rnd_f1_std = float(np.mean(rnd_f1s)), float(np.std(rnd_f1s))
    delta_acc = a_acc - rnd_acc_mean

    results[k] = {
        'selector': {'acc': a_acc, 'f1': a_f1},
        'random_k': {
            'acc_mean': rnd_acc_mean, 'acc_std': rnd_acc_std,
            'f1_mean': rnd_f1_mean, 'f1_std': rnd_f1_std,
            'acc_seeds': rnd_accs, 'f1_seeds': rnd_f1s,
        },
        'delta_acc': delta_acc,
    }
    print(f"  {k:>3}  {a_acc:>9.4f} {a_f1:>9.4f}    "
          f"{rnd_acc_mean:.4f}±{rnd_acc_std:.4f}      "
          f"{rnd_f1_mean:.4f}±{rnd_f1_std:.4f}      {delta_acc:>+7.4f}")

print()
print("Δacc = sel_acc − rnd_acc_mean  (positive ⇒ learned selector beats random)")

# Save JSON dump if requested.
if args.output_json:
    out = {
        'fold': args.fold,
        'M': M, 'ks': ks,
        'n_random_seeds': args.n_random_seeds,
        'fusion_ckpt': args.fusion_ckpt,
        'selector_ckpt': args.selector_ckpt,
        'results': {str(k): v for k, v in results.items()},
    }
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=4)
    print(f"\nWrote: {args.output_json}")
