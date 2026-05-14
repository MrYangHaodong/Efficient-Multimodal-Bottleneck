"""Top-K modality eval driven by **per-modality gradient norms**
collected during `main_v6_daliahar_fusion_only_dyna.py` training.

The dyna script saves a JSON under each ``results_fold{N}.json`` that
contains a ``final_grad_norms`` dict (and a per-epoch ``profile_log``).
Modalities whose ``input_projectors[m].weight`` accumulated **larger**
gradient norms during training are the ones the model leaned on most —
so we treat ``grad_norm_m`` as a learning-free proxy for "importance"
and keep the top-K modalities at test time.

For each K in ``--ks`` (default [2, 3, 4, 5]):

  * **gradnorm top-K**  — keep the K modalities with the highest
                          ``final_grad_norms`` value.
  * **random K (n seeds)** — for reference / lower-bound baseline.

Report clean test acc + macro-F1 for both, side by side, plus the
delta — so you can see whether the saved gradient profile carries useful
selection information.

Run:
    cd /files1/haodong/MAESTRO_ttn_robustness_old/models_export
    python eval_v6_dyna_gradnorm_topk.py \\
        --fold=0 --cuda_pick=cuda:0 \\
        --ckpt=./results_v6_fusion_dyna/<exp>/best_model_fold0.pth \\
        --results_json=./results_v6_fusion_dyna/<exp>/results_fold0.json
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
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed
from utils.dataset_cfg import DaliaHAR
from data.dataset_builder import HARDataset
from multimodal_model.v6_downsample_opt_batched import DualVideoBottleneckModelV6Downsample


# ============================ V6 cfg adaptor ============================

class _DaliaV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


# ============================ Wrapper (eval-only) ============================

class _EvalWrapper(nn.Module):
    """Channel-split wrapper that lets the caller pass a fixed subset
    of modalities per forward (matches what V6Downsample does internally
    when only K dict-keys are populated)."""

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

    def _split(self, x, keep_mods):
        high, low = {}, {}
        for m in keep_mods:
            s, e = self._mod_slices[m]
            high[m] = x[:, :, s:e]
            low[m] = x[:, :, s:e]
        return high, low

    def forward(self, x, keep_modalities):
        high, low = self._split(x, keep_modalities)
        out = self.model(high, low, training=False,
                         return_selection_info=False)
        return out[0] if isinstance(out, tuple) else out


# ============================ Eval loop ============================

@torch.no_grad()
def evaluate(model, loader, device, keep_modalities):
    model.eval()
    all_p, all_y = [], []
    for x, y in loader:
        x, y = x.to(device).float(), y.to(device)
        logits = model(x, keep_modalities)
        all_p.append(logits.argmax(dim=1).cpu())
        all_y.append(y.cpu())
    p = torch.cat(all_p).numpy()
    yy = torch.cat(all_y).numpy()
    return accuracy_score(yy, p), f1_score(yy, p, average='macro')


@torch.no_grad()
def evaluate_random_k(model, loader, device, k, seed):
    """K random modalities per BATCH (different draw each batch)."""
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(seed)
    all_p, all_y = [], []
    for x, y in loader:
        x, y = x.to(device).float(), y.to(device)
        perm = torch.randperm(len(model.modalities), generator=rng).tolist()
        keep = [model.modalities[i] for i in perm[:k]]
        logits = model(x, keep)
        all_p.append(logits.argmax(dim=1).cpu())
        all_y.append(y.cpu())
    p = torch.cat(all_p).numpy()
    yy = torch.cat(all_y).numpy()
    return accuracy_score(yy, p), f1_score(yy, p, average='macro')


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
    description='Gradient-norm top-K eval of a dyna-trained V6 fusion model.')

parser.add_argument('--ckpt', required=True, type=str,
                    help='Path to dyna best_model_fold*.pth.')
parser.add_argument('--results_json', required=True, type=str,
                    help='Path to the corresponding results_fold*.json '
                         '(needs final_grad_norms dict).')
parser.add_argument('--fold', default=0, type=int, choices=[0, 1, 2])
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--ks', nargs='+', type=int, default=[2, 3, 4, 5])
parser.add_argument('--n_random_seeds', default=5, type=int)
parser.add_argument('--output_json', default=None, type=str)

# Source of gradient norms inside the JSON.
parser.add_argument('--gradnorm_source', default='final',
                    choices=['final', 'profile_last', 'profile_mean'],
                    help="'final' = results_json['final_grad_norms']; "
                         "'profile_last' = last entry of profile_log; "
                         "'profile_mean' = average over profile_log entries.")

# Selection direction (high grad = important by default).
parser.add_argument('--prefer', default='high', choices=['high', 'low'],
                    help="'high' keeps modalities with the LARGEST grad "
                         "norms (the ones the model relied on most). "
                         "'low' keeps the smallest — useful as an ablation.")

# V6 backbone config — must match training-time config.
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

args = parser.parse_args()


# ============================ Setup ============================

device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed_num)

print(f"Device: {device}")
print(f"Fold   : {args.fold}")
print(f"Ckpt   : {args.ckpt}")
print(f"JSON   : {args.results_json}")

# Load saved grad norms from the dyna training JSON.
with open(args.results_json) as f:
    results = json.load(f)

if args.gradnorm_source == 'final':
    grad_norms = results.get('final_grad_norms', {})
elif args.gradnorm_source == 'profile_last':
    log = results.get('profile_log', [])
    grad_norms = log[-1].get('grad_norms', {}) if log else {}
elif args.gradnorm_source == 'profile_mean':
    log = results.get('profile_log', [])
    keys = set()
    for r in log:
        keys.update(r.get('grad_norms', {}).keys())
    grad_norms = {}
    for k in keys:
        vs = [r['grad_norms'][k] for r in log if k in r.get('grad_norms', {})]
        grad_norms[k] = float(np.mean(vs)) if vs else 0.0

if not grad_norms:
    raise RuntimeError(
        f"Could not pull grad_norms from {args.results_json} via "
        f"--gradnorm_source={args.gradnorm_source}.")

print(f"\nLoaded grad norms (source={args.gradnorm_source}):")
for m, g in sorted(grad_norms.items(), key=lambda kv: -kv[1]):
    print(f"  {m:<14s}  {g:.6f}")


# ============================ Data ============================

root_dir = '/files1/haodong/data/processed_dalia_activity'   # UPDATE PATH
dataset_cfg = DaliaHAR()
modalities = dataset_cfg.modalities
M = len(modalities)
fold_cfg = dataset_cfg.folds[args.fold]
eval_subjects = fold_cfg['eval_set']
print(f"\nEval subjects: {eval_subjects}")

eval_ds = HARDataset(root_dir, modalities, eval_subjects, dataset_cfg, args.transform)
eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, drop_last=True)

input_length = ((dataset_cfg.duration * dataset_cfg.base_sample_rate) // 2
                if args.transform == 'sax'
                else dataset_cfg.duration * dataset_cfg.base_sample_rate)


# ============================ Model ============================

v6_cfg = _DaliaV6Cfg(dataset_cfg)
inner = DualVideoBottleneckModelV6Downsample(
    cfg=v6_cfg, output_dim=dataset_cfg.num_classes,
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
    no_selector=True, use_weighted_factor=False, use_triton=False,
    num_classes=dataset_cfg.num_classes,
    sparse_attn_variant=args.sparse_attn_variant,
    strat_block_size=args.strat_block_size,
    downsample_min_len=args.downsample_min_len,
    use_batched_fusion=True,
    per_modal_distill=False, per_modal_downsample_min_len=4,
)
model = _EvalWrapper(inner, modalities, dataset_cfg.variates).to(device).float()

ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f"\nLoaded ckpt; missing={len(missing)} unexpected={len(unexpected)}")


# ============================ Build top-K rankings ============================

# Restrict the grad-norms dict to modalities the model actually has.
# (Defensive: a serialised JSON might include "video" or other extras.)
gn_filtered = {m: float(grad_norms.get(m, 0.0)) for m in modalities}

reverse = (args.prefer == 'high')   # high → descending
ranked = sorted(modalities, key=lambda m: gn_filtered[m], reverse=reverse)

print(f"\nModality ranking ({args.prefer}-grad-norm first):")
for i, m in enumerate(ranked):
    print(f"  {i+1}. {m:<14s} grad_norm={gn_filtered[m]:.6f}")


# ============================ Sweep ============================

ks = sorted(set(k for k in args.ks if 1 <= k <= M))

print("\n" + "=" * 96)
print(f"Eval sweep — fold {args.fold}, M={M}, random seeds={args.n_random_seeds}")
print("=" * 96)
print(f"  {'K':>3}  {'top-K modalities':<48s}  "
      f"{'gn_acc':>7s} {'gn_f1':>7s}  "
      f"{'rnd_acc(mean±std)':>20s}  {'Δacc':>7s}")
print("  " + "-" * 95)

out = {}
for k in ks:
    keep = ranked[:k]
    gn_acc, gn_f1 = evaluate(model, eval_loader, device, keep)

    rnd_accs, rnd_f1s = [], []
    for s in range(args.n_random_seeds):
        ra, rf = evaluate_random_k(model, eval_loader, device, k,
                                   seed=args.seed_num + s)
        rnd_accs.append(ra); rnd_f1s.append(rf)
    ra_m, ra_s = float(np.mean(rnd_accs)), float(np.std(rnd_accs))
    rf_m, rf_s = float(np.mean(rnd_f1s)), float(np.std(rnd_f1s))
    delta = gn_acc - ra_m

    out[k] = {
        'kept_modalities': keep,
        'gradnorm': {'acc': gn_acc, 'f1': gn_f1},
        'random_k': {
            'acc_mean': ra_m, 'acc_std': ra_s,
            'f1_mean': rf_m, 'f1_std': rf_s,
            'acc_seeds': rnd_accs, 'f1_seeds': rnd_f1s,
        },
        'delta_acc': delta,
    }

    keep_str = ','.join(keep)[:46]
    print(f"  {k:>3}  {keep_str:<48s}  "
          f"{gn_acc:>7.4f} {gn_f1:>7.4f}  "
          f"{ra_m:.4f}±{ra_s:.4f}      {delta:>+7.4f}")

print()
print("Δacc = gn_acc − rnd_acc_mean  (positive ⇒ grad-norm ranking beats random)")

if args.output_json:
    dump = {
        'fold': args.fold, 'M': M, 'ks': ks,
        'gradnorm_source': args.gradnorm_source,
        'prefer': args.prefer,
        'grad_norms': gn_filtered,
        'ranked_modalities': ranked,
        'n_random_seeds': args.n_random_seeds,
        'ckpt': args.ckpt,
        'results': {str(k): v for k, v in out.items()},
    }
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(dump, f, indent=4)
    print(f"\nWrote: {args.output_json}")
