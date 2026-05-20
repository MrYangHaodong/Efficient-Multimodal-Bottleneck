"""Compute per-sample Shapley modality importance on the DaliaHAR
fusion-only-dyna checkpoint.

Step 1 of the Shapley-distilled selector plan.  No training: the
fusion weights are kept frozen at the 2026-05-14 dyna checkpoint; we
just enumerate (or permutation-sample) subsets, forward the fusion for
each subset on the validation set, and combine the per-subset losses
via the closed-form Shapley formula

    phi_j(x_i) = sum_{S \\subseteq [p]\\{j}}  w(S) * (v(S \\cup {j}) - v(S)),
    w(S) = |S|! * (p-|S|-1)! / p!,
    v(S) = baseline_loss - CE(y_i, model(x_i; S)).

Output:
  - JSON with per-modality mean / std Shapley, per-sample top-1 stats,
    correlation with grad-norm ranking (if grad_norms_json provided).
  - Optional .npz with the full per-sample Shapley matrix Phi  [N, p].

Run example:
    python script/compute_shapley_daliahar.py \\
        --fold=0 --cuda_pick=cuda:0 \\
        --ckpt=./model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna/best_model_fold0.pth \\
        --grad_norms_json=./model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna/results_fold0.json \\
        --output_json=./model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna/shapley_fold0.json \\
        --save_phi
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from itertools import combinations
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from utils.helper_function import set_seed
from utils.dataset_cfg import DaliaHAR
from data.dataset_builder import HARDataset
from multimodal_model.v6_downsample_opt_batched import (
    DualVideoBottleneckModelV6Downsample,
)


# --------------------------------------------------------------------- #
#                         V6 config / wrapper                          #
# --------------------------------------------------------------------- #


class _DaliaV6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


def _build_v6_model(args, dataset_cfg, input_length):
    v6_cfg = _DaliaV6Cfg(dataset_cfg)
    return DualVideoBottleneckModelV6Downsample(
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
        no_selector=True,
        use_weighted_factor=False,
        use_triton=False,
        num_classes=dataset_cfg.num_classes,
        sparse_attn_variant=args.sparse_attn_variant,
        strat_block_size=args.strat_block_size,
        downsample_min_len=args.downsample_min_len,
        use_batched_fusion=True,
        per_modal_distill=False,
        per_modal_downsample_min_len=args.downsample_min_len,
    )


def _split_x(x: torch.Tensor, modalities: List[str], variates: dict):
    """Channel-split [B, T, V_total] into per-modality [B, T, V_m] dicts."""
    high, low = {}, {}
    offset = 0
    for m in modalities:
        n = variates[m]
        chunk = x[:, :, offset:offset + n]
        high[m] = chunk
        low[m] = chunk
        offset += n
    return high, low


def _forward_subset(
    model: torch.nn.Module,
    high: dict,
    low: dict,
    subset: Tuple[int, ...],
    modalities: List[str],
) -> torch.Tensor:
    """Forward V6 with only the modalities indexed in ``subset`` present
    in the input dict.  Returns logits ``[B, C]``."""
    keep = [modalities[j] for j in subset]
    high_S = {m: high[m] for m in keep if m in high}
    low_S = {m: low[m] for m in keep if m in low}
    out = model(high_S, low_S, training=False, return_selection_info=False)
    return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------- #
#                       Shapley computation                            #
# --------------------------------------------------------------------- #


def _shapley_weights(p: int) -> dict:
    """Return {|S|: |S|!*(p-|S|-1)!/p!}."""
    fact = [math.factorial(i) for i in range(p + 1)]
    return {s: fact[s] * fact[p - s - 1] / fact[p] for s in range(p)}


def _enumerate_subsets(p: int) -> List[Tuple[int, ...]]:
    """Power-set of {0,...,p-1}, NON-empty subsets first (we keep empty
    too for completeness, but it is never forwarded — its utility is
    fixed to 0 via the baseline trick)."""
    return [tuple(s) for r in range(p + 1) for s in combinations(range(p), r)]


@torch.no_grad()
def exact_shapley_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    modalities: List[str],
    variates: dict,
    num_classes: int,
) -> torch.Tensor:
    """Exact Shapley via full power-set enumeration.

    Returns ``Phi`` of shape ``[B, p]``.

    Baseline: empty-subset utility is fixed to 0 by setting
    ``v(S) = baseline_loss - loss_S`` with ``baseline_loss =
    log(num_classes)`` (uniform-prediction CE).  Any constant choice
    works — Shapley differences ``v(S∪{j}) - v(S)`` are invariant to
    an additive shift in the baseline.
    """
    p = len(modalities)
    B = x.shape[0]
    device = x.device
    high, low = _split_x(x, modalities, variates)

    subsets = _enumerate_subsets(p)
    subset_to_idx = {S: i for i, S in enumerate(subsets)}

    # v[i, k] = utility of subset_k for sample i
    v = torch.zeros(B, len(subsets), device=device)
    baseline_loss = math.log(num_classes)  # uniform prediction CE

    for k, S in enumerate(subsets):
        if len(S) == 0:
            # v(∅) = baseline - baseline = 0   (uniform-prediction CE)
            continue
        logits = _forward_subset(model, high, low, S, modalities)
        loss_S = F.cross_entropy(logits, y, reduction='none')  # [B]
        v[:, k] = baseline_loss - loss_S

    # Closed-form Shapley
    weights = _shapley_weights(p)
    Phi = torch.zeros(B, p, device=device)
    for j in range(p):
        for k, S in enumerate(subsets):
            if j in S:
                continue
            S_with = tuple(sorted(S + (j,)))
            k_with = subset_to_idx[S_with]
            w = weights[len(S)]
            Phi[:, j] += w * (v[:, k_with] - v[:, k])
    return Phi


@torch.no_grad()
def mc_shapley_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    modalities: List[str],
    variates: dict,
    num_classes: int,
    n_perm: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Permutation-sampling Shapley estimator.

    For each of ``n_perm`` random permutations pi of [p], for each
    position k=0..p-1 we evaluate ``v(S_pi[:k+1]) - v(S_pi[:k])`` and
    credit it to modality ``pi[k]``.  Unbiased; cost is
    ``n_perm * (p + 1) / unique-subsets-caching`` forwards.  We cache
    by subset (frozen-set key) to skip duplicates within and across
    permutations.

    Returns ``Phi`` of shape ``[B, p]``.
    """
    p = len(modalities)
    B = x.shape[0]
    device = x.device
    high, low = _split_x(x, modalities, variates)
    baseline_loss = math.log(num_classes)

    v_cache: dict = {}

    def utility(S: Tuple[int, ...]) -> torch.Tensor:
        key = tuple(sorted(S))
        if key in v_cache:
            return v_cache[key]
        if len(key) == 0:
            val = torch.zeros(B, device=device)
        else:
            logits = _forward_subset(model, high, low, key, modalities)
            loss_S = F.cross_entropy(logits, y, reduction='none')
            val = baseline_loss - loss_S
        v_cache[key] = val
        return val

    Phi = torch.zeros(B, p, device=device)
    perms = torch.zeros(n_perm, p, dtype=torch.long)
    for t in range(n_perm):
        perms[t] = torch.randperm(p, generator=rng)
    for t in range(n_perm):
        pi = perms[t].tolist()
        S: List[int] = []
        v_S = utility(())
        for k in range(p):
            j = pi[k]
            S_new = tuple(sorted(S + [j]))
            v_Sn = utility(S_new)
            Phi[:, j] += (v_Sn - v_S)
            S = list(S_new)
            v_S = v_Sn
    Phi /= n_perm
    return Phi


# --------------------------------------------------------------------- #
#                                CLI                                    #
# --------------------------------------------------------------------- #


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
    description='Compute per-sample Shapley modality importance on a '
                'frozen DaliaHAR fusion checkpoint.')

parser.add_argument('--ckpt', required=True, type=str,
                    help='Path to fusion_only_dyna best_model_fold{F}.pth.')
parser.add_argument('--fold', default=0, type=int, choices=[0, 1, 2])
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--split', default='val', choices=['val', 'eval', 'train'],
                    help='Which split to compute Shapley over.')
parser.add_argument('--seed_num', default=239, type=int)

parser.add_argument('--estimator', default='exact', choices=['exact', 'mc'])
parser.add_argument('--n_perm', default=10, type=int,
                    help='Permutations per batch for MC estimator.')

parser.add_argument('--grad_norms_json', default=None, type=str,
                    help='Optional path to results_fold{F}.json with '
                         'final_grad_norms — used for Spearman comparison.')
parser.add_argument('--output_json', default=None, type=str)
parser.add_argument('--save_phi', action='store_true', default=False,
                    help='Also save the full [N, p] Phi matrix as npz.')

# Model config (must match dyna training).
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_layers_per_modal', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--internal_dim', default=64, type=int)
parser.add_argument('--n_bottlenecks', default=8, type=int)
parser.add_argument('--fusion_layer', default=0, type=int)
parser.add_argument('--use_sparse_attn', default=True, type=_str2bool)
parser.add_argument('--sparse_attn_variant', default='opt', type=str)
parser.add_argument('--strat_block_size', default=8, type=int)
parser.add_argument('--downsample_min_len', default=4, type=int)
parser.add_argument('--base_factor', default=3, type=int)
parser.add_argument('--num_experts', default=4, type=int)

args = parser.parse_args()


# --------------------------------------------------------------------- #
#                                Setup                                  #
# --------------------------------------------------------------------- #


device = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed_num)
print(f'Device: {device}')
print(f'Estimator: {args.estimator}'
      + (f' (n_perm={args.n_perm})' if args.estimator == 'mc' else ''))

root_dir = '/files1/haodong/data/processed_dalia_activity'
dataset_cfg = DaliaHAR()
modalities = dataset_cfg.modalities
p = len(modalities)
num_classes = dataset_cfg.num_classes

fold_cfg = dataset_cfg.folds[args.fold]
train_subjects, eval_subjects = fold_cfg['train_set'], fold_cfg['eval_set']
val_subjects = dataset_cfg.val_set

input_length = ((dataset_cfg.duration * dataset_cfg.base_sample_rate) // 2
                if args.transform == 'sax'
                else dataset_cfg.duration * dataset_cfg.base_sample_rate)

if args.split == 'train':
    split_subjects = train_subjects
elif args.split == 'val':
    split_subjects = val_subjects
else:
    split_subjects = eval_subjects
ds = HARDataset(root_dir, modalities, split_subjects, dataset_cfg, args.transform)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, drop_last=False)
print(f'Fold {args.fold}: split={args.split}, '
      f'subjects={split_subjects}, samples={len(ds)}')
print(f'Modalities ({p}): {modalities}')
print(f'num_classes={num_classes}, input_length={input_length}')


# --------------------------------------------------------------------- #
#                            Build & load                              #
# --------------------------------------------------------------------- #


model = _build_v6_model(args, dataset_cfg, input_length)
state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
# Checkpoint was saved from V6FusionOnlyDynaWrapper, whose inner V6 lives
# under `self.model.*`.  Strip the leading "model." prefix so the bare V6
# can pick the weights up.
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k, v in state.items()
             if k.startswith('model.')}
missing, unexpected = model.load_state_dict(state, strict=False)
n_missing = len([k for k in missing if 'modality_selector' not in k])
print(f'Loaded {args.ckpt}')
print(f'  state keys missing (non-selector): {n_missing} '
      f'(total missing: {len(missing)}; unexpected: {len(unexpected)})')
if unexpected:
    print(f'  first 3 unexpected: {unexpected[:3]}')
if missing:
    print(f'  first 3 missing   : {missing[:3]}')
model.eval().to(device)


# --------------------------------------------------------------------- #
#                              Shapley                                 #
# --------------------------------------------------------------------- #


rng = torch.Generator()
rng.manual_seed(args.seed_num)

all_phi = []
all_labels = []
t0 = time.time()
for i, (x, y) in enumerate(loader):
    x = x.to(device).float()
    y = y.to(device)
    if args.estimator == 'exact':
        Phi = exact_shapley_batch(model, x, y, modalities, dataset_cfg.variates,
                                  num_classes)
    else:
        Phi = mc_shapley_batch(model, x, y, modalities, dataset_cfg.variates,
                               num_classes, n_perm=args.n_perm, rng=rng)
    all_phi.append(Phi.cpu())
    all_labels.append(y.cpu())
    if i % 5 == 0 or i == len(loader) - 1:
        elapsed = time.time() - t0
        print(f'  batch {i+1:>3}/{len(loader)}  elapsed={elapsed:6.1f}s')
Phi = torch.cat(all_phi, dim=0).numpy()          # [N, p]
labels = torch.cat(all_labels, dim=0).numpy()    # [N]
N = Phi.shape[0]
print(f'\nShapley matrix:  {Phi.shape}  (N={N}, p={p})')


# --------------------------------------------------------------------- #
#                              Statistics                              #
# --------------------------------------------------------------------- #


mean_phi = Phi.mean(axis=0)
std_phi = Phi.std(axis=0, ddof=1)
per_modality = [
    {'modality': m, 'mean_shapley': float(mean_phi[j]),
     'std_shapley': float(std_phi[j])}
    for j, m in enumerate(modalities)
]
ranked_by_mean = sorted(per_modality, key=lambda d: -d['mean_shapley'])
print('\n--- Per-modality mean Shapley ---')
for r, e in enumerate(ranked_by_mean):
    print(f'  rank {r+1}: {e["modality"]:<10} '
          f'mean={e["mean_shapley"]:+.4f}  std={e["std_shapley"]:.4f}')

# Per-sample top-1 modality histogram
top1 = np.argmax(Phi, axis=1)
top1_counts = np.bincount(top1, minlength=p)
top1_frac = top1_counts / N
print('\n--- Per-sample top-1 modality histogram ---')
for j, m in enumerate(modalities):
    print(f'  {m:<10}  count={top1_counts[j]:>4}  '
          f'frac={top1_frac[j]:.3f}')

# Per-sample Shapley variance: how often does ranking differ from
# the population mean ranking?
mean_rank = np.argsort(-mean_phi)               # decreasing mean Shapley
top1_match_with_mean = (top1 == mean_rank[0]).mean()
print(f'\nPer-sample top-1 matches population top-1: '
      f'{top1_match_with_mean*100:.1f}%')
print('(low value -> Shapley varies a lot per-sample → routing useful)')

# Compare with grad-norm ranking if provided
spearman = None
gn_ranking = None
if args.grad_norms_json is not None and os.path.exists(args.grad_norms_json):
    gn = json.load(open(args.grad_norms_json))['final_grad_norms']
    gn_vec = np.array([gn[m] for m in modalities], dtype=np.float32)
    gn_ranking = sorted(zip(modalities, gn_vec), key=lambda kv: -kv[1])

    # Spearman between mean_phi and gn_vec
    from scipy.stats import spearmanr
    rho, pval = spearmanr(mean_phi, gn_vec)
    spearman = {'rho': float(rho), 'p_value': float(pval)}
    print(f'\n--- Grad-norm comparison ---')
    print(f'  gradnorm ranking : {[m for m, _ in gn_ranking]}')
    print(f'  shapley  ranking : {[e["modality"] for e in ranked_by_mean]}')
    print(f'  Spearman rho = {rho:+.4f}  (p={pval:.4g})')


# --------------------------------------------------------------------- #
#                                Save                                  #
# --------------------------------------------------------------------- #


out = {
    'ckpt': args.ckpt,
    'fold': args.fold,
    'split': args.split,
    'estimator': args.estimator,
    'n_perm': args.n_perm if args.estimator == 'mc' else None,
    'n_samples': int(N),
    'modalities': modalities,
    'per_modality': per_modality,
    'top1_hist': {m: float(top1_frac[j]) for j, m in enumerate(modalities)},
    'top1_match_with_mean': float(top1_match_with_mean),
}
if spearman is not None:
    out['spearman_vs_gradnorm'] = spearman
    out['gradnorm_ranking'] = [m for m, _ in gn_ranking]
    out['shapley_ranking'] = [e['modality'] for e in ranked_by_mean]

if args.output_json:
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote summary to {args.output_json}')

if args.save_phi:
    npz_path = (args.output_json.replace('.json', '_phi.npz')
                if args.output_json else
                f'shapley_phi_fold{args.fold}.npz')
    np.savez(npz_path, phi=Phi, labels=labels, modalities=np.array(modalities))
    print(f'Wrote per-sample Phi to {npz_path}')

print(f'\nDone in {time.time() - t0:.1f}s.')
