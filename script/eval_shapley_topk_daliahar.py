"""Per-sample top-K modality eval driven by Shapley scores.

Companion to ``compute_shapley_daliahar.py``: same forward sweep over
the 2^p subsets, but instead of just summarising Phi we *also* evaluate
the fusion's accuracy when only the top-K modalities (chosen by
per-sample Shapley argmax) are kept.

For each K in ``--ks``:

  * **shapley-topK** — keep the K modalities with largest ``phi_j(x_i)``
                       *per sample i*.
  * **oracle-topK**  — for each sample i, pick the size-K subset S* that
                       minimises CE(y_i, mu(x_i; S)).  This is the
                       cross-entropy-aware Oracle.  Upper bound for any
                       per-sample routing rule.
  * **gradnorm-topK** — fixed global ranking by training-time grad norms
                        (only reported here for direct side-by-side; the
                        existing ``eval_v6_dyna_gradnorm_topk.py``
                        already covers this baseline).
  * **random-K**     — uniform random K-subset per sample (n seeds, mean
                       ± std).

Run example:
    python script/eval_shapley_topk_daliahar.py \\
        --fold=0 --cuda_pick=cuda:0 \\
        --ckpt=./model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna/best_model_fold0.pth \\
        --grad_norms_json=./model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna/results_fold0.json \\
        --ks 1 2 3 4 5 \\
        --output_json=./model_chkpt/DaliaHAR/multimodal_model/2026-05-14_DaliaHAR_v6_fusion_only_dyna/shapley_topk_fold0.json
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
from sklearn.metrics import f1_score, accuracy_score

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
#                              Helpers                                  #
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
        no_selector=True, use_weighted_factor=False, use_triton=False,
        num_classes=dataset_cfg.num_classes,
        sparse_attn_variant=args.sparse_attn_variant,
        strat_block_size=args.strat_block_size,
        downsample_min_len=args.downsample_min_len,
        use_batched_fusion=True, per_modal_distill=False,
        per_modal_downsample_min_len=args.downsample_min_len,
    )


def _split_x(x, modalities, variates):
    high, low = {}, {}
    offset = 0
    for m in modalities:
        n = variates[m]
        chunk = x[:, :, offset:offset + n]
        high[m] = chunk
        low[m] = chunk
        offset += n
    return high, low


@torch.no_grad()
def _forward_subset(model, high, low, subset, modalities):
    keep = [modalities[j] for j in subset]
    high_S = {m: high[m] for m in keep if m in high}
    low_S = {m: low[m] for m in keep if m in low}
    out = model(high_S, low_S, training=False, return_selection_info=False)
    return out[0] if isinstance(out, tuple) else out


def _shapley_weights(p):
    fact = [math.factorial(i) for i in range(p + 1)]
    return {s: fact[s] * fact[p - s - 1] / fact[p] for s in range(p)}


def _enumerate_subsets(p):
    return [tuple(s) for r in range(p + 1) for s in combinations(range(p), r)]


# --------------------------------------------------------------------- #
#                                CLI                                   #
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


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, type=str)
parser.add_argument('--fold', default=0, type=int, choices=[0, 1, 2])
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--transform', default='sax', type=str)
parser.add_argument('--seed_num', default=239, type=int)

parser.add_argument('--grad_norms_json', default=None, type=str,
                    help='Optional: provide gradnorm ranking from training '
                         'results_fold*.json (will report gradnorm-topK too).')
parser.add_argument('--ks', nargs='+', type=int, default=[1, 2, 3, 4, 5])
parser.add_argument('--n_random_seeds', default=5, type=int)
parser.add_argument('--output_json', default=None, type=str)

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

dataset_cfg = DaliaHAR()
modalities = dataset_cfg.modalities
p = len(modalities)
num_classes = dataset_cfg.num_classes

fold_cfg = dataset_cfg.folds[args.fold]
train_subjects, eval_subjects = fold_cfg['train_set'], fold_cfg['eval_set']

input_length = ((dataset_cfg.duration * dataset_cfg.base_sample_rate) // 2
                if args.transform == 'sax'
                else dataset_cfg.duration * dataset_cfg.base_sample_rate)

ds = HARDataset('/files1/haodong/data/processed_dalia_activity',
                modalities, eval_subjects, dataset_cfg, args.transform)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, drop_last=False)

print(f'Device: {device}')
print(f'Fold {args.fold}: eval={eval_subjects}, N={len(ds)}')
print(f'p={p} modalities, num_classes={num_classes}')


# --------------------------------------------------------------------- #
#                            Build & load                              #
# --------------------------------------------------------------------- #


model = _build_v6_model(args, dataset_cfg, input_length)
state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
if any(k.startswith('model.') for k in state.keys()):
    state = {k[len('model.'):]: v for k, v in state.items()
             if k.startswith('model.')}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f'Loaded {args.ckpt}'
      f'  (missing={len(missing)}, unexpected={len(unexpected)})')
model.eval().to(device)


# --------------------------------------------------------------------- #
#                Forward fusion on all 2^p subsets + Shapley           #
# --------------------------------------------------------------------- #


subsets = _enumerate_subsets(p)
subset_to_idx = {S: i for i, S in enumerate(subsets)}
n_subsets = len(subsets)
N_total = len(ds)
weights = _shapley_weights(p)

# logits[i, k, c]  — uses CPU memory: N * 32 * C * 4 bytes
logits_all = torch.zeros(N_total, n_subsets, num_classes)
phi_all = torch.zeros(N_total, p)
y_all = torch.zeros(N_total, dtype=torch.long)
baseline_loss = math.log(num_classes)

t0 = time.time()
cursor = 0
for bi, (x, y) in enumerate(loader):
    x = x.to(device).float()
    y_cuda = y.to(device)
    B = x.shape[0]
    high, low = _split_x(x, modalities, dataset_cfg.variates)

    v = torch.zeros(B, n_subsets, device=device)
    for k, S in enumerate(subsets):
        if len(S) == 0:
            continue
        logits = _forward_subset(model, high, low, S, modalities)
        logits_all[cursor:cursor + B, k] = logits.cpu()
        loss_S = F.cross_entropy(logits, y_cuda, reduction='none')
        v[:, k] = baseline_loss - loss_S

    # Shapley
    Phi = torch.zeros(B, p, device=device)
    for j in range(p):
        for k, S in enumerate(subsets):
            if j in S:
                continue
            k_with = subset_to_idx[tuple(sorted(S + (j,)))]
            Phi[:, j] += weights[len(S)] * (v[:, k_with] - v[:, k])

    phi_all[cursor:cursor + B] = Phi.cpu()
    y_all[cursor:cursor + B] = y
    cursor += B

    if bi % 5 == 0 or bi == len(loader) - 1:
        print(f'  batch {bi+1:>3}/{len(loader)}  elapsed={time.time()-t0:6.1f}s')

print(f'\nForward + Shapley done.  Phi shape: {tuple(phi_all.shape)}, '
      f'logits cache: {tuple(logits_all.shape)}')


# --------------------------------------------------------------------- #
#                      Build top-K subset look-up                      #
# --------------------------------------------------------------------- #


def topk_indices_per_sample(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Returns sorted top-k modality indices per sample.  scores: [N, p]"""
    _, idx = torch.topk(scores, k, dim=1)
    return idx.sort(dim=1).values  # canonicalise subset by sorting


def lookup_logits_for_per_sample_subset(idx_per_sample: torch.Tensor) -> torch.Tensor:
    """idx_per_sample: [N, K] long.  Returns [N, C]."""
    N = idx_per_sample.shape[0]
    out = torch.zeros(N, num_classes)
    for i in range(N):
        S = tuple(idx_per_sample[i].tolist())
        k_idx = subset_to_idx[S]
        out[i] = logits_all[i, k_idx]
    return out


def evaluate_logits(logits: torch.Tensor, y: torch.Tensor):
    pred = logits.argmax(dim=1)
    acc = float(accuracy_score(y.numpy(), pred.numpy()))
    f1 = float(f1_score(y.numpy(), pred.numpy(), average='macro'))
    return acc, f1


# Pre-build size-K subset lists
size_to_subset_indices = {k: [subset_to_idx[S] for S in subsets if len(S) == k]
                          for k in range(p + 1)}

# Load gradnorm ranking if available
gn_rank = None
if args.grad_norms_json is not None and os.path.exists(args.grad_norms_json):
    gn = json.load(open(args.grad_norms_json))['final_grad_norms']
    gn_vec = np.array([gn[m] for m in modalities], dtype=np.float32)
    gn_rank = list(np.argsort(-gn_vec))  # decreasing
    print(f'\nGradnorm ranking (high → low): '
          f'{[modalities[j] for j in gn_rank]}')

# Shapley population ranking
mean_phi = phi_all.mean(dim=0)
shap_pop_rank = list(torch.argsort(-mean_phi).tolist())
print(f'Shapley population ranking      : '
      f'{[modalities[j] for j in shap_pop_rank]}')


# --------------------------------------------------------------------- #
#                            Eval per K                                #
# --------------------------------------------------------------------- #


results = {'K': {}, 'modalities': modalities,
           'shapley_population_ranking':
               [modalities[j] for j in shap_pop_rank]}
if gn_rank is not None:
    results['gradnorm_ranking'] = [modalities[j] for j in gn_rank]

print('\n' + '=' * 80)
print(f'{"K":>2} {"shapley":>14} {"oracle":>14} {"gradnorm":>14} {"random":>16}')
print('-' * 80)

rng = np.random.default_rng(args.seed_num)

for K in args.ks:
    K_str = str(K)
    out_K = {}

    # --- 1. shapley-topK (per-sample) ---
    sh_idx = topk_indices_per_sample(phi_all, K)            # [N, K]
    sh_logits = lookup_logits_for_per_sample_subset(sh_idx)
    sh_acc, sh_f1 = evaluate_logits(sh_logits, y_all)
    out_K['shapley_topk'] = {'acc': sh_acc, 'f1': sh_f1}

    # --- 2. oracle-topK (per-sample best size-K subset by CE on TRUE label) ---
    if K < p:
        size_k_idx = size_to_subset_indices[K]
        # logits_all[:, size_k_idx, :] has shape [N, |C(p,K)|, C]
        # CE for each subset: F.cross_entropy with reduction='none' on a flattened batch
        logits_subset = logits_all[:, size_k_idx, :]          # [N, M, C]
        Nn, Mm, Cn = logits_subset.shape
        flat = logits_subset.reshape(Nn * Mm, Cn)
        y_rep = y_all.repeat_interleave(Mm)
        losses = F.cross_entropy(flat, y_rep, reduction='none').reshape(Nn, Mm)
        best_m = losses.argmin(dim=1)                         # [N]
        # gather corresponding logits
        oracle_logits = torch.gather(
            logits_subset, 1, best_m.view(-1, 1, 1).expand(-1, 1, Cn)).squeeze(1)
    else:
        oracle_logits = logits_all[:, subset_to_idx[tuple(range(p))], :]
    oracle_acc, oracle_f1 = evaluate_logits(oracle_logits, y_all)
    out_K['oracle_topk'] = {'acc': oracle_acc, 'f1': oracle_f1}

    # --- 3. gradnorm-topK (one global subset across all samples) ---
    if gn_rank is not None:
        gn_S = tuple(sorted(gn_rank[:K]))
        gn_k_idx = subset_to_idx[gn_S]
        gn_logits = logits_all[:, gn_k_idx, :]
        gn_acc, gn_f1 = evaluate_logits(gn_logits, y_all)
        out_K['gradnorm_topk'] = {
            'acc': gn_acc, 'f1': gn_f1,
            'kept_modalities': [modalities[j] for j in gn_rank[:K]],
        }
    else:
        gn_acc = gn_f1 = float('nan')

    # --- 4. random-K (n seeds, mean ± std) ---
    rand_accs, rand_f1s = [], []
    for s in range(args.n_random_seeds):
        rseed = args.seed_num + s
        gen = np.random.default_rng(rseed)
        # Each sample's random K-subset
        rand_idx = np.stack([gen.choice(p, K, replace=False) for _ in range(N_total)])
        rand_idx = torch.from_numpy(rand_idx).long().sort(dim=1).values
        r_logits = lookup_logits_for_per_sample_subset(rand_idx)
        a, f = evaluate_logits(r_logits, y_all)
        rand_accs.append(a)
        rand_f1s.append(f)
    rand_acc_mean = float(np.mean(rand_accs))
    rand_acc_std = float(np.std(rand_accs, ddof=1)) if len(rand_accs) > 1 else 0.0
    rand_f1_mean = float(np.mean(rand_f1s))
    rand_f1_std = float(np.std(rand_f1s, ddof=1)) if len(rand_f1s) > 1 else 0.0
    out_K['random'] = {
        'acc_mean': rand_acc_mean, 'acc_std': rand_acc_std,
        'f1_mean': rand_f1_mean, 'f1_std': rand_f1_std,
        'n_seeds': args.n_random_seeds,
    }

    results['K'][K_str] = out_K

    print(f'{K:>2} {sh_acc:>14.4f} {oracle_acc:>14.4f} '
          f'{gn_acc:>14.4f} '
          f'{rand_acc_mean:>10.4f}±{rand_acc_std:.3f}')

print('=' * 80)
print('(values are test-set accuracy)\n')


# --------------------------------------------------------------------- #
#                                Save                                  #
# --------------------------------------------------------------------- #


if args.output_json:
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    results['ckpt'] = args.ckpt
    results['fold'] = args.fold
    results['N'] = int(N_total)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote: {args.output_json}')

print(f'\nDone in {time.time()-t0:.1f}s.')
