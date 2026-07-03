#!/usr/bin/env python
"""Exact per-sample modality SHAPLEY scores for a trained seqA_prefixup model — TWO order
variants per sample per modality-combination.

For each TEST sample and each modality m, phi_m = m's average marginal contribution to the
model's softmax confidence in the TRUE class, over all 2^M modality coalitions. Because seqA's
sequential fusion is ORDER-DEPENDENT, we report two value-function variants:

  * valLOO   v(S) fuses coalition S in the model's val-LOO (method-A) order restricted to S
             — the model's deployed fusion order.  -> shapley_valLOO
  * randavg  v(S) = mean over K random fusion orders of S (all k! orders if k! <= K).
             -> shapley_randavg

  checkpoint  the one that achieved the headline best_test_acc for this cell, regardless of
              which selection method won (best_model_{loo|rand|cold}); auto-picked from the
              results json. (mosei -> cold; the other 5 datasets -> loo.)
  value fn    v(S) = softmax(seqA(S))[true_class] per sample;  v(empty) = 1/num_classes.
  masking     coalition S = feed ONLY {m in S} keys in the high/low dicts (true drop).
  order idx   seq_modality_order = positions in the canonical active list (matches
              utils/dual_order_val._eval); val-LOO order recomputed on val for THIS checkpoint.
  attention   DENSE (full). ProbSparse (eav) resamples keys with an unseeded RNG and no eval
              guard -> nondeterministic; forced dense = the deterministic value it approximates.
  Shapley     exact, all 2^M coalitions (M<=6): phi_m = sum_{S<=M\\{m}} |S|!(M-|S|-1)!/M! [v(S+m)-v(S)]
  Efficiency  sum_m phi_m == v(full) - v(empty)  (checked per sample per variant -> eff_err ~ 1e-8)

Reuses training.seqA.build_data (identical data pipeline + model geometry; sparse HAR
config.json geometry is patched from run_seqA_prefixsup_4ds.sh, see HAR_RECIPE). Test loader
= the trainer's own eval loader (HAR drop_last=True drops the final partial batch).

  python analysis/shapley_seqA_prefixup.py --dataset mosei --fold 0 \
      --exp_dir <...>/2026-06-27_CMU_MOSEI_seqA_ps_acc2_s7_seqA --tag seed7 --K 5
"""
from __future__ import annotations
import argparse, json, math, os, sys, warnings
warnings.filterwarnings('ignore')
from itertools import combinations, permutations
import numpy as np
import torch
import torch.nn.functional as F

REPO = '/files1/haodong/Efficient-Multimodal-Bottleneck'
sys.path.insert(0, REPO)
from training.seqA import build_parser, build_data   # noqa: E402
from utils.dual_order_val import loo_order            # noqa: E402

CFG_KEYS = ['head_mode', 'fusion_mode', 'seq_depth_flow', 'seq_modality_order',
            'seq_random_order', 'd_model', 'nhead', 'num_layers', 'num_layers_per_modal',
            'dropout', 'num_experts', 'base_factor', 'internal_dim', 'n_bottlenecks',
            'fusion_layer', 'use_sparse_attn', 'sparse_attn_variant', 'strat_block_size',
            'downsample_min_len', 'n_fusion_distill', 'bottleneck_agg_mode',
            'per_modal_distill', 'per_modal_downsample_min_len',
            'max_seq_len', 'time_compression_ratio', 'word_length', 'transform']

# HAR prefixsup config.json omits word_length / n_fusion_distill / input_length_override, so
# config-hydration alone builds the wrong architecture. Authoritative launch args from
# run_seqA_prefixsup_4ds.sh (verified vs each checkpoint's pos-cache + ds_conv blocks).
HAR_RECIPE = {
    'daliahar': {'word_length': 1, 'n_fusion_distill': -1, 'transform': 'sax'},
    'pamap2':   {'word_length': 1, 'transform': 'sax_noisy', 'input_length_override': 512,
                 'n_fusion_distill': -1},
    'dsads':    {'word_length': 1, 'n_fusion_distill': -1, 'transform': 'sax'},
}


def build_args(dataset, fold, exp_dir, cuda):
    p = build_parser()
    args = p.parse_args(['--dataset', dataset, '--cuda_pick', cuda,
                         '--results_dir', '/tmp/_shap_unused', '--num_epochs', '1',
                         '--fold', str(fold)])
    cfg_path = os.path.join(exp_dir, 'config.json')
    cj = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    for k in CFG_KEYS:
        if k in cj and cj[k] is not None:
            setattr(args, k, cj[k])
    if dataset == 'mosei':
        args.mosei_task = int(cj.get('num_classes', 2))
    for k, v in HAR_RECIPE.get(dataset, {}).items():
        setattr(args, k, v)
    return args, cj


def pick_best_test_ckpt(exp_dir, fold, dataset):
    """Return (ckpt_path, selection, best_test_acc) for the checkpoint that achieved the
    headline best_test_acc (loo / rand / cold), matched by test_acc; cold if neither A nor B."""
    rjp = os.path.join(exp_dir, f'results_fold{fold}.json')
    if not os.path.exists(rjp):
        rjp = os.path.join(exp_dir, 'results.json')
    rj = json.load(open(rjp))
    bt = rj['best_test_acc']
    do = rj.get('dual_order', {})
    A, B = do.get('A_valLOO', {}), do.get('B_randavg', {})
    suf = f'fold{fold}' if dataset != 'mosei' else 'val'
    if A.get('test_acc') is not None and abs(bt - A['test_acc']) < 1e-6:
        sel, fn = 'loo', f'best_model_loo_{suf}.pth'
    elif B.get('test_acc') is not None and abs(bt - B['test_acc']) < 1e-6:
        sel, fn = 'rand', f'best_model_rand_{suf}.pth'
    else:
        sel, fn = 'cold', f'best_model_cold_{suf}.pth'
    return os.path.join(exp_dir, fn), sel, bt


def _order_sets(mods, subs, vlo_order, K, seed):
    """Per coalition S: the val-LOO fusion order (positions in canonical active list) and a
    list of up-to-K random orders (all k! if k! <= K, else K distinct sampled). Deterministic."""
    rng = np.random.default_rng(seed)
    vlo_idx, rand_idx = {}, {}
    for S in subs:
        active = [mods[i] for i in sorted(S)]
        k = len(active)
        seqv = [m for m in vlo_order if m in active] if vlo_order else active
        vlo_idx[S] = [active.index(m) for m in seqv]
        allp = list(permutations(range(k)))
        if len(allp) <= K:
            chosen = allp
        else:
            pick = rng.choice(len(allp), size=K, replace=False)
            chosen = [allp[i] for i in pick]
        rand_idx[S] = [list(p) for p in chosen]
    return vlo_idx, rand_idx


@torch.no_grad()
def coalition_probs_2variants(inner, batches, mods, device, vlo_order, K, seed):
    """For every non-empty coalition return per-sample softmax [N,C] under (a) the val-LOO
    order and (b) the mean over K random orders. Plus labels [N]. Deterministic (dense attn)."""
    M = len(mods)
    subs = [frozenset(S) for k in range(1, M + 1) for S in combinations(range(M), k)]
    vlo_idx, rand_idx = _order_sets(mods, subs, vlo_order, K, seed)
    vlo_acc = {S: [] for S in subs}
    rnd_acc = {S: [] for S in subs}
    ys = []
    fuse = inner.bottleneck_fusion

    def fwd(hS, lS):
        o = inner(hS, lS, training=False, return_selection_info=False)
        return (o[0] if isinstance(o, tuple) else o)

    for high, low, y in batches:
        ys.append(y.cpu().numpy())
        h_all = {m: high[m].to(device).float() for m in mods}
        l_all = {m: low[m].to(device).float() for m in mods}
        for S in subs:
            active = [mods[i] for i in sorted(S)]
            hS = {m: h_all[m] for m in active}
            lS = {m: l_all[m] for m in active}
            fuse.seq_modality_order = vlo_idx[S]
            vlo_acc[S].append(F.softmax(fwd(hS, lS), -1).cpu().numpy())
            acc = None
            for ordr in rand_idx[S]:
                fuse.seq_modality_order = ordr
                p = F.softmax(fwd(hS, lS), -1).cpu().numpy()
                acc = p if acc is None else acc + p
            rnd_acc[S].append(acc / len(rand_idx[S]))
    fuse.seq_modality_order = None
    y = np.concatenate(ys)
    vlo = {S: np.concatenate(v, 0) for S, v in vlo_acc.items()}
    rnd = {S: np.concatenate(v, 0) for S, v in rnd_acc.items()}
    return vlo, rnd, y


def shapley(prob, y, M, num_classes):
    """Exact per-sample Shapley over all 2^M coalitions. Returns phi[N,M], v_empty."""
    N = len(y)
    idx = np.arange(N)
    v_empty = 1.0 / num_classes
    vals = {frozenset(): np.full(N, v_empty, dtype=np.float64)}
    for S, p in prob.items():
        vals[S] = p[idx, y].astype(np.float64)
    fact = math.factorial
    phi = np.zeros((N, M), dtype=np.float64)
    for m in range(M):
        others = [j for j in range(M) if j != m]
        for k in range(0, M):
            w = fact(k) * fact(M - k - 1) / fact(M)
            for S in combinations(others, k):
                phi[:, m] += w * (vals[frozenset(S + (m,))] - vals[frozenset(S)])
    return phi, v_empty


def _variant_stats(phi, prob, y, mods, v_empty):
    N, M = phi.shape
    full = frozenset(range(M))
    v_full = prob[full][np.arange(N), y]
    y_pred = prob[full].argmax(1)
    eff = phi.sum(1) - (v_full - v_empty)
    return {
        'full_coalition_test_acc': float((y_pred == y).mean()),
        'mean_v_full': float(v_full.mean()),
        'mean_abs_shapley_per_modality': {mods[i]: float(np.abs(phi[:, i]).mean()) for i in range(M)},
        'mean_shapley_per_modality': {mods[i]: float(phi[:, i].mean()) for i in range(M)},
        'efficiency_max_abs_err': float(np.abs(eff).max()),
        'efficiency_mean_abs_err': float(np.abs(eff).mean()),
    }, v_full, y_pred, eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei'])
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--exp_dir', required=True, help='experiment dir (config.json + results + checkpoints)')
    ap.add_argument('--tag', required=True, help='output label, e.g. fold0 / seed7')
    ap.add_argument('--cuda_pick', default='cuda:0')
    ap.add_argument('--K', type=int, default=5, help='# random fusion orders averaged (randavg variant)')
    ap.add_argument('--seed', type=int, default=1234, help='seed for the random-order sampling')
    ap.add_argument('--out_dir', default=os.path.join(REPO, 'results', 'shapley_seqA_prefixup'))
    a = ap.parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device(a.cuda_pick if torch.cuda.is_available() else 'cpu')

    ckpt, selection, best_test = pick_best_test_ckpt(a.exp_dir, a.fold, a.dataset)
    args, cj = build_args(a.dataset, a.fold, a.exp_dir, a.cuda_pick)
    _, val_batches, test_batches, mods, num_classes, model = build_data(args, device)
    M = len(mods)
    state = torch.load(ckpt, map_location=device)
    n_sing = sum(1 for k in state if 'singleton_heads' in k)
    model.load_state_dict(state, strict=(n_sing == 0))
    model.eval()
    inner = model.model

    # Force DENSE attention (deterministic value fn; ProbSparse would be a noisy MC draw).
    n_forced = 0
    for blk in getattr(inner.bottleneck_fusion, 'blocks', []):
        if getattr(blk, 'use_sparse_attn', False):
            blk.use_sparse_attn = False
            n_forced += 1

    # Recompute the val-LOO (method-A) order on the val set for THIS checkpoint (self-consistent
    # whether the picked checkpoint is loo / rand / cold).
    vlo_order, _ = loo_order(inner, val_batches, mods, device)
    inner.bottleneck_fusion.seq_modality_order = None
    print(f'[shapley] {a.dataset}/{a.tag} M={M} mods={mods} C={num_classes} sel={selection} '
          f'best_test={best_test:.4f} valLOO={vlo_order} K={a.K} dense_forced={n_forced} '
          f'ckpt={os.path.basename(ckpt)}')

    vlo, rnd, y = coalition_probs_2variants(inner, test_batches, mods, device, vlo_order, a.K, a.seed)
    N = len(y)
    phi_v, v_empty = shapley(vlo, y, M, num_classes)
    phi_r, _ = shapley(rnd, y, M, num_classes)
    stats_v, vfull_v, ypred_v, eff_v = _variant_stats(phi_v, vlo, y, mods, v_empty)
    stats_r, vfull_r, ypred_r, eff_r = _variant_stats(phi_r, rnd, y, mods, v_empty)

    out_dir = os.path.join(a.out_dir, a.dataset)
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f'shapley_{a.tag}.npz')
    np.savez(npz,
             shapley_valLOO=phi_v.astype(np.float32),          # [N, M]
             shapley_randavg=phi_r.astype(np.float32),         # [N, M]
             modalities=np.array(mods), valLOO_order=np.array(vlo_order),
             y_true=y.astype(np.int64),
             y_pred_valLOO=ypred_v.astype(np.int64), y_pred_randavg=ypred_r.astype(np.int64),
             v_full_valLOO=vfull_v.astype(np.float32), v_full_randavg=vfull_r.astype(np.float32),
             v_empty=np.float32(v_empty),
             eff_err_valLOO=eff_v.astype(np.float32), eff_err_randavg=eff_r.astype(np.float32),
             dataset=a.dataset, fold=a.fold, tag=a.tag, checkpoint=ckpt, selection=selection,
             best_test_acc=np.float32(best_test), K=a.K, attention='dense',
             value_fn='softmax_true_class', num_classes=num_classes)

    summary = {
        'dataset': a.dataset, 'tag': a.tag, 'fold': a.fold, 'checkpoint': ckpt,
        'selection': selection, 'best_test_acc': float(best_test),
        'n_test_samples': int(N), 'num_modalities': M, 'modalities': list(mods),
        'num_classes': int(num_classes), 'value_fn': 'softmax_true_class', 'attention': 'dense',
        'dense_forced_blocks': int(n_forced), 'v_empty': float(v_empty),
        'valLOO_order': list(vlo_order), 'K_random_orders': a.K,
        'variant_valLOO': stats_v, 'variant_randavg': stats_r,
        'npz_path': npz,
    }
    json.dump(summary, open(os.path.join(out_dir, f'shapley_{a.tag}_summary.json'), 'w'), indent=2)
    print(f'[shapley] {a.dataset}/{a.tag} N={N} sel={selection} '
          f'full_acc valLOO={stats_v["full_coalition_test_acc"]:.4f} randavg={stats_r["full_coalition_test_acc"]:.4f} '
          f'| eff_max valLOO={stats_v["efficiency_max_abs_err"]:.1e} randavg={stats_r["efficiency_max_abs_err"]:.1e} -> {npz}')
    print('   mean|phi| valLOO :', {m: round(v, 4) for m, v in stats_v['mean_abs_shapley_per_modality'].items()})
    print('   mean|phi| randavg:', {m: round(v, 4) for m, v in stats_r['mean_abs_shapley_per_modality'].items()})


if __name__ == '__main__':
    main()
