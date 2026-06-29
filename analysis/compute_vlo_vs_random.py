"""seqA adaptive early-exit curves under TWO modality orderings, on 3 datasets:
  (A) vlo    = val-LOO strong-first order
  (B) random = each sample's exit averaged over 5 random orderings (per fold,
               seeded default_rng(1234+fold)); per-tau (acc,f1,meanK) averaged
               over the 5 perms, then averaged over the 3 folds.

Reuses the data/model loading + prefix/exit helpers from the three template
scripts (compare_seqexit_admn_T256.py / compute_pamap2_T512_exit.py /
compute_seqa_loo_iemocap.py). Does NOT plot. Writes
../results_3p3/seqA_vlo_vs_random_exit_3ds.json. Run from clean_models/train/.
"""
from __future__ import annotations
import glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_function import set_seed
from models.seqA import (
    DualVideoBottleneckModelV6Downsample)

set_seed(42)
TAUS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 10.0]
N_RAND = 5
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_3p3')


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
print('Device:', dev)


# ---- generic exit simulation (shared) ----
def simulate(labels, Hs, preds, M, tau):
    N = len(labels); ek = np.full(N, M)
    for k in range(M):
        ek = np.where((ek == M) & (Hs[:, k] <= tau), k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    return float((pe == labels).mean()), float(f1_score(labels, pe, average='macro')), float(ek.mean())


def curve_for_order(labels, Hs, preds, M):
    """-> dict tau -> (acc,f1,mk) for a single (Hs,preds) realization."""
    return {t: simulate(labels, Hs, preds, M, t) for t in TAUS}


# =====================================================================
#  HAR-style datasets (daliahar uses HARDataset; pamap2 uses subjects loader)
# =====================================================================
class _V6Cfg:
    def __init__(s, mods, variates):
        s.modalities = list(mods); s.variates = {m: variates[m] for m in mods}
        f = s.modalities[0]; s.video_high_dim = s.variates[f]; s.video_low_dim = s.variates[f]


def build_har(C, mods, variates, T, NUMC, nlpm):
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=_V6Cfg(mods, variates), output_dim=NUMC, input_length=T,
        d_model=C['d_model'], nhead=C['nhead'], num_layers_per_modal=nlpm, num_layers=C['num_layers'],
        dropout=C['dropout'], verbose=False, video_low_dim=variates[mods[0]],
        video_high_dim=variates[mods[0]], use_bottleneck=True, n_bottlenecks=C['n_bottlenecks'],
        fusion_layer=C['fusion_layer'], use_sparse_moe=False, num_experts=C['num_experts'], expert_k=1,
        internal_dim=C['internal_dim'], bottleneck_head_pos=True, use_sparse_attn=C['use_sparse_attn'],
        factor=C['base_factor'], selector_video_source='high', encoder_video_source='high', no_selector=True,
        use_weighted_factor=False, use_triton=False, num_classes=NUMC, sparse_attn_variant=C['sparse_attn_variant'],
        strat_block_size=C['strat_block_size'], downsample_min_len=C['downsample_min_len'],
        use_batched_fusion=C['use_batched_fusion'], per_modal_distill=C['per_modal_distill'],
        per_modal_downsample_min_len=C['per_modal_downsample_min_len'], fusion_add_pos_embeds=False,
        fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full', fusion_mlp_ratio=1.0,
        fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'], seq_modality_order=C['seq_modality_order'],
        seq_random_order=C['seq_random_order'], bottleneck_agg_mode='gate', n_fusion_distill=-1).to(dev).float()


def sub_har(x, mods, sl):
    h = {m: x[:, :, sl[m][0]:sl[m][1]] for m in mods}
    return h, dict(h)


@torch.no_grad()
def acc_with_har(model, loader, keep, sl):
    cor = n = 0
    for x, y in loader:
        x = x.to(dev).float(); o = model(*sub_har(x, keep, sl), training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        cor += int((o.argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


@torch.no_grad()
def prefix_logits_har(model, loader, order, sl, M):
    by, labels = [], None
    for k in range(1, M + 1):
        outs, ys = [], []
        for x, y in loader:
            x = x.to(dev).float(); o = model(*sub_har(x, order[:k], sl), training=False, return_selection_info=False)
            o = o[0] if isinstance(o, tuple) else o
            outs.append(o.cpu()); ys.append(y)
        by.append(torch.cat(outs)); labels = torch.cat(ys).numpy() if labels is None else labels
    L = torch.stack(by, 1); P = F.softmax(L, -1)
    Hs = -(P * P.clamp_min(1e-9).log()).sum(-1).numpy(); preds = L.argmax(-1).numpy()
    return labels, Hs, preds


# =====================================================================
#  IEMOCAP (list-batch, high/low dicts)
# =====================================================================
def build_iemocap(C, mods, variates, input_length, NUMC):
    cfg = _V6Cfg(mods, variates)
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=cfg, output_dim=NUMC, input_length=input_length,
        d_model=C['d_model'], nhead=C['nhead'], num_layers_per_modal=C['num_layers_per_modal'],
        num_layers=C['num_layers'], dropout=C['dropout'], verbose=False,
        video_low_dim=cfg.video_low_dim, video_high_dim=cfg.video_high_dim, use_bottleneck=True,
        n_bottlenecks=C['n_bottlenecks'], fusion_layer=C['fusion_layer'], use_sparse_moe=False,
        num_experts=C['num_experts'], expert_k=1, internal_dim=C['internal_dim'], bottleneck_head_pos=True,
        use_sparse_attn=C['use_sparse_attn'], factor=C['base_factor'], selector_video_source='high',
        encoder_video_source='high', no_selector=True, use_weighted_factor=False, use_triton=False,
        num_classes=NUMC, sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
        downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
        per_modal_distill=C['per_modal_distill'], per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
        fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full',
        fusion_mlp_ratio=1.0, fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'],
        seq_modality_order=C['seq_modality_order'], seq_random_order=C['seq_random_order'],
        bottleneck_agg_mode=C.get('bottleneck_agg_mode', 'gate'), n_fusion_distill=C['n_fusion_distill']).to(dev).float()


def unpack_iemocap(batch, mods):
    *feats, label = batch
    feats = [f.to(dev).float() for f in feats]
    label = label.squeeze(-1).long()
    high = {m: feats[2 * j] for j, m in enumerate(mods)}
    low = {m: feats[2 * j + 1] for j, m in enumerate(mods)}
    return high, low, label


@torch.no_grad()
def acc_with_iemocap(model, loader, all_mods, keep):
    cor = n = 0
    for batch in loader:
        high, low, y = unpack_iemocap(batch, all_mods)
        o = model({m: high[m] for m in keep}, {m: low[m] for m in keep},
                  training=False, return_selection_info=False)
        o = o[0] if isinstance(o, tuple) else o
        cor += int((o.argmax(-1).cpu() == y).sum()); n += y.shape[0]
    return cor / n


@torch.no_grad()
def prefix_logits_iemocap(model, loader, all_mods, order, M):
    by, labels = [], None
    for k in range(1, M + 1):
        keep = order[:k]; outs, ys = [], []
        for batch in loader:
            high, low, y = unpack_iemocap(batch, all_mods)
            o = model({m: high[m] for m in keep}, {m: low[m] for m in keep},
                      training=False, return_selection_info=False)
            o = o[0] if isinstance(o, tuple) else o
            outs.append(o.cpu()); ys.append(y.cpu())
        by.append(torch.cat(outs)); labels = torch.cat(ys).numpy() if labels is None else labels
    L = torch.stack(by, 1); P = F.softmax(L, -1)
    Hs = -(P * P.clamp_min(1e-9).log()).sum(-1).numpy(); preds = L.argmax(-1).numpy()
    return labels, Hs, preds


def reorder_prefix(labels, Hs_base, preds_base, base_order, target_order):
    """Map a prefix realization computed for `base_order` onto `target_order`.

    prefix_logits for order O gives, at column k, the logits of {O[0..k]}.
    For a different permutation we'd need a *different* subset at each k, so we
    cannot reuse base columns directly -> we must recompute per order. This
    helper is intentionally NOT used (kept for clarity); each order is run
    through prefix_logits directly.
    """
    raise NotImplementedError


# =====================================================================
#  aggregation helper
# =====================================================================
def avg_curves(curve_list):
    """curve_list: list of {tau:(acc,f1,mk)} -> {tau:(acc,f1,mk)} averaged."""
    out = {}
    for t in TAUS:
        accs = [c[t][0] for c in curve_list]
        f1s = [c[t][1] for c in curve_list]
        mks = [c[t][2] for c in curve_list]
        out[t] = (float(np.mean(accs)), float(np.mean(f1s)), float(np.mean(mks)))
    return out


def to_rows(curve):
    """curve {tau:(acc,f1,mk)} -> list[[meanK,acc,f1]] in TAUS order."""
    return [[curve[t][2], curve[t][0], curve[t][1]] for t in TAUS]


OUT = {}

# =====================================================================
#  IEMOCAP
# =====================================================================
def run_iemocap():
    from data.IEMOCAP.get_data import (get_dataloader as iemocap_get_dataloader,
                                        FEATURE_DIMS, NUM_CLASSES)
    SEQA_DIR = ('/files1/haodong/MAESTRO_ttn_robustness_old/clean_models/results_3p3/'
                'seqA_iemocap/2026-06-16_IEMOCAP_v6_seqfusion_late_fusion_cs_seqA_randord')
    DATA_ROOT = '/files1/haodong/data/IEMOCAP'
    C = json.load(open(os.path.join(SEQA_DIR, 'config.json')))
    MODS = list(C['modalities']); M = len(MODS); NUMC = NUM_CLASSES
    variates = {m: FEATURE_DIMS[m] for m in MODS}
    variates_full = dict(variates); variates_full.setdefault('video', FEATURE_DIMS.get('video', 1024))
    input_length = C['max_seq_len'] if C['max_seq_len'] > 0 else 600
    gpk = json.load(open(os.path.join(R, 'seqA3p3_gflops.json')))['iemocap']['gflops_per_k']

    vlo_folds, rand_folds, orders = [], [], []
    for fold in range(3):
        _, val_loader, test_loader = iemocap_get_dataloader(
            data_root=DATA_ROOT, batch_size=64, num_workers=4, modalities=MODS,
            split_id=fold, train_shuffle=False, max_seq_len=C['max_seq_len'],
            time_compression_ratio=C['time_compression_ratio'],
            use_batched_fusion=C['use_batched_fusion'], available_sessions=None,
            split_mode='cross_subject')
        # iemocap_get_dataloader returns (train, val, test)
        model = build_iemocap(C, MODS, variates, input_length, NUMC)
        sd = torch.load(os.path.join(SEQA_DIR, f'best_model_fold{fold}.pth'),
                        map_location='cpu', weights_only=True)
        model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False)
        model.eval()
        full = acc_with_iemocap(model, val_loader, MODS, MODS)
        loo = {m: full - acc_with_iemocap(model, val_loader, MODS, [x for x in MODS if x != m]) for m in MODS}
        vlo_order = sorted(MODS, key=lambda m: -loo[m]); orders.append(vlo_order)
        print(f'  iemocap fold{fold} full_val={full:.4f} vlo_order={vlo_order}')

        labels, Hs, preds = prefix_logits_iemocap(model, test_loader, MODS, vlo_order, M)
        vlo_folds.append(curve_for_order(labels, Hs, preds, M))

        rng = np.random.default_rng(1234 + fold)
        rc = []
        for _ in range(N_RAND):
            perm = list(rng.permutation(MODS))
            lb, Hr, pr = prefix_logits_iemocap(model, test_loader, MODS, perm, M)
            rc.append(curve_for_order(lb, Hr, pr, M))
        rand_folds.append(avg_curves(rc))
        del model; torch.cuda.empty_cache()

    vlo = avg_curves(vlo_folds); rnd = avg_curves(rand_folds)
    OUT['iemocap'] = {'M': M, 'gflops_per_k': gpk, 'vlo': to_rows(vlo), 'random': to_rows(rnd)}
    return orders


# =====================================================================
#  DALIAHAR
# =====================================================================
def run_daliahar():
    from utils.dataset_cfg import DaliaHAR
    from data.dataset_builder import HARDataset
    ROOT = '/files1/haodong/data/processed_dalia_activity'
    RESROOT = '../results_3p3/T256'
    cfg = DaliaHAR(); MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
    variates = dict(cfg.variates); T = cfg.duration * cfg.base_sample_rate
    sl, off = {}, 0
    for m in MODS:
        sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
    seqa_dir = sorted(glob.glob(os.path.join(RESROOT, 'seqA', '*')))[0]
    C = json.load(open(os.path.join(seqa_dir, 'config.json')))
    gpk = [0.137238912, 0.274796416, 0.41235392, 0.549911424, 0.687534464]
    val_ds = HARDataset(ROOT, MODS, cfg.val_set, cfg, 'sax', sax_params={'alphabet_size': 20, 'word_length': 1})
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    vlo_folds, rand_folds, orders = [], [], []
    for fold in range(3):
        fc = cfg.folds[fold]
        test_ds = HARDataset(ROOT, MODS, fc['eval_set'], cfg, 'sax', sax_params={'alphabet_size': 20, 'word_length': 1})
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
        model = build_har(C, MODS, variates, T, NUMC, C['num_layers_per_modal'])
        sd = torch.load(os.path.join(seqa_dir, f'best_model_fold{fold}.pth'), map_location='cpu', weights_only=True)
        model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()
        full = acc_with_har(model, val_loader, MODS, sl)
        loo = {m: full - acc_with_har(model, val_loader, [x for x in MODS if x != m], sl) for m in MODS}
        vlo_order = sorted(MODS, key=lambda m: -loo[m]); orders.append(vlo_order)
        print(f'  daliahar fold{fold} full_val={full:.4f} vlo_order={vlo_order}')

        labels, Hs, preds = prefix_logits_har(model, test_loader, vlo_order, sl, M)
        vlo_folds.append(curve_for_order(labels, Hs, preds, M))

        rng = np.random.default_rng(1234 + fold)
        rc = []
        for _ in range(N_RAND):
            perm = list(rng.permutation(MODS))
            lb, Hr, pr = prefix_logits_har(model, test_loader, perm, sl, M)
            rc.append(curve_for_order(lb, Hr, pr, M))
        rand_folds.append(avg_curves(rc))
        del model; torch.cuda.empty_cache()

    vlo = avg_curves(vlo_folds); rnd = avg_curves(rand_folds)
    OUT['daliahar'] = {'M': M, 'gflops_per_k': gpk, 'vlo': to_rows(vlo), 'random': to_rows(rnd)}
    return orders


# =====================================================================
#  PAMAP2
# =====================================================================
def run_pamap2():
    from utils.dataset_cfg import PAMAP2
    from data.pamap2_crosssubject import load_pamap2_subjects
    cfg = PAMAP2(); MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes; T = 512
    variates = dict(cfg.variates)
    sl, off = {}, 0
    for m in MODS:
        sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
    seqa_dir = sorted(glob.glob('../results_3p3/seqA_pamap2/2026-06-20*'))[0]
    C = json.load(open(os.path.join(seqa_dir, 'config.json')))
    gpk = [0.419, 0.838, 1.257, 1.675]

    def loader(subs):
        return DataLoader(load_pamap2_subjects(cfg.cross_subject_dir, MODS, subs, cfg, 'sax_noisy',
                          sax_params={'alphabet_size': 20, 'word_length': 1}),
                          batch_size=64, shuffle=False, num_workers=4)

    val_loader = loader(cfg.val_set)
    vlo_folds, rand_folds, orders = [], [], []
    for fold in range(3):
        fc = cfg.folds[fold]; test_loader = loader(fc['eval_set'])
        model = build_har(C, MODS, variates, T, NUMC, C['num_layers_per_modal'])
        sd = torch.load(os.path.join(seqa_dir, f'best_model_fold{fold}.pth'), map_location='cpu', weights_only=True)
        model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); model.eval()
        full = acc_with_har(model, val_loader, MODS, sl)
        loo = {m: full - acc_with_har(model, val_loader, [x for x in MODS if x != m], sl) for m in MODS}
        vlo_order = sorted(MODS, key=lambda m: -loo[m]); orders.append(vlo_order)
        print(f'  pamap2 fold{fold} full_val={full:.4f} vlo_order={vlo_order}')

        labels, Hs, preds = prefix_logits_har(model, test_loader, vlo_order, sl, M)
        vlo_folds.append(curve_for_order(labels, Hs, preds, M))

        rng = np.random.default_rng(1234 + fold)
        rc = []
        for _ in range(N_RAND):
            perm = list(rng.permutation(MODS))
            lb, Hr, pr = prefix_logits_har(model, test_loader, perm, sl, M)
            rc.append(curve_for_order(lb, Hr, pr, M))
        rand_folds.append(avg_curves(rc))
        del model; torch.cuda.empty_cache()

    vlo = avg_curves(vlo_folds); rnd = avg_curves(rand_folds)
    OUT['pamap2'] = {'M': M, 'gflops_per_k': gpk, 'vlo': to_rows(vlo), 'random': to_rows(rnd)}
    return orders


if __name__ == '__main__':
    print('=== IEMOCAP ==='); io = run_iemocap()
    print('=== DALIAHAR ==='); do = run_daliahar()
    print('=== PAMAP2 ==='); po = run_pamap2()

    outpath = os.path.join(R, 'seqA_vlo_vs_random_exit_3ds.json')
    json.dump(OUT, open(outpath, 'w'), indent=2)
    print('\nsaved ->', outpath)

    # ---- sanity + small report ----
    consensus_orders = {'iemocap': io, 'daliahar': do, 'pamap2': po}
    print('\n================ SANITY / REPORT ================')
    for ds in ['iemocap', 'daliahar', 'pamap2']:
        o = OUT[ds]; M = o['M']
        vlo = np.array(o['vlo']); rnd = np.array(o['random'])  # cols [meanK,acc,f1] over TAUS
        # tau=10.0 is last -> full modality, order irrelevant
        print(f'\n[{ds}] M={M}')
        print(f'  val-LOO per-fold orders: {consensus_orders[ds]}')
        print(f'  tau=10 (full): vlo meanK={vlo[-1,0]:.3f} acc={vlo[-1,1]:.4f} | '
              f'random meanK={rnd[-1,0]:.3f} acc={rnd[-1,1]:.4f} | '
              f'|dacc|={abs(vlo[-1,1]-rnd[-1,1]):.4f} (want <=0.01)')
        # cheapest tau = tau index 0 (lowest meanK)
        print(f'  cheapest tau=0.0: vlo (meanK={vlo[0,0]:.3f}, acc={vlo[0,1]:.4f}) | '
              f'random (meanK={rnd[0,0]:.3f}, acc={rnd[0,1]:.4f}) | '
              f'gap(vlo-random)={vlo[0,1]-rnd[0,1]:+.4f}')
        print('  (meanK, vlo_acc, random_acc) at tau idx [0,3,6,9,14]:')
        for i in [0, 3, 6, 9, 14]:
            print(f'    tau={TAUS[i]:<5}: vlo(mk={vlo[i,0]:.2f}, acc={vlo[i,1]:.4f}) '
                  f'random(mk={rnd[i,0]:.2f}, acc={rnd[i,1]:.4f})')
