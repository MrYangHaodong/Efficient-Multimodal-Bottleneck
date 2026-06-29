"""Ablation: random-order training vs FIXED val-LOO-order training for seqA.
Both models are evaluated identically: modalities fed in the consensus val-LOO
order (strong-first), seq_modality_order/random overridden to None/False so the
processing order is driven purely by input order. Reports full-modality (k=M)
test acc/macro-F1 and the prefix curve k=1..M (early-exit behaviour)."""
import glob, json, os, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from compute_vlo_vs_random import (build_har, build_iemocap, sub_har, unpack_iemocap, dev)

R = os.path.dirname(os.path.abspath(__file__))
ORD = json.load(open(os.path.join(R, '../results_3p3/vlo_consensus_orders.json')))
# Random-order reference = seqA_gap_dual (same dual-order main + head_mode=gap as the
# fixed-order run; the ONLY difference is --seq_random_order vs --seq_modality_order).
RAND = {'daliahar': os.path.join(R, '../results_3p3/seqA_daliahar/2026-06-21_DaliaHAR_seqA_gap_dual_seqA_randord/'),
        'pamap2': os.path.join(R, '../results_3p3/seqA_pamap2/2026-06-21_PAMAP2_seqA_gap_dual_seqA_randord/'),
        'iemocap': os.path.join(R, '../results_3p3/seqA_iemocap/2026-06-21_IEMOCAP_seqA_gap_dual_seqA_randord/'),
        'dsads': os.path.join(R, '../results_3p3/seqA_dsads/2026-06-21_DSADS_seqA_gap_dual_seqA_randord/')}
FIX = {ds: os.path.join(R, f'../results_3p3/seqA_fixedorder/{ds}/seqA/') for ds in RAND}
# both variants use the LOO-order-selected checkpoint (matches val-LOO deployment order)
def _wpath(model_dir, fold):
    for stem in ('best_model_loo_fold', 'best_model_fold'):
        for pat in (f'/*/{stem}{fold}.pth', f'/{stem}{fold}.pth'):
            hits = glob.glob(model_dir + pat)
            if hits:
                return sorted(hits)[0]
    raise FileNotFoundError(f'no checkpoint for fold {fold} under {model_dir}')
NFOLD = {'daliahar': 3, 'pamap2': 3, 'iemocap': 3, 'dsads': 4}


def _override(C):
    C = dict(C); C['seq_modality_order'] = None; C['seq_random_order'] = False; return C


@torch.no_grad()
def _prefix_metrics_har(model, loader, order, sl, M):
    by, ys = [], None
    for k in range(1, M + 1):
        outs, lab = [], []
        for x, y in loader:
            x = x.to(dev).float()
            o = model(*sub_har(x, order[:k], sl), training=False, return_selection_info=False)
            o = o[0] if isinstance(o, tuple) else o
            outs.append(o.cpu()); lab.append(y)
        by.append(torch.cat(outs)); ys = torch.cat(lab).numpy() if ys is None else ys
    preds = torch.stack(by, 1).argmax(-1).numpy()  # (N,M)
    acc = [(preds[:, k] == ys).mean() for k in range(M)]
    f1 = [f1_score(ys, preds[:, k], average='macro') for k in range(M)]
    return np.array(acc), np.array(f1)


@torch.no_grad()
def _prefix_metrics_iemo(model, loader, all_mods, order, M):
    by, ys = [], None
    for k in range(1, M + 1):
        keep = order[:k]; outs, lab = [], []
        for batch in loader:
            high, low, y = unpack_iemocap(batch, all_mods)
            o = model({m: high[m] for m in keep}, {m: low[m] for m in keep},
                      training=False, return_selection_info=False)
            o = o[0] if isinstance(o, tuple) else o
            outs.append(o.cpu()); lab.append(y)
        by.append(torch.cat(outs)); ys = torch.cat(lab).numpy() if ys is None else ys
    preds = torch.stack(by, 1).argmax(-1).numpy()
    acc = [(preds[:, k] == ys).mean() for k in range(M)]
    f1 = [f1_score(ys, preds[:, k], average='macro') for k in range(M)]
    return np.array(acc), np.array(f1)


def test_loader_har(ds, fold):
    from utils.dataset_cfg import DaliaHAR, PAMAP2, DSADS
    SAX1 = {'alphabet_size': 20, 'word_length': 1}
    if ds == 'daliahar':
        from data.dataset_builder import HARDataset
        cfg = DaliaHAR(); eval_subj = cfg.folds[fold]['eval_set']
        ds_obj = HARDataset('/files1/haodong/data/processed_dalia_activity', list(cfg.modalities), eval_subj, cfg, 'sax', sax_params=SAX1)
    elif ds == 'pamap2':
        from data.pamap2_crosssubject import load_pamap2_subjects
        cfg = PAMAP2(); eval_subj = cfg.folds[fold]['eval_set']
        ds_obj = load_pamap2_subjects(cfg.cross_subject_dir, list(cfg.modalities), eval_subj, cfg, 'sax_noisy', sax_params=SAX1)
    else:
        from data.dataset_builder import DSADSDataset
        cfg = DSADS(); _trg = torch.load(f'/files1/haodong/data/DSADS/dsad_diversify_dict_scenario{fold}_trg.pt', weights_only=False)
        ds_obj = DSADSDataset({'samples': _trg['samples'], 'labels': _trg['labels']}, list(cfg.modalities), cfg, 'sax', sax_params=SAX1)
    return cfg, DataLoader(ds_obj, batch_size=64, shuffle=False, num_workers=4)


def eval_har(ds, model_dir):
    from utils.dataset_cfg import DaliaHAR, PAMAP2, DSADS
    cfg0 = {'daliahar': DaliaHAR, 'pamap2': PAMAP2, 'dsads': DSADS}[ds]()
    MODS = list(cfg0.modalities); M = len(MODS); variates = dict(cfg0.variates); NUMC = cfg0.num_classes
    sl, off = {}, 0
    for m in MODS:
        sl[m] = (off, off + cfg0.variates[m]); off += cfg0.variates[m]
    C = _override(json.load(open(glob.glob(model_dir + '/*/config.json')[0] if glob.glob(model_dir + '/*/config.json') else model_dir + '/config.json')))
    T = {'daliahar': 256, 'pamap2': 512, 'dsads': 125}[ds]
    order = ORD[ds]['consensus']; accs, f1s = [], []
    for fold in range(NFOLD[ds]):
        cfg, vl = test_loader_har(ds, fold)
        m = build_har(C, MODS, variates, T, NUMC, C['num_layers_per_modal'])
        sd = torch.load(_wpath(model_dir, fold), map_location='cpu', weights_only=True)
        m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); m.eval()
        a, f = _prefix_metrics_har(m, vl, order, sl, M)
        accs.append(a); f1s.append(f); del m; torch.cuda.empty_cache()
    return np.array(accs), np.array(f1s)  # (folds, M)


def eval_iemo(model_dir):
    from data.IEMOCAP.get_data import get_dataloader, FEATURE_DIMS, NUM_CLASSES
    C = _override(json.load(open(glob.glob(model_dir + '/*/config.json')[0] if glob.glob(model_dir + '/*/config.json') else model_dir + '/config.json')))
    MODS = list(C['modalities']); M = len(MODS); variates = {m: FEATURE_DIMS[m] for m in MODS}
    T = C['max_seq_len'] if C['max_seq_len'] > 0 else 600; order = ORD['iemocap']['consensus']; accs, f1s = [], []
    for fold in range(3):
        _, _, tl = get_dataloader(data_root='/files1/haodong/data/IEMOCAP', batch_size=24, num_workers=4, modalities=MODS,
                                  split_id=fold, train_shuffle=False, max_seq_len=C['max_seq_len'], time_compression_ratio=C['time_compression_ratio'],
                                  use_batched_fusion=C['use_batched_fusion'], available_sessions=None, split_mode='cross_subject')
        m = build_iemocap(C, MODS, variates, T, NUM_CLASSES)
        sd = torch.load(_wpath(model_dir, fold), map_location='cpu', weights_only=True)
        m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); m.eval()
        a, f = _prefix_metrics_iemo(m, tl, MODS, order, M)
        accs.append(a); f1s.append(f); del m; torch.cuda.empty_cache()
    return np.array(accs), np.array(f1s)


out = {}
for ds in ['iemocap', 'daliahar', 'pamap2', 'dsads']:
    rec = {}
    for tag, mdir in [('random', RAND[ds]), ('fixed', FIX[ds])]:
        a, f = (eval_iemo(mdir) if ds == 'iemocap' else eval_har(ds, mdir))
        M = a.shape[1]
        rec[tag] = {'curve_acc': a.mean(0).tolist(), 'curve_f1': f.mean(0).tolist(),
                    'full_acc': float(a[:, -1].mean()), 'full_acc_std': float(a[:, -1].std()),
                    'full_f1': float(f[:, -1].mean()), 'full_f1_std': float(f[:, -1].std()),
                    'mean_prefix_acc': float(a.mean()), 'mean_prefix_f1': float(f.mean())}
    out[ds] = rec
    r, x = rec['random'], rec['fixed']
    print(f"\n==== {ds} (M={len(r['curve_acc'])}, consensus order) ====")
    print(f"  FULL-modality acc: random {r['full_acc']:.4f}±{r['full_acc_std']:.3f} | fixed {x['full_acc']:.4f}±{x['full_acc_std']:.3f}  (Δ fixed-rand {100*(x['full_acc']-r['full_acc']):+.2f}pp)")
    print(f"  FULL-modality f1 : random {r['full_f1']:.4f}        | fixed {x['full_f1']:.4f}         (Δ {100*(x['full_f1']-r['full_f1']):+.2f}pp)")
    print(f"  mean-prefix  acc : random {r['mean_prefix_acc']:.4f}        | fixed {x['mean_prefix_acc']:.4f}         (Δ {100*(x['mean_prefix_acc']-r['mean_prefix_acc']):+.2f}pp)")
    print(f"  mean-prefix  f1  : random {r['mean_prefix_f1']:.4f}        | fixed {x['mean_prefix_f1']:.4f}         (Δ {100*(x['mean_prefix_f1']-r['mean_prefix_f1']):+.2f}pp)")
    print(f"  curve acc random: {[round(v,3) for v in r['curve_acc']]}")
    print(f"  curve acc fixed : {[round(v,3) for v in x['curve_acc']]}")

json.dump(out, open(os.path.join(R, '../results_3p3/ablation_random_vs_fixed_order.json'), 'w'), indent=2)
print('\nsaved -> results_3p3/ablation_random_vs_fixed_order.json')
