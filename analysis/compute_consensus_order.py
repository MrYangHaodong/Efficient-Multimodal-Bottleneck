"""Per-dataset consensus val-LOO modality order -> canonical indices, for the
fixed-order training ablation. Reuses helpers from compute_vlo_vs_random.py."""
import glob, json, os, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from compute_vlo_vs_random import (build_har, acc_with_har, build_iemocap, acc_with_iemocap, dev)
out = {}


def har_consensus(ds):
    from utils.dataset_cfg import DaliaHAR, PAMAP2, DSADS
    if ds == 'daliahar':
        from data.dataset_builder import HARDataset
        cfg = DaliaHAR(); ROOT = '/files1/haodong/data/processed_dalia_activity'; T = cfg.duration * cfg.base_sample_rate
        seqa = sorted(glob.glob('../results_3p3/T256/seqA/*'))[0]; folds = range(3); SAX = {'alphabet_size': 20, 'word_length': 1}
        valload = lambda: DataLoader(HARDataset(ROOT, list(cfg.modalities), cfg.val_set, cfg, 'sax', sax_params=SAX), batch_size=64, shuffle=False, num_workers=4)
    elif ds == 'pamap2':
        from data.pamap2_crosssubject import load_pamap2_subjects
        cfg = PAMAP2(); T = 512; seqa = sorted(glob.glob('../results_3p3/seqA_pamap2/2026-06-20*'))[0]; folds = range(3)
        valload = lambda: DataLoader(load_pamap2_subjects(cfg.cross_subject_dir, list(cfg.modalities), cfg.val_set, cfg, 'sax_noisy', sax_params={'alphabet_size': 20, 'word_length': 1}), batch_size=64, shuffle=False, num_workers=4)
    else:
        cfg = DSADS(); T = (cfg.duration * cfg.base_sample_rate) // 1; seqa = sorted(glob.glob('../results_3p3/dsads/seqA/*'))[0]; folds = range(4); valload = None
    MODS = list(cfg.modalities); M = len(MODS); variates = dict(cfg.variates); NUMC = cfg.num_classes
    sl, off = {}, 0
    for m in MODS:
        sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
    C = json.load(open(seqa + '/config.json')); borda = {m: 0.0 for m in MODS}
    for fold in folds:
        if ds == 'dsads':
            from data.dataset_builder import DSADSDataset
            _src = torch.load(f'/files1/haodong/data/DSADS/dsad_diversify_dict_scenario{fold}_src.pt', weights_only=False)
            _, _vx, _, _vy = train_test_split(_src['samples'], _src['labels'], test_size=0.2, random_state=2711, stratify=_src['labels'])
            vl = DataLoader(DSADSDataset({'samples': _vx, 'labels': _vy}, MODS, cfg, 'sax', sax_params={'alphabet_size': 20, 'word_length': 1}), batch_size=64, shuffle=False, num_workers=4)
        else:
            vl = valload()
        m = build_har(C, MODS, variates, T, NUMC, C['num_layers_per_modal'])
        sd = torch.load(f'{seqa}/best_model_fold{fold}.pth', map_location='cpu', weights_only=True)
        m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); m.eval()
        full = acc_with_har(m, vl, MODS, sl)
        loo = {mm: full - acc_with_har(m, vl, [x for x in MODS if x != mm], sl) for mm in MODS}
        order = sorted(MODS, key=lambda mm: -loo[mm])
        for r, mm in enumerate(order):
            borda[mm] += (M - r)
        del m; torch.cuda.empty_cache()
    cons = sorted(MODS, key=lambda mm: -borda[mm]); idx = [MODS.index(mm) for mm in cons]
    out[ds] = {'canonical': MODS, 'consensus': cons, 'indices': idx}
    print(f"{ds}: canonical={MODS}\n   consensus={cons}\n   indices={idx}")


def iemocap_consensus():
    from data.IEMOCAP.get_data import get_dataloader, FEATURE_DIMS, NUM_CLASSES
    seqa = '../results_3p3/seqA_iemocap/2026-06-16_IEMOCAP_v6_seqfusion_late_fusion_cs_seqA_randord'
    C = json.load(open(seqa + '/config.json')); MODS = list(C['modalities']); M = len(MODS); NUMC = NUM_CLASSES
    variates = {m: FEATURE_DIMS[m] for m in MODS}; T = C['max_seq_len'] if C['max_seq_len'] > 0 else 600; borda = {m: 0.0 for m in MODS}
    for fold in range(3):
        _, vl, _ = get_dataloader(data_root='/files1/haodong/data/IEMOCAP', batch_size=64, num_workers=4, modalities=MODS, split_id=fold, train_shuffle=False, max_seq_len=C['max_seq_len'], time_compression_ratio=C['time_compression_ratio'], use_batched_fusion=C['use_batched_fusion'], available_sessions=None, split_mode='cross_subject')
        m = build_iemocap(C, MODS, variates, T, NUMC); sd = torch.load(f'{seqa}/best_model_fold{fold}.pth', map_location='cpu', weights_only=True)
        m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); m.eval()
        full = acc_with_iemocap(m, vl, MODS, MODS)
        loo = {mm: full - acc_with_iemocap(m, vl, MODS, [x for x in MODS if x != mm]) for mm in MODS}
        order = sorted(MODS, key=lambda mm: -loo[mm])
        for r, mm in enumerate(order):
            borda[mm] += (M - r)
        del m; torch.cuda.empty_cache()
    cons = sorted(MODS, key=lambda mm: -borda[mm]); idx = [MODS.index(mm) for mm in cons]
    out['iemocap'] = {'canonical': MODS, 'consensus': cons, 'indices': idx}
    print(f"iemocap: canonical={MODS}\n   consensus={cons}\n   indices={idx}")


iemocap_consensus()
for ds in ['daliahar', 'pamap2', 'dsads']:
    har_consensus(ds)
json.dump(out, open('../results_3p3/vlo_consensus_orders.json', 'w'), indent=2)
print('saved -> results_3p3/vlo_consensus_orders.json')
