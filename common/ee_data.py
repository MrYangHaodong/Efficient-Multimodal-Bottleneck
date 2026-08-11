"""Shared 4-dataset loader for the early-exit baselines (MMEE / CREMA-style).
Extracted verbatim from main_dymo_proto.py `_build_loaders` so the new scripts use
the IDENTICAL data pipeline (IEMOCAP cross-subject float; daliahar/pamap2/dsads sax).

build_loaders(args) -> (train_loader, val_loader, eval_loader, cfg, input_length,
                        input_mode, batch_size, unpack)
where unpack(batch, device) -> (x [B,T,sum(variates)], y [B]).
"""
import os
import torch
from torch.utils.data import DataLoader
from common.loader_utils import _concat_iemocap


def build_loaders(args):
    ds = args.dataset

    if ds == 'iemocap':
        from data.IEMOCAP.get_data import get_dataloader as iemocap_get_dataloader, IEMOCAPDataset
        bs = args.batch_size or 32
        mods = list(IEMOCAPDataset.ALL_MODALITIES)

        class _Cfg:
            pass
        from data.IEMOCAP.get_data import FEATURE_DIMS, NUM_CLASSES
        cfg = _Cfg(); cfg.modalities = mods
        cfg.variates = {m: FEATURE_DIMS[m] for m in mods}; cfg.num_classes = NUM_CLASSES
        fold_id = args.fold if args.fold is not None else 0
        tl, vl, el = iemocap_get_dataloader(
            data_root='/home/group/maestro_visual/data/IEMOCAP', batch_size=bs, num_workers=args.num_workers,
            modalities=mods, split_id=fold_id, train_shuffle=True,
            max_seq_len=args.max_seq_len, time_compression_ratio=args.time_compression_ratio,
            use_batched_fusion=True, available_sessions=None, split_mode=args.split_mode)
        in_len = args.max_seq_len if args.max_seq_len > 0 else 600
        M = len(mods)

        def unpack(batch, device):
            *feats, label = batch
            feats = [f.to(device).float() for f in feats]
            return _concat_iemocap(feats, M), label.squeeze(-1).long().to(device)
        return tl, vl, el, cfg, in_len, 'float', bs, unpack

    if ds == 'mosei':
        # CMU-MOSEI: 3 modalities (vision/audio/text GloVe), Acc-2 classification,
        # IEMOCAP-style list batch. split_id varies the train/val partition (test fixed);
        # the 5 seeds use split_id 0 and differ only by --seed.
        from data.CMU_MOSEI.get_data import (get_dataloader as mosei_get_dataloader,
                                             CMUMOSEIDataset, FEATURE_DIMS, NUM_CLASSES)
        bs = args.batch_size or 32
        mods = list(CMUMOSEIDataset.ALL_MODALITIES)

        class _Cfg:
            pass
        cfg = _Cfg(); cfg.modalities = mods
        cfg.variates = {m: FEATURE_DIMS[m] for m in mods}; cfg.num_classes = NUM_CLASSES
        fold_id = args.fold if args.fold is not None else 0
        tl, vl, el = mosei_get_dataloader(
            batch_size=bs, num_workers=args.num_workers, modalities=mods, split_id=fold_id,
            train_shuffle=True, max_seq_len=args.max_seq_len, time_compression_ratio=args.time_compression_ratio,
            use_batched_fusion=True, task='classification2')
        in_len = args.max_seq_len if args.max_seq_len > 0 else 50
        M = len(mods)

        def unpack(batch, device):
            *feats, label = batch
            feats = [f.to(device).float() for f in feats]
            return _concat_iemocap(feats, M), label.squeeze(-1).long().to(device)
        return tl, vl, el, cfg, in_len, 'float', bs, unpack

    if ds == 'eav':
        # EAV: 3 modalities (eeg/audio/video), 5-class emotion, FLOAT input,
        # cross-subject folds. Same IEMOCAP-style list batch + concat unpack.
        from data.EAV.get_data import (get_dataloader as eav_get_dataloader,
                                        EAVDataset, FEATURE_DIMS, NUM_CLASSES)
        bs = args.batch_size or 32
        mods = list(EAVDataset.ALL_MODALITIES)

        class _Cfg:
            pass
        cfg = _Cfg(); cfg.modalities = mods
        cfg.variates = {m: FEATURE_DIMS[m] for m in mods}; cfg.num_classes = NUM_CLASSES
        fold_id = args.fold if args.fold is not None else 0
        tl, vl, el = eav_get_dataloader(
            batch_size=bs, num_workers=args.num_workers, modalities=mods, split_id=fold_id,
            train_shuffle=True, max_seq_len=args.max_seq_len,
            time_compression_ratio=args.time_compression_ratio, use_batched_fusion=True,
            split_mode=args.split_mode)
        in_len = args.max_seq_len if args.max_seq_len > 0 else 128
        M = len(mods)

        def unpack(batch, device):
            *feats, label = batch
            feats = [f.to(device).float() for f in feats]
            return _concat_iemocap(feats, M), label.squeeze(-1).long().to(device)
        return tl, vl, el, cfg, in_len, 'float', bs, unpack

    if ds == 'meld':
        from data.MELD.get_data import (get_dataloader as meld_get_dataloader,
                                         MELDDataset, FEATURE_DIMS, NUM_CLASSES)
        bs = args.batch_size or 32
        mods = list(MELDDataset.ALL_MODALITIES)

        class _Cfg:
            pass
        cfg = _Cfg(); cfg.modalities = mods
        cfg.variates = {m: FEATURE_DIMS[m] for m in mods}; cfg.num_classes = NUM_CLASSES
        fold_id = args.fold if args.fold is not None else 0
        tl, vl, el = meld_get_dataloader(
            data_root='/home/group/maestro_visual/data/MELD/features',
            batch_size=bs, num_workers=args.num_workers, modalities=mods, split_id=fold_id,
            train_shuffle=True, max_seq_len=args.max_seq_len,
            time_compression_ratio=args.time_compression_ratio, use_batched_fusion=True)
        in_len = args.max_seq_len if args.max_seq_len > 0 else 128
        M = len(mods)

        def unpack(batch, device):
            *feats, label = batch
            feats = [f.to(device).float() for f in feats]
            return _concat_iemocap(feats, M), label.squeeze(-1).long().to(device)
        return tl, vl, el, cfg, in_len, 'float', bs, unpack

    if ds == 'cmi':
        from data.CMI.get_data import (get_dataloader as cmi_get_dataloader,
                                        CMIDataset, FEATURE_DIMS, NUM_CLASSES)
        bs = args.batch_size or 32
        mods = list(CMIDataset.ALL_MODALITIES)

        class _Cfg:
            pass
        cfg = _Cfg(); cfg.modalities = mods
        cfg.variates = {m: FEATURE_DIMS[m] for m in mods}; cfg.num_classes = NUM_CLASSES
        fold_id = args.fold if args.fold is not None else 0
        tl, vl, el = cmi_get_dataloader(
            data_root='/home/group/maestro_visual/data/CMI',
            batch_size=bs, num_workers=args.num_workers, modalities=mods, split_id=fold_id,
            train_shuffle=True, max_seq_len=args.max_seq_len,
            time_compression_ratio=args.time_compression_ratio, use_batched_fusion=True,
            split_mode='cross_subject')
        in_len = args.max_seq_len if args.max_seq_len > 0 else 128
        M = len(mods)

        def unpack(batch, device):
            *feats, label = batch
            feats = [f.to(device).float() for f in feats]
            return _concat_iemocap(feats, M), label.squeeze(-1).long().to(device)
        return tl, vl, el, cfg, in_len, 'float', bs, unpack

    if ds == 'czu_mhad':
        from data.CZU_MHAD.get_data import (get_dataloader as czumhad_get_dataloader,
                                              CZUMHADDataset, FEATURE_DIMS, NUM_CLASSES)
        bs = args.batch_size or 32
        mods = list(CZUMHADDataset.ALL_MODALITIES)

        class _Cfg:
            pass
        cfg = _Cfg(); cfg.modalities = mods
        cfg.variates = {m: FEATURE_DIMS[m] for m in mods}; cfg.num_classes = NUM_CLASSES
        fold_id = args.fold if args.fold is not None else 0
        tl, vl, el = czumhad_get_dataloader(
            data_root='/home/group/maestro_visual/data/CZU-MHAD',
            batch_size=bs, num_workers=args.num_workers, modalities=mods, split_id=fold_id,
            train_shuffle=True, max_seq_len=args.max_seq_len,
            time_compression_ratio=args.time_compression_ratio, use_batched_fusion=True,
            split_mode='cross_subject')
        in_len = args.max_seq_len if args.max_seq_len > 0 else 128
        M = len(mods)

        def unpack(batch, device):
            *feats, label = batch
            feats = [f.to(device).float() for f in feats]
            return _concat_iemocap(feats, M), label.squeeze(-1).long().to(device)
        return tl, vl, el, cfg, in_len, 'float', bs, unpack

    if ds == 'mmfi':
        from data.MMFi.get_data import (get_dataloader as mmfi_get_dataloader,
                                         MMFiDataset, FEATURE_DIMS, NUM_CLASSES)
        bs = args.batch_size or 16
        mods = list(MMFiDataset.ALL_MODALITIES)

        class _Cfg:
            pass
        cfg = _Cfg(); cfg.modalities = mods
        cfg.variates = {m: FEATURE_DIMS[m] for m in mods}; cfg.num_classes = NUM_CLASSES
        fold_id = args.fold if args.fold is not None else 0
        tl, vl, el = mmfi_get_dataloader(
            data_root='/home/group/maestro_visual/data/MM-Fi',
            batch_size=bs, num_workers=0, modalities=mods, split_id=fold_id,
            train_shuffle=True, time_compression_ratio=args.time_compression_ratio)
        in_len = args.max_seq_len if args.max_seq_len > 0 else 128
        M = len(mods)
        import torch.nn.functional as _F

        def unpack(batch, device):
            *feats, label = batch
            # MMFi's native sequence length (~297) exceeds the models' positional
            # capacity (input_length + 64) — resample each modality to in_len,
            # matching CH-SIMSv2, before concatenating along the feature dim.
            parts = []
            for j in range(M):
                f = feats[2 * j].to(device).float()  # (B, T_m, C_m) high feature
                if f.shape[1] != in_len:
                    f = _F.interpolate(f.permute(0, 2, 1), size=in_len,
                                       mode='linear', align_corners=False).permute(0, 2, 1)
                parts.append(f)
            return torch.cat(parts, dim=-1), label.squeeze(-1).long().to(device)
        return tl, vl, el, cfg, in_len, 'float', bs, unpack

    if ds == 'utd_mhad':
        from data.UTD_MHAD.get_data import (get_dataloader as utdmhad_get_dataloader,
                                              UTDMHADDataset, FEATURE_DIMS, NUM_CLASSES)
        bs = args.batch_size or 32
        mods = list(UTDMHADDataset.ALL_MODALITIES)

        class _Cfg:
            pass
        cfg = _Cfg(); cfg.modalities = mods
        cfg.variates = {m: FEATURE_DIMS[m] for m in mods}; cfg.num_classes = NUM_CLASSES
        fold_id = args.fold if args.fold is not None else 0
        tl, vl, el = utdmhad_get_dataloader(
            data_root='/home/group/maestro_visual/data/UTD-MHAD',
            batch_size=bs, num_workers=args.num_workers, modalities=mods, split_id=fold_id,
            train_shuffle=True, max_seq_len=args.max_seq_len,
            time_compression_ratio=args.time_compression_ratio, use_batched_fusion=True,
            split_mode='cross_subject')
        in_len = args.max_seq_len if args.max_seq_len > 0 else 128
        M = len(mods)

        def unpack(batch, device):
            *feats, label = batch
            feats = [f.to(device).float() for f in feats]
            return _concat_iemocap(feats, M), label.squeeze(-1).long().to(device)
        return tl, vl, el, cfg, in_len, 'float', bs, unpack

    if ds == 'ch_simsv2':
        import sys as _sys
        import torch.nn.functional as _F
        _p = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '..', '..', 'MultiBench', 'datasets'))
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        from ch_simsv2.get_data import (get_dataloader as chsims_get_dataloader,
                                         FEATURE_DIMS, NUM_CLASSES)
        bs = args.batch_size or 32
        mods = ['audio', 'vision', 'text']

        class _Cfg:
            pass
        cfg = _Cfg(); cfg.modalities = mods
        cfg.variates = {m: FEATURE_DIMS[m] for m in mods}; cfg.num_classes = NUM_CLASSES
        tl, vl, el = chsims_get_dataloader(
            pkl_path='/home/group/maestro_visual/data/CH-SIMSv2/Processed/unaligned.pkl',
            batch_size=bs, num_workers=args.num_workers, task='classification',
            time_compression_ratio=args.time_compression_ratio,
            modalities=mods, train_shuffle=True)
        in_len = args.max_seq_len if args.max_seq_len > 0 else 50
        M = len(mods)

        def unpack(batch, device):
            *feats, label = batch
            # CH-SIMSv2 modalities have different T — interpolate all to in_len
            parts = []
            for j in range(M):
                f = feats[2 * j].to(device).float()  # (B, T_m, C_m)
                if f.shape[1] != in_len:
                    f = _F.interpolate(f.permute(0, 2, 1), size=in_len,
                                       mode='linear', align_corners=False).permute(0, 2, 1)
                parts.append(f)
            return torch.cat(parts, dim=-1), label.squeeze(-1).long().to(device)
        return tl, vl, el, cfg, in_len, 'float', bs, unpack

    # ---- sax HAR datasets ----
    bs = args.batch_size or 64

    def sax_unpack(batch, device):
        x, y = batch
        return x.to(device).float(), y.to(device)

    if ds == 'daliahar':
        from utils.dataset_cfg import DaliaHAR
        from data.dataset_builder import HARDataset
        root_dir = '/home/group/maestro_visual/data/processed_dalia_activity'
        cfg = DaliaHAR(); transform = 'sax'
        in_len = (cfg.duration * cfg.base_sample_rate) // args.word_length
        sax_params = {'alphabet_size': 20, 'word_length': args.word_length}
        tr, ev = (cfg.folds[args.fold]['train_set'], cfg.folds[args.fold]['eval_set']) \
            if args.fold is not None else (cfg.train_set, cfg.eval_set)
        va = cfg.val_set
        tds = HARDataset(root_dir, cfg.modalities, tr, cfg, transform, sax_params=sax_params)
        vds = HARDataset(root_dir, cfg.modalities, va, cfg, transform, sax_params=sax_params)
        eds = HARDataset(root_dir, cfg.modalities, ev, cfg, transform, sax_params=sax_params)

    elif ds == 'pamap2':
        from utils.dataset_cfg import PAMAP2
        from data.pamap2_crosssubject import load_pamap2_subjects
        cfg = PAMAP2(); transform = 'sax_noisy'
        in_len = (cfg.duration * cfg.base_sample_rate) // args.word_length
        sax_params = {'alphabet_size': 20, 'word_length': args.word_length}
        tr, ev = (cfg.folds[args.fold]['train_set'], cfg.folds[args.fold]['eval_set']) \
            if args.fold is not None else (cfg.train_set, cfg.eval_set)
        va = cfg.val_set
        tds = load_pamap2_subjects(cfg.cross_subject_dir, cfg.modalities, tr, cfg, transform, sax_params=sax_params)
        vds = load_pamap2_subjects(cfg.cross_subject_dir, cfg.modalities, va, cfg, transform, sax_params=sax_params)
        eds = load_pamap2_subjects(cfg.cross_subject_dir, cfg.modalities, ev, cfg, transform, sax_params=sax_params)

    elif ds == 'dsads':
        from utils.dataset_cfg import DSADS
        from data.dataset_builder import DSADSDataset
        from sklearn.model_selection import train_test_split
        cfg = DSADS(); transform = 'sax'
        import math as _math
        in_len = _math.ceil((cfg.duration * cfg.base_sample_rate) / args.word_length)
        sax_params = {'alphabet_size': 20, 'word_length': args.word_length}
        sc = args.fold if args.fold is not None else 0
        ROOT = '/home/group/maestro_visual/data/DSADS/'
        _src = torch.load(os.path.join(ROOT, f'dsad_diversify_dict_scenario{sc}_src.pt'), weights_only=False)
        _trg = torch.load(os.path.join(ROOT, f'dsad_diversify_dict_scenario{sc}_trg.pt'), weights_only=False)
        _trx, _vax, _try, _vay = train_test_split(_src['samples'], _src['labels'], test_size=0.2,
                                                  random_state=2711, stratify=_src['labels'])
        tds = DSADSDataset({'samples': _trx, 'labels': _try}, cfg.modalities, cfg, transform, sax_params=sax_params)
        vds = DSADSDataset({'samples': _vax, 'labels': _vay}, cfg.modalities, cfg, transform, sax_params=sax_params)
        eds = DSADSDataset(_trg, cfg.modalities, cfg, transform, sax_params=sax_params)
    else:
        raise ValueError(ds)

    print(f'Modalities: {cfg.modalities}  input_length={in_len}  transform={transform}')
    tl = DataLoader(tds, batch_size=bs, shuffle=True, num_workers=args.num_workers, drop_last=True)
    vl = DataLoader(vds, batch_size=bs, shuffle=False, num_workers=args.num_workers, drop_last=False)
    el = DataLoader(eds, batch_size=bs, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return tl, vl, el, cfg, in_len, 'sax', bs, sax_unpack
