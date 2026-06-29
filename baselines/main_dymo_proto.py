"""Faithful DyMo (Du et al., ICLR 2026) prototype baseline on our CrossModalTransformer
fusion backbone (3-enc + 3-fusion), complete-data EFFICIENCY setting.

This is the FAITHFUL DyMo training+selection pipeline:
  * mask-augmented training: each batch samples ``K`` random present-subsets and
    runs the official multi-mask forward (forward_train -> logits + L2-normalised
    projection feature);
  * official CrossEntropy + PrototypeLoss (cosine), prototype loss switched on
    after ``--pt_epoch`` with weight ``--pt_rate``;
  * per-epoch prototype recompute as the class-mean of projected cls features
    (running sums accumulated across all K masks of all batches);
  * AFTER training: per-subset Gaussian estimation on the TRAIN loader, then the
    official greedy per-sample reward selection at test time.

The selection MECHANISM is imported verbatim from
``multimodal_model/dymo_proto.py`` (CrossModalDyMoProto, estimate_subset_gaussian,
proto_greedy_select) and the loss from
``multimodal_model/dymo_official/prototype_loss.py`` (PrototypeLoss).

Dispatch on a single ``--dataset {daliahar,pamap2,wesad,iemocap}`` arg.

Run:
    python main_dymo_proto.py --dataset daliahar --fold 0 --cuda_pick cuda:0
    python main_dymo_proto.py --dataset iemocap  --fold 0 --cuda_pick cuda:0 --split_mode cross_subject
    python main_dymo_proto.py --dataset daliahar --aggregate
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _V6_ROOT)

from utils.helper_function import set_seed, count_model_parameters, AverageMeter, ProgressMeter
from utils.train_utils import FocalLoss

from multimodal_model.crossmodal_transformer import CrossModalTransformer, _concat_iemocap
from multimodal_model.dymo_proto import (
    CrossModalDyMoProto, estimate_subset_gaussian, proto_greedy_select)
from multimodal_model.dymo_official.prototype_loss import PrototypeLoss


# ============================ IEMOCAP cfg adaptor ============================

def _iemocap_cfg(modalities):
    from data.IEMOCAP.get_data import FEATURE_DIMS, NUM_CLASSES, IEMOCAPDataset
    if modalities is None:
        modalities = list(IEMOCAPDataset.ALL_MODALITIES)

    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.modalities = list(modalities)
    cfg.variates = {m: FEATURE_DIMS[m] for m in cfg.modalities}
    cfg.num_classes = NUM_CLASSES
    return cfg


# ============================ CMU-MOSEI cfg adaptor ============================

def _mosei_cfg(modalities, task='classification2'):
    from data.CMU_MOSEI.get_data import (
        FEATURE_DIMS, get_num_classes, CMUMOSEIDataset)
    if modalities is None:
        modalities = list(CMUMOSEIDataset.ALL_MODALITIES)

    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.modalities = list(modalities)
    cfg.variates = {m: FEATURE_DIMS[m] for m in cfg.modalities}
    cfg.num_classes = get_num_classes(task)
    return cfg


# ============================ Train / eval ============================

def train_epoch(loader, proto, criterion, criterion_pt, optimizer, epoch, device,
                unpack, K, pt_epoch, pt_rate, clip_grad):
    """Official DyMo train_epoch (mask-augmentation + per-epoch prototype recompute)."""
    proto.train()
    C, D = proto.num_classes, proto.proj_dim
    id2subset = proto.id2subset
    n_sub = len(id2subset)

    # frozen prototypes for this epoch's PrototypeLoss; running sums for recompute
    prototypes = proto.prototypes.clone().detach()
    prototypes_sum = torch.zeros(C, D, device=device)
    prototypes_count_sum = torch.zeros(C, 1, device=device)

    loss_meter = AverageMeter('Loss', ':.4f')
    ce_meter = AverageMeter('CE', ':.4f')
    pt_meter = AverageMeter('PT', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')

    for i, batch in enumerate(loader):
        x, y = unpack(batch, device)
        B = y.size(0)
        sample_idx = torch.randperm(n_sub)[:K]
        loss_ce_list, loss_pt_list = [], []
        y_hat_list = []
        for sid in sample_idx.tolist():
            mask = torch.tensor(id2subset[int(sid)], dtype=torch.bool, device=device)
            mask = mask.unsqueeze(0).expand(B, -1)
            logits, feat = proto.forward_train(x, mask)
            loss_ce_list.append(criterion(logits, y))
            label = F.one_hot(y, num_classes=C).float()
            loss_pt_list.append(criterion_pt(label, prototypes, feat))
            with torch.no_grad():
                class_sum, class_count = proto.cal_prototypes(label.detach(), feat.detach())
                prototypes_sum += class_sum
                prototypes_count_sum += class_count
                y_hat_list.append(logits.detach())

        loss_ce = sum(loss_ce_list) / len(loss_ce_list)
        loss_pt = sum(loss_pt_list) / len(loss_pt_list)
        loss = loss_ce + (pt_rate * loss_pt if epoch >= pt_epoch else 0.0)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(proto.parameters(), clip_grad)
        optimizer.step()

        with torch.no_grad():
            y_hat = torch.cat(y_hat_list, dim=0)
            y_rep = y.repeat(len(y_hat_list))
            pred = y_hat.argmax(1)
            acc_meter.update(pred.eq(y_rep).sum().item() / y_rep.size(0), y_rep.size(0))
        loss_meter.update(loss.item(), B)
        ce_meter.update(loss_ce.item(), B)
        pt_meter.update(loss_pt.item(), B)

        if (i % 50 == 0) or (i == len(loader) - 1):
            ProgressMeter(len(loader), [loss_meter, ce_meter, pt_meter, acc_meter],
                          prefix=f"Epoch: [{epoch}]").display(i)

    # per-epoch prototype recompute (class mean of projected cls features)
    proto.prototypes.data.copy_(
        prototypes_sum / prototypes_count_sum.clamp(min=1))
    print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  CE: {ce_meter.avg:.4f}  '
          f'PT: {pt_meter.avg:.4f}  TrainAcc: {acc_meter.avg:.4f}  '
          f'pt_on: {epoch >= pt_epoch}')
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_full(loader, proto, criterion, device, unpack):
    """Full-modality (all-present) eval -> (loss, acc, labels, preds)."""
    proto.eval()
    M = proto.M
    loss_meter, acc_meter = AverageMeter('Loss', ':.4f'), AverageMeter('Acc', ':.4f')
    all_preds, all_labels = [], []
    for batch in loader:
        x, y = unpack(batch, device)
        B = y.size(0)
        mask = torch.ones(B, M, dtype=torch.bool, device=device)
        logits, _ = proto.forward_train(x, mask)
        loss = criterion(logits, y)
        pred = logits.argmax(1)
        acc_meter.update(pred.eq(y).sum().item() / B, B)
        loss_meter.update(loss.item(), B)
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())
    return (loss_meter.avg, acc_meter.avg,
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())


@torch.no_grad()
def evaluate_select(loader, proto, sg, device, unpack):
    """DyMo greedy per-sample selection -> (acc, labels, preds, avg_selected)."""
    proto.eval()
    acc_meter = AverageMeter('Acc', ':.4f')
    all_preds, all_labels = [], []
    n_sel_sum, n_sel_cnt = 0.0, 0
    for batch in loader:
        x, y = unpack(batch, device)
        B = y.size(0)
        logits, keep = proto_greedy_select(proto, x, sg, device)
        pred = logits.argmax(1)
        acc_meter.update(pred.eq(y).sum().item() / B, B)
        n_sel_sum += keep.float().sum(dim=1).sum().item()
        n_sel_cnt += B
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())
    return (acc_meter.avg, torch.cat(all_labels).numpy(),
            torch.cat(all_preds).numpy(), n_sel_sum / max(n_sel_cnt, 1))


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
    description='Faithful DyMo prototype baseline (CrossModalTransformer 3-enc+3-fusion).')

parser.add_argument('--dataset', required=True,
                    choices=['daliahar', 'pamap2', 'wesad', 'iemocap', 'dsads', 'mosei'])
parser.add_argument('--fold', default=None, type=int, choices=[0, 1, 2, 3])
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=None, type=int,
                    help='Default per-dataset (sax=64, iemocap=32).')
parser.add_argument('--K', default=4, type=int,
                    help='Mask-augmentation count (# random subsets sampled per batch).')
parser.add_argument('--pt_epoch', default=5, type=int,
                    help='Start prototype loss after this epoch.')
parser.add_argument('--pt_rate', default=1.0, type=float, help='Prototype loss weight.')
parser.add_argument('--projection_dim', default=16, type=int)
parser.add_argument('--proto_temperature', default=0.1, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--clip_grad', default=1.0, type=float)
parser.add_argument('--seed_num', default=239, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--exp_name', default='dymo_proto', type=str)
parser.add_argument('--results_dir', default='./results_dymo_proto/', type=str)
parser.add_argument('--aggregate', action='store_true', default=False)

parser.add_argument('--loss', default='ce', choices=['ce', 'focal'])
parser.add_argument('--focal_gamma', default=2.0, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)

# HAR (SAX) knobs -- SAX granularity for daliahar:
# word_length=2 -> T=128 (default), word_length=1 -> T=256.
parser.add_argument('--word_length', default=2, type=int)

# IEMOCAP knobs
parser.add_argument('--split_mode', default='cross_subject', type=str,
                    choices=['random', 'cross_subject'])
parser.add_argument('--data_root', default='/files1/haodong/data/IEMOCAP', type=str)
parser.add_argument('--max_seq_len', default=128, type=int)
parser.add_argument('--time_compression_ratio', default=4, type=int)
parser.add_argument('--available_sessions', nargs='+', type=int, default=None)

# CMU-MOSEI knobs
parser.add_argument('--mosei_task', default=2, type=int, choices=[2, 5, 7],
                    help='CMU-MOSEI granularity: 2=Acc-2, 5=Acc-5, 7=Acc-7.')

# smoke-test hook (synthetic tiny loaders, no real data)
parser.add_argument('--smoke', action='store_true', default=False)

args = parser.parse_args()


# ============================ Setup ============================

num_epochs = args.num_epochs
device = torch.device(args.cuda_pick if torch.cuda.is_available() else "cpu")
current_date = datetime.now().strftime('%Y-%m-%d')
exp_name_full = f"{current_date}_{args.dataset}_{args.exp_name}"
exp_dir = os.path.join(args.results_dir, exp_name_full)
os.makedirs(exp_dir, exist_ok=True)


# ============================ Aggregation mode ============================

if args.aggregate:
    fold_results = []
    n_folds = 4 if args.dataset == 'dsads' else 3
    for fold_idx in range(n_folds):
        fpath = os.path.join(exp_dir, f"results_fold{fold_idx}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                fold_results.append(json.load(f))
    if not fold_results:
        print(f'No fold results found in {exp_dir}.')
        sys.exit(1)
    aggregated = {'experiment_name': exp_name_full, 'folds': fold_results}
    for key in ('best_test_acc', 'best_test_f1', 'select_test_acc',
                'select_test_f1', 'avg_selected'):
        vals = [fr[key] for fr in fold_results if key in fr]
        if vals:
            aggregated[f'{key}_mean'] = float(np.mean(vals))
            aggregated[f'{key}_std'] = float(np.std(vals))
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump(aggregated, f, indent=4)
    print(f"Aggregated -> {exp_dir}/results.json")
    for key in ('best_test_acc', 'best_test_f1', 'select_test_acc',
                'select_test_f1', 'avg_selected'):
        if f'{key}_mean' in aggregated:
            print(f"  {key}: {aggregated[f'{key}_mean']:.4f} +/- "
                  f"{aggregated[f'{key}_std']:.4f}")
    sys.exit(0)


# ============================ Per-dataset data + cfg ============================

set_seed(args.seed_num)
print(f"Device: {device}")
print(f"Experiment: {exp_name_full}")
print(f"Dataset: {args.dataset.upper()}")

# Defaults filled in per dataset below.
batch_size = args.batch_size
input_length = None
dataset_cfg = None
input_mode = 'sax'
d_model = 64
train_subjects = val_subjects = eval_subjects = None


def _build_loaders():
    """Returns (train_loader, val_loader, eval_loader, cfg, input_length, input_mode,
    d_model, batch_size, unpack)."""
    global train_subjects, val_subjects, eval_subjects
    ds = args.dataset

    if ds == 'iemocap':
        from data.IEMOCAP.get_data import get_dataloader as iemocap_get_dataloader
        bs = args.batch_size or 32
        cfg = _iemocap_cfg(None)
        modalities = cfg.modalities
        fold_id = args.fold if args.fold is not None else 0
        print(f"IEMOCAP split id: {fold_id} (fold={args.fold})  "
              f"split_mode={args.split_mode}  modalities={modalities}")
        tl, vl, el = iemocap_get_dataloader(
            data_root=args.data_root, batch_size=bs, num_workers=args.num_workers,
            modalities=modalities, split_id=fold_id, train_shuffle=True,
            max_seq_len=args.max_seq_len,
            time_compression_ratio=args.time_compression_ratio,
            use_batched_fusion=True, available_sessions=args.available_sessions,
            split_mode=args.split_mode)
        in_len = args.max_seq_len if args.max_seq_len > 0 else 600
        M = len(modalities)

        def unpack(batch, device):
            *feats, label = batch
            feats = [f.to(device).float() for f in feats]
            x = _concat_iemocap(feats, M)               # [B,T,sum(V)]
            return x, label.squeeze(-1).long().to(device)

        return tl, vl, el, cfg, in_len, 'float', 128, bs, unpack

    if ds == 'mosei':
        from data.CMU_MOSEI.get_data import get_dataloader as mosei_get_dataloader
        bs = args.batch_size or 32
        task = {2: 'classification2', 5: 'classification5',
                7: 'classification7'}[args.mosei_task]
        cfg = _mosei_cfg(None, task)
        modalities = cfg.modalities
        ml = args.max_seq_len if args.max_seq_len > 0 else 50
        print(f"CMU-MOSEI 3-seed run (seed={args.seed_num})  task={task}  "
              f"num_classes={cfg.num_classes}  modalities={modalities}")
        # split_id=0 (canonical, already speaker/video-disjoint); variance via seeds.
        tl, vl, el = mosei_get_dataloader(
            batch_size=bs, num_workers=args.num_workers,
            modalities=modalities, split_id=0, train_shuffle=True,
            max_seq_len=ml, time_compression_ratio=args.time_compression_ratio,
            use_batched_fusion=True, task=task)
        in_len = ml
        M = len(modalities)

        def unpack(batch, device):
            *feats, label = batch
            feats = [f.to(device).float() for f in feats]
            x = _concat_iemocap(feats, M)               # [B,T,sum(V)]
            return x, label.squeeze(-1).long().to(device)

        return tl, vl, el, cfg, in_len, 'float', 128, bs, unpack

    # ---- sax HAR datasets ----
    bs = args.batch_size or 64

    def sax_unpack(batch, device):
        x, y = batch
        return x.to(device).float(), y.to(device)

    if ds == 'daliahar':
        from utils.dataset_cfg import DaliaHAR
        from data.dataset_builder import HARDataset
        root_dir = '/files1/haodong/data/processed_dalia_activity'
        cfg = DaliaHAR()
        transform = 'sax'
        # word_length=2 -> 256//2=128 (default), word_length=1 -> 256.
        in_len = (cfg.duration * cfg.base_sample_rate) // args.word_length
        sax_params = {'alphabet_size': 20, 'word_length': args.word_length}
        if args.fold is not None:
            fc = cfg.folds[args.fold]
            tr, ev = fc['train_set'], fc['eval_set']
        else:
            tr, ev = cfg.train_set, cfg.eval_set
        va = cfg.val_set
        train_subjects, val_subjects, eval_subjects = tr, va, ev
        tds = HARDataset(root_dir, cfg.modalities, tr, cfg, transform, sax_params=sax_params)
        vds = HARDataset(root_dir, cfg.modalities, va, cfg, transform, sax_params=sax_params)
        eds = HARDataset(root_dir, cfg.modalities, ev, cfg, transform, sax_params=sax_params)

    elif ds == 'wesad':
        from utils.dataset_cfg import WESAD
        from data.dataset_builder import HARDataset
        root_dir = '/files1/haodong/data/WESAD/processed_wesad_activity'
        cfg = WESAD()
        transform = 'sax'
        # word_length=2 -> 256//2=128 (default), word_length=1 -> 256.
        in_len = (cfg.duration * cfg.base_sample_rate) // args.word_length
        sax_params = {'alphabet_size': 20, 'word_length': args.word_length}
        if args.fold is not None:
            fc = cfg.folds[args.fold]
            tr, ev = fc['train_set'], fc['eval_set']
        else:
            tr, ev = cfg.train_set, cfg.eval_set
        va = cfg.val_set
        train_subjects, val_subjects, eval_subjects = tr, va, ev
        tds = HARDataset(root_dir, cfg.modalities, tr, cfg, transform, sax_params=sax_params)
        vds = HARDataset(root_dir, cfg.modalities, va, cfg, transform, sax_params=sax_params)
        eds = HARDataset(root_dir, cfg.modalities, ev, cfg, transform, sax_params=sax_params)

    elif ds == 'pamap2':
        from utils.dataset_cfg import PAMAP2
        from data.pamap2_crosssubject import load_pamap2_subjects
        cfg = PAMAP2()
        transform = 'sax_noisy'
        in_len = cfg.duration * cfg.base_sample_rate         # 512
        if args.fold is not None:
            fc = cfg.folds[args.fold]
            tr, ev = fc['train_set'], fc['eval_set']
        else:
            tr, ev = cfg.train_set, cfg.eval_set
        va = cfg.val_set
        train_subjects, val_subjects, eval_subjects = tr, va, ev
        tds = load_pamap2_subjects(cfg.cross_subject_dir, cfg.modalities, tr, cfg, transform)
        vds = load_pamap2_subjects(cfg.cross_subject_dir, cfg.modalities, va, cfg, transform)
        eds = load_pamap2_subjects(cfg.cross_subject_dir, cfg.modalities, ev, cfg, transform)

    elif ds == 'dsads':
        # DSADS diversify scenarios (fold = scenario_idx 0..3), clean sax, T=125
        # (word_length=1). Same data pipeline as the clean_models DSADS mains:
        # src split 80/20 (stratified, seed 2711) for train/val; trg = test.
        from utils.dataset_cfg import DSADS
        from data.dataset_builder import DSADSDataset
        from sklearn.model_selection import train_test_split
        cfg = DSADS()
        transform = 'sax'
        in_len = (cfg.duration * cfg.base_sample_rate) // args.word_length   # 125
        sax_params = {'alphabet_size': 20, 'word_length': args.word_length}
        scenario_idx = args.fold if args.fold is not None else 0
        DATA_ROOT = '/files1/haodong/data/DSADS/'
        print(f'DSADS scenario {scenario_idx} (fold={args.fold})')
        _src = torch.load(os.path.join(DATA_ROOT, f'dsad_diversify_dict_scenario{scenario_idx}_src.pt'),
                          weights_only=False)
        _trg = torch.load(os.path.join(DATA_ROOT, f'dsad_diversify_dict_scenario{scenario_idx}_trg.pt'),
                          weights_only=False)
        _trx, _vax, _try, _vay = train_test_split(
            _src['samples'], _src['labels'], test_size=0.2,
            random_state=2711, stratify=_src['labels'])
        train_subjects = f'scenario{scenario_idx}_src_train'
        val_subjects = f'scenario{scenario_idx}_src_val'
        eval_subjects = f'scenario{scenario_idx}_trg'
        tds = DSADSDataset({'samples': _trx, 'labels': _try}, cfg.modalities, cfg, transform, sax_params=sax_params)
        vds = DSADSDataset({'samples': _vax, 'labels': _vay}, cfg.modalities, cfg, transform, sax_params=sax_params)
        eds = DSADSDataset(_trg, cfg.modalities, cfg, transform, sax_params=sax_params)
    else:
        raise ValueError(args.dataset)

    print(f'Modalities: {cfg.modalities}  input_length={in_len}  transform={transform}')
    tl = DataLoader(tds, batch_size=bs, shuffle=True, num_workers=args.num_workers, drop_last=True)
    vl = DataLoader(vds, batch_size=bs, shuffle=True, num_workers=args.num_workers, drop_last=True)
    el = DataLoader(eds, batch_size=bs, shuffle=False, num_workers=args.num_workers, drop_last=True)
    return tl, vl, el, cfg, in_len, 'sax', 64, bs, sax_unpack


# ---- synthetic tiny loaders for the CPU smoke-test ----

def _build_smoke_loaders():
    """Tiny synthetic dataset mirroring the real cfg (M, variates, num_classes)."""
    ds = args.dataset
    if ds == 'iemocap':
        cfg = _iemocap_cfg(None)
        in_len, in_mode, d_m, bs = args.max_seq_len, 'float', 128, 8
        M = len(cfg.modalities)
        n = 16

        def make():
            feats = []
            for m in cfg.modalities:
                v = cfg.variates[m]
                feats.append(torch.randn(n, in_len, v))           # high
                feats.append(torch.randn(n, in_len, v))           # low (unused)
            label = torch.randint(0, cfg.num_classes, (n, 1))
            return list(zip(*([f for f in feats] + [label])))

        # build per-sample tuples
        feats_per_mod = []
        for m in cfg.modalities:
            v = cfg.variates[m]
            feats_per_mod.append(torch.randn(n, in_len, v))
            feats_per_mod.append(torch.randn(n, in_len, v))
        label = torch.randint(0, cfg.num_classes, (n, 1))
        samples = [tuple([f[i] for f in feats_per_mod] + [label[i]]) for i in range(n)]
        loader = DataLoader(samples, batch_size=bs, shuffle=True)
        loader_eval = DataLoader(samples, batch_size=bs, shuffle=False)

        def unpack(batch, device):
            *feats, lab = batch
            feats = [f.to(device).float() for f in feats]
            x = _concat_iemocap(feats, M)
            return x, lab.squeeze(-1).long().to(device)

        return loader, loader, loader_eval, cfg, in_len, in_mode, d_m, bs, unpack

    # sax HAR
    from utils.dataset_cfg import DaliaHAR, WESAD, PAMAP2
    cfg = {'daliahar': DaliaHAR, 'wesad': WESAD, 'pamap2': PAMAP2}[ds]()
    in_len = 128 if ds != 'pamap2' else cfg.duration * cfg.base_sample_rate
    sumV = sum(cfg.variates[m] for m in cfg.modalities)
    n, bs = 16, 8
    x = torch.randint(0, 20, (n, in_len, sumV))          # sax tokens (long-castable)
    y = torch.randint(0, cfg.num_classes, (n,))
    samples = [(x[i], y[i]) for i in range(n)]
    loader = DataLoader(samples, batch_size=bs, shuffle=True)
    loader_eval = DataLoader(samples, batch_size=bs, shuffle=False)

    def unpack(batch, device):
        xx, yy = batch
        return xx.to(device).float(), yy.to(device)

    return loader, loader, loader_eval, cfg, in_len, 'sax', 64, bs, unpack


if args.smoke:
    (train_loader, val_loader, eval_loader, dataset_cfg, input_length,
     input_mode, d_model, batch_size, unpack) = _build_smoke_loaders()
else:
    (train_loader, val_loader, eval_loader, dataset_cfg, input_length,
     input_mode, d_model, batch_size, unpack) = _build_loaders()

modalities = list(dataset_cfg.modalities)
num_classes = dataset_cfg.num_classes


# ============================ Model ============================

# Cross-modal Transformer (3 per-modality enc layers + 3 fusion layers).
backbone_kwargs = dict(
    cfg=dataset_cfg, input_length=input_length, input_mode=input_mode,
    d_model=d_model, nhead=8, num_enc_layers=3, num_fusion_layers=3, dropout=0.1)
if input_mode == 'sax':
    backbone_kwargs.update(alphabet_size=20, token_dim=16)
inner = CrossModalTransformer(**backbone_kwargs)

proto = CrossModalDyMoProto(
    inner, num_classes=num_classes, projection_dim=args.projection_dim,
    proto_temperature=args.proto_temperature).to(device).float()
print(f'DyMo-proto: M={proto.M}  num_subsets={len(proto.id2subset)}  '
      f'proj_dim={proto.proj_dim}  C={num_classes}')

optimizer = optim.AdamW(proto.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = (FocalLoss(gamma=args.focal_gamma)
             if args.loss == 'focal'
             else nn.CrossEntropyLoss(label_smoothing=args.label_smoothing))
criterion_pt = PrototypeLoss(temperature=args.proto_temperature,
                             metric_name='cosine_similarity').to(device)

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

print(f'Parameters: {count_model_parameters(proto)}')


# ============================ Training loop ============================

best_val_acc, best_val_f1, best_epoch_val = 0.0, 0.0, -1
best_model_state = None

writer = SummaryWriter(
    comment=f"_{exp_name_full}_fold{args.fold}"
            if args.fold is not None else f"_{exp_name_full}")

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(
        train_loader, proto, criterion, criterion_pt, optimizer, epoch, device,
        unpack, K=args.K, pt_epoch=args.pt_epoch, pt_rate=args.pt_rate,
        clip_grad=args.clip_grad)

    val_loss, val_acc, val_y, val_p = evaluate_full(
        val_loader, proto, criterion, device, unpack)
    val_f1 = f1_score(val_y, val_p, average='macro')

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/acc', train_acc, epoch)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/acc', val_acc, epoch)
    writer.add_scalar('val/f1', val_f1, epoch)
    writer.add_scalar('schedule/lr', current_lr, epoch)
    print(f'Epoch {epoch} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')

    if val_acc > best_val_acc:
        best_val_acc, best_val_f1, best_epoch_val = val_acc, val_f1, epoch
        best_model_state = {k: v.detach().cpu().clone()
                            for k, v in proto.state_dict().items()}


# ============================ Model selection + test ============================

print(f"\nBest : epoch {best_epoch_val}, val_acc={best_val_acc:.4f}, val_f1={best_val_f1:.4f}")
if best_model_state is not None:
    proto.load_state_dict(best_model_state)
proto.eval()
ckpt_name = (f"best_model_fold{args.fold}.pth"
             if args.fold is not None else "best_val_model.pth")
torch.save(proto.state_dict(), os.path.join(exp_dir, ckpt_name))

# 1) per-subset Gaussian on the TRAIN loader (uses the loaded-best prototypes)
print("\nEstimating per-subset Gaussians on the TRAIN loader ...")
sg = estimate_subset_gaussian(proto, train_loader, device, unpack)
sg_name = (f"subset_gaussian_fold{args.fold}.pt"
           if args.fold is not None else "subset_gaussian.pt")
torch.save(sg, os.path.join(exp_dir, sg_name))

print("\n" + "=" * 80)
print("Test-set evaluation (faithful DyMo prototype)")
print("=" * 80)

# 2a) full-modality test
_, test_acc, t_y, t_p = evaluate_full(eval_loader, proto, criterion, device, unpack)
test_f1 = f1_score(t_y, t_p, average='macro')
print(f"  FULL  : test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")

# 2b) DyMo greedy selection
sel_acc, s_y, s_p, avg_selected = evaluate_select(eval_loader, proto, sg, device, unpack)
sel_f1 = f1_score(s_y, s_p, average='macro')
print(f"  DYMO  : select_acc={sel_acc:.4f}  select_f1={sel_f1:.4f}  "
      f"avg_selected={avg_selected:.2f}/{proto.M}")

writer.add_scalar('test/acc', test_acc, 0)
writer.add_scalar('test/f1', test_f1, 0)
writer.add_scalar('test/select_acc', sel_acc, 0)
writer.add_scalar('test/select_f1', sel_f1, 0)
writer.add_scalar('test/avg_selected', avg_selected, 0)
writer.close()


# ============================ Save ============================

config = {
    'experiment_name': exp_name_full, 'dataset': args.dataset,
    'method': 'dymo_proto', 'model_class': 'CrossModalDyMoProto',
    'backbone': 'CrossModalTransformer(3enc+3fusion)',
    'fold': args.fold,
    'num_epochs': num_epochs, 'batch_size': batch_size, 'seed': args.seed_num,
    'input_mode': input_mode, 'input_length': input_length, 'd_model': d_model,
    'nhead': 8, 'num_enc_layers': 3, 'num_fusion_layers': 3, 'dropout': 0.1,
    'K': args.K, 'pt_epoch': args.pt_epoch, 'pt_rate': args.pt_rate,
    'projection_dim': args.projection_dim, 'proto_temperature': args.proto_temperature,
    'lr': args.lr, 'weight_decay': args.weight_decay, 'clip_grad': args.clip_grad,
    'loss': args.loss, 'num_classes': num_classes, 'modalities': modalities,
    'split_mode': (args.split_mode if args.dataset == 'iemocap' else None),
    'train_subjects': train_subjects, 'val_subjects': val_subjects,
    'eval_subjects': eval_subjects,
}
with open(os.path.join(exp_dir, "config.json"), 'w') as f:
    json.dump(config, f, indent=4)

results = {
    'experiment_name': exp_name_full, 'fold': args.fold,
    'best_val_acc': float(best_val_acc), 'best_val_f1': float(best_val_f1),
    'best_epoch_val': int(best_epoch_val),
    'best_test_acc': float(test_acc), 'best_test_f1': float(test_f1),
    'select_test_acc': float(sel_acc), 'select_test_f1': float(sel_f1),
    'avg_selected': float(avg_selected),
}
results_filename = (f"results_fold{args.fold}.json"
                    if args.fold is not None else "results.json")
with open(os.path.join(exp_dir, results_filename), 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nBest Model (Epoch {best_epoch_val}): val_acc={best_val_acc:.4f}  "
      f"test_acc={test_acc:.4f}  select_acc={sel_acc:.4f}  avg_selected={avg_selected:.2f}")
print(f"Directory: {exp_dir}")
