"""ModalityDynMM (Xue & Marculescu, MULA/CVPR'23) on the seqA backbone — Step II.

3 expert seqA networks on nested modality subsets (1mod / partial / full, chosen by
val-LOO; see dynmm_branches.json) act as branches. A lightweight gate makes a
PER-SAMPLE choice of which branch to run, trained with task loss + a FLOP-cost
regulariser (lambda * mean_branch_weight . flop_vec) so easy samples route to the
cheap unimodal branch and hard samples to the expensive full branch. Faithful to
the paper's DiffSoftmax straight-through hard gate + frozen experts (Step II).

Eval routes each sample to its argmax branch (hard) and reports accuracy/F1 +
expected GFLOPs (branch-usage . per-branch GFLOPs). Slots into the baseline table.

Step I (train the 3 experts) = run_dynmm_experts_T256.sh. Run from clean_models/train/:
  python main_dynmm_seqA_gate.py --dataset daliahar --fold 0 --cuda_pick cuda:0 \
      --results_dir ../results_3p3/T256/dynmm_seqA --exp_name dynmm_seqA_T256
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_function import set_seed
from utils.dataset_cfg import DaliaHAR, WESAD
from data.dataset_builder import HARDataset
from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import (
    DualVideoBottleneckModelV6Downsample)

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['daliahar', 'wesad', 'pamap2', 'dsads'])
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--word_length', type=int, default=1)
ap.add_argument('--gate_epochs', type=int, default=40)
ap.add_argument('--reg', type=float, default=0.1, help='FLOP-cost reg weight (lambda)')
ap.add_argument('--batch_size', type=int, default=64)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='dynmm_seqA_T256')
ap.add_argument('--seed', type=int, default=239)
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)
BRANCHES = ['1mod', 'partial', 'full']

# per-dataset config. T = model input_length (pamap2 loads at 512 then interpolates
# to 128 inside the model). loader_kind picks HARDataset vs load_pamap2_subjects.
if args.dataset == 'daliahar':
    ROOT = '/files1/haodong/data/processed_dalia_activity'; cfg = DaliaHAR()
    RESROOT = '../results_3p3/T256'; T = (cfg.duration * cfg.base_sample_rate) // args.word_length
    LKIND, TRANSFORM = 'har', 'sax'
elif args.dataset == 'wesad':
    ROOT = '/files1/haodong/data/WESAD/processed_wesad_activity'; cfg = WESAD()
    RESROOT = '../results_3p3/T256_wesad'; T = (cfg.duration * cfg.base_sample_rate) // args.word_length
    LKIND, TRANSFORM = 'har', 'sax'
elif args.dataset == 'dsads':
    from utils.dataset_cfg import DSADS
    cfg = DSADS()
    RESROOT = '../results_3p3/dynmm_dsads'; T = (cfg.duration * cfg.base_sample_rate) // args.word_length  # 125
    LKIND, TRANSFORM = 'dsads', 'sax'
else:  # pamap2
    from utils.dataset_cfg import PAMAP2
    from data.pamap2_crosssubject import load_pamap2_subjects
    cfg = PAMAP2(); ROOT = cfg.cross_subject_dir
    RESROOT = '../results_3p3/dynmm_pamap2'; T = 512            # T=512 sax_noisy (match seqA@512, no interp)
    LKIND, TRANSFORM = 'pamap2', 'sax_noisy'
ALL_MODS = list(cfg.modalities); NUMC = cfg.num_classes
full_cm, off = {}, 0
for m in ALL_MODS:
    full_cm[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
C_full = off
BR_MODS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dynmm_branches.json')))[args.dataset]
sax = {'alphabet_size': 20, 'word_length': args.word_length}

if LKIND == 'dsads':
    # DSADS diversify scenarios (fold = scenario): src split 80/20 (stratified, seed
    # 2711) for train/val; trg = test. Always load ALL modalities (experts subset).
    from data.dataset_builder import DSADSDataset
    from sklearn.model_selection import train_test_split
    DATA_ROOT = '/files1/haodong/data/DSADS/'
    _src = torch.load(os.path.join(DATA_ROOT, f'dsad_diversify_dict_scenario{args.fold}_src.pt'), weights_only=False)
    _trg = torch.load(os.path.join(DATA_ROOT, f'dsad_diversify_dict_scenario{args.fold}_trg.pt'), weights_only=False)
    _trx, _vax, _try, _vay = train_test_split(_src['samples'], _src['labels'], test_size=0.2,
                                              random_state=2711, stratify=_src['labels'])

    def _mk(d, shuffle, drop_last):
        ds = DSADSDataset(d, ALL_MODS, cfg, 'sax', sax_params=sax)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_last)

    train_loader = _mk({'samples': _trx, 'labels': _try}, True, True)
    val_loader = _mk({'samples': _vax, 'labels': _vay}, False, False)
    test_loader = _mk(_trg, False, False)
else:
    fold_cfg = cfg.folds[args.fold]
    train_subs, eval_subs, val_subs = fold_cfg['train_set'], fold_cfg['eval_set'], cfg.val_set

    def loader(subs, shuffle, drop_last):
        if LKIND == 'har':
            ds = HARDataset(ROOT, ALL_MODS, subs, cfg, 'sax', sax_params=sax)   # always load ALL modalities
        else:
            ds = load_pamap2_subjects(ROOT, ALL_MODS, subs, cfg, TRANSFORM)      # default sax (T=512 -> interp)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_last)

    train_loader = loader(train_subs, True, True)
    val_loader = loader(val_subs, False, False)
    test_loader = loader(eval_subs, False, False)


class _V6Cfg:
    def __init__(s, mods, variates):
        s.modalities = list(mods); s.variates = {m: variates[m] for m in mods}
        f = s.modalities[0]; s.video_high_dim = s.variates[f]; s.video_low_dim = s.variates[f]


def build_expert(mods, C):
    vc = _V6Cfg(mods, cfg.variates)
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=vc, output_dim=NUMC, input_length=T, d_model=C['d_model'], nhead=C['nhead'],
        num_layers_per_modal=C['num_layers_per_modal'], num_layers=C['num_layers'], dropout=C['dropout'],
        verbose=False, video_low_dim=vc.video_low_dim, video_high_dim=vc.video_high_dim, use_bottleneck=True,
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
        bottleneck_agg_mode='gate', n_fusion_distill=-1).to(dev).float()


# ---- load the 3 frozen experts ----
# IMPORTANT: the seqA expert's sequential fusion has order-tied position/order
# embeddings, so it MUST be built+fed in the SAME modality order it was trained in
# (config['modalities'], canonical), NOT the dynmm_branches.json order. The set of
# modalities is identical; only the order differs. Using the json order silently
# permutes the inputs and collapses the expert to ~random accuracy.
experts, expert_mods = [], []
for br in BRANCHES:
    rd = sorted(glob.glob(os.path.join(RESROOT, 'dynmm_experts', br, '*')))
    rd = [d for d in rd if os.path.isfile(os.path.join(d, f'best_model_fold{args.fold}.pth'))]
    assert rd, f'no trained expert for {args.dataset}/{br} fold{args.fold}'
    rundir = rd[0]
    C = json.load(open(os.path.join(rundir, 'config.json')))
    mods = C['modalities']                       # trained (canonical) order, not BR_MODS[br]
    e = build_expert(mods, C)
    sd = torch.load(os.path.join(rundir, f'best_model_fold{args.fold}.pth'), map_location='cpu', weights_only=True)
    e.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False)
    for p in e.parameters():
        p.requires_grad_(False)
    e.eval(); experts.append(e); expert_mods.append(mods)
    print(f'[expert] {br}: mods={mods} <- {rundir}')


def expert_fwd(e, mods, x):
    high = {m: x[:, :, full_cm[m][0]:full_cm[m][1]] for m in mods}
    out = e(high, dict(high), training=False, return_selection_info=False)
    return out[0] if isinstance(out, tuple) else out


# ---- per-branch GFLOPs (relative; for the FLOP reg + reporting) ----
from torch.utils.flop_counter import FlopCounterMode
flop_vec = []
xb0 = next(iter(val_loader))[0][:1].float().to(dev)
for e, mods in zip(experts, expert_mods):
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        expert_fwd(e, mods, xb0)
    flop_vec.append(fc.get_total_flops() / 1e9)
flop_vec = torch.tensor(flop_vec, device=dev, dtype=torch.float32)
flop_norm = flop_vec / flop_vec.max()
print(f'[flops] per-branch GFLOPs={[round(f,4) for f in flop_vec.tolist()]} (norm={[round(f,3) for f in flop_norm.tolist()]})')


# ---- gate (DynMM MLP over pooled input) ----
class Gate(nn.Module):
    def __init__(s, c_in, n):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(2 * c_in, 128), nn.ReLU(), nn.Linear(128, n))

    def forward(s, x):                                    # x [B,T,C]
        feat = torch.cat([x.mean(1), x.std(1)], dim=1)    # [B, 2C]
        return s.net(feat)


def diff_softmax(logits, tau=1.0, hard=True):
    y = (logits / tau).softmax(-1)
    if hard:
        idx = y.max(-1, keepdim=True)[1]
        yh = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        return yh - y.detach() + y
    return y


gate = Gate(C_full, len(BRANCHES)).to(dev)
opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-2)
for ep in range(args.gate_epochs):
    gate.train()
    for x, y in train_loader:
        x, y = x.float().to(dev), y.to(dev)
        with torch.no_grad():
            preds = torch.stack([expert_fwd(e, m, x) for e, m in zip(experts, expert_mods)], 1)  # [B,3,NUMC]
        w = diff_softmax(gate(x), tau=1.0, hard=True)                          # [B,3] straight-through
        out = (w.unsqueeze(-1) * preds).sum(1)                                 # [B,NUMC]
        loss = F.cross_entropy(out, y) + args.reg * (w.mean(0) * flop_norm).sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0); opt.step()
gate.eval()


# ---- eval: hard route each sample to its argmax branch ----
@torch.no_grad()
def evaluate(ld):
    preds_all, ys, choice = [], [], []
    for x, y in ld:
        x = x.float().to(dev)
        pr = torch.stack([expert_fwd(e, m, x) for e, m in zip(experts, expert_mods)], 1)  # [B,3,NUMC]
        b = gate(x).argmax(-1)                                                  # [B] hard branch
        out = pr[torch.arange(len(x)), b]
        preds_all.append(out.argmax(-1).cpu()); ys.append(y); choice.append(b.cpu())
    return (torch.cat(preds_all).numpy(), torch.cat(ys).numpy(), torch.cat(choice).numpy())


vp, vy, vc = evaluate(val_loader)
tp, ty, tc = evaluate(test_loader)
usage = np.bincount(tc, minlength=len(BRANCHES)) / len(tc)
mean_gflops = float((torch.tensor(usage, dtype=torch.float32) * flop_vec.cpu()).sum())
ta = float((tp == ty).mean()); tf = float(f1_score(ty, tp, average='macro'))
va = float((vp == vy).mean())

exp_dir = os.path.join(args.results_dir, args.exp_name); os.makedirs(exp_dir, exist_ok=True)
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'dynmm_seqA',
       'best_val_acc': va, 'best_val_f1': float(f1_score(vy, vp, average='macro')),
       'best_test_acc': ta, 'best_test_f1': tf,
       'dynmm': {'branches': BRANCHES, 'branch_mods': {b: BR_MODS[b] for b in BRANCHES},
                 'test_branch_usage': usage.tolist(), 'per_branch_gflops': flop_vec.tolist(),
                 'mean_gflops': mean_gflops, 'reg_lambda': args.reg,
                 'per_branch_test_acc': [float(((tp[tc == i] == ty[tc == i]).mean()) if (tc == i).any() else 0.0)
                                         for i in range(len(BRANCHES))]},
       'n_test': len(ty)}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
torch.save(gate.state_dict(), os.path.join(exp_dir, f'gate_fold{args.fold}.pth'))
print(f"[DynMM-seqA] {args.dataset} f{args.fold} test_acc={ta:.4f}/{tf:.4f} "
      f"usage(1mod/partial/full)={usage.round(2).tolist()} mean_GFLOPs={mean_gflops:.3f} "
      f"(full={flop_vec[-1]:.3f}) -> {exp_dir}")
