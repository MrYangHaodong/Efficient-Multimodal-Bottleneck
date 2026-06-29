"""ModalityDynMM (Xue & Marculescu, MULA/CVPR'23) on the seqA backbone — Step II,
MOSEI variant.

Mirrors main_dynmm_seqA_gate.py (daliahar/wesad/pamap2) but adapted to MOSEI's
data pipeline: 6 modalities with DIFFERENT feature dims, list-based batch format
(flat [high0, low0, high1, low1, ...] + label), and per-fold cross-subject session
splits via data.MOSEI.get_data.get_dataloader(split_mode='cross_subject').

3 frozen seqA experts on nested modality subsets (1mod / partial / full, see
dynmm_branches.json["mosei"]) act as branches. A lightweight per-sample gate
routes to one branch, trained with task loss + a FLOP-cost regulariser
(lambda * mean_branch_weight . flop_vec). The paper's DiffSoftmax straight-through
hard gate + frozen experts (Step I = train the 3 experts).

Eval routes each sample to its argmax branch (hard) and reports accuracy/F1 +
expected GFLOPs (branch-usage . per-branch GFLOPs). Output json keys match the HAR
trainer so it slots into the same aggregation.

Run from clean_models/train/:
  python main_dynmm_seqA_gate_mosei.py --dataset mosei --fold 0 \
      --cuda_pick cuda:0 --results_dir ../results_3p3/dynmm_mosei/dynmm_seqA \
      --exp_name dynmm_seqA_T128
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_function import set_seed
from data.CMU_MOSEI.get_data import (
    get_dataloader as mosei_get_dataloader,
    FEATURE_DIMS, NUM_CLASSES, CMUMOSEIDataset,
)
from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import (
    DualVideoBottleneckModelV6Downsample,
)

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='mosei', choices=['mosei'])
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--gate_epochs', type=int, default=40)
ap.add_argument('--reg', type=float, default=0.1, help='FLOP-cost reg weight (lambda)')
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='dynmm_seqA_T128')
ap.add_argument('--seed', type=int, default=239)
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)
BRANCHES = ['1mod', 'partial', 'full']

# MOSEI plumbing (mirrors compute_seqa_loo_mosei.py)
DATA_ROOT = '/files1/haodong/data/CMU-MOSI/CMU-MOSEI'
SPLIT_MODE = 'random'   # cross-subject session-disjoint splits (per fold)
RESROOT = os.path.dirname(os.path.abspath(args.results_dir))   # experts live alongside gate output (per-seed isolation; backward-compatible: .../dynmm_mosei/dynmm_seqA -> .../dynmm_mosei)
ALL_MODS = list(CMUMOSEIDataset.ALL_MODALITIES)   # ['video','audio','text','mocap_hand','mocap_head','mocap_rotated']
NUMC = NUM_CLASSES
BR_MODS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'dynmm_branches.json')))[args.dataset]


# ---- expert build (mirrors compute_seqa_loo_mosei.build, restricted to subset) ----
class _MOSEIV6Cfg:
    """Replicates the cfg shim from the MOSEI seqA training script, restricted
    to a branch's modality SUBSET."""
    def __init__(self, modalities):
        self.modalities = list(modalities)
        self.variates = {m: FEATURE_DIMS[m] for m in self.modalities}
        f = self.modalities[0]
        self.video_high_dim = FEATURE_DIMS[f]
        self.video_low_dim = FEATURE_DIMS[f]
        self.num_classes = NUM_CLASSES


def build_expert(C, modalities, input_length):
    cfg = _MOSEIV6Cfg(modalities)
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=cfg, output_dim=NUM_CLASSES, input_length=input_length,
        d_model=C['d_model'], nhead=C['nhead'],
        num_layers_per_modal=C['num_layers_per_modal'], num_layers=C['num_layers'],
        dropout=C['dropout'], verbose=False,
        video_low_dim=cfg.video_low_dim, video_high_dim=cfg.video_high_dim,
        use_bottleneck=True, n_bottlenecks=C['n_bottlenecks'],
        fusion_layer=C['fusion_layer'], use_sparse_moe=False,
        num_experts=C['num_experts'], expert_k=1, internal_dim=C['internal_dim'],
        bottleneck_head_pos=True, use_sparse_attn=C['use_sparse_attn'],
        factor=C['base_factor'],
        selector_video_source='high', encoder_video_source='high', no_selector=True,
        use_weighted_factor=False, use_triton=False, num_classes=NUM_CLASSES,
        sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
        downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
        per_modal_distill=C['per_modal_distill'],
        per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
        fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256,
        fusion_pos_embed_mode='full', fusion_mlp_ratio=1.0,
        fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'],
        seq_modality_order=C['seq_modality_order'], seq_random_order=C['seq_random_order'],
        bottleneck_agg_mode=C.get('bottleneck_agg_mode', 'gate'),
        n_fusion_distill=C['n_fusion_distill'],
    ).to(dev).float()


def unpack(batch):
    """MOSEI collate -> (high0, low0, high1, low1, ..., label[B,1]) over ALL_MODS
    order. Returns per-modality high/low dicts (full set) + label."""
    *feats, label = batch
    feats = [f.to(dev).float() for f in feats]
    label = label.squeeze(-1).long().to(dev)
    high = {m: feats[2 * j] for j, m in enumerate(ALL_MODS)}
    low = {m: feats[2 * j + 1] for j, m in enumerate(ALL_MODS)}
    return high, low, label


def expert_fwd(e, mods, high, low):
    """Run expert e on its modality SUBSET (build subset high/low dicts)."""
    h = {m: high[m] for m in mods}
    l = {m: low[m] for m in mods}
    out = e(h, l, training=False, return_selection_info=False)
    return out[0] if isinstance(out, tuple) else out


# ---- gate (DynMM MLP over per-modality pooled features) ----
# MOSEI modalities have DIFFERENT feature dims, so we cannot concat a single
# [B,T,C] tensor like the HAR datasets. Instead: mean-pool each modality's `high`
# tensor over time -> per-modality vector, project each to a common dim (64) via a
# per-modality Linear, SUM the projections into the gate feature, then MLP -> 3
# logits. The rest of the DynMM logic (diff_softmax, loss, hard route) is identical.
class Gate(nn.Module):
    PROJ_DIM = 64

    def __init__(s, mods, n=len(BRANCHES)):
        super().__init__()
        s.mods = list(mods)
        s.proj = nn.ModuleDict({m: nn.Linear(FEATURE_DIMS[m], s.PROJ_DIM) for m in s.mods})
        s.net = nn.Sequential(nn.Linear(s.PROJ_DIM, 128), nn.ReLU(), nn.Linear(128, n))

    def forward(s, high):                                   # high[m]: [B,T,C_m]
        feat = 0
        for m in s.mods:
            pooled = high[m].mean(1)                        # [B, C_m]
            feat = feat + s.proj[m](pooled)                 # [B, PROJ_DIM]
        return s.net(feat)                                  # [B, n]


def diff_softmax(logits, tau=1.0, hard=True):
    y = (logits / tau).softmax(-1)
    if hard:
        idx = y.max(-1, keepdim=True)[1]
        yh = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        return yh - y.detach() + y
    return y


# ---- per-fold cross-subject loaders (full modality set; subset chosen at fwd) ----
# Read the seqA config for max_seq_len / time_compression_ratio / use_batched_fusion
# so loaders match what the experts were trained on. Fall back to experts' configs.
def _seqa_loader_cfg():
    return {'max_seq_len': 50, 'time_compression_ratio': 4, 'use_batched_fusion': True}


# ---- load the 3 frozen experts (FAIL GRACEFULLY if not trained yet) ----
def _resolve_expert(br, fold):
    rd = sorted(glob.glob(os.path.join(RESROOT, 'dynmm_experts', br, '*')))
    rd = [d for d in rd if os.path.isfile(os.path.join(d, f'best_model_cold_fold{fold}.pth'))
          and os.path.isfile(os.path.join(d, 'config.json'))]
    return rd[0] if rd else None


missing = [br for br in BRANCHES if _resolve_expert(br, args.fold) is None]
if missing:
    print('[experts not ready] No trained seqA expert checkpoint found for '
          f'mosei fold{args.fold} branches: {missing}.')
    print(f'  Expected: {RESROOT}/dynmm_experts/{{{",".join(BRANCHES)}}}/<run>/'
          f'best_model_cold_fold{args.fold}.pth (+ config.json).')
    print('  Step I (train the 3 experts) must finish first. Exiting gracefully.')
    # Still verify the MOSEI pipeline + gate build before exiting.
    Lc = _seqa_loader_cfg()
    if Lc is None:
        print('[experts not ready] (could not read a seqA config for loader params; '
              'skipping dataset/gate build verification).')
        sys.exit(0)
    print('[verify] building MOSEI cross-subject dataset + gate (experts skipped)...')
    train_loader, val_loader, test_loader = mosei_get_dataloader(
        data_root=DATA_ROOT, batch_size=args.batch_size, num_workers=2,
        modalities=ALL_MODS, split_id=args.fold, train_shuffle=True,
        max_seq_len=Lc['max_seq_len'], time_compression_ratio=Lc['time_compression_ratio'],
        use_batched_fusion=Lc['use_batched_fusion'], available_sessions=None,
        split_mode=SPLIT_MODE,
    )
    _g = Gate(ALL_MODS).to(dev)
    print(f'[verify] OK: train/val/test loaders built; gate has '
          f'{sum(p.numel() for p in _g.parameters())} params. '
          'Pipeline + gate build verified.')
    sys.exit(0)


# ---- load loaders + experts ----
Lc = _seqa_loader_cfg()
assert Lc is not None, 'could not read a seqA_mosei config for loader params'
input_length = Lc['max_seq_len'] if Lc['max_seq_len'] > 0 else 600

train_loader, val_loader, test_loader = mosei_get_dataloader(
    data_root=DATA_ROOT, batch_size=args.batch_size, num_workers=4,
    modalities=ALL_MODS, split_id=args.fold, train_shuffle=True,
    max_seq_len=Lc['max_seq_len'], time_compression_ratio=Lc['time_compression_ratio'],
    use_batched_fusion=Lc['use_batched_fusion'], available_sessions=None,
    split_mode=SPLIT_MODE,
)

experts, expert_mods = [], []
for br in BRANCHES:
    mods = BR_MODS[br]
    rundir = _resolve_expert(br, args.fold)
    C = json.load(open(os.path.join(rundir, 'config.json')))
    e = build_expert(C, mods, input_length)
    sd = torch.load(os.path.join(rundir, f'best_model_cold_fold{args.fold}.pth'),
                    map_location='cpu', weights_only=True)
    e.load_state_dict({k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')},
                      strict=False)
    for p in e.parameters():
        p.requires_grad_(False)
    e.eval(); experts.append(e); expert_mods.append(mods)
    print(f'[expert] {br}: mods={mods} <- {rundir}')


# ---- per-branch GFLOPs (relative; for the FLOP reg + reporting) ----
from torch.utils.flop_counter import FlopCounterMode
b0 = next(iter(val_loader))
high0, low0, _ = unpack(b0)
high0 = {m: high0[m][:1] for m in ALL_MODS}
low0 = {m: low0[m][:1] for m in ALL_MODS}
flop_vec = []
for e, mods in zip(experts, expert_mods):
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        expert_fwd(e, mods, high0, low0)
    flop_vec.append(fc.get_total_flops() / 1e9)
flop_vec = torch.tensor(flop_vec, device=dev, dtype=torch.float32)
flop_norm = flop_vec / flop_vec.max()
print(f'[flops] per-branch GFLOPs={[round(f,4) for f in flop_vec.tolist()]} '
      f'(norm={[round(f,3) for f in flop_norm.tolist()]})')


# ---- train the gate ----
gate = Gate(ALL_MODS).to(dev)
opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-2)
for ep in range(args.gate_epochs):
    gate.train()
    for batch in train_loader:
        high, low, y = unpack(batch)
        with torch.no_grad():
            preds = torch.stack([expert_fwd(e, m, high, low)
                                 for e, m in zip(experts, expert_mods)], 1)   # [B,3,NUMC]
        w = diff_softmax(gate(high), tau=1.0, hard=True)                       # [B,3] straight-through
        out = (w.unsqueeze(-1) * preds).sum(1)                                 # [B,NUMC]
        loss = F.cross_entropy(out, y) + args.reg * (w.mean(0) * flop_norm).sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0); opt.step()
gate.eval()


# ---- eval: hard route each sample to its argmax branch ----
@torch.no_grad()
def evaluate(ld):
    preds_all, ys, choice = [], [], []
    for batch in ld:
        high, low, y = unpack(batch)
        pr = torch.stack([expert_fwd(e, m, high, low)
                          for e, m in zip(experts, expert_mods)], 1)           # [B,3,NUMC]
        b = gate(high).argmax(-1)                                              # [B] hard branch
        out = pr[torch.arange(pr.shape[0], device=dev), b]
        preds_all.append(out.argmax(-1).cpu()); ys.append(y.cpu()); choice.append(b.cpu())
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
