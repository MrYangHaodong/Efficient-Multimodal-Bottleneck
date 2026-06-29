"""ADMN baseline built on the trained seqA backbone (faithful 2-stage, NeurIPS 2025).

Per the user's spec:
  * Stage 1 (LayerDrop): initialise the seqA model from its already-trained
    per-dataset checkpoint, then fine-tune the whole backbone with RANDOM
    per-(modality, 2nd/3rd-encoder-layer) LayerDrop -> a backbone robust to any
    subset of per-modality encoder layers being dropped. (seqA's per-modal
    encoder has per_modal_distill=False, so dropping a layer preserves the
    sequence length -> clean residual-passthrough gating.)
  * Stage 2 (controller): freeze the Stage-1 backbone and train the
    ADMNLayerController (Gumbel top-k under a layer budget B) to pick, per
    sample, which droppable encoder layers run.

Outputs (so it slots into aggregate_baselines / the GFLOPs plots like the other
methods): results_fold{F}.json with best_test_acc/f1 = the controller's
accuracy/F1 at budget B, plus an 'admn' block (static-frozen top-B / full / none
references), and admn_model_fold{F}.pth (Stage-1 backbone + controller weights).

Scope: T=256 (word_length=1), daliahar + wesad.

Run from clean_models/train/:
  python main_admn_seqA.py --dataset daliahar --fold 0 --cuda_pick cuda:0 \
      --results_dir ../results_3p3/T256/admn_seqA --exp_name admn_seqA_T256
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # clean_models/
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.helper_function import set_seed
from utils.dataset_cfg import DaliaHAR, WESAD
from data.dataset_builder import HARDataset
from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import (
    DualVideoBottleneckModelV6Downsample)
from multimodal_model.admn_baseline import ADMNLayerController

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['daliahar', 'wesad'])
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--word_length', type=int, default=1)            # T=256
ap.add_argument('--transform', default='sax')
ap.add_argument('--stage1_epochs', type=int, default=30)         # LayerDrop FT (from seqA init -> converges fast)
ap.add_argument('--ctrl_epochs', type=int, default=40)           # controller
ap.add_argument('--batch_size', type=int, default=64)
ap.add_argument('--lr', type=float, default=1e-4)                # Stage-1 FT lr
ap.add_argument('--admn_budget', type=int, default=-1,           # -1 = ceil(M*n_drop/2)
                help='# droppable encoder-layers to KEEP (top-k over M*(L-1))')
ap.add_argument('--seqa_dir', default=None,
                help='seqA run dir holding config.json + best_model_fold*.pth (auto if None)')
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='admn_seqA_T256')
ap.add_argument('--seed', type=int, default=239)
args = ap.parse_args()

dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

# ----------------------------- data -----------------------------
if args.dataset == 'daliahar':
    ROOT = '/files1/haodong/data/processed_dalia_activity'
    cfg = DaliaHAR()
    SEQA_GLOB = '../results_3p3/T256/seqA/*'
else:
    ROOT = '/files1/haodong/data/WESAD/processed_wesad_activity'
    cfg = WESAD()
    SEQA_GLOB = '../results_3p3/T256_wesad/seqA/*'

MODS = list(cfg.modalities)
VARIATES = {m: cfg.variates[m] for m in MODS}
M = len(MODS)
NUMC = cfg.num_classes
fold_cfg = cfg.folds[args.fold]
train_subs, eval_subs = fold_cfg['train_set'], fold_cfg['eval_set']
val_subs = cfg.val_set
input_length = (cfg.duration * cfg.base_sample_rate) // args.word_length
sax_params = {'alphabet_size': 20, 'word_length': args.word_length}
print(f'[ADMN-seqA] {args.dataset} fold{args.fold} M={M} T={input_length} '
      f'train={train_subs} val={val_subs} test={eval_subs}')


def loader(subs, shuffle, drop_last):
    ds = HARDataset(ROOT, MODS, subs, cfg, args.transform, sax_params=sax_params)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=4, drop_last=drop_last)


train_loader = loader(train_subs, True, True)
val_loader = loader(val_subs, False, False)
test_loader = loader(eval_subs, False, False)

# ----------------------------- build seqA backbone + load checkpoint -----------------------------
seqa_dir = args.seqa_dir or sorted(glob.glob(SEQA_GLOB))[0]
C = json.load(open(os.path.join(seqa_dir, 'config.json')))
ckpt_path = os.path.join(seqa_dir, f'best_model_fold{args.fold}.pth')
assert os.path.isfile(ckpt_path), f'missing seqA ckpt {ckpt_path}'
print(f'[ADMN-seqA] seqA backbone <- {ckpt_path}')


class _V6Cfg:
    def __init__(self, ds_cfg):
        self.modalities = ds_cfg.modalities
        self.variates = ds_cfg.variates
        first = ds_cfg.modalities[0]
        self.video_high_dim = ds_cfg.variates[first]
        self.video_low_dim = ds_cfg.variates[first]


v6_cfg = _V6Cfg(cfg)
inner = DualVideoBottleneckModelV6Downsample(
    head_mode='gap', cfg=v6_cfg, output_dim=NUMC, input_length=input_length,
    d_model=C['d_model'], nhead=C['nhead'],
    num_layers_per_modal=C['num_layers_per_modal'], num_layers=C['num_layers'],
    dropout=C['dropout'], verbose=False,
    video_low_dim=v6_cfg.video_low_dim, video_high_dim=v6_cfg.video_high_dim,
    use_bottleneck=True, n_bottlenecks=C['n_bottlenecks'], fusion_layer=C['fusion_layer'],
    use_sparse_moe=False, num_experts=C['num_experts'], expert_k=1,
    internal_dim=C['internal_dim'], bottleneck_head_pos=True,
    use_sparse_attn=C['use_sparse_attn'], factor=C['base_factor'],
    selector_video_source='high', encoder_video_source='high', no_selector=True,
    use_weighted_factor=False, use_triton=False, num_classes=NUMC,
    sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
    downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
    per_modal_distill=C['per_modal_distill'],
    per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
    fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full',
    fusion_mlp_ratio=1.0,
    fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'],
    seq_modality_order=C['seq_modality_order'], seq_random_order=C['seq_random_order'],
    bottleneck_agg_mode='gate', n_fusion_distill=-1,
).to(dev).float()

# checkpoint was saved as the V6FusionOnlyDynaWrapper state (keys prefixed 'model.')
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
inner_state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
missing, unexpected = inner.load_state_dict(inner_state, strict=False)
assert not unexpected, f'unexpected keys: {unexpected[:5]}'
# only singleton-head / wrapper-only keys may be missing; backbone must be complete
bad = [k for k in missing if 'singleton' not in k]
assert not bad, f'missing backbone keys: {bad[:8]}'
print(f'[ADMN-seqA] loaded ({len(inner_state)} tensors; {len(missing)} non-backbone missing)')

n_drop = C['num_layers_per_modal'] - 1
assert n_drop >= 1, 'ADMN needs num_layers_per_modal >= 2'
B_budget = args.admn_budget if args.admn_budget > 0 else int(math.ceil(M * n_drop / 2))
channel_map, off = {}, 0
for m in MODS:
    channel_map[m] = list(range(off, off + VARIATES[m])); off += VARIATES[m]


def split_dicts(x):
    high, low, o = {}, {}, 0
    for m in MODS:
        n = VARIATES[m]
        high[m] = x[:, :, o:o + n]; low[m] = x[:, :, o:o + n]; o += n
    return high, low


def fwd(x, layer_keep=None):
    """x: [B,T,C] float. layer_keep: [B,M,n_drop] or None."""
    high, low = split_dicts(x)
    lk = layer_keep.permute(1, 0, 2).contiguous() if layer_keep is not None else None
    out = inner(high, low, training=inner.training,
                return_selection_info=False, layer_keep=lk)
    return out[0] if isinstance(out, tuple) else out


# ----------------------------- Stage 1: LayerDrop fine-tune -----------------------------
opt = torch.optim.AdamW(inner.parameters(), lr=args.lr, weight_decay=1e-4)
sched = CosineAnnealingLR(opt, T_max=args.stage1_epochs, eta_min=1e-6)


@torch.no_grad()
def _eval(ld, mode):
    """val acc. mode='full' -> all encoder layers on; mode='robust' -> mean acc
    over a FIXED set of random per-(modality,layer) LayerDrop masks. Because we
    init from already-trained seqA, full-mask val peaks at epoch 0 and would pick
    a non-LayerDrop-robust checkpoint; the 'robust' metric matches the Stage-1
    objective (tolerate any dropped-layer config) so we select on it instead.
    Masks are RNG-seeded identically every call -> comparable across epochs."""
    inner.eval()
    correct = total = 0
    g = torch.Generator(device=dev); g.manual_seed(12345)
    for x, y in ld:
        x, y = x.float().to(dev), y.to(dev)
        if mode == 'full':
            masks = [torch.ones(len(x), M, n_drop, device=dev)]
        else:
            masks = []
            for _ in range(4):
                dens = torch.rand(len(x), 1, 1, generator=g, device=dev)
                masks.append((torch.rand(len(x), M, n_drop, generator=g, device=dev) < dens).float())
        for keep in masks:
            correct += int((fwd(x, keep).argmax(-1) == y).sum().item()); total += len(y)
    return correct / total


best_val, best_state = -1.0, None
for ep in range(args.stage1_epochs):
    inner.train()
    for x, y in train_loader:
        x, y = x.float().to(dev), y.to(dev)
        dens = torch.rand(len(x), 1, 1, device=dev)                # per-sample keep density
        keep = (torch.rand(len(x), M, n_drop, device=dev) < dens).float()
        loss = F.cross_entropy(fwd(x, keep), y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), 1.0); opt.step()
    sched.step()
    vr = _eval(val_loader, 'robust')                               # LayerDrop-aware selection
    if vr > best_val:
        best_val = vr
        best_state = {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()}
    if ep % 5 == 0 or ep == args.stage1_epochs - 1:
        print(f'[stage1] ep{ep:3d} val_robust={vr:.4f} (best={best_val:.4f}) '
              f'val_full={_eval(val_loader, "full"):.4f}')
inner.load_state_dict(best_state)

# ----------------------------- Stage 2: controller (frozen backbone) -----------------------------
for p in inner.parameters():
    p.requires_grad_(False)
inner.eval()
ctrl = ADMNLayerController(channel_map, MODS, VARIATES, budget=B_budget, n_drop=n_drop).to(dev)
copt = torch.optim.AdamW(ctrl.parameters(), lr=1e-3, weight_decay=1e-4)
for ep in range(args.ctrl_epochs):
    ctrl.train()
    temp = max(0.5, 2.0 * (1 - ep / args.ctrl_epochs))
    for x, y in train_loader:
        x, y = x.float().to(dev), y.to(dev)
        mask, _ = ctrl(x, temp=temp)                               # [B,M,n_drop] straight-through
        loss = F.cross_entropy(fwd(x, mask), y)
        copt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0); copt.step()
ctrl.eval()

# ----------------------------- Stage 3: eval (controller vs static-top-B / full / none) -----------------------------
yte = torch.cat([y for _, y in test_loader]).numpy()


@torch.no_grad()
def eval_mask(maskfn):
    preds = []
    for x, _ in test_loader:
        x = x.float().to(dev)
        preds.append(fwd(x, maskfn(x)).argmax(-1).cpu())
    return torch.cat(preds).numpy()


# controller val-frequency -> static top-B layer set (matched compute)
with torch.no_grad():
    freq = torch.zeros(M, n_drop, device=dev)
    for x, _ in val_loader:
        freq += ctrl(x.float().to(dev))[0].sum(0)
flat_idx = torch.topk(freq.flatten(), B_budget).indices
static_vec = torch.zeros(M * n_drop, device=dev); static_vec[flat_idx] = 1.0
static_vec = static_vec.view(M, n_drop)

ctrl_pred = eval_mask(lambda x: ctrl(x)[0])
static_pred = eval_mask(lambda x: static_vec.unsqueeze(0).expand(len(x), -1, -1))
full_pred = eval_mask(lambda x: torch.ones(len(x), M, n_drop, device=dev))
none_pred = eval_mask(lambda x: torch.zeros(len(x), M, n_drop, device=dev))

ca = float((ctrl_pred == yte).mean()); cf = float(f1_score(yte, ctrl_pred, average='macro'))
sa = float((static_pred == yte).mean()); sf = float(f1_score(yte, static_pred, average='macro'))

# ----------------------------- save -----------------------------
exp_dir = os.path.join(args.results_dir, f"{args.exp_name}")
os.makedirs(exp_dir, exist_ok=True)
res = {
    'dataset': args.dataset, 'fold': args.fold, 'M': M, 'T': input_length,
    'admn_budget': B_budget, 'n_drop': n_drop, 'n_droppable': M * n_drop,
    'best_val_acc': best_val, 'best_val_f1': cf,         # controller is the reported method
    'best_test_acc': ca, 'best_test_f1': cf,
    'admn': {
        'controller_acc': ca, 'controller_f1': cf,
        'static_acc': sa, 'static_f1': sf, 'adv_over_static': ca - sa,
        'full_acc': float((full_pred == yte).mean()),
        'none_acc': float((none_pred == yte).mean()),
        'static_layers': [f'{MODS[i // n_drop]}:L{int(i % n_drop) + 1}' for i in flat_idx.cpu().tolist()],
        'stage1_best_val_robust': best_val,
    },
    'seqa_ckpt': ckpt_path, 'n_test': len(yte),
}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
torch.save({'backbone': inner.state_dict(), 'controller': ctrl.state_dict(),
            'budget': B_budget, 'n_drop': n_drop},
           os.path.join(exp_dir, f'admn_model_fold{args.fold}.pth'))
if not os.path.isfile(os.path.join(exp_dir, 'config.json')):
    json.dump({**C, 'method': 'admn_seqA', 'stage1_epochs': args.stage1_epochs,
               'ctrl_epochs': args.ctrl_epochs, 'admn_budget': B_budget},
              open(os.path.join(exp_dir, 'config.json'), 'w'), indent=2)
print(f"[ADMN-seqA] {args.dataset} f{args.fold} budget={B_budget}/{M * n_drop} "
      f"controller={ca:.4f}/{cf:.4f} static={sa:.4f} adv={ca - sa:+.4f} "
      f"| full={res['admn']['full_acc']:.4f} none={res['admn']['none_acc']:.4f} "
      f"| stage1_val={best_val:.4f} -> {exp_dir}")
