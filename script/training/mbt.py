"""Vanilla MBT — plain Multimodal Bottleneck Transformer classifier baseline. ONE trainer
for all datasets via --dataset. Backbone: models.mbt_vanilla.VanillaMBTClf (genuine shared
bottleneck-token fusion, SimpleMBTFusion), single forward pass, ALL modalities, NO early-exit /
gating / threshold. Plain cross-entropy, AdamW + Cosine, grad-clip 1.0, best-VAL selection,
test at the best-val checkpoint.

Same structure params as the other d128 baselines / seqA fusion: d_model=128, num_layers=6,
nhead=8, fusion_layer=3, n_bottlenecks=4.

  python training/mbt.py --dataset iemocap --fold 0 --results_dir ./results_mbt
  python training/mbt.py --dataset cmi     --fold 0 --results_dir ./results_mbt
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ee_data import build_loaders
from multimodal_model.mbt_vanilla import VanillaMBTClf


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei', 'meld', 'cmi', 'czu_mhad', 'mmfi', 'utd_mhad', 'ch_simsv2'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='mbt')
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=1)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject', choices=['random', 'cross_subject'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--num_layers', type=int, default=6)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--mlp_dim', type=int, default=128)
ap.add_argument('--fusion_layer', type=int, default=3)
ap.add_argument('--n_bottlenecks', type=int, default=4)
ap.add_argument('--warmup_epochs', type=int, default=0,
                help='Epochs with zero modality dropout before ramp')
ap.add_argument('--ramp_epochs', type=int, default=0,
                help='Epochs to linearly ramp drop prob from 0 to --max_drop_prob')
ap.add_argument('--max_drop_prob', type=float, default=0.0,
                help='Final per-sample per-modality dropout probability (0 = disabled)')
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)
model = VanillaMBTClf(
    cfg=cfg, num_classes=cfg.num_classes,
    d_model=args.d_model, num_layers=args.num_layers, num_heads=args.nhead,
    mlp_dim=args.mlp_dim, fusion_layer=args.fusion_layer, n_bottlenecks=args.n_bottlenecks,
).to(dev).float()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
ce = nn.CrossEntropyLoss()
print(f'[MBT] {args.dataset} M={cfg.modalities} in_mode={in_mode} in_len={in_len} '
      f'd={args.d_model} L={args.num_layers} h={args.nhead} fuse@{args.fusion_layer} '
      f'nb={args.n_bottlenecks} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)

# build modality slice map for per-sample zeroing
_slices = {}
_offset = 0
for _m in cfg.modalities:
    _n = cfg.variates[_m]
    _slices[_m] = (_offset, _offset + _n)
    _offset += _n


def _drop_schedule(ep, warmup, ramp, max_prob):
    if ep < warmup or max_prob == 0.0:
        return 0.0
    if ep < warmup + ramp:
        return max_prob * (ep - warmup) / ramp
    return max_prob


def _apply_mod_dropout(x, modalities, slices, drop_prob):
    """Per-sample Bernoulli zeroing: whole modality branch set to 0 for dropped samples."""
    x = x.clone()
    B = x.size(0)
    for b in range(B):
        drops = [m for m in modalities if torch.rand(1).item() < drop_prob]
        if len(drops) == len(modalities):
            drops = drops[:-1]  # guarantee at least one modality survives
        for m in drops:
            s, e = slices[m]
            x[b, :, s:e] = 0.0
    return x


@torch.no_grad()
def evaluate(loader):
    model.eval(); P, Y = [], []
    for batch in loader:
        x, y = unpack(batch, dev)
        logits = model(x)
        P.append(logits.argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


best, best_state = -1, None
for ep in range(args.num_epochs):
    drop_prob = _drop_schedule(ep, args.warmup_epochs, args.ramp_epochs, args.max_drop_prob)
    model.train()
    for batch in tl:
        x, y = unpack(batch, dev)
        if drop_prob > 0:
            x = _apply_mod_dropout(x, cfg.modalities, _slices, drop_prob)
        logits = model(x)
        loss = ce(logits, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sched.step()
    va, vf = evaluate(vl)
    if ep % 5 == 0 or ep == args.num_epochs - 1:
        print(f'  ep{ep:03d} drop={drop_prob:.3f} val acc={va:.4f} f1={vf:.4f}', flush=True)
    if va > best:
        best = va; best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
ta, tf = evaluate(el)
exp_dir = os.path.join(args.results_dir, args.exp_name)
os.makedirs(exp_dir, exist_ok=True)
torch.save(best_state, os.path.join(exp_dir, f'best_model_fold{args.fold}.pth'))
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'mbt_vanilla',
       'model_class': 'VanillaMBTClf', 'd_model': args.d_model, 'num_layers': args.num_layers,
       'nhead': args.nhead, 'fusion_layer': args.fusion_layer, 'n_bottlenecks': args.n_bottlenecks,
       'best_val_acc': best, 'best_test_acc': ta, 'best_test_f1': tf}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
print(f'[MBT] {args.dataset} TEST acc={ta:.4f} f1={tf:.4f} (best val {best:.4f}) -> {exp_dir}', flush=True)
