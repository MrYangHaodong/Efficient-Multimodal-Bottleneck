"""AdaMML baseline (whole-sample modality selection) on EAV.

EAV variant of main_adamml_ours.py. Identical AdaMML logic (model build,
per-modality encoder + policy GFLOPs measurement, 3-stage training: warmup mode='all'
-> joint mode='policy' with Gumbel temp decay + cost penalty -> finetune frozen policy,
evaluate with hard decisions, same results-json + save format) — only the DATA PIPELINE
is adapted.

EAV specifics:
  * 6 modalities (video/audio/text/mocap_hand/mocap_head/mocap_rotated) with DIFFERENT
    feature dims (FEATURE_DIMS) -> used as `mod_dims`; AdaMMLNet builds a per-modality
    encoder so heterogeneous dims are fine. NUM_CLASSES=4.
  * cross-subject session-disjoint splits (split_mode='cross_subject', split_id=fold).
  * list-based batch [high0, low0, high1, low1, ..., label]; we take the per-modality
    `high` tensors and build xd = {m: high[m]} for ALL 6 modalities.
  * max_len for AdaMMLNet = the sequence length T after time compression, inferred from
    one batch (high[m].shape[1]; all modalities share the same padded T).

Run from clean_models/train/.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_function import set_seed
from data.EAV.get_data import (
    get_dataloader as eav_get_dataloader,
    FEATURE_DIMS, NUM_CLASSES, EAVDataset,
)
from multimodal_model.adamml_baseline import AdaMMLNet

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='eav', choices=['eav'])
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--num_layers', type=int, default=6)          # per-modality transformer depth (spec)
ap.add_argument('--warmup_epochs', type=int, default=40)
ap.add_argument('--joint_epochs', type=int, default=40)
ap.add_argument('--finetune_epochs', type=int, default=20)
ap.add_argument('--gamma', type=float, default=0.5, help='efficiency (cost) penalty weight')
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--p_lr', type=float, default=1e-3)           # policy lr (AdaMML uses a higher p_lr)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='adamml_ours')
ap.add_argument('--seed', type=int, default=239)
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

# ---- EAV plumbing (mirrors main_dynmm_seqA_gate_eav.py) ----
DATA_ROOT = '/files1/haodong/data/EAV'
SPLIT_MODE = 'cross_subject'
MODS = list(EAVDataset.ALL_MODALITIES)   # ['video','audio','text','mocap_hand','mocap_head','mocap_rotated']
M = len(MODS); NUMC = NUM_CLASSES
VAR = {m: FEATURE_DIMS[m] for m in MODS}      # per-modality in_dim -> mod_dims


# ---- loader params: read a seqA_eav config (max_seq_len/compression/batched) so
# loaders match what the experts/seqA were trained on (mirrors the dynmm script). ----
def _seqa_loader_cfg():
    seqa = sorted(glob.glob('./results_eav/seqA*/*')) + sorted(glob.glob('../results_3p3/seqA_eav/*'))
    for d in seqa:
        cp = os.path.join(d, 'config.json')
        if os.path.isfile(cp):
            c = json.load(open(cp))
            if 'max_seq_len' in c:
                return c
    return None


Lc = _seqa_loader_cfg()
assert Lc is not None, 'could not read a seqA_eav config for loader params'

train_loader, val_loader, test_loader = eav_get_dataloader(
    data_root=DATA_ROOT, batch_size=args.batch_size, num_workers=4,
    modalities=MODS, split_id=args.fold, train_shuffle=True,
    max_seq_len=Lc['max_seq_len'], time_compression_ratio=Lc['time_compression_ratio'],
    use_batched_fusion=Lc['use_batched_fusion'], available_sessions=None,
    split_mode=SPLIT_MODE,
)


def unpack(batch):
    """EAV collate -> (high0, low0, high1, low1, ..., label[B,1]). Returns the
    per-modality high tensors as xd = {m: high[m]} (AdaMML uses the high path) + label."""
    *feats, label = batch
    feats = [f.to(dev).float() for f in feats]
    label = label.squeeze(-1).long().to(dev)
    xd = {m: feats[2 * j] for j, m in enumerate(MODS)}   # high path per modality
    return xd, label


# ---- infer T (max_len) from one batch: all modalities share the same padded T ----
xd0, _ = unpack(next(iter(val_loader)))
T = xd0[MODS[0]].shape[1]
print(f'[AdaMML] {args.dataset} f{args.fold} M={M} T={T} d={args.d_model} L={args.num_layers}')

model = AdaMMLNet(VAR, MODS, args.d_model, nhead=8, num_layers=args.num_layers,
                  num_classes=NUMC, dropout=0.1, max_len=T).to(dev).float()

# ---- per-modality encoder + policy GFLOPs (for cost penalty + reporting) ----
xd1 = {m: xd0[m][:1] for m in MODS}
enc_g = {}
for m in MODS:
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        model.encoders[m](xd1[m])
    enc_g[m] = fc.get_total_flops() / 1e9
fc = FlopCounterMode(display=False)
with fc, torch.no_grad():
    model.policy(xd1)
policy_g = fc.get_total_flops() / 1e9
cost_vec = torch.tensor([enc_g[m] for m in MODS], device=dev)
cost_norm = cost_vec / cost_vec.sum()
full_g = sum(enc_g.values()) + policy_g
print(f'[gflops] per-mod enc={[round(enc_g[m],4) for m in MODS]} policy={policy_g:.4f} full={full_g:.4f}')

ce = nn.CrossEntropyLoss()
main_params = [p for n, p in model.named_parameters() if not n.startswith('policy.')]
opt = torch.optim.AdamW(main_params, lr=args.lr, weight_decay=1e-4)
popt = torch.optim.AdamW(model.policy.parameters(), lr=args.p_lr, weight_decay=1e-4)


@torch.no_grad()
def evaluate(ld, mode='policy'):
    model.eval(); preds, ys, decs = [], [], []
    for batch in ld:
        xd, y = unpack(batch)
        out, dec, _ = model(xd, mode=mode)
        preds.append(out.argmax(-1).cpu()); ys.append(y.cpu()); decs.append(dec.cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    usage = torch.cat(decs).mean(0).numpy() if decs else np.zeros(M)
    return float((p == t).mean()), float(f1_score(t, p, average='macro')), usage


# ---- Stage 1: warmup main net (all modalities) ----
for ep in range(args.warmup_epochs):
    model.train()
    for batch in train_loader:
        xd, y = unpack(batch)
        out, _, _ = model(xd, mode='all')
        loss = ce(out, y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(main_params, 1.0); opt.step()
    va, _, _ = evaluate(val_loader, 'all'); print(f'[stage1 warmup] val_all_acc={va:.4f}')

# ---- Stage 2: joint policy + main (Gumbel temp decay + cost penalty) ----
T0, T1 = 5.0, 0.5
for ep in range(args.joint_epochs):
    model.train()
    model.policy.set_temperature(T0 * (T1 / T0) ** (ep / max(1, args.joint_epochs - 1)))
    for batch in train_loader:
        xd, y = unpack(batch)
        out, dec, _ = model(xd, mode='policy')
        cost = (dec.mean(0) * cost_norm).sum()                # expected per-modality compute
        loss = ce(out, y) + args.gamma * cost
        opt.zero_grad(); popt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); popt.step()
    va, vf, vu = evaluate(val_loader, 'policy'); print(f'[stage2 joint] val_acc={va:.4f} usage={vu.round(2).tolist()}')

# ---- Stage 3: finetune main on frozen policy ----
for p in model.policy.parameters():
    p.requires_grad_(False)
for ep in range(args.finetune_epochs):
    model.train(); model.policy.eval()
    for batch in train_loader:
        xd, y = unpack(batch)
        out, _, _ = model(xd, mode='policy')
        loss = ce(out, y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(main_params, 1.0); opt.step()

# ---- eval ----
va, vfa, vu = evaluate(val_loader, 'policy')
ta, tf, tu = evaluate(test_loader, 'policy')
mean_g = float((torch.tensor(tu) * cost_vec.cpu()).sum() + policy_g)
exp_dir = os.path.join(args.results_dir, args.exp_name); os.makedirs(exp_dir, exist_ok=True)
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'adamml_ours',
       'best_val_acc': va, 'best_val_f1': vfa, 'best_test_acc': ta, 'best_test_f1': tf,
       'adamml': {'modalities': MODS, 'test_usage': tu.tolist(), 'per_mod_gflops': [enc_g[m] for m in MODS],
                  'policy_gflops': policy_g, 'mean_gflops': mean_g, 'full_gflops': full_g,
                  'gamma': args.gamma}, 'n_test': len(tu)}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
torch.save(model.state_dict(), os.path.join(exp_dir, f'adamml_fold{args.fold}.pth'))
print(f"[AdaMML] {args.dataset} f{args.fold} test_acc={ta:.4f}/{tf:.4f} usage={tu.round(2).tolist()} "
      f"mean_GFLOPs={mean_g:.3f} (full={full_g:.3f}) -> {exp_dir}")
