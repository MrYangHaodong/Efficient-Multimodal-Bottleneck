"""AdaMML (Panda et al., ICCV'21) — whole-sample modality selection. ONE trainer for
all 6 datasets via --dataset (iemocap / daliahar / pamap2 / dsads / eav / mosei).

main net = per-modality 6-layer Transformer encoders + LOGITS late fusion; policy net
previews a cheap per-modality summary and emits per-modality Gumbel keep/drop decisions.
Faithful 3-stage AdaMML training:
  Stage 1 warmup  : main net, mode='all' (every modality), CE only.
  Stage 2 joint   : main + policy, mode='policy', Gumbel temp-decay (5.0 -> 0.5) +
                    GAMMA * per-modality FLOP cost penalty (expected compute).
  Stage 3 finetune: main net on the FROZEN learned policy, CE only.
Eval = hard per-modality decisions. Reports acc/F1, per-modality usage, expected GFLOPs
(sum enc_gflops*usage + policy) measured via FlopCounterMode.

The shared common.ee_data.build_loaders pipeline gives a CONCATENATED x [B,T,sum(variates)]
(IEMOCAP/eav/mosei cross-subject float; daliahar/pamap2/dsads integer SAX tokens).
AdaMML's encoders take a PER-MODALITY dict {m: (B,T,C_m)}, so x is split back per modality
via the shared channel map (_build_channel_map) and cast to float (encoders are nn.Linear,
no token embedding — matches the original HAR adamml trainer).

  python training/adamml.py --dataset iemocap  --fold 0 --results_dir ./results_adamml
  python training/adamml.py --dataset daliahar --fold 0 --results_dir ./results_adamml
  python training/adamml.py --dataset eav      --fold 0 --results_dir ./results_adamml
  python training/adamml.py --dataset dsads    --fold 0 --results_dir ./results_adamml  # fold = scenario
  python training/adamml.py --dataset mosei    --fold 0 --results_dir ./results_adamml
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ee_data import build_loaders
from multimodal_model.adamml import AdaMMLNet
from common.loader_utils import _build_channel_map

try:
    from utils.helper_function import set_seed
except Exception:                                       # fallback if helper unavailable
    def set_seed(seed):
        torch.manual_seed(seed); np.random.seed(seed)

ap = argparse.ArgumentParser(description='AdaMML — 6-dataset trainer')
ap.add_argument('--dataset', required=True,
                choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei', 'meld', 'cmi', 'czu_mhad', 'mmfi', 'utd_mhad', 'ch_simsv2'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--warmup_epochs', type=int, default=40)
ap.add_argument('--joint_epochs', type=int, default=40)
ap.add_argument('--finetune_epochs', type=int, default=20)
ap.add_argument('--gamma', type=float, default=0.5, help='efficiency (cost) penalty weight')
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--p_lr', type=float, default=1e-3, help='policy lr (AdaMML uses a higher p_lr)')
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='adamml')
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--dim_feedforward', type=int, default=None)  # None -> 4*d_model (original); set to d_model for seqA parity
ap.add_argument('--num_layers', type=int, default=6)          # per-modality transformer depth (spec)
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=1)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject', choices=['random', 'cross_subject'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--drop_warmup', type=int, default=0)
ap.add_argument('--drop_ramp', type=int, default=0)
ap.add_argument('--max_drop_prob', type=float, default=0.0)
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)
MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
VAR = {m: cfg.variates[m] for m in MODS}                 # per-modality in_dim -> mod_dims

# AdaMML's encoders take a PER-MODALITY dict {m: (B,T,C_m)}; ee_data's unpack gives a
# CONCATENATED x [B,T,sum(variates)]. Split x back by cfg.variates via the shared map.
_chan_map = _build_channel_map(MODS, VAR)
_boundaries = {m: (cols[0], cols[-1] + 1) for m, cols in _chan_map.items()}


def to_dict(x):                                          # [B,T,sum(C)] -> {m: [B,T,C_m]} float
    return {m: x[:, :, a:b].float() for m, (a, b) in _boundaries.items()}


def unpack_dict(batch, device):
    x, y = unpack(batch, device)                         # x: (B, T, sum(variates))
    return to_dict(x), y


# ---- infer T (max_len) from one batch: all modalities share the same padded T ----
xd0, _ = unpack_dict(next(iter(vl)), dev)
T = xd0[MODS[0]].shape[1]
print(f'[AdaMML] {args.dataset} f{args.fold} M={M} T={T} d={args.d_model} L={args.num_layers}')

model = AdaMMLNet(VAR, MODS, args.d_model, nhead=8, num_layers=args.num_layers,
                  num_classes=NUMC, dropout=0.1, max_len=T,
                  dim_feedforward=args.dim_feedforward).to(dev).float()


def _drop_sched_a(ep, dw, dr, mp):
    if ep < dw or mp == 0.0: return 0.0
    if ep < dw + dr: return mp * (ep - dw) / dr
    return mp

def _apply_mod_dropout_dict(xd, dp):
    if dp <= 0.0: return xd
    xd = {m: v.clone() for m, v in xd.items()}; B = next(iter(xd.values())).size(0); mods = list(xd)
    for b in range(B):
        drops = [m for m in mods if torch.rand(1).item() < dp]
        if len(drops) == len(mods): drops = drops[:-1]
        for m in drops: xd[m][b] = 0.0
    return xd

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
        xd, y = unpack_dict(batch, dev)
        out, dec, _ = model(xd, mode=mode)
        preds.append(out.argmax(-1).cpu()); ys.append(y.cpu()); decs.append(dec.cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    usage = torch.cat(decs).mean(0).numpy() if decs else np.zeros(M)
    return float((p == t).mean()), float(f1_score(t, p, average='macro')), usage


# ---- Stage 1: warmup main net (all modalities) ----
_gep = 0
for ep in range(args.warmup_epochs):
    model.train()
    for batch in tl:
        xd, y = unpack_dict(batch, dev)
        xd = _apply_mod_dropout_dict(xd, _drop_sched_a(_gep, args.drop_warmup, args.drop_ramp, args.max_drop_prob))
        out, _, _ = model(xd, mode='all')
        loss = ce(out, y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(main_params, 1.0); opt.step()
    va, _, _ = evaluate(vl, 'all'); print(f'[stage1 warmup] ep{ep:03d} val_all_acc={va:.4f}')
    _gep += 1

# ---- Stage 2: joint policy + main (Gumbel temp decay + cost penalty) ----
T0, T1 = 5.0, 0.5
for ep in range(args.joint_epochs):
    model.train()
    model.policy.set_temperature(T0 * (T1 / T0) ** (ep / max(1, args.joint_epochs - 1)))
    for batch in tl:
        xd, y = unpack_dict(batch, dev)
        xd = _apply_mod_dropout_dict(xd, _drop_sched_a(_gep, args.drop_warmup, args.drop_ramp, args.max_drop_prob))
        out, dec, _ = model(xd, mode='policy')
        cost = (dec.mean(0) * cost_norm).sum()                # expected per-modality compute
        loss = ce(out, y) + args.gamma * cost
        opt.zero_grad(); popt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); popt.step()
    va, vf, vu = evaluate(vl, 'policy'); print(f'[stage2 joint] ep{ep:03d} val_acc={va:.4f} usage={vu.round(2).tolist()}')
    _gep += 1

# ---- Stage 3: finetune main on frozen policy (best-val checkpoint) ----
for p in model.policy.parameters():
    p.requires_grad_(False)
best_ft_acc, best_ft_f1, best_ft_state = -1.0, -1.0, None
for ep in range(args.finetune_epochs):
    model.train(); model.policy.eval()
    for batch in tl:
        xd, y = unpack_dict(batch, dev)
        out, _, _ = model(xd, mode='policy')
        loss = ce(out, y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(main_params, 1.0); opt.step()
    va, vf, _ = evaluate(vl, 'policy')
    print(f'[stage3 finetune] ep{ep:03d} val_acc={va:.4f}')
    if va > best_ft_acc:
        best_ft_acc = va; best_ft_f1 = vf
        best_ft_state = copy.deepcopy(model.state_dict())
model.load_state_dict(best_ft_state)

# ---- eval ----
_, _, vu = evaluate(vl, 'policy')
ta, tf, tu = evaluate(el, 'policy')
mean_g = float((torch.tensor(tu) * cost_vec.cpu()).sum() + policy_g)
exp_dir = os.path.join(args.results_dir, args.exp_name); os.makedirs(exp_dir, exist_ok=True)
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'adamml',
       'best_val_acc': best_ft_acc, 'best_val_f1': best_ft_f1, 'best_test_acc': ta, 'best_test_f1': tf,
       'adamml': {'modalities': MODS, 'test_usage': tu.tolist(), 'per_mod_gflops': [enc_g[m] for m in MODS],
                  'policy_gflops': policy_g, 'mean_gflops': mean_g, 'full_gflops': full_g,
                  'gamma': args.gamma}, 'n_test': len(tu)}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
torch.save(best_ft_state, os.path.join(exp_dir, f'best_model_fold{args.fold}.pth'))
print(f"[AdaMML] {args.dataset} f{args.fold} test_acc={ta:.4f}/{tf:.4f} usage={tu.round(2).tolist()} "
      f"mean_GFLOPs={mean_g:.3f} (full={full_g:.3f}) -> {exp_dir}")
