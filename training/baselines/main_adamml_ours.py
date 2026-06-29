"""AdaMML baseline (whole-sample modality selection) on our time-series datasets.
main net = per-modality 6-layer Transformer + logits late fusion; policy net previews
a cheap summary and emits per-modality Gumbel keep/drop decisions. Faithful 3-stage
AdaMML training: warmup main (all modalities) -> joint policy+main (Gumbel temp-decay
+ per-modality cost penalty) -> finetune main on the learned policy.

HAR datasets (daliahar/wesad/pamap2). Eval = hard per-modality decisions; reports
acc/F1, per-modality usage, expected GFLOPs (sum enc_gflops*usage + policy). Saves in
the same json format as the DynMM baseline so it slots into the comparison.
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
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_function import set_seed
from utils.dataset_cfg import DaliaHAR, WESAD
from data.dataset_builder import HARDataset
from multimodal_model.adamml_baseline import AdaMMLNet

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['daliahar', 'wesad', 'pamap2', 'dsads'])
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--d_model', type=int, default=64)
ap.add_argument('--num_layers', type=int, default=6)          # per-modality transformer depth (spec)
ap.add_argument('--warmup_epochs', type=int, default=40)
ap.add_argument('--joint_epochs', type=int, default=40)
ap.add_argument('--finetune_epochs', type=int, default=20)
ap.add_argument('--gamma', type=float, default=0.5, help='efficiency (cost) penalty weight')
ap.add_argument('--batch_size', type=int, default=64)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--p_lr', type=float, default=1e-3)           # policy lr (AdaMML uses a higher p_lr)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='adamml_ours')
ap.add_argument('--seed', type=int, default=239)
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

LKIND = 'har'
if args.dataset == 'daliahar':
    ROOT = '/files1/haodong/data/processed_dalia_activity'; cfg = DaliaHAR(); WL = 1; TRANSFORM = 'sax'
elif args.dataset == 'wesad':
    ROOT = '/files1/haodong/data/WESAD/processed_wesad_activity'; cfg = WESAD(); WL = 1; TRANSFORM = 'sax'
elif args.dataset == 'dsads':
    from utils.dataset_cfg import DSADS
    cfg = DSADS(); WL = 1; TRANSFORM = 'sax'; LKIND = 'dsads'         # T=125 clean sax, 4 scenarios
else:
    from utils.dataset_cfg import PAMAP2
    from data.pamap2_crosssubject import load_pamap2_subjects
    cfg = PAMAP2(); ROOT = cfg.cross_subject_dir; WL = 1; TRANSFORM = 'sax_noisy'; LKIND = 'pamap2'   # T=512 sax_noisy (match seqA@512)
MODS = list(cfg.modalities); M = len(MODS); NUMC = cfg.num_classes
VAR = {m: cfg.variates[m] for m in MODS}
cm, off = {}, 0
for m in MODS:
    cm[m] = (off, off + VAR[m]); off += VAR[m]
sax = {'alphabet_size': 20, 'word_length': WL}

if LKIND == 'dsads':
    # DSADS diversify scenarios (fold = scenario): src split 80/20 (stratified, seed
    # 2711) for train/val; trg = test. Same pipeline as the other DSADS mains.
    from data.dataset_builder import DSADSDataset
    from sklearn.model_selection import train_test_split
    DATA_ROOT = '/files1/haodong/data/DSADS/'
    _src = torch.load(os.path.join(DATA_ROOT, f'dsad_diversify_dict_scenario{args.fold}_src.pt'), weights_only=False)
    _trg = torch.load(os.path.join(DATA_ROOT, f'dsad_diversify_dict_scenario{args.fold}_trg.pt'), weights_only=False)
    _trx, _vax, _try, _vay = train_test_split(_src['samples'], _src['labels'], test_size=0.2,
                                              random_state=2711, stratify=_src['labels'])

    def _mk(d, shuffle, drop_last):
        ds = DSADSDataset(d, MODS, cfg, 'sax', sax_params=sax)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_last)

    train_loader = _mk({'samples': _trx, 'labels': _try}, True, True)
    val_loader = _mk({'samples': _vax, 'labels': _vay}, False, False)
    test_loader = _mk(_trg, False, False)
else:
    fold_cfg = cfg.folds[args.fold]
    train_subs, eval_subs, val_subs = fold_cfg['train_set'], fold_cfg['eval_set'], cfg.val_set

    def loader(subs, shuffle, drop_last):
        if LKIND == 'har':
            ds = HARDataset(ROOT, MODS, subs, cfg, 'sax', sax_params=sax)
        else:
            ds = load_pamap2_subjects(ROOT, MODS, subs, cfg, TRANSFORM, sax_params=sax)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_last)

    train_loader = loader(train_subs, True, True)
    val_loader = loader(val_subs, False, False)
    test_loader = loader(eval_subs, False, False)
T = next(iter(val_loader))[0].shape[1]
print(f'[AdaMML] {args.dataset} f{args.fold} M={M} T={T} d={args.d_model} L={args.num_layers}')


def split(x):                                                 # [B,T,C] -> dict m -> [B,T,V_m]
    return {m: x[:, :, cm[m][0]:cm[m][1]] for m in MODS}


model = AdaMMLNet(VAR, MODS, args.d_model, nhead=8, num_layers=args.num_layers,
                  num_classes=NUMC, dropout=0.1, max_len=T).to(dev).float()

# ---- per-modality encoder + policy GFLOPs (for cost penalty + reporting) ----
x1 = next(iter(val_loader))[0][:1].float().to(dev)
xd1 = split(x1)
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
    for x, y in ld:
        x = x.float().to(dev)
        out, dec, _ = model(split(x), mode=mode)
        preds.append(out.argmax(-1).cpu()); ys.append(y); decs.append(dec.cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    usage = torch.cat(decs).mean(0).numpy() if decs else np.zeros(M)
    return float((p == t).mean()), float(f1_score(t, p, average='macro')), usage


# ---- Stage 1: warmup main net (all modalities) ----
for ep in range(args.warmup_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.float().to(dev), y.to(dev)
        out, _, _ = model(split(x), mode='all')
        loss = ce(out, y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(main_params, 1.0); opt.step()
va, _, _ = evaluate(val_loader, 'all'); print(f'[stage1 warmup] val_all_acc={va:.4f}')

# ---- Stage 2: joint policy + main (Gumbel temp decay + cost penalty) ----
T0, T1 = 5.0, 0.5
for ep in range(args.joint_epochs):
    model.train()
    model.policy.set_temperature(T0 * (T1 / T0) ** (ep / max(1, args.joint_epochs - 1)))
    for x, y in train_loader:
        x, y = x.float().to(dev), y.to(dev)
        out, dec, _ = model(split(x), mode='policy')
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
    for x, y in train_loader:
        x, y = x.float().to(dev), y.to(dev)
        out, _, _ = model(split(x), mode='policy')
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
