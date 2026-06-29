"""Original-DynMM gate over 3 frozen branches on CMU-MOSEI (Acc-2). Step II.
Loads the 3 experts (1mod/partial/full) from RESROOT=dirname(results_dir)/experts/<br>/, measures
per-branch GFLOPs (FlopCounterMode in TRAIN mode so TransformerEncoderLayer is counted), trains the
DiffSoftmax gate with CE + reg*expected-FLOP. Eval = hard argmax routing; reports acc/f1, branch
usage, mean GFLOPs. NO distillation. Run from clean_models/train/."""
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.CMU_MOSEI.get_data import get_dataloader, FEATURE_DIMS, NUM_CLASSES
from multimodal_model.dynmm_orig_baseline import OrigDynMMExpert, OrigDynMM3Branch
from utils.helper_function import set_seed

ALL = ['vision', 'audio', 'text']
ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--results_dir', required=True)              # gate output dir; experts at dirname(.)/experts
ap.add_argument('--exp_name', default='dynmm_orig_gate')
ap.add_argument('--gate_epochs', type=int, default=40)
ap.add_argument('--reg', type=float, default=0.1)
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--tau', type=float, default=1.0)
ap.add_argument('--max_seq_len', type=int, default=50)
ap.add_argument('--tcr', type=int, default=1)
ap.add_argument('--cuda_pick', default='cuda:0')
args = ap.parse_args()
set_seed(args.seed)
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
RESROOT = os.path.dirname(os.path.abspath(args.results_dir))
BRANCHES = ['1mod', 'partial', 'full']

tl, vl, el = get_dataloader(batch_size=args.batch_size, num_workers=4, modalities=ALL, split_id=0,
                            train_shuffle=True, max_seq_len=args.max_seq_len,
                            time_compression_ratio=args.tcr, use_batched_fusion=True, task='classification2')


def unpack(batch):
    *feats, label = batch
    fd = {m: feats[2 * j].to(dev).float() for j, m in enumerate(ALL)}
    gate_in = torch.cat([fd[m] for m in ALL], dim=-1)            # (B,T,409)
    y = label.long().to(dev); y = y.squeeze(-1) if y.ndim > 1 else y
    return fd, gate_in, y


# ---- load 3 frozen experts ----
def load_expert(br):
    cands = [d for d in sorted(glob.glob(os.path.join(RESROOT, 'experts', br, '*')))
             if os.path.isfile(os.path.join(d, 'best_model.pth'))]
    if not cands:
        raise FileNotFoundError(f'no expert for branch {br} under {RESROOT}/experts/{br}/*/best_model.pth')
    d = cands[0]; C = json.load(open(os.path.join(d, 'config.json')))
    e = OrigDynMMExpert(C['modalities'], {m: FEATURE_DIMS[m] for m in C['modalities']}, NUM_CLASSES).to(dev)
    e.load_state_dict(torch.load(os.path.join(d, 'best_model.pth'), map_location='cpu')); e.eval()
    for p in e.parameters(): p.requires_grad = False
    return e, C['modalities']


experts, exp_mods = [], []
for br in BRANCHES:
    e, m = load_expert(br); experts.append(e); exp_mods.append(m)

# ---- per-branch GFLOPs (train mode -> count TransformerEncoderLayer) ----
fd0, gin0, _ = unpack(next(iter(el)))
fd1 = {m: v[:1] for m, v in fd0.items()}
flop_vec = []
for e in experts:
    e.train()
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc: e(fd1)
    flop_vec.append(fc.get_total_flops() / 1e9); e.eval()
print(f'[flops] per-branch GFLOPs={[round(g,4) for g in flop_vec]}', flush=True)

model = OrigDynMM3Branch(experts, sum(FEATURE_DIMS[m] for m in ALL), flop_vec, tau=args.tau, hard=True).to(dev)
opt = torch.optim.AdamW(model.gate.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.gate_epochs)
ce = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(loader):
    model.eval(); P, Y, W = [], [], []
    for b in loader:
        fd, gin, y = unpack(b); out, w = model(fd, gin)        # hard routing (hard=True)
        P.append(out.argmax(-1).cpu()); Y.append(y.cpu()); W.append(w.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy(); W = torch.cat(W).numpy()
    usage = W.mean(0); mean_g = float((usage * np.array(flop_vec)).sum())
    return (P == Y).mean(), f1_score(Y, P, average='macro'), usage, mean_g


best = {'va': -1, 'ta': 0, 'tf': 0, 'usage': None, 'mg': 0}
for ep in range(args.gate_epochs):
    model.train(); [e.eval() for e in model.experts]                 # keep frozen experts deterministic
    for b in tl:
        fd, gin, y = unpack(b); opt.zero_grad()
        out, reg = model(fd, gin, additional_loss=True)
        (ce(out, y) + args.reg * reg).backward()
        torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.0); opt.step()
    sched.step()
    va, vf, _, _ = evaluate(vl); ta, tf, us, mg = evaluate(el)
    if va > best['va']:
        best = {'va': va, 'ta': ta, 'tf': tf, 'usage': us.tolist(), 'mg': mg}
    print(f'ep{ep} val_acc={va:.4f} test_acc={ta:.4f} usage={us.round(2).tolist()} GFLOPs={mg:.3f}', flush=True)

exp_dir = os.path.join(args.results_dir, args.exp_name); os.makedirs(exp_dir, exist_ok=True)
res = {'dataset': 'mosei', 'method': 'dynmm_orig_3branch', 'seed': args.seed, 'branches': BRANCHES,
       'best_val_acc': float(best['va']), 'best_test_acc': float(best['ta']), 'best_test_f1': float(best['tf']),
       'usage_1mod_partial_full': best['usage'], 'mean_gflops': best['mg'], 'per_branch_gflops': flop_vec,
       'reg': args.reg, 'tcr': args.tcr}
json.dump(res, open(os.path.join(exp_dir, 'results_fold0.json'), 'w'), indent=2)
print(f"[DynMM-orig] mosei s{args.seed} test={best['ta']:.4f}/{best['tf']:.4f} "
      f"usage={best['usage']} GFLOPs={best['mg']:.3f} -> {exp_dir}")
