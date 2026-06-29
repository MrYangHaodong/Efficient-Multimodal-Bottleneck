"""Original-DynMM gate over 3 frozen branches, generalized to the 5 datasets. Step II.
Loads experts from RESROOT=dirname(results_dir)/experts/<br>/, measures per-branch GFLOPs
(FlopCounterMode in TRAIN mode), trains DiffSoftmax gate with CE + reg*expected-FLOP. NO distillation.
Eval = hard argmax routing; reports acc/f1, branch usage, mean GFLOPs. Per cross-subject fold."""
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ee_data import build_loaders
from multimodal_model.dynmm_orig_baseline import OrigDynMMExpertG, OrigGateG, OrigDynMM3BranchG
from utils.helper_function import set_seed

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='dynmm_orig_gate')
ap.add_argument('--gate_epochs', type=int, default=40)
ap.add_argument('--reg', type=float, default=0.1)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--tau', type=float, default=1.0)
ap.add_argument('--embed_dim', type=int, default=60)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=2)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject')
ap.add_argument('--cuda_pick', default='cuda:0')
args = ap.parse_args()
set_seed(args.seed)
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
RESROOT = os.path.dirname(os.path.abspath(args.results_dir))
BRANCHES = ['1mod', 'partial', 'full']
BR_MODS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dynmm_branches.json')))[args.dataset]

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)


def load_expert(br):
    cands = [d for d in sorted(glob.glob(os.path.join(RESROOT, 'experts', br, '*')))
             if os.path.isfile(os.path.join(d, 'best_model.pth'))]
    if not cands:
        raise FileNotFoundError(f'no expert for {br} under {RESROOT}/experts/{br}/*/best_model.pth')
    C = json.load(open(os.path.join(cands[0], 'config.json')))
    e = OrigDynMMExpertG(C['modalities'], cfg.modalities, cfg.variates, cfg.num_classes,
                         input_mode=in_mode, embed_dim=args.embed_dim).to(dev)
    e.load_state_dict(torch.load(os.path.join(cands[0], 'best_model.pth'), map_location='cpu')); e.eval()
    for p in e.parameters(): p.requires_grad = False
    return e


experts = [load_expert(br) for br in BRANCHES]

# per-branch GFLOPs (train mode -> counts TransformerEncoderLayer)
x0, _ = unpack(next(iter(el)), dev); x1 = x0[:1]
flop_vec = []
for e in experts:
    e.train(); fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc: e(x1)
    flop_vec.append(fc.get_total_flops() / 1e9); e.eval()
print(f'[flops] {args.dataset} per-branch GFLOPs={[round(g, 4) for g in flop_vec]}', flush=True)

gate = OrigGateG(cfg.modalities, cfg.variates, len(BRANCHES), input_mode=in_mode, gate_dim=10)
model = OrigDynMM3BranchG(experts, gate, flop_vec, tau=args.tau, hard=True).to(dev)
opt = torch.optim.AdamW(model.gate.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.gate_epochs)
ce = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(loader):
    model.eval(); P, Y, W = [], [], []
    for b in loader:
        x, y = unpack(b, dev); out, w = model(x); P.append(out.argmax(-1).cpu()); Y.append(y.cpu()); W.append(w.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy(); W = torch.cat(W).numpy()
    usage = W.mean(0); return (P == Y).mean(), f1_score(Y, P, average='macro'), usage, float((usage * np.array(flop_vec)).sum())


best = {'va': -1, 'ta': 0, 'tf': 0, 'usage': None, 'mg': 0}
for ep in range(args.gate_epochs):
    model.train(); [e.eval() for e in model.experts]
    for b in tl:
        x, y = unpack(b, dev); opt.zero_grad()
        out, reg = model(x, additional_loss=True)
        (ce(out, y) + args.reg * reg).backward()
        torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.0); opt.step()
    sched.step()
    va, vf, _, _ = evaluate(vl); ta, tf, us, mg = evaluate(el)
    if va > best['va']:
        best = {'va': va, 'ta': ta, 'tf': tf, 'usage': us.tolist(), 'mg': mg}
    print(f'ep{ep} val_acc={va:.4f} test_acc={ta:.4f} usage={us.round(2).tolist()} GFLOPs={mg:.3f}', flush=True)

exp_dir = os.path.join(args.results_dir, args.exp_name); os.makedirs(exp_dir, exist_ok=True)
json.dump({'dataset': args.dataset, 'method': 'dynmm_orig_3branch', 'fold': args.fold, 'branches': BRANCHES,
           'best_val_acc': float(best['va']), 'best_test_acc': float(best['ta']), 'best_test_f1': float(best['tf']),
           'usage_1mod_partial_full': best['usage'], 'mean_gflops': best['mg'], 'per_branch_gflops': flop_vec, 'reg': args.reg},
          open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
print(f"[DynMM-orig] {args.dataset} f{args.fold} test={best['ta']:.4f}/{best['tf']:.4f} "
      f"usage={best['usage']} GFLOPs={best['mg']:.3f} -> {exp_dir}")
