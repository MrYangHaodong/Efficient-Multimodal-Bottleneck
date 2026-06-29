"""CREMA-style baseline on IEMOCAP / daliahar / pamap2 / dsads (concept port of
CREMA: per-modality modular adapters + modality-sequential training + modality-
adaptive early exit). MODALITY-axis early exit. See crema_style_baseline.py.

  python main_crema_4ds.py --dataset iemocap  --fold 0 --cuda_pick cuda:0
  python main_crema_4ds.py --dataset pamap2   --fold 0
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ee_data import build_loaders
from multimodal_model.crema_style_baseline import CREMAStyleClassifier

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'mosei'])
ap.add_argument('--fold', type=int, default=None)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--weight_decay', type=float, default=1e-4)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--word_length', type=int, default=2)
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject', choices=['random', 'cross_subject'])
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--num_layers', type=int, default=3)
ap.add_argument('--adapter_dim', type=int, default=32)
ap.add_argument('--num_query', type=int, default=4)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', default='./results_crema')
ap.add_argument('--exp_name', default='crema_style')
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed); np.random.seed(args.seed)
TAUS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)
model = CREMAStyleClassifier(cfg, in_len, input_mode=in_mode, d_model=args.d_model, nhead=args.nhead,
                             num_layers=args.num_layers, adapter_dim=args.adapter_dim,
                             num_query=args.num_query).to(dev).float()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
ce = nn.CrossEntropyLoss()
M = model.M
print(f'[CREMA-style] {args.dataset} M={cfg.modalities} layers={args.num_layers} '
      f'adapter={args.adapter_dim} q={args.num_query} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')


@torch.no_grad()
def eval_full(loader):
    model.eval(); P, Y = [], []
    for batch in loader:
        x, y = unpack(batch, dev)
        out = model(x)['prefix_logits'][-1]          # all modalities
        P.append(out.argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


@torch.no_grad()
def exit_curve(loader):
    model.eval(); rows = []
    for tau in TAUS:
        P, Y, K = [], [], []
        for batch in loader:
            x, y = unpack(batch, dev)
            logit, k = model.exit_inference(x, tau)
            P.append(logit.argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy()); K.append(k.cpu().numpy())
        p, y, k = np.concatenate(P), np.concatenate(Y), np.concatenate(K)
        rows.append((float(k.mean()), float((p == y).mean()), float(f1_score(y, p, average='macro'))))
    return sorted(rows)


best, best_state = -1, None
for ep in range(args.num_epochs):
    model.train()
    for batch in tl:
        x, y = unpack(batch, dev)
        prefix = model(x)['prefix_logits']
        loss = sum(ce(lg, y) for lg in prefix) / len(prefix)    # modality-sequential: supervise every prefix
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sched.step()
    va, vf = eval_full(vl)
    if ep % 5 == 0 or ep == args.num_epochs - 1:
        print(f'  ep{ep:03d} val acc={va:.4f} f1={vf:.4f}')
    if va > best:
        best = va; best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
ta, tf = eval_full(el)
curve = exit_curve(el)
exp_dir = os.path.join(args.results_dir, f'{args.exp_name}_{args.dataset}',
                       f'fold{args.fold}' if args.fold is not None else 'run')
os.makedirs(exp_dir, exist_ok=True)
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'crema_style',
       'best_val_acc': best, 'best_test_acc': ta, 'best_test_f1': tf, 'M': M,
       'exit_curve_meanMods_acc_f1': curve, 'taus': TAUS}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'
                                  if args.fold is not None else 'results.json'), 'w'), indent=2)
print(f'[CREMA-style] {args.dataset} TEST acc={ta:.4f} f1={tf:.4f} | '
      f'exit-curve acc {curve[0][1]:.3f}->{curve[-1][1]:.3f} @ meanMods {curve[0][0]:.2f}->{curve[-1][0]:.2f} '
      f'(of {M}) -> {exp_dir}')
