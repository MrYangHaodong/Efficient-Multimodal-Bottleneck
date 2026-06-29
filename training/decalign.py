"""DecAlign baseline — ONE trainer for all 6 datasets via ``--dataset``.

DecAlign (decoupled alignment) is a multimodal fusion model trained with a
special loss that combines the task cross-entropy with two alignment terms:

    total = CE(logits, y)
            + alpha1 * dec_loss                 # decouple specific vs common
            + alpha2 * (hete_loss + homo_loss)  # heterogeneity OT + homogeneity MMD

  python training/decalign.py --dataset iemocap  --fold 0 --cuda_pick cuda:0 --results_dir ./results_decalign
  python training/decalign.py --dataset daliahar --fold 0 --results_dir ./results_decalign
  python training/decalign.py --dataset dsads    --fold 0 --results_dir ./results_decalign   # fold = scenario

Replaces the 7 per-dataset main_decalign_{daliahar,dsads,eav,iemocap,mosei,
pamap2,wesad}_dyna.py scripts (wesad dropped -> 6 datasets). Per-dataset
hyperparameters (d_model / num_prototypes / transform) are unified behind
defaults keyed off the loader input mode (float vs sax), with CLI overrides.
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ee_data import build_loaders
from models.decalign import DecAlignDSADS


def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


ap = argparse.ArgumentParser(description='DecAlign — unified 6-dataset trainer.')
ap.add_argument('--dataset', required=True,
                choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--weight_decay', type=float, default=1e-4)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='decalign')
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=2)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject', choices=['random', 'cross_subject'])
# DecAlign hyperparameters (None -> per-dataset default keyed off input mode).
ap.add_argument('--d_model', type=int, default=None)
ap.add_argument('--num_prototypes', type=int, default=None)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--num_layers', type=int, default=3)
ap.add_argument('--conv1d_kernel', type=int, default=3)
ap.add_argument('--ot_reg', type=float, default=0.1)
ap.add_argument('--ot_num_iters', type=int, default=50)
ap.add_argument('--alpha1', type=float, default=0.1, help='decouple-loss weight')
ap.add_argument('--alpha2', type=float, default=0.01, help='alignment-loss weight (hete + homo)')
ap.add_argument('--dropout', type=float, default=0.1)
ap.add_argument('--alphabet_size', type=int, default=20)
ap.add_argument('--token_dim', type=int, default=16)
ap.add_argument('--clip_grad', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

# ---------------- Data ----------------
tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)

# float datasets (iemocap/mosei/eav): d_model=128, K=3, transform=None
# sax datasets   (daliahar/pamap2/dsads): d_model=64,  K=5, transform=<loader>
if in_mode == 'sax':
    d_model = args.d_model if args.d_model is not None else 64
    num_prototypes = args.num_prototypes if args.num_prototypes is not None else 5
    # The HAR loaders tokenize with 'sax' (daliahar/dsads) / 'sax_noisy' (pamap2);
    # 'sax' is sufficient to trigger the model's token-embedding path for either.
    transform = 'sax'
else:
    d_model = args.d_model if args.d_model is not None else 128
    num_prototypes = args.num_prototypes if args.num_prototypes is not None else 3
    transform = None

# ---------------- Model ----------------
model = DecAlignDSADS(
    cfg=cfg, num_classes=cfg.num_classes, input_length=in_len,
    d_model=d_model, nhead=args.nhead, num_layers=args.num_layers,
    num_prototypes=num_prototypes, ot_reg=args.ot_reg, ot_num_iters=args.ot_num_iters,
    conv1d_kernel=args.conv1d_kernel, dropout=args.dropout,
    alpha1=args.alpha1, alpha2=args.alpha2, transform=transform,
    alphabet_size=args.alphabet_size, token_dim=args.token_dim, verbose=True,
).to(dev).float()

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
ce = nn.CrossEntropyLoss()
print(f'[DecAlign] {args.dataset} M={cfg.modalities} in_mode={in_mode} d_model={d_model} '
      f'K={num_prototypes} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')


def decalign_loss(out, y):
    """DecAlign special loss: task CE + alpha1*decouple + alpha2*(hete + homo)."""
    return (ce(out['logits'], y)
            + args.alpha1 * out['dec_loss']
            + args.alpha2 * (out['hete_loss'] + out['homo_loss']))


@torch.no_grad()
def evaluate(loader):
    model.eval(); P, Y = [], []
    for batch in loader:
        x, y = unpack(batch, dev)
        out = model(x)
        P.append(out['logits'].argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


# ---------------- Train (best-VAL selection) ----------------
best_val, best_state = -1.0, None
for ep in range(args.num_epochs):
    model.train()
    for batch in tl:
        x, y = unpack(batch, dev)
        out = model(x)
        loss = decalign_loss(out, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
    sched.step()
    va, vf = evaluate(vl)
    if ep % 5 == 0 or ep == args.num_epochs - 1:
        print(f'  ep{ep:03d} val acc={va:.4f} f1={vf:.4f}')
    if va > best_val:
        best_val = va; best_state = copy.deepcopy(model.state_dict())

# ---------------- Test ----------------
model.load_state_dict(best_state)
ta, tf = evaluate(el)
exp_dir = os.path.join(args.results_dir, f'{args.exp_name}_{args.dataset}',
                       f'fold{args.fold}' if args.fold is not None else 'run')
os.makedirs(exp_dir, exist_ok=True)
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'decalign',
       'best_val_acc': best_val, 'best_test_acc': ta, 'best_test_f1': tf,
       'd_model': d_model, 'num_prototypes': num_prototypes, 'in_mode': in_mode,
       'alpha1': args.alpha1, 'alpha2': args.alpha2}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'
                                 if args.fold is not None else 'results.json'), 'w'), indent=2)
print(f'[DecAlign] {args.dataset} TEST acc={ta:.4f} f1={tf:.4f} (best_val={best_val:.4f}) -> {exp_dir}')
