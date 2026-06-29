"""Original-DynMM expert (MultiBench Transformer + Concat, NO distill) on ONE modality subset,
generalized to the 5 datasets (iemocap/daliahar/pamap2/dsads/eav) via ee_data.build_loaders
(float or SAX). Per cross-subject fold. Saves best_model.pth + config.json for the gate."""
import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ee_data import build_loaders
from multimodal_model.dynmm_orig_baseline import OrigDynMMExpertG
from utils.helper_function import set_seed

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav'])
ap.add_argument('--modalities', nargs='+', required=True)
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', required=True)
ap.add_argument('--num_epochs', type=int, default=30)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--embed_dim', type=int, default=60)
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=2)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject')
ap.add_argument('--cuda_pick', default='cuda:0')
args = ap.parse_args()
set_seed(args.seed)
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)
SUB = list(args.modalities)
model = OrigDynMMExpertG(SUB, cfg.modalities, cfg.variates, cfg.num_classes,
                         input_mode=in_mode, embed_dim=args.embed_dim).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
ce = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(loader):
    model.eval(); P, Y = [], []
    for b in loader:
        x, y = unpack(b, dev); P.append(model(x).argmax(-1).cpu()); Y.append(y.cpu())
    import numpy as np; P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    return (P == Y).mean(), f1_score(Y, P, average='macro')


best = {'va': -1, 'state': None, 'ta': 0, 'tf': 0}
for ep in range(args.num_epochs):
    model.train()
    for b in tl:
        x, y = unpack(b, dev); opt.zero_grad(); ce(model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sched.step()
    va, vf = evaluate(vl); ta, tf = evaluate(el)
    if va > best['va']:
        best = {'va': va, 'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}, 'ta': ta, 'tf': tf}
    print(f'ep{ep} val_acc={va:.4f} test_acc={ta:.4f}', flush=True)

exp_dir = os.path.join(args.results_dir, args.exp_name); os.makedirs(exp_dir, exist_ok=True)
torch.save(best['state'], os.path.join(exp_dir, 'best_model.pth'))
json.dump({'dataset': args.dataset, 'modalities': SUB, 'num_classes': cfg.num_classes, 'fold': args.fold,
           'input_mode': in_mode, 'embed_dim': args.embed_dim,
           'best_val_acc': float(best['va']), 'best_test_acc': float(best['ta']), 'best_test_f1': float(best['tf'])},
          open(os.path.join(exp_dir, 'config.json'), 'w'), indent=2)
print(f"[expert] {args.dataset} {SUB} f{args.fold} best_val={best['va']:.4f} test={best['ta']:.4f}/{best['tf']:.4f} -> {exp_dir}")
