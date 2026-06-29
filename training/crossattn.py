"""CrossAttn — CrossAttnV6Clf fusion baseline. ONE trainer for all 6 datasets via
--dataset. The backbone is ``models.crossattn.CrossAttnV6Clf`` (standalone): it
consumes a single concatenated ``(B, T, sum(variates))`` tensor (modalities along
the feature dim) and returns ``(logits, dynamic_factor)``. We discard the factor.
Plain cross-entropy on the logits, AdamW + Cosine, grad-clip 1.0, best-VAL model
selection, test at the best-val checkpoint.

CrossAttnV6Clf projects each modality slice with a Linear, so it consumes the
concatenated tensor identically for float datasets (IEMOCAP/EAV/MOSEI) and for the
SAX HAR datasets (daliahar/pamap2/dsads) — the SAX token indices are fed as float
values exactly as in the original main_crossattn_v6_daliahar.py baseline. The model
has no separate sax embedding path, so ``input_mode`` only documents the regime.

  python training/crossattn.py --dataset iemocap  --fold 0 --results_dir ./results_crossattn
  python training/crossattn.py --dataset daliahar --fold 0 --results_dir ./results_crossattn
  python training/crossattn.py --dataset eav      --fold 0 --results_dir ./results_crossattn
  python training/crossattn.py --dataset dsads    --fold 0 --results_dir ./results_crossattn  # fold = scenario
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ee_data import build_loaders
from models.crossattn import CrossAttnV6Clf


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='crossattn')
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=2)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject', choices=['random', 'cross_subject'])
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(42)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)
model = CrossAttnV6Clf(
    cfg=cfg,
    num_classes=cfg.num_classes,
    input_length=in_len,
    verbose=True,
).to(dev).float()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
ce = nn.CrossEntropyLoss()
print(f'[CrossAttn] {args.dataset} M={cfg.modalities} in_mode={in_mode} in_len={in_len} '
      f'params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')


@torch.no_grad()
def evaluate(loader):
    model.eval(); P, Y = [], []
    for batch in loader:
        x, y = unpack(batch, dev)
        logits, _ = model(x, modality_dropout_prob=0.0, training=False)  # all modalities present
        P.append(logits.argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


best, best_state = -1, None
for ep in range(args.num_epochs):
    model.train()
    for batch in tl:
        x, y = unpack(batch, dev)
        logits, _ = model(x, modality_dropout_prob=0.0, training=True)
        loss = ce(logits, y)                               # plain CE on the logits
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sched.step()
    va, vf = evaluate(vl)
    if ep % 5 == 0 or ep == args.num_epochs - 1:
        print(f'  ep{ep:03d} val acc={va:.4f} f1={vf:.4f}')
    if va > best:
        best = va; best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
ta, tf = evaluate(el)
exp_dir = os.path.join(args.results_dir, args.exp_name)
os.makedirs(exp_dir, exist_ok=True)
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'crossattn',
       'model_class': 'CrossAttnV6Clf',
       'best_val_acc': best, 'best_test_acc': ta, 'best_test_f1': tf}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
print(f'[CrossAttn] {args.dataset} TEST acc={ta:.4f} f1={tf:.4f} (best val {best:.4f}) -> {exp_dir}')
