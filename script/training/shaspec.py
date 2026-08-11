"""ShaSpec (Shared-Specific) baseline trainer for all 6 datasets via --dataset.

Replaces the 7 per-dataset scripts
``training/baselines/main_shaspec_{daliahar,dsads,eav,iemocap,mosei,pamap2,wesad}_dyna.py``
(wesad dropped -> 6 datasets: iemocap daliahar pamap2 dsads eav mosei).

Backbone: GenericShaSpecClassifier (shared encoder tied across modalities +
per-modality specific encoders + strict compositional fusion + domain head).
Special loss PRESERVED (shaspec_losses):
    total = task_CE
            + alpha * shared_consistency  (L1 alignment of shared features across modalities)
            + beta  * specific_domain_CE  (per-modality domain classification on specifics)
A constant per-modality dropout (--mod_drop) is applied during training so the
shared/specific split is exercised under missing modalities. Mirrors
``training/baselines/main_mmee_4ds.py``.

  python training/shaspec.py --dataset iemocap  --fold 0 --results_dir ./results_shaspec
  python training/shaspec.py --dataset daliahar --fold 0 --results_dir ./results_shaspec
"""
from __future__ import annotations
import os, sys, argparse, json, warnings; warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ee_data import build_loaders
from multimodal_model.shaspec import GenericShaSpecClassifier, shaspec_losses

try:
    from utils.helper_function import set_seed
except Exception:  # pragma: no cover - fallback if helper unavailable
    def set_seed(seed):
        torch.manual_seed(seed); np.random.seed(seed)

ap = argparse.ArgumentParser(description='ShaSpec baseline (6 datasets).')
ap.add_argument('--dataset', required=True,
                choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei', 'meld', 'cmi', 'czu_mhad', 'mmfi', 'utd_mhad', 'ch_simsv2'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='shaspec')
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=1)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject',
                choices=['random', 'cross_subject'])
# ShaSpec hyper-parameters (unified defaults across datasets).
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--dropout', type=float, default=0.2)
ap.add_argument('--alpha', type=float, default=0.1, help='Shared-consistency loss weight')
ap.add_argument('--beta', type=float, default=0.02, help='Specific-domain CE loss weight')
ap.add_argument('--mod_drop', type=float, default=0.4,
                help='Per-modality dropout prob during training')
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)
model = GenericShaSpecClassifier(
    cfg, in_len, input_mode=in_mode,
    d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
    dropout=args.dropout).to(dev).float()
mod_drop_probs = {m: args.mod_drop for m in cfg.modalities} if args.mod_drop > 0 else None

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
print(f'[ShaSpec] {args.dataset} M={cfg.modalities} in_mode={in_mode} '
      f'params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')


@torch.no_grad()
def evaluate(loader):
    model.eval(); P, Y = [], []
    for batch in loader:
        x, y = unpack(batch, dev)
        out = model(x, modality_dropout_probs=None, training=False)
        P.append(out['logits'].argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


best, best_state = -1.0, None
for ep in range(args.num_epochs):
    model.train()
    for batch in tl:
        x, y = unpack(batch, dev)
        out = model(x, modality_dropout_probs=mod_drop_probs, training=True)
        loss, _ = shaspec_losses(out, y, alpha=args.alpha, beta=args.beta)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sched.step()
    va, vf = evaluate(vl)
    if ep % 5 == 0 or ep == args.num_epochs - 1:
        print(f'  ep{ep:03d} val acc={va:.4f} f1={vf:.4f}')
    if va > best:
        best = va; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
ta, tf = evaluate(el)
exp_dir = os.path.join(args.results_dir, args.exp_name)
os.makedirs(exp_dir, exist_ok=True)
torch.save(best_state, os.path.join(exp_dir, f'best_model_fold{args.fold}.pth'))
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'shaspec',
       'best_val_acc': best, 'best_test_acc': ta, 'best_test_f1': tf}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
print(f'[ShaSpec] {args.dataset} TEST acc={ta:.4f} f1={tf:.4f} -> {exp_dir}')
