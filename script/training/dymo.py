"""DyMo (dymo_proto) — faithful DyMo prototype baseline on the cross-modal
Transformer fusion backbone, complete-data EFFICIENCY setting. ONE trainer for all
6 datasets via --dataset.

Recipe (preserved from the original main_dymo_proto.py):
  * mask-augmented training: each batch samples ``K`` random present-subsets and
    runs the multi-mask forward (forward_train -> logits + L2-normalised feature);
  * loss = CrossEntropy + pt_rate * PrototypeLoss(cosine), prototype loss switched
    on after ``--pt_epoch``;
  * per-epoch prototype recompute as the class-mean of projected cls features;
  * AFTER training: per-subset Gaussian estimation on the TRAIN loader, then the
    greedy per-sample reward selection at test time.
Best-VAL model selection (full-modality val acc); test reported at best-val.

  python training/dymo.py --dataset iemocap  --fold 0 --results_dir ./results_dymo
  python training/dymo.py --dataset daliahar --fold 0 --results_dir ./results_dymo
  python training/dymo.py --dataset eav      --fold 0 --results_dir ./results_dymo
  python training/dymo.py --dataset dsads    --fold 0 --results_dir ./results_dymo  # fold = scenario
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ee_data import build_loaders
from multimodal_model.dymo import (CrossModalDyMoProto, PrototypeLoss,
                         estimate_subset_gaussian, proto_greedy_select)
from multimodal_model.crossmodal_transformer import CrossModalTransformer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['iemocap', 'daliahar', 'pamap2', 'dsads', 'eav', 'mosei', 'meld', 'cmi', 'czu_mhad', 'mmfi', 'utd_mhad', 'ch_simsv2'])
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--num_epochs', type=int, default=100)
ap.add_argument('--batch_size', type=int, default=None)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', default='dymo')
ap.add_argument('--max_seq_len', type=int, default=128)
ap.add_argument('--time_compression_ratio', type=int, default=4)
ap.add_argument('--word_length', type=int, default=1)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--split_mode', default='cross_subject', choices=['random', 'cross_subject'])
# DyMo-specific knobs
ap.add_argument('--K', type=int, default=4, help='mask-augmentation count (# random subsets per batch)')
ap.add_argument('--pt_epoch', type=int, default=5, help='start prototype loss after this epoch')
ap.add_argument('--pt_rate', type=float, default=1.0, help='prototype-loss weight')
ap.add_argument('--projection_dim', type=int, default=16)
ap.add_argument('--proto_temperature', type=float, default=0.1)
ap.add_argument('--d_model', type=int, default=64)
ap.add_argument('--nhead', type=int, default=8)
ap.add_argument('--num_enc_layers', type=int, default=3)
ap.add_argument('--num_fusion_layers', type=int, default=3)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--drop_warmup', type=int, default=0)
ap.add_argument('--drop_ramp', type=int, default=0)
ap.add_argument('--max_drop_prob', type=float, default=0.0)
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

tl, vl, el, cfg, in_len, in_mode, bs, unpack = build_loaders(args)

_mslices = {}; _off = 0
for _m in cfg.modalities:
    _n = cfg.variates[_m]; _mslices[_m] = (_off, _off + _n); _off += _n

def _drop_sched(ep, dw, dr, mp):
    if ep < dw or mp == 0.0: return 0.0
    if ep < dw + dr: return mp * (ep - dw) / dr
    return mp

def _apply_mod_dropout(x, dp):
    if dp <= 0.0: return x
    x = x.clone(); B = x.size(0); mods = list(_mslices)
    for b in range(B):
        drops = [m for m in mods if torch.rand(1).item() < dp]
        if len(drops) == len(mods): drops = drops[:-1]
        for m in drops:
            s, e = _mslices[m]; x[b, :, s:e] = 0.0
    return x

# Cross-modal Transformer backbone (per-modality encoders + cross-modal fusion).
backbone_kwargs = dict(cfg=cfg, input_length=in_len, input_mode=in_mode,
                       d_model=args.d_model, nhead=args.nhead,
                       num_enc_layers=args.num_enc_layers,
                       num_fusion_layers=args.num_fusion_layers, dropout=0.1)
if in_mode == 'sax':
    backbone_kwargs.update(alphabet_size=20, token_dim=16)
inner = CrossModalTransformer(**backbone_kwargs)
proto = CrossModalDyMoProto(inner, num_classes=cfg.num_classes,
                            projection_dim=args.projection_dim,
                            proto_temperature=args.proto_temperature).to(dev).float()

opt = torch.optim.AdamW(proto.parameters(), lr=args.lr, weight_decay=1e-4)
from torch.optim.lr_scheduler import CosineAnnealingLR
sched = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
ce = nn.CrossEntropyLoss()
criterion_pt = PrototypeLoss(temperature=args.proto_temperature, metric_name='cosine_similarity').to(dev)
print(f'[DyMo] {args.dataset} M={cfg.modalities} num_subsets={len(proto.id2subset)} '
      f'proj_dim={proto.proj_dim} K={args.K} pt_rate={args.pt_rate} '
      f'params={sum(p.numel() for p in proto.parameters())/1e6:.2f}M')


def train_epoch(epoch):
    """Mask-augmented DyMo train epoch + per-epoch prototype recompute."""
    proto.train()
    C, D = proto.num_classes, proto.proj_dim
    id2subset = proto.id2subset
    n_sub = len(id2subset)
    prototypes = proto.prototypes.clone().detach()                          # frozen for this epoch
    prototypes_sum = torch.zeros(C, D, device=dev)
    prototypes_count_sum = torch.zeros(C, 1, device=dev)
    for batch in tl:
        x, y = unpack(batch, dev)
        x = _apply_mod_dropout(x, _drop_sched(epoch, args.drop_warmup, args.drop_ramp, args.max_drop_prob))
        B = y.size(0)
        sample_idx = torch.randperm(n_sub)[:args.K]
        loss_ce_list, loss_pt_list = [], []
        for sid in sample_idx.tolist():
            mask = torch.tensor(id2subset[int(sid)], dtype=torch.bool, device=dev)
            mask = mask.unsqueeze(0).expand(B, -1)
            logits, feat = proto.forward_train(x, mask)
            loss_ce_list.append(ce(logits, y))
            label = F.one_hot(y, num_classes=C).float()
            loss_pt_list.append(criterion_pt(label, prototypes, feat))
            with torch.no_grad():
                class_sum, class_count = proto.cal_prototypes(label.detach(), feat.detach())
                prototypes_sum += class_sum
                prototypes_count_sum += class_count
        loss_ce = sum(loss_ce_list) / len(loss_ce_list)
        loss_pt = sum(loss_pt_list) / len(loss_pt_list)
        loss = loss_ce + (args.pt_rate * loss_pt if epoch >= args.pt_epoch else 0.0)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(proto.parameters(), 1.0); opt.step()
    # per-epoch prototype recompute (class mean of projected cls features)
    proto.prototypes.data.copy_(prototypes_sum / prototypes_count_sum.clamp(min=1))


@torch.no_grad()
def eval_full(loader):
    """Full-modality (all-present) eval -> (acc, macro-f1)."""
    proto.eval(); P, Y = [], []
    M = proto.M
    for batch in loader:
        x, y = unpack(batch, dev)
        mask = torch.ones(y.size(0), M, dtype=torch.bool, device=dev)
        logits, _ = proto.forward_train(x, mask)
        P.append(logits.argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return float((p == y).mean()), float(f1_score(y, p, average='macro'))


@torch.no_grad()
def eval_select(loader, sg):
    """DyMo greedy per-sample selection -> (acc, macro-f1, avg_selected)."""
    proto.eval(); P, Y = [], []
    n_sel_sum, n_sel_cnt = 0.0, 0
    for batch in loader:
        x, y = unpack(batch, dev)
        logits, keep = proto_greedy_select(proto, x, sg, dev)
        P.append(logits.argmax(-1).cpu().numpy()); Y.append(y.cpu().numpy())
        n_sel_sum += keep.float().sum(dim=1).sum().item(); n_sel_cnt += y.size(0)
    p, y = np.concatenate(P), np.concatenate(Y)
    return (float((p == y).mean()), float(f1_score(y, p, average='macro')),
            n_sel_sum / max(n_sel_cnt, 1))


best, best_state = -1, None
for ep in range(args.num_epochs):
    train_epoch(ep)
    sched.step()
    va, vf = eval_full(vl)
    if ep % 5 == 0 or ep == args.num_epochs - 1:
        print(f'  ep{ep:03d} val acc={va:.4f} f1={vf:.4f} pt_on={ep >= args.pt_epoch}')
    if va > best:
        best = va; best_state = copy.deepcopy(proto.state_dict())

proto.load_state_dict(best_state)

# per-subset Gaussian on the TRAIN loader (uses the loaded-best prototypes)
print('Estimating per-subset Gaussians on the TRAIN loader ...')
sg = estimate_subset_gaussian(proto, tl, dev, unpack)

ta, tf = eval_full(el)                                  # full-modality test
sa, sf, avg_sel = eval_select(el, sg)                   # DyMo greedy selection test

exp_dir = os.path.join(args.results_dir, args.exp_name)
os.makedirs(exp_dir, exist_ok=True)
torch.save(best_state, os.path.join(exp_dir, f'best_model_fold{args.fold}.pth'))
res = {'dataset': args.dataset, 'fold': args.fold, 'method': 'dymo',
       'model_class': 'CrossModalDyMoProto',
       'best_val_acc': best, 'best_test_acc': ta, 'best_test_f1': tf,
       'select_test_acc': sa, 'select_test_f1': sf, 'avg_selected': avg_sel,
       'K': args.K, 'pt_epoch': args.pt_epoch, 'pt_rate': args.pt_rate,
       'projection_dim': args.projection_dim, 'proto_temperature': args.proto_temperature}
json.dump(res, open(os.path.join(exp_dir, f'results_fold{args.fold}.json'), 'w'), indent=2)
print(f'[DyMo] {args.dataset} TEST full acc={ta:.4f} f1={tf:.4f} | '
      f'select acc={sa:.4f} f1={sf:.4f} avg_selected={avg_sel:.2f}/{proto.M} '
      f'(best val {best:.4f}) -> {exp_dir}')
