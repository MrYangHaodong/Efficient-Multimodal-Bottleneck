"""seqA adaptive early-exit on CMU-MOSEI (FROZEN-BERT regime) via the
DISCRETIZE-FOR-DECISION scheme: train seqA as a 7-class sentiment classifier
(Acc-7 bins), so each prefix has a softmax -> use its ENTROPY as the per-sample
exit signal. Modalities processed in val-LOO order (text->vision->audio); exit
after the first k whose entropy <= tau. Reports the exit curve (meanK, Acc-7,
Acc-2) so it can be plotted vs per-K GFLOPs like the GloVe-regime exit.

Run from clean_models/train/:  python seqA_frozen_cls7_exit.py --cuda_pick cuda:0
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.CMU_MOSEI.get_data_mmsa import get_mmsa_dataloader, FEATURE_DIMS
from models.seqA import (
    DualVideoBottleneckModelV6Downsample)

ap = argparse.ArgumentParser()
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--num_epochs', type=int, default=30)
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', default='./results_mosei_mmsa_frozen')
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed); np.random.seed(args.seed)
MODS = ['vision', 'audio', 'text']; M = 3; T = 50; NUMC = 7    # 7 sentiment bins
TAUS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 10.0]


def to_bin(reg):                       # [-3,3] float -> {0..6}
    return torch.clamp(torch.round(reg), -3, 3).long() + 3


class _Cfg:
    modalities = list(MODS); variates = dict(FEATURE_DIMS)   # text=768 frozen
    video_high_dim = 35; video_low_dim = 35; num_classes = NUMC


def build():
    return DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=_Cfg(), output_dim=NUMC, input_length=T, d_model=128, nhead=8,
        num_layers_per_modal=3, num_layers=3, dropout=0.1, verbose=False,
        video_low_dim=35, video_high_dim=35, use_bottleneck=True, n_bottlenecks=4, fusion_layer=0,
        use_sparse_moe=False, num_experts=4, expert_k=1, internal_dim=128, bottleneck_head_pos=True,
        use_sparse_attn=False, factor=10, selector_video_source='high', encoder_video_source='high',
        no_selector=True, use_weighted_factor=False, use_triton=False, num_classes=NUMC,
        sparse_attn_variant='opt', strat_block_size=8, downsample_min_len=64, use_batched_fusion=True,
        per_modal_distill=False, per_modal_downsample_min_len=4, fusion_add_pos_embeds=False,
        fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full', fusion_mlp_ratio=1.0,
        fusion_mode='sequential', seq_depth_flow=False, seq_modality_order=None, seq_random_order=False,
        bottleneck_agg_mode='gate', n_fusion_distill=-1).to(dev)


def fwd(model, high, keep):
    h = {m: high[m] for m in keep}
    o = model(h, h, training=model.training, return_selection_info=False)
    return o[0] if isinstance(o, tuple) else o


tr, va, te = get_mmsa_dataloader(batch_size=args.batch_size, num_workers=4, text_mode='frozen')


def preload(ld):
    out = []
    for txt, v, a, y in ld:
        out.append(({'vision': v.to(dev), 'audio': a.to(dev), 'text': txt.to(dev)}, y))
    return out


val_b, test_b = preload(va), preload(te)
model = build()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
ce = nn.CrossEntropyLoss()
print(f'[seqA-frozen-cls7] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')


def acc2_acc7(pred_bin, true_reg):                 # pred_bin {0..6}, true_reg float
    sent = pred_bin.astype(float) - 3.0
    acc7 = float((pred_bin == (np.clip(np.round(true_reg), -3, 3) + 3)).mean())
    nz = true_reg != 0
    a2 = float(((sent[nz] > 0) == (true_reg[nz] > 0)).mean())
    return a2, acc7


@torch.no_grad()
def eval_full(batches, keep=MODS):
    model.eval(); P, Y = [], []
    for high, y in batches:
        P.append(fwd(model, high, keep).argmax(-1).cpu().numpy()); Y.append(y.numpy())
    return acc2_acc7(np.concatenate(P), np.concatenate(Y))


# ---- train (CE on 7-bin labels; select best val non0-acc2) ----
best = -1; best_state = None
for ep in range(args.num_epochs):
    model.train()
    for txt, v, a, y in tr:                     # iterate loader directly (shuffled, per-batch to GPU)
        high = {'vision': v.to(dev), 'audio': a.to(dev), 'text': txt.to(dev)}
        logits = fwd(model, high, MODS)
        loss = ce(logits, to_bin(y).to(dev))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    a2, a7 = eval_full(val_b)
    if ep % 3 == 0 or ep == args.num_epochs - 1:
        print(f'  ep{ep:02d} val: acc2={a2:.4f} acc7={a7:.4f}')
    if a2 > best:
        best = a2; best_state = copy.deepcopy(model.state_dict())
model.load_state_dict(best_state); model.eval()


# ---- val-LOO modality order (strong-first) ----
full = eval_full(val_b, MODS)[0]
loo = {m: full - eval_full(val_b, [x for x in MODS if x != m])[0] for m in MODS}
order = sorted(MODS, key=lambda m: -loo[m])
print(f'\nval-LOO acc2 drops: ' + ', '.join(f'{m}={loo[m]:+.3f}' for m in order) + f'  -> order {order}')


# ---- prefix softmax entropy on TEST, in val-LOO order ----
@torch.no_grad()
def prefix(batches, order):
    by, Y = [], []
    for k in range(1, M + 1):
        outs = []
        for high, y in batches:
            outs.append(fwd(model, high, order[:k]).cpu())
            if k == 1:
                Y.append(y)
        by.append(torch.cat(outs))
    L = torch.stack(by, 1); P = F.softmax(L, -1)
    H = -(P * P.clamp_min(1e-9).log()).sum(-1).numpy()     # (N, M) entropy per prefix
    preds = L.argmax(-1).numpy()                           # (N, M) bin per prefix
    return torch.cat(Y).numpy(), H, preds


true_reg, H, preds = prefix(test_b, order)
print('\n=== discretize-for-decision EXIT CURVE (frozen-BERT seqA, entropy threshold) ===')
print(f"  {'tau':>5} {'meanK':>6} {'Acc-2':>6} {'Acc-7':>6}")
curve = []
for tau in TAUS:
    N = len(true_reg); ek = np.full(N, M)
    for k in range(M):
        ek = np.where((ek == M) & (H[:, k] <= tau), k + 1, ek)
    pe = preds[np.arange(N), ek - 1]
    a2, a7 = acc2_acc7(pe, true_reg)
    curve.append((float(ek.mean()), a2, a7))
    print(f"  {tau:5.2f} {ek.mean():6.2f} {a2:6.4f} {a7:6.4f}")

exit_curve = sorted(curve)
out = {'order': order, 'loo': {m: float(loo[m]) for m in MODS},
       'exit_curve_meanK_acc2_acc7': exit_curve,
       'full_acc2': exit_curve[-1][1], 'full_acc7': exit_curve[-1][2],
       'k1_acc2': exit_curve[0][1], 'k1_acc7': exit_curve[0][2]}
os.makedirs(args.results_dir, exist_ok=True)
json.dump(out, open(os.path.join(args.results_dir, 'seqA_frozen_cls7_exit.json'), 'w'), indent=2)
print(f"\nfull(M=3): Acc-2={exit_curve[-1][1]:.4f} Acc-7={exit_curve[-1][2]:.4f} | "
      f"k=1(text): Acc-2={exit_curve[0][1]:.4f} Acc-7={exit_curve[0][2]:.4f}")
print('saved -> results_mosei_mmsa_frozen/seqA_frozen_cls7_exit.json')
