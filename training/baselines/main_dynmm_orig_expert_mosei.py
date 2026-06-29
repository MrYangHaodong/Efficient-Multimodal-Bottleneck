"""Original-DynMM expert (MultiBench Transformer + Concat late fusion) on CMU-MOSEI, ONE branch.
Acc-2 classification (CE). Saves best_model.pth + config.json so the gate can rebuild it.
Run: python main_dynmm_orig_expert_mosei.py --modalities text vision --seed 7 --results_dir ... --exp_name ..."""
import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.CMU_MOSEI.get_data import get_dataloader, FEATURE_DIMS, NUM_CLASSES
from multimodal_model.dynmm_orig_baseline import OrigDynMMExpert
from utils.helper_function import set_seed

ALL = ['vision', 'audio', 'text']
ap = argparse.ArgumentParser()
ap.add_argument('--modalities', nargs='+', required=True)
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--results_dir', required=True)
ap.add_argument('--exp_name', required=True)
ap.add_argument('--num_epochs', type=int, default=30)
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--max_seq_len', type=int, default=50)
ap.add_argument('--tcr', type=int, default=1)            # original DynMM = full sequence (no compression)
ap.add_argument('--cuda_pick', default='cuda:0')
args = ap.parse_args()
set_seed(args.seed)
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
SUB = list(args.modalities)

tl, vl, el = get_dataloader(batch_size=args.batch_size, num_workers=4, modalities=ALL, split_id=0,
                            train_shuffle=True, max_seq_len=args.max_seq_len,
                            time_compression_ratio=args.tcr, use_batched_fusion=True, task='classification2')


def unpack(batch):
    *feats, label = batch
    fd = {m: feats[2 * j].to(dev).float() for j, m in enumerate(ALL)}
    y = label.long().to(dev); y = y.squeeze(-1) if y.ndim > 1 else y
    return fd, y


model = OrigDynMMExpert(SUB, {m: FEATURE_DIMS[m] for m in SUB}, NUM_CLASSES).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
ce = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(loader):
    model.eval(); P, Y = [], []
    for b in loader:
        fd, y = unpack(b); P.append(model(fd).argmax(-1).cpu()); Y.append(y.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    return (P == Y).mean(), f1_score(Y, P, average='macro')


best = {'va': -1, 'state': None, 'ta': 0, 'tf': 0}
for ep in range(args.num_epochs):
    model.train()
    for b in tl:
        fd, y = unpack(b); opt.zero_grad(); loss = ce(model(fd), y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sched.step()
    va, vf = evaluate(vl); ta, tf = evaluate(el)
    if va > best['va']:
        best = {'va': va, 'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}, 'ta': ta, 'tf': tf}
    print(f'ep{ep} val_acc={va:.4f} test_acc={ta:.4f}', flush=True)

exp_dir = os.path.join(args.results_dir, args.exp_name); os.makedirs(exp_dir, exist_ok=True)
torch.save(best['state'], os.path.join(exp_dir, 'best_model.pth'))
json.dump({'modalities': SUB, 'num_classes': NUM_CLASSES, 'seed': args.seed, 'tcr': args.tcr,
           'best_val_acc': float(best['va']), 'best_test_acc': float(best['ta']), 'best_test_f1': float(best['tf'])},
          open(os.path.join(exp_dir, 'config.json'), 'w'), indent=2)
print(f"[expert] {SUB} s{args.seed} best_val={best['va']:.4f} best_test={best['ta']:.4f}/{best['tf']:.4f} -> {exp_dir}")
