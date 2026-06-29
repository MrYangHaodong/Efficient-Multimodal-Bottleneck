"""seqA on CMU-MOSEI in the MMSA / BERT regime (matches DecAlign / MMSA setup).

Data : MMSA aligned_50.pkl (text_bert tokens + FACET vision35 + COVAREP audio74).
Text : BERT-base-uncased, FINETUNED (lr 1e-5), -> (B,50,768) token features.
Model: DualVideoBottleneckModelV6Downsample (our seqA), text modality dim = 768,
       output_dim = 1 (regression). MSE loss. Report Acc-2(has0/non0)/Acc-7/MAE/Corr.
Selection: best val non0_acc2 (deepcopy). Run from clean_models/train/.
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.CMU_MOSEI.get_data_mmsa import get_mmsa_dataloader, mosei_metrics, FEATURE_DIMS
from multimodal_model.v6_downsample_opt_batched_late_fusion_seqfusion import (
    DualVideoBottleneckModelV6Downsample)
from transformers import BertModel

ap = argparse.ArgumentParser()
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--num_epochs', type=int, default=12)
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--lr', type=float, default=5e-5)          # seqA + projections
ap.add_argument('--bert_lr', type=float, default=1e-5)     # finetuned BERT
ap.add_argument('--weight_decay', type=float, default=1e-3)
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--n_bottlenecks', type=int, default=4)
ap.add_argument('--num_layers', type=int, default=3)
ap.add_argument('--num_layers_per_modal', type=int, default=3)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', default='./results_mosei_mmsa')
ap.add_argument('--exp_name', default='seqA_mmsa_bert')
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed); np.random.seed(args.seed)
MODS = ['vision', 'audio', 'text']; T = 50


class _Cfg:
    modalities = list(MODS)
    variates = dict(FEATURE_DIMS)            # text=768
    video_high_dim = FEATURE_DIMS['vision']; video_low_dim = FEATURE_DIMS['vision']
    num_classes = 1


class SeqA_BERT(nn.Module):
    """BERT text encoder + seqA regression head."""
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.seqa = DualVideoBottleneckModelV6Downsample(
            head_mode='gap', cfg=_Cfg(), output_dim=1, input_length=T,
            d_model=args.d_model, nhead=8, num_layers_per_modal=args.num_layers_per_modal,
            num_layers=args.num_layers, dropout=0.1, verbose=False,
            video_low_dim=FEATURE_DIMS['vision'], video_high_dim=FEATURE_DIMS['vision'],
            use_bottleneck=True, n_bottlenecks=args.n_bottlenecks, fusion_layer=0,
            use_sparse_moe=False, num_experts=4, expert_k=1, internal_dim=args.d_model,
            bottleneck_head_pos=True, use_sparse_attn=False, factor=10,
            selector_video_source='high', encoder_video_source='high', no_selector=True,
            use_weighted_factor=False, use_triton=False, num_classes=1,
            sparse_attn_variant='opt', strat_block_size=8, downsample_min_len=64,
            use_batched_fusion=True, per_modal_distill=False, per_modal_downsample_min_len=4,
            fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full',
            fusion_mlp_ratio=1.0, fusion_mode='sequential', seq_depth_flow=False,
            seq_modality_order=None, seq_random_order=False, bottleneck_agg_mode='gate',
            n_fusion_distill=-1)

    def forward(self, text_bert, vision, audio):
        ids, mask, ttype = text_bert[:, 0], text_bert[:, 1], text_bert[:, 2]
        txt = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=ttype).last_hidden_state  # (B,50,768)
        high = {'vision': vision, 'audio': audio, 'text': txt}
        out = self.seqa(high, high, training=self.training, return_selection_info=False)
        out = out[0] if isinstance(out, tuple) else out
        return out.squeeze(-1)                                # (B,)


tr, va, te = get_mmsa_dataloader(batch_size=args.batch_size, num_workers=4)
model = SeqA_BERT().to(dev)
bert_p = list(model.bert.parameters())
other_p = [p for n, p in model.named_parameters() if not n.startswith('bert.')]
opt = torch.optim.AdamW([{'params': bert_p, 'lr': args.bert_lr},
                         {'params': other_p, 'lr': args.lr}], weight_decay=args.weight_decay)
mse = nn.MSELoss()
print(f'[seqA-MMSA-BERT] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M '
      f'(bert={sum(p.numel() for p in bert_p)/1e6:.1f}M)  lr={args.lr}/bert{args.bert_lr}')


@torch.no_grad()
def evaluate(ld):
    model.eval(); P, Y = [], []
    for tb, v, a, y in ld:
        o = model(tb.to(dev), v.to(dev), a.to(dev))
        P.append(o.cpu().numpy()); Y.append(y.numpy())
    return mosei_metrics(np.concatenate(P), np.concatenate(Y))


best = {'non0_acc2': -1}; best_state = None
for ep in range(args.num_epochs):
    model.train()
    for tb, v, a, y in tr:
        o = model(tb.to(dev), v.to(dev), a.to(dev))
        loss = mse(o, y.to(dev))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    vm = evaluate(va)
    print(f'  ep{ep:02d} val: non0_acc2={vm["non0_acc2"]:.4f} has0_acc2={vm["has0_acc2"]:.4f} '
          f'acc7={vm["acc7"]:.4f} MAE={vm["mae"]:.4f} corr={vm["corr"]:.4f}')
    if vm['non0_acc2'] > best['non0_acc2']:
        best = {**vm, 'epoch': ep}; best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
tm = evaluate(te)
print('\n=== TEST (best-val ckpt) ===')
for k, v in tm.items():
    print(f'  {k:12s} {v:.4f}')

exp_dir = os.path.join(args.results_dir, f'{args.exp_name}_s{args.seed}'); os.makedirs(exp_dir, exist_ok=True)
json.dump({'best_val': best, 'test': tm, 'args': vars(args)},
          open(os.path.join(exp_dir, 'results.json'), 'w'), indent=2)
print(f'\n[seqA-MMSA-BERT] TEST non0_acc2={tm["non0_acc2"]:.4f} has0_acc2={tm["has0_acc2"]:.4f} '
      f'acc7={tm["acc7"]:.4f} MAE={tm["mae"]:.4f} corr={tm["corr"]:.4f} -> {exp_dir}')
