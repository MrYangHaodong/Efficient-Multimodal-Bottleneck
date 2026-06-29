"""crossattn / multimodn / shaspec / decalign baselines on CMU-MOSEI in the
MMSA / BERT regime (matches DecAlign / MMSA). Finetuned BERT text encoder ->
(B,50,768); concat [vision35, audio74, text768] -> baseline (output_dim=1) ->
MSE regression. Reports Acc-2(has0/non0)/Acc-7/MAE/Corr. Mirrors
main_v6_mosei_mmsa_bert.py for the seqA case.

Run from clean_models/train/:
  python main_baselines_mosei_mmsa_bert.py --model multimodn --cuda_pick cuda:1
"""
from __future__ import annotations
import argparse, copy, glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.CMU_MOSEI.get_data_mmsa import get_mmsa_dataloader, mosei_metrics, FEATURE_DIMS
from transformers import BertModel

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, choices=['crossattn', 'multimodn', 'shaspec', 'decalign'])
ap.add_argument('--cuda_pick', default='cuda:0')
ap.add_argument('--num_epochs', type=int, default=10)
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--bert_lr', type=float, default=1e-5)
ap.add_argument('--weight_decay', type=float, default=1e-3)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--results_dir', default='./results_mosei_mmsa')
args = ap.parse_args()
dev = torch.device(args.cuda_pick if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed); np.random.seed(args.seed)
MODS = ['vision', 'audio', 'text']; T = 50


class _Cfg:
    def __init__(s):
        s.modalities = list(MODS)
        s.variates = dict(FEATURE_DIMS)        # text=768
        s.num_classes = 1                      # regression


def _glove_cfg(model):
    d = sorted(glob.glob(f'results_mosei_3seed/2026-*_CMU_MOSEI_{model}*acc2_s42*'))
    d = [x for x in d if 'fixed' in x or model != 'crossattn']
    return json.load(open(os.path.join(sorted(d)[-1], 'config.json')))


def build_baseline(model, cj):
    cfg = _Cfg()
    if model == 'crossattn':
        from multimodal_model.cross_attn_v6_clf import CrossAttnV6Clf
        return CrossAttnV6Clf(cfg=cfg, num_classes=1, input_length=T, d_model=cj['d_model'],
            nhead=cj['nhead'], num_layers_per_modal=cj['num_layers_per_modal'], num_layers=cj['num_layers'],
            dropout=cj['dropout'], base_factor=cj['base_factor'], num_experts=cj['num_experts'],
            use_sparse_attn=cj['use_sparse_attn'], sparse_attn_variant=cj['sparse_attn_variant'],
            strat_block_size=cj['strat_block_size'], per_modal_distill=cj['per_modal_distill'],
            per_modal_downsample_min_len=cj['per_modal_downsample_min_len'], verbose=False)
    if model == 'multimodn':
        from multimodal_model.multimodn_baseline import GenericMultiModNClassifier
        return GenericMultiModNClassifier(cfg=cfg, input_length=T, input_mode='float',
            d_model=cj['d_model'], nhead=cj['nhead'], num_layers=cj['num_layers'],
            num_layers_fus=cj['num_layers_fus'], num_layers_pred=cj['num_layers_pred'],
            dropout=cj['dropout'], state_size=(cj.get('state_size') or None))
    if model == 'shaspec':
        from multimodal_model.shaspec_baseline import GenericShaSpecClassifier
        return GenericShaSpecClassifier(cfg=cfg, input_length=T, input_mode='float',
            d_model=cj['d_model'], nhead=cj['nhead'], num_layers=cj['num_layers'], dropout=cj['dropout'])
    if model == 'decalign':
        from multimodal_model.decalign_baseline import build_decalign
        m = build_decalign(cfg=cfg, input_length=T, d_model=cj['d_model'], nhead=cj['nhead'],
            num_layers=cj['num_layers'], num_prototypes=cj['num_prototypes'], ot_reg=cj['ot_reg'],
            ot_num_iters=cj['ot_num_iters'], conv1d_kernel=cj['conv1d_kernel'], dropout=cj['dropout'],
            alpha1=cj['alpha1'], alpha2=cj['alpha2'], transform=None, verbose=False)
        z = torch.tensor(0.0, device=dev)
        m.compute_decoupling_loss = m.compute_hetero_loss = m.compute_homo_loss = lambda *a, **k: z
        return m


def _pred(out):
    if isinstance(out, dict):
        out = out.get('logits', next(iter(out.values())))
    elif isinstance(out, (tuple, list)):
        out = out[0]
    return out.squeeze(-1)


class Baseline_BERT(nn.Module):
    def __init__(s, model, cj):
        super().__init__()
        s.model_name = model
        s.bert = BertModel.from_pretrained('bert-base-uncased')
        s.net = build_baseline(model, cj)

    def forward(s, text_bert, vision, audio):
        ids, mask, tt = text_bert[:, 0], text_bert[:, 1], text_bert[:, 2]
        txt = s.bert(input_ids=ids, attention_mask=mask, token_type_ids=tt).last_hidden_state  # (B,50,768)
        x = torch.cat([vision, audio, txt], dim=-1)         # (B,50,35+74+768)  order = MODS
        if s.model_name == 'crossattn':
            out = s.net(x, modality_dropout_prob=0.0, training=s.training)
        elif s.model_name == 'multimodn':
            out = s.net(x, modality_dropout_probs=None, training=s.training)
        else:
            out = s.net(x)
        return _pred(out)


cj = _glove_cfg(args.model)
tr, va, te = get_mmsa_dataloader(batch_size=args.batch_size, num_workers=4)
model = Baseline_BERT(args.model, cj).to(dev)
bert_p = list(model.bert.parameters())
other_p = [p for n, p in model.named_parameters() if not n.startswith('bert.')]
opt = torch.optim.AdamW([{'params': bert_p, 'lr': args.bert_lr},
                         {'params': other_p, 'lr': args.lr}], weight_decay=args.weight_decay)
mse = nn.MSELoss()
print(f'[{args.model}-MMSA-BERT] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  '
      f'lr={args.lr}/bert{args.bert_lr}')


@torch.no_grad()
def evaluate(ld):
    model.eval(); P, Y = [], []
    for tb, v, a, y in ld:
        P.append(model(tb.to(dev), v.to(dev), a.to(dev)).cpu().numpy()); Y.append(y.numpy())
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
exp_dir = os.path.join(args.results_dir, f'{args.model}_mmsa_bert_s{args.seed}'); os.makedirs(exp_dir, exist_ok=True)
json.dump({'model': args.model, 'best_val': best, 'test': tm, 'args': vars(args)},
          open(os.path.join(exp_dir, 'results.json'), 'w'), indent=2)
print(f'\n[{args.model}-MMSA-BERT] TEST non0_acc2={tm["non0_acc2"]:.4f} has0_acc2={tm["has0_acc2"]:.4f} '
      f'acc7={tm["acc7"]:.4f} MAE={tm["mae"]:.4f} corr={tm["corr"]:.4f} -> {exp_dir}')
