"""Numerical-equivalence gate: per-prefix logits from ONE full forward (return_prefix_logits)
must equal separate k-only forwards (the inference patience path). Uses a gap_dual daliahar model."""
import glob, json, torch
from torch.utils.data import DataLoader
from compute_vlo_vs_random import build_har, sub_har, dev
from utils.dataset_cfg import DaliaHAR
from data.dataset_builder import HARDataset

cfg = DaliaHAR(); MODS = list(cfg.modalities); M = len(MODS)
variates = dict(cfg.variates); NUMC = cfg.num_classes; T = 256
sl, off = {}, 0
for m in MODS:
    sl[m] = (off, off + cfg.variates[m]); off += cfg.variates[m]
seqa = sorted(glob.glob('../results_3p3/seqA_daliahar/2026-06-21_DaliaHAR_seqA_gap_dual_seqA_randord'))[0]
C = json.load(open(seqa + '/config.json'))
m = build_har(C, MODS, variates, T, NUMC, C['num_layers_per_modal'])
sd = torch.load(seqa + '/best_model_loo_fold0.pth', map_location='cpu', weights_only=True)
m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')}, strict=False); m.eval()

SAX = {'alphabet_size': 20, 'word_length': 1}
loader = DataLoader(HARDataset('/files1/haodong/data/processed_dalia_activity', MODS, cfg.val_set, cfg, 'sax', sax_params=SAX),
                    batch_size=32, shuffle=False, num_workers=2)
x, y = next(iter(loader)); x = x.to(dev).float()

with torch.no_grad():
    high, low = sub_har(x, MODS, sl)                      # full, modalities in canonical order
    out_full, prefix_logits = m(high, low, training=False, return_selection_info=False, return_prefix_logits=True)
    print('prefix_logits shape:', tuple(prefix_logits.shape), '| M =', M)
    # k-only forwards (inference path), same modality order
    maxabs = []
    for k in range(1, M + 1):
        keep = MODS[:k]
        h = {mm: high[mm] for mm in keep}; l = {mm: low[mm] for mm in keep}
        ok = m(h, l, training=False, return_selection_info=False)
        ok = ok[0] if isinstance(ok, tuple) else ok
        diff = (ok - prefix_logits[:, k - 1]).abs().max().item()
        maxabs.append(diff)
        print(f'  k={k}: max|prefix_from_full - k_only| = {diff:.3e}')
    full_eq = (out_full - prefix_logits[:, -1]).abs().max().item()
    print(f'prefix_logits[:,-1] vs output: max abs = {full_eq:.3e}')
    print('PASS' if max(maxabs) < 1e-3 and full_eq < 1e-4 else 'FAIL (mismatch!)')
