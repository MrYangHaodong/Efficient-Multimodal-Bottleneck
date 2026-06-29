"""TRUE GFLOPs (FlopCounterMode + analytic SDPA patch, CPU single-count) for the
crossattn/multimodn/shaspec/decalign/dymo baselines on EAV at T=128, FLOAT input.
Same method as measure_baselines_gflops_mosei.py.
Output -> results_3p3/eav/baselines_gflops_eav.json.
Run from clean_models/train/:  python ../results_3p3/measure_baselines_gflops_eav.py
"""
import glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

_HERE = os.path.dirname(os.path.abspath(__file__)); _CM = os.path.dirname(_HERE)
sys.path.insert(0, _CM)
torch.backends.mha.set_fastpath_enabled(False)
from data.EAV.get_data import FEATURE_DIMS, EAVDataset           # noqa: E402

DEV = torch.device('cpu')
T = 128
MODS = list(EAVDataset.ALL_MODALITIES)
RESROOT = os.path.join(_CM, 'train', 'results_eav')

_ATTN = {'f': 0}; _orig = F.scaled_dot_product_attention
def _patched(q, k, v, *a, **kw):
    lead = 1
    for s in q.shape[:-2]:
        lead *= s
    _ATTN['f'] += 4 * lead * q.shape[-2] * k.shape[-2] * q.shape[-1]
    return _orig(q, k, v, *a, **kw)


class _Cfg:
    def __init__(s, m, v, n):
        s.modalities, s.variates, s.num_classes = list(m), dict(v), n


def cfg_of(cj):
    mods = cj.get('modalities') or list(MODS)
    return _Cfg(mods, {m: FEATURE_DIMS[m] for m in mods}, cj.get('num_classes') or 5)


def run_dir(method):
    ds = sorted(glob.glob(os.path.join(RESROOT, method, '*')))
    for d in reversed(ds):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'config.json')):
            return d
    return None


@torch.no_grad()
def measure(fwd):
    _ATTN['f'] = 0
    fc = FlopCounterMode(display=False)
    with fc:
        fwd()
    lib = fc.get_total_flops()
    return lib / 1e9, (lib + _ATTN['f']) / 1e9


def build_fwd(method, cj, cfg, C):
    x = torch.randn(1, T, C).to(DEV)                       # EAV: float input
    if method == 'crossattn':
        from multimodal_model.cross_attn_v6_clf import CrossAttnV6Clf
        m = CrossAttnV6Clf(cfg=cfg, num_classes=cfg.num_classes, input_length=T, d_model=cj['d_model'],
            nhead=cj['nhead'], num_layers_per_modal=cj['num_layers_per_modal'], num_layers=cj['num_layers'],
            dropout=cj['dropout'], base_factor=cj['base_factor'], num_experts=cj['num_experts'],
            use_sparse_attn=cj['use_sparse_attn'], sparse_attn_variant=cj['sparse_attn_variant'],
            strat_block_size=cj['strat_block_size'], per_modal_distill=cj['per_modal_distill'],
            per_modal_downsample_min_len=cj['per_modal_downsample_min_len'], verbose=False).to(DEV).eval()
        return lambda: m(x, modality_dropout_prob=0.0, training=False)
    if method == 'multimodn':
        from multimodal_model.multimodn_baseline import GenericMultiModNClassifier
        m = GenericMultiModNClassifier(cfg=cfg, input_length=T, input_mode='float',
            d_model=cj['d_model'], nhead=cj['nhead'], num_layers=cj['num_layers'],
            num_layers_fus=cj['num_layers_fus'], num_layers_pred=cj['num_layers_pred'],
            dropout=cj['dropout'], state_size=(cj.get('state_size') or None)).to(DEV).eval()
        return lambda: m(x, modality_dropout_probs=None, training=False)
    if method == 'decalign':
        from multimodal_model.decalign_baseline import build_decalign
        m = build_decalign(cfg=cfg, input_length=T, d_model=cj['d_model'], nhead=cj['nhead'],
            num_layers=cj['num_layers'], num_prototypes=cj['num_prototypes'], ot_reg=cj['ot_reg'],
            ot_num_iters=cj['ot_num_iters'], conv1d_kernel=cj['conv1d_kernel'], dropout=cj['dropout'],
            alpha1=cj['alpha1'], alpha2=cj['alpha2'], transform=None, verbose=False).to(DEV).eval()
        z = torch.tensor(0.0)
        m.compute_decoupling_loss = m.compute_hetero_loss = m.compute_homo_loss = lambda *a, **k: z
        return lambda: m(x)
    if method == 'shaspec':
        from multimodal_model.shaspec_baseline import GenericShaSpecClassifier
        m = GenericShaSpecClassifier(cfg=cfg, input_length=T, input_mode='float',
            d_model=cj['d_model'], nhead=cj['nhead'], num_layers=cj['num_layers'],
            dropout=cj['dropout']).to(DEV).eval()
        return lambda: m(x)
    if method == 'dymo':
        from multimodal_model.crossmodal_transformer import CrossModalTransformer
        m = CrossModalTransformer(cfg=cfg, input_length=T, input_mode='float',
            d_model=cj.get('d_model', 128), nhead=cj.get('nhead', 8),
            num_enc_layers=cj.get('num_enc_layers', 3), num_fusion_layers=cj.get('num_fusion_layers', 3),
            dropout=cj.get('dropout', 0.1)).to(DEV).eval()
        mask = torch.ones(1, len(cfg.modalities), dtype=torch.bool, device=DEV)
        return lambda: m(x, mask)


F.scaled_dot_product_attention = _patched
print(f"{'method':10s} {'lib':>9s} {'TRUE(T128)':>11s}")
out = {}
for method in ['crossattn', 'multimodn', 'decalign', 'shaspec', 'dymo']:
    rd = run_dir(method)
    if rd is None:
        print(f"{method:10s}  <no run dir>"); continue
    cj = json.load(open(os.path.join(rd, 'config.json')))
    cfg = cfg_of(cj); C = sum(cfg.variates[m] for m in cfg.modalities)
    try:
        lib, true = measure(build_fwd(method, cj, cfg, C))
        out[method] = {'eav': {'lib_gflops': lib, 'true_gflops': true}}
        print(f"{method:10s} {lib:9.4f} {true:11.4f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{method:10s}  ERROR: {e}")
F.scaled_dot_product_attention = _orig
os.makedirs(os.path.join(_HERE, 'eav'), exist_ok=True)
json.dump(out, open(os.path.join(_HERE, 'eav', 'baselines_gflops_eav.json'), 'w'), indent=2)
print('saved -> results_3p3/eav/baselines_gflops_eav.json')
