"""torch.compile per-sample (batch=1) GPU latency for seqA (per-K, adaptive-exit
curve) + crossattn/multimodn/shaspec/decalign baselines on CMU-MOSEI (M=3, T=50,
FLOAT). Same method as measure_latency_compile_4ds.py: compile (reduce-overhead =
CUDA graphs -> default -> eager) then CUDA-event median over many iters.
seqA per-K uses the consensus val-LOO order (text->vision->audio) from exit_mosei.json.
Output -> results_3p3/latency_compile_mosei.json.
Run from clean_models/train/:  python ../results_3p3/measure_latency_compile_mosei.py
"""
import glob, json, os, sys, statistics, warnings
warnings.filterwarnings('ignore')
import torch

_HERE = os.path.dirname(os.path.abspath(__file__)); _CM = os.path.dirname(_HERE)
sys.path.insert(0, _CM)
from data.CMU_MOSEI.get_data import FEATURE_DIMS, CMUMOSEIDataset            # noqa: E402
from models.seqA import (  # noqa: E402
    DualVideoBottleneckModelV6Downsample)

DEV = torch.device('cuda:0')
RES = os.path.join(_CM, 'train', 'results_mosei_3seed')
T = 50; MODS = list(CMUMOSEIDataset.ALL_MODALITIES); NUMC = 2
VAR = {m: FEATURE_DIMS[m] for m in MODS}
ORDER = json.load(open(os.path.join(_HERE, 'mosei', 'exit_mosei.json')))['mosei']['consensus_order']
N_WARMUP, N_ITER = 30, 200

SEQA_DIR = sorted(glob.glob(os.path.join(RES, '2026-*_seqA_lr2e5e30_acc2_s42_seqA')))[-1]
MO_PAT = {'crossattn': '2026-*_CMU_MOSEI_crossattn_fixed_acc2_s42',
          'multimodn': '2026-*_CMU_MOSEI_multimodn_acc2_s42',
          'shaspec':   '2026-*_CMU_MOSEI_shaspec_acc2_s42',
          'decalign':  '2026-*_CMU_MOSEI_decalign_acc2_s42'}


class _Cfg:
    def __init__(s, m, v, n): s.modalities, s.variates, s.num_classes = list(m), dict(v), n


def _time(fn):
    with torch.no_grad():
        for _ in range(N_WARMUP):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(N_ITER):
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def latency(fwd):
    for mode in ('reduce-overhead', 'default'):
        try:
            c = torch.compile(fwd, mode=None if mode == 'default' else mode)
            t = _time(c); torch.compiler.reset(); return t, mode
        except Exception as ex:
            print(f'    [compile {mode} failed: {type(ex).__name__}: {str(ex)[:100]}]')
            torch.compiler.reset()
    return _time(fwd), 'eager'


def build_seqa():
    C = json.load(open(os.path.join(SEQA_DIR, 'config.json')))

    class _VC:
        modalities = list(MODS); variates = dict(VAR)
        video_high_dim = VAR[MODS[0]]; video_low_dim = VAR[MODS[0]]
    m = DualVideoBottleneckModelV6Downsample(
        head_mode='gap', cfg=_VC(), output_dim=NUMC, input_length=T, d_model=C['d_model'], nhead=C['nhead'],
        num_layers_per_modal=C['num_layers_per_modal'], num_layers=C['num_layers'], dropout=C['dropout'], verbose=False,
        video_low_dim=VAR[MODS[0]], video_high_dim=VAR[MODS[0]], use_bottleneck=True, n_bottlenecks=C['n_bottlenecks'],
        fusion_layer=C['fusion_layer'], use_sparse_moe=False, num_experts=C['num_experts'], expert_k=1,
        internal_dim=C['internal_dim'], bottleneck_head_pos=True, use_sparse_attn=C['use_sparse_attn'],
        factor=C['base_factor'], selector_video_source='high', encoder_video_source='high', no_selector=True,
        use_weighted_factor=False, use_triton=False, num_classes=NUMC,
        sparse_attn_variant=C['sparse_attn_variant'], strat_block_size=C['strat_block_size'],
        downsample_min_len=C['downsample_min_len'], use_batched_fusion=C['use_batched_fusion'],
        per_modal_distill=C['per_modal_distill'], per_modal_downsample_min_len=C['per_modal_downsample_min_len'],
        fusion_add_pos_embeds=False, fusion_pos_embeds_max_len=256, fusion_pos_embed_mode='full', fusion_mlp_ratio=1.0,
        fusion_mode=C['fusion_mode'], seq_depth_flow=C['seq_depth_flow'], seq_modality_order=None,
        seq_random_order=False, bottleneck_agg_mode=C.get('bottleneck_agg_mode', 'gate'),
        n_fusion_distill=C.get('n_fusion_distill', -1)).to(DEV).float().eval()
    return m


def seqa_latency_per_k():
    m = build_seqa(); per_k, modes = [], []
    for k in range(1, len(MODS) + 1):
        keep = ORDER[:k]                                  # consensus val-LOO order
        high = {mm: torch.randn(1, T, VAR[mm]).to(DEV) for mm in keep}
        low = {mm: v.clone() for mm, v in high.items()}
        fwd = (lambda h, l: (lambda: m(h, l, training=False, return_selection_info=False)))(high, low)
        t, mode = latency(fwd); per_k.append(t); modes.append(mode)
    return per_k, modes


def build_base_fwd(method):
    cd = sorted(glob.glob(os.path.join(RES, MO_PAT[method])))[-1]
    cj = json.load(open(os.path.join(cd, 'config.json')))
    mods = cj.get('modalities') or list(MODS)
    cfg = _Cfg(mods, {m: VAR[m] for m in mods}, 2)
    C = sum(cfg.variates[m] for m in cfg.modalities)
    x = torch.randn(1, T, C).to(DEV)
    if method == 'crossattn':
        from models.crossattn import CrossAttnV6Clf
        m = CrossAttnV6Clf(cfg=cfg, num_classes=2, input_length=T, d_model=cj['d_model'], nhead=cj['nhead'],
            num_layers_per_modal=cj['num_layers_per_modal'], num_layers=cj['num_layers'], dropout=cj['dropout'],
            base_factor=cj['base_factor'], num_experts=cj['num_experts'], use_sparse_attn=cj['use_sparse_attn'],
            sparse_attn_variant=cj['sparse_attn_variant'], strat_block_size=cj['strat_block_size'],
            per_modal_distill=cj['per_modal_distill'], per_modal_downsample_min_len=cj['per_modal_downsample_min_len'],
            verbose=False).to(DEV).eval()
        return lambda: m(x, modality_dropout_prob=0.0, training=False)
    if method == 'multimodn':
        from models.multimodn import GenericMultiModNClassifier
        m = GenericMultiModNClassifier(cfg=cfg, input_length=T, input_mode='float', d_model=cj['d_model'],
            nhead=cj['nhead'], num_layers=cj['num_layers'], num_layers_fus=cj['num_layers_fus'],
            num_layers_pred=cj['num_layers_pred'], dropout=cj['dropout'],
            state_size=(cj.get('state_size') or None)).to(DEV).eval()
        return lambda: m(x, modality_dropout_probs=None, training=False)
    if method == 'decalign':
        from models.decalign import build_decalign
        m = build_decalign(cfg=cfg, input_length=T, d_model=cj['d_model'], nhead=cj['nhead'], num_layers=cj['num_layers'],
            num_prototypes=cj['num_prototypes'], ot_reg=cj['ot_reg'], ot_num_iters=cj['ot_num_iters'],
            conv1d_kernel=cj['conv1d_kernel'], dropout=cj['dropout'], alpha1=cj['alpha1'], alpha2=cj['alpha2'],
            transform=None, verbose=False).to(DEV).eval()
        z = torch.tensor(0.0, device=DEV)
        m.compute_decoupling_loss = m.compute_hetero_loss = m.compute_homo_loss = lambda *a, **k: z
        return lambda: m(x)
    if method == 'shaspec':
        from models.shaspec import GenericShaSpecClassifier
        m = GenericShaSpecClassifier(cfg=cfg, input_length=T, input_mode='float', d_model=cj['d_model'],
            nhead=cj['nhead'], num_layers=cj['num_layers'], dropout=cj['dropout']).to(DEV).eval()
        return lambda: m(x)


print(f'=== CMU-MOSEI (M={len(MODS)}, T={T}, float) — seqA order={ORDER} ===')
pk, modes = seqa_latency_per_k()
print(f'  seqA latency_per_k(ms)={[round(t, 3) for t in pk]} compile={modes}')
rec = {'M': len(MODS), 'T': T, 'seqA_latency_per_k_ms': pk, 'seqA_compile': modes}
for method in ['crossattn', 'multimodn', 'shaspec', 'decalign']:
    try:
        t, mode = latency(build_base_fwd(method))
    except Exception as ex:
        print(f'  {method}: FAILED {type(ex).__name__}: {str(ex)[:100]}'); t, mode = None, 'fail'
    rec[method] = {'latency_ms': t, 'compile': mode}
    print(f'  {method}: {t if t is None else round(t, 3)} ms ({mode})')
json.dump({'mosei': rec}, open(os.path.join(_HERE, 'latency_compile_mosei.json'), 'w'), indent=2)
print('saved -> results_3p3/latency_compile_mosei.json')
