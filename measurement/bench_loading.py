#!/usr/bin/env python3
"""Loading-inclusive latency benchmark — storage -> RAM -> GPU -> forward.

Unlike bench_gpu.py / bench_cpu.py (which pre-load every tensor to the device
before timing), here the timer starts BEFORE anything is read from disk.  Each
timed call is the full pipeline a deployed system actually pays:

    t_start
    np.load(mod.npy)                    <- disk -> CPU RAM
    torch.from_numpy(...).to(dev)       <- CPU RAM -> GPU
    model forward                       <- GPU compute
    cuda.synchronize()
    t_end

This is where sequential acquisition earns its keep.  AdaMML, DynMM and DyMo all
need every modality present before inference begins, so they pay M file reads on
every sample.  seqA+RL only needs one modality to start: the next file is not
read until the policy has decided it wants that modality, and files it never
acquires are never read at all.

Two modes for seqA+RL (baselines are identical in both — always M loads):

  sequential  load each modality on demand, interleaved with the RL decisions.
              Only the acquired modalities are ever read.  This is the honest
              cost of a sequential-acquisition system.
  bundled     load all M modalities upfront, then walk.  Pays the same I/O as
              the baselines, so the delta between the two modes isolates exactly
              what sequential loading buys.

Run prepare_samples.py once first to write the per-modality sample files.

Usage:
  cd runs/measurement
  python prepare_samples.py --n 50           # once (--n 0 for full test sets)
  CUDA_VISIBLE_DEVICES=0 python bench_loading.py --n 50 --mode both
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from models.dynmm import OrigGateG, OrigDynMMExpertG, diff_softmax
from models.dymo import proto_greedy_select, estimate_subset_gaussian
from models.adamml import AdaMMLNet

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CKPT_DIR   = os.path.join(HERE, 'checkpoints')
DATA_DIR   = os.path.join(HERE, 'data')
SG_DIR     = os.path.join(HERE, 'dymo_sg')
REBAL_CKPT = os.path.join(CKPT_DIR, 'seqA_mmfi_rebalanced',
                          '2026-07-24_mmfi_pv3_branch_16_seqAbranch')
DROP_CKPT  = os.path.join(CKPT_DIR, 'dropout_robustness_seqA_dropaug',
                          '2026-08-09_utd_mhad_pv3_branch_16_seqAbranch')

# Stop bias in the RL walk: STOP when Q[STOP] + OURS_DELTA >= max Q[acquire].
# Set per-dataset by main() before each walk.
OURS_DELTA = 0.1


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset configuration
# ══════════════════════════════════════════════════════════════════════════════
DS_STATIC = {
    'iemocap': dict(
        mods=['video', 'audio', 'text', 'mocap_hand', 'mocap_head', 'mocap_rotated'],
        variates={'video': 1024, 'audio': 1280, 'text': 768,
                  'mocap_hand': 18, 'mocap_head': 6, 'mocap_rotated': 165},
        num_classes=4, T=100),
    'cmi': dict(
        mods=['imu', 'thm', 'tof_1', 'tof_2', 'tof_3', 'tof_4', 'tof_5'],
        variates={'imu': 7, 'thm': 5, 'tof_1': 64, 'tof_2': 64,
                  'tof_3': 64, 'tof_4': 64, 'tof_5': 64},
        num_classes=18, T=128),
    'czu_mhad': dict(
        mods=['depth', 'skeleton', 'sensor_head', 'sensor_torso',
              'sensor_rarm', 'sensor_larm', 'sensor_leg'],
        variates={'depth': 1024, 'skeleton': 100, 'sensor_head': 14, 'sensor_torso': 14,
                  'sensor_rarm': 14, 'sensor_larm': 14, 'sensor_leg': 14},
        num_classes=22, T=128),
    'mmfi': dict(
        mods=['infra1', 'wifi', 'mmwave', 'lidar', 'depth'],
        variates={'infra1': 34, 'wifi': 3420, 'mmwave': 128, 'lidar': 128, 'depth': 1024},
        num_classes=27, T=128),
    'dsads': dict(
        mods=['torso', 'right_arm', 'left_arm', 'right_leg', 'left_leg'],
        variates={'torso': 9, 'right_arm': 9, 'left_arm': 9, 'right_leg': 9, 'left_leg': 9},
        num_classes=19, T=63),
    'eav': dict(
        mods=['eeg', 'audio', 'video'],
        variates={'eeg': 30, 'audio': 768, 'video': 1024},
        num_classes=5, T=128),
    'utd_mhad': dict(
        mods=['skeleton', 'inertial', 'depth', 'rgb'],
        variates={'skeleton': 60, 'inertial': 6, 'depth': 1024, 'rgb': 1024},
        num_classes=27, T=128),
}

# pol      : policy bundle dir holding {ds}_fold{k}.pt
# delta    : OURS_DELTA stop bias for this dataset
# folds    : number of test folds
# has_*    : baseline checkpoint availability
DS_CFG = {
    'iemocap':  dict(pol='rl_policy_net_lam0.5',          delta=0.1, folds=3,
                     has_dynmm=True,  has_dymo=True),
    'mmfi':     dict(pol='rl_policy_net_mmfi_rebal',      delta=0.1, folds=3,
                     has_dynmm=True,  has_dymo=True),
    'cmi':      dict(pol='rl_policy_net_lam0.5',          delta=0.1, folds=3,
                     has_dynmm=True,  has_dymo=True),
    'czu_mhad': dict(pol='rl_policy_net_lam0.5',          delta=0.1, folds=3,
                     has_dynmm=True,  has_dymo=True),
    'dsads':    dict(pol='rl_policy_net_k16p_lam0.5',     delta=0.1, folds=4,
                     has_dynmm=False, has_dymo=False),
    'eav':      dict(pol='rl_policy_net_k16p_lam0.5',     delta=0.1, folds=3,
                     has_dynmm=False, has_dymo=False),
    'utd_mhad': dict(pol='rl_policy_net_utd_mhad_dropaug', delta=0.0, folds=3,
                     has_dynmm=True,  has_dymo=True),
}

DISPLAY = {'iemocap': 'IEMOCAP', 'mmfi': 'MM-Fi*', 'cmi': 'CMI',
           'czu_mhad': 'CZU-MHAD', 'dsads': 'DSADS', 'eav': 'EAV',
           'utd_mhad': 'UTD-MHAD**'}


class DsCfg:
    """Static per-dataset facts + the seqA K=16 checkpoint directory."""
    def __init__(self, ds):
        st = DS_STATIC[ds]
        self.ds       = ds
        self.mods     = st['mods']
        self.variates = st['variates']
        self.M        = len(self.mods)
        self.sum_var  = sum(self.variates.values())
        self.num_cls  = st['num_classes']
        self.T        = st['T']
        g = glob.glob(os.path.join(CKPT_DIR, f'*{ds}_pv3_branch_16_seqAbranch'))
        assert g, f'no K=16 seqA_branched dir for {ds}'
        self.seq16 = g[0]


# ══════════════════════════════════════════════════════════════════════════════
#  Test data
# ══════════════════════════════════════════════════════════════════════════════
def load_real_test(ds, fold, dev):
    """data/<ds>_fold<k>_test.pt -> (batches, mods, variates, C, in_len)."""
    d = torch.load(os.path.join(DATA_DIR, f'{ds}_fold{fold}_test.pt'), map_location='cpu')
    mods, variates = list(d['mods']), dict(d['variates'])
    C, in_len = int(d['num_classes']), int(d['in_len'])
    batches = []
    for b in d['batches']:
        xd = {m: b['high'][m].float().to(dev) for m in mods}
        y = b['y'].to(dev)
        batches.append((xd, y, y.shape[0]))
    return batches, mods, variates, C, in_len


def make_samples(batches, mods, n=0):
    """Explode batches into per-sample dicts of [1,T,V] tensors already on device."""
    samples = [{m: xd[m][i:i + 1].contiguous() for m in mods}
               for xd, y, B in batches for i in range(B)]
    labels = [int(y[i].item()) if y.ndim == 1 else int(y[i].argmax().item())
              for xd, y, B in batches for i in range(B)]
    return (samples[:n], labels[:n]) if n else (samples, labels)


# ══════════════════════════════════════════════════════════════════════════════
#  RL policy (dqn_qprior, deterministic eps=0 rollout)
# ══════════════════════════════════════════════════════════════════════════════
class ResidualQ(nn.Module):
    """f_theta — same 2-hidden-layer MLP as dqn_policy_qprior.ResidualQ."""
    def __init__(self, Fdim, A, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(Fdim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, A))

    def forward(self, x):
        return self.net(x)


def load_policy(path, dev):
    blob = torch.load(path, map_location=dev, weights_only=False)
    cfg = blob['cfg']
    net = ResidualQ(cfg['fdim'], cfg['M'] + 1).to(dev)
    net.load_state_dict(blob['state_dict'])
    net.eval()

    def _np(v):
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        return np.asarray(v, np.float32)
    vv = {frozenset(k): _np(v) for k, v in blob['vv'].items()}
    return net, vv, cfg


def _state(Sset, soft, M, avail):
    """[mask | avail | softmax | H | margin | depth/M | |avail|/M]"""
    mask = np.zeros(M, np.float32)
    for m in Sset:
        mask[m] = 1.0
    p = np.clip(soft, 1e-8, 1.0)
    H = float(-(p * np.log(p)).sum())
    srt = np.sort(soft)
    margin = float(srt[-1] - srt[-2])
    return np.concatenate([mask, avail.astype(np.float32), soft.astype(np.float32),
                           [H], [margin], [len(Sset) / M],
                           [float(avail.sum()) / M]]).astype(np.float32)


def _qprior(Sset, soft, vv, lam, M, avail, wc, wa, wcost):
    """Q[STOP]=wc*conf ; Q[acquire m]=wc*conf + wa*adv[m] - wcost*(lam/|A|)"""
    conf = float(soft.max())
    adv = np.zeros(M, np.float32)
    tab = vv.get(frozenset(Sset))            # [C,M] conditional gain table
    if tab is not None:
        adv = soft.astype(np.float32) @ tab
        for m in Sset:
            adv[m] = 0.0
    sc = lam / max(1.0, float(avail.sum()))
    Q = np.zeros(M + 1, np.float32)
    Q[M] = wc * conf
    Q[:M] = wc * conf + wa * adv - wcost * sc
    return Q


# ══════════════════════════════════════════════════════════════════════════════
#  seqA wrappers
# ══════════════════════════════════════════════════════════════════════════════
class SeqAStep(nn.Module):
    """One acquisition step: encode a modality then fuse it into the bottleneck."""
    def __init__(self, branch):
        super().__init__()
        self.branch = branch

    def forward(self, x_m, bottleneck):
        tokens = self.branch.encode(x_m)
        return self.branch.fuse(tokens, bottleneck)


class SeqAHead(nn.Module):
    """Read logits off the running token sum."""
    def __init__(self, seqa):
        super().__init__()
        self.fn = seqa.final_norm
        self.rg = seqa.regressor

    def forward(self, tok_sum, tok_count):
        return self.rg(self.fn(tok_sum / tok_count))


# ══════════════════════════════════════════════════════════════════════════════
#  C2: stacked-BMM pre-encode
# ══════════════════════════════════════════════════════════════════════════════
class StackedBmmEncoder(nn.Module):
    """M per-modality encoder blocks fused into single BMM calls.

    Converts M independent TransformerBlock chains into stacked-weight tensors
    [M, ...] so the whole encode runs as one grouped BMM per layer — no Python
    loop over modalities, no per-modality CUDA kernel dispatch.
    """

    def __init__(self, branches, mods):
        super().__init__()
        b0 = branches[mods[0]]
        n_enc = len(b0.encoder)
        D = b0.encoder[0].ln1.weight.shape[0]
        self.n_layers = n_enc
        self.M = len(mods)
        self.D = D
        self.nhead = b0.encoder[0].attn.num_heads
        self.dh = D // self.nhead
        self.mods = mods

        for l in range(n_enc):
            self.register_buffer(f'l{l}_qkv_w', torch.stack(
                [branches[m].encoder[l].attn.in_proj_weight.T for m in mods]))
            self.register_buffer(f'l{l}_qkv_b', torch.stack(
                [branches[m].encoder[l].attn.in_proj_bias for m in mods]))
            self.register_buffer(f'l{l}_ao_w', torch.stack(
                [branches[m].encoder[l].attn.out_proj.weight.T for m in mods]))
            self.register_buffer(f'l{l}_ao_b', torch.stack(
                [branches[m].encoder[l].attn.out_proj.bias for m in mods]))
            self.register_buffer(f'l{l}_ln1_w', torch.stack(
                [branches[m].encoder[l].ln1.weight for m in mods]))
            self.register_buffer(f'l{l}_ln1_b', torch.stack(
                [branches[m].encoder[l].ln1.bias for m in mods]))
            self.register_buffer(f'l{l}_ln2_w', torch.stack(
                [branches[m].encoder[l].ln2.weight for m in mods]))
            self.register_buffer(f'l{l}_ln2_b', torch.stack(
                [branches[m].encoder[l].ln2.bias for m in mods]))
            self.register_buffer(f'l{l}_mlp_w1', torch.stack(
                [branches[m].encoder[l].mlp[0].weight.T for m in mods]))
            self.register_buffer(f'l{l}_mlp_b1', torch.stack(
                [branches[m].encoder[l].mlp[0].bias for m in mods]))
            self.register_buffer(f'l{l}_mlp_w2', torch.stack(
                [branches[m].encoder[l].mlp[3].weight.T for m in mods]))
            self.register_buffer(f'l{l}_mlp_b2', torch.stack(
                [branches[m].encoder[l].mlp[3].bias for m in mods]))

        self.register_buffer('enc_norm_w', torch.stack(
            [branches[m].enc_norm.weight for m in mods]))
        self.register_buffer('enc_norm_b', torch.stack(
            [branches[m].enc_norm.bias for m in mods]))
        self.register_buffer('in_norm_w', torch.stack(
            [branches[m].in_norm.weight for m in mods]))
        self.register_buffer('in_norm_b', torch.stack(
            [branches[m].in_norm.bias for m in mods]))

    def _ln(self, x, w, b):
        return F.layer_norm(x, (self.D,)) * w[:, None, :] + b[:, None, :]

    def _attn_block(self, x, qkv_w, qkv_b, ao_w, ao_b, ln1_w, ln1_b):
        M, T, D = x.shape
        res = x
        x = self._ln(x, ln1_w, ln1_b)
        qkv = torch.bmm(x, qkv_w) + qkv_b[:, None, :]
        q, k, v = qkv.chunk(3, dim=-1)
        nh, dh = self.nhead, self.dh
        q = q.view(M, T, nh, dh).permute(0, 2, 1, 3).reshape(M * nh, T, dh)
        k = k.view(M, T, nh, dh).permute(0, 2, 1, 3).reshape(M * nh, T, dh)
        v = v.view(M, T, nh, dh).permute(0, 2, 1, 3).reshape(M * nh, T, dh)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        a = a.reshape(M, nh, T, dh).permute(0, 2, 1, 3).reshape(M, T, D)
        return res + torch.bmm(a, ao_w) + ao_b[:, None, :]

    def _mlp_block(self, x, mlp_w1, mlp_b1, mlp_w2, mlp_b2, ln2_w, ln2_b):
        res = x
        x = self._ln(x, ln2_w, ln2_b)
        x = torch.bmm(x, mlp_w1) + mlp_b1[:, None, :]
        x = F.gelu(x)
        return res + torch.bmm(x, mlp_w2) + mlp_b2[:, None, :]

    @torch.inference_mode()
    def forward(self, x_stacked):
        """[M, B, T, D] -> [M, B, T, D] (encoded tokens for all M modalities)."""
        M, B, T, D = x_stacked.shape
        x = x_stacked.view(M, B * T, D) if B > 1 else x_stacked.squeeze(1)
        x = self._ln(x, self.in_norm_w, self.in_norm_b)
        for l in range(self.n_layers):
            x = self._attn_block(x,
                    getattr(self, f'l{l}_qkv_w'), getattr(self, f'l{l}_qkv_b'),
                    getattr(self, f'l{l}_ao_w'),  getattr(self, f'l{l}_ao_b'),
                    getattr(self, f'l{l}_ln1_w'), getattr(self, f'l{l}_ln1_b'))
            x = self._mlp_block(x,
                    getattr(self, f'l{l}_mlp_w1'), getattr(self, f'l{l}_mlp_b1'),
                    getattr(self, f'l{l}_mlp_w2'), getattr(self, f'l{l}_mlp_b2'),
                    getattr(self, f'l{l}_ln2_w'),  getattr(self, f'l{l}_ln2_b'))
        x = self._ln(x, self.enc_norm_w, self.enc_norm_b)
        return x.unsqueeze(1) if B == 1 else x.view(M, B, T, D)


def build_stacked_encoder(seqa, mods, dev, verbose=True):
    """Build the stacked encoder and assert it matches per-branch encode()."""
    enc = StackedBmmEncoder(seqa.branches, mods).to(dev).eval()
    dtype = next(seqa.parameters()).dtype
    with torch.inference_mode():
        torch.manual_seed(0)
        probes = {m: torch.randn(1, 100, seqa.branches[m].projector.in_features,
                                 device=dev, dtype=dtype) for m in mods}
        ref = torch.stack([seqa.branches[m].encode(probes[m]) for m in mods])
        proj = torch.stack([seqa.branches[m].projector(probes[m]) for m in mods])
        max_err = (enc(proj) - ref).abs().max().item()
    tol = 5e-2 if dtype == torch.float16 else 5e-4
    if verbose:
        print(f'    StackedBmmEncoder parity: max_err={max_err:.2e} (tol={tol:.0e})', flush=True)
    assert max_err < tol, f'Parity fail: max_err={max_err:.2e}'
    return enc


# ══════════════════════════════════════════════════════════════════════════════
#  Walk variants — the RL decision loop
# ══════════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def walk_a0(s, seqa, steps, head, net, vv, cfg, mods, dev):
    """A0: encode+fuse each acquired modality sequentially inside the loop."""
    M, md, lam = cfg['M'], cfg['md'], cfg['lam']
    wc, wa, wcost = cfg['W_CONF'], cfg['W_ADV'], cfg['W_COST']
    avail = np.ones(M, np.float32)
    bn = seqa.bottleneck.expand(1, -1, -1).contiguous()
    tok_sum = torch.zeros(1, seqa.d_model, device=dev)
    tok_count = 0
    S, Sset = [], set()
    soft = np.full(cfg['C'], 1.0 / cfg['C'], np.float32)
    logits = None
    while len(S) < md:
        st = torch.from_numpy(_state(Sset, soft, M, avail)).to(dev)
        Q = net(st).cpu().numpy() + _qprior(Sset, soft, vv, lam, M, avail, wc, wa, wcost)
        legal = [m for m in range(M) if m not in Sset]
        if not legal:
            break
        if len(S) > 0 and Q[M] + OURS_DELTA >= max(Q[m] for m in legal):
            break
        a = max(legal, key=lambda m: Q[m])
        S.append(a); Sset.add(a)
        new_bn, out_tok = steps[mods[a]](s[mods[a]], bn)
        bn = seqa._agg_bottleneck(len(S) - 1, bn, new_bn)
        tok_sum += out_tok.sum(1); tok_count += out_tok.shape[1]
        logits = head(tok_sum, float(tok_count))
        soft = F.softmax(logits.float(), -1)[0].cpu().numpy()
    return logits, len(S)


@torch.inference_mode()
def walk_c2(s, seqa, enc_stacked, head, net, vv, cfg, mods, dev, conf_thresh=0.0):
    """C2: stacked-BMM pre-encode all M, then a fuse-only walk (tensor indexing).

    conf_thresh > 0 adds an entropy early exit on top of the delta stop rule.
    """
    M, md, lam = cfg['M'], cfg['md'], cfg['lam']
    wc, wa, wcost = cfg['W_CONF'], cfg['W_ADV'], cfg['W_COST']
    avail = np.ones(M, np.float32)

    proj = torch.stack([seqa.branches[m].projector(s[m]) for m in mods])
    pre_stacked = enc_stacked(proj)

    bn = seqa.bottleneck.expand(1, -1, -1).contiguous()
    tok_sum = torch.zeros(1, seqa.d_model, device=dev, dtype=pre_stacked.dtype)
    tok_count = 0
    S, Sset = [], set()
    soft = np.full(cfg['C'], 1.0 / cfg['C'], np.float32)
    logits = None
    while len(S) < md:
        st = torch.from_numpy(_state(Sset, soft, M, avail)).to(dev)
        Q = net(st).cpu().numpy() + _qprior(Sset, soft, vv, lam, M, avail, wc, wa, wcost)
        legal = [m for m in range(M) if m not in Sset]
        if not legal:
            break
        if len(S) > 0 and Q[M] + OURS_DELTA >= max(Q[m] for m in legal):
            break
        a = max(legal, key=lambda m: Q[m])
        S.append(a); Sset.add(a)
        new_bn, out_tok = seqa.branches[mods[a]].fuse(pre_stacked[a], bn)
        bn = seqa._agg_bottleneck(len(S) - 1, bn, new_bn)
        tok_sum += out_tok.sum(1); tok_count += out_tok.shape[1]
        logits = head(tok_sum, float(tok_count))
        soft = F.softmax(logits.float(), -1)[0].cpu().numpy()
        if conf_thresh > 0.0 and soft.max() > conf_thresh:
            break
    return logits, len(S)


# ══════════════════════════════════════════════════════════════════════════════
#  Baseline forwards — only the selected sub-network runs
# ══════════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def fwd_adamml(adamml, s, mods):
    dec, plog = adamml.policy(s)
    if dec.sum() == 0:
        dec[0, plog[..., 1].argmax(1)] = 1.0          # guarantee >=1 modality
    kept = [m for i, m in enumerate(mods) if dec[0, i] > 0.5]
    w = F.softmax(adamml.lf, 0)
    logits, wsum = None, 0.0
    for i, m in enumerate(mods):
        if dec[0, i] > 0.5:
            lm = adamml.encoders[m](s[m])[0]           # ONLY kept encoders run
            logits = w[i] * lm if logits is None else logits + w[i] * lm
            wsum = wsum + w[i]
    return logits / wsum.clamp_min(1e-6), ('adamml', frozenset(kept))


@torch.inference_mode()
def fwd_dynmm(gate, experts, branches, tau, s_cat):
    sel = diff_softmax(gate(s_cat), tau=tau, hard=True).argmax(1).item()
    br = branches[sel]
    return experts[br](s_cat), ('dynmm', br)          # ONLY selected expert runs


@torch.inference_mode()
def fwd_dymo(proto, s_cat, sg, dev):
    logits, mask = proto_greedy_select(proto, s_cat, sg, dev)   # greedy O(M^2)
    return logits, ('dymo', int(mask.long().sum().item()))


# ══════════════════════════════════════════════════════════════════════════════
#  Model loaders
# ══════════════════════════════════════════════════════════════════════════════
def _build_seqa(ckpt_dir, D, fold, device):
    from models.seqA_branched import SeqABranched
    j = json.load(open(os.path.join(ckpt_dir, f'results_fold{fold}.json')))
    model = SeqABranched(
        modalities=D.mods, feat_dims=D.variates, num_classes=D.num_cls,
        d_model=j['d_model'], nhead=j['nhead'],
        num_enc_layers=j['num_enc_layers'], num_fusion_layers=j['num_fusion_layers'],
        n_bottlenecks=j['n_bottlenecks'],
        use_sparse_attn=j.get('use_sparse_attn', False),
        sparse_attn_variant=j.get('sparse_attn_variant', 'opt'),
        factor=j.get('base_factor', 5),
        seq_agg_mode=j.get('seq_agg_mode', 'mean'),
        seq_random_order=False, max_len=D.T,
        num_fusion_distill=j.get('num_fusion_distill', 0),
    ).to(device).float()
    ck = torch.load(os.path.join(ckpt_dir, f'best_model_fold{fold}.pth'), map_location=device)
    model.load_state_dict(ck); model.eval()
    return model


def _build_seqa_mmfi_rebal(fold, device):
    """MM-Fi rebalanced uses num_fusion_distill=4, which only the main repo's
    SeqABranched supports — import it with the local models/ shadowed out."""
    main_repo = os.path.dirname(RUNS)
    saved = sys.path[:]
    sys.path = [main_repo] + [p for p in sys.path if os.path.abspath(p or '.') != HERE]
    try:
        for k in list(sys.modules.keys()):
            if k in ('models', 'models.seqA_branched', 'models.seqA'):
                del sys.modules[k]
        from models.seqA_branched import SeqABranched as _SB
    finally:
        sys.path = saved

    D = DsCfg('mmfi')
    j = json.load(open(os.path.join(REBAL_CKPT, f'results_fold{fold}.json')))
    model = _SB(
        modalities=D.mods, feat_dims=D.variates, num_classes=D.num_cls,
        d_model=j['d_model'], nhead=j['nhead'],
        num_enc_layers=j['num_enc_layers'], num_fusion_layers=j['num_fusion_layers'],
        n_bottlenecks=j['n_bottlenecks'],
        use_sparse_attn=j.get('use_sparse_attn', False),
        sparse_attn_variant=j.get('sparse_attn_variant', 'opt'),
        factor=j.get('base_factor', 5),
        seq_agg_mode=j.get('seq_agg_mode', 'mean'),
        seq_random_order=False, max_len=D.T,
        num_fusion_distill=j.get('num_fusion_distill', 0),
    ).to(device).float()
    ck = torch.load(os.path.join(REBAL_CKPT, f'best_model_fold{fold}.pth'), map_location=device)
    model.load_state_dict(ck); model.eval()
    return model


def load_seqa(ds, fold, device):
    if ds == 'mmfi':
        return _build_seqa_mmfi_rebal(fold, device)
    ckpt = DROP_CKPT if ds == 'utd_mhad' else DsCfg(ds).seq16
    return _build_seqa(ckpt, DsCfg(ds), fold, device)


def load_adamml_g00(ds, fold, device):
    """AdaMML with gamma=0 and ffn=128 — the fair-compute configuration."""
    D = DsCfg(ds)
    ck = torch.load(os.path.join(CKPT_DIR, f'adamml_ffn128_g00_{ds}', 'adamml',
                                 f'best_model_fold{fold}.pth'), map_location=device)
    pos_key = next(k for k in ck if k.endswith('.pos'))
    ffn_key = next((k for k in ck if '.enc.layers.0.linear1.bias' in k), None)
    model = AdaMMLNet(D.variates, D.mods, d_model=128, nhead=8, num_layers=6,
                      num_classes=D.num_cls, dropout=0.1,
                      max_len=ck[pos_key].shape[1],
                      dim_feedforward=ck[ffn_key].shape[0] if ffn_key else None
                      ).to(device).float()
    model.load_state_dict(ck); model.eval()
    return model


def load_dynmm_real_gate(ds, fold, mods, variates, C, dev):
    """Trained DynMM gate + all branch experts."""
    base = os.path.join(CKPT_DIR, f'dynmm_{ds}', f'fold{fold}')
    rj = json.load(open(os.path.join(CKPT_DIR, f'dynmm_{ds}',
                                     f'results_fold{fold}.json')))
    branches = rj.get('usage_branches', rj.get('branches', ['1mod', 'full']))
    experts, nmods = {}, {}
    for br in branches:
        cfg = json.load(open(os.path.join(base, 'experts', br, 'config.json')))
        sub = list(cfg['modalities']); nmods[br] = len(sub)
        ck = torch.load(os.path.join(base, 'experts', br, 'best_model.pth'), map_location=dev)
        emode = 'sax' if any(k.endswith('token_embedding.weight') for k in ck) else 'float'
        e = OrigDynMMExpertG(sub, list(mods), dict(variates), C, input_mode=emode,
                             embed_dim=128, nhead=8, num_layers=6).to(dev).float()
        e.load_state_dict(ck); e.eval(); experts[br] = e
    gck = torch.load(os.path.join(base, 'gate.pth'), map_location=dev)
    gmode = 'sax' if any(k.endswith('token_embedding.weight') for k in gck) else 'float'
    gate = OrigGateG(list(mods), dict(variates), len(branches), input_mode=gmode,
                     gate_dim=int(rj.get('gate_dim', 16)), nhead=int(rj.get('nhead', 8)),
                     num_layers=int(rj.get('num_layers', 6))).to(dev).float()
    gate.load_state_dict(gck); gate.eval()
    return gate, experts, nmods, branches, float(rj.get('tau', 1.0))


def load_dymo_with_sg(ds, fold, device):
    """DyMo prototype model + its cached subset-Gaussian statistics."""
    from models.dymo import CrossModalDyMoProto
    from models.crossmodal_transformer import CrossModalTransformer
    D = DsCfg(ds)
    base = os.path.join(CKPT_DIR, f'dymo_{ds}')
    ck = torch.load(os.path.join(base, f'best_model_fold{fold}.pth'), map_location=device)
    j = json.load(open(os.path.join(base, f'results_fold{fold}.json')))
    d_model = ck['model.pos_embedding'].shape[2]
    input_len = ck['model.pos_embedding'].shape[1] - 64
    n_enc = len([k for k in ck if f'model.encoders.{D.mods[0]}.encoder.layers.' in k
                 and k.endswith('.self_attn.in_proj_weight')])
    n_fus = len([k for k in ck if 'model.fusion.layers.' in k
                 and k.endswith('.self_attn.in_proj_weight')])

    class _Cfg: pass
    cfg = _Cfg(); cfg.modalities = D.mods; cfg.variates = D.variates
    cfg.num_classes = D.num_cls
    mode = 'sax' if any(k.endswith('token_embedding.weight') for k in ck) else 'float'
    inner = CrossModalTransformer(cfg, input_len, input_mode=mode, d_model=d_model,
                                  nhead=8, num_enc_layers=n_enc,
                                  num_fusion_layers=n_fus, dropout=0.1).to(device).float()
    proto = CrossModalDyMoProto(inner, D.num_cls, projection_dim=j['projection_dim'],
                                proto_temperature=j['proto_temperature']).to(device).float()
    proto.load_state_dict(ck); proto.eval()

    sg_path = os.path.join(SG_DIR, f'{ds}_fold{fold}.pt')
    assert os.path.exists(sg_path), f'missing cached subset Gaussians: {sg_path}'
    return proto, torch.load(sg_path, map_location=device)


# ══════════════════════════════════════════════════════════════════════════════
#  Timed loops — the timer starts BEFORE any modality is read from disk
# ══════════════════════════════════════════════════════════════════════════════
def _sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def _timed(fwd, samples, labels, dev, warmup):
    """Run fwd(sample) -> (logits, depth) over all samples, timing each call."""
    for s in samples[:warmup]:
        fwd(s)
    _sync(dev)
    preds, lats, depths = [], [], []
    for s in samples:
        t0 = time.perf_counter()
        logits, depth = fwd(s)
        _sync(dev)
        lats.append((time.perf_counter() - t0) * 1e3)
        depths.append(depth)
        preds.append(int(logits.argmax(-1).item()) if logits is not None else 0)
    return (f1_score(labels, preds, average='macro'),
            np.array(lats), float(np.mean(depths)))

# ══════════════════════════════════════════════════════════════════════════════
#  Per-modality sample files on disk
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_DIR = os.path.join(HERE, 'samples')


def get_sample_dirs(ds, fold, n=0):
    """Sorted per-sample directories written by prepare_samples.py."""
    fold_dir = os.path.join(SAMPLE_DIR, f'{ds}_fold{fold}')
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(
            f'Sample dir not found: {fold_dir}\n'
            f'Run:  python prepare_samples.py  first.')
    dirs = sorted(os.path.join(fold_dir, d) for d in os.listdir(fold_dir)
                  if d.startswith('sample_')
                  and os.path.isdir(os.path.join(fold_dir, d)))
    return dirs[:n] if n else dirs


def load_one_mod(sample_dir, mod, dev):
    """Disk -> CPU RAM -> GPU for a SINGLE modality. The unit of sequential I/O."""
    x = np.load(os.path.join(sample_dir, f'{mod}.npy'))
    return torch.from_numpy(x).unsqueeze(0).float().to(dev)


def load_all_mods(sample_dir, mods, dev):
    """Disk -> CPU RAM -> GPU for ALL M modalities. What the baselines must pay."""
    return {m: load_one_mod(sample_dir, m, dev) for m in mods}


def read_label(sample_dir):
    """Read the label OUTSIDE the timed region — it is not part of inference."""
    return int(np.load(os.path.join(sample_dir, 'label.npy')))


# ══════════════════════════════════════════════════════════════════════════════
#  Sequential-loading walk — the point of the whole benchmark
# ══════════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def walk_a0_disk(sample_dir, seqa, steps, head, net, vv, cfg, mods, dev):
    """A0 walk that reads each modality from disk at the moment it is acquired.

    Identical decisions to walk_a0 — the only difference is WHEN the data
    arrives. No modality file is opened until the policy has chosen it, so a
    sample that stops after k acquisitions performs exactly k file reads.
    """
    M, md, lam = cfg['M'], cfg['md'], cfg['lam']
    wc, wa, wcost = cfg['W_CONF'], cfg['W_ADV'], cfg['W_COST']
    avail = np.ones(M, np.float32)
    bn = seqa.bottleneck.expand(1, -1, -1).contiguous()
    tok_sum = torch.zeros(1, seqa.d_model, device=dev)
    tok_count = 0
    S, Sset = [], set()
    soft = np.full(cfg['C'], 1.0 / cfg['C'], np.float32)
    logits = None
    while len(S) < md:
        st = torch.from_numpy(_state(Sset, soft, M, avail)).to(dev)
        Q = net(st).cpu().numpy() + _qprior(Sset, soft, vv, lam, M, avail, wc, wa, wcost)
        legal = [m for m in range(M) if m not in Sset]
        if not legal:
            break
        if len(S) > 0 and Q[M] + OURS_DELTA >= max(Q[m] for m in legal):
            break
        a = max(legal, key=lambda m: Q[m])
        S.append(a); Sset.add(a)
        x_t = load_one_mod(sample_dir, mods[a], dev)      # <- the on-demand read
        new_bn, out_tok = steps[mods[a]](x_t, bn)
        bn = seqa._agg_bottleneck(len(S) - 1, bn, new_bn)
        tok_sum += out_tok.sum(1); tok_count += out_tok.shape[1]
        logits = head(tok_sum, float(tok_count))
        soft = F.softmax(logits.float(), -1)[0].cpu().numpy()
    return logits, len(S)


def _timed_loading(fwd, sample_dirs, dev, warmup):
    """Time fwd(sample_dir) -> (logits, depth) with disk I/O inside the timer.

    The label is read before the clock starts: it is ground truth, not input.
    """
    for sd in sample_dirs[:warmup]:
        fwd(sd)
    _sync(dev)
    preds, labels, lats, depths = [], [], [], []
    for sd in sample_dirs:
        lbl = read_label(sd)
        t0 = time.perf_counter()
        logits, depth = fwd(sd)
        _sync(dev)
        lats.append((time.perf_counter() - t0) * 1e3)
        depths.append(depth); labels.append(lbl)
        preds.append(int(logits.argmax(-1).item()) if logits is not None else 0)
    return (f1_score(labels, preds, average='macro'),
            np.array(lats), float(np.mean(depths)))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global OURS_DELTA
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',      type=int, default=50, help='samples per fold (0=all)')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--gpu',    type=int, default=0)
    ap.add_argument('--cpu',    action='store_true', help='force CPU')
    ap.add_argument('--mode',   choices=['sequential', 'bundled', 'both'],
                    default='sequential',
                    help='how seqA+RL loads its modalities (baselines always load all M)')
    ap.add_argument('--datasets', nargs='+', default=list(DS_CFG), choices=list(DS_CFG))
    ap.add_argument('--out', default=os.path.join(HERE, 'results_loading.json'))
    a = ap.parse_args()

    dev = torch.device('cpu') if a.cpu or not torch.cuda.is_available() \
          else torch.device(f'cuda:{a.gpu}')
    want_seq = a.mode in ('sequential', 'both')
    want_bun = a.mode in ('bundled', 'both')

    print(f'\n=== Loading-inclusive benchmark (disk -> RAM -> {dev.type.upper()} -> forward) ===')
    print(f'device={dev}  n/fold={a.n or "all"}  warmup={a.warmup}  mode={a.mode}')
    print(f'seqA+RL loads only the modalities it acquires; baselines load all M.\n')

    results = {}
    for ds in a.datasets:
        dcfg = DS_CFG[ds]
        OURS_DELTA = dcfg['delta']
        pol_dir = os.path.join(HERE, dcfg['pol'])
        D = DsCfg(ds)
        mods = D.mods
        results[ds] = {}
        print(f'── {DISPLAY[ds]}  (δ={dcfg["delta"]}, {dcfg["folds"]} folds, M={D.M}) ──')

        keys = (['a0_seq'] if want_seq else []) + (['a0_bundled'] if want_bun else [])
        keys += ['adamml']
        keys += ['dynmm'] if dcfg['has_dynmm'] else []
        keys += ['dymo'] if dcfg['has_dymo'] else []
        acc = {k: dict(f1s=[], lats=[], mods=[]) for k in keys}

        for fold in range(dcfg['folds']):
            sample_dirs = get_sample_dirs(ds, fold, a.n)
            print(f'  fold{fold}  n={len(sample_dirs)}')

            # ── seqA+RL ────────────────────────────────────────────────────
            seqa = load_seqa(ds, fold, dev)
            head = SeqAHead(seqa).eval().to(dev)
            steps = {m: SeqAStep(seqa.branches[m]).eval().to(dev) for m in mods}
            net, vv, cfg = load_policy(os.path.join(pol_dir, f'{ds}_fold{fold}.pt'), dev)
            net.eval()

            if want_seq:
                f1, lats, mdp = _timed_loading(
                    lambda sd: walk_a0_disk(sd, seqa, steps, head, net, vv, cfg, mods, dev),
                    sample_dirs, dev, a.warmup)
                acc['a0_seq']['f1s'].append(f1); acc['a0_seq']['lats'].append(lats.mean())
                acc['a0_seq']['mods'].append(mdp)
                print(f'    A0 seq-load    F1={f1:.4f}  lat={lats.mean():6.2f}ms  '
                      f'mods={mdp:.2f}/{D.M}  [reads {mdp:.2f}/{D.M} files]')

            if want_bun:
                def _fwd_bundled(sd):
                    s = load_all_mods(sd, mods, dev)      # all M read upfront
                    return walk_a0(s, seqa, steps, head, net, vv, cfg, mods, dev)

                f1, lats, mdp = _timed_loading(_fwd_bundled, sample_dirs, dev, a.warmup)
                acc['a0_bundled']['f1s'].append(f1)
                acc['a0_bundled']['lats'].append(lats.mean())
                acc['a0_bundled']['mods'].append(mdp)
                print(f'    A0 bundled     F1={f1:.4f}  lat={lats.mean():6.2f}ms  '
                      f'mods={mdp:.2f}/{D.M}  [reads {D.M}/{D.M} files]')

            del seqa, head, steps, net
            if dev.type == 'cuda':
                torch.cuda.empty_cache()

            # ── AdaMML ─────────────────────────────────────────────────────
            adamml = load_adamml_g00(ds, fold, dev)

            def _fwd_am(sd):
                s = load_all_mods(sd, mods, dev)
                logits, route = fwd_adamml(adamml, s, mods)
                return logits, len(route[1])

            f1, lats, mdp = _timed_loading(_fwd_am, sample_dirs, dev, a.warmup)
            acc['adamml']['f1s'].append(f1); acc['adamml']['lats'].append(lats.mean())
            acc['adamml']['mods'].append(mdp)
            print(f'    AdaMML         F1={f1:.4f}  lat={lats.mean():6.2f}ms  '
                  f'mods={mdp:.2f}/{D.M}  [reads {D.M}/{D.M} files]')
            del adamml
            if dev.type == 'cuda':
                torch.cuda.empty_cache()

            # ── DynMM ──────────────────────────────────────────────────────
            if dcfg['has_dynmm']:
                try:
                    gate, experts, nmods, branches, tau = load_dynmm_real_gate(
                        ds, fold, mods, D.variates, D.num_cls, dev)

                    def _fwd_dyn(sd):
                        s = load_all_mods(sd, mods, dev)
                        s_cat = torch.cat([s[m] for m in mods], dim=-1)
                        logits, route = fwd_dynmm(gate, experts, branches, tau, s_cat)
                        return logits, nmods[route[1]]

                    f1, lats, mdp = _timed_loading(_fwd_dyn, sample_dirs, dev, a.warmup)
                    acc['dynmm']['f1s'].append(f1); acc['dynmm']['lats'].append(lats.mean())
                    acc['dynmm']['mods'].append(mdp)
                    print(f'    DynMM          F1={f1:.4f}  lat={lats.mean():6.2f}ms  '
                          f'mods={mdp:.2f}/{D.M}  [reads {D.M}/{D.M} files]')
                    del gate, experts
                except Exception as e:
                    print(f'    DynMM   SKIP: {e}')
                    for k in ('f1s', 'lats', 'mods'):
                        acc['dynmm'][k].append(float('nan'))
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()

            # ── DyMo ───────────────────────────────────────────────────────
            if dcfg['has_dymo']:
                try:
                    proto, sg = load_dymo_with_sg(ds, fold, dev)

                    def _fwd_dymo(sd):
                        s = load_all_mods(sd, mods, dev)
                        s_cat = torch.cat([s[m] for m in mods], dim=-1)
                        logits, route = fwd_dymo(proto, s_cat, sg, dev)
                        return logits, route[1]

                    f1, lats, mdp = _timed_loading(_fwd_dymo, sample_dirs, dev, a.warmup)
                    acc['dymo']['f1s'].append(f1); acc['dymo']['lats'].append(lats.mean())
                    acc['dymo']['mods'].append(mdp)
                    print(f'    DyMo           F1={f1:.4f}  lat={lats.mean():6.2f}ms  '
                          f'mods={mdp:.2f}/{D.M}  [reads {D.M}/{D.M} files]')
                    del proto
                except Exception as e:
                    print(f'    DyMo    SKIP: {e}')
                    for k in ('f1s', 'lats', 'mods'):
                        acc['dymo'][k].append(float('nan'))
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()

        for model, v in acc.items():
            results[ds][model] = dict(
                f1=float(np.nanmean(v['f1s'])), lat=float(np.nanmean(v['lats'])),
                lat_std=float(np.nanstd(v['lats'])), mods=float(np.nanmean(v['mods'])),
                M=D.M)
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    MODELS = [k for k in ['a0_seq', 'a0_bundled', 'adamml', 'dynmm', 'dymo']
              if any(k in results[ds] for ds in results)]
    NAMES = {'a0_seq': 'seqA+RL seq', 'a0_bundled': 'seqA+RL bundled',
             'adamml': 'AdaMML', 'dynmm': 'DynMM', 'dymo': 'DyMo'}

    print(f'\n{"":14}' + ''.join(f'  {NAMES[m]:>17}' for m in MODELS))
    print(f'{"Dataset":<14}' + ''.join(f'  {"lat(ms)/F1":>17}' for m in MODELS))
    print('-' * (14 + 19 * len(MODELS)))
    for ds in results:
        print(f'{DISPLAY[ds]:<14}', end='')
        for m in MODELS:
            r = results[ds].get(m)
            if r is None or np.isnan(r['lat']):
                print(f'  {"—":>17}', end='')
            else:
                print(f'  {r["lat"]:8.2f}ms/{r["f1"]:.3f}', end='')
        print()

    ours = 'a0_seq' if want_seq else 'a0_bundled'
    print(f'\nSpeedup {NAMES[ours]} vs baseline  (lat_baseline / lat_ours):')
    for bm in [m for m in MODELS if m != ours]:
        spds = [results[ds][bm]['lat'] / results[ds][ours]['lat']
                for ds in results
                if bm in results[ds] and not np.isnan(results[ds][bm]['lat'])]
        if spds:
            print(f'  vs {NAMES[bm]:<16}: {np.exp(np.mean(np.log(spds))):.3f}× geomean  '
                  f'(per-ds: {[f"{s:.2f}" for s in spds]})')

    print(f'\nModality files read per sample:')
    for ds in results:
        r = results[ds].get('a0_seq')
        if r:
            M = r['M']
            print(f'  {DISPLAY[ds]:<14}: seqA+RL {r["mods"]:.2f}/{M}   baselines {M}/{M}   '
                  f'(skips {M - r["mods"]:.2f} files on average)')

    with open(a.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {a.out}')


if __name__ == '__main__':
    main()
