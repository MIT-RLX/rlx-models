#!/usr/bin/env python3
"""Re-export ChatterBox S3Gen flow decoder as a LOOP-friendly set of 3 small ONNX
graphs instead of the monolithic 23,934-node `conditional_decoder.onnx` (which
statically unrolled the 10-step CFM solver ~10x).

Outputs (into weights/tts/chatterbox/onnx/):
  flow_encoder.onnx : (speech_tokens, prompt_token, prompt_feat, embedding)
                      -> (mu, mask, spks, cond)   [runs ONCE]
  cfm_estimator.onnx: (x, mask, mu, t, spks, cond) -> dxdt   [the loop body, run N x]
  hift_vocoder.onnx : (speech_feat) -> waveform   [runs ONCE]

The Rust side then drives the CFM Euler solver (CFG-doubled batch, split,
`(1+cfg)*cond - cfg*uncond`, `x += (r-t)*dxdt`) — see solve_euler in
chatterbox/models/s3gen/flow_matching.py. Compile once, run the estimator N x.
"""
import os, sys, torch
import torch.nn as nn

OUT = sys.argv[1] if len(sys.argv) > 1 else "weights/tts/chatterbox/onnx"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

# Load S3Gen directly from the cached checkpoint — bypass ChatterboxTTS.__init__
# (it dies on the optional `perth` watermarker, which we don't need for export).
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from chatterbox.models.s3gen import S3Gen

ckpt = Path(snapshot_download("ResembleAI/chatterbox", allow_patterns=["s3gen.safetensors"]))
print("loading S3Gen from", ckpt)
s3 = S3Gen()
s3.load_state_dict(load_file(ckpt / "s3gen.safetensors"), strict=False)
s3.eval()
flow = s3.flow                        # CausalMaskedDiffWithXvec
cfm = flow.decoder                    # CausalConditionalCFM
est = cfm.estimator                   # ConditionalDecoder (the loop body)
est.eval(); flow.eval(); s3.eval()
print("estimator dtype:", est.dtype, "cfm cfg_rate:", cfm.inference_cfg_rate,
      "n_timesteps default:", getattr(cfm, "t_scheduler", "?"))

# ── 1) CFM estimator (the loop body) ────────────────────────────────────────
# solve_euler builds CFG-doubled batches of size 2B; export at batch 2, dynamic T.
class Est(nn.Module):
    def __init__(self, e): super().__init__(); self.e = e
    def forward(self, x, mask, mu, t, spks, cond):
        return self.e.forward(x=x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=None)

T = 200
xb   = torch.randn(2, 80, T)
mb   = torch.ones(2, 1, T)
mub  = torch.randn(2, 80, T)
tb   = torch.rand(2)
spb  = torch.randn(2, 80)
cob  = torch.randn(2, 80, T)
with torch.no_grad():
    ref = Est(est)(xb, mb, mub, tb, spb, cob)
print("estimator out:", tuple(ref.shape))
# Export with a STATIC batch=2 (the CFG-doubled batch is always 2 for a single
# utterance). A dynamic batch axis makes the importer mis-resolve the attention
# mask reshape at batch>1 (2·T length), forcing 2× batch-1 runs; a static batch
# bakes `[2,8,T,T]` shapes so ONE batch-2 run per solver step works — halving the
# estimator invocations. Only T stays dynamic.
torch.onnx.export(
    Est(est), (xb, mb, mub, tb, spb, cob), f"{OUT}/cfm_estimator.onnx",
    input_names=["x", "mask", "mu", "t", "spks", "cond"], output_names=["dxdt"],
    dynamic_axes={k: {2: "T"} for k in ["x", "mask", "mu", "cond", "dxdt"]},
    opset_version=17, do_constant_folding=True,
)
import onnx
print("cfm_estimator.onnx nodes:", len(onnx.load(f"{OUT}/cfm_estimator.onnx", load_external_data=False).graph.node))

# ── 2) HiFT vocoder SPECTRAL HEAD (mel -> magnitude, phase) ─────────────────
# `torch.istft` has no ONNX op, so we stop at (magnitude, phase) and do the tiny
# n_fft=16 / hop=4 ISTFT in Rust (the LuxTTS/Kokoro playbook). The forward STFT of
# the NSF source (`_stft`) exports fine to the ONNX STFT op (opset 17).
import math, torch.nn.functional as F
mw = s3.mel2wav  # HiFTGenerator

class STFTConv(nn.Module):
    """torch.stft(n_fft, hop, hann, center=True, return_complex=True) as Conv1d —
    exportable to ONNX (no complex/STFT op). Matches _stft's (real, imag) output."""
    def __init__(self, n_fft, hop, window):
        super().__init__()
        n = torch.arange(n_fft).float(); k = torch.arange(n_fft // 2 + 1).float()
        ang = (2 * math.pi / n_fft) * k[:, None] * n[None, :]  # [n_bins, n_fft]
        self.register_buffer("cos_k", (window * torch.cos(ang))[:, None, :])
        self.register_buffer("sin_k", (-window * torch.sin(ang))[:, None, :])
        self.hop, self.pad = hop, n_fft // 2
    def forward(self, x):  # x [B, L] -> real,imag [B, n_bins, T]
        x = F.pad(x[:, None], (self.pad, self.pad), mode="reflect")
        return F.conv1d(x, self.cos_k, stride=self.hop), F.conv1d(x, self.sin_k, stride=self.hop)

_stftconv = STFTConv(mw.istft_params["n_fft"], mw.istft_params["hop_len"], mw.stft_window)

# Make the NSF SineGen DETERMINISTIC — the reference adds a random per-harmonic
# phase offset (Uniform(-π,π)) + noise even at inference, which (a) makes the
# ONNX non-reproducible and (b) leans on RandomUniform/Normal ops. A zero-phase,
# noise-free sine source is an equally valid vocoder excitation and imports
# cleanly on every backend.
import types
def _det_sinegen(self, f0):
    F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1)), device=f0.device)
    for i in range(self.harmonic_num + 1):
        F_mat[:, i:i + 1, :] = f0 * (i + 1) / self.sampling_rate
    theta = 2 * math.pi * (torch.cumsum(F_mat, dim=-1) % 1)
    sine = self.sine_amp * torch.sin(theta)
    uv = self._f02uv(f0)
    sine = sine * uv
    return sine, uv, torch.zeros_like(sine)
mw.m_source.l_sin_gen.forward = types.MethodType(_det_sinegen, mw.m_source.l_sin_gen)
# sanity: manual STFT-conv matches torch.stft
_t = torch.randn(1, 400)
_ra, _ia = _stftconv(_t); _rb, _ib = mw._stft(_t)
print("STFTConv vs torch.stft max_abs:", (_ra - _rb).abs().max().item(), (_ia - _ib).abs().max().item())

class HiftHead(nn.Module):
    def __init__(self, m, stftc): super().__init__(); self.m = m; self.stftc = stftc
    def forward(self, speech_feat):
        m = self.m
        f0 = m.f0_predictor(speech_feat)
        s = m.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = m.m_source(s)
        s = s.transpose(1, 2)
        sr, si_ = self.stftc(s.squeeze(1))
        s_stft = torch.cat([sr, si_], dim=1)
        x = m.conv_pre(speech_feat)
        for i in range(m.num_upsamples):
            x = F.leaky_relu(x, m.lrelu_slope)
            x = m.ups[i](x)
            if i == m.num_upsamples - 1:
                x = m.reflection_pad(x)
            sd = m.source_downs[i](s_stft)
            sd = m.source_resblocks[i](sd)
            x = x + sd
            xs = None
            for j in range(m.num_kernels):
                r = m.resblocks[i * m.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / m.num_kernels
        x = F.leaky_relu(x)
        x = m.conv_post(x)
        nfft = m.istft_params["n_fft"]
        magnitude = torch.exp(x[:, : nfft // 2 + 1, :]).clip(max=1e2)
        phase = torch.sin(x[:, nfft // 2 + 1:, :])
        return magnitude, phase
feat = torch.randn(1, 80, T)
with torch.no_grad():
    mag, ph = HiftHead(mw, _stftconv)(feat)
print("hift head out: magnitude", tuple(mag.shape), "phase", tuple(ph.shape),
      "| istft n_fft", mw.istft_params["n_fft"], "hop", mw.istft_params["hop_len"])
torch.onnx.export(
    HiftHead(mw, _stftconv), (feat,), f"{OUT}/hift_head.onnx",
    input_names=["speech_feat"], output_names=["magnitude", "phase"],
    dynamic_axes={"speech_feat": {0: "b", 2: "T"}, "magnitude": {0: "b", 2: "T"}, "phase": {0: "b", 2: "T"}},
    opset_version=17, do_constant_folding=True,
)
print("hift_head.onnx nodes:", len(onnx.load(f"{OUT}/hift_head.onnx", load_external_data=False).graph.node))

# ── 3) Flow encoder (token + prompt + xvec -> mu, mask, spks, cond) ─────────
class FlowEnc(nn.Module):
    def __init__(self, flow): super().__init__(); self.f = flow
    def forward(self, token, prompt_token, prompt_feat, embedding):
        f = self.f
        emb = F.normalize(embedding, dim=1)
        spks = f.spk_embed_affine_layer(emb)                 # [1,80]
        tok = torch.cat([prompt_token, token], dim=1).long() # [1, Np+Nt]
        token_len = torch.tensor([tok.shape[1]], dtype=torch.long)
        mask1 = torch.ones(1, tok.shape[1], 1)
        te = f.input_embedding(tok) * mask1
        h, _ = f.encoder(te, token_len)                      # [1,T,C]
        h = f.encoder_proj(h)                                # [1,T,80]
        # cond = [prompt_feat ; zeros] along time (first mel_len1 = reference mel).
        # A `conds[:, :mel_len1] = prompt_feat` slice-assign does NOT survive
        # torch.onnx (it scatters ~nothing at a runtime mel_len1 != trace) — build
        # it with an explicit concat of prompt_feat and a dynamic zero tail.
        mel_len2 = h.shape[1] - prompt_feat.shape[1]
        tail = torch.zeros(1, mel_len2, 80, dtype=prompt_feat.dtype)
        conds = torch.cat([prompt_feat, tail], dim=1).transpose(1, 2)  # [1,80,T]
        mask = torch.ones(1, 1, h.shape[1])
        mu = h.transpose(1, 2).contiguous()                  # [1,80,T]
        return mu, mask, spks, conds
tok = torch.randint(0, 6000, (1, 40))
ptok = torch.randint(0, 6000, (1, 30))
pfeat = torch.randn(1, 60, 80)
xvec = torch.randn(1, 192)
try:
    with torch.no_grad():
        mu, mk, sp, cd = FlowEnc(flow)(tok, ptok, pfeat, xvec)
    print("flow_encoder out: mu", tuple(mu.shape), "mask", tuple(mk.shape), "spks", tuple(sp.shape), "cond", tuple(cd.shape))
    torch.onnx.export(
        FlowEnc(flow), (tok, ptok, pfeat, xvec), f"{OUT}/flow_encoder.onnx",
        input_names=["token", "prompt_token", "prompt_feat", "embedding"],
        output_names=["mu", "mask", "spks", "cond"],
        dynamic_axes={"token": {1: "Nt"}, "prompt_token": {1: "Np"}, "prompt_feat": {1: "Lp"},
                      "mu": {2: "T"}, "mask": {2: "T"}, "cond": {2: "T"}},
        opset_version=17, do_constant_folding=True,
    )
    print("flow_encoder.onnx nodes:", len(onnx.load(f"{OUT}/flow_encoder.onnx", load_external_data=False).graph.node))
except Exception as e:
    print("FLOW ENCODER export failed:", repr(e)[:300])

print("DONE")
