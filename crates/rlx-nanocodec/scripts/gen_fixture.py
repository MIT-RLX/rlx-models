#!/usr/bin/env python3
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Decode-only parity fixture for NVIDIA NanoCodec (nvidia/nemo-nano-codec-22khz-*).
# Transcribes NeMo's verbatim forward math (audio_codec_modules.py + common/parts
# /utils.py): GroupFiniteScalarQuantizer.decode + CausalHiFiGANDecoder. No NeMo
# runtime needed — the .nemo is just a tar of model_config.yaml + model_weights.ckpt.
import os, json, tarfile, math
import torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

REPO = os.environ.get("NANO_REPO", "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
UP = [7, 7, 6, 3, 2]
BASE = 864
LEVELS = [9, 8, 8, 7]
NUM_GROUPS = 4
torch.manual_seed(0)

nemo = hf_hub_download(REPO, REPO.split("/")[-1] + ".nemo")
work = "/tmp/nanocodec"
os.makedirs(work, exist_ok=True)
with tarfile.open(nemo) as t:
    t.extractall(work)
sd = torch.load(os.path.join(work, "model_weights.ckpt"), map_location="cpu", weights_only=False)
if "state_dict" in sd:
    sd = sd["state_dict"]


def wn(prefix):
    """Fold weight-norm: w = g * v / ||v||_{dims>=1} (parametrizations.weight_norm, dim=0)."""
    g = sd[f"{prefix}.weight_g"].float()
    v = sd[f"{prefix}.weight_v"].float()
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
    w = g * v / norm
    b = sd[f"{prefix}.bias"].float()
    return w, b


def causal_conv(x, prefix, dilation=1):
    w, b = wn(prefix)
    k = w.shape[-1]
    pad = (k - 1) * dilation  # padding_total (stride 1), all on the left, zeros
    x = F.pad(x, (pad, 0))
    return F.conv1d(x, w, b, dilation=dilation)


def causal_convT(x, prefix, stride, groups):
    w, b = wn(prefix)
    k = w.shape[-1]
    y = F.conv_transpose1d(x, w, b, stride=stride, groups=groups)
    trim_right = k - stride  # padding_right (trim_right_ratio=1), padding_left=0
    return y[..., : y.shape[-1] - trim_right]


def snake(x, alpha, eps=1e-9):
    return x + (alpha + eps).reciprocal() * torch.sin(alpha * x).pow(2)


def half_snake(x, alpha):
    h = x.shape[1] // 2
    s = snake(x[:, :h, :], alpha)
    l = F.leaky_relu(x[:, h:, :], 0.01)
    return torch.cat([s, l], dim=1)


def residual_block(x, prefix, k, dil):
    a0 = sd[f"{prefix}.input_activation.activation.snake_act.alpha"].float()
    h = half_snake(x, a0)
    h = causal_conv(h, f"{prefix}.input_conv.conv", dilation=dil)
    a1 = sd[f"{prefix}.skip_activation.activation.snake_act.alpha"].float()
    h = half_snake(h, a1)
    h = causal_conv(h, f"{prefix}.skip_conv.conv", dilation=1)
    return x + h


def res_layer(x, prefix):
    # HiFiGANResLayer: mean over kernel-size blocks; each block = sequential over dilations.
    outs = []
    for ks_i, k in enumerate([3, 7, 11]):
        h = x
        for di, dil in enumerate([1, 3, 5]):
            h = residual_block(h, f"{prefix}.res_blocks.{ks_i}.res_blocks.{di}", k, dil)
        outs.append(h)
    return sum(outs) / len(outs)


def fsq_group_decode(indices, levels):
    # indices [B,T] int -> codes [B, D, T] continuous, centered.
    base = torch.cumprod(torch.tensor([1] + levels[:-1]), 0).view(1, -1, 1)
    nl = torch.tensor(levels).view(1, -1, 1)
    idx = indices.unsqueeze(1)  # [B,1,T]
    nonneg = (idx // base) % nl  # [B,D,T]
    scale = offset = nl // 2
    return (nonneg - offset) / scale


def decode(codes):
    # codes [num_groups, B, T] -> latent [B,16,T] -> wav
    groups = codes.chunk(NUM_GROUPS, dim=0)
    lat = torch.cat([fsq_group_decode(g.squeeze(0), LEVELS) for g in groups], dim=1).float()
    x = causal_conv(lat, "audio_decoder.pre_conv.conv")
    in_ch = BASE
    for i, rate in enumerate(UP):
        alpha = sd[f"audio_decoder.activations.{i}.activation.snake_act.alpha"].float()
        x = half_snake(x, alpha)
        out_ch = in_ch // 2
        x = causal_convT(x, f"audio_decoder.up_sample_conv_layers.{i}.conv", rate, groups=out_ch)
        x = res_layer(x, f"audio_decoder.res_layers.{i}")
        in_ch = out_ch
    alpha = sd["audio_decoder.post_activation.activation.snake_act.alpha"].float()
    x = half_snake(x, alpha)
    x = causal_conv(x, "audio_decoder.post_conv.conv")
    x = torch.clamp(x, -1.0, 1.0)
    return x.squeeze(1)


T = 16
codes = torch.stack([torch.randint(0, math.prod(LEVELS), (1, T)) for _ in range(NUM_GROUPS)], dim=0)
with torch.no_grad():
    wav = decode(codes).cpu().numpy().reshape(-1)
print("wav", wav.shape, "range", float(wav.min()), float(wav.max()))

# Export folded weights with Rust-friendly flat keys.
out = {}
for k in sd:
    if k.endswith(".weight_v"):
        prefix = k[: -len(".weight_v")]
        w, b = wn(prefix)
        out[prefix + ".weight"] = w.contiguous()
        out[prefix + ".bias"] = b.contiguous()
    elif k.endswith(".alpha"):
        out[k] = sd[k].float().contiguous()
os.makedirs(OUT, exist_ok=True)
save_file(out, os.path.join(OUT, "nano_dec.safetensors"))
json.dump({"codes": codes.reshape(NUM_GROUPS, T).tolist(), "wav": wav.tolist(), "T": T},
          open(os.path.join(OUT, "nano_ref.json"), "w"))
print("saved", len(out), "tensors")
