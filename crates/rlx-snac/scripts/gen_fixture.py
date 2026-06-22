#!/usr/bin/env python3
"""Generate the SNAC 24 kHz decoder parity fixture for rlx-snac.

Folds weight_norm, exports the decoder + quantizer weights to safetensors with
the key names rlx-snac expects, and runs the official SNAC decoder with
*deterministic* noise (captured + saved) so the Rust port can match it bit-exactly.

Run:  /tmp/snacenv/bin/python crates/rlx-snac/scripts/gen_fixture.py
"""
import json
import os
import torch
import numpy as np
from safetensors.torch import save_file
import snac.layers as L
from snac import SNAC

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

# --- deterministic noise: patch NoiseBlock to use a seeded generator + record ---
captured = []
gen = torch.Generator().manual_seed(20240617)

def patched_noise_forward(self, x):
    B, C, T = x.shape
    noise = torch.randn((B, 1, T), generator=gen, dtype=x.dtype)
    captured.append(noise[0, 0].cpu().numpy().astype(np.float32).copy())
    h = self.linear(x)
    return x + noise * h

L.NoiseBlock.forward = patched_noise_forward

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
sd = model.state_dict()

# --- fold weight_norm parametrizations: {base}.original0(g)/original1(v) -> {base}.weight ---
folded = {}
bases = set()
for k in sd:
    if k.endswith(".parametrizations.weight.original0"):
        bases.add(k[: -len(".parametrizations.weight.original0")])

for k, v in sd.items():
    if ".parametrizations.weight.original" in k:
        continue
    folded[k] = v

for base in bases:
    g = sd[base + ".parametrizations.weight.original0"]  # [out,1,1]
    v = sd[base + ".parametrizations.weight.original1"]  # [out,in,k]
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
    w = g * v / norm
    folded[base + ".weight"] = w

# keep encoder.* / decoder.* / quantizer.* (folded), as contiguous f32
export = {}
for k, v in folded.items():
    if k.startswith("decoder.") or k.startswith("quantizer.") or k.startswith("encoder."):
        export[k] = v.contiguous().to(torch.float32)
save_file(export, os.path.join(OUT, "snac24_decoder.safetensors"))
print("wrote weights:", len(export), "tensors")

# --- encode reference: fixed PCM (multiple of total stride) → latent + codes ---
torch.manual_seed(101)
enc_len = 512 * 8  # 8 latent frames (divisible by coarsest vq stride 4)
pcm = torch.randn(1, 1, enc_len)
with torch.no_grad():
    z_lat = model.encoder(pcm)          # [1, latent, T]
    _, enc_codes = model.quantizer(z_lat)
enc = {
    "pcm": pcm[0, 0].cpu().numpy().astype(np.float32).tolist(),
    "latent": z_lat[0].cpu().numpy().astype(np.float32).reshape(-1).tolist(),
    "latent_shape": list(z_lat.shape[1:]),
    "codes": [c[0].cpu().numpy().astype(np.int64).tolist() for c in enc_codes],
}
with open(os.path.join(OUT, "snac24_encode_ref.json"), "w") as f:
    json.dump(enc, f)
print("encode: pcm", enc_len, "latent", list(z_lat.shape[1:]), "codes", [len(c) for c in enc["codes"]])

# --- deterministic codes + reference decode ---
torch.manual_seed(7)
base_len = 6
strides = [4, 2, 1]
t_base = base_len * strides[0]
codes = []
for s in strides:
    n = t_base // s
    codes.append(torch.randint(0, 4096, (1, n), dtype=torch.long))

with torch.no_grad():
    wav = model.decode(codes)  # [1, 1, T]

wav = wav[0, 0].cpu().numpy().astype(np.float32)
ref = {
    "codes": [c[0].cpu().numpy().astype(np.int64).tolist() for c in codes],
    "noise": [n.tolist() for n in captured],
    "wav": wav.tolist(),
}
with open(os.path.join(OUT, "snac24_ref.json"), "w") as f:
    json.dump(ref, f)
print("codes levels:", [len(c) for c in ref["codes"]])
print("noise planes:", [len(n) for n in ref["noise"]])
print("wav samples:", len(ref["wav"]), "abs-max", float(np.abs(wav).max()))
