#!/usr/bin/env python3
"""EnCodec 24 kHz parity fixture: fold weight_norm, export weights (HF key
names), and dump encode/decode references + pre/post-LSTM intermediates.

Run: /tmp/snacenv/bin/python crates/rlx-encodec/scripts/gen_fixture.py
"""
import json, os
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import EncodecModel

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

model = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()
cfg = model.config

# capture EVERY EncodecLSTM call (encoder fires first, decoder later) so we can
# pick the encoder one regardless of call order.
lstm_calls = []
import transformers.models.encodec.modeling_encodec as menc
_orig = menc.EncodecLSTM.forward
def cap_forward(self, x):
    pre = x.detach().cpu().numpy().astype(np.float32).copy()
    y = _orig(self, x)
    lstm_calls.append((pre, y.detach().cpu().numpy().astype(np.float32).copy()))
    return y
menc.EncodecLSTM.forward = cap_forward

sd = model.state_dict()
folded = {}
bases = {k[: -len(".parametrizations.weight.original0")] for k in sd if k.endswith(".parametrizations.weight.original0")}
for k, v in sd.items():
    if ".parametrizations.weight.original" in k:
        continue
    folded[k] = v
for b in bases:
    g = sd[b + ".parametrizations.weight.original0"]
    v = sd[b + ".parametrizations.weight.original1"]
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
    folded[b + ".weight"] = g * v / norm
export = {k: v.contiguous().to(torch.float32) for k, v in folded.items()
          if k.startswith(("encoder.", "decoder.", "quantizer."))}
save_file(export, os.path.join(OUT, "encodec24.safetensors"))
print("weights:", len(export))

# --- encode/decode reference at 6 kbps (8 codebooks) ---
torch.manual_seed(3)
enc_len = 320 * 8  # hop=320 → 8 latent frames
pcm = torch.randn(1, 1, enc_len)
with torch.no_grad():
    enc = model.encode(pcm, bandwidth=6.0)
    codes = enc.audio_codes  # [1, 1, n_q, T]
    audio = model.decode(codes, enc.audio_scales)[0]  # [1,1,T]

codes_np = codes[0, 0].cpu().numpy().astype(np.int64)  # [n_q, T]
enc_pre, enc_post = lstm_calls[0]  # encoder LSTM fires first
ref = {
    "pcm": pcm[0, 0].cpu().numpy().astype(np.float32).tolist(),
    "codes": codes_np.tolist(),                       # [n_q][T]
    "wav": audio[0, 0].cpu().numpy().astype(np.float32).tolist(),
    "lstm_pre": enc_pre.reshape(-1).tolist(),         # encoder LSTM input  [B,C,T]→flat
    "lstm_pre_shape": list(enc_pre.shape),
    "lstm_post": enc_post.reshape(-1).tolist(),
    "n_q": int(codes_np.shape[0]),
}
with open(os.path.join(OUT, "encodec24_ref.json"), "w") as f:
    json.dump(ref, f)
print("codes", codes_np.shape, "wav", audio.shape[-1], "lstm_pre", enc_pre.shape, "lstm calls", len(lstm_calls))
