#!/usr/bin/env python3
"""SpeechTokenizer (fnlp/SpeechTokenizer) parity fixture: fold weight_norm,
export weights, dump encode/decode references + encoder(bidi)/decoder LSTM
pre/post intermediates.

Run: /tmp/snacenv/bin/python crates/rlx-speechtokenizer/scripts/gen_fixture.py
"""
import json, os
import numpy as np
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
from speechtokenizer import SpeechTokenizer
import speechtokenizer.modules.lstm as L

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

cfg_p = hf_hub_download("fnlp/SpeechTokenizer", "speechtokenizer_hubert_avg/config.json")
ckpt_p = hf_hub_download("fnlp/SpeechTokenizer", "speechtokenizer_hubert_avg/SpeechTokenizer.pt")
model = SpeechTokenizer.load_from_checkpoint(cfg_p, ckpt_p).eval()

lstm_calls = []
_orig = L.SLSTM.forward
def cap(self, x):
    pre = x.detach().cpu().numpy().astype(np.float32).copy()
    y = _orig(self, x)
    lstm_calls.append((pre, y.detach().cpu().numpy().astype(np.float32).copy()))
    return y
L.SLSTM.forward = cap

sd = model.state_dict()
folded, bases = {}, {k[:-len(".weight_g")] for k in sd if k.endswith(".weight_g")}
for k, v in sd.items():
    if k.endswith(".weight_g") or k.endswith(".weight_v"):
        continue
    folded[k] = v
for b in bases:
    g = sd[b + ".weight_g"]; v = sd[b + ".weight_v"]
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
    folded[b + ".weight"] = g * v / norm
export = {k: v.contiguous().to(torch.float32) for k, v in folded.items()
          if k.startswith(("encoder.", "decoder.", "quantizer."))}
save_file(export, os.path.join(OUT, "speechtokenizer.safetensors"))
print("weights:", len(export))

torch.manual_seed(5)
enc_len = 320 * 8  # hop=320 → 8 frames
wav = torch.randn(1, 1, enc_len)
with torch.no_grad():
    codes = model.encode(wav)          # [n_q, B, T]
    rec = model.decode(codes)          # [B, 1, T] or [B, T]
codes_np = codes[:, 0].cpu().numpy().astype(np.int64)  # [n_q, T]
rec = rec.squeeze().cpu().numpy().astype(np.float32)
enc_pre, enc_post = lstm_calls[0]      # encoder bidi LSTM first
dec_pre, dec_post = lstm_calls[1]
ref = {
    "pcm": wav[0, 0].cpu().numpy().astype(np.float32).tolist(),
    "codes": codes_np.tolist(),
    "wav": rec.reshape(-1).tolist(),
    "enc_lstm_pre": enc_pre.reshape(-1).tolist(),
    "enc_lstm_pre_shape": list(enc_pre.shape),
    "enc_lstm_post": enc_post.reshape(-1).tolist(),
    "enc_lstm_post_shape": list(enc_post.shape),
    "dec_lstm_pre": dec_pre.reshape(-1).tolist(),
    "dec_lstm_post": dec_post.reshape(-1).tolist(),
    "n_q": int(codes_np.shape[0]),
}
json.dump(ref, open(os.path.join(OUT, "speechtokenizer_ref.json"), "w"))
print("codes", codes_np.shape, "wav", rec.shape, "enc_lstm pre", enc_pre.shape, "post", enc_post.shape, "calls", len(lstm_calls))
