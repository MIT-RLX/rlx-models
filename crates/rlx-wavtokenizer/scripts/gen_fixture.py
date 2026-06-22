#!/usr/bin/env python3
"""WavTokenizer (novateur/WavTokenizer small 320/24k) decoder parity fixture.
Run from a checkout of github.com/jishengpeng/WavTokenizer on sys.path."""
import sys, json, os, numpy as np, torch
sys.path.insert(0, os.environ.get("WAVTOK_REPO", "/tmp/WavTokenizer"))
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from decoder.pretrained import WavTokenizer
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)
cfg = hf_hub_download("novateur/WavTokenizer", "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
ckpt = hf_hub_download("novateur/WavTokenizer", "WavTokenizer_small_320_24k_4096.ckpt")
m = WavTokenizer.from_pretrained0802(cfg, ckpt).eval()
sd = m.state_dict()
folded, bases = {}, {k[:-len(".weight_g")] for k in sd if k.endswith(".weight_g")}
for k, v in sd.items():
    if k.endswith(".weight_g") or k.endswith(".weight_v"): continue
    folded[k] = v
for b in bases:
    g, v = sd[b+".weight_g"], sd[b+".weight_v"]
    folded[b+".weight"] = g * v / v.flatten(1).norm(dim=1).view(-1, *([1]*(v.dim()-1)))
export = {k: t.contiguous().float() for k, t in folded.items() if k.startswith(("backbone.","head.","feature_extractor.","quantizer."))}
save_file(export, os.path.join(OUT, "wavtokenizer.safetensors"))
bw = torch.tensor([0]); torch.manual_seed(1); wav = torch.randn(1, 320*8)
with torch.no_grad():
    emb = m.feature_extractor.encodec.encoder(wav.unsqueeze(1))  # pre-VQ [1,512,T]
    feats, codes = m.encode_infer(wav, bandwidth_id=bw)
    out = m.decode(feats, bandwidth_id=bw)
ref = {
    "feats": feats[0].cpu().numpy().astype(np.float32).reshape(-1).tolist(),  # [512, T]
    "feats_shape": list(feats.shape[1:]),
    "codes": codes[0, 0].cpu().numpy().astype(np.int64).tolist(),  # [T]
    "wav": out[0].cpu().numpy().astype(np.float32).tolist(),
    "pcm": wav[0].cpu().numpy().astype(np.float32).tolist(),
    "emb": emb[0].cpu().numpy().astype(np.float32).reshape(-1).tolist(),
}
json.dump(ref, open(os.path.join(OUT, "wavtokenizer_ref.json"), "w"))
print("weights", len(export), "feats", feats.shape, "codes", codes.shape, "wav", out.shape)
