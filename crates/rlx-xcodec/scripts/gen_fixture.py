#!/usr/bin/env python3
"""XCodec2 (HKUSTAudio/xcodec2) decoder-backbone parity fixture.
Random continuous embedding [1024,T] → generator(vq=False) → wav. Avoids w2v-BERT."""
import sys, importlib.util, os, types, json, numpy as np, torch
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors.torch import save_file, load_file
repo = snapshot_download("HKUSTAudio/xcodec2", allow_patterns=["*.py","vq/*.py","config.json"])
sys.path.insert(0, repo)
pkg=types.ModuleType("vq"); pkg.__path__=[os.path.join(repo,"vq")]; sys.modules["vq"]=pkg
def load(n):
    s=importlib.util.spec_from_file_location(f"vq.{n}",os.path.join(repo,"vq",f"{n}.py")); m=importlib.util.module_from_spec(s); sys.modules[f"vq.{n}"]=m; s.loader.exec_module(m); return m
for d in ["module","bs_roformer5","codec_decoder_vocos"]: load(d)
import vq.codec_decoder_vocos as C
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

gen = C.CodecDecoderVocos().eval()
# load generator weights from the full model.safetensors
msd = load_file(hf_hub_download("HKUSTAudio/xcodec2", "model.safetensors"))
gsd = {k[len("generator."):]: v for k, v in msd.items() if k.startswith("generator.")}
# fold any weight_norm
folded, bases = {}, {k[:-9] for k in gsd if k.endswith(".weight_g")}
for k,v in gsd.items():
    if k.endswith(".weight_g") or k.endswith(".weight_v"): continue
    folded[k]=v
for b in bases:
    g,v=gsd[b+".weight_g"],gsd[b+".weight_v"]; folded[b+".weight"]=g*v/v.flatten(1).norm(dim=1).view(-1,*([1]*(v.dim()-1)))
missing, unexpected = gen.load_state_dict(folded, strict=False)
print("missing", len(missing), "unexpected", len(unexpected), "e.g.", missing[:2], unexpected[:2])
export = {k: v.contiguous().float() for k,v in folded.items() if k.startswith(("backbone.","head."))}
save_file(export, os.path.join(OUT, "xcodec_dec.safetensors"))

torch.manual_seed(2); T=8
emb = torch.randn(1, T, 1024)  # [B, T, 1024] (generator input layout)
with torch.no_grad():
    wav = gen(emb, vq=False)[0]
ref = {"emb": emb[0].cpu().numpy().astype(np.float32).reshape(-1).tolist(), "emb_shape": [T,1024],
       "wav": wav.squeeze().cpu().numpy().astype(np.float32).tolist()}
json.dump(ref, open(os.path.join(OUT,"xcodec_ref.json"),"w"))
print("weights", len(export), "emb", emb.shape, "wav", wav.shape)
