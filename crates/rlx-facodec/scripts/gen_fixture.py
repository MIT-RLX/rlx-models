#!/usr/bin/env python3
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Generate a decoder-only parity fixture for FACodec (amphion/naturalspeech3_facodec).
# Feeds a random latent emb [1,256,T] + speaker embedding [1,256] straight into
# FACodecDecoder.inference (bypassing the VQ encoder) and dumps the folded
# (weight-norm-removed) decoder weights + reference waveform.
#
# Requires the Amphion `ns3_codec` module on disk (models/codec/ns3_codec from
# github.com/open-mmlab/Amphion). `melspec`/`pyworld` are stubbed since the
# decoder path does not touch them.
import sys, types, importlib.util, os, json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

NS3 = os.environ.get("NS3_CODEC_DIR", "/tmp/amphion_ns3")
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")

torch.manual_seed(0)
sys.modules["pyworld"] = types.ModuleType("pyworld")
pkg = types.ModuleType("ns3"); pkg.__path__ = [NS3]; sys.modules["ns3"] = pkg
mel = types.ModuleType("ns3.melspec"); mel.MelSpectrogram = type("M", (torch.nn.Module,), {})
sys.modules["ns3.melspec"] = mel
spec = importlib.util.spec_from_file_location("ns3.facodec", os.path.join(NS3, "facodec.py"), submodule_search_locations=[NS3])
m = importlib.util.module_from_spec(spec); sys.modules["ns3.facodec"] = m; spec.loader.exec_module(m)

dec = m.FACodecDecoder(in_channels=256, upsample_initial_channel=1024, ngf=32, up_ratios=(5, 5, 4, 2), vq_dim=256)
sd = torch.load(hf_hub_download("amphion/naturalspeech3_facodec", "ns3_facodec_decoder.bin"), map_location="cpu")
dec.load_state_dict(sd, strict=False)
dec.eval()
filt = [b for n, b in dec.named_buffers() if n.endswith("filter") and "model" in n][0].detach().reshape(-1)
dec.remove_weight_norm()

T = 48
emb = torch.randn(1, 256, T)
spk = torch.randn(1, 256)
with torch.no_grad():
    wav = dec.inference(emb, spk).cpu().numpy().reshape(-1)

out = {k: v.contiguous().float() for k, v in dec.state_dict().items() if k.startswith("model.") or k.startswith("timbre_linear")}
out["_filter"] = filt.float()
os.makedirs(OUT, exist_ok=True)
save_file(out, os.path.join(OUT, "facodec_dec.safetensors"))
json.dump({"emb": emb.reshape(-1).tolist(), "spk": spk.reshape(-1).tolist(), "wav": wav.tolist(), "T": T},
          open(os.path.join(OUT, "facodec_ref.json"), "w"))
print(f"saved {len(out)} tensors; wav {wav.shape}")
