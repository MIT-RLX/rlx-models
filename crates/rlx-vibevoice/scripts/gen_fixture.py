#!/usr/bin/env python3
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Decode-only parity fixture for VibeVoice's acoustic σ-VAE tokenizer
# (microsoft/VibeVoice-1.5B). Feeds a random latent [1,64,T] straight into the
# TokenizerDecoder (bypassing the encoder). Uses the ORIGINAL modular file from
# the microsoft/VibeVoice repo (its weight keys match the checkpoint; the
# transformers-bundled port renames them). The trailing AutoModel.register lines
# are stripped to avoid double-registration.
import sys, types, importlib.util, json, os, urllib.request
import torch
from types import SimpleNamespace
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
URL = "https://raw.githubusercontent.com/microsoft/VibeVoice/main/vibevoice/modular/modular_vibevoice_tokenizer.py"
torch.manual_seed(0)

raw = urllib.request.urlopen(URL).read().decode()
raw = raw[: raw.find("AutoModel.register")]  # strip registration + __all__
path = "/tmp/vv_tok_stripped.py"
open(path, "w").write(raw)

pkg = types.ModuleType("vv"); pkg.__path__ = []; sys.modules["vv"] = pkg
cm = types.ModuleType("vv.configuration_vibevoice")
for n in ["VibeVoiceAcousticTokenizerConfig", "VibeVoiceSemanticTokenizerConfig"]:
    setattr(cm, n, type(n, (object,), {}))
sys.modules["vv.configuration_vibevoice"] = cm
spec = importlib.util.spec_from_file_location("vv.tok", path, submodule_search_locations=[])
m = importlib.util.module_from_spec(spec); m.__package__ = "vv"; sys.modules["vv.tok"] = m
spec.loader.exec_module(m)

cfg = SimpleNamespace(dimension=64, channels=1, n_filters=32, ratios=[8, 5, 5, 4, 2, 2],
    depths=[8, 3, 3, 3, 3, 3, 3], causal=True, kernel_size=7, last_kernel_size=7, norm="none",
    norm_params={}, pad_mode="constant", bias=True, layernorm="RMSNorm", layernorm_eps=1e-5,
    layernorm_elementwise_affine=True, trim_right_ratio=1.0, drop_path_rate=0.0,
    mixer_layer="depthwise_conv", layer_scale_init_value=1e-6, disable_last_norm=True, n_residual_layers=1)
dec = m.TokenizerDecoder(cfg).eval()

idx = json.load(open(hf_hub_download("microsoft/VibeVoice-1.5B", "model.safetensors.index.json")))
need = {k: v for k, v in idx["weight_map"].items() if "acoustic_tokenizer.decoder" in k}
store = {}
for sh in set(need.values()):
    store.update(load_file(hf_hub_download("microsoft/VibeVoice-1.5B", sh)))
sd = {k[len("model.acoustic_tokenizer.decoder."):]: store[k].float() for k in need}
miss, unexp = dec.load_state_dict(sd, strict=False)
assert not miss and not unexp, (len(miss), len(unexp))

T = 4
lat = torch.randn(1, 64, T) * 5.0  # scale up so the deep net produces a non-trivial signal
with torch.no_grad():
    wav = dec(lat).cpu().numpy().reshape(-1)
print("wav", wav.shape, "std", float(wav.std()), "max", float(abs(wav).max()))

os.makedirs(OUT, exist_ok=True)
save_file({k: v.contiguous().float() for k, v in dec.state_dict().items()}, os.path.join(OUT, "vv_dec.safetensors"))
json.dump({"latent": lat.reshape(64, T).reshape(-1).tolist(), "wav": wav.tolist(), "T": T},
          open(os.path.join(OUT, "vv_ref.json"), "w"))
print("saved", len(sd), "decoder tensors")
