#!/usr/bin/env python3
"""Truncate the mlx-community gemma-3n E2B 4bit checkpoint to N text layers.

Writes two dirs:
  <DST>          : rlx layout  (language_model.model.*), single model.safetensors
  <DST>-mlxlm    : mlx-lm layout (model.language_model.*)

Handles the per-layer packed tables (embed_tokens_per_layer col-slice +
per_layer_model_projection row-slice) so the depth-N config is self-consistent.

Usage: trunc_gemma3n.py SRC DST N
"""
import json, os, re, sys
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

SRC, DST, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(f"{SRC}/config.json"))
tc = cfg["text_config"]
PW = tc["hidden_size_per_layer_input"]        # 256
NL_FULL = tc["num_hidden_layers"]             # 30
GS = cfg["quantization"]["group_size"]        # 64
VPU = 8                                        # 4-bit values per uint32 word

# For depth N we want BOTH attention types present (mlx-lm indexes both).
layer_types = list(tc["layer_types"])[:N]
if "full_attention" not in layer_types:
    layer_types[-1] = "full_attention"        # relabel the last kept layer
if "sliding_attention" not in layer_types:
    layer_types[0] = "sliding_attention"

f = safe_open(f"{SRC}/model.safetensors", "pt")
keys = list(f.keys())


def keep(k):
    if not k.startswith("language_model.model."):
        return False                          # drop vision/audio towers
    m = re.match(r"language_model\.model\.layers\.(\d+)\.", k)
    if m:
        return int(m.group(1)) < N
    return True                               # embed/norm/altup/per_layer_*


tensors = {}
for k in keys:
    if not keep(k):
        continue
    tt = f.get_tensor(k)
    if tt.dtype == torch.bfloat16:
        tt = tt.float()                       # bf16 unsupported by numpy; upcast
    t = tt.numpy()
    # Column-slice the layer-major per-layer embedding table to N layers.
    if k == "language_model.model.embed_tokens_per_layer.weight":
        t = t[:, : N * PW // VPU]             # packed uint32 words
    elif k == "language_model.model.embed_tokens_per_layer.scales":
        t = t[:, : N * PW // GS]
    elif k == "language_model.model.embed_tokens_per_layer.biases":
        t = t[:, : N * PW // GS]
    # Row-slice the per-layer model projection (out = NL*PW) to N layers.
    elif k == "language_model.model.per_layer_model_projection.weight":
        t = t[: N * PW, :]
    elif k == "language_model.model.per_layer_model_projection.scales":
        t = t[: N * PW, :]
    elif k == "language_model.model.per_layer_model_projection.biases":
        t = t[: N * PW, :]
    tensors[k] = np.ascontiguousarray(t)

os.makedirs(DST, exist_ok=True)
save_file(tensors, f"{DST}/model.safetensors")

# Truncated config (self-consistent at depth N, no KV sharing).
tc2 = dict(tc)
tc2["num_hidden_layers"] = N
tc2["num_kv_shared_layers"] = 0
tc2["layer_types"] = layer_types
tc2["activation_sparsity_pattern"] = list(tc["activation_sparsity_pattern"])[:N]
if isinstance(tc.get("intermediate_size"), list):
    tc2["intermediate_size"] = list(tc["intermediate_size"])[:N]
cfg2 = dict(cfg)
cfg2["text_config"] = tc2
json.dump(cfg2, open(f"{DST}/config.json", "w"), indent=1)
for extra in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json"):
    p = f"{SRC}/{extra}"
    if os.path.exists(p):
        open(f"{DST}/{extra}", "wb").write(open(p, "rb").read())
print(f"[rlx]   wrote {len(tensors)} tensors → {DST}  (N={N}, layer_types={layer_types})")

# ── mlx-lm layout: language_model.model.* → model.language_model.* ──
MDST = f"{DST}-mlxlm"
os.makedirs(MDST, exist_ok=True)
PFX = "language_model.model."
ren = {"model.language_model." + k[len(PFX):]: v for k, v in tensors.items()}
save_file(ren, f"{MDST}/model.safetensors")
json.dump(cfg2, open(f"{MDST}/config.json", "w"), indent=1)
for extra in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json"):
    p = f"{SRC}/{extra}"
    if os.path.exists(p):
        open(f"{MDST}/{extra}", "wb").write(open(p, "rb").read())
print(f"[mlxlm] wrote {len(ren)} tensors → {MDST}")
