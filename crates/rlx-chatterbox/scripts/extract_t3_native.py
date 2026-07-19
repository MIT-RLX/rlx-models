#!/usr/bin/env python3
"""Extract the ChatterBox T3 Llama LM weights from the ONNX *weight container*
into a clean safetensors that ``rlx-llama32`` can load directly — enabling a
hand-authored native rlx graph (real KV-cache) instead of importing the ONNX
*graph*.

The ONNX `language_model_fp16.onnx` is used ONLY as a tensor store here; we do
NOT import its computation graph. Every tensor is a standard Llama tensor:

  onnx  model.layers.{i}.attn.qkv_proj.MatMul.weight  [in=1024, out=3072]  (fp16)
  ->    model.layers.{i}.self_attn.qkv.weight          [out=3072, in=1024]  (fp16)

ONNX MatMul stores weights as [in, out] (x @ W); PyTorch/HF Linear + rlx-llama32
expect [out, in], so every 2-D projection is transposed. Norms (1-D) pass through.
qkv stays FUSED — rlx-llama32 `load_self_attn_qkv` splits it by q/kv dims.

Also extracts the additive embedding tables from `embed_tokens.onnx`
(speech_emb / speech_pos_emb / text_emb / text_pos_emb) for the Rust-side
inputs-embeds construction, plus the baked rope tables (cos/sin) for parity.

Output: weights/tts/chatterbox/native/{t3_lm.safetensors, embed_tables.safetensors, rope_cache.safetensors, t3_config.json}
Deterministic, no download; stays bit-identical (fp16) to the validated pipeline.
"""
import json
import os
import re
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CB = os.path.join(ROOT, "weights", "tts", "chatterbox")
LM_ONNX = os.path.join(CB, "onnx", "language_model_fp16.onnx")
EMB_ONNX = os.path.join(CB, "onnx", "embed_tokens.onnx")
OUT = os.path.join(CB, "native")
os.makedirs(OUT, exist_ok=True)


def load_inits(path):
    m = onnx.load(path, load_external_data=True)
    return {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}


def main():
    print(f"[extract] loading {LM_ONNX}")
    lm = load_inits(LM_ONNX)

    # --- resolve true layer count from the graph (config says 30, graph has ids 0..?) ---
    layer_ids = sorted({int(x.group(1)) for n in lm if (x := re.search(r"layers\.(\d+)\.", n))})
    per_layer_keys = [
        "input_layernorm.weight",
        "attn.qkv_proj.MatMul.weight",
        "attn.o_proj.MatMul.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.MatMul.weight",
        "mlp.up_proj.MatMul.weight",
        "mlp.down_proj.MatMul.weight",
    ]
    complete = []
    for i in layer_ids:
        have = [k for k in per_layer_keys if f"model.layers.{i}.{k}" in lm]
        if len(have) == len(per_layer_keys):
            complete.append(i)
        else:
            print(f"[extract]   layer {i}: PARTIAL ({len(have)}/{len(per_layer_keys)}) -> {have}")
    n_layers = len(complete)
    assert complete == list(range(n_layers)), f"non-contiguous layers: {complete}"
    print(f"[extract] complete transformer layers: {n_layers} (ids {complete[0]}..{complete[-1]})")

    out = {}

    def T(name):  # transpose 2-D [in,out] -> [out,in]
        a = lm[name]
        assert a.ndim == 2, f"{name} not 2-D: {a.shape}"
        return np.ascontiguousarray(a.T)

    for i in complete:
        lp = f"model.layers.{i}"
        out[f"{lp}.input_layernorm.weight"] = lm[f"{lp}.input_layernorm.weight"]
        out[f"{lp}.post_attention_layernorm.weight"] = lm[f"{lp}.post_attention_layernorm.weight"]
        # fused qkv onnx [in=1024, out=3072] -> HF [out=3072, in=1024]; rows
        # [0:1024]=q [1024:2048]=k [2048:3072]=v. rlx-flow wants SEPARATE
        # q_proj/k_proj/v_proj (each HF [1024,1024]); it transposes on load.
        qkv = T(f"{lp}.attn.qkv_proj.MatMul.weight")  # [3072,1024]
        H2 = 1024
        out[f"{lp}.self_attn.q_proj.weight"] = np.ascontiguousarray(qkv[0:H2])
        out[f"{lp}.self_attn.k_proj.weight"] = np.ascontiguousarray(qkv[H2:2 * H2])
        out[f"{lp}.self_attn.v_proj.weight"] = np.ascontiguousarray(qkv[2 * H2:3 * H2])
        out[f"{lp}.self_attn.o_proj.weight"] = T(f"{lp}.attn.o_proj.MatMul.weight")
        out[f"{lp}.mlp.gate_proj.weight"] = T(f"{lp}.mlp.gate_proj.MatMul.weight")
        out[f"{lp}.mlp.up_proj.weight"] = T(f"{lp}.mlp.up_proj.MatMul.weight")
        out[f"{lp}.mlp.down_proj.weight"] = T(f"{lp}.mlp.down_proj.MatMul.weight")

    # final norm — the graph names it as a phantom "layer 30" final_norm_layernorm
    # (SkipSimplifiedLayerNormalization before the lm_head MatMul).
    norm_key = next(
        (k for k in lm if k in ("model.norm.weight", "norm.weight")),
        None,
    )
    if norm_key is None:
        norm_key = next(k for k in lm if k.endswith("final_norm_layernorm.weight"))
    out["model.norm.weight"] = lm[norm_key]
    print(f"[extract] final norm <- {norm_key} {lm[norm_key].shape}")

    # untied lm head: onnx [1024,8194] -> [8194,1024]
    out["lm_head.weight"] = T("lm_head.MatMul.weight")
    print(f"[extract] lm_head {out['lm_head.weight'].shape} (untied)")

    save_file(out, os.path.join(OUT, "t3_lm.safetensors"))
    print(f"[extract] wrote t3_lm.safetensors ({len(out)} tensors)")

    # --- embedding tables (additive; used to build inputs_embeds in Rust) ---
    print(f"[extract] loading {EMB_ONNX}")
    emb = load_inits(EMB_ONNX)
    tables = {}
    for k in ("text_emb.weight", "speech_emb.weight", "text_pos_emb.weight", "speech_pos_emb.weight"):
        if k in emb:
            tables[k] = emb[k].astype(np.float32)
            print(f"[extract]   {k} {emb[k].shape}")
    save_file(tables, os.path.join(OUT, "embed_tables.safetensors"))

    # --- baked rope tables (ground truth for parity check vs llama3 recompute) ---
    rope = {}
    for k in ("cos_cache", "sin_cache"):
        if k in lm:
            rope[k] = lm[k].astype(np.float32)
            print(f"[extract]   {k} {lm[k].shape}")
    if rope:
        save_file(rope, os.path.join(OUT, "rope_cache.safetensors"))

    cfg = {
        "vocab_size": 8194,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": n_layers,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 64,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        # NOTE: config.json declares rope_scaling=llama3, but the ONNX export
        # baked PLAIN rope (verified: reconstructed inv_freq matches plain
        # theta=500000 to 1.2e-4 fp16 noise; llama3 gives Δcos=1.83). Use plain
        # rope (rope_scaling=None) for parity with the validated pipeline.
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "hidden_act": "silu",
        "max_position_embeddings": 131072,
    }
    with open(os.path.join(OUT, "t3_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[extract] wrote t3_config.json (num_hidden_layers={n_layers})")
    print("[extract] DONE ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
