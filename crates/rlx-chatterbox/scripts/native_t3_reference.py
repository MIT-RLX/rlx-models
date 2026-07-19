#!/usr/bin/env python3
"""Numpy reference forward for the extracted native T3 Llama — the parity oracle
for the hand-authored `rlx-llama32` graph.

Implements a plain pre-norm Llama (RMSNorm, fused-QKV MHA, NeoX rope
theta=500000 NO scaling, SwiGLU, untied lm_head) directly on
weights/tts/chatterbox/native/t3_lm.safetensors, on a DETERMINISTIC
inputs_embeds, and dumps raw-f32 binaries the Rust parity harness reads back:

  native/parity_inputs_embeds.bin   [1, T, 1024]  f32 little-endian
  native/parity_ref_logits.bin      [1, T, 8194]  f32 little-endian
  native/parity_meta.json           {T, hidden, vocab}

This is a self-contained numeric gate: numpy-ref vs rlx-native, both on the SAME
extracted weights. It validates the rlx graph math (rope/norm/attention/mlp/head)
independent of onnxruntime. Whisper round-trip separately validates the math
matches ChatterBox.
"""
import json
import os

import numpy as np
from safetensors.numpy import load_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NAT = os.path.join(ROOT, "weights", "tts", "chatterbox", "native")

H, NH, HD, NL, VOCAB, INTER = 1024, 16, 64, 30, 8194, 4096
EPS, THETA = 1e-5, 500000.0
T = 8  # short prompt for a fast gate


def rmsnorm(x, w):
    v = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(v + EPS) * w).astype(np.float64)


def rope_tables(T):
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))  # [32]
    pos = np.arange(T, dtype=np.float64)[:, None]                        # [T,1]
    freqs = pos * inv[None, :]                                           # [T,32]
    emb = np.concatenate([freqs, freqs], axis=-1)                        # [T,64]
    return np.cos(emb), np.sin(emb)


def rotate_half(x):  # NeoX / HF style
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos, sin):  # x:[T,NH,HD] cos/sin:[T,HD]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return x * c + rotate_half(x) * s


def main():
    w = load_file(os.path.join(NAT, "t3_lm.safetensors"))
    w = {k: v.astype(np.float64) for k, v in w.items()}

    rng = np.random.default_rng(1234)
    emb = (rng.standard_normal((T, H)) * 0.1).astype(np.float64)  # deterministic

    cos, sin = rope_tables(T)
    mask = np.triu(np.full((T, T), -1e30, dtype=np.float64), k=1)

    h = emb.copy()
    for i in range(NL):
        lp = f"model.layers.{i}"
        # --- attention ---
        x = rmsnorm(h, w[f"{lp}.input_layernorm.weight"])
        qkv = x @ w[f"{lp}.self_attn.qkv.weight"].T           # [T,3072]
        q, k, v = qkv[:, :H], qkv[:, H:2 * H], qkv[:, 2 * H:]
        q = q.reshape(T, NH, HD); k = k.reshape(T, NH, HD); v = v.reshape(T, NH, HD)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        # [NH,T,T]
        scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(HD)
        scores = scores + mask[None]
        scores = scores - scores.max(-1, keepdims=True)
        p = np.exp(scores); p /= p.sum(-1, keepdims=True)
        ctx = np.einsum("hts,shd->thd", p, v).reshape(T, H)
        attn = ctx @ w[f"{lp}.self_attn.o_proj.weight"].T
        h = h + attn
        # --- mlp ---
        x = rmsnorm(h, w[f"{lp}.post_attention_layernorm.weight"])
        gate = x @ w[f"{lp}.mlp.gate_proj.weight"].T
        up = x @ w[f"{lp}.mlp.up_proj.weight"].T
        act = (gate / (1.0 + np.exp(-gate))) * up            # silu(gate)*up
        h = h + act @ w[f"{lp}.mlp.down_proj.weight"].T

    h = rmsnorm(h, w["model.norm.weight"])
    logits = h @ w["lm_head.weight"].T                        # [T,VOCAB]

    emb.astype("<f4").tofile(os.path.join(NAT, "parity_inputs_embeds.bin"))
    logits.astype("<f4").tofile(os.path.join(NAT, "parity_ref_logits.bin"))
    with open(os.path.join(NAT, "parity_meta.json"), "w") as f:
        json.dump({"T": T, "hidden": H, "vocab": VOCAB}, f)
    print(f"[ref] T={T} logits {logits.shape}  argmax(last)={int(logits[-1].argmax())}")
    print(f"[ref] last-row logits[:5]={np.round(logits[-1,:5],4)}")
    print("[ref] wrote parity_inputs_embeds.bin / parity_ref_logits.bin")


if __name__ == "__main__":
    main()
