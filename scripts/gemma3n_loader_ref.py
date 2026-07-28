#!/usr/bin/env python3
"""mlx reference for the rlx gemma-3n mlx-affine loader cross-check.

Dequantizes the same tensors `examples/mlx_loader_check.rs` dumps, using MLX's
own `mx.dequantize`, and prints len/finite/sum/head8 so the two can be compared
element-for-element. Proves the rlx loader's tensor naming + affine dequant +
bf16 scales/biases decode match mlx bit-for-bit.

Usage:  PYTHONPATH=/Users/Shared/mlx-lm python3 scripts/gemma3n_loader_ref.py <dir>
"""
import struct, json, sys
import numpy as np
import mlx.core as mx


def load_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    return hdr, base


def read_tensor(path, hdr, base, name):
    t = hdr[name]
    b0, b1 = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + b0)
        raw = f.read(b1 - b0)
    dt = t["dtype"]
    shape = t["shape"]
    if dt == "U32":
        arr = np.frombuffer(raw, dtype="<u4").reshape(shape)
        return mx.array(arr)
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return mx.array(u16.view(np.float32).reshape(shape))
    if dt in ("F32", "F16"):
        npdt = {"F32": "<f4", "F16": "<f2"}[dt]
        return mx.array(np.frombuffer(raw, dtype=npdt).astype(np.float32).reshape(shape))
    raise ValueError(dt)


def dequant(path, hdr, base, module, bits=4, gs=64):
    w = read_tensor(path, hdr, base, f"{module}.weight")
    s = read_tensor(path, hdr, base, f"{module}.scales")
    b = read_tensor(path, hdr, base, f"{module}.biases")
    out = mx.dequantize(w, s, b, group_size=gs, bits=bits)
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def show(tag, v):
    v = np.asarray(v).reshape(-1)
    print(f"{tag}: len={v.size} finite={np.isfinite(v).all()} "
          f"sum={float(v.astype(np.float64).sum()):.6f} head8={[round(float(x),6) for x in v[:8]]}")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else ".mlx-test/gemma3n-e2b-4bit"
    path = f"{d}/model.safetensors"
    hdr, base = load_header(path)
    P = "language_model.model"

    q = dequant(path, hdr, base, f"{P}.layers.0.self_attn.q_proj")
    print(f"q_proj shape={q.shape}"); show("q_proj.row0", q[0])

    dp = dequant(path, hdr, base, f"{P}.layers.0.mlp.down_proj")
    print(f"down_proj shape={dp.shape}"); show("down_proj.row0", dp[0])

    plmp = dequant(path, hdr, base, f"{P}.per_layer_model_projection")
    print(f"per_layer_model_projection shape={plmp.shape}"); show("plmp.row0", plmp[0])

    norm = np.array(read_tensor(path, hdr, base, f"{P}.norm.weight"))
    show("norm", norm)

    emb = dequant(path, hdr, base, f"{P}.embed_tokens")
    print(f"embed shape={emb.shape}")
    show("embed.tok2", emb[2])
    show("embed.tok818", emb[818])


if __name__ == "__main__":
    main()
