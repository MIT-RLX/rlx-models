#!/usr/bin/env python3
"""Compare RoPE+Householder vs TorchScript internals (debug)."""
from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import load_file

SQRT3 = 3**0.5


def apply_rope3d(x, pos, log_freq, reflect_vec, eye, tau=300.0):
    # x: (B,H,N,D), pos: (B,N,3)
    b, h, n, d = x.shape
    pos_n = pos / tau
    pos5 = pos_n.reshape(b, 1, n, 1, 3)
    freq = pos5 * torch.exp(log_freq) / SQRT3
    cos = torch.cos(freq)
    sin = torch.sin(freq)
    cos = torch.where(torch.isnan(cos), torch.ones_like(cos), cos)
    sin = torch.where(torch.isnan(sin), torch.zeros_like(sin), sin)
    cos = cos.repeat_interleave(2, dim=-1).reshape(b, h, n, d)
    sin = sin.repeat_interleave(2, dim=-1).reshape(b, h, n, d)
    x_odd = -x[..., 1::2]
    x_even = x[..., 0::2]
    rot = torch.stack([x_odd, x_even], dim=-1).reshape_as(x)
    rotated = x * cos + rot * sin
    np.save("/tmp/hoct_rope_rotated.npy", rotated.numpy())
    v = reflect_vec / reflect_vec.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    outer = v.unsqueeze(-1) * v.unsqueeze(-2)
    refl = eye - 2 * outer
    return torch.einsum("b h n d, h d e -> b h n e", rotated, refl)


def main() -> None:
    torch.manual_seed(0)
    sd = load_file("/tmp/hoct-inspect/weights/general_v0.safetensors")
    prefix = "node_blocks.0.attn.pos_enc"
    log_freq = sd[f"{prefix}.log_freq"]
    reflect_vec = sd[f"{prefix}.reflect_vec"]
    eye = sd[f"{prefix}.eye"]
    x = torch.randn(1, 4, 8, 72)
    pos = torch.randn(1, 8, 3) * 100
    y = apply_rope3d(x, pos, log_freq, reflect_vec, eye)
    print("python rope sample", y[0, 0, 0, :5].tolist())
    np.save("/tmp/hoct_rope_ref.npy", y.numpy())
    np.save("/tmp/hoct_rope_x.npy", x.numpy())
    np.save("/tmp/hoct_rope_pos.npy", pos.numpy())


if __name__ == "__main__":
    main()
