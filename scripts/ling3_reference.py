#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Ling 3.0 / BailingMoeV3 reference forward, for rlx-ling parity tests.

Emits a self-contained fixture (config + synthetic safetensors + reference
logits) that `crates/rlx-ling/tests/parity_reference.rs` replays through the RLX
graph:

    python3 scripts/ling3_reference.py .fixtures/ling3-parity
    RLX_LING_PARITY_DIR=.fixtures/ling3-parity cargo test -p rlx-ling --test parity_reference

This is a *transcription* of `modeling_bailing_moe_v3.py` plus the FLA kernels it
calls, not an import of them: `chunk_kda` / `fused_recurrent_kda` need Triton, so
the delta-net recurrence is written out sequentially here. The pieces that were
easy to get wrong upstream, and their sources:

  * KDA gate — `fla/ops/kda/gate.py::naive_kda_lowerbound_gate`. With
    `kda_lower_bound` set the gate is `lb·σ(exp(A_log)·(g + dt_bias))`, NOT a
    clamped `-exp(A_log)·softplus(·)`. `A_log` is per head.
  * Output norm — `fla/modules/fused_norm_gate.py`. The `sigmoid` gate is applied
    *after* normalising and scaling by the weight.
  * The recurrence itself matches `Op::GatedDeltaNet{gate_per_channel}`
    (`rlx-cpu/src/gdn.rs::gdn_step_blas_pc`): decay the key rows, delta-correct
    against the decayed state, read out with `q/√n`.
"""

import argparse
import json
import math
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

# A scaled-down Ling-3.0-tiny: same structural relationships (layer_group_size
# cycle, MLA head split, grouped router, one dense layer), small enough to run in
# a test. Must stay in sync with `tiny_config()` in the Rust parity test.
CONFIG = {
    "model_type": "bailing_hybrid",
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 24,
    "num_hidden_layers": 8,
    "num_attention_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 600000.0,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_shared_experts": 1,
    "moe_intermediate_size": 8,
    "moe_shared_expert_intermediate_size": 8,
    "n_group": 2,
    "topk_group": 1,
    "routed_scaling_factor": 2.5,
    "first_k_dense_replace": 1,
    "q_lora_rank": 12,
    "kv_lora_rank": 10,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 4,
    "v_head_dim": 8,
    "rope_interleave": True,
    "gated_attention_proj_granularity_type": "head_wise",
    "layer_group_size": 4,
    "short_conv_kernel_size": 4,
    "no_kda_lora": True,
    "kda_safe_gate": True,
    "kda_lower_bound": -5.0,
    "tie_word_embeddings": False,
}

L2NORM_EPS = 1e-6


def attn_kind(cfg, i):
    gs = cfg["layer_group_size"]
    whole = cfg["num_hidden_layers"] // gs * gs
    return "mla" if ((i + 1) % gs == 0 or i >= whole) else "kda"


def is_moe(cfg, i):
    return cfg["num_experts"] > 0 and i >= cfg["first_k_dense_replace"]


def make_weights(cfg, gen):
    """Synthetic checkpoint in HF Bailing naming/layout."""

    def w(*shape):
        # Small values keep the MoE router away from ties, where a top-k
        # tie-break difference would show up as a spurious parity failure.
        return (torch.rand(*shape, generator=gen, dtype=torch.float32) - 0.5) * 0.2

    h = cfg["hidden_size"]
    hh = cfg["num_attention_heads"]
    hd = cfg["head_dim"]
    proj = hh * hd
    qk = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    ql, kvl = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    rope, nope, vd = cfg["qk_rope_head_dim"], cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    mi = cfg["moe_intermediate_size"]
    si = cfg["moe_shared_expert_intermediate_size"] * cfg["num_shared_experts"]
    e = cfg["num_experts"]

    t = {"model.word_embeddings.weight": w(cfg["vocab_size"], h)}
    for i in range(cfg["num_hidden_layers"]):
        lp = f"model.layers.{i}"
        t[f"{lp}.input_layernorm.weight"] = w(h) + 1.0
        t[f"{lp}.post_attention_layernorm.weight"] = w(h) + 1.0
        at = f"{lp}.attention"
        if attn_kind(cfg, i) == "mla":
            t[f"{at}.q_a_proj.weight"] = w(ql, h)
            t[f"{at}.q_a_layernorm.weight"] = w(ql) + 1.0
            t[f"{at}.q_b_proj.weight"] = w(hh * qk, ql)
            t[f"{at}.kv_a_proj_with_mqa.weight"] = w(kvl + rope, h)
            t[f"{at}.kv_a_layernorm.weight"] = w(kvl) + 1.0
            t[f"{at}.kv_b_proj.weight"] = w(hh * (nope + vd), kvl)
            t[f"{at}.g_proj.weight"] = w(hh, h)
            t[f"{at}.dense.weight"] = w(h, hh * vd)
        else:
            for p in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
                t[f"{at}.{p}.weight"] = w(proj, h)
            for c in ("q_conv1d", "k_conv1d", "v_conv1d"):
                t[f"{at}.{c}.weight"] = w(proj, 1, cfg["short_conv_kernel_size"])
            t[f"{at}.b_proj.weight"] = w(hh, h)
            # A_log = log(U(1,16)) as in the reference init; per head.
            t[f"{at}.A_log"] = torch.log(
                torch.rand(hh, generator=gen, dtype=torch.float32) * 15.0 + 1.0
            )
            t[f"{at}.dt_bias"] = w(proj)
            t[f"{at}.o_norm.weight"] = w(hd) + 1.0
            t[f"{at}.o_proj.weight"] = w(h, proj)
        mlp = f"{lp}.mlp"
        if is_moe(cfg, i):
            t[f"{mlp}.gate.weight"] = w(e, h)
            t[f"{mlp}.gate.expert_bias"] = w(e)
            for ei in range(e):
                b = f"{mlp}.experts.{ei}"
                t[f"{b}.gate_proj.weight"] = w(mi, h)
                t[f"{b}.up_proj.weight"] = w(mi, h)
                t[f"{b}.down_proj.weight"] = w(h, mi)
            t[f"{mlp}.shared_experts.gate_proj.weight"] = w(si, h)
            t[f"{mlp}.shared_experts.up_proj.weight"] = w(si, h)
            t[f"{mlp}.shared_experts.down_proj.weight"] = w(h, si)
        else:
            di = cfg["intermediate_size"]
            t[f"{mlp}.gate_proj.weight"] = w(di, h)
            t[f"{mlp}.up_proj.weight"] = w(di, h)
            t[f"{mlp}.down_proj.weight"] = w(h, di)
    t["model.norm.weight"] = w(h) + 1.0
    t["lm_head.weight"] = w(cfg["vocab_size"], h)
    return t


# ─────────────────────────── layer primitives ───────────────────────────


def rms_norm(x, weight, eps):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return weight * (x.float() * torch.rsqrt(var + eps))


def l2norm(x, eps=L2NORM_EPS):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_interleave(x, cos, sin):
    """`apply_rotary_pos_emb_interleave` on `[s, heads, rope]`.

    De-interleaves each head's pairs into half-split order, then rotates. cos/sin
    are `[s, rope/2]`, doubled to `[s, rope]` as `cat(freqs, freqs)`.
    """
    s, hh, d = x.shape
    x = x.view(s, hh, d // 2, 2).transpose(3, 2).reshape(s, hh, d)
    c = torch.cat((cos, cos), dim=-1).unsqueeze(1)
    n = torch.cat((sin, sin), dim=-1).unsqueeze(1)
    return x * c + rotate_half(x) * n


def short_conv_silu(x, weight, k):
    """Causal depthwise conv1d (`ShortConvolution`, activation silu) on `[s, c]`."""
    s, c = x.shape
    padded = torch.cat([x.new_zeros(k - 1, c), x], dim=0)
    out = x.new_zeros(s, c)
    for j in range(k):
        out = out + padded[j : j + s] * weight[:, 0, j]
    return F.silu(out)


def kda_layer(cfg, t, at, x):
    """`BailingMoeV3KimiDeltaAttention.forward` on `[s, hidden]`."""
    hh, hd = cfg["num_attention_heads"], cfg["head_dim"]
    s = x.shape[0]
    k_size = cfg["short_conv_kernel_size"]

    q = short_conv_silu(x @ t[f"{at}.q_proj.weight"].T, t[f"{at}.q_conv1d.weight"], k_size)
    k = short_conv_silu(x @ t[f"{at}.k_proj.weight"].T, t[f"{at}.k_conv1d.weight"], k_size)
    v = short_conv_silu(x @ t[f"{at}.v_proj.weight"].T, t[f"{at}.v_conv1d.weight"], k_size)
    q = l2norm(q.view(s, hh, hd))
    k = l2norm(k.view(s, hh, hd))
    v = v.view(s, hh, hd)

    g = (x @ t[f"{at}.f_proj.weight"].T).view(s, hh, hd) + t[f"{at}.dt_bias"].view(hh, hd)
    a_exp = torch.exp(t[f"{at}.A_log"]).view(1, hh, 1)
    lb = cfg["kda_lower_bound"]
    if lb is not None:
        g_log = lb * torch.sigmoid(a_exp * g)
    else:
        g_log = -a_exp * F.softplus(g)
    beta = torch.sigmoid(x @ t[f"{at}.b_proj.weight"].T)  # [s, hh]

    # Gated delta rule, state [hh, hd(key), hd(value)].
    scale = 1.0 / math.sqrt(hd)
    state = x.new_zeros(hh, hd, hd)
    o = x.new_zeros(s, hh, hd)
    for ti in range(s):
        state = state * torch.exp(g_log[ti]).unsqueeze(-1)  # decay key rows
        sk = torch.einsum("hij,hi->hj", state, k[ti])
        d = (v[ti] - sk) * beta[ti].unsqueeze(-1)
        state = state + torch.einsum("hi,hj->hij", k[ti], d)
        o[ti] = torch.einsum("hij,hi->hj", state, q[ti]) * scale

    # FusedRMSNormGated(sigmoid): normalise+scale, then gate.
    normed = rms_norm(o, t[f"{at}.o_norm.weight"], cfg["rms_norm_eps"])
    gate = torch.sigmoid((x @ t[f"{at}.g_proj.weight"].T).view(s, hh, hd))
    return (normed * gate).reshape(s, hh * hd) @ t[f"{at}.o_proj.weight"].T


def mla_layer(cfg, t, at, x, cos, sin):
    """`BailingMoeV3MultiLatentAttention.forward` on `[s, hidden]`."""
    hh = cfg["num_attention_heads"]
    nope, rope, vd = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    qk = nope + rope
    s = x.shape[0]

    q = rms_norm(x @ t[f"{at}.q_a_proj.weight"].T, t[f"{at}.q_a_layernorm.weight"], cfg["rms_norm_eps"])
    q = (q @ t[f"{at}.q_b_proj.weight"].T).view(s, hh, qk)
    q_nope, q_rot = q[..., :nope], q[..., nope:]

    ckv = x @ t[f"{at}.kv_a_proj_with_mqa.weight"].T
    k_lora, k_rot = ckv[:, : cfg["kv_lora_rank"]], ckv[:, cfg["kv_lora_rank"] :]
    kv = rms_norm(k_lora, t[f"{at}.kv_a_layernorm.weight"], cfg["rms_norm_eps"])
    kv = (kv @ t[f"{at}.kv_b_proj.weight"].T).view(s, hh, nope + vd)
    k_nope, value = kv[..., :nope], kv[..., nope:]

    q_rot = apply_rope_interleave(q_rot, cos, sin)
    k_rot = apply_rope_interleave(k_rot.view(s, 1, rope), cos, sin).expand(s, hh, rope)

    query = torch.cat([q_nope, q_rot], dim=-1).transpose(0, 1)  # [hh, s, qk]
    key = torch.cat([k_nope, k_rot], dim=-1).transpose(0, 1)
    val = value.transpose(0, 1)  # [hh, s, vd]

    scores = query @ key.transpose(1, 2) * (qk**-0.5)
    mask = torch.full((s, s), float("-inf")).triu(1)
    probs = torch.softmax(scores + mask, dim=-1, dtype=torch.float32)
    o = (probs @ val).transpose(0, 1)  # [s, hh, vd]

    if cfg["gated_attention_proj_granularity_type"] == "head_wise":
        o = o * torch.sigmoid(x @ t[f"{at}.g_proj.weight"].T).unsqueeze(-1)
    elif cfg["gated_attention_proj_granularity_type"] == "element_wise":
        o = o * torch.sigmoid((x @ t[f"{at}.g_proj.weight"].T).view(s, hh, vd))
    return o.reshape(s, hh * vd) @ t[f"{at}.dense.weight"].T


def group_limited_topk(cfg, scores):
    """`BailingMoeV3Gate.group_limited_topk` — top-2-per-group group score."""
    n_tok, n_exp = scores.shape
    n_group, topk_group = cfg["n_group"], cfg["topk_group"]
    group_scores = scores.view(n_tok, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(n_tok, n_group, n_exp // n_group)
        .reshape(n_tok, -1)
    )
    masked = scores.masked_fill(~score_mask.bool(), float("-inf"))
    return torch.topk(masked, k=cfg["num_experts_per_tok"], dim=-1)[1]


def moe_layer(cfg, t, mlp, x):
    logits = x @ t[f"{mlp}.gate.weight"].T
    scores = torch.sigmoid(logits)
    topk_idx = group_limited_topk(cfg, scores + t[f"{mlp}.gate.expert_bias"])
    picked = torch.gather(scores, 1, topk_idx)
    weights = picked / (picked.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * cfg["routed_scaling_factor"]

    out = torch.zeros_like(x)
    for ki in range(cfg["num_experts_per_tok"]):
        for ti in range(x.shape[0]):
            b = f"{mlp}.experts.{int(topk_idx[ti, ki])}"
            xi = x[ti : ti + 1]
            y = (F.silu(xi @ t[f"{b}.gate_proj.weight"].T) * (xi @ t[f"{b}.up_proj.weight"].T))
            out[ti] += (y @ t[f"{b}.down_proj.weight"].T)[0] * weights[ti, ki]

    sh = f"{mlp}.shared_experts"
    y = F.silu(x @ t[f"{sh}.gate_proj.weight"].T) * (x @ t[f"{sh}.up_proj.weight"].T)
    return out + y @ t[f"{sh}.down_proj.weight"].T


def dense_mlp(t, mlp, x):
    y = F.silu(x @ t[f"{mlp}.gate_proj.weight"].T) * (x @ t[f"{mlp}.up_proj.weight"].T)
    return y @ t[f"{mlp}.down_proj.weight"].T


def forward(cfg, t, ids):
    s = len(ids)
    half = cfg["qk_rope_head_dim"] // 2
    inv = cfg["rope_theta"] ** (
        -2.0 * torch.arange(half, dtype=torch.float64) / cfg["qk_rope_head_dim"]
    )
    ang = torch.arange(s, dtype=torch.float64).unsqueeze(1) * inv.unsqueeze(0)
    cos, sin = ang.cos().float(), ang.sin().float()

    x = t["model.word_embeddings.weight"][torch.tensor(ids)]
    for i in range(cfg["num_hidden_layers"]):
        lp = f"model.layers.{i}"
        normed = rms_norm(x, t[f"{lp}.input_layernorm.weight"], cfg["rms_norm_eps"])
        at = f"{lp}.attention"
        if attn_kind(cfg, i) == "mla":
            x = x + mla_layer(cfg, t, at, normed, cos, sin)
        else:
            x = x + kda_layer(cfg, t, at, normed)
        normed = rms_norm(x, t[f"{lp}.post_attention_layernorm.weight"], cfg["rms_norm_eps"])
        mlp = f"{lp}.mlp"
        x = x + (moe_layer(cfg, t, mlp, normed) if is_moe(cfg, i) else dense_mlp(t, mlp, normed))
    x = rms_norm(x, t["model.norm.weight"], cfg["rms_norm_eps"])
    return x @ t["lm_head.weight"].T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=pathlib.Path, help="fixture directory to write")
    ap.add_argument("--seq", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    cfg = dict(CONFIG)
    t = make_weights(cfg, gen)
    ids = [(i * 7) % cfg["vocab_size"] for i in range(args.seq)]

    with torch.no_grad():
        logits = forward(cfg, t, ids)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    save_file({k: v.contiguous() for k, v in t.items()}, str(args.out / "model.safetensors"))
    (args.out / "input_ids.json").write_text(json.dumps(ids))
    (args.out / "logits.f32").write_bytes(
        logits.to(torch.float32).contiguous().numpy().tobytes()
    )
    print(f"wrote {args.out} — logits {tuple(logits.shape)}, "
          f"range [{logits.min():.4f}, {logits.max():.4f}]")


if __name__ == "__main__":
    main()
