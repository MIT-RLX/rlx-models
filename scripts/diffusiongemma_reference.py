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
"""DiffusionGemma reference forward, for rlx-diffusiongemma parity tests.

Emits a self-contained fixture (config + synthetic safetensors + reference
tensors) that `crates/rlx-diffusiongemma/tests/parity_reference.rs` replays
through the RLX graphs:

    python3 scripts/diffusiongemma_reference.py .fixtures/diffusiongemma-parity
    RLX_DG_PARITY_DIR=.fixtures/diffusiongemma-parity \
        cargo test -p rlx-diffusiongemma --test parity_reference

This is a *transcription* of `transformers/models/diffusion_gemma/
modeling_diffusion_gemma.py` (plus the Gemma 4 router/experts and
`_compute_proportional_rope_parameters` it reuses), not an import of it —
DiffusionGemma needs transformers >= 5.8, and the point is to pin the arithmetic
independently. The details that are easy to get wrong, and where they come from:

  * **Proportional RoPE** — `modeling_rope_utils.py::
    _compute_proportional_rope_parameters`. `rope_angles = int(p·head_dim // 2)`
    slots are rotated with the exponent divided by the *full* `head_dim`; the
    remaining `head_dim/2 - rope_angles` slots get `inv_freq = 0`, so the table
    stays full width and those channels pass through. Also note the head_dim
    here is the *per-layer* one (512 on full-attention layers), resolved via
    `per_layer_config`.
  * **V aliases K** — full-attention layers have no `v_proj`; V is the K
    projection taken *before* `k_norm`, and never RoPE'd. `v_norm` is
    `with_scale=False`, so it carries no weight.
  * **`scaling = 1.0`** — not `head_dim**-0.5`; Q/K are per-head RMS-normed.
  * **Gemma4RMSNorm has no `1 +`** — unlike Gemma 2/3, the weight multiplies the
    normalized value directly.
  * **Two-branch FFN** — the router scores the *unnormalized* residual while the
    experts consume `pre_feedforward_layernorm_2(residual)`.
  * **Sliding mask** — `kv_idx > q_idx - sliding_window`, i.e. `sliding_window`
    positions including the query.
  * **Decoder** — bidirectional over `[cache ; canvas]`, no mask; the cache is
    windowed for sliding layers. Logits are soft-capped and *then* divided by
    the step temperature, and the self-conditioning signal is built from those
    scaled logits.
"""

import argparse
import json
import math
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

TINY = {
    "model_type": "diffusion_gemma",
    "canvas_length": 4,

    "vision_config": {
        "hidden_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "head_dim": 12,
        "intermediate_size": 20,
        "patch_size": 2,
        "pooling_kernel_size": 2,
        "position_embedding_size": 32,
        "rms_norm_eps": 1e-6,
        "standardize": True,
        "use_clipped_linears": False,
        "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
    },
    "text_config": {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 12,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 8,
        "global_head_dim": 16,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "sliding_window": 4,
        "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30.0,
        "num_experts": 4,
        "top_k_experts": 2,
        "moe_intermediate_size": 6,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    },
}


class VCfg:
    def __init__(self, raw):
        for k, v in raw.items():
            setattr(self, k, v)


class Cfg:
    def __init__(self, raw):
        t = raw["text_config"]
        self.canvas_length = raw["canvas_length"]
        self.image_token_id = raw.get("image_token_id", 258880)
        self.vision = VCfg(raw["vision_config"]) if raw.get("vision_config") else None
        for k, v in t.items():
            setattr(self, k, v)

    def is_full(self, i):
        return self.layer_types[i] == "full_attention"

    def head_dim_at(self, i):
        return self.global_head_dim if self.is_full(i) else self.head_dim

    def kv_heads_at(self, i):
        return self.num_global_key_value_heads if self.is_full(i) else self.num_key_value_heads

    def k_eq_v(self, i):
        return self.is_full(i)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def rms_norm(x, weight, eps):
    """Gemma4RMSNorm. No `1 +` on the weight, and eps is inside the sqrt."""
    out = x.float()
    out = out * torch.pow(out.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        out = out * weight.float()
    return out.type_as(x)


def inv_freq_for(cfg, layer):
    """Per-layer inverse frequencies, always head_dim/2 long."""
    hd = cfg.head_dim_at(layer)
    params = cfg.rope_parameters["full_attention" if cfg.is_full(layer) else "sliding_attention"]
    base = params["rope_theta"]
    if params.get("rope_type", "default") == "proportional":
        prop = params.get("partial_rotary_factor", 1.0)
        # int(rope_proportion * head_dim // 2): the floor divide binds after
        # the multiply.
        rope_angles = int(prop * hd // 2)
        rotated = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / hd)
        )
        nope = hd // 2 - rope_angles
        inv = torch.cat([rotated, torch.zeros(nope)]) if nope > 0 else rotated
        return inv / params.get("factor", 1.0)
    return 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))


def rope_tables(cfg, layer, offset, length):
    """(cos, sin) of shape [length, head_dim/2] — the rlx half-width layout."""
    inv = inv_freq_for(cfg, layer)
    pos = torch.arange(offset, offset + length, dtype=torch.float)
    ang = pos[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """NeoX rotate-half on [b, s, h, d] with half-width tables [s, d/2]."""
    full_cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]
    full_sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
    d = x.shape[-1] // 2
    rotated = torch.cat([-x[..., d:], x[..., :d]], dim=-1)
    return x * full_cos + rotated * full_sin


def repeat_kv(x, group):
    """[b, s, kvh, d] -> [b, s, kvh*group, d], each head repeated in place."""
    if group == 1:
        return x
    b, s, h, d = x.shape
    return x[:, :, :, None, :].expand(b, s, h, group, d).reshape(b, s, h * group, d)


def attention(q, k, v, mask):
    """Eager SDPA with scaling = 1.0. q/k/v are [b, s, heads, d]."""
    q = q.transpose(1, 2)  # [b, h, sq, d]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2))  # scaling = 1.0
    if mask is not None:
        scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float())  # [b, h, sq, d]
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def causal_sliding_mask(seq, window):
    """HF: causal AND kv_idx > q_idx - window."""
    qi = torch.arange(seq)[:, None]
    ki = torch.arange(seq)[None, :]
    allowed = (ki <= qi) & (ki > qi - window)
    return torch.where(allowed, 0.0, float("-inf"))


def causal_mask(seq):
    qi = torch.arange(seq)[:, None]
    ki = torch.arange(seq)[None, :]
    return torch.where(ki <= qi, 0.0, float("-inf"))


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


def qkv(w, p, cfg, layer, x, cos, sin):
    """Projections + per-head norms + RoPE. Returns q [b,s,h,d] and (k, v)."""
    hd = cfg.head_dim_at(layer)
    nh = cfg.num_attention_heads
    nkv = cfg.kv_heads_at(layer)
    b, s, _ = x.shape
    eps = cfg.rms_norm_eps

    q = (x @ w[f"{p}.self_attn.q_proj.weight"].T).view(b, s, nh, hd)
    q = rms_norm(q, w[f"{p}.self_attn.q_norm.weight"], eps)
    q = apply_rope(q, cos, sin)

    k_raw = (x @ w[f"{p}.self_attn.k_proj.weight"].T).view(b, s, nkv, hd)
    # V aliases the PRE-k_norm K projection on full-attention layers.
    v_raw = (
        k_raw
        if cfg.k_eq_v(layer)
        else (x @ w[f"{p}.self_attn.v_proj.weight"].T).view(b, s, nkv, hd)
    )
    k = rms_norm(k_raw, w[f"{p}.self_attn.k_norm.weight"], eps)
    k = apply_rope(k, cos, sin)
    v = rms_norm(v_raw, None, eps)  # v_norm: with_scale=False
    return q, k, v


def gated_mlp(w, p, x):
    """down(gelu_tanh(gate(x)) * up(x))."""
    gate = x @ w[f"{p}.gate_proj.weight"].T
    up = x @ w[f"{p}.up_proj.weight"].T
    act = torch.nn.functional.gelu(gate, approximate="tanh")
    return (act * up) @ w[f"{p}.down_proj.weight"].T


def router(w, p, cfg, flat):
    """Gemma4TextRouter -> (top_idx, top_weights)."""
    eps = cfg.rms_norm_eps
    h = rms_norm(flat, None, eps)  # norm: with_scale=False
    h = h * w[f"{p}.router.scale"] * (cfg.hidden_size**-0.5)
    scores = h @ w[f"{p}.router.proj.weight"].T
    probs = torch.softmax(scores.float(), dim=-1)
    top_w, top_i = torch.topk(probs, k=cfg.top_k_experts, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    top_w = top_w * w[f"{p}.router.per_expert_scale"][top_i]
    return top_i, top_w


def experts(w, p, cfg, flat, top_i, top_w):
    """Gemma4TextExperts over the stacked banks."""
    gate_up = w[f"{p}.experts.gate_up_proj"]  # [E, 2i, hidden]
    down = w[f"{p}.experts.down_proj"]  # [E, hidden, i]
    out = torch.zeros_like(flat)
    inter = cfg.moe_intermediate_size
    for tok in range(flat.shape[0]):
        for slot in range(cfg.top_k_experts):
            e = int(top_i[tok, slot])
            gu = flat[tok] @ gate_up[e].T  # [2i]
            g, u = gu[:inter], gu[inter:]
            hx = torch.nn.functional.gelu(g, approximate="tanh") * u
            out[tok] += (hx @ down[e].T) * top_w[tok, slot]
    return out


def layer_forward(w, cfg, layer, x, cos, sin, mask, scalar_key, enc_kv=None):
    """One DiffusionGemma layer. `enc_kv` switches on the decoder path."""
    p = f"model.decoder.layers.{layer}"
    eps = cfg.rms_norm_eps
    hd = cfg.head_dim_at(layer)
    nkv = cfg.kv_heads_at(layer)
    group = cfg.num_attention_heads // nkv

    residual = x
    h = rms_norm(x, w[f"{p}.input_layernorm.weight"], eps)
    q, k, v = qkv(w, p, cfg, layer, h, cos, sin)
    tap = (k.clone(), v.clone())
    if enc_kv is not None:
        k = torch.cat([enc_kv[0], k], dim=1)
        v = torch.cat([enc_kv[1], v], dim=1)
    attn = attention(q, repeat_kv(k, group), repeat_kv(v, group), mask)
    attn = attn @ w[f"{p}.self_attn.o_proj.weight"].T
    attn = rms_norm(attn, w[f"{p}.post_attention_layernorm.weight"], eps)
    residual = residual + attn

    # Branch 1: shared expert.
    b1 = gated_mlp(w, f"{p}.mlp", rms_norm(residual, w[f"{p}.pre_feedforward_layernorm.weight"], eps))
    b1 = rms_norm(b1, w[f"{p}.post_feedforward_layernorm_1.weight"], eps)

    # Branch 2: routed experts. The router sees the RAW residual.
    flat = residual.reshape(-1, cfg.hidden_size)
    top_i, top_w = router(w, p, cfg, flat)
    expert_in = rms_norm(flat, w[f"{p}.pre_feedforward_layernorm_2.weight"], eps)
    b2 = experts(w, p, cfg, expert_in, top_i, top_w).reshape(residual.shape)
    b2 = rms_norm(b2, w[f"{p}.post_feedforward_layernorm_2.weight"], eps)

    out = rms_norm(b1 + b2, w[f"{p}.post_feedforward_layernorm.weight"], eps)
    out = residual + out
    return out * w[scalar_key], tap


def encoder(w, cfg, ids):
    """Causal prefill; returns (hidden, [(k, v) per layer])."""
    x = w["model.decoder.embed_tokens.weight"][ids][None] * math.sqrt(cfg.hidden_size)
    seq = ids.shape[0]
    taps = []
    for l in range(cfg.num_hidden_layers):
        cos, sin = rope_tables(cfg, l, 0, seq)
        mask = causal_mask(seq) if cfg.is_full(l) else causal_sliding_mask(seq, cfg.sliding_window)
        key = f"model.encoder.language_model.layers.{l}.layer_scalar"
        x, tap = layer_forward(w, cfg, l, x, cos, sin, mask, key)
        taps.append(tap)
    return rms_norm(x, w["model.decoder.norm.weight"], cfg.rms_norm_eps), taps


def decoder(w, cfg, canvas_ids, sc_signal, taps, prompt_len, temperature):
    """Bidirectional denoiser over the canvas; returns (logits, soft_embeds)."""
    eps = cfg.rms_norm_eps
    hidden = cfg.hidden_size
    embed = w["model.decoder.embed_tokens.weight"]
    canvas = canvas_ids.shape[0]

    x = embed[canvas_ids][None] * math.sqrt(hidden)
    # Self-conditioning.
    sc = rms_norm(sc_signal, w["model.decoder.self_conditioning.pre_norm.weight"], eps)
    sc = gated_mlp(w, "model.decoder.self_conditioning", sc)
    x = rms_norm(x + sc, None, eps)  # post_norm: with_scale=False

    for l in range(cfg.num_hidden_layers):
        cos, sin = rope_tables(cfg, l, prompt_len, canvas)
        keep = min(prompt_len, cfg.sliding_window) if not cfg.is_full(l) else prompt_len
        enc_kv = (taps[l][0][:, prompt_len - keep :], taps[l][1][:, prompt_len - keep :])
        key = f"model.decoder.layers.{l}.layer_scalar"
        # No mask: the denoiser attends bidirectionally over cache and canvas.
        x, _ = layer_forward(w, cfg, l, x, cos, sin, None, key, enc_kv=enc_kv)

    x = rms_norm(x, w["model.decoder.norm.weight"], eps)
    cap = cfg.final_logit_softcapping
    logits = (x @ embed.T).float()
    logits = torch.tanh(logits / cap) * cap
    logits = logits / temperature
    soft = torch.softmax(logits, dim=-1) @ embed.float() * math.sqrt(hidden)
    return logits, soft



# --------------------------------------------------------------------------
# Vision tower (gemma4_vision) + projector
# --------------------------------------------------------------------------


def vision_inv_freq(v):
    """Per-axis frequencies: spatial = head_dim/2, stepped by 2."""
    spatial = v.head_dim // 2
    base = v.rope_parameters["rope_theta"]
    return 1.0 / (base ** (torch.arange(0, spatial, 2, dtype=torch.float) / spatial))


def vision_rope_tables(v, positions):
    """(cos, sin) of shape [P, head_dim]: x-half then y-half, each cat(f, f)."""
    inv = vision_inv_freq(v)
    xs = torch.tensor([p[0] for p in positions], dtype=torch.float)
    ys = torch.tensor([p[1] for p in positions], dtype=torch.float)
    ax = xs[:, None] * inv[None, :]
    ay = ys[:, None] * inv[None, :]
    cos = torch.cat([ax.cos(), ax.cos(), ay.cos(), ay.cos()], dim=-1)
    sin = torch.cat([ax.sin(), ax.sin(), ay.sin(), ay.sin()], dim=-1)
    return cos, sin


def apply_2d_rope(x, cos, sin):
    """x: [1, P, heads, head_dim]; cos/sin: [P, head_dim]. Rotate each axis half
    independently (HF apply_multidimensional_rope with ndim=2)."""
    d = x.shape[-1]
    q = d // 4
    h = d // 2
    x_lo, x_hi = x[..., :q], x[..., q:h]
    y_lo, y_hi = x[..., h : h + q], x[..., h + q :]
    rot = torch.cat([-x_hi, x_lo, -y_hi, y_lo], dim=-1)
    return x * cos[None, :, None, :] + rot * sin[None, :, None, :]


def vision_layer(w, p, v, x, cos, sin):
    eps = v.rms_norm_eps
    nh, hd = v.num_attention_heads, v.head_dim
    b, P, _ = x.shape

    residual = x
    h = rms_norm(x, w[f"{p}.input_layernorm.weight"], eps)
    q = (h @ w[f"{p}.self_attn.q_proj.linear.weight"].T).view(b, P, nh, hd)
    k = (h @ w[f"{p}.self_attn.k_proj.linear.weight"].T).view(b, P, nh, hd)
    val = (h @ w[f"{p}.self_attn.v_proj.linear.weight"].T).view(b, P, nh, hd)
    q = apply_2d_rope(rms_norm(q, w[f"{p}.self_attn.q_norm.weight"], eps), cos, sin)
    k = apply_2d_rope(rms_norm(k, w[f"{p}.self_attn.k_norm.weight"], eps), cos, sin)
    val = rms_norm(val, None, eps)
    attn = attention(q, k, val, None)  # scaling = 1.0, bidirectional
    attn = attn @ w[f"{p}.self_attn.o_proj.linear.weight"].T
    attn = rms_norm(attn, w[f"{p}.post_attention_layernorm.weight"], eps)
    residual = residual + attn

    h = rms_norm(residual, w[f"{p}.pre_feedforward_layernorm.weight"], eps)
    gate = h @ w[f"{p}.mlp.gate_proj.linear.weight"].T
    up = h @ w[f"{p}.mlp.up_proj.linear.weight"].T
    ffn = (torch.nn.functional.gelu(gate, approximate="tanh") * up)
    ffn = ffn @ w[f"{p}.mlp.down_proj.linear.weight"].T
    ffn = rms_norm(ffn, w[f"{p}.post_feedforward_layernorm.weight"], eps)
    return residual + ffn


def vision_tower(w, cfg, pixels, positions, pool, out_len):
    """pixels: [1, P, 3*patch^2] in [0,1]. Returns soft tokens [1, L, text_hidden]."""
    v = cfg.vision
    vp = "model.encoder.vision_tower"
    eps = v.rms_norm_eps

    scaled = 2 * (pixels - 0.5)
    x = scaled @ w[f"{vp}.patch_embedder.input_proj.weight"].T
    table = w[f"{vp}.patch_embedder.position_embedding_table"]
    xi = torch.tensor([p[0] for p in positions])
    yi = torch.tensor([p[1] for p in positions])
    x = x + (table[0][xi] + table[1][yi])[None]

    taps = {"patch_embed": x.clone(), "rope_cos": None}
    cos, sin = vision_rope_tables(v, positions)
    taps["rope_cos"], taps["rope_sin"] = cos, sin
    for i in range(v.num_hidden_layers):
        x = vision_layer(w, f"{vp}.encoder.layers.{i}", v, x, cos, sin)
    taps["encoder_out"] = x.clone()

    # Pool (k^2 average by position), then sqrt(hidden) scale.
    pooled = pool @ x[0].float()
    pooled = pooled * (v.hidden_size ** 0.5)
    if v.standardize:
        pooled = (pooled - w[f"{vp}.std_bias"].float()) * w[f"{vp}.std_scale"].float()
    taps["pooled"] = pooled.clone()

    # Projector: scale-free RMS norm, then linear into LM width.
    pj = "model.encoder.embed_vision"
    normed = rms_norm(pooled, None, eps)
    soft = normed @ w[f"{pj}.embedding_projection.weight"].T
    _ = out_len
    return soft[None], taps


def make_vision_weights(cfg, w, gen):
    def r(*shape):
        return (torch.rand(*shape, generator=gen) - 0.5) * 0.2

    v = cfg.vision
    vp = "model.encoder.vision_tower"
    h = v.hidden_size
    w[f"{vp}.patch_embedder.input_proj.weight"] = r(h, 3 * v.patch_size**2)
    w[f"{vp}.patch_embedder.position_embedding_table"] = r(2, v.position_embedding_size, h)
    w[f"{vp}.std_bias"] = r(h)
    w[f"{vp}.std_scale"] = r(h) + 1.0
    for i in range(v.num_hidden_layers):
        p = f"{vp}.encoder.layers.{i}"
        for n in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            w[f"{p}.{n}.weight"] = r(h)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w[f"{p}.self_attn.{n}.linear.weight"] = r(h, h)
        w[f"{p}.self_attn.q_norm.weight"] = r(v.head_dim)
        w[f"{p}.self_attn.k_norm.weight"] = r(v.head_dim)
        w[f"{p}.mlp.gate_proj.linear.weight"] = r(v.intermediate_size, h)
        w[f"{p}.mlp.up_proj.linear.weight"] = r(v.intermediate_size, h)
        w[f"{p}.mlp.down_proj.linear.weight"] = r(h, v.intermediate_size)
    w["model.encoder.embed_vision.embedding_projection.weight"] = r(cfg.hidden_size, h)
    return w


def pool_matrix(positions, k, out_len):
    """HF _avg_pool_by_positions as an explicit [out_len, P] matrix."""
    P = len(positions)
    m = torch.zeros(out_len, P)
    max_x = max(p[0] for p in positions) + 1
    cols = max_x // k
    for pi, (x, y) in enumerate(positions):
        m[(x // k) + cols * (y // k), pi] = 1.0 / (k * k)
    return m


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------


def make_weights(cfg, gen):
    def r(*shape):
        return (torch.rand(*shape, generator=gen) - 0.5) * 0.2

    w = {}
    h = cfg.hidden_size
    w["model.decoder.embed_tokens.weight"] = r(cfg.vocab_size, h)
    w["model.decoder.norm.weight"] = r(h)
    sc = "model.decoder.self_conditioning"
    w[f"{sc}.pre_norm.weight"] = r(h)
    w[f"{sc}.gate_proj.weight"] = r(cfg.intermediate_size, h)
    w[f"{sc}.up_proj.weight"] = r(cfg.intermediate_size, h)
    w[f"{sc}.down_proj.weight"] = r(h, cfg.intermediate_size)

    for l in range(cfg.num_hidden_layers):
        p = f"model.decoder.layers.{l}"
        hd = cfg.head_dim_at(l)
        q_dim = cfg.num_attention_heads * hd
        kv_dim = cfg.kv_heads_at(l) * hd
        for n in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ):
            w[f"{p}.{n}.weight"] = r(h)
        w[f"{p}.layer_scalar"] = torch.tensor([1.05])
        w[f"model.encoder.language_model.layers.{l}.layer_scalar"] = torch.tensor([0.93])
        w[f"{p}.self_attn.q_proj.weight"] = r(q_dim, h)
        w[f"{p}.self_attn.k_proj.weight"] = r(kv_dim, h)
        if not cfg.k_eq_v(l):
            w[f"{p}.self_attn.v_proj.weight"] = r(kv_dim, h)
        w[f"{p}.self_attn.q_norm.weight"] = r(hd)
        w[f"{p}.self_attn.k_norm.weight"] = r(hd)
        w[f"{p}.self_attn.o_proj.weight"] = r(h, q_dim)
        w[f"{p}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, h)
        w[f"{p}.mlp.up_proj.weight"] = r(cfg.intermediate_size, h)
        w[f"{p}.mlp.down_proj.weight"] = r(h, cfg.intermediate_size)
        w[f"{p}.router.proj.weight"] = r(cfg.num_experts, h)
        w[f"{p}.router.scale"] = r(h) + 1.0
        w[f"{p}.router.per_expert_scale"] = r(cfg.num_experts) + 1.0
        w[f"{p}.experts.gate_up_proj"] = r(cfg.num_experts, 2 * cfg.moe_intermediate_size, h)
        w[f"{p}.experts.down_proj"] = r(cfg.num_experts, h, cfg.moe_intermediate_size)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--prompt-len", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    out_dir = out
    cfg = Cfg(TINY)
    gen = torch.Generator().manual_seed(args.seed)
    w = make_weights(cfg, gen)
    if cfg.vision is not None:
        w = make_vision_weights(cfg, w, gen)

    prompt_len = args.prompt_len
    canvas = cfg.canvas_length
    temperature = 0.8
    ids = torch.tensor([(i * 7 + 3) % cfg.vocab_size for i in range(prompt_len)])
    canvas_ids = torch.tensor([(i * 7 + 3) % cfg.vocab_size for i in range(canvas)])
    sc_signal = torch.full((1, canvas, cfg.hidden_size), 0.0)

    # Vision: a 4x4 patch grid pooled 2x2 into 4 soft tokens.
    v = cfg.vision
    grid = 4
    positions = [(x, y) for y in range(grid) for x in range(grid)]
    n_patches = len(positions)
    n_soft = n_patches // (v.pooling_kernel_size**2)
    torch.manual_seed(args.seed + 1)
    pixels = torch.rand(1, n_patches, 3 * v.patch_size**2, generator=gen)
    pool = pool_matrix(positions, v.pooling_kernel_size, n_soft)

    with torch.no_grad():
        soft_tokens, vtaps = vision_tower(w, cfg, pixels, positions, pool, n_soft)
        hidden, taps = encoder(w, cfg, ids)
        logits, soft = decoder(w, cfg, canvas_ids, sc_signal, taps, prompt_len, temperature)
        # A second pass with a non-zero self-conditioning signal, which is what
        # every denoising step after the first actually sees.
        logits2, soft2 = decoder(w, cfg, canvas_ids, soft, taps, prompt_len, temperature)

    save_file({k: v.contiguous() for k, v in w.items()}, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(TINY, indent=2))

    def dump(name, t):
        (out / f"{name}.bin").write_bytes(
            t.detach().float().contiguous().numpy().tobytes()
        )

    dump("encoder_hidden", hidden)
    for l, (k, v) in enumerate(taps):
        dump(f"enc_k_{l}", k)
        dump(f"enc_v_{l}", v)
    dump("logits", logits)
    dump("soft_embeds", soft)
    dump("logits_sc", logits2)
    dump("soft_embeds_sc", soft2)
    dump("vision_pixels", pixels)
    dump("vision_pool", pool)
    dump("soft_tokens", soft_tokens)
    dump("vision_patch_embed", vtaps["patch_embed"])
    dump("vision_encoder_out", vtaps["encoder_out"])
    dump("vision_pooled", vtaps["pooled"])
    dump("vision_rope_cos", vtaps["rope_cos"])
    dump("vision_rope_sin", vtaps["rope_sin"])

    # PIL-resize cases for the Rust preprocessor: one downscale, one upscale,
    # one non-square. These pin `resize_bicubic_u8` against Pillow itself,
    # which is what HF's image processor calls under the hood.
    from PIL import Image

    resize_cases = []
    rng = np.random.RandomState(args.seed)
    for i, (sh, sw, dh, dw) in enumerate([(137, 91, 96, 48), (40, 30, 96, 144), (64, 64, 48, 96)]):
        arr = rng.randint(0, 256, size=(sh, sw, 3), dtype=np.uint8)
        resized = np.asarray(Image.fromarray(arr, "RGB").resize((dw, dh), Image.BICUBIC))
        (out_dir / f"resize_src_{i}.bin").write_bytes(arr.tobytes())
        (out_dir / f"resize_dst_{i}.bin").write_bytes(resized.tobytes())
        resize_cases.append({"src_h": sh, "src_w": sw, "dst_h": dh, "dst_w": dw})

    # Chat prompts rendered by the model's own chat_template.jinja, with the
    # processor's image-token expansion applied on top — exactly the string
    # `rlx_diffusiongemma::prompt::format_chat` must produce.
    chat_cases = []
    try:
        from jinja2 import Environment

        tmpl_path = pathlib.Path(__file__).parent.parent / "crates/rlx-diffusiongemma/fixtures/chat_template.jinja"
        if tmpl_path.is_file():
            tmpl = Environment(extensions=["jinja2.ext.do"]).from_string(tmpl_path.read_text())
            BOI, IMG, EOI = "<|image>", "<|image|>", "<image|>"
            specs = [
                ("plain", [{"role": "user", "content": "  Why is the sky blue?  "}], False, []),
                ("system", [{"role": "system", "content": "You are terse."},
                            {"role": "user", "content": "Hi"}], False, []),
                ("thinking", [{"role": "user", "content": "Hi"}], True, []),
                ("multi", [{"role": "user", "content": "2+2?"},
                           {"role": "assistant", "content": "4"},
                           {"role": "user", "content": "and 3+3?"}], False, []),
                ("image", [{"role": "user", "content": [{"type": "image"},
                                                        {"type": "text", "text": "What is this?"}]}], False, [3]),
                ("two_images", [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                                             {"type": "text", "text": "Compare these."}]}], False, [2, 4]),
            ]
            for name, msgs, think, soft in specs:
                rendered = tmpl.render(messages=msgs, bos_token="<bos>",
                                       add_generation_prompt=True, enable_thinking=think)
                # Gemma4Processor.replace_image_token, per image in order.
                # Split-then-rejoin, NOT repeated `replace`: the slots inserted
                # for image i are themselves `<|image|>` and would be re-matched
                # when expanding image i+1.
                parts = rendered.split(IMG)
                assert len(parts) - 1 == len(soft), f"{name}: placeholder/count mismatch"
                rendered = parts[0]
                for n, tail in zip(soft, parts[1:]):
                    rendered += f"{BOI}{IMG * n}{EOI}" + tail
                chat_cases.append({"name": name, "soft_tokens": soft, "expected": rendered})
    except ImportError:
        print("jinja2 not installed - skipping chat template cases")

    meta = {
        "chat_cases": chat_cases,
        "resize_cases": resize_cases,
        "vision": {
            "patches": n_patches,
            "soft_tokens": n_soft,
            "grid": grid,
            "positions_x": [p[0] for p in positions],
            "positions_y": [p[1] for p in positions],
        },
        "prompt_len": prompt_len,
        "canvas": canvas,
        "temperature": temperature,
        "prompt_ids": ids.tolist(),
        "canvas_ids": canvas_ids.tolist(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote fixture to {out}")
    print(f"  encoder hidden {tuple(hidden.shape)}  logits {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
