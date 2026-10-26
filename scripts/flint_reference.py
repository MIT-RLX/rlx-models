#!/usr/bin/env python3
"""Reference logits for rlx-flint, transcribed from Recursive's own SGLang model.

Flint ships no transformers modeling code: the only implementation Recursive
publishes is `flint_sglang/model.py` inside `sglang/flint_sglang-*.whl` in the
model repo, which needs CUDA and SGLang. This script is that file's forward
pass rewritten in plain torch, float32 on CPU, so rlx-flint has something
independent to be compared against on any machine.

What it transcribes, piece by piece (the SGLang class each comes from):

  * `FlintAttention` (Qwen3MoeAttention): q/k/v projections, per-head RMSNorm
    on q and k, NeoX RoPE over the whole head on layers NOT in
    `no_rope_layer_indices`, scale head_dim^-0.5, and a learned per-head sink
    logit (`self_attn.softmax_offset`) that joins the softmax denominator with
    no value. Sliding layers see the current token and `sliding_window - 1`
    before it (SGLang's `sliding_window_size = sliding_window - 1`).
  * `FlintMLP` (Qwen2MoeMLP): down(silu(gate(x)) * up(x)).
  * `FlintSparseMoeBlock`: sigmoid router scores, top-k on scores (+ the
    optional `mlp.gate.expert_bias`, selection only), weights renormalized
    with `sum + 1e-20`, plus an ungated shared expert.
  * `FlintDecoderLayer`: pre-norm residual blocks; `mlp_only_layers` dense.

RoPE angles are computed in float64 and rounded to float32 once (SGLang
builds its cos/sin cache in float32). The difference is far below bf16
resolution and keeps this script and the port on the same tables.

Weights are read lazily, one tensor at a time, and routed experts only when a
token fires them, so the real 80 GB checkpoint runs on a machine with less
RAM than that (slowly). `--layers N` truncates the stack to its first N
layers, which is how a range-fetched subset is checked.

Usage:

  python3 scripts/flint_reference.py --model-dir weights/lm/flint \\
      --ids 150003,9906,11,1879 --out /tmp/flint_ref.safetensors [--layers 3]
  python3 scripts/flint_reference.py --model-dir DIR --text "Hello" --chat ...

Output (safetensors): `ids` [seq] (int64), `logits` [seq, vocab] (f32) —
or only the last row with `--last-only` — and `hidden.N` [seq, hidden] after
every layer when `--dump-hidden` is given.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class Checkpoint:
    """Name → tensor, read on demand from a (sharded) safetensors directory."""

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        index = model_dir / "model.safetensors.index.json"
        if index.is_file():
            self.where = json.loads(index.read_text())["weight_map"]
        else:
            self.where = {}
            for f in sorted(model_dir.glob("*.safetensors")):
                with safe_open(str(f), "pt") as h:
                    for k in h.keys():
                        self.where[k] = f.name
        self.handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str) -> torch.Tensor:
        shard = self.where[name]
        h = self.handles.get(shard)
        if h is None:
            h = safe_open(str(self.dir / shard), "pt")
            self.handles[shard] = h
        return h.get_tensor(name).to(torch.float32)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_tables(seq: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv = [1.0 / theta ** (2.0 * i / head_dim) for i in range(half)]
    ang = [[p * f for f in inv] for p in range(seq)]
    cos = torch.tensor([[math.cos(a) for a in row] for row in ang], dtype=torch.float32)
    sin = torch.tensor([[math.sin(a) for a in row] for row in ang], dtype=torch.float32)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [seq, heads, head_dim]; NeoX pairing (i, i + half).
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([a * c - b * s, b * c + a * s], dim=-1)


def swiglu(ck: Checkpoint, prefix: str, x: torch.Tensor) -> torch.Tensor:
    g = x @ ck.get(f"{prefix}.gate_proj.weight").T
    u = x @ ck.get(f"{prefix}.up_proj.weight").T
    return (torch.nn.functional.silu(g) * u) @ ck.get(f"{prefix}.down_proj.weight").T


def attention(ck, cfg, layer: int, x: torch.Tensor, cos, sin) -> torch.Tensor:
    p = f"model.layers.{layer}.self_attn"
    seq = x.shape[0]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    q = (x @ ck.get(f"{p}.q_proj.weight").T).view(seq, nh, hd)
    k = (x @ ck.get(f"{p}.k_proj.weight").T).view(seq, nkv, hd)
    v = (x @ ck.get(f"{p}.v_proj.weight").T).view(seq, nkv, hd)
    q = rms_norm(q, ck.get(f"{p}.q_norm.weight"), eps)
    k = rms_norm(k, ck.get(f"{p}.k_norm.weight"), eps)
    if layer not in set(cfg.get("no_rope_layer_indices") or []):
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    group = nh // nkv
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q, k) * hd**-0.5  # [nh, seq, seq]
    qi = torch.arange(seq)[:, None]
    ki = torch.arange(seq)[None, :]
    visible = ki <= qi
    sliding = (cfg.get("layer_types") or [])[layer] == "sliding_attention"
    if sliding and cfg.get("use_sliding_window", True) and cfg.get("sliding_window"):
        visible &= (qi - ki) <= cfg["sliding_window"] - 1
    scores = scores.masked_fill(~visible[None], float("-inf"))

    sink = ck.get(f"{p}.softmax_offset").view(nh, 1, 1).expand(nh, seq, 1)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :seq]
    out = torch.einsum("hqk,khd->qhd", probs, v).reshape(seq, nh * hd)
    return out @ ck.get(f"{p}.o_proj.weight").T


def moe(ck, cfg, layer: int, x: torch.Tensor) -> torch.Tensor:
    p = f"model.layers.{layer}.mlp"
    top_k = cfg["num_experts_per_tok"]
    scores = torch.sigmoid(x @ ck.get(f"{p}.gate.weight").T)  # [seq, E]
    choice = scores
    if ck.has(f"{p}.gate.expert_bias"):
        choice = scores + ck.get(f"{p}.gate.expert_bias")[None, :]
    ids = torch.topk(choice, k=top_k, dim=-1).indices
    w = scores.gather(1, ids)
    if cfg.get("norm_topk_prob", True):
        w = w / (w.sum(-1, keepdim=True) + 1e-20)

    out = torch.zeros_like(x)
    for e in sorted(set(ids.flatten().tolist())):
        rows, slot = (ids == e).nonzero(as_tuple=True)
        y = swiglu(ck, f"{p}.experts.{e}", x[rows])
        out.index_add_(0, rows, y * w[rows, slot][:, None])
    if cfg.get("shared_expert_intermediate_size", 0) > 0:
        out = out + swiglu(ck, f"{p}.shared_expert", x)
    return out


def forward(ck, cfg, ids: list[int], n_layers: int, dump_hidden: bool):
    eps = cfg["rms_norm_eps"]
    seq = len(ids)
    emb = ck.get("model.embed_tokens.weight")
    h = emb[torch.tensor(ids)]
    del emb
    cos, sin = rope_tables(seq, cfg["head_dim"], float(cfg.get("rope_theta", 10000.0)))
    dense = set(cfg.get("mlp_only_layers") or [])
    hidden = {}
    for layer in range(n_layers):
        p = f"model.layers.{layer}"
        a = attention(ck, cfg, layer, rms_norm(h, ck.get(f"{p}.input_layernorm.weight"), eps), cos, sin)
        h = h + a
        x = rms_norm(h, ck.get(f"{p}.post_attention_layernorm.weight"), eps)
        h = h + (swiglu(ck, f"{p}.mlp", x) if layer in dense else moe(ck, cfg, layer, x))
        if dump_hidden:
            hidden[f"hidden.{layer}"] = h.clone()
        print(f"  layer {layer} done", flush=True)
    h = rms_norm(h, ck.get("model.norm.weight"), eps)
    logits = h @ ck.get("lm_head.weight").T
    return logits, hidden


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--ids", help="comma-separated token ids")
    ap.add_argument("--text", help="encode with the checkpoint's tokenizer.json")
    ap.add_argument("--chat", action="store_true", help="wrap --text as one user turn + generation prompt")
    ap.add_argument("--layers", type=int, help="run only the first N layers")
    ap.add_argument("--last-only", action="store_true", help="store only the last row of logits")
    ap.add_argument("--dump-hidden", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    cfg = json.loads((args.model_dir / "config.json").read_text())
    ck = Checkpoint(args.model_dir)
    if args.ids:
        ids = [int(t) for t in args.ids.split(",") if t.strip()]
    elif args.text is not None:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
        text = f"<|user|>{args.text}<|assistant|>" if args.chat else args.text
        ids = tok.encode(text, add_special_tokens=False).ids
    else:
        ap.error("one of --ids / --text is required")
    n_layers = args.layers or cfg["num_hidden_layers"]
    print(f"flint reference: {len(ids)} tokens, {n_layers} layers", flush=True)
    logits, hidden = forward(ck, cfg, ids, n_layers, args.dump_hidden)
    if args.last_only:
        logits = logits[-1:]
    out = {"ids": torch.tensor(ids, dtype=torch.int64), "logits": logits.contiguous()}
    out.update({k: v.contiguous() for k, v in hidden.items()})
    save_file(out, str(args.out))
    print(f"wrote {args.out}: logits {tuple(logits.shape)}, argmax(last) = {int(logits[-1].argmax())}")


if __name__ == "__main__":
    main()
