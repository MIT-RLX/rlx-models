#!/usr/bin/env python3
"""Reference outputs for rlx-pplx-decider, from Perplexity's own `pplx_decider` code.

Runs the released `Decider` (protocol v3, exact padded path, float32 on CPU) over a
request suite and writes, per request: the rendered prompt, its token ids, the
readout marker positions, the raw readout logits of every question's valid options
and the formatted answers. rlx-pplx-decider's tests compare against this file.

Two weight sources:

  * default: the base checkpoint's bf16 safetensors, upcast to f32;
  * --gguf FILE: the same code, with every text-backbone tensor and the readout
    replaced by the DEQUANTIZED tensors of a GGUF (e.g. the Q8_0 release). This is
    the reference for running that GGUF: quantization error is shared, so what is
    left is the implementation. The GGUF→HF mapping inverts llama.cpp's qwen35
    converter (V-head reorder, A = -exp(A_log), +1 on norms) and asserts every
    tensor lands within quantization error of the bf16 original, so a wrong
    mapping fails loudly instead of producing a plausible reference.

Usage (venv with transformers>=5.17, torch, torchvision, pillow, gguf):

  python scripts/pplx_decider_reference.py --checkpoint weights/lm/pplx-decider-flash \\
      --requests crates/rlx-pplx-decider/tests/fixtures/requests.json --out ref_f32.json
  python scripts/pplx_decider_reference.py ... --gguf weights/lm/pplx-decider-flash-gguf/decider-flash-v1.2-q8_0.gguf \\
      --out ref_q8_0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


class LazyGguf:
    """Dequantize one tensor at a time (the whole model in f32 is ~18 GB)."""

    def __init__(self, path: Path) -> None:
        from gguf import GGUFReader

        self.tensors = {t.name: t for t in GGUFReader(str(path)).tensors}
        self.read: set[str] = set()

    def __call__(self, name: str) -> torch.Tensor:
        from gguf.quants import dequantize

        t = self.tensors[name]
        self.read.add(name)
        shape = [int(d) for d in reversed(t.shape)]  # gguf lists innermost first
        data = dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)
        return torch.from_numpy(data.reshape(shape))


def ungroup_v(x: torch.Tensor, dim: int, n_k: int, per_k: int, head: int) -> torch.Tensor:
    """Inverse of llama.cpp's `_reorder_v_heads` (tiled -> grouped by K head)."""
    shape = list(x.shape)
    new = shape[:dim] + [per_k, n_k, head] + shape[dim + 1 :]
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return x.reshape(new).permute(perm).reshape(shape)


def load_gguf_into(decider, gguf_path: Path) -> dict[str, float]:
    """Overwrite the text backbone and readout with the GGUF's dequantized weights."""
    cfg = decider.backbone.config.text_config
    n_k, n_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    hk, hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    per_k = n_v // n_k
    take = LazyGguf(gguf_path)

    lm = decider.backbone.language_model
    mapped: dict[str, object] = {
        "embed_tokens.weight": lambda: take("token_embd.weight"),
        "norm.weight": lambda: take("output_norm.weight") - 1.0,
    }
    for i, kind in enumerate(cfg.layer_types):
        b, p = f"blk.{i}.", f"layers.{i}."

        def norm(gg: str) -> object:
            return lambda: take(gg) - 1.0

        def plain(gg: str) -> object:
            return lambda: take(gg)

        def grouped(gg: str, dim: int, head: int) -> object:
            return lambda: ungroup_v(take(gg), dim, n_k, per_k, head)

        def grouped_vec(gg: str) -> torch.Tensor:
            return ungroup_v(take(gg).unsqueeze(-1), 0, n_k, per_k, 1).squeeze(-1)

        def qk_then_v(gg: str) -> object:
            def f() -> torch.Tensor:
                w = take(gg)
                qk = 2 * n_k * hk
                return torch.cat([w[:qk], ungroup_v(w[qk:], 0, n_k, per_k, hv)])

            return f

        mapped[p + "input_layernorm.weight"] = norm(b + "attn_norm.weight")
        mapped[p + "post_attention_layernorm.weight"] = norm(b + "post_attention_norm.weight")
        for hf, gg in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")):
            mapped[p + f"mlp.{hf}.weight"] = plain(b + gg + ".weight")
        if kind == "full_attention":
            a = p + "self_attn."
            for hf, gg in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
                mapped[a + hf + ".weight"] = plain(b + gg + ".weight")
            mapped[a + "q_norm.weight"] = norm(b + "attn_q_norm.weight")
            mapped[a + "k_norm.weight"] = norm(b + "attn_k_norm.weight")
        else:
            a = p + "linear_attn."
            mapped[a + "in_proj_qkv.weight"] = qk_then_v(b + "attn_qkv.weight")
            mapped[a + "in_proj_z.weight"] = grouped(b + "attn_gate.weight", 0, hv)
            mapped[a + "in_proj_a.weight"] = grouped(b + "ssm_alpha.weight", 0, 1)
            mapped[a + "in_proj_b.weight"] = grouped(b + "ssm_beta.weight", 0, 1)
            mapped[a + "A_log"] = (lambda gg: lambda: torch.log(-grouped_vec(gg)))(b + "ssm_a")
            mapped[a + "dt_bias"] = (lambda gg: lambda: grouped_vec(gg))(b + "ssm_dt.bias")
            mapped[a + "conv1d.weight"] = (lambda f: lambda: f().unsqueeze(1))(qk_then_v(b + "ssm_conv1d.weight"))
            mapped[a + "norm.weight"] = plain(b + "ssm_norm.weight")
            mapped[a + "out_proj.weight"] = grouped(b + "ssm_out.weight", 1, hv)

    errors: dict[str, float] = {}
    params = dict(lm.named_parameters())
    if set(params) != set(mapped):
        raise SystemExit(f"mapping mismatch: missing {set(params) - set(mapped)}, extra {set(mapped) - set(params)}")
    for name, make in mapped.items():
        new = make()
        param = params[name]
        old = param.detach().float()
        if tuple(new.shape) != tuple(old.shape):
            raise SystemExit(f"{name}: gguf {tuple(new.shape)} vs hf {tuple(old.shape)}")
        scale = old.abs().max().item() or 1.0
        rel = (new - old).abs().max().item() / scale
        errors[name] = rel
        # Q8_0 is ~1/127 of the block max; A_log goes through log(exp()) of an f32.
        if rel > 2e-2:
            raise SystemExit(f"{name}: dequantized GGUF is {rel:.3g} (relative) from the bf16 tensor; mapping is wrong")
        with torch.no_grad():
            param.copy_(new.to(param.dtype))
        del new, old

    readout = take("output.weight")  # [vocab, hidden], zero outside the code tokens
    ids = decider_token_ids(decider)
    rows = readout[ids]
    old = decider.readout.weight.detach().float()
    rel = (rows - old).abs().max().item() / old.abs().max().item()
    if rel > 2e-2:
        raise SystemExit(f"readout: {rel:.3g} from readout.safetensors; token_ids mapping is wrong")
    errors["readout"] = rel
    rest = torch.ones(readout.shape[0], dtype=torch.bool)
    rest[ids] = False
    if readout[rest].abs().max().item() != 0.0:
        raise SystemExit("output.weight is not zero outside the code tokens")
    with torch.no_grad():
        decider.readout.weight.copy_(rows.to(decider.readout.weight.dtype))
    leftover = sorted(k for k in take.tensors if k not in take.read and not k.startswith("mtp."))
    if leftover:
        raise SystemExit(f"unread GGUF tensors: {leftover[:8]}")
    return errors


def load_mmproj_into(decider, mmproj_path: Path) -> float:
    """Overwrite the vision tower with an mmproj GGUF (llama.cpp's qwen3vl layout)."""
    take = LazyGguf(mmproj_path)
    vis = decider.backbone.visual
    cfg = decider.backbone.config.vision_config
    mapped: dict[str, object] = {
        "patch_embed.proj.weight": lambda: torch.stack(
            [take("v.patch_embd.weight"), take("v.patch_embd.weight.1")], dim=2
        ),
        "patch_embed.proj.bias": lambda: take("v.patch_embd.bias"),
        "pos_embed.weight": lambda: take("v.position_embd.weight"),
        "merger.norm.weight": lambda: take("v.post_ln.weight"),
        "merger.norm.bias": lambda: take("v.post_ln.bias"),
        "merger.linear_fc1.weight": lambda: take("mm.0.weight"),
        "merger.linear_fc1.bias": lambda: take("mm.0.bias"),
        "merger.linear_fc2.weight": lambda: take("mm.2.weight"),
        "merger.linear_fc2.bias": lambda: take("mm.2.bias"),
    }
    names = {"norm1": "ln1", "norm2": "ln2", "attn.qkv": "attn_qkv", "attn.proj": "attn_out",
             "mlp.linear_fc1": "ffn_up", "mlp.linear_fc2": "ffn_down"}
    for i in range(cfg.depth):
        for hf, gg in names.items():
            for kind in ("weight", "bias"):
                mapped[f"blocks.{i}.{hf}.{kind}"] = (lambda k: lambda: take(k))(f"v.blk.{i}.{gg}.{kind}")
    params = dict(vis.named_parameters())
    if set(params) != set(mapped):
        raise SystemExit(f"mmproj mapping mismatch: missing {set(params) - set(mapped)}, extra {set(mapped) - set(params)}")
    worst = 0.0
    for name, make in mapped.items():
        new, param = make(), params[name]
        old = param.detach().float()
        if tuple(new.shape) != tuple(old.shape):
            raise SystemExit(f"{name}: mmproj {tuple(new.shape)} vs hf {tuple(old.shape)}")
        rel = (new - old).abs().max().item() / (old.abs().max().item() or 1.0)
        worst = max(worst, rel)
        # F16 holds every bf16 value of a sane weight exactly.
        if rel > 1e-3:
            raise SystemExit(f"{name}: mmproj is {rel:.3g} (relative) from the bf16 tensor; mapping is wrong")
        with torch.no_grad():
            param.copy_(new.to(param.dtype))
    leftover = sorted(k for k in take.tensors if k not in take.read)
    if leftover:
        raise SystemExit(f"unread mmproj tensors: {leftover[:8]}")
    return worst


def resized_image_sha256(pixel_values: torch.Tensor, grid: list[int]) -> str:
    """sha256 of the resized uint8 HWC image, rebuilt from the processor's patches.

    `pixel_values` rows are 2x2 merge windows of 16x16 patches, each row
    `(channel, temporal copy, y, x)`; normalization is `(x - 127.5) / 127.5`.
    """
    import hashlib

    _, gh, gw = grid
    p = pixel_values.reshape(gh // 2, gw // 2, 2, 2, 3, 2, 16, 16)[:, :, :, :, :, 0]
    img = p.permute(4, 0, 2, 5, 1, 3, 6).reshape(3, gh * 16, gw * 16)  # C, H, W
    img = img.permute(1, 2, 0)
    u8 = torch.round(img * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    return hashlib.sha256(u8.contiguous().numpy().tobytes()).hexdigest()


def decider_token_ids(decider) -> list[int]:
    cfg = json.loads((decider.checkpoint / "decision_config.json").read_text())
    return list(cfg["token_ids"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gguf", type=Path)
    ap.add_argument("--mmproj", type=Path, help="with --gguf: also replace the vision tower")
    ap.add_argument("--dump-dir", type=Path, help="write full vision embeddings here (.npy per request)")
    ap.add_argument("--hidden", type=int, default=8, help="leading hidden values to dump per marker")
    args = ap.parse_args()

    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from pplx_decider import Decider

    torch.manual_seed(0)
    decider = Decider.from_pretrained(args.checkpoint, device="cpu")
    weights = {"source": "bf16 safetensors upcast to f32"}
    if args.gguf:
        errs = load_gguf_into(decider, args.gguf)
        weights = {
            "source": f"dequantized {args.gguf.name}",
            "max_relative_vs_bf16": max(errs.values()),
        }
    if args.mmproj:
        weights["mmproj"] = args.mmproj.name
        weights["mmproj_max_relative_vs_bf16"] = load_mmproj_into(decider, args.mmproj)

    requests = json.loads(args.requests.read_text())
    base = args.requests.parent
    for request in requests:  # image paths are relative to the suite file
        if "images" in request:
            request["images"] = [str(base / p) for p in request["images"]]
    captured: list[torch.Tensor] = []
    vision: list[torch.Tensor] = []
    hook = decider.backbone.language_model.register_forward_hook(
        lambda _m, _i, out: captured.append(out.last_hidden_state.detach())
    )
    vhook = decider.backbone.visual.register_forward_hook(
        lambda _m, _i, out: vision.append(out.pooler_output.detach())
    )
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for n, request in enumerate(requests):
        captured.clear()
        vision.clear()
        chunk = decider.encoder.plan(request, decider.max_length)
        assert len(chunk) == 1, "the suite must fit one sequence"
        enc = chunk[0]
        logits = decider.logits([request])[0]
        hidden = captured[-1][0]  # one unpadded row
        emb = vision[-1] if vision else None
        response = decider.decide(request)
        result = {
            "text": decider.encoder.text(request),
            "input_ids": enc.input_ids.tolist(),
            "markers": enc.markers.tolist(),
            "logits": logits,
            "marker_hidden": [hidden[m, : args.hidden].tolist() for m in enc.markers.tolist()],
            "response": response,
        }
        if enc.image_grid_thw is not None:
            grids = enc.image_grid_thw.tolist()
            sizes = [t * h * w for t, h, w in grids]
            pv = torch.split(enc.pixel_values, sizes)
            pos, _ = decider.backbone.get_rope_index(
                enc.input_ids[None], enc.mm_token_type_ids[None], enc.image_grid_thw
            )
            result["image_grid_thw"] = grids
            result["image_sha256"] = [resized_image_sha256(p, g) for p, g in zip(pv, grids)]
            result["position_ids"] = pos[:, 0].tolist()
            result["vision_rows"] = int(emb.shape[0])
            result["vision_head"] = [emb[r, : args.hidden].tolist() for r in (0, emb.shape[0] // 2, emb.shape[0] - 1)]
            if args.dump_dir:
                np.save(args.dump_dir / f"vision_{n}.npy", emb.float().numpy())
        results.append(result)
    hook.remove()
    vhook.remove()
    out = {
        "checkpoint": args.checkpoint.name,
        "weights": weights,
        "temperature": decider.temperature,
        "results": results,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
