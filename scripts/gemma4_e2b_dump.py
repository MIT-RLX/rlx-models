#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# GPLv3 — see repository LICENSE.
"""Inspect and dump references for google/gemma-4-E2B-it-qat-mobile-transformers.

Two modes:

  --inspect   Print every safetensors tensor (name, dtype, shape) plus the
              `quantization_config` from config.json. Used to confirm the exact
              Per-Layer-Embedding / quant tensor names + packing before we write
              the Rust loader. No model construction — just header reads.

  --reference Build the HF model, run a fixed prompt, and dump the text-LM final
              logits (and optionally per-layer hidden states) to <out>/ as raw
              little-endian f32 `.bin` files + a JSON sidecar of shapes. These
              are the parity fixtures the Rust tests compare against.

Examples:
  python3 scripts/gemma4_e2b_dump.py --inspect
  python3 scripts/gemma4_e2b_dump.py --reference --out fixtures/gemma4_e2b \
      --prompt "The capital of France is" --hidden-states
"""

import argparse
import json
import os
import struct
import sys

MODEL = "google/gemma-4-E2B-it-qat-mobile-transformers"


def _snapshot_dir() -> str:
    from huggingface_hub import snapshot_download

    # Already-downloaded files are reused; nothing re-fetched.
    return snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"])


def cmd_inspect(_args) -> int:
    from safetensors import safe_open

    d = _snapshot_dir()
    cfg = json.load(open(os.path.join(d, "config.json")))
    print("=== quantization_config ===")
    print(json.dumps(cfg.get("quantization_config", {}), indent=2))
    print("\n=== text_config (key fields) ===")
    tc = cfg.get("text_config", {})
    for k in (
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "hidden_size_per_layer_input", "vocab_size_per_layer_input",
        "num_kv_shared_layers", "use_double_wide_mlp", "head_dim",
        "global_head_dim", "num_attention_heads", "num_key_value_heads",
    ):
        print(f"  {k}: {tc.get(k)}")

    st = os.path.join(d, "model.safetensors")
    print(f"\n=== tensors in {os.path.basename(st)} ===")
    seen_patterns = {}
    with safe_open(st, framework="pt") as f:
        for name in f.keys():
            sl = f.get_slice(name)
            shape = sl.get_shape()
            dtype = sl.get_dtype()
            # Collapse layer indices to {i} for a compact unique view.
            import re
            pat = re.sub(r"\.\d+\.", ".{i}.", name)
            if pat not in seen_patterns:
                seen_patterns[pat] = (dtype, shape)
                print(f"  {dtype:>8}  {str(shape):>22}  {name}")
    print(f"\n=== {len(seen_patterns)} unique name patterns ===")
    for pat, (dt, sh) in sorted(seen_patterns.items()):
        print(f"  {dt:>8}  {str(sh):>22}  {pat}")
    return 0


def _write_bin(path: str, tensor) -> list:
    import torch

    t = tensor.detach().to(torch.float32).contiguous().cpu().numpy()
    with open(path, "wb") as fh:
        fh.write(t.tobytes())
    return list(t.shape)


def cmd_reference(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids
    meta = {"prompt": args.prompt, "input_ids": ids[0].tolist(), "shapes": {}}
    with torch.no_grad():
        out = model(ids, output_hidden_states=args.hidden_states, use_cache=False)

    meta["shapes"]["logits"] = _write_bin(os.path.join(args.out, "logits.bin"), out.logits)
    # Last-token logits are the most-used parity signal.
    meta["shapes"]["last_logits"] = _write_bin(
        os.path.join(args.out, "last_logits.bin"), out.logits[:, -1, :]
    )
    if args.hidden_states:
        for i, hs in enumerate(out.hidden_states):
            meta["shapes"][f"hidden_{i}"] = _write_bin(
                os.path.join(args.out, f"hidden_{i}.bin"), hs
            )
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"wrote reference to {args.out}/ ({len(meta['shapes'])} tensors)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--out", default="fixtures/gemma4_e2b")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--hidden-states", action="store_true")
    args = ap.parse_args()
    if args.inspect:
        return cmd_inspect(args)
    if args.reference:
        return cmd_reference(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
