#!/usr/bin/env python3
"""Compute reference Jacobians with the Python `jlens`, for comparison with rlx.

Everything self-consistent inside rlx-jlens still leaves one question open: does
it agree with the implementation it was ported from? This dumps `J_l` from the
reference for a fixed prompt so `compare_reference.rs` can diff it entry-wise.

The exact token ids are dumped alongside, because a tokenizer mismatch would
show up as a numerical disagreement and be blamed on the estimator.

    python3 reference_jacobian.py --model ../../weights/Qwen3-0.6B \
        --out /tmp/ref.npz --layers 3,4,5 --target 6 --seq 16
"""

import argparse
import json
import sys

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="safetensors path")
    ap.add_argument("--prompt", default="The capital of France is Paris and the capital of Japan is Tokyo.")
    ap.add_argument("--layers", default="3,4,5")
    ap.add_argument("--target", type=int, default=6)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--skip-first", type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, "/Users/Shared/jacobian-lens")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.fitting import jacobian_for_prompt
    from jlens.hf import from_hf

    layers = [int(x) for x in args.layers.split(",")]

    # f32 on CPU: the point is to compare estimators, not to reproduce a
    # particular kernel's rounding, and rlx runs this comparison on CPU too.
    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    model = from_hf(hf, tok)

    ids = model.encode(args.prompt, max_length=args.seq)
    print(f"prompt -> {ids.shape[1]} tokens: {ids[0].tolist()}", file=sys.stderr)

    jac, seq_len, n_valid = jacobian_for_prompt(
        model,
        args.prompt,
        layers,
        target_layer=args.target,
        dim_batch=args.dim_batch,
        max_seq_len=args.seq,
        skip_first=args.skip_first,
    )
    print(f"seq_len={seq_len} n_valid_positions={n_valid}", file=sys.stderr)

    # safetensors rather than npz so the Rust side can read this with the
    # dependency it already has, and f32 so the comparison is not floored by the
    # artifact format the way a f16 lens file would be.
    from safetensors.numpy import save_file

    save_file(
        {f"J.{l}": np.ascontiguousarray(jac[l].numpy().astype(np.float32)) for l in layers},
        args.out,
        metadata={
            "ids": ",".join(str(int(v)) for v in ids[0].cpu().tolist()),
            "target": str(args.target),
            "dim_batch": str(args.dim_batch),
            "skip_first": str(args.skip_first),
        },
    )
    meta = {
        "layers": layers,
        "target": args.target,
        "seq": int(seq_len),
        "n_valid": int(n_valid),
        "dim_batch": args.dim_batch,
        "skip_first": args.skip_first,
        "d_model": int(jac[layers[0]].shape[0]),
        "input_ids": ids[0].cpu().tolist(),
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out} ({len(layers)} layers, d_model {meta['d_model']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
