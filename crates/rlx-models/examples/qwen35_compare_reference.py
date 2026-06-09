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

#
"""reference companion for `examples/qwen35_compare.rs`.

Runs the same GGUF through `llama-cpp-python` or the Rust reference
(`cargo run --features parity-llama -p rlx-models --example qwen35_compare_reference`).
prints the top-K in the same `REF_LOGIT idx=… token=… value=…`
format. Pipe both binaries' stdout to a diff tool or score the
top-K overlap.

Usage:
    pip install llama-cpp-python
    python3 qwen35_compare_reference.py <weights.gguf> \\
        --prompt-ids 1,2,3 [--top-k 16]

Then compare:
    cargo run --release -p rlx-models --example qwen35_compare -- \\
        <weights.gguf> --packed --prompt-ids 1,2,3 --top-k 16 \\
        | sort > rlx.out
    python3 qwen35_compare_reference.py <weights.gguf> \\
        --prompt-ids 1,2,3 --top-k 16 | sort > ref.out
    diff <(grep -oP 'token=\\d+' rlx.out) <(grep -oP 'token=\\d+' ref.out)

A perfect parity yields identical top-K id ordering; ranking
divergences (esp. swap of #1) are the loudest signal that the
forward graph deviates from the reference.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="qwen35 parity reference (llama-cpp-python)")
    p.add_argument("weights", help="path to .gguf")
    p.add_argument(
        "--prompt-ids",
        default="1,2,3",
        help="comma-separated u32 token ids (default: 1,2,3)",
    )
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        help="context size (default: model's max)",
    )
    args = p.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError:
        print(
            "ERROR: `llama-cpp-python` not installed. Run:\n"
            "    pip install llama-cpp-python\n"
            "(set CMAKE_ARGS='-DGGML_METAL=on' on macOS for Metal builds).",
            file=sys.stderr,
        )
        return 2

    prompt_ids = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
    if not prompt_ids:
        print("ERROR: empty prompt-ids", file=sys.stderr)
        return 2

    print(f"# Loading {args.weights} via llama-cpp-python…", file=sys.stderr)
    llm = Llama(
        model_path=args.weights,
        n_ctx=args.n_ctx or 4096,
        logits_all=True,
        verbose=False,
    )

    # Direct token-id eval (skip tokenizer to match the RLX caller).
    llm.eval(prompt_ids)
    logits = llm.eval_logits[-1]  # last-token logits
    n_vocab = len(logits)
    print(f"# REF logits: len={n_vocab}", file=sys.stderr)

    # Top-K.
    pairs = sorted(enumerate(logits), key=lambda kv: kv[1], reverse=True)[: args.top_k]
    for rank, (tok_id, val) in enumerate(pairs):
        print(f"REF_LOGIT idx={rank} token={tok_id} value={val:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
