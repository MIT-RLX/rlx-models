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

"""Tokenize a user message with the official MiniCPM5 chat template, then run rlx-minicpm5.

Usage:
  pip install transformers
  just fetch-minicpm5
  just minicpm5-chat "What is 2+2?"
  MINICPM5_MODEL_DIR=/path/to/MiniCPM5-1B RLX_MINICPM5_DEVICE=cpu just minicpm5-chat "Hello"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    # CPU decode is correct; MLX uses one-shot decode (see --no-bucketed-decode in Rust).
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description="MiniCPM5 chat → rlx-minicpm5")
    p.add_argument(
        "message",
        nargs="?",
        default="What is the capital of France? Reply in one short sentence.",
    )
    p.add_argument(
        "--model-dir",
        default=os.environ.get(
            "MINICPM5_MODEL_DIR", "/tmp/rlx-weights/MiniCPM5-1B"
        ),
    )
    p.add_argument(
        "--device",
        default=os.environ.get("RLX_MINICPM5_DEVICE", "auto"),
        help="cpu|mlx|metal|auto (auto=cpu for reliable decode)",
    )
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-seq", type=int, default=0, help="0 = auto from prompt+decode")
    p.add_argument("--dry-run", action="store_true", help="print cargo command only")
    args = p.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("pip install transformers")

    model_id = "openbmb/MiniCPM5-1B"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    messages = [{"role": "user", "content": args.message}]
    prompt_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok.encode(prompt_text, add_special_tokens=False)
    weights = os.path.join(args.model_dir, "model-00000-of-00001.safetensors")
    tokenizer = os.path.join(args.model_dir, "tokenizer.json")
    prompt_ids = ",".join(str(i) for i in ids)
    device = pick_device(args.device)
    max_seq = args.max_seq or max(64, len(ids) + args.max_tokens + 16)

    print(f"# user: {args.message!r}")
    print(f"# prompt tokens: {len(ids)}  device: {device}  max_seq: {max_seq}")

    root = os.environ.get("RLX_MODELS_ROOT", ".")
    bin_path = os.path.join(root, "target", "release", "rlx-minicpm5")
    use_bin = os.path.isfile(bin_path)

    base: list[str] = (
        [bin_path]
        if use_bin
        else [
            "cargo",
            "run",
            "-p",
            "rlx-minicpm5",
            "--features",
            "tokenizer,mlx,metal",
            "--release",
            "--",
        ]
    )
    cmd = base + [
        "--weights",
        weights,
        "--tokenizer",
        tokenizer,
        "--device",
        device,
        "--prompt-ids",
        prompt_ids,
        "--max-tokens",
        str(args.max_tokens),
        "--max-seq",
        str(max_seq),
        "--no-stream",
    ]
    if args.dry_run:
        print(" ".join(cmd))
        return

    t0 = time.perf_counter()
    subprocess.check_call(cmd, cwd=root)
    print(f"# wall time: {time.perf_counter() - t0:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
