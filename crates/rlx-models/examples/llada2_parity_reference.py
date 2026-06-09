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

"""PyTorch reference for RLX LLaDA2 parity (TIDE modeling_llada2_moe.py).

Usage (from repo root, with model weights + transformers on PYTHONPATH):

    export LLADA2_MODEL_DIR=/path/to/hf/checkpoint  # config.json + model.safetensors
    python3 rlx-models/examples/llada2_parity_reference.py \\
        --prompt-ids 1,2,3 --seq-len 8 --top-k 8

Prints REF_LOGIT lines for comparison with:
    cargo run --release -p rlx-models --example llada2_compare -- ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-dir",
        default=os.environ.get("LLADA2_MODEL_DIR", "/Users/Shared/TIDE/model"),
    )
    p.add_argument("--prompt-ids", default="1,2,3")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--top-k", type=int, default=8)
    args = p.parse_args()

    model_dir = args.model_dir
    sys.path.insert(0, model_dir)
    try:
        import torch
        from configuration_llada2_moe import LLaDA2MoeConfig
        from modeling_llada2_moe import LLaDA2MoeModelLM
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Need torch + TIDE model sources on PYTHONPATH", file=sys.stderr)
        return 2

    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        raw = json.load(f)
    cfg = LLaDA2MoeConfig(**{k: v for k, v in raw.items() if k != "dtype"})

    prompt = [int(x) for x in args.prompt_ids.split(",") if x.strip()]
    seq_len = args.seq_len
    block_length = min(32, seq_len)
    num_blocks = (len(prompt) + seq_len - len(prompt) + block_length - 1) // block_length
    total = num_blocks * block_length
    if total < seq_len:
        total = seq_len

    device = "cpu"
    model = LLaDA2MoeModelLM(cfg).to(device).eval()

    x = torch.full((1, total), cfg.pad_token_id if hasattr(cfg, "pad_token_id") else 156895, dtype=torch.long)
    plen = len(prompt)
    x[0, :plen] = torch.tensor(prompt, dtype=torch.long)
    pos = torch.arange(total, device=device).unsqueeze(0)

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
        .log()[:, :, :seq_len, :seq_len]
    )

    with torch.no_grad():
        out = model(
            input_ids=x[:, :seq_len],
            attention_mask=attn,
            position_ids=pos[:, :seq_len],
        )
        logits = out.logits[0].float()

    for pos in range(min(seq_len, logits.shape[0])):
        row = logits[pos]
        vals, idx = torch.topk(row, args.top_k)
        for i in range(args.top_k):
            print(f"REF_LOGIT pos={pos} rank={i} token={idx[i].item()} value={vals[i].item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
