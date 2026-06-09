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

"""PyTorch / HuggingFace Gemma last-token logits for RLX parity (JSON lines on stdout)."""

from __future__ import annotations

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "usage: gemma_parity_reference.py WEIGHTS.safetensors CONFIG.json",
            file=sys.stderr,
        )
        return 2

    weights_path = sys.argv[1]
    config_path = sys.argv[2]

    # Derive model directory from config path (HF layout).
    import os

    model_dir = os.path.dirname(os.path.abspath(config_path))
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    prompt_ids = [2, 106, 164, 207, 417, 521, 897]
    if tok.eos_token_id is not None:
        prompt_ids.append(tok.eos_token_id)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, -1, :].float().cpu().tolist()

    emit(
        {
            "prompt_ids": prompt_ids,
            "logits": logits,
            "top1": int(max(range(len(logits)), key=lambda i: logits[i])),
        }
    )
    return 0


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
