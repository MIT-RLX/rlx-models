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

"""Bake ref_code + ref_spk_embedding fixtures for the Rust voice-clone bench.

Uses the qwen_tts Python package against the Base checkpoint. Emits a single
JSON file with ICL-ready arrays; the Rust bench loads it as the input to
`build_icl_prompt` / `build_x_vector_prompt`.

Inputs:
  --base-dir   path to Qwen3-TTS-12Hz-0.6B-Base
  --ref-wav    path to a 24 kHz reference WAV
  --ref-text   reference transcript (required for ICL)
  --out-json   output path

The same fixture serves both modes — ICL uses ref_code+ref_text+spk, x-vector
uses only spk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_FLASH_ATTN", "1")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True, type=Path)
    p.add_argument("--ref-wav", required=True, type=Path)
    p.add_argument("--ref-text", required=True, type=str)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = p.parse_args()

    import torch
    from qwen_tts import Qwen3TTSModel

    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    print(f"loading Base from {args.base_dir} on {args.device}", file=sys.stderr)
    model = Qwen3TTSModel.from_pretrained(
        str(args.base_dir),
        torch_dtype=torch.float32,
        device_map=args.device,
        attn_implementation="sdpa",
    )

    print(f"baking from {args.ref_wav}", file=sys.stderr)
    items = model.create_voice_clone_prompt(
        ref_audio=str(args.ref_wav),
        ref_text=args.ref_text,
        x_vector_only_mode=False,
    )
    item = items[0]
    ref_code = item.ref_code.detach().cpu().numpy()
    spk = item.ref_spk_embedding.detach().cpu().to(torch.float32).numpy()

    if ref_code.ndim == 1:
        ref_code = ref_code.reshape(-1, 1)
    n_frames, n_groups = ref_code.shape
    print(f"ref_code: {n_frames} frames x {n_groups} groups", file=sys.stderr)
    print(f"spk_embedding: {spk.shape} dtype={spk.dtype}", file=sys.stderr)

    payload = {
        "ref_text": args.ref_text,
        "ref_wav": str(args.ref_wav),
        "n_frames": int(n_frames),
        "n_groups": int(n_groups),
        "ref_code": [[int(x) for x in row] for row in ref_code.tolist()],
        "spk_dim": int(spk.shape[-1]),
        "ref_spk_embedding": [float(x) for x in spk.reshape(-1).tolist()],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump(payload, fh)
    print(f"wrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
