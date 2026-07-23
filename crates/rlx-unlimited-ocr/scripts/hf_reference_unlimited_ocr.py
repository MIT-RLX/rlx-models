#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# HuggingFace reference probes for baidu/Unlimited-OCR stage-wise / e2e parity.
#
# Usage:
#   python3 scripts/hf_reference_unlimited_ocr.py \
#     --model-dir "$RLX_UNLIMITED_OCR_DIR" \
#     --image fixtures/sample.jpg \
#     --mode base \
#     --max-new-tokens 32 \
#     --out /tmp/uo_ref.json

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", choices=("base", "gundam", "multi"), default="base")
    ap.add_argument("--prompt", default="<image>document parsing.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(args.model_dir)
    image_path = Path(args.image)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    image = Image.open(image_path).convert("RGB")

    # Prefer the checkpoint's `infer` helper when present.
    if hasattr(model, "infer"):
        crop_mode = args.mode == "gundam"
        base_size = 1024
        image_size = 640 if crop_mode else 1024
        with torch.no_grad():
            text = model.infer(
                tokenizer,
                prompt=args.prompt,
                image=image,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                eval_mode=True,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                no_repeat_ngram_size=35,
                ngram_window=128 if args.mode != "multi" else 1024,
            )
        # `infer` typically returns decoded text; also try to recover ids if exposed.
        token_ids = None
        if isinstance(text, tuple):
            text, token_ids = text[0], list(text[1]) if len(text) > 1 else None
        payload = {
            "mode": args.mode,
            "prompt": args.prompt,
            "text": text if isinstance(text, str) else str(text),
            "token_ids": token_ids,
            "image": str(image_path),
            "model_dir": str(model_dir),
        }
    else:
        payload = {
            "error": "model.infer missing — custom code not loaded",
            "mode": args.mode,
        }

    # Lightweight tokenizer probe for placeholder expansion.
    image_token_id = 128815
    patch_size = 16
    downsample_ratio = 4
    if args.mode == "gundam":
        q_base = math.ceil((1024 // patch_size) / downsample_ratio)
        q_tile = math.ceil((640 // patch_size) / downsample_ratio)
        payload["num_queries_base"] = q_base
        payload["num_queries_tile"] = q_tile
        payload["base_placeholder_tokens"] = q_base * (q_base + 1) + 1
    else:
        q = math.ceil((1024 // patch_size) / downsample_ratio)
        payload["num_queries"] = q
        payload["base_placeholder_tokens"] = q * (q + 1) + 1
    payload["image_token_id"] = image_token_id
    payload["bos_id"] = 0
    payload["eos_id"] = 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
