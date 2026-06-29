#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Batch-export AIF token-dynamics probes for VLMEvalKit-style JSONL datasets.
#
# ```bash
# RLX_QWEN25_VL_HF_DIR=/path/to/Qwen2.5-VL-7B-Instruct \
# python3 scripts/aif_export_probes.py \
#   --jsonl /path/realworldqa.jsonl \
#   --image-root /path/images \
#   --out-dir /tmp/aif-probes \
#   --limit 100 \
#   --vlmevalkit-prompt
# ```

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "crates/rlx-models/tests/qwen25_vl_parity_helpers"
sys.path.insert(0, str(HELPER))

from aif_probe import resolve_model_dir, run_hf_prefill_probe, save_probe_sample  # noqa: E402


def load_jsonl(path: Path, image_root: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("id") or row.get("question_id") or f"line_{line_no}"
            rel = row.get("image_path") or row.get("image")
            if not rel:
                raise SystemExit(f"line {line_no}: missing image path")
            yield sid, image_root / rel, row["question"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export AIF probes for a VQA JSONL")
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--hf-dir", default=None, help="HF model dir (or RLX_QWEN25_VL_HF_DIR)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vlmevalkit-prompt", action="store_true")
    args = ap.parse_args()

    if args.hf_dir:
        import os

        os.environ["RLX_QWEN25_VL_HF_DIR"] = str(args.hf_dir)
    model_dir = resolve_model_dir()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for sid, image_path, question in load_jsonl(args.jsonl, args.image_root):
        if args.limit and n >= args.limit:
            break
        if not image_path.is_file():
            print(f"skip missing {image_path}", file=sys.stderr)
            continue
        safe_id = str(sid).replace("/", "_")
        print(f"[{n + 1}] probe {safe_id} …", flush=True)
        result = run_hf_prefill_probe(
            model_dir=model_dir,
            image_path=str(image_path),
            question=question,
            device=args.device,
            vlmevalkit=args.vlmevalkit_prompt,
        )
        save_probe_sample(args.out_dir, safe_id, result)
        n += 1

    print(f"wrote {n} probes to {args.out_dir}")


if __name__ == "__main__":
    main()
