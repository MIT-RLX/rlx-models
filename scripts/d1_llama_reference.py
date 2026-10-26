#!/usr/bin/env python3
"""Quantized-model references for `crates/rlx-d1`: llama.cpp's own d1 port.

Sends every case in `crates/rlx-d1/tests/fixtures/d1_cases.json` to a running
`llama-server` (`/v1/systemone`, llama.cpp's `server-decision.cpp`) loaded
with a `LiquidAI/d1-3B-GGUF` text model and its mmproj, and writes the
answers as a fixture. Pictures are sent as data URLs.

    llama-server -m d1-3B-Q4_K_M.gguf --mmproj mmproj-d1-3B-Q8_0.gguf -ngl 99 --port 8091
    python3 scripts/d1_llama_reference.py --url http://127.0.0.1:8091 \\
        --images /tmp/d1img --out crates/rlx-d1/tests/fixtures/d1_reference_q4_k_m.json

Run the server on a GPU backend (`-ngl 99`). llama.cpp's CPU path quantizes
activations to 8 bits before every K-quant matmul (`vec_dot_type`); its GPU
paths, like rlx, keep them in float. On a flat answer the CPU path moves
probabilities by ~1e-2 (d1's `urgency` score: 0.366 CPU vs 0.3776 Metal and
0.3777 rlx), which is llama.cpp's choice, not the checkpoint's.

Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path

CASES = Path(__file__).resolve().parent.parent / "crates/rlx-d1/tests/fixtures/d1_cases.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--backend", default="metal (-ngl 99)", help="recorded in the fixture")
    ap.add_argument("--mmproj", default="mmproj-d1-3B-Q8_0.gguf", help="recorded in the fixture")
    args = ap.parse_args()

    props = json.load(urllib.request.urlopen(f"{args.url}/props"))
    results = []
    for case in json.loads(Path(args.cases).read_text()):
        body = {"state": case["state"], "questions": case["questions"]}
        images = []
        for name in case.get("images", []):
            data = (Path(args.images) / name).read_bytes()
            mime = "image/png" if name.endswith(".png") else "image/jpeg"
            images.append(f"data:{mime};base64,{base64.b64encode(data).decode()}")
        if images:
            body["images"] = images
        req = urllib.request.Request(
            f"{args.url}/v1/systemone",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        out = json.load(urllib.request.urlopen(req))
        out.pop("model", None)
        print(f"[d1-llama] {case['name']}: {out['usage']}", flush=True)
        results.append({"name": case["name"], "system_one": out})

    Path(args.out).write_text(json.dumps({
        "llama_cpp": props.get("build_info", ""),
        "backend": args.backend,
        "model": Path(props.get("model_path", "")).name,
        "mmproj": args.mmproj,
        "cases": results,
    }, ensure_ascii=False, indent=1))
    print(f"[d1-llama] wrote {args.out}")


if __name__ == "__main__":
    main()
