#!/usr/bin/env python3
"""Fit per-bin residual AEC mask and export safetensors for rlx-aec.

Usage:
  python3 scripts/aec_train_residual_export.py \\
    --dataset-dir /tmp/aec_dataset \\
    --out crates/rlx-aec/weights/residual_aec.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path


def read_wav(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "r") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    pcm = []
    for i in range(0, len(raw), 2):
        pcm.append(struct.unpack("<h", raw[i : i + 2])[0] / 32768.0)
    return sr, pcm


def write_safetensors(path: Path, scale: list[float], bias: list[float]) -> None:
    import json as js

    tensors = {"scale": scale, "bias": bias}
    blobs = []
    headers = {}
    offset = 0
    for name, data in tensors.items():
        raw = struct.pack(f"<{len(data)}f", *data)
        headers[name] = {
            "dtype": "F32",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header = js.dumps(headers).encode("utf-8")
    pad = (8 - (len(header) % 8)) % 8
    header += b" " * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for b in blobs:
            f.write(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-fft", type=int, default=1024)
    args = ap.parse_args()

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    n_bins = args.n_fft * 2
    scale = [1.0] * n_bins
    bias = [0.0] * n_bins

    # Teacher: identity mask (linear AEC handles echo; residual is passthrough v1).
    # Future: fit scale/bias from STFT(error) → STFT(clean) regression.
    _ = manifest

    write_safetensors(args.out, scale, bias)
    print(f"exported identity residual mask → {args.out} ({n_bins} bins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
