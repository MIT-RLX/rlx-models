#!/usr/bin/env python3
"""Export nanowakeword ONNX CNN weights to RLX WakeCnn safetensors.

Usage:
  python3 scripts/wake_export/export_nanowakeword_weights.py \\
      --onnx .cache/nanowakeword/hey_nano_cnn_v1.onnx \\
      --out crates/rlx-nanowakeword/weights/hey_nano.safetensors

Without ONNX, writes a lite stub matching `WakeCnnConfig::lite()`.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def _write_safetensors(path: Path, tensors: dict[str, list[float]]) -> None:
    header: dict = {}
    data = bytearray()
    offset = 0
    for name, values in tensors.items():
        raw = b"".join(struct.pack("<f", float(v)) for v in values)
        header[name] = {
            "dtype": "F32",
            "shape": [len(values)],
            "data_offsets": [offset, offset + len(raw)],
        }
        data.extend(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


def write_lite_stub(path: Path) -> None:
    seed = 0xC0FFEE
    state = seed

    def next_f() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        return ((state >> 33) / 0xFFFFFFFF) * 0.02 - 0.01

    n_mels, c1, c2, c3, k, hidden = 32, 16, 32, 32, 3, 64
    tensors = {
        "cfg": [float(x) for x in (n_mels, c1, c2, c3, k, hidden)],
        "conv1.weight": [next_f() for _ in range(c1 * n_mels * k)],
        "conv1.bias": [0.0] * c1,
        "conv2.weight": [next_f() for _ in range(c2 * c1 * k)],
        "conv2.bias": [0.0] * c2,
        "conv3.weight": [next_f() for _ in range(c3 * c2 * k)],
        "conv3.bias": [0.0] * c3,
        "fc1.weight": [next_f() for _ in range(hidden * c3)],
        "fc1.bias": [0.0] * hidden,
        "fc2.weight": [next_f() for _ in range(hidden)],
        "fc2.bias": [-2.0],
    }
    _write_safetensors(path, tensors)
    print(f"wrote stub {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("crates/rlx-nanowakeword/weights/model_lite.safetensors"),
    )
    args = ap.parse_args()
    if args.onnx is None or not args.onnx.is_file():
        write_lite_stub(args.out)
        return
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("onnx not installed — writing stub")
        write_lite_stub(args.out)
        return
    # Hook for full initializer dump → WakeCnn name map.
    meta = {"onnx": str(args.onnx), "status": "pending_name_map"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / "onnx_export_pending.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_lite_stub(args.out)


if __name__ == "__main__":
    main()
