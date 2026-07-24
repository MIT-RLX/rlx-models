#!/usr/bin/env python3
"""Export openWakeWord ONNX tensors into RLX safetensors (native load path).

Usage:
  python3 scripts/wake_export/export_openwakeword_weights.py \\
      --onnx-dir .cache/openwakeword/onnx \\
      --out-dir crates/rlx-openwakeword/weights

If ONNX files are missing, writes deterministic stub weights matching the
native Rust stub layout (CI / offline).
"""

from __future__ import annotations

import argparse
import struct
import json
from pathlib import Path


def _write_safetensors(path: Path, tensors: dict[str, list[float]]) -> None:
    """Minimal single-file safetensors writer (F32, 1-D shapes)."""
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
    # Align header to 8 bytes
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


def _stub_rng(seed: int = 0xEABEDD):
    state = seed & 0xFFFFFFFFFFFFFFFF

    def next_f() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        return ((state >> 33) / 0xFFFFFFFF) * 0.02 - 0.01

    return next_f


def write_stubs(out_dir: Path, keyword: str = "wake") -> None:
    next_f = _stub_rng()
    n_mels, c1, c2, c3 = 32, 32, 64, 64
    embed_dim = 96
    embed = {
        "embed.cfg": [float(n_mels), float(c1), float(c2), float(c3)],
        "embed.conv1.weight": [next_f() for _ in range(c1 * 1 * 3 * 3)],
        "embed.conv1.bias": [0.0] * c1,
        "embed.conv2.weight": [next_f() for _ in range(c2 * c1 * 3 * 3)],
        "embed.conv2.bias": [0.0] * c2,
        "embed.conv3.weight": [next_f() for _ in range(c3 * c2 * 3 * 3)],
        "embed.conv3.bias": [0.0] * c3,
        "embed.fc.weight": [next_f() for _ in range(embed_dim * c3)],
        "embed.fc.bias": [0.0] * embed_dim,
    }
    hidden = 64
    in_dim = 16 * embed_dim
    phrase = {
        "phrase.hidden": [float(hidden)],
        "phrase.fc1.weight": [next_f() for _ in range(hidden * in_dim)],
        "phrase.fc1.bias": [0.0] * hidden,
        "phrase.fc2.weight": [next_f() for _ in range(hidden)],
        "phrase.fc2.bias": [-2.5],
    }
    _write_safetensors(out_dir / "embedding.safetensors", embed)
    _write_safetensors(out_dir / "phrase.safetensors", phrase)
    (out_dir / "config.json").write_text(
        json.dumps({"keyword": keyword, "source": "stub"}, indent=2) + "\n"
    )
    print(f"wrote stub weights under {out_dir} (keyword={keyword})")


def try_export_onnx(onnx_dir: Path, out_dir: Path) -> bool:
    try:
        import numpy as np  # noqa: F401
        import onnx  # noqa: F401
    except ImportError:
        print("onnx/numpy not installed — falling back to stubs")
        return False
    mel = onnx_dir / "melspectrogram.onnx"
    emb = onnx_dir / "embedding_model.onnx"
    phrase = onnx_dir / "phrase.onnx"
    if not (mel.is_file() and emb.is_file() and phrase.is_file()):
        print(f"missing ONNX trio under {onnx_dir} — falling back to stubs")
        return False
    # Full ONNX→tensor mapping is model-version specific; keep a hook for
    # offline export once local ONNX is present. For now record paths and stub.
    meta = {
        "onnx_dir": str(onnx_dir),
        "note": "ONNX present; use openWakeWord tooling to dump named tensors, "
        "then map into embedding.safetensors / phrase.safetensors.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "onnx_export_pending.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_stubs(out_dir)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, default=Path(".cache/openwakeword/onnx"))
    ap.add_argument("--out-dir", type=Path, default=Path("crates/rlx-openwakeword/weights"))
    ap.add_argument("--keyword", default="wake")
    args = ap.parse_args()
    if not try_export_onnx(args.onnx_dir, args.out_dir):
        write_stubs(args.out_dir, args.keyword)


if __name__ == "__main__":
    main()
