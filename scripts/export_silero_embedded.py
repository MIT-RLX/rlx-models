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

"""Export Silero VAD ONNX 16 kHz branch weights to legacy RLXV blob.

Prefer `export_silero_onnx_weights.py` (safetensors) for rlx-vad embedding.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MAGIC = b"RLXV"
VERSION = 1


def branch_16k(model: onnx.ModelProto):
    if_node = next(n for n in model.graph.node if n.op_type == "If")
    return if_node.attribute[0].g


def float_tensors(g) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for n in g.node:
        if n.op_type != "Constant":
            continue
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                arr = numpy_helper.to_array(a.t)
                if arr.dtype == np.float32 and arr.ndim >= 1:
                    out.append(arr.astype(np.float32))
    by_shape: dict[tuple, np.ndarray] = {}
    for arr in out:
        by_shape[tuple(arr.shape)] = arr.reshape(-1) if arr.shape == (1,) else arr
    return by_shape


def export(onnx_path: Path, out_path: Path) -> None:
    model = onnx.load(str(onnx_path))
    by_shape = float_tensors(branch_16k(model))
    expected = [
        (130, 1, 128),
        (128, 65, 3),
        (128,),
        (64, 128, 3),
        (64,),
        (64, 64, 3),
        (64,),
        (128, 64, 3),
        (128,),
        (512, 128),
        (512, 128),
        (512,),
        (512,),
        (1, 128, 1),
        (1,),
    ]
    parts = []
    for shape in expected:
        key = shape
        if shape not in by_shape:
            raise RuntimeError(f"missing tensor shape {shape}")
        arr = by_shape[key]
        if shape == (1,):
            parts.append(arr.reshape(1))
        else:
            parts.append(arr)
    blob = bytearray(MAGIC)
    blob.extend(struct.pack("<I", VERSION))
    for arr in parts:
        blob.extend(arr.tobytes(order="C"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    print(f"wrote {out_path} ({len(blob)} bytes, {len(parts)} tensors)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "onnx",
        nargs="?",
        type=Path,
        default=Path("/tmp/silero_vad.onnx"),
    )
    p.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path("crates/rlx-vad/weights/silero_vad_16k.safetensors"),
    )
    args = p.parse_args()
    if not args.onnx.is_file():
        raise SystemExit(f"missing ONNX: {args.onnx}")
    export(args.onnx, args.out)


if __name__ == "__main__":
    main()
