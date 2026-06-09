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

"""Export Silero VAD ONNX branch weights to safetensors for rlx-vad embedding.

The 16 kHz export matches `crates/rlx-vad/weights/silero_vad_16k.safetensors`
(`include_bytes!` in Rust). This is **not** the Hugging Face file of the same
name (that file is the 8 kHz graph).

Example::

    curl -sL -o /tmp/silero_vad.onnx \\
      https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
    python3 scripts/export_silero_onnx_weights.py /tmp/silero_vad.onnx \\
      crates/rlx-vad/weights/silero_vad_16k.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


def branch_graph(model: onnx.ModelProto, sixteen_k: bool):
    if_node = next(n for n in model.graph.node if n.op_type == "If")
    # Equal(sr, 16000): true -> then_branch (16 kHz), false -> else_branch (8 kHz)
    return if_node.attribute[0].g if sixteen_k else if_node.attribute[1].g


def float_tensors_by_shape(g) -> dict[tuple, np.ndarray]:
    by_shape: dict[tuple, np.ndarray] = {}
    for n in g.node:
        if n.op_type != "Constant":
            continue
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                arr = numpy_helper.to_array(a.t)
                if arr.dtype == np.float32 and arr.ndim >= 1:
                    by_shape[tuple(arr.shape)] = arr.astype(np.float32)
    return by_shape


def pick(by_shape: dict[tuple, np.ndarray], shape: tuple, label: str) -> np.ndarray:
    if shape not in by_shape:
        raise KeyError(f"missing tensor shape {shape} ({label})")
    return by_shape[shape]


def export(onnx_path: Path, out_path: Path, sixteen_k: bool = True) -> None:
    model = onnx.load(str(onnx_path))
    by_shape = float_tensors_by_shape(branch_graph(model, sixteen_k))
    if sixteen_k:
        shapes = [
            ("stft_conv.weight", (130, 1, 128)),
            ("conv1.weight", (128, 65, 3)),
            ("conv1.bias", (128,)),
            ("conv2.weight", (64, 128, 3)),
            ("conv2.bias", (64,)),
            ("conv3.weight", (64, 64, 3)),
            ("conv3.bias", (64,)),
            ("conv4.weight", (128, 64, 3)),
            ("conv4.bias", (128,)),
            ("lstm_cell.weight_ih", (512, 128)),
            ("lstm_cell.weight_hh", (512, 128)),
            ("lstm_cell.bias_ih", (512,)),
            ("lstm_cell.bias_hh", (512,)),
            ("final_conv.weight", (1, 128, 1)),
            ("final_conv.bias", (1,)),
        ]
    else:
        shapes = [
            ("stft_conv.weight", (258, 1, 256)),
            ("conv1.weight", (128, 129, 3)),
            ("conv1.bias", (128,)),
            ("conv2.weight", (64, 128, 3)),
            ("conv2.bias", (64,)),
            ("conv3.weight", (64, 64, 3)),
            ("conv3.bias", (64,)),
            ("conv4.weight", (128, 64, 3)),
            ("conv4.bias", (128,)),
            ("lstm_cell.weight_ih", (512, 128)),
            ("lstm_cell.weight_hh", (512, 128)),
            ("lstm_cell.bias_ih", (512,)),
            ("lstm_cell.bias_hh", (512,)),
            ("final_conv.weight", (1, 128, 1)),
            ("final_conv.bias", (1,)),
        ]

    tensors = {name: pick(by_shape, shape, name) for name, shape in shapes}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, {'16k' if sixteen_k else '8k'})")
    for name, arr in tensors.items():
        print(f"  {name}: {tuple(arr.shape)} {arr.dtype}")


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
    p.add_argument("--8k", dest="eight_k", action="store_true")
    args = p.parse_args()
    if not args.onnx.is_file():
        raise SystemExit(f"missing ONNX: {args.onnx}")
    export(args.onnx, args.out, sixteen_k=not args.eight_k)


if __name__ == "__main__":
    main()
