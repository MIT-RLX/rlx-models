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

"""Export TEN-VAD (`ten-vad.onnx` + `src/coeff.h`) to the safetensors bundle
embedded by `rlx-ten-vad`.

Two things happen here that the Rust side would otherwise have to redo at load
time:

* **LSTM gate reorder** — ONNX packs `i, o, f, c`; `rlx_ir::Op::Lstm` uses the
  PyTorch order `i, f, g, o`. The two ONNX bias halves (`Wb`, `Rb`) are summed
  into the single combined bias `Op::Lstm` expects.
* **Length axis moves H-ward** — after the first (genuinely 2D) conv every
  activation is `[N, C, 1, W]`. rlx graphs carry 1-D convs as `[N, C, T, 1]`
  (length in H), so the separable kernels are reshaped `[16,1,1,3] → [16,1,3,1]`.

The DSP tables from `src/coeff.h` (Hann-768 analysis window, per-feature mean /
std) ride along in the same file so the crate has a single embedded asset.

Example::

    git clone --depth 1 https://github.com/TEN-framework/ten-vad /tmp/ten-vad
    python3 scripts/export_ten_vad_onnx_weights.py /tmp/ten-vad \\
      crates/rlx-ten-vad/weights/ten_vad.safetensors
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

HIDDEN = 64
FEA_LEN = 41
WINDOW_SZ = 768

# ONNX initializer names are TensorFlow export paths; map them to short ones.
VAD = "StatefulPartitionedCall/vad_model"
ONNX_NAMES = {
    "conv0.depthwise.weight": "const_fold_opt__178",
    "conv0.pointwise.weight": f"{VAD}/separable_conv2d/separable_conv2d/ReadVariableOp_1:0",
    "conv0.bias": f"{VAD}/separable_conv2d/BiasAdd/ReadVariableOp:0",
    "sep1.depthwise.weight": "const_fold_opt__179",
    "sep1.pointwise.weight": f"{VAD}/separable_conv1d/ExpandDims_2:0",
    "sep1.bias": f"{VAD}/separable_conv1d/BiasAdd/ReadVariableOp:0",
    "sep2.depthwise.weight": "const_fold_opt__180",
    "sep2.pointwise.weight": f"{VAD}/separable_conv1d_1/ExpandDims_2:0",
    "sep2.bias": f"{VAD}/separable_conv1d_1/BiasAdd/ReadVariableOp:0",
    "dense1.weight": f"{VAD}/dense_3/Tensordot/ReadVariableOp:0",
    "dense1.bias": f"{VAD}/dense_3/BiasAdd/ReadVariableOp:0",
    "dense2.weight": f"{VAD}/dense_5/Tensordot/ReadVariableOp:0",
    "dense2.bias": f"{VAD}/dense_5/BiasAdd/ReadVariableOp:0",
}
LSTM_ONNX = {
    "lstm1": ("W0__70", "R0__71", "B0__72"),
    "lstm2": ("W0__99", "R0__100", "B0__101"),
}

EXPECT_SHAPES = {
    "conv0.depthwise.weight": (1, 1, 3, 3),
    "conv0.pointwise.weight": (16, 1, 1, 1),
    "conv0.bias": (16,),
    "sep1.depthwise.weight": (16, 1, 3, 1),
    "sep1.pointwise.weight": (16, 16, 1, 1),
    "sep1.bias": (16,),
    "sep2.depthwise.weight": (16, 1, 3, 1),
    "sep2.pointwise.weight": (16, 16, 1, 1),
    "sep2.bias": (16,),
    "lstm1.weight_ih": (4 * HIDDEN, 80),
    "lstm1.weight_hh": (4 * HIDDEN, HIDDEN),
    "lstm1.bias": (4 * HIDDEN,),
    "lstm2.weight_ih": (4 * HIDDEN, HIDDEN),
    "lstm2.weight_hh": (4 * HIDDEN, HIDDEN),
    "lstm2.bias": (4 * HIDDEN,),
    "dense1.weight": (128, 32),
    "dense1.bias": (32,),
    "dense2.weight": (32, 1),
    "dense2.bias": (1,),
    "feature.mean": (FEA_LEN,),
    "feature.std": (FEA_LEN,),
    "stft.window": (WINDOW_SZ,),
}


def gate_reorder(rows: np.ndarray) -> np.ndarray:
    """ONNX `i, o, f, c` gate blocks → PyTorch / rlx `i, f, g, o`."""
    i, o, f, c = np.split(rows, 4, axis=0)
    return np.concatenate([i, f, c, o], axis=0)


def parse_coeff_array(src: str, name: str, count: int) -> np.ndarray:
    match = re.search(re.escape(name) + r"\[[^\]]*\]\s*=\s*\{(.*?)\};", src, re.S)
    if match is None:
        raise SystemExit(f"{name} not found in coeff.h")
    body = match.group(1).replace("f,", ",").replace("f}", "}")
    values = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", body)]
    if len(values) != count:
        raise SystemExit(f"{name}: expected {count} values, found {len(values)}")
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path, help="TEN-framework/ten-vad checkout")
    ap.add_argument("out", type=Path, help="destination .safetensors")
    args = ap.parse_args()

    model = onnx.load(str(args.repo / "src/onnx_model/ten-vad.onnx"))
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    out: dict[str, np.ndarray] = {}
    for short, onnx_name in ONNX_NAMES.items():
        if onnx_name not in init:
            raise SystemExit(f"missing initializer {onnx_name}")
        arr = np.asarray(init[onnx_name], dtype=np.float32)
        if short.startswith(("sep1.depthwise", "sep2.depthwise")):
            # [C_out, 1, 1, 3] → [C_out, 1, 3, 1]: same data, length axis to H.
            arr = arr.reshape(arr.shape[0], 1, arr.shape[3], 1)
        out[short] = np.ascontiguousarray(arr)

    for short, (w_name, r_name, b_name) in LSTM_ONNX.items():
        w = np.asarray(init[w_name], dtype=np.float32)[0]  # [4H, input]
        r = np.asarray(init[r_name], dtype=np.float32)[0]  # [4H, H]
        b = np.asarray(init[b_name], dtype=np.float32)[0]  # [8H] = Wb ‖ Rb
        out[f"{short}.weight_ih"] = np.ascontiguousarray(gate_reorder(w))
        out[f"{short}.weight_hh"] = np.ascontiguousarray(gate_reorder(r))
        out[f"{short}.bias"] = np.ascontiguousarray(
            gate_reorder(b[: 4 * HIDDEN] + b[4 * HIDDEN :])
        )

    coeff = (args.repo / "src/coeff.h").read_text()
    out["feature.mean"] = parse_coeff_array(coeff, "AUP_AED_FEATURE_MEANS", FEA_LEN)
    out["feature.std"] = parse_coeff_array(coeff, "AUP_AED_FEATURE_STDS", FEA_LEN)
    out["stft.window"] = parse_coeff_array(
        coeff, "AUP_AED_STFTWindow_Hann768", WINDOW_SZ
    )

    for name, expect in EXPECT_SHAPES.items():
        got = out[name].shape
        if got != expect:
            raise SystemExit(f"{name}: shape {got}, expected {expect}")
    if set(out) != set(EXPECT_SHAPES):
        raise SystemExit(f"tensor set mismatch: {sorted(set(out) ^ set(EXPECT_SHAPES))}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out))
    total = sum(v.size for v in out.values())
    print(f"wrote {args.out} — {len(out)} tensors, {total} f32 values")


if __name__ == "__main__":
    main()
