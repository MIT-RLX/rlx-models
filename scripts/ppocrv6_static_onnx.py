#!/usr/bin/env python3
"""Materialize a PP-OCRv6 ONNX with static NCHW input dims (for rlx-onnx-import)."""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import shape_inference


def materialize(src: Path, dst: Path, n: int, c: int, h: int, w: int) -> None:
    m = onnx.load(str(src))
    del m.graph.value_info[:]
    for inp in m.graph.input:
        if inp.name != "x":
            continue
        while len(inp.type.tensor_type.shape.dim):
            inp.type.tensor_type.shape.dim.pop()
        for v in (n, c, h, w):
            d = inp.type.tensor_type.shape.dim.add()
            d.dim_value = int(v)
    for out in m.graph.output:
        while len(out.type.tensor_type.shape.dim):
            out.type.tensor_type.shape.dim.pop()
    m2 = shape_inference.infer_shapes(m)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m2, str(dst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--c", type=int, default=3)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    args = ap.parse_args()
    materialize(args.src, args.dst, args.n, args.c, args.h, args.w)
    print(args.dst)


if __name__ == "__main__":
    main()
