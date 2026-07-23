#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Export PP-OCRv6 ONNX → RLX rewrites + safetensors for native inference.
#
# Used by `just fetch-ppocrv6-tiny|small`. Writes `inference_rlx.onnx` (emit
# input only) plus `{stem}.safetensors` and `model.safetensors` (runtime).
#
# Example:
#   python3 scripts/export_ppocrv6_onnx_weights.py \
#     .cache/ppocrv6/tiny/det/inference.onnx \
#     .cache/ppocrv6/tiny/det --stem ppocrv6_tiny_det

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from safetensors.numpy import save_file


def rewrite_hardsigmoid(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    new_nodes = []
    names = {n.name for n in g.node}
    init_names = {i.name for i in g.initializer}

    def uniq(base: str) -> str:
        i = 0
        while f"{base}_{i}" in names or f"{base}_{i}" in init_names:
            i += 1
        names.add(f"{base}_{i}")
        return f"{base}_{i}"

    for node in g.node:
        if node.op_type != "HardSigmoid":
            new_nodes.append(node)
            continue
        alpha, beta = 0.2, 0.5
        for a in node.attribute:
            if a.name == "alpha":
                alpha = a.f
            elif a.name == "beta":
                beta = a.f
        x, y = node.input[0], node.output[0]
        a_name, b_name = uniq(f"{node.name}_alpha"), uniq(f"{node.name}_beta")
        z_name, o_name = uniq(f"{node.name}_zero"), uniq(f"{node.name}_one")
        for name, val in [
            (a_name, alpha),
            (b_name, beta),
            (z_name, 0.0),
            (o_name, 1.0),
        ]:
            g.initializer.append(
                numpy_helper.from_array(np.array(val, dtype=np.float32), name=name)
            )
            init_names.add(name)
        mul_out, add_out = uniq(f"{node.name}_mul"), uniq(f"{node.name}_add")
        new_nodes.append(
            helper.make_node("Mul", [x, a_name], [mul_out], name=uniq(f"{node.name}_Mul"))
        )
        new_nodes.append(
            helper.make_node(
                "Add", [mul_out, b_name], [add_out], name=uniq(f"{node.name}_Add")
            )
        )
        new_nodes.append(
            helper.make_node(
                "Clip", [add_out, z_name, o_name], [y], name=uniq(f"{node.name}_Clip")
            )
        )
    del g.node[:]
    g.node.extend(new_nodes)
    return model


def export_safetensors(model: onnx.ModelProto, out_path: Path) -> int:
    tensors = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        if arr.dtype not in (np.float32, np.float16, np.int64, np.int32):
            continue
        key = init.name.replace("/", ".")
        tensors[key] = np.ascontiguousarray(arr)
    save_file(tensors, str(out_path))
    return len(tensors)


def rewrite_same_upper_spatial(
    model: onnx.ModelProto, op_types: set[str]
) -> onnx.ModelProto:
    """Replace Conv/MaxPool auto_pad=SAME_* with Pad + valid op.

    `rlx-onnx-import`'s `onnx_pads` ignores `auto_pad`, so SAME_UPPER 2×2
    stride-1 convs run with pad=0 and shrink the map (48→46). MLX Concat then
    fails; CPU/Metal can appear to work with stale declared shapes. Explicit
    end-pad + valid op matches ORT and keeps all backends aligned.
    """
    g = model.graph
    new_nodes = []
    names = {n.name for n in g.node}
    init_names = {i.name for i in g.initializer}

    def uniq(base: str) -> str:
        i = 0
        while f"{base}_{i}" in names or f"{base}_{i}" in init_names:
            i += 1
        names.add(f"{base}_{i}")
        return f"{base}_{i}"

    for node in g.node:
        if node.op_type not in op_types:
            new_nodes.append(node)
            continue
        auto_pad = b"NOTSET"
        k = [1, 1]
        s = [1, 1]
        d = [1, 1]
        other_attrs = []
        for a in node.attribute:
            if a.name == "auto_pad":
                auto_pad = a.s
            elif a.name == "kernel_shape":
                k = list(a.ints)
            elif a.name == "strides":
                s = list(a.ints)
            elif a.name == "dilations":
                d = list(a.ints)
            elif a.name == "pads":
                continue  # replace
            else:
                other_attrs.append(a)
        if auto_pad not in (b"SAME_UPPER", b"SAME_LOWER"):
            new_nodes.append(node)
            continue
        # Spatial pad for 'same' output with stride 1: total = dil*(k-1)
        # (general SAME formula with out≈in). For k=2,s=1,d=1 → total=1.
        def axis_pads(ki: int, si: int, di: int) -> tuple[int, int]:
            # ONNX SAME: pad_total = (out-1)*stride + (k-1)*dil + 1 - in
            # with out = ceil(in/stride). For s=1: pad_total = (k-1)*dil.
            if si == 1:
                total = di * (ki - 1)
            else:
                # conservative: enough to keep size for typical even dims
                total = max(0, di * (ki - 1) - (si - 1))
            if auto_pad == b"SAME_UPPER":
                begin, end = total // 2, total - total // 2
            else:
                end, begin = total // 2, total - total // 2
            return begin, end

        hb, he = axis_pads(k[0], s[0], d[0])
        wb, we = axis_pads(k[1] if len(k) > 1 else k[0], s[1] if len(s) > 1 else s[0], d[1] if len(d) > 1 else d[0])
        if hb == 0 and he == 0 and wb == 0 and we == 0:
            nn = helper.make_node(
                node.op_type,
                list(node.input),
                list(node.output),
                name=node.name,
                kernel_shape=k,
                strides=s,
                pads=[0, 0, 0, 0],
            )
            if any(x != 1 for x in d):
                nn.attribute.append(helper.make_attribute("dilations", d))
            for a in other_attrs:
                nn.attribute.append(a)
            new_nodes.append(nn)
            continue

        # ONNX Pad pads: [n0,c0,h0,w0, n1,c1,h1,w1]
        pads = [0, 0, hb, wb, 0, 0, he, we]
        pads_name = uniq(f"{node.name}_same_pads")
        g.initializer.append(
            numpy_helper.from_array(np.array(pads, dtype=np.int64), name=pads_name)
        )
        init_names.add(pads_name)
        pad_out = uniq(f"{node.name}_same_padded")
        new_nodes.append(
            helper.make_node(
                "Pad",
                [node.input[0], pads_name],
                [pad_out],
                name=uniq(f"{node.name}_SamePad"),
                mode="constant",
            )
        )
        conv_inputs = [pad_out] + list(node.input)[1:]
        nn = helper.make_node(
            node.op_type,
            conv_inputs,
            list(node.output),
            name=uniq(f"{node.name}_valid"),
            kernel_shape=k,
            strides=s,
            pads=[0, 0, 0, 0],
        )
        if any(x != 1 for x in d):
            nn.attribute.append(helper.make_attribute("dilations", d))
        for a in other_attrs:
            nn.attribute.append(a)
        new_nodes.append(nn)
    del g.node[:]
    g.node.extend(new_nodes)
    return model


def rewrite_for_rlx(model: onnx.ModelProto) -> onnx.ModelProto:
    model = rewrite_hardsigmoid(model)
    # Conv + MaxPool SAME_* → Pad + valid (fixes MLX Concat + CPU MaxPool OOB).
    model = rewrite_same_upper_spatial(model, {"Conv", "MaxPool"})
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("onnx", type=Path, help="source inference.onnx")
    ap.add_argument(
        "out_dir",
        type=Path,
        help="directory for inference_rlx.onnx + *.safetensors",
    )
    ap.add_argument(
        "--stem",
        default=None,
        help="safetensors stem (default: derived from parent dirs)",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(args.onnx))
    model = rewrite_for_rlx(model)
    rlx_path = args.out_dir / "inference_rlx.onnx"
    onnx.save(model, str(rlx_path))
    stem = args.stem or "ppocrv6_weights"
    st_path = args.out_dir / f"{stem}.safetensors"
    n = export_safetensors(model, st_path)
    # Runtime native loaders expect `model.safetensors` beside the named export.
    model_st = args.out_dir / "model.safetensors"
    shutil.copy2(st_path, model_st)
    print(f"wrote {rlx_path}")
    print(f"wrote {st_path} and {model_st} ({n} tensors)")


if __name__ == "__main__":
    main()
