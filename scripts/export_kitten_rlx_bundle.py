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

"""Export a KittenTTS ONNX model to an RLX compile bundle (weights + graph IR).

The bundle replaces ONNX Runtime at inference time: Rust loads graph.json +
weights.safetensors, builds an RLX HIR module, and compiles with Session::compile.

Usage:
  python3 scripts/export_kitten_rlx_bundle.py \\
    /path/to/kitten_tts_nano_v0_8.onnx \\
    /path/to/out/kitten-rlx-bundle

Requires: pip install onnx onnxshape numpy safetensors

Rewrites applied at export:
  - MatMulInteger → f32 MatMul (when possible)
  - duration feedback: `duration` → `__onnx_import__/duration_carry` on `/Expand_1`, `/Where_1`
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnx
    from onnx import TensorProto, numpy_helper, shape_inference
except ImportError:
    print("install: pip install onnx", file=sys.stderr)
    raise


def _tensor_dtype(elem_type: int) -> str:
    m = {
        TensorProto.FLOAT: "f32",
        TensorProto.INT64: "i64",
        TensorProto.INT32: "i32",
        TensorProto.BOOL: "bool",
    }
    return m.get(elem_type, f"type_{elem_type}")


def _shape_from_value_info(vi) -> list[Any]:
    if vi.type.HasField("tensor_type"):
        out = []
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_value:
                out.append(int(d.dim_value))
            elif d.dim_param:
                out.append(d.dim_param)
            else:
                out.append("?")
        return out
    return []


def _rewrite_quantized_matmul(m: onnx.ModelProto) -> onnx.ModelProto:
    """Fold MatMulInteger(+static weight) + DynamicQuantizeLinear activations into MatMul."""
    graph = m.graph
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    producers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for o in node.output:
            producers[o] = node

    def trace_pre_quant(name: str) -> str | None:
        """Name of the f32 tensor feeding DynamicQuantizeLinear (activation side)."""
        if name in init:
            return None
        n = producers.get(name)
        if n is None:
            return name
        if n.op_type in ("DynamicQuantizeLinear", "Cast"):
            return trace_pre_quant(n.input[0])
        return name

    new_nodes: list[onnx.NodeProto] = []
    removed: set[str] = set()
    replaced: dict[str, str] = {}

    for node in graph.node:
        if node.op_type != "MatMulInteger":
            new_nodes.append(node)
            continue
        act_q, w_q, act_zp, w_zp = node.input[:4]
        if w_q not in init:
            new_nodes.append(node)
            continue
        act_f32 = trace_pre_quant(act_q)
        if act_f32 is None:
            new_nodes.append(node)
            continue
        w = init[w_q].astype(np.float32)
        # weight scale from paired DynamicQuantize on initializer path is 1/s for int8
        w_scale_name = w_q.replace("_quantized", "") + "_scale"
        w_zp_name = w_q.replace("_quantized", "") + "_zero_point"
        scale_arr = init.get(w_scale_name)
        zp_arr = init.get(w_zp_name)
        if scale_arr is not None:
            s = float(np.asarray(scale_arr).reshape(-1)[0])
            z = int(np.asarray(zp_arr).reshape(-1)[0]) if zp_arr is not None else 0
            w = (w.astype(np.float32) - z) * s
        w_name = node.name + "_f32_weight"
        init[w_name] = w
        mm = onnx.helper.make_node(
            "MatMul",
            [act_f32, w_name],
            list(node.output),
            name=node.name + "_f32",
        )
        new_nodes.append(mm)
        removed.add(node.name)
        for o in node.output:
            replaced[o] = o

    # Drop DynamicQuantizeLinear nodes whose outputs are no longer consumed.
    consumers: dict[str, int] = defaultdict(int)
    for node in new_nodes:
        for i in node.input:
            consumers[i] += 1

    final_nodes = []
    for node in new_nodes:
        if node.op_type == "DynamicQuantizeLinear":
            if all(o not in consumers or consumers[o] == 0 for o in node.output):
                continue
        final_nodes.append(node)

    del graph.node[:]
    graph.node.extend(final_nodes)
    # Re-add any new initializers
    existing = {t.name for t in graph.initializer}
    for name, arr in init.items():
        if name.endswith("_f32_weight") and name not in existing:
            graph.initializer.append(numpy_helper.from_array(arr, name=name))
    return m


DURATION_CARRY = "__onnx_import__/duration_carry"


def _rewrite_duration_carry(nodes: list[dict[str, Any]]) -> None:
    """Break the duration feedback cycle for ORT single-pass import semantics."""
    for node in nodes:
        if node.get("name") in ("/Expand_1", "/Where_1"):
            node["inputs"] = [
                DURATION_CARRY if inp == "duration" else inp for inp in node["inputs"]
            ]


def _export_subgraph(g: onnx.GraphProto, vi_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Serialize a nested ONNX graph (e.g. Loop body) for runtime lowering."""
    nodes_out: list[dict[str, Any]] = []
    for node in g.node:
        attrs: dict[str, Any] = {}
        for a in node.attribute:
            if a.type == onnx.AttributeProto.INT:
                attrs[a.name] = int(a.i)
            elif a.type == onnx.AttributeProto.INTS:
                attrs[a.name] = [int(x) for x in a.ints]
            elif a.type == onnx.AttributeProto.FLOAT:
                attrs[a.name] = float(a.f)
            elif a.type == onnx.AttributeProto.STRING:
                attrs[a.name] = a.s.decode() if isinstance(a.s, bytes) else a.s
            elif a.type == onnx.AttributeProto.TENSOR:
                attrs[a.name] = numpy_helper.to_array(a.t).tolist()
            elif a.type == onnx.AttributeProto.GRAPH:
                attrs[a.name] = _export_subgraph(a.g, vi_map)
        out_shapes = [vi_map.get(o, {}) for o in node.output]
        nodes_out.append(
            {
                "name": node.name or node.op_type,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attrs": attrs,
                "output_meta": out_shapes,
            }
        )
    inits = {
        t.name: numpy_helper.to_array(t).tolist()
        for t in g.initializer
    }
    return {
        "inputs": [i.name for i in g.input],
        "outputs": [o.name for o in g.output],
        "initializers": inits,
        "nodes": nodes_out,
    }


def export_bundle(onnx_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(onnx_path))
    ops_before = Counter(n.op_type for n in m.graph.node)
    if ops_before["MatMulInteger"]:
        m = _rewrite_quantized_matmul(m)
    m = shape_inference.infer_shapes(m)

    graph = m.graph
    init_arrays = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    # value name -> shape/dtype from value_info
    vi_map: dict[str, dict[str, Any]] = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        vi_map[vi.name] = {
            "shape": _shape_from_value_info(vi),
            "dtype": _tensor_dtype(vi.type.tensor_type.elem_type)
            if vi.type.HasField("tensor_type")
            else "unknown",
        }

    nodes_out: list[dict[str, Any]] = []
    for node in graph.node:
        attrs: dict[str, Any] = {}
        for a in node.attribute:
            if a.type == onnx.AttributeProto.INT:
                attrs[a.name] = int(a.i)
            elif a.type == onnx.AttributeProto.INTS:
                attrs[a.name] = [int(x) for x in a.ints]
            elif a.type == onnx.AttributeProto.FLOAT:
                attrs[a.name] = float(a.f)
            elif a.type == onnx.AttributeProto.FLOATS:
                attrs[a.name] = [float(x) for x in a.floats]
            elif a.type == onnx.AttributeProto.STRING:
                attrs[a.name] = a.s.decode() if isinstance(a.s, bytes) else a.s
            elif a.type == onnx.AttributeProto.TENSOR:
                attrs[a.name] = numpy_helper.to_array(a.t).tolist()
            elif a.type == onnx.AttributeProto.GRAPH:
                attrs[a.name] = _export_subgraph(a.g, vi_map)
        out_shapes = [vi_map.get(o, {}) for o in node.output]
        nodes_out.append(
            {
                "name": node.name or node.op_type,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attrs": attrs,
                "output_meta": out_shapes,
            }
        )

    manifest = {
        "source_onnx": str(onnx_path),
        "opset": [(o.domain, o.version) for o in m.opset_import],
        "inputs": [
            {"name": i.name, **vi_map.get(i.name, {})} for i in graph.input
        ],
        "outputs": [
            {"name": o.name, **vi_map.get(o.name, {})} for o in graph.output
        ],
        "node_count": len(nodes_out),
        "initializer_count": len(init_arrays),
        "op_histogram": dict(Counter(n["op"] for n in nodes_out)),
    }

    _rewrite_duration_carry(nodes_out)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "graph.json").write_text(json.dumps(nodes_out))

    # safetensors
    try:
        from safetensors.numpy import save_file
    except ImportError:
        print("install: pip install safetensors", file=sys.stderr)
        raise
    tensors = {k: np.ascontiguousarray(v) for k, v in init_arrays.items()}
    save_file(tensors, str(out_dir / "weights.safetensors"))

    print(f"Wrote bundle to {out_dir}")
    print(f"  nodes={len(nodes_out)} initializers={len(tensors)}")
    blockers = [
        "MatMulInteger",
        "ConvInteger",
        "DynamicQuantizeLinear",
        "DynamicQuantizeLSTM",
    ]
    hist = manifest["op_histogram"]
    for b in blockers:
        if hist.get(b):
            print(f"  WARNING: blocker op {b}: {hist[b]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("onnx_path", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    export_bundle(args.onnx_path, args.out_dir)


if __name__ == "__main__":
    main()
