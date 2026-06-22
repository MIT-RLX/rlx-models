#!/usr/bin/env python3
"""Assemble the RLX asset bundle for TinyTTS (https://github.com/tronghieuit/tiny-tts).

TinyTTS is a MeloTTS / VITS2-style English model exported as four ONNX subgraphs
(text_encoder, duration_predictor, flow, decoder) plus a NumPy "glue" stage for
monotonic alignment + latent sampling. RLX imports each ONNX graph into the
rlx-ir HIR and runs it on any backend; the glue stage is reimplemented in Rust.

This script:
  1. Rewrites unsupported `Split` nodes into equivalent `Slice` nodes so the
     rlx-onnx-import op registry can lower every graph.
  2. Copies the four graphs into the bundle's `onnx/` dir.
  3. Reuses the (byte-identical MeloTTS) English text frontend assets — the
     symbol table, CMUdict, g2p_en checkpoint, POS tagger and BERT tokenizer —
     from an existing inflect-nano bundle, since both models share the exact
     same `tiny_tts/text/symbols.py` inventory (219 symbols, EN tone_start=7).
  4. Writes `config.json`.

Usage:
  python scripts/export_tiny_tts.py \
      --onnx /tmp/tiny-tts/onnx \
      --frontend weights/inflect-nano-rlx/frontend \
      --out weights/tiny-tts-rlx
"""
import argparse
import json
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def constant_to_initializer(model: onnx.ModelProto) -> int:
    """Promote every `Constant` node to a graph initializer.

    rlx-onnx-import resolves initializers but not all standalone `Constant`
    nodes (e.g. the scalar divisor feeding `decoder`'s `x / num_kernels`). This
    is the canonical first step of constant folding and introduces no op fusion.
    """
    graph = model.graph
    keep, n = [], 0
    for node in graph.node:
        if node.op_type != "Constant":
            keep.append(node)
            continue
        out = node.output[0]
        tensor = None
        for attr in node.attribute:
            if attr.name == "value":
                tensor = attr.t
            elif attr.name == "value_float":
                tensor = numpy_helper.from_array(np.array(attr.f, dtype=np.float32))
            elif attr.name == "value_floats":
                tensor = numpy_helper.from_array(np.array(attr.floats, dtype=np.float32))
            elif attr.name == "value_int":
                tensor = numpy_helper.from_array(np.array(attr.i, dtype=np.int64))
            elif attr.name == "value_ints":
                tensor = numpy_helper.from_array(np.array(attr.ints, dtype=np.int64))
        if tensor is None:
            # Unhandled Constant variant — leave the node in place.
            keep.append(node)
            continue
        init = onnx.TensorProto()
        init.CopyFrom(tensor)
        init.name = out
        graph.initializer.append(init)
        n += 1
    del graph.node[:]
    graph.node.extend(keep)
    return n


def _resolve_split_sizes(node, axis_dim, n_out, initializers):
    """Return the list of split sizes for a Split node."""
    # opset >= 13: optional second input holds the split tensor.
    if len(node.input) > 1 and node.input[1]:
        init = initializers.get(node.input[1])
        if init is not None:
            return [int(v) for v in numpy_helper.to_array(init).tolist()]
    # opset < 13: 'split' attribute.
    for attr in node.attribute:
        if attr.name == "split":
            return [int(v) for v in attr.ints]
    # Otherwise: even split across the outputs.
    if axis_dim is None:
        raise ValueError(f"cannot infer split sizes for {node.name}: unknown axis dim")
    if axis_dim % n_out != 0:
        raise ValueError(f"uneven default split for {node.name}: {axis_dim}/{n_out}")
    return [axis_dim // n_out] * n_out


def decompose_split(model: onnx.ModelProto):
    """Replace every `Split` node with a set of `Slice` nodes.

    Returns (model, count). Shape inference is run on a copy to resolve even-split
    axis sizes; the rewritten nodes are applied to that same copy.
    """
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    # value_info shape lookup (axis dim resolution for even splits).
    shapes = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.HasField("dim_value") else None)
        shapes[vi.name] = dims
    initializers = {init.name: init for init in graph.initializer}

    new_nodes = []
    added_inits = []
    n_split = 0
    for node in graph.node:
        if node.op_type != "Split":
            new_nodes.append(node)
            continue
        n_split += 1
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
        data = node.input[0]
        in_dims = shapes.get(data)
        rank = len(in_dims) if in_dims else None
        norm_axis = axis if axis >= 0 else (axis + rank if rank else axis)
        axis_dim = in_dims[norm_axis] if (in_dims and rank) else None
        sizes = _resolve_split_sizes(node, axis_dim, len(node.output), initializers)

        offset = 0
        for i, out in enumerate(node.output):
            if not out:
                offset += sizes[i]
                continue
            pfx = f"{node.name}_slice{i}" if node.name else f"{out}_slice"
            starts = numpy_helper.from_array(
                np.array([offset], dtype=np.int64), pfx + "_starts")
            ends = numpy_helper.from_array(
                np.array([offset + sizes[i]], dtype=np.int64), pfx + "_ends")
            axes = numpy_helper.from_array(
                np.array([norm_axis], dtype=np.int64), pfx + "_axes")
            steps = numpy_helper.from_array(
                np.array([1], dtype=np.int64), pfx + "_steps")
            added_inits += [starts, ends, axes, steps]
            new_nodes.append(helper.make_node(
                "Slice",
                inputs=[data, starts.name, ends.name, axes.name, steps.name],
                outputs=[out],
                name=pfx,
            ))
            offset += sizes[i]

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(added_inits)
    return model, n_split


def pin_batch_dim(model: onnx.ModelProto):
    """Pin the leading (batch) axis of every graph input/output to a literal 1.

    rlx-onnx-import maps *any* symbolic dim string (`B`, `T`, …) to its single
    `sequence_length` knob. That is exactly what we want for the length axis, but
    the batch axis must stay 1 — so we make it a concrete `dim_value=1`. The
    remaining symbolic length axis is left as-is (→ bound to sequence_length).
    """
    for vi in list(model.graph.input) + list(model.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if len(dims) >= 1:
            dims[0].ClearField("dim_param")
            dims[0].dim_value = 1
    return model


def process_onnx(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = {}
    for name in ["text_encoder", "duration_predictor", "flow", "decoder"]:
        src = os.path.join(src_dir, name + ".onnx")
        model = onnx.load(src)
        model = pin_batch_dim(model)
        n_const = constant_to_initializer(model)
        model, n = decompose_split(model)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        # The decoder upsamples length 512× through 5 ConvTranspose layers. ONNX
        # leaves those lengths symbolic (`unk__*`), and rlx-onnx-import prefers a
        # node's declared value_info over its own conv formula — collapsing the
        # length to `sequence_length`. Strip internal value_info so the importer
        # recomputes the (correct) upsampled lengths. The other three graphs keep
        # length T throughout, where the value_info heuristic is already correct
        # (and they use Range/Equal/ConstantOfShape that infer_output can't carry).
        stripped = ""
        if name == "decoder":
            del model.graph.value_info[:]
            stripped = " [value_info stripped]"
        dst = os.path.join(out_dir, name + ".onnx")
        onnx.save(model, dst)
        print(f"  {name}: {n_const} Const->init, {n} Split->Slice"
              f"  ({os.path.getsize(dst)} bytes){stripped}")
        meta[name] = n
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="dir with the 4 TinyTTS .onnx files")
    ap.add_argument("--frontend", required=True,
                    help="existing MeloTTS frontend asset dir to reuse")
    ap.add_argument("--out", required=True, help="output bundle dir")
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)

    print("[1/3] processing ONNX graphs (Split -> Slice)…")
    process_onnx(args.onnx, os.path.join(out, "onnx"))

    print("[2/3] copying frontend assets…")
    fe_out = os.path.join(out, "frontend")
    if os.path.abspath(fe_out) != os.path.abspath(args.frontend):
        if os.path.exists(fe_out):
            shutil.rmtree(fe_out)
        shutil.copytree(args.frontend, fe_out)
    # Sanity: the symbol table must be the MeloTTS 219-symbol inventory.
    sym = json.load(open(os.path.join(fe_out, "symbols.json")))
    assert sym["language_id_map"]["EN"] == 2, "frontend symbols.json is not MeloTTS-EN"
    assert sym["language_tone_start_map"]["EN"] == 7
    print(f"      symbols ok: {len(sym['symbols'])} symbols, EN tone_start=7")

    print("[3/3] writing config.json…")
    config = {
        "model": "TinyTTS",
        "source": "https://github.com/tronghieuit/tiny-tts",
        "sample_rate": 44100,
        "add_blank": True,
        "language": "EN",
        "speakers": {"MALE": 0},
        "default_speaker": "MALE",
        # VITS sampling controls (match tiny_tts/infer_onnx.py defaults).
        "noise_scale": 0.667,
        "noise_scale_w": 0.8,
        "length_scale": 1.0,
        # ONNX channel widths (read so the Rust glue allocates z_p correctly).
        "inter_channels": 80,
        "gin_channels": 80,
    }
    json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
