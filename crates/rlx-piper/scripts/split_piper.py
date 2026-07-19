#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Split the monolithic piper VITS ONNX into two fixed-shape subgraphs for the
# native (ort-free) RLX runner, plus a raw dump of the StochasticDurationPredictor
# (dp) weights the runner needs. The dp itself is NOT exported as a graph — its
# rational-quadratic-spline coupling flow uses boolean-mask indexing
# (`inputs[inside_mask]`), a data-dependent NonZero/GatherND flatten that no
# static-shape importer can rank. Instead the dp runs in Rust (crate `sdp.rs`),
# where the spline is a trivial per-element inside/outside branch.
#
#   enc_p.onnx    [input, input_lengths]
#               → m_p    [1,192,T]  (/enc_p/Split_output_0)
#               → logs_p [1,192,T]  (/enc_p/Split_output_1)
#               → dp_in  [1,192,T]  (/enc_p/encoder/Mul_2_output_0)  ← feeds Rust dp
#     ── Rust StochasticDurationPredictor: dp_in → durations[T] ──
#     ── Rust length regulator + z_p = m_p' + randn·exp(logs_p')·noise_scale ──
#   flow_dec.onnx [z_p [1,192,T'], y_mask [1,1,T']] → waveform [1,1,1,T'·hop]
#
# Usage: python split_piper.py <model.onnx> <out_dir>
import sys, os, json
import numpy as np
import onnx
from onnx import shape_inference, numpy_helper as nh
from onnx.utils import Extractor

ENC_OUT = ["/enc_p/Split_output_0", "/enc_p/Split_output_1", "/enc_p/encoder/Mul_2_output_0"]
FLOW_IN = ["/Add_output_0", "/Cast_2_output_0"]


def rename_dims(graph, mapping):
    """Rewrite dim_param labels graph-wide so the runner can bind stable names."""
    for coll in (graph.input, graph.output, graph.value_info):
        for v in coll:
            t = v.type.tensor_type
            for d in t.shape.dim:
                if d.HasField("dim_param") and d.dim_param in mapping:
                    d.dim_param = mapping[d.dim_param]


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    m = onnx.load(src)
    # Shape-inference pass: the exported dp/spline region ships without shapes;
    # infer_shapes(data_prop=True) fills them so the extractor cuts cleanly.
    m = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    ex = Extractor(m)

    enc = ex.extract_model(["input", "input_lengths"], ENC_OUT)
    onnx.save(enc, os.path.join(out, "enc_p.onnx"))

    flow = ex.extract_model(FLOW_IN, ["output"])
    # Canonicalize the flow_dec dynamic dims: the shared frame length T' (axis-2 of
    # both inputs) → "frames"; the batch-like axis-0 dims → "batch".
    tprime = flow.graph.input[0].type.tensor_type.shape.dim[2].dim_param
    batch0 = flow.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    batch1 = flow.graph.input[1].type.tensor_type.shape.dim[0].dim_param
    rename_dims(flow.graph, {tprime: "frames", batch0: "batch", batch1: "batch"})
    onnx.save(flow, os.path.join(out, "flow_dec.onnx"))

    # Dump the dp weights as a flat little-endian f32 blob + a JSON manifest of
    # {name: [shape]} so the Rust runner can load them without an ONNX parser.
    W = {i.name: nh.to_array(i) for i in m.graph.initializer if i.name.startswith("dp.")}
    manifest = {}
    with open(os.path.join(out, "dp_weights.f32"), "wb") as f:
        off = 0
        for name in sorted(W):
            a = W[name].astype("<f4").ravel()
            f.write(a.tobytes())
            manifest[name] = {"shape": list(W[name].shape), "offset": off, "numel": int(a.size)}
            off += int(a.size)
    with open(os.path.join(out, "dp_manifest.json"), "w") as f:
        json.dump({"weights": manifest, "num_bins": 10, "tail_bound": 5.0,
                   "filter_channels": 192, "eps": 1e-5}, f, indent=2)
    print(f"wrote enc_p.onnx, flow_dec.onnx, dp_weights.f32 ({len(W)} tensors), dp_manifest.json → {out}")


if __name__ == "__main__":
    main()
