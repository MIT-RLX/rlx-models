#!/usr/bin/env python3
"""Generate a tiny ONNX LSTM + an onnxruntime reference, for rlx import parity.

Writes into <outdir>:
  lstm.onnx   forward LSTM, static shapes
  x.f32       input X   [seq, batch, input]  raw little-endian f32
  y_ref.f32   ORT Y out [seq, num_dir, batch, hidden] raw f32
  meta.json   shapes
Run: python3 gen_lstm.py <outdir> [--bidir]
"""
import json
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

SEQ, BATCH, INPUT, HIDDEN = 5, 1, 3, 4

def build(outdir, bidir):
    dirs = 2 if bidir else 1
    direction = "bidirectional" if bidir else "forward"
    rng = np.random.default_rng(0)
    W = rng.standard_normal((dirs, 4 * HIDDEN, INPUT)).astype(np.float32) * 0.5
    R = rng.standard_normal((dirs, 4 * HIDDEN, HIDDEN)).astype(np.float32) * 0.5
    B = rng.standard_normal((dirs, 8 * HIDDEN)).astype(np.float32) * 0.1
    X = rng.standard_normal((SEQ, BATCH, INPUT)).astype(np.float32)

    x_in = helper.make_tensor_value_info("X", TensorProto.FLOAT, [SEQ, BATCH, INPUT])
    y_out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [SEQ, dirs, BATCH, HIDDEN])
    inits = [
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(R, "R"),
        numpy_helper.from_array(B, "Bias"),
    ]
    node = helper.make_node(
        "LSTM", ["X", "W", "R", "Bias"], ["Y", "Yh", "Yc"],
        hidden_size=HIDDEN, direction=direction,
    )
    yh = helper.make_tensor_value_info("Yh", TensorProto.FLOAT, [dirs, BATCH, HIDDEN])
    yc = helper.make_tensor_value_info("Yc", TensorProto.FLOAT, [dirs, BATCH, HIDDEN])
    graph = helper.make_graph([node], "lstm_test", [x_in], [y_out, yh, yc], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, f"{outdir}/lstm.onnx")

    sess = ort.InferenceSession(f"{outdir}/lstm.onnx", providers=["CPUExecutionProvider"])
    y_ref = sess.run(["Y"], {"X": X})[0].astype(np.float32)

    X.tofile(f"{outdir}/x.f32")
    y_ref.tofile(f"{outdir}/y_ref.f32")
    json.dump(
        {"seq": SEQ, "batch": BATCH, "input": INPUT, "hidden": HIDDEN,
         "dirs": dirs, "direction": direction, "y_shape": list(y_ref.shape)},
        open(f"{outdir}/meta.json", "w"), indent=2,
    )
    print(f"wrote {outdir}: X{list(X.shape)} Y{list(y_ref.shape)} dir={direction}")
    print("y_ref[0,:, 0, :] =", y_ref[0].reshape(dirs, HIDDEN))

if __name__ == "__main__":
    outdir = sys.argv[1]
    bidir = "--bidir" in sys.argv
    build(outdir, bidir)
