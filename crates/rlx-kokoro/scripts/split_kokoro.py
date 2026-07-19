#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Split the monolithic Kokoro-82M ONNX graph into the two fixed-shape subgraphs
# the native RLX runner ([`rlx_kokoro::native::NativeKokoro`]) consumes, plus the
# ISTFT `window_sum`. The data-dependent length regulator and the ISTFT
# overlap-add normalization are done in Rust — this decomposition is verified
# bit-exact (cosine 1.0, max_abs 0) against the monolithic model.
#
#   encoder.onnx      [input_ids, style, speed]
#                   → /encoder/Transpose_output_0                 prosody [1,640,seq]
#                   → /encoder/text_encoder/Transpose_2_output_0  text    [1,512,seq]
#                   → /encoder/Cast_output_0                      dur     [1,seq] i64
#   decoder_raw.onnx  [/encoder/MatMul_output_0, /encoder/MatMul_1_output_0, style]
#                   → /decoder/decoder/generator/istft/stft/Squeeze_1_output_0  (raw, pre-ISTFT-norm)
#   window_sum.f32    ISTFT overlap-add normalization window (len n_fft=20)
#
# Usage:
#   python split_kokoro.py <model.onnx> <out_dir>
# e.g. python split_kokoro.py weights/tts/kokoro-82m/onnx/model.onnx \
#          weights/tts/kokoro-82m/onnx/rlx-split
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper as nh
from onnx.utils import Extractor

# Graph-split boundary tensors (stable across the onnx-community Kokoro exports).
PROSODY = "/encoder/Transpose_output_0"
TEXT = "/encoder/text_encoder/Transpose_2_output_0"
PDUR = "/encoder/Cast_output_0"
EN = "/encoder/MatMul_output_0"
ASR = "/encoder/MatMul_1_output_0"
# Cut the decoder BEFORE the ISTFT NonZero/ScatterND overlap-add normalization
# (which has no static-shape RLX lowering); the Rust side finishes it.
RAW = "/decoder/decoder/generator/istft/stft/Squeeze_1_output_0"
WINDOW_SUM_SUFFIX = "istft.stft.window_sum"


def extract(model, inputs, outputs, dst):
    # `Extractor` (not `extract_model`) with no post-check: the raw-decoder cut
    # leaves a graph onnx.checker rejects (dangling ISTFT scatter tail), but it
    # imports + runs fine.
    ex = Extractor(model)
    sub = ex.extract_model(inputs, outputs)
    onnx.save(sub, dst)


def main(src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(src)

    extract(model, ["input_ids", "style", "speed"], [PROSODY, TEXT, PDUR],
            os.path.join(out_dir, "encoder.onnx"))
    print("wrote encoder.onnx")

    extract(model, [EN, ASR, "style"], [RAW],
            os.path.join(out_dir, "decoder_raw.onnx"))
    print("wrote decoder_raw.onnx")

    wsum = None
    for init in model.graph.initializer:
        if init.name.endswith(WINDOW_SUM_SUFFIX):
            wsum = nh.to_array(init).astype(np.float32).ravel()
            break
    if wsum is None:
        raise SystemExit(f"window_sum initializer (*{WINDOW_SUM_SUFFIX}) not found")
    wsum.tofile(os.path.join(out_dir, "window_sum.f32"))
    print(f"wrote window_sum.f32 (len {wsum.size})")
    print(f"native Kokoro bundle ready: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: split_kokoro.py <model.onnx> <out_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
