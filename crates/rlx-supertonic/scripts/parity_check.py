#!/usr/bin/env python3
"""Numerical parity check: rlx-supertonic vs Python onnxruntime, identical inputs.

rlx-supertonic calls the same onnxruntime C++ library the reference uses, so with
IDENTICAL inputs (tokens + style + the sampled noise) the output must be
bit-identical. This confirms it end-to-end.

  1) dump rlx's exact ONNX inputs + output:
       RLX_ST_PARITY_DUMP=/tmp/stp cargo run -p rlx-supertonic --bin rlx-supertonic -- \
           --data weights/tts/supertonic-3 --text "Numerical parity check." --voice F1 --out /tmp/x.wav
  2) compare against onnxruntime fed those same inputs:
       python parity_check.py /tmp/stp weights/tts/supertonic-3/onnx

Result on reference run: cosine=0.99999994, max_abs=0.0 (bit-exact).
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort


def main(pdir: str, onnx_dir: str) -> None:
    m = json.load(open(f"{pdir}/meta.json"))
    t, l, ch, total = m["t"], m["l"], m["ch"], m["total"]
    ids = np.fromfile(f"{pdir}/ids.i64", dtype=np.int64).reshape(1, t)
    sttl = np.fromfile(f"{pdir}/style_ttl.f32", dtype=np.float32).reshape(1, m["ttl_rows"], m["ttl_cols"])
    noise = np.fromfile(f"{pdir}/noise.f32", dtype=np.float32).reshape(1, ch, l)
    rlx_audio = np.fromfile(f"{pdir}/audio_rlx.f32", dtype=np.float32)
    tm = np.ones((1, 1, t), np.float32)
    lm = np.ones((1, 1, l), np.float32)

    def sess(n):
        return ort.InferenceSession(os.path.join(onnx_dir, n), providers=["CPUExecutionProvider"])

    te, ve, vo = sess("text_encoder.onnx"), sess("vector_estimator.onnx"), sess("vocoder.onnx")
    text_emb = te.run(None, {"text_ids": ids, "style_ttl": sttl, "text_mask": tm})[0]
    xt = noise.copy()
    for step in range(total):
        xt = ve.run(None, {
            "noisy_latent": xt, "text_emb": text_emb, "style_ttl": sttl,
            "latent_mask": lm, "text_mask": tm,
            "current_step": np.array([step], np.float32),
            "total_step": np.array([total], np.float32),
        })[0]
    wav = vo.run(None, {"latent": xt})[0].reshape(-1)
    n = min(len(rlx_audio), len(wav))
    a, b = rlx_audio[:n], wav[:n]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"cosine={cos:.8f}  max_abs={np.max(np.abs(a - b)):.3e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
