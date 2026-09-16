#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.
#
# Fit linear body map B: enc ≈ h @ B from Apple capture_work speech dumps.
# Writes body_map_ls.B.bin under RLX_ASR_DIR (or --out).
#
# Usage:
#   python3 fit_body_map_ls.py \
#     --capture-work ".../asr cache/capture_work" \
#     --out /Users/Shared/translator/models/rlx-asr/body_map_ls.B.bin
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from e2e_native_whole import load_native_pack
from audio_io import asr_dir

DIM = 512
OUT_T = 64
SUB = 6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--min-speech-frames", type=int, default=5)
    args = ap.parse_args()

    pack = load_native_pack()
    W = pack["frontend.input_proj_eff.W"].astype(np.float32)
    if W.shape == (DIM, 80):
        W = W.T
    b = pack.get("frontend.input_proj_eff.b")

    xs, ys = [], []
    for d in sorted(args.capture_work.iterdir()):
        if not d.is_dir() or d.name.startswith("synth_"):
            continue
        fp, ep, wp = d / "e5_feature.bin", d / "e5_encoder_cache.bin", d / "e5_wp_prob.bin"
        if not all(p.is_file() for p in (fp, ep, wp)):
            continue
        feat = np.fromfile(fp, dtype=np.float32)
        if feat.size % 80:
            continue
        T = feat.size // 80
        if T < OUT_T * SUB:
            continue
        L = np.fromfile(wp, dtype=np.float32).reshape(-1, 6081)[:OUT_T]
        if (L.argmax(1) != 0).sum() < args.min_speech_frames:
            continue
        feat = feat.reshape(T, 80)
        enc = np.fromfile(ep, dtype=np.float32).reshape(-1, DIM)[:OUT_T]
        h = feat[np.arange(OUT_T) * SUB] @ W
        if b is not None:
            h = h + b.astype(np.float32)
        xs.append(h.astype(np.float32))
        ys.append(enc.astype(np.float32))

    if not xs:
        raise SystemExit(f"no speech captures under {args.capture_work}")

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    B = np.linalg.solve(X.T @ X + args.lam * np.eye(DIM), X.T @ Y).astype(np.float32)
    out = args.out or (asr_dir() / "body_map_ls.B.bin")
    out.parent.mkdir(parents=True, exist_ok=True)
    B.astype("<f4").tofile(out)

    step = max(1, len(X) // 500)
    corr = float(
        np.mean(
            [np.corrcoef((X @ B)[i], Y[i])[0, 1] for i in range(0, len(X), step)]
        )
    )
    print(f"wrote {out} frames={len(X)} train_frame_corr={corr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
