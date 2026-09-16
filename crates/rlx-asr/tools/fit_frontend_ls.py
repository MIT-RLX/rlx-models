#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.
#
# Fit per-bin affine frontend LS: our mel → Apple e5_feature.
# Writes frontend_fbank_ls_cross_wav_{a,b}.bin under --out-dir / RLX_ASR_DIR.
#
# Usage:
#   python3 fit_frontend_ls.py \
#     --pair ask_not.wav:.../ask_not_00/e5_feature.bin \
#     --pair moon.wav:.../moon_00/e5_feature.bin
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from audio_io import asr_dir, mel_from_wav


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pair",
        action="append",
        required=True,
        help="wav:e5_feature.bin (repeatable)",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=389)
    args = ap.parse_args()

    xs, ys = [], []
    for spec in args.pair:
        wav_s, feat_s = spec.split(":", 1)
        wav, feat_p = Path(wav_s), Path(feat_s)
        our, _ = mel_from_wav(wav)
        tf = np.fromfile(feat_p, dtype=np.float32)
        if tf.size % 80:
            raise SystemExit(f"bad feature size {feat_p}")
        tf = tf.reshape(-1, 80)
        n = min(len(our), len(tf), args.max_frames)
        xs.append(our[:n])
        ys.append(tf[:n])
        print(
            f"{wav.name}: n={n} rawcorr={np.corrcoef(our[:n].ravel(), tf[:n].ravel())[0,1]:.3f}"
        )

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    a = np.zeros(80, np.float32)
    b = np.zeros(80, np.float32)
    for i in range(80):
        x = X[:, i]
        y = Y[:, i]
        vx = float(np.var(x)) + 1e-8
        a[i] = float(np.cov(x, y, bias=True)[0, 1] / vx)
        b[i] = float(y.mean() - a[i] * x.mean())
    cal = X * a + b
    corr = float(np.corrcoef(cal.ravel(), Y.ravel())[0, 1])
    out = args.out_dir or asr_dir()
    out.mkdir(parents=True, exist_ok=True)
    a.astype("<f4").tofile(out / "frontend_fbank_ls_cross_wav_a.bin")
    b.astype("<f4").tofile(out / "frontend_fbank_ls_cross_wav_b.bin")
    print(f"wrote {out} frames={len(X)} corr={corr:.3f} a=[{a.min():.3f},{a.max():.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
