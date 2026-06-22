#!/usr/bin/env python3
"""Generate synthetic echo dataset triples for AEC residual training.

Usage:
  python3 scripts/aec_synthetic_dataset.py --out-dir /tmp/aec_dataset --n 32
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path


def synth_tone(n: int, sr: int, hz: float, amp: float) -> list[float]:
    return [amp * math.sin(2.0 * math.pi * hz * i / sr) for i in range(n)]


def write_wav(path: Path, pcm: list[float], sr: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for s in pcm:
            v = max(-1.0, min(1.0, s))
            frames.extend(struct.pack("<h", int(v * 32767)))
        wf.writeframes(frames)


def delayed_echo(clean: list[float], far: list[float], delay: int, alpha: float) -> list[float]:
    n = len(clean)
    mic = clean[:]
    for i in range(len(far)):
        t = i + delay
        if t < n:
            mic[t] += alpha * far[i]
    return mic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=32, help="number of clips")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--sr", type=int, default=16_000)
    args = ap.parse_args()

    sr = args.sr
    n = int(args.seconds * sr)
    manifest = []
    for idx in range(args.n):
        delay = 80 + (idx * 37) % 400
        alpha = 0.4 + 0.01 * (idx % 25)
        clean_hz = 200.0 + (idx % 7) * 40.0
        far_hz = 420.0 + (idx % 5) * 55.0
        clean = synth_tone(n, sr, clean_hz, 0.35)
        far = synth_tone(n, sr, far_hz, 0.45)
        mic = delayed_echo(clean, far, delay, alpha)
        stem = args.out_dir / f"clip_{idx:04d}"
        write_wav(stem.with_name(stem.name + "_clean.wav"), clean, sr)
        write_wav(stem.with_name(stem.name + "_far.wav"), far, sr)
        write_wav(stem.with_name(stem.name + "_mic.wav"), mic, sr)
        manifest.append(
            {
                "id": idx,
                "delay": delay,
                "alpha": alpha,
                "clean": str(stem.with_name(stem.name + "_clean.wav")),
                "far": str(stem.with_name(stem.name + "_far.wav")),
                "mic": str(stem.with_name(stem.name + "_mic.wav")),
            }
        )
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} clips → {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
