#!/usr/bin/env python3
"""Bench AEC with optional Speex/pyroomacoustics (Python-only baselines).

Outputs JSON with MSE improvement vs clean target.

Usage:
  python3 scripts/aec_bench_speex.py --out /tmp/aec_speex.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
import wave
from pathlib import Path


def synth(n: int, sr: int, delay: int, alpha: float) -> tuple[list[float], list[float], list[float]]:
    clean = [0.35 * math.sin(2 * math.pi * 280 * i / sr) for i in range(n)]
    far = [0.42 * math.sin(2 * math.pi * 520 * i / sr) for i in range(n)]
    mic = clean[:]
    for i in range(len(far)):
        t = i + delay
        if t < n:
            mic[t] += alpha * far[i]
    return clean, far, mic


def mse(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum((a[i] - b[i]) ** 2 for i in range(n)) / max(n, 1)


def mse_improve_db(mic: list[float], out: list[float], clean: list[float]) -> float:
    m0 = mse(mic, clean)
    m1 = mse(out, clean)
    if m0 <= 1e-12 or m1 <= 1e-12:
        return 0.0
    return 10.0 * math.log10(m0 / m1)


def nlms_cancel(mic: list[float], far: list[float], delay: int, mu: float = 0.05, filt_len: int = 512) -> list[float]:
    """Simple time-domain NLMS baseline."""
    out = mic[:]
    w = [0.0] * filt_len
    aligned = [0.0] * (len(far) + delay)
    aligned[delay : delay + len(far)] = far
    eps = 1e-8
    for i in range(len(mic)):
        x = [aligned[i - k] if i >= k else 0.0 for k in range(filt_len)]
        p = sum(v * v for v in x) + eps
        y = sum(w[k] * x[k] for k in range(filt_len))
        e = mic[i] - y
        out[i] = e
        for k in range(filt_len):
            w[k] += mu * e * x[k] / p
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--sr", type=int, default=16_000)
    args = ap.parse_args()

    n = int(args.seconds * args.sr)
    delay = 200
    alpha = 0.65
    clean, far, mic = synth(n, args.sr, delay, alpha)

    rows = []
    t0 = time.perf_counter()
    py_out = nlms_cancel(mic, far, delay)
    rows.append(
        {
            "label": "python-nlms",
            "mse_improve_db": mse_improve_db(mic, py_out, clean),
            "seconds": time.perf_counter() - t0,
        }
    )

    try:
        import pyroomacoustics as pra  # type: ignore

        room = pra.ShoeBox([4, 4, 2.5], fs=args.sr, max_order=8, absorption=0.25)
        room.add_source([1.0, 1.0, 1.5])
        room.add_microphone([2.5, 2.5, 1.5])
        room.add_source([3.0, 1.0, 1.5], signal=far)
        t0 = time.perf_counter()
        room.simulate()
        rir_mic = room.mic_array.signals[0, :n]
        if len(rir_mic) < n:
            rir_mic = list(rir_mic) + [0.0] * (n - len(rir_mic))
        rows.append(
            {
                "label": "pyroomacoustics-rir",
                "mse_improve_db": mse_improve_db(mic, rir_mic[:n], clean),
                "seconds": time.perf_counter() - t0,
            }
        )
    except ImportError:
        rows.append({"label": "pyroomacoustics-rir", "skipped": "pip install pyroomacoustics"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {args.out}")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
