#!/usr/bin/env python3
"""Warm forecast latency of the upstream `timestral` package (CPU), to compare
with `cargo run --release -p rlx-timestral --example bench` on the same input.

    PYTHONPATH=.cache/timestral/src/src python3 scripts/timestral_bench_torch.py \
        DIR series.csv [HORIZON] [RUNS]

`series.csv` is one value per line. Prints the best and median of RUNS warm
calls with the default options (two passes: symmetric averaging).
"""

import sys
import time

import numpy as np
import torch
from timestral import TimestralForecaster


def main() -> None:
    weights, csv = sys.argv[1], sys.argv[2]
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    series = np.loadtxt(csv, dtype=np.float64)[None, None]
    fc = TimestralForecaster.from_pretrained(weights)
    with torch.inference_mode():
        fc.forecast(series, horizon)
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            fc.forecast(series, horizon)
            times.append((time.perf_counter() - t) * 1e3)
    times.sort()
    print(f"pytorch cpu: context {series.shape[-1]}, horizon {horizon}: "
          f"warm best {times[0]:.1f} ms, median {times[len(times) // 2]:.1f} ms")


if __name__ == "__main__":
    main()
