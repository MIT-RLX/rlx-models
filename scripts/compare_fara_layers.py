"""Compare RLX vs HF Fara layer dumps (last-token hiddens).

.venv-hf/bin/python scripts/compare_fara_layers.py \\
  --rlx /tmp/fara_rlx_layers --hf /tmp/fara_hf_layers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlx", default="/tmp/fara_rlx_layers")
    ap.add_argument("--hf", default="/tmp/fara_hf_layers")
    args = ap.parse_args()
    rlx_dir = Path(args.rlx)
    hf_dir = Path(args.hf)

    names = ["embed"] + [f"layer_{i:02d}" for i in range(32)] + ["logits"]
    print(f"{'name':12s} {'cos':>8s} {'l2_ratio':>9s} {'maxdiff':>10s} {'rlx_absmax':>11s} {'hf_absmax':>10s}")
    first_bad = None
    for name in names:
        rp = rlx_dir / f"{name}.npy"
        hp = hf_dir / f"{name}.npy"
        if not rp.is_file() or not hp.is_file():
            print(f"{name:12s} MISSING rlx={rp.is_file()} hf={hp.is_file()}")
            continue
        r = np.load(rp)
        h = np.load(hp)
        # logits may differ in trailing vocab pad
        n = min(r.size, h.size)
        r = r.reshape(-1)[:n]
        h = h.reshape(-1)[:n]
        c = cos(r, h)
        lr = float(np.linalg.norm(r) / (np.linalg.norm(h) + 1e-12))
        md = float(np.max(np.abs(r - h)))
        print(
            f"{name:12s} {c:8.5f} {lr:9.4f} {md:10.4f} "
            f"{float(np.abs(r).max()):11.4f} {float(np.abs(h).max()):10.4f}"
        )
        if first_bad is None and (not np.isfinite(c) or c < 0.99):
            first_bad = name
    if first_bad:
        print(f"\nfirst divergence (cos < 0.99): {first_bad}")
    else:
        print("\nall compared tensors cos >= 0.99")


if __name__ == "__main__":
    main()
