#!/usr/bin/env python3
"""Dump feature / parental-softmax / ILP reference fixtures for rlx-hoct."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Pure-numpy ports of hoct formulas (no tracksdata required).


def border_dist_nd(coords: np.ndarray, shape: tuple[int, ...], cutoff: float = 5.0) -> np.ndarray:
    shape_a = np.asarray(shape)[None, :]
    distance = np.minimum(coords, shape_a - coords).min(axis=1)
    return 1.0 - np.minimum(1.0, distance / cutoff)


def parental_softmax(sim_exp: np.ndarray, orphan_exp: float, target: np.ndarray, delta_t: np.ndarray):
    """Per-(target, delta_t) normalize; return similarity + orphan_prob by target."""
    out = np.zeros_like(sim_exp)
    orphan_by = {}
    for t in np.unique(target):
        for dt in np.unique(delta_t[target == t]):
            m = (target == t) & (delta_t == dt)
            denom = sim_exp[m].sum() + orphan_exp
            out[m] = sim_exp[m] / denom
            orphan_by[(int(t), int(dt))] = float(orphan_exp / denom)
    return out, orphan_by


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pipeline"
    out.mkdir(parents=True, exist_ok=True)

    # Synthetic 2D labels → border_dist at centroids
    labels = np.zeros((3, 16, 16), dtype=np.uint32)
    labels[0, 4:6, 4:7] = 1
    labels[1, 5:7, 5:8] = 1
    labels[2, 6:8, 6:9] = 1
    labels[0, 12:14, 12:14] = 2
    labels[1, 12:14, 11:13] = 2
    labels[2, 11:13, 11:13] = 2
    np.save(out / "labels.npy", labels)

    # Parental softmax micro-fixture
    sim_exp = np.array([2.0, 1.0, 0.5, 3.0], dtype=np.float32)
    target = np.array([1, 1, 1, 2], dtype=np.int64)
    delta_t = np.array([1, 1, 2, 1], dtype=np.int64)
    sim, orphans = parental_softmax(sim_exp, 1.0, target, delta_t)
    np.save(out / "soft_sim_exp.npy", sim_exp)
    np.save(out / "soft_target.npy", target)
    np.save(out / "soft_delta_t.npy", delta_t)
    np.save(out / "soft_similarity.npy", sim.astype(np.float32))
    with (out / "soft_orphans.json").open("w") as f:
        json.dump({f"{k[0]},{k[1]}": v for k, v in orphans.items()}, f, indent=2)

    # Border dist reference for centroids of label 1 / 2 on frame 0
    coords = np.array([[0.0, 4.5, 5.0], [0.0, 12.5, 12.5]], dtype=np.float32)
    bd = border_dist_nd(coords, (1, 16, 16), 5.0)
    np.save(out / "border_dist.npy", bd.astype(np.float32))
    print("wrote", out)


if __name__ == "__main__":
    main()
