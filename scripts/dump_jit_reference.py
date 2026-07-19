#!/usr/bin/env python3
"""Run HOCT TorchScript on a fixed-seed batch; save edge logits for Rust parity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pt", type=Path, help="Path to general_v0.pt")
    p.add_argument("-o", "--out", type=Path, default=Path("/tmp/hoct_ref_logits.npy"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--edges", type=int, default=12)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    m = torch.jit.load(str(args.pt), map_location="cpu")
    m.eval()

    b, n, e, d = 1, args.nodes, args.edges, 19
    node_features = torch.randn(b, n, d)
    node_pos = torch.randn(b, n, 3) * 100
    edge_pos = torch.randn(b, e, 3) * 100
    edge_indices = torch.randint(0, n, (b, e, 2))
    node_mask = torch.ones(b, n, dtype=torch.bool)
    edge_mask = torch.ones(b, e, dtype=torch.bool)

    with torch.no_grad():
        logits, _, _, orphan = m(
            node_features, node_pos, edge_pos, edge_indices, node_mask, edge_mask
        )
    assert orphan.abs().sum().item() == 0.0
    arr = logits.cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, arr)
    stem = args.out.with_suffix("")
    np.save(f"{stem}_node_features.npy", node_features.numpy())
    np.save(f"{stem}_node_pos.npy", node_pos.numpy())
    np.save(f"{stem}_edge_pos.npy", edge_pos.numpy())
    np.save(f"{stem}_edge_indices.npy", edge_indices.numpy().astype(np.int64))
    np.save(f"{stem}_node_mask.npy", node_mask.numpy())
    np.save(f"{stem}_edge_mask.npy", edge_mask.numpy())
    print(f"wrote logits {arr.shape} -> {args.out}")
    print(f"wrote inputs with stem {stem}_*.npy")
    print("sample", arr[0, :5, 0].tolist())


if __name__ == "__main__":
    main()
