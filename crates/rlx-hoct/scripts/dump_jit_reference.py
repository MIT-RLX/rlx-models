#!/usr/bin/env python3
"""Dump JIT reference tensors for rlx-hoct parity tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, default=Path("/tmp/hoct_ref_logits"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = torch.jit.load(str(args.pt), map_location="cpu")
    m.eval()
    torch.manual_seed(args.seed)
    b, n, e = 1, 4, 3
    nf = torch.randn(b, n, 19)
    npos = torch.randn(b, n, 3) * 20
    eidx = torch.tensor([[[0, 1], [1, 2], [2, 3]]], dtype=torch.long)
    src, tgt = eidx[..., 0], eidx[..., 1]
    epos = 0.5 * (
        torch.gather(npos, 1, src.unsqueeze(-1).expand(b, e, 3))
        + torch.gather(npos, 1, tgt.unsqueeze(-1).expand(b, e, 3))
    )
    nmask = torch.ones(b, n, dtype=torch.bool)
    emask = torch.ones(b, e, dtype=torch.bool)
    with torch.no_grad():
        logits, node_h, edge_h, orphan = m.forward(nf, npos, epos, eidx, nmask, emask)

    stem = args.out_prefix
    np.save(f"{stem}.npy", logits.numpy())
    np.save(f"{stem}_node_h.npy", node_h.numpy())
    np.save(f"{stem}_edge_h.npy", edge_h.numpy())
    np.save(f"{stem}_node_features.npy", nf.numpy())
    np.save(f"{stem}_node_pos.npy", npos.numpy())
    np.save(f"{stem}_edge_pos.npy", epos.numpy())
    np.save(f"{stem}_edge_indices.npy", eidx.numpy())
    np.save(f"{stem}_node_mask.npy", nmask.numpy())
    np.save(f"{stem}_edge_mask.npy", emask.numpy())
    print("wrote", stem, "logits", logits.flatten().tolist())


if __name__ == "__main__":
    main()
