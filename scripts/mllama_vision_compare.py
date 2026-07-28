#!/usr/bin/env python3
# Compare RLX mllama vision output (--dump-vision) against the HF reference.
#
#   python3 scripts/mllama_vision_compare.py --ref out/mllama_ref_cross_states.npy \
#       --rlx out/mllama_rlx_vision
#
# --rlx points at the prefix written by `rlx-mllama --dump-vision <prefix>`
# (reads <prefix>.f32 + <prefix>.json). Prints per-tile and overall cosine.
import argparse, json
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="HF cross_states .npy [n_tiles, num_patches, hidden]")
    ap.add_argument("--rlx", required=True, help="RLX dump prefix (<prefix>.f32 + <prefix>.json)")
    args = ap.parse_args()

    ref = np.load(args.ref).astype(np.float64)  # [T, P, H]
    with open(f"{args.rlx}.json") as f:
        shape = json.load(f)["shape"]
    rlx = np.fromfile(f"{args.rlx}.f32", dtype="<f4").astype(np.float64).reshape(shape)

    if ref.shape != rlx.shape:
        print(f"SHAPE MISMATCH ref {ref.shape} vs rlx {rlx.shape}")
        # still attempt a flattened compare on the min size
        n = min(ref.size, rlx.size)
        a, b = ref.reshape(-1)[:n], rlx.reshape(-1)[:n]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"flattened cosine (min {n}): {cos:.6f}")
        return

    a, b = ref.reshape(-1), rlx.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.abs(ref - rlx).mean())
    print(f"overall cosine {cos:.6f}  mean|Δ| {mad:.3e}  shape {ref.shape}")
    for t in range(ref.shape[0]):
        at, bt = ref[t].reshape(-1), rlx[t].reshape(-1)
        ct = float(at @ bt / (np.linalg.norm(at) * np.linalg.norm(bt) + 1e-12))
        print(f"  tile {t}: cosine {ct:.6f}")


if __name__ == "__main__":
    main()
