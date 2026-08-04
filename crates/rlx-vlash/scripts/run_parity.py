#!/usr/bin/env python
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# GPL-3.0-only (the rlx-vlash crate); VLASH itself is Apache-2.0.
"""End-to-end VLASH π₀ / π₀.₅ parity: run the ORIGINAL implementation and the
rlx-vlash port on identical inputs, then compare stage-by-stage.

Pipeline:
  1. `vlash_ref_dump.py` runs upstream VLASH on CPU/float32 → `<name>.bin`
     (reference) + the shared inputs (pixel_values / token_ids / state / noise).
  2. `cargo run --example dump_intermediates` runs rlx-vlash on those SAME
     inputs → `rlx_<name>.bin`.
  3. This script loads both and prints a cosine / max|Δ| table with PASS/FAIL
     (cosine > --threshold, default 0.999); exits non-zero on any failure.

Prereqs: a Python env with upstream VLASH installed (see `vlash_ref_dump.py`);
a local checkpoint dir with `model.safetensors` + `tokenizer.json` for the Rust
side (or an HF repo id — resolved via `huggingface_hub`); a Rust toolchain.

Usage:
    python run_parity.py --variant pi05 \
        --checkpoint lerobot/pi05_base \
        --out ~/.cache/rlx-vlash/fixtures/pi05 \
        [--rlx-model <dir with model.safetensors>] \
        [--threshold 0.999] [--prompt "pick up the cube"] [--skip-ref]
"""

import argparse
import os
import subprocess
import sys

import numpy as np

STAGES = ["image_features_raw", "prefix_embeds", "velocity_step0", "actions_padded"]
HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)  # crates/rlx-vlash


def load_bin(path):
    return np.fromfile(path, dtype="<f4")


def cosine(a, b):
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def resolve_rlx_model(checkpoint):
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=checkpoint, allow_patterns=["*.safetensors", "*.json"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["pi0", "pi05"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rlx-model", default=None)
    ap.add_argument("--prompt", default="do the task")
    ap.add_argument("--prompt-len", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.999)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument(
        "--tokens",
        default="",
        help="comma-separated fixed token ids (bypasses the gated PaliGemma tokenizer)",
    )
    ap.add_argument("--skip-ref", action="store_true", help="reuse an existing reference dump")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) reference dump (original implementation).
    if not args.skip_ref:
        print("== [1/3] running upstream VLASH reference dump ==")
        subprocess.run(
            [
                sys.executable, os.path.join(HERE, "vlash_ref_dump.py"),
                "--variant", args.variant,
                "--checkpoint", args.checkpoint,
                "--out", args.out,
                "--prompt", args.prompt,
                "--prompt-len", str(args.prompt_len),
                "--num-images", str(args.num_images),
                "--num-steps", str(args.num_steps),
                "--tokens", args.tokens,
            ],
            check=True,
        )

    # 2) rlx-vlash dump on the same inputs.
    rlx_model = args.rlx_model or resolve_rlx_model(args.checkpoint)
    print(f"== [2/3] running rlx-vlash (model={rlx_model}) ==")
    subprocess.run(
        [
            "cargo", "run", "--release", "--example", "dump_intermediates",
            "--manifest-path", os.path.join(CRATE, "Cargo.toml"),
            "--", "--variant", args.variant, "--model", rlx_model, "--fixture", args.out,
        ],
        check=True,
    )

    # 3) compare.
    print("== [3/3] comparison (cosine / max|Δ|) ==")
    print(f"  {'stage':22s} {'cosine':>10s} {'max|Δ|':>12s} {'n':>8s}  result")
    all_ok = True
    for stage in STAGES:
        ref_p = os.path.join(args.out, f"{stage}.bin")
        rlx_p = os.path.join(args.out, f"rlx_{stage}.bin")
        if not (os.path.isfile(ref_p) and os.path.isfile(rlx_p)):
            print(f"  {stage:22s} {'(missing)':>10s}")
            continue
        ref, rlx = load_bin(ref_p), load_bin(rlx_p)
        if len(ref) != len(rlx):
            print(f"  {stage:22s}  LENGTH MISMATCH ref={len(ref)} rlx={len(rlx)}  FAIL")
            all_ok = False
            continue
        cos = cosine(ref, rlx)
        mx = float(np.max(np.abs(ref.astype(np.float64) - rlx.astype(np.float64))))
        ok = cos > args.threshold
        all_ok = all_ok and ok
        print(f"  {stage:22s} {cos:10.6f} {mx:12.3e} {len(ref):8d}  {'PASS' if ok else 'FAIL'}")

    print("\n" + ("ALL STAGES PASS ✅" if all_ok else "PARITY FAILED ❌"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
