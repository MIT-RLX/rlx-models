#!/usr/bin/env python3
# Fit encoder.body_out_ls.A from Apple capture_work (teacher mel + encoder cache).
#
#   enc_fold = subsample(teacher_feat) @ input_proj @ bodyR
#   A = ridge_ls(enc_fold, enc_teacher)
#
# Usage:
#   python3 fit_body_out_ls.py --capture ~/.cache/asr/capture_work --out body_out_ls.A.bin
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DIM = 512
OUT_T = 64
SUB = 6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True, help="capture_work root")
    ap.add_argument("--pack", type=Path, default=None, help="model.rlxp dir or file")
    ap.add_argument("--out", type=Path, required=True, help="body_out_ls.A.bin output")
    ap.add_argument("--lam", type=float, default=0.1, help="ridge lambda")
    ap.add_argument("--holdout", nargs="*", default=[], help="utt dir names to exclude")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from e2e_native_whole import load_native_pack

    pack = load_native_pack(args.pack)
    W = pack["frontend.input_proj_eff.W"].astype(np.float32)
    if W.shape == (DIM, 80):
        W = W.T
    b = pack.get("frontend.input_proj_eff.b")
    R = pack["frontend.body_residual_ls.R"].astype(np.float32)
    hold = set(args.holdout)

    xs, ys = [], []
    n_utts = 0
    for d in sorted(args.capture.iterdir()):
        if not d.is_dir() or d.name in hold:
            continue
        feat_p, enc_p = d / "e5_feature.bin", d / "e5_encoder_cache.bin"
        if not (feat_p.is_file() and enc_p.is_file()):
            continue
        feat = np.fromfile(feat_p, dtype=np.float32)
        enc_t = np.fromfile(enc_p, dtype=np.float32)
        if feat.size % 80:
            continue
        t_len = feat.size // 80
        if t_len < OUT_T * SUB:
            continue
        feat = feat.reshape(t_len, 80)
        enc_t = enc_t.reshape(-1, DIM)
        mel = feat[np.arange(OUT_T) * SUB]
        h = mel @ W
        if b is not None:
            h = h + b.astype(np.float32)
        enc = h @ R
        n = min(enc.shape[0], enc_t.shape[0])
        xs.append(enc[:n])
        ys.append(enc_t[:n])
        n_utts += 1

    if not xs:
        raise SystemExit(f"no usable captures under {args.capture}")

    X = np.concatenate(xs, 0)
    Y = np.concatenate(ys, 0)
    lam = args.lam
    A = np.linalg.solve(X.T @ X + lam * np.eye(DIM), X.T @ Y).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    A.astype("<f4").tofile(args.out)
    corr = np.mean([np.corrcoef((X @ A)[i], Y[i])[0, 1] for i in range(min(500, len(X)))])
    print(f"wrote {args.out}  utts={n_utts}  frames={X.shape[0]}  train_corr={corr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
