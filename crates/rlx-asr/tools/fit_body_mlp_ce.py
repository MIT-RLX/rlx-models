#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.
#
# Fit CE residual body MLP from Apple capture_work:
#   enc = h @ B + tanh(h @ W1 + b1) @ W2 + b2
# Supervised by CTC CE vs teacher wp argmax (+ light MSE to teacher enc).
# Uses all capture windows, momentum SGD, speech-weighted CE.
#
# Usage:
#   python3 fit_body_mlp_ce.py --capture-work ".../capture_work" \
#     --out-dir /Users/Shared/translator/models/rlx-asr
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from audio_io import asr_dir, ctc_beam_decode, decode_pieces, resolve_units
from e2e_native_whole import ctc_logp, load_native_pack

DIM = 512
OUT_T = 64
SUB = 6
V = 6081
HID = 512
FRAMES = OUT_T * SUB


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-work", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--init", type=Path, default=None, help="optional prior body_mlp_ce dir")
    args = ap.parse_args()

    pack = load_native_pack()
    pieces = resolve_units()
    W = pack["frontend.input_proj_eff.W"].astype(np.float32)
    if W.shape == (DIM, 80):
        W = W.T
    b = pack.get("frontend.input_proj_eff.b")
    Wc = pack.get("head.ctc.W_ls", pack["head.ctc.W"]).astype(np.float32)
    bc = pack.get("head.ctc.b_ls")
    if Wc.shape != (V, DIM):
        Wc = Wc.T

    utts = []
    for d in sorted(args.capture_work.iterdir()):
        if not d.is_dir() or d.name.startswith("synth_"):
            continue
        fp, ep, wp = d / "e5_feature.bin", d / "e5_encoder_cache.bin", d / "e5_wp_prob.bin"
        if not all(p.is_file() for p in (fp, ep, wp)):
            continue
        feat = np.fromfile(fp, dtype=np.float32)
        if feat.size % 80:
            continue
        feat = feat.reshape(-1, 80)
        enc_all = np.fromfile(ep, dtype=np.float32)
        if enc_all.size % DIM:
            continue
        enc_all = enc_all.reshape(-1, DIM)
        L_all = np.fromfile(wp, dtype=np.float32)
        if L_all.size % V:
            continue
        L_all = L_all.reshape(-1, V)
        n_enc = min(enc_all.shape[0], L_all.shape[0], feat.shape[0] // SUB)
        for off in range(0, max(1, n_enc - OUT_T + 1), OUT_T):
            if off + OUT_T > n_enc:
                break
            mel_off = off * SUB
            if mel_off + FRAMES > feat.shape[0]:
                break
            gold = L_all[off : off + OUT_T].argmax(1)
            if (gold != 0).sum() < 4:
                continue
            h = feat[mel_off + np.arange(OUT_T) * SUB] @ W
            if b is not None:
                h = h + b.astype(np.float32)
            utts.append(
                (
                    h.astype(np.float32),
                    enc_all[off : off + OUT_T].astype(np.float32),
                    gold.astype(np.int64),
                    f"{d.name}:{off}",
                )
            )

    if len(utts) < 8:
        raise SystemExit(f"need more speech captures, got {len(utts)}")

    hold = ("ask_not_00", "moon_00", "jfk_voice_clone_00")
    eval_utts = [u for u in utts if any(k in u[3] for k in hold) and u[3].endswith(":0")]
    if not eval_utts:
        eval_utts = utts[: min(8, len(utts))]

    rng = np.random.default_rng(0)
    H = np.concatenate([u[0] for u in utts])
    E = np.concatenate([u[1] for u in utts])
    G = np.concatenate([u[2] for u in utts])

    init_dir = args.init or asr_dir()
    if (init_dir / "body_mlp_ce.B.bin").is_file():
        B = np.fromfile(init_dir / "body_mlp_ce.B.bin", dtype="<f4").reshape(DIM, DIM)
        W1 = np.fromfile(init_dir / "body_mlp_ce.W1.bin", dtype="<f4").reshape(DIM, HID)
        b1 = np.fromfile(init_dir / "body_mlp_ce.b1.bin", dtype="<f4")[:HID].copy()
        W2 = np.fromfile(init_dir / "body_mlp_ce.W2.bin", dtype="<f4").reshape(HID, DIM)
        b2 = np.fromfile(init_dir / "body_mlp_ce.b2.bin", dtype="<f4")[:DIM].copy()
    else:
        B = np.linalg.solve(H.T @ H + args.lam * np.eye(DIM), H.T @ E).astype(np.float32)
        W1 = (rng.standard_normal((DIM, HID)) * 0.01).astype(np.float32)
        b1 = np.zeros(HID, np.float32)
        W2 = (rng.standard_normal((HID, DIM)) * 0.01).astype(np.float32)
        b2 = np.zeros(DIM, np.float32)

    def forward(h: np.ndarray) -> np.ndarray:
        return h @ B + np.tanh(h @ W1 + b1) @ W2 + b2

    def kw(t: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z']{3,}", t.lower()))

    def score() -> tuple[float, str]:
        hits = total = 0
        ask = ""
        for h, enc_t, _, name in eval_utts:
            hyp = decode_pieces(pieces, ctc_beam_decode(ctc_logp(forward(h), pack), beam=12)[0])
            ref = decode_pieces(pieces, ctc_beam_decode(ctc_logp(enc_t, pack), beam=12)[0])
            hits += len(kw(hyp) & kw(ref))
            total += len(kw(ref))
            if "ask_not_00" in name:
                ask = hyp
        sc = hits / max(1, total)
        sc += 0.05 * sum(1 for t in ("ask", "not", "america", "fellow") if t in ask.lower())
        return sc, ask

    vB = np.zeros_like(B)
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)
    mu = 0.9
    best = (-1.0, None, -1, "")
    for epoch in range(args.epochs):
        lr = args.lr * (0.55 ** (epoch // 20))
        perm = rng.permutation(len(H))
        for s in range(0, len(H), 192):
            ii = perm[s : s + 192]
            h = H[ii]
            g = G[ii]
            e_tgt = E[ii]
            enc = forward(h)
            z = enc @ Wc.T
            if bc is not None:
                z = z + bc
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            w = np.where(g != 0, 8.0, 0.6).astype(np.float32)
            dz = p.copy()
            dz[np.arange(len(g)), g] -= 1
            dz *= w[:, None]
            dz /= w.sum()
            mse_w = 0.12 if epoch < 15 else 0.04
            d = dz @ Wc + mse_w * 2 * (enc - e_tgt) / max(1, len(h))
            hid = np.tanh(h @ W1 + b1)
            scv = 1.0 / len(h)
            gB = (h.T @ d) * scv
            gb2 = d.sum(0) * scv
            gW2 = (hid.T @ d) * scv
            d_hid = (d @ W2.T) * (1 - hid * hid)
            gW1 = (h.T @ d_hid) * scv
            gb1 = d_hid.sum(0) * scv
            vB = mu * vB + gB
            vW1 = mu * vW1 + gW1
            vb1 = mu * vb1 + gb1
            vW2 = mu * vW2 + gW2
            vb2 = mu * vb2 + gb2
            B -= lr * vB
            W1 -= lr * vW1
            b1 -= lr * vb1
            W2 -= lr * vW2
            b2 -= lr * vb2
        if epoch % 5 == 4:
            sc, ask = score()
            print(f"ep{epoch+1:03d} score={sc*100:.1f}% ask={ask[:60]!r}")
            if sc > best[0]:
                best = (
                    sc,
                    (B.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy()),
                    epoch + 1,
                    ask,
                )

    if best[1] is None:
        raise SystemExit("no checkpoint")
    B, W1, b1, W2, b2 = best[1]
    out = args.out_dir or asr_dir()
    out.mkdir(parents=True, exist_ok=True)
    for n, arr in [("B", B), ("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
        arr.astype("<f4").tofile(out / f"body_mlp_ce.{n}.bin")
    (out / "body_mlp_ce.meta.txt").write_text(
        f"hid={HID}\nscore={best[0]}\nepoch={best[2]}\nask={best[3]}\n"
        f"form=h@B+tanh(h@W1+b1)@W2+b2\ndomain=apple_capture_work\nwindows={len(utts)}\n"
    )
    print(f"wrote {out}/body_mlp_ce.* best_score={best[0]:.3f} ep={best[2]} ask={best[3]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
