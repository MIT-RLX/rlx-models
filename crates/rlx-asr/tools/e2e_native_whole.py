#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.
#
# Folded / hybrid native ASR E2E:
#   folded = wav → mel → input_proj_eff → body_map_ls.B (or body_residual_ls.R) → CTC beam → text
#   hybrid = teacher encoder cache → native CTC head (parity gate; needs teacher_work captures)
#
# Usage:
#   python3 e2e_native_whole.py --wav .cache/conformer-ctc/sample.wav
#   python3 e2e_native_whole.py --wav a.wav --mode hybrid --teacher-work .cache/asr/asr_weights/e2e_native/teacher_work
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_io import (  # noqa: E402
    BLANK,
    FRAMES,
    asr_dir,
    ctc_beam_decode,
    decode_pieces,
    mel_chunks_from_wav,
    resolve_units,
)

DIM = 512
OUT_T = 64
SUB = 6
V_CTC = 6081


def load_native_pack(native: Path | None = None) -> dict:
    """Load folded encoder + CTC tensors from `model.rlxp` or legacy GGUF."""
    from gguf_io import load_encoder_pack, resolve_pack

    _ = native  # legacy CLI flag
    path = resolve_pack()
    if path is None:
        raise SystemExit(
            "model.rlxp / model.gguf not found under weights/asr (run: just fetch-rlx-asr)"
        )
    en = load_encoder_pack(path)
    missing = []
    if "frontend.input_proj_eff.W" not in en:
        missing.append("frontend.input_proj_eff.W")
    if "frontend.body_residual_ls.R" not in en:
        missing.append("frontend.body_residual_ls.R")
    if "head.ctc.W_ls" not in en and "head.ctc.W" not in en:
        missing.append("head.ctc.W_ls")
    if missing:
        raise SystemExit(
            f"GGUF {path} incomplete: missing {', '.join(missing)}. "
            "Re-run: just fetch-rlx-asr"
        )
    return en


def ctc_logp(enc: np.ndarray, pack: dict) -> np.ndarray:
    Wc = pack.get("head.ctc.W_ls", pack.get("head.ctc.W")).astype(np.float32)
    bc = pack.get("head.ctc.b_ls")
    if Wc.shape == (V_CTC, DIM):
        logits = enc.astype(np.float32) @ Wc.T
    else:
        logits = enc.astype(np.float32) @ Wc
    if bc is not None:
        logits = logits + bc.astype(np.float32)
    m = logits.max(axis=1, keepdims=True)
    return (logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))).astype(
        np.float32
    )


def _load_body_mlp_ce() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load CE residual MLP: enc = h@B + tanh(h@W1+b1)@W2 + b2."""
    if os.environ.get("RLX_ASR_BODY_MLP", "auto").lower() in {
        "0",
        "false",
        "off",
        "none",
        "no",
    }:
        return None
    root = asr_dir()
    paths = [root / f"body_mlp_ce.{n}.bin" for n in ("B", "W1", "b1", "W2", "b2")]
    if not all(p.is_file() for p in paths):
        return None
    B = np.fromfile(paths[0], dtype="<f4").reshape(DIM, DIM)
    W1 = np.fromfile(paths[1], dtype="<f4").reshape(DIM, DIM)
    b1 = np.fromfile(paths[2], dtype="<f4")
    W2 = np.fromfile(paths[3], dtype="<f4").reshape(DIM, DIM)
    b2 = np.fromfile(paths[4], dtype="<f4")
    if b1.size < DIM or b2.size < DIM:
        return None
    return B, W1, b1[:DIM], W2, b2[:DIM]


def _load_body_map(pack: dict) -> np.ndarray | None:
    """Prefer capture-fit `body_map_ls.B` over pack `body_residual_ls.R`."""
    if "encoder.body_map_ls.B" in pack:
        B = pack["encoder.body_map_ls.B"].astype(np.float32)
        if B.shape == (DIM, DIM):
            return B
    root = asr_dir()
    for name in ("body_map_ls.B.bin",):
        p = root / name
        if p.is_file() and p.stat().st_size == DIM * DIM * 4:
            return np.fromfile(p, dtype="<f4").reshape(DIM, DIM)
    return None


def _load_h_map() -> np.ndarray | None:
    mode = os.environ.get("RLX_ASR_H_MAP", "auto").lower()
    if mode in {"0", "false", "off", "none", "no"}:
        return None
    if mode == "auto":
        # Match Rust: only on LS frontend (live path).
        fe = os.environ.get("RLX_ASR_FRONTEND", "auto").lower()
        if fe in {"raw", "cal", "calibrated", "silence"}:
            return None
    p = asr_dir() / "body_h_map_ls.M.bin"
    if p.is_file() and p.stat().st_size == DIM * DIM * 4:
        return np.fromfile(p, dtype="<f4").reshape(DIM, DIM)
    return None


def forward_folded(feat389: np.ndarray, pack: dict) -> tuple[np.ndarray, np.ndarray]:
    W = pack["frontend.input_proj_eff.W"].astype(np.float32)
    b = pack.get("frontend.input_proj_eff.b")
    if W.shape == (DIM, 80):
        W = W.T
    mel = feat389[np.arange(OUT_T) * SUB].astype(np.float32)
    h = mel @ W
    if b is not None:
        h = h + b.astype(np.float32)
    if (m := _load_h_map()) is not None:
        h = h @ m
    mlp = _load_body_mlp_ce()
    if mlp is not None:
        B, W1, b1, W2, b2 = mlp
        enc = h @ B + np.tanh(h @ W1 + b1) @ W2 + b2
    else:
        body = _load_body_map(pack)
        if body is None:
            body = pack["frontend.body_residual_ls.R"].astype(np.float32)
        enc = h @ body.astype(np.float32)
    return enc.astype(np.float32), ctc_logp(enc, pack)


def teacher_encoder_chunks(teacher_work: Path, wav: Path) -> list[np.ndarray] | None:
    """Load Apple E5 encoder caches [64×512] when golden teacher_work exists."""
    stem = Path(wav).stem
    d = teacher_work / stem
    if not d.is_dir():
        return None
    encs = []
    for chunk_dir in sorted(d.glob("chunk_*")):
        p = chunk_dir / "e5_encoder_cache.bin"
        if not p.is_file():
            return None
        encs.append(np.fromfile(p, dtype=np.float32).reshape(OUT_T, DIM))
    return encs or None


def forward_hybrid_teacher_enc(enc: np.ndarray, pack: dict) -> tuple[np.ndarray, np.ndarray]:
    return enc.astype(np.float32), ctc_logp(enc, pack)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", nargs="+", required=True)
    ap.add_argument("--native", type=Path, default=None)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument(
        "--mode",
        choices=("folded", "hybrid", "both"),
        default="folded",
        help="folded = mel→bodyR→CTC; hybrid = teacher encoder cache→CTC (parity); both = compare",
    )
    ap.add_argument(
        "--teacher-work",
        type=Path,
        default=None,
        help="teacher_work root with <wav_stem>/chunk_*/e5_encoder_cache.bin (hybrid/both)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.mode in ("hybrid", "both") and args.teacher_work is None:
        tw = asr / "asr_weights" / "e2e_native" / "teacher_work"
        if tw.is_dir():
            args.teacher_work = tw

    asr = asr_dir()
    out_dir = args.out or (asr / "e2e_native")
    out_dir.mkdir(parents=True, exist_ok=True)

    pieces = resolve_units(asr)
    try:
        pack = load_native_pack(args.native)
    except SystemExit as e:
        print(f"soft skip: {e}", flush=True)
        summary = {
            "n_wavs": 0,
            "n_folded": 0,
            "n_folded_nonempty": 0,
            "gates": {"e2e_folded_native": "SKIP", "e2e_native": "SKIP"},
            "verdict": f"e2e_native=SKIP: {e}",
        }
        (out_dir / "e2e_native_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0

    reports = []
    n_folded_nonempty = 0
    n_folded = 0
    n_hybrid = 0
    n_hybrid_match = 0

    for wav_s in args.wav:
        wav = Path(wav_s)
        print(f"\n=== {wav} ===", flush=True)
        row: dict = {"wav": str(wav), "mode": args.mode}

        if args.mode in ("hybrid", "both") and args.teacher_work is not None:
            encs = teacher_encoder_chunks(args.teacher_work, wav)
            if encs:
                t0 = time.time()
                wps = []
                for enc in encs:
                    _, logp = forward_hybrid_teacher_enc(enc, pack)
                    wps.append(logp)
                wp = np.concatenate(wps, 0)
                ids, score = ctc_beam_decode(wp, blank=BLANK, beam=args.beam)
                text = decode_pieces(pieces, ids)
                ms = (time.time() - t0) * 1000
                n_hybrid += 1
                row.update(
                    {
                        "hybrid_text": text,
                        "hybrid_ids": ids,
                        "hybrid_score": score,
                        "hybrid_ms": ms,
                    }
                )
                print(f"  hybrid (teacher enc): {text!r}  score={score:.2f}  ({ms:.0f} ms)")
            else:
                row["hybrid_skip"] = f"no teacher_work under {args.teacher_work}/{wav.stem}"
                print(f"  hybrid skip: {row['hybrid_skip']}")

        if args.mode in ("folded", "both"):
            chunks = mel_chunks_from_wav(wav, asr=asr, hop_frames=FRAMES, max_chunks=32)
            if not chunks:
                print("  no mel chunks — skip folded")
            else:
                print(f"  chunks={len(chunks)}", flush=True)
                row["n_chunks"] = len(chunks)
                t0 = time.time()
                encs, wps = [], []
                for ch in chunks:
                    enc, logp = forward_folded(ch, pack)
                    encs.append(enc)
                    wps.append(logp)
                wp = np.concatenate(wps, 0)
                ids, score = ctc_beam_decode(wp, blank=BLANK, beam=args.beam)
                text = decode_pieces(pieces, ids)
                ms = (time.time() - t0) * 1000
                n_folded += 1
                if text.strip():
                    n_folded_nonempty += 1
                print(f"  folded native: {text!r}  score={score:.2f}  ({ms:.0f} ms)")
                row.update(
                    {
                        "folded_text": text,
                        "folded_ids": ids,
                        "folded_score": score,
                        "folded_ms": ms,
                    }
                )

        if "hybrid_text" in row and "folded_text" in row:
            row["hybrid_text_match_folded"] = row["hybrid_text"] == row["folded_text"]
            if row.get("hybrid_text") == row.get("folded_text") and row["hybrid_text"].strip():
                n_hybrid_match += 1
        reports.append(row)

    folded_gate = "SOFT_PASS" if n_folded_nonempty > 0 else "HARD_FAIL"
    if n_folded == 0 and args.mode in ("folded", "both"):
        folded_gate = "SKIP"
    hybrid_gate = "SKIP"
    if n_hybrid > 0:
        hybrid_gate = "HARD_PASS" if n_hybrid_match == n_hybrid else "SOFT_PASS"
    summary = {
        "n_wavs": len(reports),
        "n_hybrid": n_hybrid,
        "n_hybrid_text_match_teacher": n_hybrid if hybrid_gate == "HARD_PASS" else 0,
        "n_folded": n_folded,
        "n_folded_nonempty": n_folded_nonempty,
        "gates": {
            "e2e_folded_native": folded_gate,
            "e2e_hybrid_native_ctc": hybrid_gate,
            "e2e_native": hybrid_gate if n_hybrid else folded_gate,
        },
        "report": str(out_dir / "e2e_native_report.json"),
        "verdict": (
            f"e2e_native={hybrid_gate if n_hybrid else folded_gate}: "
            f"hybrid {n_hybrid}/{len(reports)}; folded nonempty {n_folded_nonempty}/{n_folded}."
        ),
    }
    (out_dir / "e2e_native_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    (out_dir / "e2e_native_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'e2e_native_report.json'}")
    return 0 if folded_gate != "HARD_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
