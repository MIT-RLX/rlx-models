#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.
#
# Folded native ASR E2E:
#   wav → calibrated fbank → input_proj_eff + body_residual_ls.R → CTC beam → text
#
# Usage:
#   python3 e2e_native_whole.py --wav .cache/conformer-ctc/sample.wav
#   python3 e2e_native_whole.py --wav a.wav --mode folded
from __future__ import annotations

import argparse
import json
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
    """Load folded encoder + CTC tensors from `weights/asr/model.gguf`."""
    from gguf_io import load_encoder_pack, resolve_gguf

    _ = native  # legacy CLI flag; GGUF is canonical
    gguf = resolve_gguf()
    if gguf is None:
        raise SystemExit(
            "model.gguf not found under weights/asr (run: just asr-pack-gguf)"
        )
    en = load_encoder_pack(gguf)
    missing = []
    if "frontend.input_proj_eff.W" not in en:
        missing.append("frontend.input_proj_eff.W")
    if "frontend.body_residual_ls.R" not in en:
        missing.append("frontend.body_residual_ls.R")
    if "head.ctc.W_ls" not in en and "head.ctc.W" not in en:
        missing.append("head.ctc.W_ls")
    if missing:
        raise SystemExit(
            f"GGUF {gguf} incomplete: missing {', '.join(missing)}. "
            "Re-run: just asr-pack-gguf"
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


def forward_folded(feat389: np.ndarray, pack: dict) -> tuple[np.ndarray, np.ndarray]:
    W = pack["frontend.input_proj_eff.W"].astype(np.float32)
    b = pack.get("frontend.input_proj_eff.b")
    R = pack["frontend.body_residual_ls.R"].astype(np.float32)
    if W.shape == (DIM, 80):
        W = W.T
    mel = feat389[np.arange(OUT_T) * SUB].astype(np.float32)
    h = mel @ W
    if b is not None:
        h = h + b.astype(np.float32)
    enc = h + h @ R.T
    return enc.astype(np.float32), ctc_logp(enc, pack)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", nargs="+", required=True)
    ap.add_argument("--native", type=Path, default=None)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument(
        "--mode",
        choices=("folded",),
        default="folded",
        help="folded = mel → input_proj + body R → CTC (only supported mode)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

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

    for wav_s in args.wav:
        wav = Path(wav_s)
        print(f"\n=== {wav} ===", flush=True)
        chunks = mel_chunks_from_wav(wav, asr=asr, hop_frames=FRAMES, max_chunks=32)
        if not chunks:
            print("  no mel chunks — skip")
            continue
        print(f"  chunks={len(chunks)}", flush=True)
        row: dict = {"wav": str(wav), "n_chunks": len(chunks), "mode": "folded"}

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
        reports.append(row)

    folded_gate = "SOFT_PASS" if n_folded_nonempty > 0 else "HARD_FAIL"
    if n_folded == 0:
        folded_gate = "SKIP"
    summary = {
        "n_wavs": len(reports),
        "n_folded": n_folded,
        "n_folded_nonempty": n_folded_nonempty,
        "gates": {
            "e2e_folded_native": folded_gate,
            "e2e_native": folded_gate,
        },
        "report": str(out_dir / "e2e_native_report.json"),
        "verdict": (
            f"e2e_native={folded_gate}: folded nonempty {n_folded_nonempty}/{n_folded}."
        ),
    }
    (out_dir / "e2e_native_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    (out_dir / "e2e_native_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'e2e_native_report.json'}")
    return 0 if folded_gate != "HARD_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
