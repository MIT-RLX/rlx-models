#!/usr/bin/env python3
"""Encode a mono reference WAV into Orpheus SNAC audio token ids.

Usage:
  python3 scripts/orpheus_encode_reference.py \\
    --wav assets/jfk/jfk_voice_clone.wav \\
    --transcript "Ask not what your country can do for you." \\
    --out /tmp/jfk_orpheus_ref.json

Requires: pip install snac torch safetensors soundfile
Optional: pip install whisper for --transcript auto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SNAC_TOKEN_OFFSET = 128_256 + 10  # CUSTOM_TOKEN_BASE + 10


def load_mono_24k(path: Path):
    try:
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        print(f"error: {e}\ninstall: pip install soundfile numpy", file=sys.stderr)
        raise SystemExit(1) from e

    pcm, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sr != 24_000:
        import torch
        import torchaudio

        t = torch.from_numpy(pcm).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, 24_000)
        pcm = t.squeeze(0).numpy()
    return pcm


def snac_levels_to_token_ids(codes) -> tuple[list[int], list[int]]:
    """Orpheus interleaved layout (see canopyai/Orpheus-TTS issues #153/#200)."""
    token_ids: list[int] = []
    frame_codes: list[int] = []
    n_frames = codes[0].shape[1]
    for i in range(n_frames):
        frame = [
            int(codes[0][0][i].item()),
            int(codes[1][0][2 * i].item()),
            int(codes[2][0][4 * i].item()),
            int(codes[2][0][4 * i + 1].item()),
            int(codes[1][0][2 * i + 1].item()),
            int(codes[2][0][4 * i + 2].item()),
            int(codes[2][0][4 * i + 3].item()),
        ]
        frame_codes.extend(frame)
        for slot, code in enumerate(frame):
            token_ids.append(SNAC_TOKEN_OFFSET + code + slot * 4096)
    return token_ids, frame_codes


def transcribe_whisper(wav_path: Path) -> str:
    try:
        import whisper
    except ImportError as e:
        print(
            "error: whisper not installed; pass --transcript explicitly",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    model = whisper.load_model("base")
    result = model.transcribe(str(wav_path), language="en")
    return result["text"].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True, help="mono reference WAV")
    parser.add_argument(
        "--transcript",
        default="auto",
        help='spoken text in the clip, or "auto" for Whisper base',
    )
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--max-seconds", type=float, default=12.0, help="trim reference")
    args = parser.parse_args()

    if not args.wav.is_file():
        print(f"error: missing wav {args.wav}", file=sys.stderr)
        return 1

    try:
        import torch
        from snac import SNAC
    except ImportError as e:
        print(f"error: {e}\ninstall: pip install snac torch", file=sys.stderr)
        return 1

    pcm = load_mono_24k(args.wav)
    max_samples = int(args.max_seconds * 24_000)
    if len(pcm) > max_samples:
        pcm = pcm[:max_samples]

    if args.transcript == "auto":
        transcript = transcribe_whisper(args.wav)
        print(f"[orpheus_encode] whisper transcript: {transcript!r}")
    else:
        transcript = args.transcript.strip()
    if not transcript:
        print("error: empty transcript", file=sys.stderr)
        return 1

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    with torch.inference_mode():
        wave = torch.from_numpy(pcm).float().unsqueeze(0).unsqueeze(0)
        codes = model.encode(wave)

    token_ids, frame_codes = snac_levels_to_token_ids(codes)
    payload = {
        "transcript": transcript,
        "token_ids": token_ids,
        "frame_codes": frame_codes,
        "sample_rate": 24_000,
        "wav": str(args.wav.resolve()),
        "num_frames": len(frame_codes) // 7,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {len(token_ids)} audio tokens ({payload['num_frames']} frames) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
