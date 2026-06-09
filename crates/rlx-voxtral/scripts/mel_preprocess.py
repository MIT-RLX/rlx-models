#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Whisper log-mel frontend matching HF VoxtralProcessor (30 s pad, 128 bins)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoProcessor


def pcm_from_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(path, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sr != 16_000:
        raise SystemExit(f"expected 16 kHz wav, got {sr}")
    return pcm


def mel_from_pcm(processor, pcm: np.ndarray, language: str | None) -> tuple[np.ndarray, list[int]]:
    from mistral_common.protocol.transcription.request import TranscriptionRequest

    import io
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, pcm, 16_000, format="WAV")
    buf.seek(0)
    req = TranscriptionRequest.from_openai(
        {
            "model": "mistralai/Voxtral-Mini-3B-2507",
            "file": buf,
            "language": language,
        }
    )
    tokenized = processor.tokenizer.tokenizer.encode_transcription(req)
    audio_kwargs = {
        "sampling_rate": 16_000,
        "padding": True,
        "truncation": False,
        "pad_to_multiple_of": 480_000,
        "return_tensors": "pt",
    }
    feats = processor._retrieve_input_features(
        [el.audio_array for el in tokenized.audios],
        max_source_positions=3_000,
        **audio_kwargs,
    )
    mel = feats[0].numpy().astype(np.float32)
    return mel, tokenized.tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--language", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(str(args.model_dir))
    if args.wav is None:
        pcm = np.zeros(16_000, dtype=np.float32)
    else:
        pcm = pcm_from_wav(args.wav)

    language = args.language or None
    mel, tokens = mel_from_pcm(processor, pcm, language)
    if args.json:
        payload = {
            "n_mels": int(mel.shape[0]),
            "n_frames": int(mel.shape[1]),
            "mel": mel.reshape(-1).tolist(),
            "tokens": tokens,
        }
        json.dump(payload, sys.stdout)
    else:
        flat = mel.reshape(-1)
        print(f"MEL {flat.size}", " ".join(f"{x:.9g}" for x in flat))
        print(f"TOKENS {len(tokens)}", " ".join(str(t) for t in tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
