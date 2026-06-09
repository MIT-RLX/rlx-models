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

"""HF reference CustomVoice synthesis (MPS/CUDA/CPU). Use for finetuned checkpoints until native parity matches."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-TTS CustomVoice via HuggingFace qwen_tts")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--text", required=True)
    p.add_argument("--speaker", required=True)
    p.add_argument("--language", default="english")
    p.add_argument("--out-wav", required=True, type=Path)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = p.parse_args()

    src = Path(__file__).resolve().parents[1] / ".cache/qwen3-tts/Qwen3-TTS-src"
    if src.is_dir():
        sys.path.insert(0, str(src))

    import torch
    from qwen_tts import Qwen3TTSModel

    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    model = Qwen3TTSModel.from_pretrained(
        str(args.model_dir),
        device_map=args.device,
        attn_implementation="sdpa",
    )
    wavs, sr = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        subtalker_dosample=False,
    )
    pcm = np.asarray(wavs[0], dtype=np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    args.out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out_wav), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm16.tobytes())
    rms = float(np.sqrt((pcm**2).mean()))
    print(f"wrote {args.out_wav} ({len(pcm)} samples @ {sr} Hz, rms={rms:.4f})")


if __name__ == "__main__":
    main()
