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

"""HF reference for Qwen3-TTS parity (greedy CustomVoice).

Protocol:
  META text=... speaker=... language=...
  CODEC_FRAMES <n_frames> <16*n ints>
  PCM <n> <floats...>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--text", default="Hi.")
    p.add_argument("--speaker", default="vivian")
    p.add_argument("--language", default="english")
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    m = Qwen3TTSModel.from_pretrained(args.model_dir, torch_dtype=torch.float32, device_map="cpu")
    wavs, sr = m.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        do_sample=False,
        subtalker_dosample=False,
        max_new_tokens=args.max_new_tokens,
    )
    text = args.text
    print(f"META text={text!r} speaker={args.speaker!r} language={args.language!r} sr={sr}")

    input_ids = m._tokenize_texts([m._build_assistant_text(text)])[0]
    gen = m._merge_generate_kwargs(do_sample=False, subtalker_dosample=False, max_new_tokens=args.max_new_tokens)
    codes_list, _ = m.model.generate(
        input_ids=[input_ids],
        languages=[args.language],
        speakers=[args.speaker],
        non_streaming_mode=True,
        **gen,
    )
    codes = codes_list[0].cpu().numpy().astype(np.int64)
    flat = codes.reshape(-1).tolist()
    print(f"CODEC_FRAMES {codes.shape[0]} {' '.join(str(int(x)) for x in flat)}")
    pcm = wavs[0].astype(np.float32)
    print(f"PCM {pcm.size} {' '.join(f'{x:.8g}' for x in pcm.tolist())}")


if __name__ == "__main__":
    main()
