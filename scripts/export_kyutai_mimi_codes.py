#!/usr/bin/env python3
"""Export Moshi TTS Mimi code frames for Rust parity tests.

Writes JSON to /tmp/py_mimi_codes.json (override with RLX_KYUTAI_MIMI_CODES_REF).

Usage:
    .venv-kyutai-moshi/bin/python3 scripts/export_kyutai_mimi_codes.py
    RLX_KYUTAI_MIMI_CODES_REF=/path/out.json scripts/export_kyutai_mimi_codes.py
"""

from __future__ import annotations

import json
import os
import sys

import torch

torch.manual_seed(42)

import moshi.utils.compile as muc

muc.torch.compile = lambda fn, **kw: fn

from moshi.models import loaders
from moshi.models.tts import TTSModel

PROMPT = "Hello world, this is a test of the Kyutai text to speech system."
VOICE = "alba-mackenna/casual.wav"
OUT = os.environ.get("RLX_KYUTAI_MIMI_CODES_REF", "/tmp/py_mimi_codes.json")


def main() -> None:
    info = loaders.CheckpointInfo.from_hf_repo("kyutai/tts-1.6b-en_fr")
    tts = TTSModel.from_checkpoint_info(info, n_q=32, temp=0.0, device="cpu")
    entries = [tts.prepare_script([PROMPT], padding_between=1)]
    voice = tts.get_voice_path(VOICE)
    attrs = [tts.make_condition_attributes([voice], cfg_coef=2.0)]
    result = tts.generate(entries, attrs)
    delay = tts.delay_steps
    end = result.end_steps[0]
    trimmed = []
    for i, frame in enumerate(result.frames):
        if i < delay + 2:
            continue
        idx = i - (delay + 2)
        if end is not None and idx >= end:
            break
        trimmed.append([int(x) for x in frame[0, 1:, 0].tolist()])
    payload = {"delay": delay, "end": end, "trimmed": trimmed}
    with open(OUT, "w") as f:
        json.dump(payload, f)
    print(f"exported {len(trimmed)} frames to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
