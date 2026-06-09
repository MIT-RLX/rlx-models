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

"""Run Qwen3-TTS finetuning/sft_12hz.py with sdpa (MPS-safe) instead of flash_attention_2."""

from __future__ import annotations

import os
import runpy
import sys

import qwen_tts.inference.qwen3_tts_model as m

_orig = m.Qwen3TTSModel.from_pretrained


def _from_pretrained(*args, **kwargs):
    if kwargs.get("attn_implementation") == "flash_attention_2":
        kwargs["attn_implementation"] = "sdpa"
    return _orig(*args, **kwargs)


m.Qwen3TTSModel.from_pretrained = _from_pretrained

def _patch_sft_text_projection(src: str) -> str:
    """0.6B talker uses text_hidden=2048 → project to hidden=1024 before adding codec embeds."""
    old = (
        "input_text_embedding = model.talker.model.text_embedding(input_text_ids) "
        "* text_embedding_mask"
    )
    new = (
        "input_text_embedding = model.talker.text_projection("
        "model.talker.model.text_embedding(input_text_ids)) * text_embedding_mask"
    )
    if old not in src:
        if "text_projection" in src:
            return src
        raise SystemExit("sft_12hz.py changed — update qwen3_tts_sft_mps.py patch")
    return src.replace(old, new, 1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: qwen3_tts_sft_mps.py /path/to/sft_12hz.py [args...]")
    script = os.path.abspath(sys.argv[1])
    finetune_dir = os.path.dirname(script)
    os.chdir(finetune_dir)
    sys.path.insert(0, finetune_dir)
    patched = os.path.join(finetune_dir, ".sft_12hz_rlx_patched.py")
    src_text = open(script, encoding="utf-8").read()
    open(patched, "w", encoding="utf-8").write(_patch_sft_text_projection(src_text))
    sys.argv = [patched, *sys.argv[2:]]
    runpy.run_path(patched, run_name="__main__")
