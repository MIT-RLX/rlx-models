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

"""Dump HF prefill inputs_embeds for layer-0 isolation."""

from __future__ import annotations

import argparse
import json

import torch
from qwen_tts import Qwen3TTSModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    m = Qwen3TTSModel.from_pretrained(args.model_dir, torch_dtype=torch.float32, device_map="cpu")
    text = "Hi."
    ids = m._tokenize_texts([m._build_assistant_text(text)])[0]

    talker_dec = m.model.talker.model
    embeds_holder: list = []

    def pre_hook(_mod, _args, kwargs):
        ie = kwargs.get("inputs_embeds")
        if ie is not None and ie.shape[1] > 1:
            embeds_holder.append(ie.detach().clone())

    h = talker_dec.register_forward_pre_hook(pre_hook, with_kwargs=True)
    gen = m._merge_generate_kwargs(do_sample=False, subtalker_dosample=False, max_new_tokens=2)
    m.model.generate(
        input_ids=[ids],
        languages=["english"],
        speakers=["vivian"],
        non_streaming_mode=True,
        **gen,
    )
    h.remove()
    ie = embeds_holder[0]
    seq = ie.shape[1]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"seq": seq, "embeds": ie[0].reshape(-1).cpu().tolist()},
            f,
        )
    print(f"wrote seq={seq}")


if __name__ == "__main__":
    main()
