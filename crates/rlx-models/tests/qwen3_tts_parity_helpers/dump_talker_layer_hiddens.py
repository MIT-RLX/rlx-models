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

"""Dump talker prefill last-token hidden after each decoder layer (HF reference)."""

from __future__ import annotations

import argparse
import json
import sys

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
    embeds_holder: list[torch.Tensor] = []

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
    if not embeds_holder:
        print("no prefill embeds captured", file=sys.stderr)
        sys.exit(1)

    ie = embeds_holder[0]
    seq = ie.shape[1]
    attn = torch.ones(1, seq, dtype=torch.long)

    layer_last: list[list[float]] = []

    def layer_hook(_mod, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        layer_last.append(hs[0, -1].detach().cpu().tolist())

    hooks = [layer.register_forward_hook(layer_hook) for layer in talker_dec.layers]
    norm_hook = talker_dec.norm.register_forward_hook(layer_hook)

    with torch.no_grad():
        talker_dec(inputs_embeds=ie, attention_mask=attn, use_cache=False)

    for hk in hooks:
        hk.remove()
    norm_hook.remove()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seq": seq, "layer_last": layer_last}, f)
    print(f"wrote {len(layer_last)} rows (layers+norm) seq={seq}")


if __name__ == "__main__":
    main()
