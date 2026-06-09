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

"""Dump layer-0 Q/K (head 0, last token) after RoPE for eager parity."""

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
    ie = embeds_holder[0]
    seq = ie.shape[1]
    layer0 = talker_dec.layers[0]
    attn = layer0.self_attn

    with torch.no_grad():
        pos = torch.arange(seq, device=ie.device).view(1, 1, -1).expand(3, 1, -1).float()
        pe = talker_dec.rotary_emb(ie, pos)
        h = layer0.input_layernorm(ie)
        hidden_shape = (*h.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(h).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(h).view(hidden_shape)).transpose(1, 2)
        cos, sin = pe
        from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb

        q, k = apply_multimodal_rotary_pos_emb(
            q, k, cos, sin, attn.rope_scaling["mrope_section"], attn.rope_scaling["interleaved"]
        )
        q_last = q[0, 0, -1, :16].tolist()
        k_last = k[0, 0, -1, :16].tolist()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"q_head0_last16": q_last, "k_head0_last16": k_last}, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
