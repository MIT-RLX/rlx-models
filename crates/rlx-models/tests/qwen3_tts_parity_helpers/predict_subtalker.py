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

"""Greedy code-predictor (subtalker) for one talker frame — matches HF `code_predictor.generate`."""

from __future__ import annotations

import argparse
import json
import sys

import torch
from qwen_tts import Qwen3TTSModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--past-hidden", required=True, help="JSON float array [hidden]")
    p.add_argument("--group0", type=int, required=True)
    args = p.parse_args()

    past = torch.tensor(json.loads(args.past_hidden), dtype=torch.float32).view(1, 1, -1)
    g0 = torch.tensor([[args.group0]], dtype=torch.long)

    m = Qwen3TTSModel.from_pretrained(args.model_dir, dtype=torch.float32, device_map="cpu")
    talker = m.model.talker
    cp = talker.code_predictor
    n_groups = m.model.config.talker_config.num_code_groups
    last_id = talker.get_input_embeddings()(g0)
    out = cp.generate(
        inputs_embeds=torch.cat((past, last_id), dim=1),
        max_new_tokens=n_groups - 1,
        do_sample=False,
        top_k=1,
        top_p=1.0,
        temperature=0.0,
        return_dict_in_generate=False,
    )
    seq = torch.cat((g0, out), dim=-1)[0].tolist()
    print(json.dumps(seq))


if __name__ == "__main__":
    main()
