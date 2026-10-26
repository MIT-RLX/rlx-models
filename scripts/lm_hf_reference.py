#!/usr/bin/env python3
"""Greedy reference for a small HF causal LM, in f32.

Writes JSON with the prompt ids, the greedy continuation and the
last-prompt-position logits, for the env-gated parity tests:

    python3 scripts/lm_hf_reference.py weights/lm/lfm2.5-230m /tmp/lfm25-230m.ref.json
    RLX_LFM_HF_DIR=weights/lm/lfm2.5-230m RLX_LFM_HF_REF=/tmp/lfm25-230m.ref.json \
        cargo test -p rlx-lfm --release --test hf_parity_live -- --nocapture

The reference must be f32: a bf16 reference flips near-tie greedy tokens.
Even in f32, transformers' cached `generate` and its full forward can disagree
on a near-tie, so `margins` records each step's top-2 gap for the test to use.
"""

import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

PROMPT = "The history of the printing press begins in"
N_NEW = 24


def main() -> None:
    model_dir, out = sys.argv[1], sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else PROMPT
    if prompt.replace(",", "").isdigit():
        ids = [int(x) for x in prompt.split(",")]
    else:
        ids = AutoTokenizer.from_pretrained(model_dir)(prompt)["input_ids"]
    # Multimodal checkpoints (Qwen3.5) nest the LM under `text_config` and only
    # load through the image-text class; given input_ids alone it runs text-only.
    has_text_config = hasattr(AutoConfig.from_pretrained(model_dir), "text_config")
    cls = AutoModelForImageTextToText if has_text_config else AutoModelForCausalLM
    model = cls.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x).logits[0, -1].tolist()
        gen = model.generate(x, max_new_tokens=N_NEW, do_sample=False)[0, len(ids):].tolist()
        # Top-1 minus top-2 at every greedy step, teacher-forced in one forward:
        # a step whose margin is within float noise may legitimately flip.
        full = model(torch.tensor([ids + gen])).logits[0, len(ids) - 1 : -1]
        top2 = full.topk(2, dim=-1).values
        margins = (top2[:, 0] - top2[:, 1]).tolist()
    json.dump(
        {"prompt_ids": ids, "greedy": gen, "margins": margins, "last_logits": logits},
        open(out, "w"),
    )
    print(f"wrote {out}: greedy={gen}")


if __name__ == "__main__":
    main()
