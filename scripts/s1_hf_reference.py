#!/usr/bin/env python3
"""HF reference dump for S1-mini (superwhisper/s1-mini).

Produces the ground truth `rlx-s1` is checked against:

  * the prompt string HF's `apply_chat_template(..., enable_thinking=False)`
    builds, so `rlx_s1::render_prompt` can be compared byte-for-byte;
  * its token ids, so the Rust tokenizer bridge can be compared id-for-id;
  * the greedy completion for each case, so decode can be compared token-for-token.

Usage:
    python3 scripts/s1_hf_reference.py \
        --weights /Volumes/FOUR/weights/lm/s1-mini \
        --out /tmp/s1_reference.json
    # prompts + ids only (no forward pass):
    python3 scripts/s1_hf_reference.py --weights ... --out ... --no-generate
"""

from __future__ import annotations

import argparse
import json
import sys

# Exact system prompt from the model card. Re-wording it changes the model's
# behavior, so it is duplicated verbatim in `crates/rlx-s1/src/prompt.rs`.
SYSTEM = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)

# The model card's worked examples, all under the default control line, plus a
# few that exercise the other axes.
CASES = [
    ("so um i need to like send the the report by uh friday no wait make that thursday",
     "semi-formal", "prose", "general"),
    ("i think the answer is forty two no sorry forty three", "semi-formal", "prose", "general"),
    ("let's meet at half past two tomorrow uh actually make it three fifteen p m",
     "semi-formal", "prose", "general"),
    ("the invoice came to twenty three thousand four hundred and fifty dollars and it's due on "
     "march third twenty twenty six", "semi-formal", "prose", "general"),
    ("send it to support at superwhisper dot com", "semi-formal", "prose", "general"),
    ("um", "semi-formal", "prose", "general"),
    ("hmm im gonna be late theres a cute dog outside i cant just walk past him",
     "casual", "prose", "general"),
    ("hmm im gonna be late theres a cute dog outside i cant just walk past him",
     "semi-casual", "prose", "general"),
    ("hmm im gonna be late theres a cute dog outside i cant just walk past him",
     "formal", "prose", "general"),
    ("so for the trip we need to pack sunscreen and then also a first aid kit and um chargers "
     "for everything", "semi-formal", "lists", "general"),
    ("hey sarah just wanted to follow up on the proposal can you send the numbers by end of "
     "week thanks john", "semi-formal", "prose", "email"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="s1-mini checkpoint directory")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--no-generate", action="store_true", help="prompts + ids only")
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="0 = the card's 1.3*prompt+32 heuristic")
    # rlx computes in f32, so f32 is the right control for a token-identity
    # gate. `auto` picks the checkpoint's bf16 and will flip near-tie argmaxes.
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16", "auto"])
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.weights)
    model = None
    if not args.no_generate:
        import torch
        from transformers import AutoModelForCausalLM

        dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
        model = AutoModelForCausalLM.from_pretrained(args.weights, dtype=dtype)
        model.eval()
        torch.manual_seed(0)

    out = {"system": SYSTEM, "dtype": args.dtype, "cases": []}
    for transcript, styling, structure, context in CASES:
        control = f"[Styling: {styling}] [Structure: {structure}] [Context: {context}]"
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{control}\n{transcript}"},
        ]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(prompt, return_tensors=None)["input_ids"]
        rec = {
            "transcript": transcript,
            "styling": styling,
            "structure": structure,
            "context": context,
            "prompt": prompt,
            "prompt_ids": list(map(int, ids)),
        }
        if model is not None:
            import torch

            n_new = args.max_new_tokens or (int(len(ids) * 1.3 + 0.999) + 32)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=n_new,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            new_ids = gen[0][enc["input_ids"].shape[1]:].tolist()
            rec["output_ids"] = list(map(int, new_ids))
            rec["output"] = tok.decode(new_ids, skip_special_tokens=True)
            print(f"{control}\n  in : {transcript}\n  out: {rec['output']!r}", flush=True)
        out["cases"].append(rec)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out} ({len(out['cases'])} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
