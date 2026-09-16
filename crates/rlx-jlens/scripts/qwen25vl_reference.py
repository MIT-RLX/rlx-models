#!/usr/bin/env python3
"""Per-layer reference hidden states for a Qwen2.5-VL *text-only* prompt.

The VL trunk's real-weight forward is degenerate in rlx on every backend, which
rules out a kernel bug and points at the model definition. This dumps what the
reference produces at each layer boundary so the first divergent layer can be
found by bisection, using the smallest prompt that shows the fault.

    python3 crates/rlx-jlens/scripts/qwen25vl_reference.py \
        --weights /Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct \
        --out /tmp/vl_ref.safetensors

Writes `hidden.{l}` for `l` in `0..=n_layers` (l=0 is the embedding output,
l=k is what leaves block k-1), plus `logits`, all f32.
"""

import argparse

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.weights)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.weights, dtype=dtype, device_map="cpu"
    )
    model.eval()

    ids = tok(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids
    print("ids", ids.tolist())

    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)

    tensors = {}
    for i, h in enumerate(out.hidden_states):
        tensors[f"hidden.{i}"] = h[0].to(torch.float32).contiguous()
    tensors["logits"] = out.logits[0].to(torch.float32).contiguous()
    tensors["ids"] = ids[0].to(torch.int32).contiguous()

    save_file(tensors, args.out)
    top = out.logits[0, -1].topk(5)
    print("out", args.out, "layers", len(out.hidden_states))
    print("top5:", [(tok.decode([i]), round(v.item(), 3)) for v, i in zip(top.values, top.indices)])


if __name__ == "__main__":
    main()
