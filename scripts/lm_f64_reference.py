#!/usr/bin/env python3
"""Float64 logits from transformers, for `examples/lm_f64_check.rs`.

    python3 scripts/lm_f64_reference.py DIR 785,6722,315,9625,374 out.f64

Writes raw little-endian f64 logits `[seq, vocab]` for the given token ids.
The model is loaded and run in float64 with eager attention.

`torch_dtype=float64` alone is not an f64 computation: transformers' norms
upcast with `.float()` / `.to(torch.float32)` (a downcast from f64), eager
attention runs `softmax(..., dtype=torch.float32)`, and the rotary tables are
built from a float32 `inv_freq`. All of these are redirected to float64 here,
so every step runs in f64.
"""

import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def force_float64() -> None:
    to = torch.Tensor.to

    def to64(self, *args, **kwargs):
        args = tuple(torch.float64 if a is torch.float32 else a for a in args)
        if kwargs.get("dtype") is torch.float32:
            kwargs["dtype"] = torch.float64
        return to(self, *args, **kwargs)

    def float64(self, *args, **kwargs):
        # Straight to f64: `flt(self).double()` would round through f32.
        return to(self, torch.float64)

    def softmax64(x, *args, dtype=None, **kwargs):
        if dtype is torch.float32:
            dtype = torch.float64
        return softmax(x, *args, dtype=dtype, **kwargs)

    softmax = torch.nn.functional.softmax
    torch.Tensor.to = to64
    torch.Tensor.float = float64
    torch.nn.functional.softmax = softmax64


def main() -> None:
    force_float64()
    model_dir, ids, out = sys.argv[1], sys.argv[2], sys.argv[3]
    ids = [int(t) for t in ids.split(",")]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float64, attn_implementation=os.environ.get("ATTN", "eager")
    )
    model.eval()
    # Buffers built at init (rotary `inv_freq`) predate the patch's effect on
    # `from_pretrained`'s own casts; make them f64 too.
    for name, buf in model.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.double()
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
    np.asarray(logits, dtype="<f8").tofile(out)
    print(f"{out}: {tuple(logits.shape)}, max |logit| {logits.abs().max().item():.2f}")


if __name__ == "__main__":
    main()
