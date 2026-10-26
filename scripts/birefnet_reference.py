#!/usr/bin/env python3
"""BiRefNet reference dumps for rlx-birefnet parity (the model's own code via transformers).

Writes raw little-endian f32 `.bin` files + `meta.json` into --out:
  input      the normalized network input [3, S, S]
  bb{1..4}   backbone stage outputs for the full-resolution pass [C, h, w]
  enc{1..4}  encoder features after the two-scale concat (+ context on enc4)
  sq4        after the squeeze block
  logits     decoder output [1, S, S] (pre-sigmoid)

  python3 scripts/birefnet_reference.py --model weights/image/birefnet/BiRefNet_lite \\
      --image IMG --size 512 --out DIR
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def save(out, name, t):
    a = t.detach().float().cpu().numpy().astype(np.float32)
    a.tofile(os.path.join(out, name + ".bin"))
    return list(a.shape)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoModelForImageSegmentation

    m = AutoModelForImageSegmentation.from_pretrained(a.model, trust_remote_code=True).eval().float()
    img = Image.open(a.image).convert("RGB").resize((a.size, a.size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = ((x - mean) / std)[None]

    meta = {"size": a.size, "image": a.image, "input": save(a.out, "input", x[0])}
    bb = m.bb(x)
    for i, f in enumerate(bb):
        meta[f"bb{i + 1}"] = save(a.out, f"bb{i + 1}", f[0])
    (x1, x2, x3, x4), _ = m.forward_enc(x)
    for i, f in enumerate((x1, x2, x3, x4)):
        meta[f"enc{i + 1}"] = save(a.out, f"enc{i + 1}", f[0])
    sq = m.squeeze_module(x4) if m.config.squeeze_block else x4
    meta["sq4"] = save(a.out, "sq4", sq[0])
    logits = m.decoder([x, x1, x2, x3, sq])[-1]
    meta["logits"] = save(a.out, "logits", logits[0])
    full = m(x)[-1]
    meta["full_vs_staged_max"] = float((full - logits).abs().max())
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
