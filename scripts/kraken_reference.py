#!/usr/bin/env python3
"""Kraken reference dumps for rlx-kraken parity (Kraken's own VGSL code).

Loads a recognition `.mlmodel` with `kraken.lib.vgsl.TorchVGSLModel`, runs
Kraken's line transform for baseline segmentations (`ImageInputTransforms`
with `valid_norm=False`: grayscale → Lanczos resize to the model height →
pad 16 px of white left and right → [0, 1] → `max − x`), the network, a
softmax, greedy CTC and the codec.

Writes raw little-endian f32 `.bin` files + `meta.json` into --out:
  input     the network input [C, H, W]
  logits    network output [classes, W'] (pre-softmax)
and the transcription in `meta.json`.

Needs Kraken itself (Python 3.10-3.13):
  uv venv -p python3.12 .venv-kraken && VIRTUAL_ENV=.venv-kraken uv pip install kraken==7.1.1
  .venv-kraken/bin/python scripts/kraken_reference.py \\
      --model weights/ocr/kraken-catmus-medieval/catmus-medieval.mlmodel \\
      --image line.png --out DIR
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
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
    p.add_argument("--out", required=True)
    p.add_argument("--padding", type=int, default=16)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from kraken.lib.ctc_decoder import greedy_decoder
    from kraken.lib.dataset.utils import ImageInputTransforms
    from kraken.lib.vgsl import TorchVGSLModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = TorchVGSLModel.load_model(a.model)
    m.eval()
    batch, channels, height, width = m.input
    ts = ImageInputTransforms(batch, height, width, channels, (a.padding, 0), valid_norm=False)
    x = ts(Image.open(a.image))
    logits, _ = m.nn(x[None])
    logits = logits[0, :, 0, :]  # [classes, W']
    probs = logits.softmax(0)
    labels = greedy_decoder(probs)[0]
    text = "".join(c for c, *_ in m.codec.decode(labels))

    meta = {
        "image": a.image,
        "spec": m.spec,
        "padding": a.padding,
        "input": save(a.out, "input", x),
        "logits": save(a.out, "logits", logits),
        "text": text,
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("input", "logits", "text")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
