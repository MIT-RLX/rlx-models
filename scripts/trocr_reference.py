#!/usr/bin/env python3
"""TrOCR reference dumps for rlx-trocr parity (transformers' own model).

Runs `VisionEncoderDecoderModel` + `TrOCRProcessor` on one line image and
writes raw little-endian f32 `.bin` files + `meta.json` into --out:
  pixels    pixel_values [3, 384, 384]
  encoder   encoder last_hidden_state [S, d_enc]
  logits    decoder logits [T, vocab], teacher-forced on the greedy output
and the greedy token ids + transcription in `meta.json`.

  uv venv -p python3.12 .venv-trocr
  VIRTUAL_ENV=.venv-trocr uv pip install torch transformers==5.3.0 sentencepiece pillow
  .venv-trocr/bin/python scripts/trocr_reference.py \\
      --model weights/ocr/trocr-small-handwritten --image line.jpg --out DIR
"""

import argparse
import json
import os

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
    p.add_argument("--max-new-tokens", type=int, default=64)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    proc = TrOCRProcessor.from_pretrained(a.model)
    model = VisionEncoderDecoderModel.from_pretrained(a.model).eval().float()
    img = Image.open(a.image).convert("RGB")
    px = proc(images=img, return_tensors="pt").pixel_values
    enc = model.encoder(pixel_values=px).last_hidden_state
    ids = model.generate(px, max_new_tokens=a.max_new_tokens, num_beams=1, do_sample=False)
    ids = ids[0].tolist()
    logits = model(pixel_values=px, decoder_input_ids=torch.tensor([ids[:-1]])).logits[0]
    text = proc.batch_decode([ids], skip_special_tokens=True)[0]

    meta = {
        "image": a.image,
        "processor": type(proc.image_processor).__name__,
        "pixels": save(a.out, "pixels", px[0]),
        "encoder": save(a.out, "encoder", enc[0]),
        "logits": save(a.out, "logits", logits),
        "ids": ids,
        "text": text,
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("processor", "ids", "text")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
