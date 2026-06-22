#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Dump an OpenCLIP (BioCLIP-2 / ViT-L-14) reference for rlx parity tests.
#
# Produces $RLX_BIOCLIP2_FIXTURE/reference.json with, for a deterministic
# synthetic image and a fixed label set:
#   - pixel_values:  the post-preprocess [3*224*224] tensor (what
#                    model.encode_image receives) — fed directly into the
#                    rlx vision graph so resize/crop differences don't
#                    pollute the network-parity comparison.
#   - image_features: raw (unnormalized) [768] image embedding.
#   - texts / token_ids / text_features: per-label tokenization + raw [768].
#   - logit_scale and logits_per_image (normalized + scaled).
#
# Usage:
#   pip install open_clip_torch torch numpy pillow
#   RLX_BIOCLIP2_FIXTURE=/tmp/bioclip2_fixture \
#   RLX_BIOCLIP2_MODEL=weights/bioclip-2 \
#   python3 scripts/bioclip2_dump_reference.py

import json
import os

import numpy as np
import torch
from PIL import Image

import open_clip

FIXTURE = os.environ.get("RLX_BIOCLIP2_FIXTURE", "/tmp/bioclip2_fixture")
MODEL_DIR = os.environ.get("RLX_BIOCLIP2_MODEL", "weights/bioclip-2")
LABELS = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a black and white striped fish",
]


def synth_image(size=256):
    """Deterministic RGB gradient + checkerboard image."""
    yy, xx = np.mgrid[0:size, 0:size]
    r = (xx * 255 // size).astype(np.uint8)
    g = (yy * 255 // size).astype(np.uint8)
    b = (((xx // 16 + yy // 16) % 2) * 200 + 30).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def main():
    os.makedirs(FIXTURE, exist_ok=True)
    ckpt = os.path.join(MODEL_DIR, "open_clip_pytorch_model.bin")
    pretrained = ckpt if os.path.exists(ckpt) else "hf-hub:imageomics/bioclip-2"
    print(f"[ref] loading ViT-L-14 pretrained={pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=pretrained
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    img = synth_image()
    pixel = preprocess(img).unsqueeze(0)  # [1,3,224,224]

    tok = tokenizer(LABELS)  # [n,77]

    with torch.no_grad():
        image_features = model.encode_image(pixel)          # [1,768] raw
        text_features = model.encode_text(tok)              # [n,768] raw
        img_n = image_features / image_features.norm(dim=-1, keepdim=True)
        txt_n = text_features / text_features.norm(dim=-1, keepdim=True)
        scale = model.logit_scale.exp()
        logits = (scale * img_n @ txt_n.t())                # [1,n]

    ref = {
        "pixel_values": pixel[0].flatten().tolist(),
        "image_features": image_features[0].tolist(),
        "texts": LABELS,
        "token_ids": tok.tolist(),
        "text_features": text_features.tolist(),
        "logit_scale": float(model.logit_scale.item()),
        "logits_per_image": logits[0].tolist(),
    }
    out = os.path.join(FIXTURE, "reference.json")
    with open(out, "w") as f:
        json.dump(ref, f)
    print(f"[ref] wrote {out}")
    print(f"[ref] logits_per_image={ref['logits_per_image']}")


if __name__ == "__main__":
    main()
