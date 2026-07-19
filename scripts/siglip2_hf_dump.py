#!/usr/bin/env python3
# RLX — SigLIP 2 reference dump for rlx-siglip2 parity tests.
#
# Runs the HuggingFace fixed-resolution SigLIP 2 model on a deterministic
# synthetic image + a few prompts, and writes reference.json:
#   pixel_values      flat [3*H*W]      (SigLIP-normalized, slow/PIL processor)
#   token_ids         [[u32; ctx]; N]   (processor, padding=max_length=64)
#   image_features    flat [D]          (get_image_features → image_embeds, un-normalized)
#   text_features     [[f32; D]; N]     (get_text_features → text_embeds, un-normalized)
#   logits_per_image  flat [N]          (model(...).logits_per_image row 0)
#
# Usage:
#   python scripts/siglip2_hf_dump.py \
#       --model weights/siglip2-base-224 --out weights/siglip2-base-224/fixture
import argparse, json, os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

PROMPTS = ["a photo of a cat", "a photo of a dog", "2 cats", "a green field"]


def synthetic_image(h=256, w=256):
    # Must match the tests' synthetic image byte-for-byte.
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            rgb[y, x, 0] = (x * 255 // w) & 0xFF
            rgb[y, x, 1] = (y * 255 // h) & 0xFF
            rgb[y, x, 2] = (((x // 16 + y // 16) % 2) * 200 + 30) & 0xFF
    return Image.fromarray(rgb, "RGB")


def is_naflex(model_dir):
    import json
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    return cfg.get("model_type") == "siglip2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    naflex = is_naflex(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    # Fixed-res: slow (PIL) processor to match rlx's PIL-faithful resampler.
    # NaFlex: the fast processor is the only one (pixel_values are fed to the
    # model directly, so the encoder parity is isolated from preprocessing).
    proc = AutoProcessor.from_pretrained(args.model, use_fast=naflex)

    # A non-square image for NaFlex exercises the position-embedding resize.
    img = synthetic_image(224, 320) if naflex else synthetic_image(256, 256)
    enc = proc(
        text=PROMPTS,
        images=[img],
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**enc)
        # Raw (un-normalized) pooler outputs — what rlx encode_* returns.
        # transformers 5.x normalizes out.image_embeds/out.text_embeds, so
        # read the sub-models' pooler_output directly for magnitude parity.
        if naflex:
            vout = model.vision_model(
                enc["pixel_values"],
                attention_mask=enc["pixel_attention_mask"],
                spatial_shapes=enc["spatial_shapes"],
            )
        else:
            vout = model.vision_model(enc["pixel_values"])
        image_embeds = vout.pooler_output.detach().cpu().float().numpy()
        text_embeds = model.text_model(enc["input_ids"]).pooler_output
        text_embeds = text_embeds.detach().cpu().float().numpy()
        logits_per_image = out.logits_per_image.detach().cpu().float().numpy()

    ref = {
        "pixel_values": enc["pixel_values"][0].flatten().tolist(),
        "token_ids": enc["input_ids"].to(torch.int64).tolist(),
        "image_features": image_embeds[0].flatten().tolist(),
        "text_features": [row.tolist() for row in text_embeds],
        "logits_per_image": logits_per_image[0].tolist(),
        "logit_scale": float(model.logit_scale.detach().item()),
        "logit_bias": float(model.logit_bias.detach().item()),
        "prompts": PROMPTS,
        "naflex": naflex,
    }
    if naflex:
        ss = enc["spatial_shapes"][0].tolist()
        ref["spatial_shapes"] = [int(ss[0]), int(ss[1])]  # (nph, npw)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "reference.json")
    with open(path, "w") as f:
        json.dump(ref, f)
    print("wrote", path)
    print("  image_embeds dim", len(ref["image_features"]))
    print("  logits_per_image", [round(x, 3) for x in ref["logits_per_image"]])
    print("  logit_scale", ref["logit_scale"], "logit_bias", ref["logit_bias"])


if __name__ == "__main__":
    main()
