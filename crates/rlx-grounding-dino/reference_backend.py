#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# Reference dumper for Grounding DINO parity testing. Runs the HF
# `transformers` GroundingDinoForObjectDetection and writes per-stage tensors
# plus the final detections to a directory for the Rust tests to compare.
#
# Usage:
#   python reference_backend.py --image path.jpg --text "a cat. a remote control." \
#       --out /tmp/gdino_ref [--model IDEA-Research/grounding-dino-base]
#
# Requires: torch, transformers>=4.40, pillow, numpy.

import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    os.makedirs(args.out, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model).eval()

    image = Image.open(args.image).convert("RGB")
    text = args.text if args.text.strip().endswith(".") else args.text.strip() + "."
    inputs = processor(images=image, text=text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Final boxes/logits.
    np.save(os.path.join(args.out, "logits.npy"), outputs.logits[0].float().numpy())
    np.save(os.path.join(args.out, "pred_boxes.npy"), outputs.pred_boxes[0].float().numpy())

    # Post-processed detections.
    # transformers >=5 renamed `box_threshold` -> `threshold`.
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    raw_labels = results.get("labels", results.get("text_labels", results.get("text", [])))
    dets = {
        "scores": results["scores"].tolist(),
        "boxes": results["boxes"].tolist(),
        "labels": list(raw_labels),
    }
    # `labels` may be a list of strings (phrases) depending on transformers version.
    if not isinstance(dets["labels"], list):
        dets["labels"] = list(dets["labels"])
    with open(os.path.join(args.out, "detections.json"), "w") as f:
        json.dump(dets, f, indent=2)

    # Tokenization for the Rust side.
    with open(os.path.join(args.out, "inputs.json"), "w") as f:
        json.dump(
            {
                "input_ids": inputs.input_ids[0].tolist(),
                "attention_mask": inputs.attention_mask[0].tolist(),
                "image_size": list(image.size),  # (w, h)
                "text": text,
            },
            f,
            indent=2,
        )
    print(f"[reference_backend] wrote outputs to {args.out}")
    print(f"  detections: {len(dets['scores'])}")


if __name__ == "__main__":
    main()
