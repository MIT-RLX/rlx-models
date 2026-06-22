#!/usr/bin/env python3
"""HF Grounding DINO reference dump for parity bisection against rlx-grounding-dino."""
import sys, json
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

REPO = "IDEA-Research/grounding-dino-base"
img_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gdino_cats.jpg"
prompt = sys.argv[2] if len(sys.argv) > 2 else "a cat. a remote control."

proc = AutoProcessor.from_pretrained(REPO)
model = AutoModelForZeroShotObjectDetection.from_pretrained(REPO).eval()

image = Image.open(img_path).convert("RGB")
inputs = proc(images=image, text=prompt, return_tensors="pt")
print("== tokenizer ==")
print("input_ids:", inputs["input_ids"][0].tolist())
print("attention_mask:", inputs["attention_mask"][0].tolist())
print("token_type_ids:", inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"]))[0].tolist())
print("pixel_values shape:", tuple(inputs["pixel_values"].shape))
print("pixel_values[0,:,0,0]:", inputs["pixel_values"][0, :, 0, 0].tolist())

with torch.no_grad():
    out = model(**inputs, output_hidden_states=False)

logits = out.logits[0]            # [num_queries, num_text_tokens]
boxes = out.pred_boxes[0]         # [num_queries, 4] cxcywh normalized
scores = logits.sigmoid()
maxq = scores.max(dim=1).values   # per-query best token score
top = torch.topk(maxq, 5)
print("== final ==")
print("logits shape:", tuple(logits.shape), "boxes shape:", tuple(boxes.shape))
print("top5 query max-sigmoid scores:", [round(v, 4) for v in top.values.tolist()])
for i in top.indices.tolist():
    print(f"  q{i}: score={maxq[i].item():.4f} box(cxcywh)={[round(x,3) for x in boxes[i].tolist()]}")

results = proc.post_process_grounded_object_detection(
    out, inputs["input_ids"], threshold=0.3, text_threshold=0.25,
    target_sizes=[image.size[::-1]],
)[0]
print("== detections (thr 0.3/0.25) ==")
for s, l, b in zip(results["scores"].tolist(), results["labels"], results["boxes"].tolist()):
    print(f"  {l}: score={s:.3f} box={[round(x,1) for x in b]}")

# Save text features for intermediate comparison.
np.save("/tmp/gdino_hf_input_ids.npy", inputs["input_ids"][0].numpy())
print("saved input_ids to /tmp/gdino_hf_input_ids.npy")
