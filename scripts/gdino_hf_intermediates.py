#!/usr/bin/env python3
"""Dump HF Grounding DINO intermediate-stage magnitude stats for parity bisection."""
import sys
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

def stat(name, t):
    if not isinstance(t, torch.Tensor):
        return
    v = t.detach().float().flatten()
    finite = v[torch.isfinite(v)]
    head = [round(x, 3) for x in v[:5].tolist()]
    print(f"DBG {name}: shape={tuple(t.shape)} mean={finite.mean():.4f} std={finite.std():.4f} "
          f"min={finite.min():.4f} max={finite.max():.4f} head={head}")

captured = {}
def hook(name):
    def fn(mod, inp, out):
        o = out
        if isinstance(o, (tuple, list)):
            o = next((x for x in o if isinstance(x, torch.Tensor)), None)
        if isinstance(o, torch.Tensor):
            captured[name] = o.detach()
    return fn

m = model.model
hooks = [
    m.backbone.register_forward_hook(hook("backbone_out")),
    m.text_backbone.register_forward_hook(hook("text_backbone_out")),
    m.encoder.register_forward_hook(hook("encoder_out")),
    m.decoder.register_forward_hook(hook("decoder_out")),
]
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)
for h in hooks:
    h.remove()

stat("pixel_values", inputs["pixel_values"])
for k in ("backbone_out", "text_backbone_out", "encoder_out", "decoder_out"):
    if k in captured:
        stat(k, captured[k])
# Output dataclass fields
for fld in ("encoder_last_hidden_state_vision", "encoder_last_hidden_state_text",
            "last_hidden_state", "init_reference_points"):
    t = getattr(out, fld, None)
    if isinstance(t, torch.Tensor):
        stat(fld, t)
stat("final_logits", out.logits)
stat("final_boxes", out.pred_boxes)
print("logits.sigmoid().max():", out.logits.sigmoid().max().item())
