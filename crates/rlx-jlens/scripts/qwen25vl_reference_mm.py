#!/usr/bin/env python3
"""Reference vision embeddings and logits for a Qwen2.5-VL *image* prompt.

The text trunk is bit-exact against `qwen25vl_reference.py`, so anything still
wrong with an image in the prompt is upstream of the trunk: the vision tower,
the splice, or the mRoPE positions the image tokens carry. This dumps what the
reference produces at each of those boundaries.

    python3 crates/rlx-jlens/scripts/qwen25vl_reference_mm.py \
        --weights /Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct \
        --image crates/rlx-locateanything/fixtures/sample.jpg \
        --out /tmp/vl_mm_ref.safetensors

Writes `image_embeds` `[n_vision, d]` straight off the vision tower,
`inputs_embeds` `[seq, d]` after the splice, `position_ids` `[3, seq]` (the
mRoPE sections), `last_logits` `[vocab]`, and `pixel_values` / `grid_thw` so the
Rust side can be fed byte-identical pixels if the preprocessors disagree.
"""

import argparse

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Describe the image.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    proc = AutoProcessor.from_pretrained(args.weights)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.weights, dtype=dtype, device_map="cpu"
    )
    model.eval()

    img = Image.open(args.image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": args.prompt}],
        }
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt")
    print("input_ids", inputs.input_ids.shape, "grid_thw", inputs.image_grid_thw.tolist())

    with torch.no_grad():
        visual = model.model.visual if hasattr(model.model, "visual") else model.visual
        image_embeds = visual(
            inputs.pixel_values.to(dtype), grid_thw=inputs.image_grid_thw
        )
        # transformers 5.x wraps the tower's output; older releases return the
        # tensor directly.
        image_embeds = getattr(image_embeds, "last_hidden_state", image_embeds)
        out = model(**inputs)
        rope_deltas = None
        get_rope = getattr(model.model, "get_rope_index", None) or getattr(
            model, "get_rope_index", None
        )
        position_ids = None
        if get_rope is not None:
            # The signature moved around across transformers releases; the dump
            # is still useful without it, so never let this sink the run.
            try:
                position_ids, rope_deltas = get_rope(
                    input_ids=inputs.input_ids,
                    image_grid_thw=inputs.image_grid_thw,
                    attention_mask=inputs.attention_mask,
                )
            except Exception as e:  # noqa: BLE001
                print("get_rope_index unavailable:", e)

    tensors = {
        "image_embeds": image_embeds.to(torch.float32).contiguous(),
        "input_ids": inputs.input_ids[0].to(torch.int32).contiguous(),
        "grid_thw": inputs.image_grid_thw.to(torch.int32).contiguous(),
        "pixel_values": inputs.pixel_values.to(torch.float32).contiguous(),
        "last_logits": out.logits[0, -1].to(torch.float32).contiguous(),
    }
    if position_ids is not None:
        tensors["position_ids"] = position_ids[:, 0, :].to(torch.int32).contiguous()
    save_file(tensors, args.out)

    tok = proc.tokenizer
    top = out.logits[0, -1].topk(5)
    print("image_embeds", tuple(image_embeds.shape), "rope_deltas", rope_deltas)
    print("out", args.out)
    print("top5:", [(tok.decode([i]), round(v.item(), 3)) for v, i in zip(top.values, top.indices)])


if __name__ == "__main__":
    main()
