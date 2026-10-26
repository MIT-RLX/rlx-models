#!/usr/bin/env python3
"""Reference dump for `rlx-minicpm-v` parity: HF transformers MiniCPM-V 4.6 / 4.7.

Runs one image + question through the HF processor and model in float32 and
writes every intermediate the Rust side checks against:

  <out>.safetensors
    pixel_values     [1, 3, 14, L]   processor output (NaViT-packed crops)
    target_sizes     [n_crops, 2]    patch grid per crop (thumbnail first)
    input_ids        [S]
    mm_token_type_ids[S]             0 text, 1 image
    position_ids     [3, S]          canvas M-RoPE (4.7) / arange (4.6)
    vit_merged.{i}   [h*w/4, 1152]   vision tower after the window merger (16x)
    vision_hidden.{i}[n, 1152]       vision tower output (post_layernorm)
    features.{i}     [n, hidden]     merger output per crop, in LM space
    last_logits      [vocab]         prefill logits at the last prompt token
  <out>.json         prompt text, greedy ids, top-2 margins, config echo

Usage:
  .venv-minicpmv/bin/python -I scripts/minicpmv_reference.py \
      weights/vision/minicpm-v-4.7-1b crates/rlx-locateanything/fixtures/sample.jpg \
      /tmp/minicpmv47 --question "Describe this image." --n-new 24

The model must run in float32: a bf16 reference flips near-tie greedy steps.
"""

import argparse
import json

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--question", default="Describe this image in one sentence.")
    ap.add_argument("--n-new", type=int, default=24)
    ap.add_argument("--max-slice-nums", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    image = Image.open(args.image).convert("RGB")
    # JPEG decoders disagree by a few levels (PIL's libjpeg vs Rust's `image`),
    # so hand the Rust side PIL's decoded pixels losslessly.
    decoded = args.out + ".png"
    image.save(decoded)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": args.question}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    proc_kwargs = {}
    if args.max_slice_nums is not None:
        proc_kwargs["max_slice_nums"] = args.max_slice_nums
    inputs = processor(images=[image], text=[prompt], return_tensors="pt", **proc_kwargs)

    tensors = {
        "pixel_values": inputs["pixel_values"].float().contiguous(),
        "target_sizes": inputs["target_sizes"].to(torch.int32).contiguous(),
        "input_ids": inputs["input_ids"][0].to(torch.int64).contiguous(),
    }
    mm_types = inputs.get("mm_token_type_ids")
    if mm_types is None:
        mm_types = (inputs["input_ids"] == model.config.image_token_id).to(torch.int64)
    tensors["mm_token_type_ids"] = mm_types[0].to(torch.int64).contiguous()

    inner = model.model
    with torch.no_grad():
        if hasattr(inner, "get_rope_index"):
            pos, _ = inner.get_rope_index(
                inputs["input_ids"],
                mm_token_type_ids=mm_types,
                target_sizes=inputs["target_sizes"],
            )
            tensors["position_ids"] = pos[:, 0, :].to(torch.int64).contiguous()
        else:
            s = inputs["input_ids"].shape[1]
            tensors["position_ids"] = torch.arange(s).expand(3, s).contiguous()

        # Window-merger output, captured by hook (16x mode only).
        merged = []
        hook = inner.vision_tower.vit_merger.register_forward_hook(
            lambda _m, _i, out: merged.append(out[0].detach().clone())
        )
        feats = inner.get_image_features(inputs["pixel_values"], inputs["target_sizes"])
        hook.remove()

        ts = inputs["target_sizes"].tolist()
        # The tower runs all crops in one packed sequence; split it back up.
        hidden = feats.last_hidden_state[0]
        merged_all = merged[0] if merged else None
        off_h = off_m = 0
        for i, (h, w) in enumerate(ts):
            n = h * w // 4
            tensors[f"vision_hidden.{i}"] = hidden[off_h : off_h + n].contiguous()
            off_h += n
            if merged_all is not None:
                tensors[f"vit_merged.{i}"] = merged_all[off_m : off_m + n].contiguous()
                off_m += n
            tensors[f"features.{i}"] = feats.pooler_output[i].float().contiguous()

        out = model(**inputs)
        tensors["last_logits"] = out.logits[0, -1].float().contiguous()

        gen = model.generate(
            **inputs, max_new_tokens=args.n_new, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    new_ids = gen.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    margins = []
    for step in gen.scores:
        top2 = torch.topk(step[0].float(), 2).values
        margins.append(float(top2[0] - top2[1]))

    save_file(tensors, args.out + ".safetensors")
    meta = {
        "model_dir": args.model_dir,
        "image": decoded,
        "source_image": args.image,
        "question": args.question,
        "prompt": prompt,
        "image_size": list(image.size),
        "target_sizes": ts,
        "greedy": new_ids,
        "margins": margins,
        "text": processor.tokenizer.decode(new_ids, skip_special_tokens=True),
        "max_slice_nums": args.max_slice_nums,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("target_sizes", "greedy", "text")}, indent=1))


if __name__ == "__main__":
    main()
