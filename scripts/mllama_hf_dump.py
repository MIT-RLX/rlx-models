#!/usr/bin/env python3
# RLX — Llama-3.2-Vision (mllama) HF reference dump for RLX parity.
#
# Runs the HuggingFace MllamaForConditionalGeneration on one image + prompt and
# dumps the tensors the RLX port must reproduce:
#   - pixel_values, aspect_ratio_ids, aspect_ratio_mask  (processor output)
#   - cross_attention_states  [n_tiles, num_patches, text_hidden]  (vision+projector, real tiles only)
#   - prompt token ids and greedily generated token ids
#
# Usage:
#   python3 scripts/mllama_hf_dump.py --ckpt <dir> --image <path> \
#       [--prompt "Describe this image."] [--max-new 32] [--out out/mllama_ref] [--device auto]
#
# On the RLX side compare:
#   - vision cross_states  vs  cross_attention_states  (cosine per row)
#   - generated ids        vs  gen_ids  (exact prefix)
import argparse, json, os
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Llama-3.2-*-Vision checkpoint dir")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe this image in detail.")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--out", default="out/mllama_ref")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--raw", action="store_true", help="use --prompt verbatim (no chat template)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dtype = torch.float32
    device_map = None if args.device == "cpu" else args.device
    model = MllamaForConditionalGeneration.from_pretrained(
        args.ckpt, dtype=dtype, device_map=device_map
    )
    if device_map is None:
        model = model.to("cpu")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)

    image = Image.open(args.image).convert("RGB")
    if args.raw:
        text = args.prompt
    else:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    dev = model.device
    with torch.no_grad():
        # Vision tower + projector -> cross_attention_states.
        vision_outputs = model.model.vision_model(
            pixel_values=inputs["pixel_values"],
            aspect_ratio_ids=inputs["aspect_ratio_ids"],
            aspect_ratio_mask=inputs["aspect_ratio_mask"],
        )
        cas = vision_outputs.last_hidden_state
        cas = model.model.multi_modal_projector(cas).reshape(-1, cas.shape[-2], model.config.text_config.hidden_size)
        cas = cas.float().cpu().numpy()  # [B*media*tiles, num_patches, hidden] (tiles padded to max_num_tiles)

    # Slice to the real tiles for this image (drop padded tiles).
    ar_mask = inputs["aspect_ratio_mask"].cpu().numpy().reshape(-1)  # [max_num_tiles]
    n_real = int(ar_mask.sum())
    cas_real = cas[:n_real]

    prompt_ids = inputs["input_ids"].cpu().numpy().reshape(-1).astype(np.int64)

    with torch.no_grad():
        gen = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new)
    gen_ids = gen[0].cpu().numpy().astype(np.int64)
    new_ids = gen_ids[prompt_ids.shape[0]:]

    np.save(f"{args.out}_pixel_values.npy", inputs["pixel_values"].cpu().float().numpy())
    np.save(f"{args.out}_aspect_ratio_ids.npy", inputs["aspect_ratio_ids"].cpu().numpy())
    np.save(f"{args.out}_cross_states.npy", cas_real)
    meta = {
        "ckpt": args.ckpt,
        "image": args.image,
        "prompt": args.prompt,
        "aspect_ratio_id": int(inputs["aspect_ratio_ids"].cpu().numpy().reshape(-1)[0]),
        "n_real_tiles": n_real,
        "num_patches": int(cas.shape[1]),
        "text_hidden": int(cas.shape[2]),
        "cross_states_shape": list(cas_real.shape),
        "prompt_ids": prompt_ids.tolist(),
        "gen_ids": new_ids.tolist(),
        "gen_text": processor.decode(new_ids, skip_special_tokens=True),
    }
    with open(f"{args.out}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out}_cross_states.npy {cas_real.shape} and {args.out}_meta.json")
    print("gen:", meta["gen_text"])


if __name__ == "__main__":
    main()
