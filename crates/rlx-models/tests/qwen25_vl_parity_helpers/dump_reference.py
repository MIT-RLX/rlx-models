#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Dump HuggingFace Qwen2.5-VL activations for Rust parity tests.

import json
import os
import sys

import numpy as np
from PIL import Image

from aif_probe import (
    blocked_keys_from_entropy,
    build_aif_probe,
    compute_dynamics_decode_step,
    compute_dynamics_eq2_prefill,
    dynamics_mode,
    find_vision_span,
    resolve_model_dir,
)


def env(name: str) -> str:
    v = os.environ.get(name)
    if v is None:
        print(f"missing env var: {name}", file=sys.stderr)
        sys.exit(2)
    return v


def main() -> None:
    image_path = env("RLX_QWEN25_VL_IMAGE")
    out_dir = env("RLX_QWEN25_VL_OUT_DIR")
    os.makedirs(out_dir, exist_ok=True)

    model_dir = resolve_model_dir()
    device = os.environ.get("RLX_QWEN25_VL_DEVICE", "cpu")
    prompt = os.environ.get(
        "RLX_QWEN25_VL_PROMPT",
        "Describe this image briefly.",
    )

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = torch.float32
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.bfloat16
    elif device == "mps" and torch.backends.mps.is_available():
        device = "mps"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_dir)
    image = Image.open(image_path).convert("RGB")
    source_w, source_h = image.size
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size

    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    ids_list = inputs["input_ids"][0].tolist()
    vision_start_idx, n_vision = find_vision_span(ids_list, image_pad_id)
    if n_vision == 0:
        print("warning: no <|image_pad|> run in input_ids", file=sys.stderr)

    mode = dynamics_mode()
    seq_len = int(inputs["input_ids"].shape[-1])
    vision_dynamics = None

    with torch.no_grad():
        vision_out = model.get_image_features(
            inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        if mode == "decode_step" and n_vision > 0:
            prefill = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=True,
                return_dict=True,
            )
            next_id = prefill.logits[:, -1:].argmax(dim=-1)
            decode = model(
                input_ids=next_id,
                past_key_values=prefill.past_key_values,
                output_attentions=True,
                return_dict=True,
            )
            out = prefill
            if decode.attentions is not None:
                vision_dynamics = compute_dynamics_decode_step(
                    decode.attentions, vision_start_idx, n_vision
                )
        else:
            out = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            if out.attentions is not None and n_vision > 0:
                vision_dynamics = compute_dynamics_eq2_prefill(
                    out.attentions, vision_start_idx, n_vision, seq_len
                )

    vision_mu = None
    vision_token_entropy = None
    aif_ratio = None
    aif_s0 = None
    aif_blocked = None
    if vision_dynamics is not None:
        probe = build_aif_probe(vision_dynamics)
        vision_mu = probe["mu"]
        vision_token_entropy = probe["token_entropy"]
        aif_s0 = probe["s0"]
        aif_ratio = probe["mask_ratio"]
        aif_blocked = blocked_keys_from_entropy(
            vision_token_entropy, vision_start_idx, aif_ratio
        )

    if isinstance(vision_out.pooler_output, (list, tuple)):
        vision_emb = vision_out.pooler_output[0]
    else:
        vision_emb = vision_out.pooler_output
    vision_emb = vision_emb.float().cpu().numpy()

    logits = out.logits[0, -1].float().cpu().numpy()
    hidden = out.hidden_states[-1][0, -1].float().cpu().numpy()
    input_ids = inputs["input_ids"][0].cpu().numpy()

    grid = None
    resized_w = None
    resized_h = None
    if "image_grid_thw" in inputs:
        g = inputs["image_grid_thw"][0].tolist()
        grid = {"t": g[0], "h": g[1], "w": g[2]}
        resized_h = int(grid["h"] * patch_size)
        resized_w = int(grid["w"] * patch_size)

    meta = {
        "model_dir": model_dir,
        "image": image_path,
        "prompt": prompt,
        "chat_text": text,
        "source_w": int(source_w),
        "source_h": int(source_h),
        "patch_size": int(patch_size),
        "merge_size": int(merge_size),
        "seq_len": int(inputs["input_ids"].shape[-1]),
        "vocab_size": int(logits.shape[0]),
        "hidden_size": int(hidden.shape[0]),
        "vision_start_idx": int(vision_start_idx),
        "n_vision_tokens": int(n_vision),
        "image_pad_id": int(image_pad_id),
        "vision_proj_dim": int(vision_emb.shape[-1]),
        "vision_rows": int(vision_emb.shape[0]),
        "aif_dynamics_mode": mode,
    }
    if resized_w is not None:
        meta["resized_w"] = resized_w
        meta["resized_h"] = resized_h
    if grid is not None:
        meta["image_grid_thw"] = grid
    if vision_mu is not None:
        meta["aif_mask_ratio"] = aif_ratio
        meta["aif_s0"] = aif_s0
        meta["aif_n_layers"] = int(vision_dynamics.shape[1])
        meta["aif_blocked_keys"] = aif_blocked

    np.save(os.path.join(out_dir, "logits_last.npy"), logits)
    np.save(os.path.join(out_dir, "hidden_last.npy"), hidden)
    np.save(os.path.join(out_dir, "input_ids.npy"), input_ids)
    np.save(os.path.join(out_dir, "vision_embeddings.npy"), vision_emb)
    if vision_dynamics is not None:
        np.save(
            os.path.join(out_dir, "vision_dynamics.npy"),
            vision_dynamics.astype(np.float32),
        )
    if vision_mu is not None:
        np.save(
            os.path.join(out_dir, "vision_mu_scores.npy"),
            np.asarray(vision_mu, dtype=np.float32),
        )
    if vision_token_entropy is not None:
        np.save(
            os.path.join(out_dir, "vision_token_entropy.npy"),
            np.asarray(vision_token_entropy, dtype=np.float32),
        )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"wrote reference to {out_dir} (seq={meta['seq_len']} "
        f"vision={meta['vision_rows']}x{meta['vision_proj_dim']}"
        f"{f' aif_ratio={aif_ratio:.3f}' if aif_ratio is not None else ''})"
    )


if __name__ == "__main__":
    main()
