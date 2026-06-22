#!/usr/bin/env python3
# RLX — Florence-2 HF reference dumper for parity tests.
#
# Loads microsoft/Florence-2 with trust_remote_code, runs a fixed
# image+task through the staged forward, and writes a JSON payload consumed
# by `crates/rlx-florence2/tests/florence2_hf_parity.rs`:
#
#   {
#     "pixel_values":  [..],  # [3*H*W]
#     "img_size":      H,
#     "input_ids":     [..],  # text prompt token ids (BART)
#     "image_features":[..],  # [577*1024]
#     "encoder_hidden":[..],  # [(577+T)*1024]
#     "seq":           577+T,
#     "decoder_input_ids": [..],   # full greedy prefix incl. start token
#     "step0_logits":  [..],  # [vocab]  logits for the first decode step
#     "greedy":        [..],  # greedy token ids (no beams)
#     "beam":          [..],  # num_beams=3 token ids
#   }
import argparse
import json
import sys
import types
from pathlib import Path

# Florence-2's remote modeling file statically imports `flash_attn`, but only
# uses it under `is_flash_attn_2_available()` (False on CPU). Register a stub so
# the import check passes without a real CUDA flash-attn build.
if "flash_attn" not in sys.modules:
    import importlib.machinery

    fa = types.ModuleType("flash_attn")
    fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    fa.flash_attn_func = None
    fa.flash_attn_varlen_func = None
    sys.modules["flash_attn"] = fa
    bp = types.ModuleType("flash_attn.bert_padding")
    bp.index_first_axis = bp.pad_input = bp.unpad_input = None
    sys.modules["flash_attn.bert_padding"] = bp

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--task", default="<CAPTION>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default=None, help="real image path (else synthetic)")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    from PIL import Image

    if args.image:
        pil = Image.open(args.image).convert("RGB")
        image = np.asarray(pil, dtype=np.uint8)
        h, w = image.shape[0], image.shape[1]
    else:
        # Deterministic synthetic RGB image so Rust + Python share input.
        rng = np.random.default_rng(0)
        h = w = 768
        image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        pil = Image.fromarray(image, mode="RGB")

    inputs = processor(text=args.task, images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(torch.float32)
    input_ids = inputs["input_ids"]
    out = {
        "pixel_values": pixel_values.reshape(-1).tolist(),
        "img_size": int(pixel_values.shape[-1]),
        "input_ids": input_ids.reshape(-1).tolist(),
        "image_rgb": image.reshape(-1).tolist(),
        "image_hw": [int(h), int(w)],
    }

    with torch.no_grad():
        # Stage 1: image features.
        image_features = model._encode_image(pixel_values)  # [1, 577, 1024]
        out["image_features"] = image_features.reshape(-1).tolist()

        # Stage 2: merge with text embeds → encoder.
        inputs_embeds = model.get_input_embeddings()(input_ids)
        merged, attn = model._merge_input_ids_with_image_features(
            image_features, inputs_embeds
        )
        encoder = model.get_encoder()
        enc_out = encoder(inputs_embeds=merged, attention_mask=attn)
        encoder_hidden = enc_out.last_hidden_state  # [1, seq, 1024]
        out["encoder_hidden"] = encoder_hidden.reshape(-1).tolist()
        out["seq"] = int(encoder_hidden.shape[1])

        # Stage 3: first decode step logits (decoder_start_token_id = 2).
        start = model.config.text_config.decoder_start_token_id
        dec_in = torch.tensor([[start]], dtype=torch.long)
        lm = model.language_model
        step = lm(
            encoder_outputs=(encoder_hidden,),
            decoder_input_ids=dec_in,
            attention_mask=attn,
        )
        step0 = step.logits[0, -1, :].to(torch.float32)
        out["step0_logits"] = step0.tolist()

        # Stage 4: greedy + beam generation (full model.generate).
        greedy = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        out["greedy"] = greedy[0].reshape(-1).tolist()

        beam = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            num_beams=3,
            do_sample=False,
        )
        out["beam"] = beam[0].reshape(-1).tolist()

        # Stage 5: HF post-processed answer for the task (boxes / text / etc.).
        gen_text = processor.batch_decode(beam, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            gen_text, task=args.task, image_size=(w, h)
        )
        out["hf_answer"] = json.loads(json.dumps(parsed[args.task], default=list))

    Path(args.out).write_text(json.dumps(out))
    print(
        f"wrote {args.out}: seq={out['seq']} "
        f"input_ids={out['input_ids']} greedy_len={len(out['greedy'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
