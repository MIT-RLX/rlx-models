#!/usr/bin/env python3
"""Reference dump for `rlx-minicpm-v` parity on the resampler generation
(`model_type = minicpmv`, e.g. openbmb/MiniCPM-V-4): OpenBMB's remote code,
run in float32 on CPU. Needs transformers 4.5x (the remote code predates 5.x),
torch, torchvision, pillow.

  <out>.safetensors
    pixel.{i}          [3, 14, h*w*14]  processor output per crop
    tgt_sizes          [n_crops, 2]     patch grid per crop (thumbnail first)
    input_ids          [S]
    image_bound        [n_crops, 2]     [start, end) of each crop's tokens
    vision_hidden.{i}  [h*w, 1152]      SigLIP output (post_layernorm)
    features.{i}       [query_num, hidden]  resampler output per crop
    last_logits        [vocab]
  <out>.json  prompt, greedy ids, top-2 margins, text

Usage:
  .venv-sala/bin/python -I scripts/minicpmv_legacy_reference.py \
      weights/vision/minicpm-v-4 crates/rlx-locateanything/fixtures/sample.jpg \
      .cache/minicpmv/v4 [--widen]

--video CLIP [--fps N] [--force-packing K] (4.5) runs the model card's packed
video instead of the image: `encode_video` (ported here onto ffprobe / ffmpeg,
the decoder the Rust side runs), frames grouped with temporal ids,
`use_image_id=False`, `max_slice_nums=1`; features.{g} is then one group.

--widen loads the checkpoint in its own 16-bit type (bf16, or f16 for 2.5)
and widens each LM decoder layer to f32
only for its own forward (the tower, resampler, embeddings and head stay f32):
every product is f32 with the checkpoint's values, at about half the memory of
an f32 load — what an 8B model (4.5) needs.
"""

import argparse
import builtins
import json
import math
import os
import sys
import typing

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoModel, AutoProcessor


def encode_video(path, choose_fps, force_packing, max_frames=180, max_packing=3):
    """The MiniCPM-V-4_5 card's `encode_video`, on ffprobe metadata."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    from minicpmv_video_reference import decode, probe

    width, height, total, fps = probe(path)
    duration = total / fps
    if choose_fps * int(duration) <= max_frames:
        packing = 1
        choose = round(min(choose_fps, round(fps)) * min(max_frames, duration))
    else:
        packing = math.ceil(duration * choose_fps / max_frames)
        if packing <= max_packing:
            choose = round(duration * choose_fps)
        else:
            choose = round(max_frames * max_packing)
            packing = max_packing
    gap = total / choose
    idx = np.array([int(i * gap + gap / 2) for i in range(choose)])
    if force_packing:
        packing = min(force_packing, max_packing)
    scale = np.arange(0, duration, 0.1)
    ts = idx / fps
    nearest = scale[np.abs(ts[:, None] - scale[None, :]).argmin(axis=1)]
    ts_id = (nearest / 0.1).astype(np.int32)
    groups = [ts_id[i:i + packing].tolist() for i in range(0, len(ts_id), packing)]
    frames = decode(path, idx.tolist(), width, height)
    return frames, idx.tolist(), groups, (width, height, total, fps)


def widen_layers(model) -> None:
    for m in (model.vpm, model.resampler, model.llm.model.embed_tokens,
              model.llm.model.norm, model.llm.lm_head):
        m.float()
    if hasattr(model.llm.model, "rotary_emb"):
        model.llm.model.rotary_emb.float()

    storage = next(model.llm.model.layers[0].parameters()).dtype

    def cast(module, dtype):
        for p in module.parameters():
            p.data = p.data.to(dtype)

    def pre(module, args, kwargs):
        cast(module, torch.float32)
        args = tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a
                     for a in args)
        return args, kwargs

    def post(module, _args, _kwargs, out):
        cast(module, storage)
        return out

    for layer in model.llm.model.layers:
        layer.register_forward_pre_hook(pre, with_kwargs=True)
        layer.register_forward_hook(post, with_kwargs=True)


def main() -> None:
    # 2.5's resampler.py annotates with `List`, which it expected
    # `from torch.nn.functional import *` to bring in; torch 2.x no longer does.
    builtins.List = typing.List
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("out")
    ap.add_argument("--question", default="Describe this image in one sentence.")
    ap.add_argument("--n-new", type=int, default=24)
    ap.add_argument("--widen", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--force-packing", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True,
        torch_dtype="auto" if args.widen else torch.float32,
        attn_implementation="eager",
    ).eval()
    if args.widen:
        widen_layers(model)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    video = None
    if args.video:
        frames, idx, groups, vmeta = encode_video(args.video, args.fps, args.force_packing)
        images = [Image.fromarray(f) for f in frames]
        video = {"video": os.path.abspath(args.video), "frame_indices": idx,
                 "temporal_groups": groups, "total_frames": vmeta[2], "fps": vmeta[3],
                 "choose_fps": args.fps, "force_packing": args.force_packing}
        import subprocess
        video["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                         text=True).stdout.splitlines()[0]
        content = "\n".join(["(<image>./</image>)"] * len(images) + [args.question])
        image = None
    else:
        image = Image.open(args.image).convert("RGB")
        decoded = args.out + ".png"
        image.save(decoded)  # JPEG decoders disagree; the Rust side reads this.
        # `MiniCPMV.chat`'s prompt: the image tag, a newline, the question.
        content = "(<image>./</image>)\n" + args.question
    msgs = [{"role": "user", "content": content}]
    # `enable_thinking=False` is `chat()`'s default (only Qwen3 templates read it).
    prompt = processor.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if video is not None:
        inputs = processor([prompt], [images], max_slice_nums=1, use_image_id=False,
                           temporal_ids=[groups], return_tensors="pt")
    elif getattr(processor.image_processor, "version", None) == 2.5:
        # 2.5's processor takes one prompt and a flat image list.
        inputs = processor(prompt, [image], return_tensors="pt")
    else:
        inputs = processor([prompt], [[image]], return_tensors="pt")

    pixel = inputs["pixel_values"][0]
    tgt = inputs["tgt_sizes"][0]
    tensors = {
        "tgt_sizes": tgt.to(torch.int32).contiguous(),
        "input_ids": inputs["input_ids"][0].to(torch.int64).contiguous(),
        "image_bound": inputs["image_bound"][0].to(torch.int64).contiguous(),
    }
    for i, p in enumerate(pixel):
        tensors[f"pixel.{i}"] = p.float().contiguous()

    data = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
        "tgt_sizes": inputs["tgt_sizes"],
        "image_bound": inputs["image_bound"],
    }
    if video is not None:
        data["temporal_ids"] = inputs["temporal_ids"]
    with torch.no_grad():
        hidden = []
        hook = model.vpm.register_forward_hook(
            lambda _m, _i, out: hidden.append(out.last_hidden_state.detach().clone())
        )
        embeds, vision = model.get_vllm_embedding(data)
        hook.remove()
        vh = torch.cat(hidden, 0)
        if video is None:
            for i, (h, w) in enumerate(tgt.tolist()):
                tensors[f"vision_hidden.{i}"] = vh[i, : h * w].contiguous()
                tensors[f"features.{i}"] = vision[0][i].float().contiguous()
        else:
            for g in range(len(vision[0])):
                tensors[f"features.{g}"] = vision[0][g].float().contiguous()

        out = model.llm(inputs_embeds=embeds)
        tensors["last_logits"] = out.logits[0, -1].float().contiguous()
        gen = model.llm.generate(
            inputs_embeds=embeds, max_new_tokens=args.n_new, do_sample=False, num_beams=1,
            output_scores=True, return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.pad_token_id or 0,
        )
    new_ids = gen.sequences[0].tolist()[-len(gen.scores):]
    margins = [float(torch.topk(s[0].float(), 2).values.diff().neg()) for s in gen.scores]

    save_file(tensors, args.out + ".safetensors")
    meta = {
        "model_dir": args.model_dir,
        "question": args.question,
        "prompt": prompt,
        "target_sizes": tgt.tolist(),
        "greedy": new_ids,
        "margins": margins,
        "text": processor.tokenizer.decode(new_ids, skip_special_tokens=True),
    }
    if video is None:
        meta.update(image=decoded, source_image=args.image, image_size=list(image.size))
    else:
        meta.update(video)
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("target_sizes", "greedy", "text")}, indent=1))


if __name__ == "__main__":
    main()
