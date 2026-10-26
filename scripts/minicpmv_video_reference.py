#!/usr/bin/env python3
"""Video reference dump for `rlx-minicpm-v` parity: HF transformers MiniCPM-V
4.6 / 4.7 on one video + question, in float32.

Frames are sampled with the HF video processor's own `sample_frames` and
decoded with the `ffmpeg` CLI (rgb24), the same decoder the Rust side runs, so
both read identical pixels. The processor is then called with
`do_sample_frames=False` on those frames.

  <out>.safetensors
    pixel_values_videos [1, 3, 14, L]  NaViT-packed crops of every frame
    target_sizes_videos [n_crops, 2]   patch grid per crop
    input_ids           [S]
    mm_token_type_ids   [S]            0 text, 2 video
    position_ids        [3, S]         canvas M-RoPE (4.7) / arange (4.6)
    features.{i}        [n, hidden]    merger output per crop, in LM space
    last_logits         [vocab]
  <out>.json  sampled frame indices, video metadata, settings, greedy ids,
              top-2 margins, text

Usage:
  .venv-minicpmv/bin/python -I scripts/minicpmv_video_reference.py \
      weights/vision/minicpm-v-4.7-1b clip.mp4 .cache/minicpmv/video47 \
      --max-num-frames 8 --stack-frames 3 --max-slice-nums 1
"""

import argparse
import json
import os
import subprocess
from fractions import Fraction

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,avg_frame_rate", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    fps = float(Fraction(s["avg_frame_rate"]))
    total = int(s["nb_frames"])
    return int(s["width"]), int(s["height"]), total, fps


def decode(path, indices, width, height):
    """Frames `indices` (any order, repeats allowed) as [T, H, W, 3] uint8."""
    unique = sorted(set(indices))
    select = "+".join(f"eq(n\\,{i})" for i in unique)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='{select}'",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    assert len(frames) == len(unique), (len(frames), len(unique))
    by_index = dict(zip(unique, frames))
    return np.stack([by_index[i] for i in indices])


def ffmpeg_version() -> str:
    out = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
    return out.stdout.splitlines()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--question", default="Describe this video in two sentences.")
    ap.add_argument("--n-new", type=int, default=24)
    ap.add_argument("--max-num-frames", type=int, default=8)
    ap.add_argument("--stack-frames", type=int, default=1)
    ap.add_argument("--max-slice-nums", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    width, height, total, fps = probe(args.video)
    metadata = VideoMetadata(total_num_frames=total, fps=fps, width=width, height=height,
                             duration=total / fps)
    indices = processor.video_processor.sample_frames(
        metadata, max_num_frames=args.max_num_frames, stack_frames=args.stack_frames
    )
    frames = decode(args.video, indices, width, height)
    metadata.frames_indices = list(indices)

    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": args.question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        videos=[frames], text=[prompt], return_tensors="pt", video_metadata=[metadata],
        do_sample_frames=False, max_num_frames=args.max_num_frames,
        stack_frames=args.stack_frames, max_slice_nums=args.max_slice_nums,
        return_mm_token_type_ids=True,
    )

    tensors = {
        "pixel_values_videos": inputs["pixel_values_videos"].float().contiguous(),
        "target_sizes_videos": inputs["target_sizes_videos"].to(torch.int32).contiguous(),
        "input_ids": inputs["input_ids"][0].to(torch.int64).contiguous(),
        "mm_token_type_ids": inputs["mm_token_type_ids"][0].to(torch.int64).contiguous(),
    }
    inner = model.model
    with torch.no_grad():
        if hasattr(inner, "get_rope_index"):
            pos, _ = inner.get_rope_index(
                inputs["input_ids"],
                mm_token_type_ids=inputs["mm_token_type_ids"],
                target_sizes_videos=inputs["target_sizes_videos"],
            )
            tensors["position_ids"] = pos[:, 0, :].to(torch.int64).contiguous()
        else:
            s = inputs["input_ids"].shape[1]
            tensors["position_ids"] = torch.arange(s).expand(3, s).contiguous()

        feats = inner.get_video_features(
            inputs["pixel_values_videos"], inputs["target_sizes_videos"]
        )
        for i, f in enumerate(feats.pooler_output):
            tensors[f"features.{i}"] = f.float().contiguous()

        out = model(**inputs)
        tensors["last_logits"] = out.logits[0, -1].float().contiguous()
        gen = model.generate(
            **inputs, max_new_tokens=args.n_new, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    new_ids = gen.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    margins = [float(torch.topk(s[0].float(), 2).values.diff().neg()) for s in gen.scores]

    save_file(tensors, args.out + ".safetensors")
    meta = {
        "model_dir": args.model_dir,
        "video": os.path.abspath(args.video),
        "question": args.question,
        "prompt": prompt,
        # Builds convert YUV to RGB differently; the Rust side holds pixels to
        # equality only under the same one.
        "ffmpeg": ffmpeg_version(),
        "frame_indices": list(map(int, indices)),
        "total_frames": total,
        "fps": fps,
        "size": [width, height],
        "max_num_frames": args.max_num_frames,
        "stack_frames": args.stack_frames,
        "max_slice_nums": args.max_slice_nums,
        "target_sizes": inputs["target_sizes_videos"].tolist(),
        "greedy": new_ids,
        "margins": margins,
        "text": processor.tokenizer.decode(new_ids, skip_special_tokens=True),
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("frame_indices", "greedy", "text")}, indent=1))


if __name__ == "__main__":
    main()
