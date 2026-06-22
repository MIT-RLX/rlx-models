#!/usr/bin/env python3
"""Headless moshi_mlx bench (Metal) — one-way zeros input, no sounddevice."""

import argparse
import time
import wave
import struct

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from moshi_mlx import models, utils


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-wav", required=True)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("-q", "--quantized", type=int, choices=[4, 8], default=4)
    p.add_argument("--hf-repo", default="kyutai/moshiko-mlx-q4")
    p.add_argument("--moshi-weight")
    p.add_argument("--mimi-weight")
    args = p.parse_args()

    import huggingface_hub

    def dl(path: str) -> str:
        return huggingface_hub.hf_hub_download(args.hf_repo, path)

    if args.moshi_weight:
        model_file = args.moshi_weight
    elif args.quantized == 8:
        model_file = dl("model.q8.safetensors")
    elif args.quantized == 4:
        model_file = dl("model.q4.safetensors")
    else:
        model_file = dl("model.safetensors")

    mimi_file = args.mimi_weight or dl("tokenizer-e351c8d8-checkpoint125.safetensors")

    t0 = time.perf_counter()
    lm_config = models.config_v0_1()
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if args.quantized:
        group_size = 32 if args.quantized == 4 else 64
        nn.quantize(model, bits=args.quantized, group_size=group_size)
    model.load_weights(model_file, strict=True)
    model.warmup()
    load_ms = (time.perf_counter() - t0) * 1000

    import rustymimi

    mimi = rustymimi.StreamTokenizer(mimi_file)

    def run_steps(n: int) -> tuple[list[float], int]:
        gen = models.LmGen(
            model=model,
            max_steps=n + 5,
            text_sampler=utils.Sampler(),
            audio_sampler=utils.Sampler(),
            check=False,
        )
        pcm_out: list[float] = []
        out_frames = 0
        zeros = np.zeros(1920, dtype=np.float32)
        for _ in range(n):
            mimi.encode(zeros)
            while True:
                enc = mimi.get_encoded()
                if enc is not None:
                    break
            data = mx.array(enc).transpose(1, 0)[:, :8]
            gen.step(data)
            audio_tokens = gen.last_audio_tokens()
            if audio_tokens is not None:
                mimi.decode(np.array(audio_tokens, dtype=np.uint32))
                while True:
                    dec = mimi.get_decoded()
                    if dec is not None:
                        pcm_out.extend(dec.tolist())
                        out_frames += 1
                        break
        return pcm_out, out_frames

    if args.warmup:
        run_steps(args.warmup)

    t1 = time.perf_counter()
    pcm, out_frames = run_steps(args.max_steps)
    gen_ms = (time.perf_counter() - t1) * 1000

    out = args.out_wav
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        ints = [int(max(-1.0, min(1.0, x)) * 32767) for x in pcm]
        w.writeframes(struct.pack("<" + "h" * len(ints), *ints))

    audio_s = len(pcm) / 24000
    ms_per_frame = gen_ms / max(out_frames, 1)
    rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 0
    print(f"backend=moshi_mlx_q{args.quantized}_metal")
    print(f"load_ms={load_ms:.1f}")
    print(f"gen_ms={gen_ms:.1f}")
    print(f"max_steps={args.max_steps}")
    print(f"out_frames={out_frames}")
    print(f"out_samples={len(pcm)}")
    print(f"audio_s={audio_s:.3f}")
    print(f"ms_per_frame={ms_per_frame:.1f}")
    print(f"rtf={rtf:.2f}")
    print(f"out_wav={out}")


if __name__ == "__main__":
    main()
