#!/usr/bin/env python3
"""Moonshine streaming (v2) reference: encoder states and greedy tokens, f32.

    python3 scripts/moonshine_streaming_reference.py weights/asr/moonshine-streaming-tiny \
        assets/harness/jfk_16k.wav /tmp/moonshine-streaming-tiny.ref.json

Needs a transformers with `moonshine_streaming` (5.x). The JSON holds the
encoder output after `final_norm`, the decoder-side encoder states (after the
learned position embedding and the width projection), and the greedy token
ids for `--max-new-tokens`.
"""

import argparse
import json
import wave

import numpy as np
import torch
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, "need 16 kHz 16-bit PCM"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("wav")
    ap.add_argument("out")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    audio = read_wav(args.wav)
    proc = AutoProcessor.from_pretrained(args.model_dir)
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).eval()
    inputs = proc(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        enc = model.model.encoder(inputs["input_values"], attention_mask=inputs["attention_mask"])
        hidden = enc.last_hidden_state
        dec = model.model.decoder
        t = hidden.shape[1]
        dec_states = dec.proj(hidden + dec.pos_emb(torch.arange(t)))
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)[0]

    tokens = gen.tolist()
    out = {
        "num_samples": len(audio),
        "encoder_hidden": hidden[0].flatten().tolist(),
        "encoder_seq": t,
        "encoder_mask": enc.attention_mask[0].int().tolist() if enc.attention_mask is not None else None,
        "decoder_encoder_states": dec_states[0].flatten().tolist(),
        "tokens": tokens,
        "text": proc.batch_decode([tokens], skip_special_tokens=True)[0],
    }
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}: enc_seq={t} tokens={len(tokens)} text={out['text']!r}")


if __name__ == "__main__":
    main()
