#!/usr/bin/env python3
"""Bake HF transformers Mimi encode codes for Rust parity (kyutai/mimi).

Usage:
  python3 scripts/mimi_hf_parity.py \\
    --wav crates/rlx-qwen3-tts/examples/audio/ask_not.wav \\
    --out crates/rlx-mimi/tests/fixtures/hf_ask_not.json

Requires: pip install torch transformers soundfile
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="kyutai/mimi")
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-quantizers", type=int, default=32)
    args = p.parse_args()

    import torch
    import soundfile as sf
    from transformers import AutoFeatureExtractor, MimiModel

    wav, sr = sf.read(args.wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    model = MimiModel.from_pretrained(args.model)
    model.eval()

    target_sr = fe.sampling_rate
    if sr != target_sr:
        import torchaudio

        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        wav = wav_t.squeeze(0).numpy()
        sr = target_sr

    inputs = fe(raw_audio=wav, sampling_rate=sr, return_tensors="pt")
    input_pcm = inputs["input_values"].squeeze().tolist()
    with torch.no_grad():
        enc = model.encode(inputs["input_values"], num_quantizers=args.num_quantizers)
        codes = enc.audio_codes.squeeze(0).cpu().tolist()  # [K, T]
        dec = model.decode(torch.tensor(codes).unsqueeze(0))
        recon = dec[0].squeeze().cpu().numpy().tolist()
        if isinstance(recon[0], list):
            recon = [x for row in recon for x in row]

    payload = {
        "model": args.model,
        "wav": str(args.wav),
        "sample_rate": sr,
        "pcm_samples": len(input_pcm),
        "input_pcm": input_pcm,
        "num_quantizers": args.num_quantizers,
        "num_frames": len(codes[0]) if codes else 0,
        "codes_hf_layout": codes,
        "recon_pcm": recon,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} ({payload['num_frames']} frames × {args.num_quantizers} codebooks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
