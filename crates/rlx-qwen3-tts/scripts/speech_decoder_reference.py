"""The official `qwen-tts` 12 Hz speech tokenizer decoding fixed codes: the
reference the crate's `St12HzDecoder` is checked against.

    python scripts/speech_decoder_reference.py --model-dir weights/tts/qwen3-tts-0.6b-customvoice \
        --out tests/fixtures/st12hz_decode

Writes `codes.json` (`[frames][16]`, seeded) and `pcm.f32` (little-endian
f32 samples of the decoded waveform).
"""

import argparse
import json
import os

import numpy as np
import torch
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    tok = Qwen3TTSTokenizer.from_pretrained(
        os.path.join(args.model_dir, "speech_tokenizer"), device_map="cpu"
    )
    rng = np.random.default_rng(args.seed)
    codes = rng.integers(0, 2048, size=(args.frames, 16), dtype=np.int64)
    wavs, sr = tok.decode({"audio_codes": torch.from_numpy(codes)})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "codes.json"), "w") as f:
        json.dump(codes.tolist(), f)
    np.asarray(wavs[0], dtype="<f4").tofile(os.path.join(args.out, "pcm.f32"))
    print(len(wavs[0]), sr)


if __name__ == "__main__":
    main()
