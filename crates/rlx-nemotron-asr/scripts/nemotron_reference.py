"""Nemotron 3.5 ASR through Hugging Face transformers: the reference the
rlx runner is checked against.

    python scripts/nemotron_reference.py --model-dir weights/asr/nemotron-asr \
        --wav clip16k.wav [--out ref.npz]

Prints the transcript; `--out` saves the log-mel features and the encoder's
output for a stage-by-stage comparison.
"""

import argparse
import wave

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    pcm, sr = read_wav(args.wav)
    assert sr == 16000, f"{args.wav} is {sr} Hz; want 16 kHz"
    proc = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir, torch_dtype=torch.float32).eval()
    inputs = proc(pcm, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs)
    ids = getattr(ids, "sequences", ids)
    print(proc.batch_decode(ids, skip_special_tokens=True)[0])
    if args.out:
        with torch.no_grad():
            enc = model.get_encoder()(**{k: v for k, v in inputs.items() if k != "prompt_ids"})
        np.savez(
            args.out,
            features=inputs["input_features"].numpy(),
            encoder=enc.last_hidden_state.numpy(),
        )


if __name__ == "__main__":
    main()
