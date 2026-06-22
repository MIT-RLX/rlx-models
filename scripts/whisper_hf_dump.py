#!/usr/bin/env python3
"""Dump HF Whisper base.en reference for the JFK clip: mel, encoder output, transcript.

Writes raw f32 little-endian:
  /tmp/hf_mel.bin  (n_mels * n_frames, row-major [n_mels, n_frames])
  /tmp/hf_enc.bin  (enc_seq * d_model, row-major [enc_seq, d_model])
and prints shapes/stats + the greedy transcript (sanity).
"""
import sys
import numpy as np
import torch
import wave

MODEL_DIR = "/Users/Shared/rlx-models/.cache/whisper-base.en"
WAV = "/Users/Shared/rlx-models/.cache/whisper-bench/jfk_16k.wav"


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    assert sr == 16000, sr
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


def main():
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperFeatureExtractor,
        AutoTokenizer,
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    n_mels = model.config.num_mel_bins
    fe = WhisperFeatureExtractor(feature_size=n_mels, sampling_rate=16000)
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception as e:
        print(f"tokenizer load failed: {e}")
        tok = None

    pcm = load_wav(WAV)
    print(f"pcm len={len(pcm)} ({len(pcm)/16000:.2f}s) n_mels={n_mels}")

    feats = fe(pcm, sampling_rate=16000, return_tensors="pt")
    mel = feats.input_features  # [1, n_mels, n_frames]
    print(f"hf mel shape={tuple(mel.shape)} mean={mel.mean():.6f} std={mel.std():.6f}")
    mel.numpy().astype("<f4").tofile("/tmp/hf_mel.bin")

    with torch.no_grad():
        enc = model.model.encoder(mel).last_hidden_state  # [1, enc_seq, d]
    print(f"hf enc shape={tuple(enc.shape)} mean={enc.mean():.6f} std={enc.std():.6f}")
    enc.numpy().astype("<f4").tofile("/tmp/hf_enc.bin")

    with torch.no_grad():
        gen = model.generate(mel, max_new_tokens=64)
    ids = gen[0].tolist()
    print(f"hf gen ids={ids}")
    if tok is not None:
        text = tok.decode(gen[0], skip_special_tokens=True)
        print(f"hf transcript={text!r}")


if __name__ == "__main__":
    sys.exit(main())
