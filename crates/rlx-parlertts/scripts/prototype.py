#!/usr/bin/env python3
"""Parler-TTS ONNX pipeline prototype (validate BEFORE the Rust port).

Drives the exported text_encoder + decoder ONNX with onnxruntime through the
9-codebook DELAY-PATTERN AR loop, decodes with transformers DacModel, and writes
a wav. Resolves the open question: does the transcript condition via the T5
encoder input (this export has NO prompt input on the decoder)?

  python3 crates/rlx-parlertts/scripts/prototype.py
"""
import sys, numpy as np, onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
W = ROOT / "weights/tts/parlertts"
K = 9                    # num_codebooks
BOS = 1025               # decoder_start_token_id
PAD = 1024              # pad_token_id (== eos)
VOCAB = 1088            # per-codebook vocab (codes 0..1023 + specials)
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 120

TEXT = sys.argv[1] if len(sys.argv) > 1 else "The quick brown fox jumps over the lazy dog."
DESC = "A clear female voice speaks slowly."

tok = Tokenizer.from_file(str(W / "tokenizer.json"))
enc_sess = ort.InferenceSession(str(W / "onnx/text_encoder.onnx"))
dec_sess = ort.InferenceSession(str(W / "onnx/decoder.onnx"))


def encode(text):
    ids = np.array([tok.encode(text).ids + [1]], dtype=np.int64)  # T5 eos=1
    mask = np.ones_like(ids)
    (hs,) = enc_sess.run(None, {"input_ids": ids, "attention_mask": mask})
    return hs.astype(np.float32), mask


def build_delay(codes):
    # codes: list of [9] per generation step -> [9, T] aligned (un-delayed)
    T = len(codes)
    arr = np.full((K, T), PAD, dtype=np.int64)
    for t, c in enumerate(codes):
        arr[:, t] = c
    # un-delay: codebook k was produced delayed by k -> shift left by k
    out = np.full((K, T), PAD, dtype=np.int64)
    for k in range(K):
        out[k, : T - k] = arr[k, k:T]
    return out


def run(enc_text, label):
    hs, emask = encode(enc_text)
    # decoder_input_ids [1, 9, 1] all BOS
    dids = np.full((1, K, 1), BOS, dtype=np.int64)
    codes = []
    for step in range(MAX_STEPS):
        (logits,) = dec_sess.run(
            None,
            {"decoder_input_ids": dids, "encoder_hidden_states": hs, "encoder_attention_mask": emask},
        )
        # logits: [9, T, 1088] = [codebook, seq, vocab]; take last position
        lg = np.asarray(logits)  # [9, T, 1088]
        nxt = lg[:, -1, :].argmax(-1).astype(np.int64)  # [9] greedy per codebook
        # delay: codebook k only real once step>=k; else feed BOS/pad
        for k in range(K):
            if step < k:
                nxt[k] = BOS
        codes.append(nxt.copy())
        if step >= K and (nxt == PAD).all():
            break
        dids = np.concatenate([dids, nxt.reshape(1, K, 1)], axis=2)
    aligned = build_delay(codes)
    # clamp to valid DAC range [0,1023]
    valid_T = aligned.shape[1]
    while valid_T > 0 and (aligned[:, valid_T - 1] >= 1024).any():
        valid_T -= 1
    aligned = aligned[:, :valid_T]
    print(f"[{label}] steps={len(codes)} aligned codes {aligned.shape} "
          f"range[{aligned.min()},{aligned.max()}] cb0[:12]={aligned[0,:12].tolist()}")
    np.save(W / f"proto_codes_{label}.npy", aligned)
    return aligned


print("=== A: transcript -> encoder ===")
run(TEXT, "transcript")
print("=== B: description -> encoder ===")
run(DESC, "description")
print("codes dumped; decode with DAC + whisper to see which path speaks the words")
