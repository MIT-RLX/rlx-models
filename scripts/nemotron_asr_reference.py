#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# Distributed WITHOUT ANY WARRANTY; see the GNU GPL v3 for details.
"""Dump NeMo reference tensors for the Nemotron 3.5 ASR model so the Rust
port (`rlx-nemotron-asr`) can be checked element-by-element.

This is a *parity reference* generator — it is NOT part of the runtime path
(RLX loads the `.nemo` natively via `rlx-nemo`). Run it on a machine with a
working NeMo + PyTorch install:

    pip install "nemo_toolkit[asr]"
    python scripts/nemotron_asr_reference.py \
        --nemo nemotron-3.5-asr-streaming-0.6b.nemo \
        --wav  sample16k.wav \
        --out  fixtures/nemotron_asr_ref.json

It writes JSON with:
  * config         — the resolved encoder/decoder hyperparameters,
  * mel            — log-mel features [n_mels, n_frames] (row-major),
  * encoder_hidden — FastConformer output [t, d_model] (row-major),
  * tokens         — greedy RNN-T token ids,
  * text           — the decoded transcript,
  * key_shapes     — every state-dict tensor name -> shape (to reconcile the
                     Rust `weights::keys` module against this checkpoint).

`rlx-nemotron-asr` tests load this JSON and compare stage-by-stage.
"""

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nemo", required=True, help="path to the .nemo checkpoint")
    ap.add_argument("--wav", required=True, help="16 kHz mono wav to transcribe")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--target-lang", default="auto", help="language code or 'auto'")
    args = ap.parse_args()

    try:
        import numpy as np
        import soundfile as sf
        import torch
        import nemo.collections.asr as nemo_asr
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"missing NeMo/torch environment: {e}", file=sys.stderr)
        print("install with: pip install 'nemo_toolkit[asr]' soundfile", file=sys.stderr)
        return 2

    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cpu")
    model.eval()

    # ── full state-dict key -> shape (reconcile with weights::keys) ──
    key_shapes = {k: list(v.shape) for k, v in model.state_dict().items()}

    # ── audio -> mel via the model's own preprocessor ──
    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != model.cfg.preprocessor.sample_rate:
        print(
            f"warning: wav sr {sr} != model sr {model.cfg.preprocessor.sample_rate}",
            file=sys.stderr,
        )
    sig = torch.tensor(audio).unsqueeze(0)
    length = torch.tensor([sig.shape[1]])

    with torch.no_grad():
        mel, mel_len = model.preprocessor(input_signal=sig, length=length)
        enc, enc_len = model.encoder(audio_signal=mel, length=mel_len)
        # encoder output is [B, D, T]; transpose to [T, D].
        enc_td = enc[0].transpose(0, 1).contiguous()

    hyps = model.transcribe([args.wav], batch_size=1)
    text = hyps[0].text if hasattr(hyps[0], "text") else str(hyps[0])

    out = {
        "config": {
            "sample_rate": int(model.cfg.preprocessor.sample_rate),
            "n_mels": int(model.cfg.preprocessor.features),
            "d_model": int(model.cfg.encoder.d_model),
            "n_layers": int(model.cfg.encoder.n_layers),
            "n_heads": int(model.cfg.encoder.n_heads),
        },
        "mel": {
            "shape": list(mel[0].shape),
            "data": mel[0].flatten().tolist(),
        },
        "encoder_hidden": {
            "shape": list(enc_td.shape),
            "data": enc_td.flatten().tolist(),
        },
        "text": text,
        "key_shapes": key_shapes,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: text={text!r}, enc={list(enc_td.shape)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
