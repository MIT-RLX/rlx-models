#!/usr/bin/env python3
"""Capture Inflect-Nano-v1 parity fixtures for the RLX Rust port.

For each corpus sentence, runs the *reference* PyTorch pipeline end-to-end and
records every stage so the Rust crate can be tested at each boundary:

  - text -> token IDs           (frontend parity)
  - token IDs -> encoder/durations/mel   (acoustic parity)
  - mel -> waveform             (vocoder parity)
  - text -> waveform            (e2e parity)

Outputs a JSON manifest + per-sentence safetensors (mel, wav, encoded, durations).
Deterministic: MicroFastSpeech.infer and the vocoder use no sampling.

    .venv-inflect/bin/python scripts/inflect_nano_reference.py \
        --repo /tmp/inflect-nano --out weights/inflect-nano-rlx/fixtures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

CORPUS = [
    "The weather is nice today, and I feel very relaxed.",
    "Hello, world!",
    "Dr. Smith paid $42.50 at 3:15 pm.",
    "In 1999 we found 1024 reasons.",
    "Zephyrization glorptastic woogle.",
    "It costs 7 dollars, maybe 8.",
]


def load_models(repo: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from train_hifigan_oracle_v1 import HifiGanGenerator, make_config
    from train_inflect_micro_fastspeech_v3_pitch import MicroFastSpeech, MicroFastSpeechConfig

    ack = torch.load(repo / "weights" / "inflect_nano_v1_acoustic.pt",
                     map_location=device, weights_only=True)
    acfg = MicroFastSpeechConfig(**ack["config"])
    acoustic = MicroFastSpeech(acfg).to(device)
    acoustic.load_state_dict(ack["model"])
    acoustic.eval()
    speakers = ack.get("speakers") or {"mark": 0}

    vck = torch.load(repo / "weights" / "inflect_nano_v1_vocoder.pt",
                     map_location=device, weights_only=True)
    vcfg = make_config((vck.get("config") or {}).get("variant", "snake_v2mid"))
    vocoder = HifiGanGenerator(vcfg).to(device)
    vocoder.load_state_dict(vck["generator"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    return acoustic, vocoder, speakers


def text_to_tokens(repo: Path, text: str):
    from tiny_tts.nn import commons
    from tiny_tts.text import phonemes_to_ids
    from tiny_tts.text.english import grapheme_to_phoneme, normalize_text
    from tiny_tts.utils import ADD_BLANK
    from tinytts_text_cleaning import clean_tinytts_text

    cleaned = clean_tinytts_text(text)
    normalized = normalize_text(cleaned)
    phones, tones, _ = grapheme_to_phoneme(normalized)
    phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")
    if ADD_BLANK:
        phone_ids = commons.insert_blanks(phone_ids, 0)
        tone_ids = commons.insert_blanks(tone_ids, 0)
        lang_ids = commons.insert_blanks(lang_ids, 0)
    return phones, tones, phone_ids, tone_ids, lang_ids


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    acoustic, vocoder, speakers = load_models(args.repo, device)
    spk = int(speakers.get("mark", next(iter(speakers.values()), 0)))

    manifest = {"sample_rate": 24000, "speaker": spk, "cases": []}
    for i, text in enumerate(CORPUS):
        phones, tones, phone_ids, tone_ids, lang_ids = text_to_tokens(args.repo, text)
        phone = torch.LongTensor(phone_ids).unsqueeze(0)
        tone = torch.LongTensor(tone_ids).unsqueeze(0)
        lang = torch.LongTensor(lang_ids).unsqueeze(0)
        speaker = torch.LongTensor([spk])

        # Intermediate: encoder output + predicted durations (mirror infer()).
        token_mask = torch.ones_like(phone, dtype=torch.bool)
        encoded = acoustic.encode(phone, tone, lang, speaker, token_mask)
        log_dur, energy, bright, pitch = acoustic.predict_prosody(encoded, token_mask)
        pred_dur = torch.expm1(log_dur).clamp(0, 80) * 1.0
        durations = torch.round(pred_dur).long().clamp_min(1).masked_fill(~token_mask, 0)

        mel = acoustic.infer(phone, tone, lang, speaker)  # [1, 80, T]
        wav = vocoder(mel).squeeze().detach().cpu().numpy().astype(np.float32)

        tensors = {
            "encoded": encoded.squeeze(0).cpu().numpy().astype(np.float32),
            "log_dur": log_dur.squeeze(0).cpu().numpy().astype(np.float32),
            "durations": durations.squeeze(0).cpu().numpy().astype(np.int64),
            "mel": mel.squeeze(0).cpu().numpy().astype(np.float32),
            "wav": wav,
        }
        save_file(tensors, str(args.out / f"case_{i}.safetensors"))
        manifest["cases"].append({
            "index": i,
            "text": text,
            "phones": phones,
            "tones": tones,
            "phone_ids": [int(x) for x in phone_ids],
            "tone_ids": [int(x) for x in tone_ids],
            "lang_ids": [int(x) for x in lang_ids],
            "mel_shape": list(mel.shape[1:]),
            "wav_len": int(wav.shape[-1]),
            "fixture": f"case_{i}.safetensors",
        })
        print(f"[{i}] '{text[:40]}' tokens={len(phone_ids)} frames={mel.shape[-1]} wav={wav.shape[-1]}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(CORPUS)} fixtures to {args.out}")


if __name__ == "__main__":
    main()
