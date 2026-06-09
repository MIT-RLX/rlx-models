#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""HF reference for Voxtral parity (mistralai/Voxtral-Mini-3B-2507).

Prints a line-oriented protocol consumed by `voxtral_hf_parity.rs`:

  META <key=value ...>
  MEL <n> <floats...>
  INPUT_IDS <n> <ints...>
  STEM <n> <floats...>
  ENCODER <n> <floats...>
  PROJECTOR <n> <floats...>
  LOGITS <n> <floats...>
  GREEDY <n> <ints...>
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from mistral_common.protocol.transcription.request import TranscriptionRequest
from transformers import AutoProcessor, VoxtralForConditionalGeneration


def synth_pcm(seconds: float, sr: int = 16_000) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / sr
    return (
        0.25 * np.sin(2 * np.pi * 440.0 * t)
        + 0.10 * np.sin(2 * np.pi * 880.0 * t)
        + 0.05 * np.sin(2 * np.pi * 220.0 * t)
    ).astype(np.float32)


def pcm_to_wav_buffer(pcm: np.ndarray, sr: int = 16_000) -> io.BytesIO:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV")
    buf.seek(0)
    return buf


def load_model(model_dir: Path) -> tuple[VoxtralForConditionalGeneration, AutoProcessor]:
    repo = str(model_dir) if model_dir.is_dir() else "mistralai/Voxtral-Mini-3B-2507"
    processor = AutoProcessor.from_pretrained(repo)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()
    return model, processor


def build_inputs(
    processor: AutoProcessor,
    model_id: str,
    pcm: np.ndarray,
    language: str | None,
) -> dict:
    buf = pcm_to_wav_buffer(pcm)
    req = TranscriptionRequest.from_openai(
        {"model": model_id, "file": buf, "language": language}
    )
    tokenized = processor.tokenizer.tokenizer.encode_transcription(req)
    input_ids = torch.tensor([tokenized.tokens], dtype=torch.long)
    audio_arrays = [el.audio_array for el in tokenized.audios]
    input_features = processor._retrieve_input_features(
        audio_arrays,
        3_000,
        sampling_rate=16_000,
        padding=True,
        truncation=False,
        pad_to_multiple_of=480_000,
        return_tensors="pt",
    )
    return {
        "input_ids": input_ids,
        "input_features": input_features.float(),
        "tokens": tokenized.tokens,
    }


def emit_line(tag: str, values) -> None:
    flat = []
    for v in values:
        if isinstance(v, (float, np.floating)):
            flat.append(f"{float(v):.17g}")
        else:
            flat.append(str(int(v)))
    print(f"{tag} {len(flat)}", " ".join(flat))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--duration-sec", type=float, default=1.0)
    ap.add_argument("--language", default="en")
    ap.add_argument(
        "--probe",
        choices=("all", "encoder", "projector", "prefill", "greedy"),
        default="all",
        help="Legacy alias; all probes emit the full reference in one pass.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=3)
    args = ap.parse_args()
    _ = args.probe

    model_id = (
        args.model_dir.name
        if args.model_dir.is_dir()
        else "mistralai/Voxtral-Mini-3B-2507"
    )
    model, processor = load_model(args.model_dir)
    pcm = synth_pcm(args.duration_sec)
    inputs = build_inputs(processor, model_id, pcm, args.language)

    mel = inputs["input_features"]
    ids = inputs["input_ids"]
    batch, n_mels, mel_frames = mel.shape
    enc_seq = model.audio_tower(mel).last_hidden_state.shape[1]

    print(
        "META "
        f"batch={batch} "
        f"n_mels={n_mels} "
        f"mel_frames={mel_frames} "
        f"enc_seq={enc_seq} "
        f"n_tokens={ids.shape[1]} "
        f"n_audio={int((ids == model.config.audio_token_id).sum())} "
        f"vocab={model.config.text_config.vocab_size} "
        f"hidden={model.config.text_config.hidden_size}"
    )

    mel_flat = mel.reshape(-1).tolist()
    emit_line("MEL", mel_flat)
    emit_line("INPUT_IDS", ids.reshape(-1).tolist())

    with torch.no_grad():
        conv1 = torch.nn.functional.gelu(model.audio_tower.conv1(mel))
        emit_line("CONV1_PRE", model.audio_tower.conv1(mel).reshape(-1).tolist())
        emit_line("CONV1", conv1.reshape(-1).tolist())
        conv2 = torch.nn.functional.gelu(model.audio_tower.conv2(conv1))
        conv2t = conv2.permute(0, 2, 1)
        enc_seq = conv2t.shape[1]
        pos = model.audio_tower.embed_positions.weight[:enc_seq]
        stem = conv2t + pos
        emit_line("STEM", stem.reshape(-1).tolist())
        emit_line("CONV2", conv2.reshape(-1).tolist())

        enc_out = model.audio_tower(mel).last_hidden_state
        emit_line("ENCODER", enc_out.reshape(-1).tolist())

        grouped = enc_out.reshape(-1, model.config.audio_config.intermediate_size)
        proj_out = model.multi_modal_projector(grouped)
        emit_line("PROJECTOR", proj_out.reshape(-1).tolist())

        outputs = model(
            input_ids=ids,
            input_features=mel,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :].reshape(-1)
        emit_line("LOGITS", logits.tolist())

        gen = [ids[0].tolist()]
        past = outputs.past_key_values
        next_id = int(logits.argmax(dim=-1).item())
        for _ in range(args.max_new_tokens):
            if next_id == 0:
                break
            gen[0].append(next_id)
            step = model(
                input_ids=torch.tensor([[next_id]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
            past = step.past_key_values
            next_id = int(step.logits[:, -1, :].argmax(dim=-1).item())
        emit_line("GREEDY", gen[0])

    return 0


if __name__ == "__main__":
    sys.exit(main())
