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

"""Bake layer-by-layer ECAPA-TDNN intermediates for parity-testing the Rust port.

Inputs:
  --base-dir  Qwen3-TTS Base checkpoint
  --ref-wav   24 kHz mono WAV
  --out       safetensors path with: pcm, mel, block_{0..3}_out, mfa, asp, fc, xvec

Mel + ECAPA params match `Qwen3TTSModel.extract_speaker_embedding`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_FLASH_ATTN", "1")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True, type=Path)
    p.add_argument("--ref-wav", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import save_file
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    pcm, sr = sf.read(str(args.ref_wav), dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    assert sr == 24000, f"expected 24 kHz, got {sr}"
    print(f"pcm: {pcm.shape} dtype={pcm.dtype}", file=sys.stderr)

    model = Qwen3TTSModel.from_pretrained(
        str(args.base_dir), torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )

    # Replicate extract_speaker_embedding step-by-step.
    audio_t = torch.from_numpy(pcm).unsqueeze(0)
    mels = mel_spectrogram(
        audio_t,
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    print(f"mels: {tuple(mels.shape)}", file=sys.stderr)

    enc = model.model.speaker_encoder
    out = {
        "pcm": torch.from_numpy(pcm.copy()),
        "mel": mels.clone().detach(),
    }

    # Replicate encoder forward with intermediates.
    hidden = mels.transpose(1, 2)
    block_outs = []
    for i, layer in enumerate(enc.blocks):
        hidden = layer(hidden)
        out[f"block_{i}_out"] = hidden.clone().detach()
        block_outs.append(hidden)
    cat = torch.cat([h for h in block_outs[1:]], dim=1)
    out["concat_blocks_1plus"] = cat.clone().detach()
    mfa = enc.mfa(cat)
    out["mfa"] = mfa.clone().detach()
    asp = enc.asp(mfa)
    out["asp"] = asp.clone().detach()
    fc = enc.fc(asp)
    out["fc"] = fc.clone().detach()
    xvec = fc.squeeze(-1)
    out["xvec"] = xvec.clone().detach()

    # Sanity-check: full forward matches stepwise.
    full = enc(mels)
    diff = (full - xvec).abs().max().item()
    print(f"stepwise vs forward max-abs diff = {diff:.3e}", file=sys.stderr)
    assert diff < 1e-5, diff

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = {k: v.contiguous() for k, v in out.items()}
    save_file(out, str(args.out))
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
