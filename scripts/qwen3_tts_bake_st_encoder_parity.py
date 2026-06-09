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

"""Bake layer-by-layer intermediates for the speech tokenizer encoder.

Outputs a single safetensors file with:
  - pcm: input WAV (mono float32, 24 kHz)
  - input: input_values reshape (1, 1, T)
  - enc_layer_{i}_out: hidden states after each encoder.layers[i]
  - enc_out: final encoder convolutional output (before downsample)
  - downsampled: after `encoder.downsample` (after Mimi encoder transformer + downsample conv)
  - codes: final discrete codec frames (T x num_quantizers) integer tensor

Used for layer-by-layer parity tests in the Rust port.
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
    from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiResnetBlock
    from qwen_tts import Qwen3TTSModel

    pcm, sr = sf.read(str(args.ref_wav), dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    assert sr == 24000, f"expected 24 kHz, got {sr}"
    print(f"pcm: {pcm.shape}", file=sys.stderr)

    model = Qwen3TTSModel.from_pretrained(
        str(args.base_dir), torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )
    st = model.model.speech_tokenizer.model
    encoder = st.encoder.encoder  # MimiEncoder
    input_values = torch.from_numpy(pcm).unsqueeze(0).unsqueeze(0)  # [B=1, C=1, T]

    out = {
        "pcm": torch.from_numpy(pcm.copy()),
        "input_values": input_values.clone().detach(),
    }

    # Layer-by-layer through MimiEncoder.
    hidden = input_values
    for i, layer in enumerate(encoder.layers):
        if isinstance(layer, (MimiConv1d, MimiResnetBlock)):
            hidden = layer(hidden, padding_cache=None)
        else:
            hidden = layer(hidden)
        # Skip recording ELU outputs to keep file lean (just the conv-bearing layers).
        out[f"enc_layer_{i}_out"] = hidden.clone().detach()
    out["enc_out"] = hidden.clone().detach()

    # Apply encoder_transformer manually so we can capture its output and per-layer
    # intermediates.
    transformer = st.encoder.encoder_transformer
    embeds = hidden.transpose(1, 2)
    out["pre_transformer"] = embeds.clone().detach()
    # Walk transformer layers individually so we can dump per-layer outputs.
    h = embeds
    T = h.shape[1]
    position_ids = torch.arange(T).unsqueeze(0)
    # Build sliding-window causal mask explicitly: shape [B=1, 1, T, T], fill -inf above diag
    # or beyond sliding window (window=250). With T=122 < 250, this reduces to a pure causal mask.
    from transformers.masking_utils import create_sliding_window_causal_mask
    cfg = transformer.config
    attn_mask = create_sliding_window_causal_mask(
        cfg, input_embeds=h, attention_mask=None, cache_position=position_ids.squeeze(0),
        past_key_values=None, position_ids=position_ids,
    )
    print(f"  attn_mask shape: {None if attn_mask is None else tuple(attn_mask.shape)}", file=sys.stderr)
    for i, layer in enumerate(transformer.layers):
        outs = layer(
            hidden_states=h,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )
        h = outs[0] if isinstance(outs, tuple) else outs
        out[f"tf_layer_{i}_out"] = h.clone().detach()
    post_t = h
    out["post_transformer"] = post_t.clone().detach()
    # Downsample takes [B, C, T]; output [B, C, T/2].
    pre_ds = post_t.transpose(1, 2)
    out["pre_downsample"] = pre_ds.clone().detach()
    ds = st.encoder.downsample(pre_ds, padding_cache=None)
    out["post_downsample"] = ds.clone().detach()

    # Final codes from the full encode path.
    enc_out = st.encoder.encode(input_values=input_values, return_dict=True)
    out["audio_codes"] = enc_out.audio_codes.clone().detach().to(torch.int64)

    # Sanity print
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = {k: v.contiguous() for k, v in out.items()}
    save_file(out, str(args.out))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
