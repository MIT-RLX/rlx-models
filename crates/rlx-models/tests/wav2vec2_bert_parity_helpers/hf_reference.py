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

"""HF reference for Wav2Vec2-BERT encoder parity (facebook/w2v-bert-2.0).

Usage (from repo root):
  python3 rlx-models/tests/wav2vec2_bert_parity_helpers/hf_reference.py \\
    --model-dir /path/to/w2v-bert-2.0 \\
    --duration-sec 1.0 \\
    --seq 128

Prints a simple text protocol:
  SHAPE <batch> <seq> <hidden> <feat_dim>
  FEAT <n> floats...
  MASK <n> floats...
  HIDDEN <n> floats...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2BertModel
from transformers.models.seamless_m4t.feature_extraction_seamless_m4t import (
    SeamlessM4TFeatureExtractor,
)


def synth_waveform(sr: int, seconds: float) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / sr
    return (
        0.25 * np.sin(2 * np.pi * 440.0 * t)
        + 0.10 * np.sin(2 * np.pi * 880.0 * t)
        + 0.05 * np.sin(2 * np.pi * 220.0 * t)
    ).astype(np.float32)


def load_model(model_dir: Path) -> Wav2Vec2BertModel:
    if model_dir.is_dir():
        return Wav2Vec2BertModel.from_pretrained(str(model_dir))
    return Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path("facebook/w2v-bert-2.0"),
        help="HF model directory or repo id",
    )
    ap.add_argument("--duration-sec", type=float, default=1.0)
    ap.add_argument(
        "--seq",
        type=int,
        default=0,
        help="Right-pad/truncate frames to this seq (0 = native length)",
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=-1,
        help="Encoder layers to keep (-1 = full model, 0 = feature projection only)",
    )
    ap.add_argument(
        "--probe",
        choices=[
            "enc_in",
            "after_ffn1",
            "after_attn",
            "after_conv",
            "after_ffn2",
            "final",
        ],
        help="Return an intermediate activation from encoder layer 0 instead of last_hidden_state",
    )
    ap.add_argument("--config-only", action="store_true")
    args = ap.parse_args()

    sr = 16_000
    wav = synth_waveform(sr, args.duration_sec)
    fe = SeamlessM4TFeatureExtractor(
        feature_size=80,
        sampling_rate=sr,
        num_mel_bins=80,
        padding_value=1.0,
        stride=2,
    )
    out = fe(wav, return_tensors="pt", padding=False, do_normalize_per_mel_bins=True)
    feat = out["input_features"].numpy().astype(np.float32)  # [1, S, 160]
    mask = out["attention_mask"].numpy().astype(np.float32)  # [1, S]

    seq = feat.shape[1]
    if args.seq > 0:
        target = args.seq
        dim = feat.shape[2]
        if seq < target:
            pad = np.full((1, target - seq, dim), 1.0, dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=1)
            mask = np.concatenate(
                [mask, np.zeros((1, target - seq), dtype=np.float32)], axis=1
            )
        else:
            feat = feat[:, :target, :]
            mask = mask[:, :target]
        seq = target

    feat = torch.from_numpy(feat)
    mask = torch.from_numpy(mask)

    if args.config_only:
        cfg_path = args.model_dir / "config.json"
        if cfg_path.exists():
            print(json.dumps(json.loads(cfg_path.read_text())))
        else:
            from transformers import AutoConfig

            print(json.dumps(AutoConfig.from_pretrained(str(args.model_dir)).to_dict()))
        return 0

    model = load_model(args.model_dir)
    model.eval()
    if args.num_layers >= 0:
        import torch.nn as nn

        model.encoder.layers = nn.ModuleList(
            list(model.encoder.layers[: args.num_layers])
        )
    with torch.no_grad():
        fp_out, _ = model.feature_projection(feat)
        enc_in = fp_out.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        if args.probe == "enc_in":
            hidden = enc_in
        elif args.probe and args.num_layers >= 1:
            attn_mask = (1.0 - mask)[:, None, None, :] * torch.finfo(torch.float32).min
            attn_mask = attn_mask.expand(-1, 1, seq, seq)
            L = model.encoder.layers[0]
            x = enc_in
            residual = x
            x = L.ffn1_layer_norm(x)
            x = L.ffn1(x)
            after_ffn1 = x * 0.5 + residual
            if args.probe == "after_ffn1":
                hidden = after_ffn1
            else:
                residual = after_ffn1
                x = L.self_attn_layer_norm(after_ffn1)
                sa, _ = L.self_attn(x, attention_mask=attn_mask)
                sa = L.self_attn_dropout(sa)
                after_attn = sa + residual
                if args.probe == "after_attn":
                    hidden = after_attn
                else:
                    residual = after_attn
                    after_conv = residual + L.conv_module(after_attn, attention_mask=mask)
                    if args.probe == "after_conv":
                        hidden = after_conv
                    else:
                        residual = after_conv
                        x = L.ffn2_layer_norm(after_conv)
                        x = L.ffn2(x)
                        after_ffn2 = x * 0.5 + residual
                        if args.probe == "after_ffn2":
                            hidden = after_ffn2
                        else:
                            hidden = L.final_layer_norm(after_ffn2)
        else:
            hidden = model(
                input_features=feat,
                attention_mask=mask,
            ).last_hidden_state

    b, s, h = hidden.shape
    fd = feat.shape[2]
    hidden = hidden.detach().numpy().astype(np.float32)
    print(f"SHAPE {b} {s} {h} {fd}")
    print("FEAT", " ".join(f"{v:.9g}" for v in feat.numpy().reshape(-1)))
    print("MASK", " ".join(f"{v:.9g}" for v in mask.numpy().reshape(-1)))
    print("HIDDEN", " ".join(f"{v:.9g}" for v in hidden.reshape(-1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
