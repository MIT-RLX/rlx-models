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

"""
Download Bio_ClinicalBERT, run forward on a fixed sentence, dump:
  inputs.json          (input_ids, attention_mask, token_type_ids — all length=SEQ)
  hidden_states.npy    last_hidden_state  [seq, hidden_size]  float32
  meta.json            { model, sentence, seq, hidden_size, vocab_size }

The Rust parity binary reads inputs.json + meta.json, runs the same forward,
and a comparison script diffs hidden_states.npy vs hidden_states_rlx.bin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BertForMaskedLM

DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_SENTENCE = (
    "The patient was admitted with chest pain and shortness of breath; "
    "ECG showed ST-segment elevation in the anterior leads."
)
DEFAULT_SEQ = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sentence", default=DEFAULT_SENTENCE)
    ap.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/rlx-clinicalbert-parity"),
        help="Output directory for inputs.json + hidden_states.npy",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)

    print(f"loading tokenizer + model from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    encoded = tok(
        args.sentence,
        return_tensors="pt",
        padding="max_length",
        max_length=args.seq,
        truncation=True,
        return_token_type_ids=True,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded["token_type_ids"],
        )
    # last_hidden_state: [1, seq, hidden]
    hidden = outputs.last_hidden_state[0].cpu().numpy().astype(np.float32)
    pooler_output = outputs.pooler_output[0].cpu().numpy().astype(np.float32)

    # MLM head: re-load as BertForMaskedLM to get the head modules attached.
    print(f"loading {args.model} as BertForMaskedLM for MLM head ...")
    mlm = BertForMaskedLM.from_pretrained(args.model, torch_dtype=torch.float32)
    mlm.eval()
    with torch.no_grad():
        mlm_out = mlm(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded["token_type_ids"],
        )
    mlm_logits = mlm_out.logits[0].cpu().numpy().astype(np.float32)  # [seq, vocab]

    cfg = model.config
    meta = {
        "model": args.model,
        "sentence": args.sentence,
        "seq": int(encoded["input_ids"].shape[1]),
        "hidden_size": int(cfg.hidden_size),
        "vocab_size": int(cfg.vocab_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "intermediate_size": int(cfg.intermediate_size),
        "max_position_embeddings": int(cfg.max_position_embeddings),
        "type_vocab_size": int(cfg.type_vocab_size),
        "layer_norm_eps": float(cfg.layer_norm_eps),
        "hidden_act": cfg.hidden_act,
    }

    inputs = {
        "input_ids": encoded["input_ids"][0].tolist(),
        "attention_mask": encoded["attention_mask"][0].tolist(),
        "token_type_ids": encoded["token_type_ids"][0].tolist(),
    }

    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / "inputs.json").write_text(json.dumps(inputs))
    np.save(args.out / "hidden_states.npy", hidden)
    np.save(args.out / "pooler_output.npy", pooler_output)
    np.save(args.out / "mlm_logits.npy", mlm_logits)

    print(f"meta:     {meta}")
    print(f"seq:      {meta['seq']}")
    print(f"hidden:   {hidden.shape}")
    print(f"pooler:   {pooler_output.shape}")
    print(f"mlm:      {mlm_logits.shape}")
    print(f"wrote:    {args.out}/meta.json, inputs.json, hidden_states.npy, pooler_output.npy, mlm_logits.npy")


if __name__ == "__main__":
    main()
