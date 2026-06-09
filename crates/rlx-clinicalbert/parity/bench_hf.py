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

"""HF PyTorch wall-clock baseline for the same parity sentence."""
from __future__ import annotations

import argparse
import time
from statistics import mean, median

import torch
from transformers import AutoModel, AutoTokenizer, BertForMaskedLM

DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_SENTENCE = (
    "The patient was admitted with chest pain and shortness of breath; "
    "ECG showed ST-segment elevation in the anterior leads."
)


def time_block(fn, n_iter: int) -> tuple[float, float]:
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return median(times), mean(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sentence", default=DEFAULT_SENTENCE)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())  # respect default thread count
    print(f"torch threads: {torch.get_num_threads()}")

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    mlm = BertForMaskedLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()

    encoded = tok(
        args.sentence,
        return_tensors="pt",
        padding="max_length",
        max_length=args.seq,
        truncation=True,
        return_token_type_ids=True,
    )

    # Warm-up
    with torch.no_grad():
        for _ in range(2):
            _ = enc(**encoded)
            _ = mlm(**encoded)

    # Encoder + pooler (one HF call gives both).
    def run_encoder():
        with torch.no_grad():
            _ = enc(**encoded)

    def run_mlm():
        with torch.no_grad():
            _ = mlm(**encoded)

    med_e, mean_e = time_block(run_encoder, args.iters)
    med_m, mean_m = time_block(run_mlm, args.iters)

    print(f"HF encoder+pooler  median {med_e*1000:.2f} ms  mean {mean_e*1000:.2f} ms")
    print(f"HF MLM (full fwd)  median {med_m*1000:.2f} ms  mean {mean_m*1000:.2f} ms")


if __name__ == "__main__":
    main()
