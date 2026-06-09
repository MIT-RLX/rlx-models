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

"""Python Transformers Qwen3 timing and CPU/MPS parity matrix.

Uses the same token grid as the Rust Qwen3 matrix example. The output is
CSV-compatible and intentionally mirrors the Rust columns where possible.
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM


TOKEN_POOL = [
    1, 17, 42, 314, 2718, 9001, 27182, 8128, 65535, 12345, 256, 1024,
    4096, 16384, 32768, 100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000,
    12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000,
    22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000,
    32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000, 41000,
    42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000, 51000,
    52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 60000, 61000,
    62000, 63000, 64000, 65000, 66000, 67000, 68000, 69000, 70000, 71000,
    72000, 73000, 74000, 75000, 76000, 77000, 78000, 79000, 80000, 81000,
    82000, 83000, 84000, 85000, 86000, 87000, 88000, 89000, 90000, 91000,
    92000, 93000, 94000, 95000, 96000, 97000, 98000, 99000, 100000,
    101000, 102000, 103000, 104000, 105000, 106000, 107000, 108000,
    109000, 110000,
]


def make_ids(batch: int, seq: int, device: str) -> torch.Tensor:
    rows = []
    for b in range(batch):
        off = (b * 7) % len(TOKEN_POOL)
        rows.append([TOKEN_POOL[(off + i) % len(TOKEN_POOL)] for i in range(seq)])
    return torch.tensor(rows, dtype=torch.long, device=device)


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def top1_match(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    aa = a.reshape(-1, a.shape[-1]).argmax(dim=-1)
    bb = b.reshape(-1, b.shape[-1]).argmax(dim=-1)
    return int((aa == bb).sum().item()), int(aa.numel())


def metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float, float, int, int]:
    af = a.detach().float().cpu().reshape(a.shape[0], -1)
    bf = b.detach().float().cpu().reshape(b.shape[0], -1)
    diff = (af - bf).abs()
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=-1)
    t1, total = top1_match(a.detach().cpu(), b.detach().cpu())
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(cos.mean().item()),
        float(cos.min().item()),
        t1,
        total,
    )


@torch.no_grad()
def run_case(model, device: str, batch: int, seq: int, keep: bool, reps: int):
    x = make_ids(batch, seq, device)
    kwargs = {"input_ids": x, "use_cache": False}
    if keep:
        kwargs["logits_to_keep"] = 1
    for _ in range(2):
        out = model(**kwargs).logits
    sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = model(**kwargs).logits
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, times


def emit_row(
    backend: str,
    mode: str,
    batch: int,
    seq: int,
    shape_ok: bool,
    metric_values: tuple[float, float, float, float, int, int],
    times: Iterable[float],
    status: str,
    message: str,
) -> None:
    times = list(times)
    print(
        "prefill,python,{backend},{mode},{batch},{seq},{shape_ok},"
        "{max_abs:.6f},{mean_abs:.6f},{cos_mean:.7f},{cos_min:.7f},"
        "{top1_match},{top1_total},{min_ms:.1f},{median_ms:.1f},{status},{message}".format(
            backend=backend,
            mode=mode,
            batch=batch,
            seq=seq,
            shape_ok=str(shape_ok).lower(),
            max_abs=metric_values[0],
            mean_abs=metric_values[1],
            cos_mean=metric_values[2],
            cos_min=metric_values[3],
            top1_match=metric_values[4],
            top1_total=metric_values[5],
            min_ms=min(times),
            median_ms=statistics.median(times),
            status=status,
            message=message,
        )
    )


def main() -> None:
    model_dir = os.environ.get("RLX_QWEN3_DIR", "/Users/Shared/rlx/weights/Qwen3-0.6B")
    reps = max(1, int(os.environ.get("RLX_QWEN3_MATRIX_REPS", "3")))
    target = "mps" if torch.backends.mps.is_available() else "cpu"

    print(
        "kind,impl,backend,mode,batch,seq,shape_ok,max_abs,mean_abs,cos_mean,cos_min,"
        "top1_match,top1_total,min_ms,median_ms,status,message"
    )
    cpu_model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32
    ).to("cpu").eval()
    target_model = cpu_model if target == "cpu" else AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32
    ).to(target).eval()

    for batch in (1, 2, 4):
        for seq in (8, 32, 64, 128):
            for keep, mode in ((False, "full"), (True, "last")):
                ref, _ = run_case(cpu_model, "cpu", batch, seq, keep, 1)
                got, times = run_case(target_model, target, batch, seq, keep, reps)
                shape_ok = tuple(ref.shape) == tuple(got.shape)
                vals = metrics(got, ref)
                status = "ok" if shape_ok and vals[4] == vals[5] else "fail"
                emit_row(target.upper(), mode, batch, seq, shape_ok, vals, times, status, "cpu_ref")


if __name__ == "__main__":
    main()
