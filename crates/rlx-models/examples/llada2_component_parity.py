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

"""Emit JSON lines for RLX ↔ TIDE PyTorch component parity (standalone torch refs)."""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def ramp(n: int, scale: float) -> list[float]:
    return [0.001 + scale * i * 0.01 for i in range(n)]


def get_num_transfer_tokens(block_length: int, steps: int) -> list[int]:
    if steps == 0:
        return []
    base = block_length // steps
    remainder = block_length % steps
    out = [base] * steps
    for i in range(remainder):
        out[i] += 1
    return out


def group_limited_topk(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = scores.size()
    epg = num_experts // n_group
    group_scores = scores.view(num_tokens, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, epg)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    return torch.topk(masked_scores, k=top_k, dim=-1)


def gate_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    top_k: int,
    routed_scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden.float(), weight.float())
    scores = torch.sigmoid(logits.float())
    scores_for_routing = scores + expert_bias
    _, topk_idx = group_limited_topk(scores_for_routing, n_group, topk_group, top_k)
    gathered = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
    if top_k > 1:
        topk_weight = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        topk_weight = gathered
    topk_weight = topk_weight * routed_scaling
    return topk_idx, topk_weight


def main() -> int:
    for block_length, steps in [(32, 32), (10, 3), (7, 4)]:
        emit(
            {
                "test": "transfer_schedule",
                "block_length": block_length,
                "steps": steps,
                "schedule": get_num_transfer_tokens(block_length, steps),
            }
        )

    seq_len, block_length = 8, 4
    num_blocks = (seq_len + block_length - 1) // block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, dtype=torch.float32))
    attn = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
        .log()[:, :, :seq_len, :seq_len]
    )
    flat = attn.reshape(-1)
    emit(
        {
            "test": "block_mask",
            "seq_len": seq_len,
            "block_length": block_length,
            "mask": ["-inf" if math.isinf(v) else float(v) for v in flat.tolist()],
        }
    )

    scores = torch.tensor(
        [[0.1, 0.9, 0.2, 0.8], [0.5, 0.5, 0.5, 0.5]],
        dtype=torch.float32,
    )
    probs, idx = group_limited_topk(scores, n_group=2, topk_group=1, top_k=2)
    emit(
        {
            "test": "group_limited_topk",
            "indices": [int(x) for x in idx.reshape(-1).tolist()],
            "probs": [float(x) for x in probs.reshape(-1).tolist()],
        }
    )

    h, e, rows = 16, 4, 4
    hidden = torch.tensor(
        [[0.01 * i for i in range(h)] for _ in range(rows)], dtype=torch.float32
    )
    # RLX stores router row-major [hidden, expert]; F.linear weight is [expert, hidden].
    weight = torch.tensor(ramp(h * e, 1.1), dtype=torch.float32).reshape(h, e).t()
    bias = torch.tensor(ramp(e, 0.01), dtype=torch.float32)
    top_idx, top_weight = gate_forward(
        hidden, weight, bias, n_group=2, topk_group=1, top_k=2, routed_scaling=2.5
    )
    emit(
        {
            "test": "gate_forward",
            "indices": [int(x) for x in top_idx.reshape(-1).tolist()],
            "weights": [float(x) for x in top_weight.reshape(-1).tolist()],
        }
    )

    for step in range(4):
        refresh = (1 == 0) or (step % 2 == 0)
        emit({"test": "refresh", "num_block": 1, "prefill": 0, "step": step, "refresh": refresh})

    emit({"test": "done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
