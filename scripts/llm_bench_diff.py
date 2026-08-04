#!/usr/bin/env python3
# RLX models — LLM benchmark harness: per-document agreement diff.
# SPDX-License-Identifier: GPL-3.0-only
#
# Compares the per-document prediction dumps written by `rlx-llm-bench --dump`
# and `scripts/llm_bench_ref.py --dump`. When both ran the SAME model weights on
# the SAME docs, high agreement proves the two harnesses implement identical,
# correct scoring (aggregate-accuracy matching alone can coincide by luck).
#
#   python3 scripts/llm_bench_diff.py mmlu  rlx_mmlu.jsonl ref_mmlu.jsonl
#   python3 scripts/llm_bench_diff.py gsm8k rlx_gsm8k.jsonl ref_gsm8k.jsonl

import json
import sys


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def num_eq(a, b):
    if a is None or b is None:
        return a == b
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return str(a) == str(b)


def mmlu(rlx_path, ref_path):
    r, f = load(rlx_path), load(ref_path)
    n = min(len(r), len(f))
    best = sum(1 for a, b in zip(r, f) if a["best"] == b["best"])
    norm = sum(1 for a, b in zip(r, f) if a["best_norm"] == b["best_norm"])
    acc_r = sum(1 for a in r[:n] if a["best"] == a["gold"]) / n
    acc_f = sum(1 for b in f[:n] if b["best"] == b["gold"]) / n
    print(f"MMLU  n={n}")
    print(f"  per-doc argmax agreement (best):      {best}/{n}  ({best/n:.1%})")
    print(f"  per-doc argmax agreement (best_norm): {norm}/{n}  ({norm/n:.1%})")
    print(f"  accuracy  rlx={acc_r:.4f}  torch={acc_f:.4f}")
    for a, b in zip(r, f):
        if a["best"] != b["best"]:
            print(f"    disagree i={a['i']:>2}  rlx={a['best']} torch={b['best']} gold={a['gold']}")


def gsm8k(rlx_path, ref_path):
    r, f = load(rlx_path), load(ref_path)
    n = min(len(r), len(f))
    pred = sum(1 for a, b in zip(r, f) if num_eq(a["pred"], b["pred"]))
    flag = sum(1 for a, b in zip(r, f) if a["correct"] == b["correct"])
    acc_r = sum(1 for a in r[:n] if a["correct"]) / n
    acc_f = sum(1 for b in f[:n] if b["correct"]) / n
    print(f"GSM8K n={n}")
    print(f"  per-doc predicted-number agreement:   {pred}/{n}  ({pred/n:.1%})")
    print(f"  per-doc correct-flag agreement:       {flag}/{n}  ({flag/n:.1%})")
    print(f"  accuracy  rlx={acc_r:.4f}  torch={acc_f:.4f}")
    for a, b in zip(r, f):
        if not num_eq(a["pred"], b["pred"]):
            print(f"    disagree i={a['i']:>2}  rlx={a['pred']} torch={b['pred']} gold={a['gold']}")


if __name__ == "__main__":
    task, rlx_path, ref_path = sys.argv[1], sys.argv[2], sys.argv[3]
    (mmlu if task == "mmlu" else gsm8k)(rlx_path, ref_path)
