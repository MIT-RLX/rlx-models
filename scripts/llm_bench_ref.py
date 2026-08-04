#!/usr/bin/env python3
# RLX models — LLM benchmark harness: Python reference.
# SPDX-License-Identifier: GPL-3.0-only
#
# Reference implementation of rlx-llm-bench's MMLU / GSM8K scoring, using
# HuggingFace transformers. It mirrors `crates/rlx-llm-bench/src/quality/mod.rs`
# EXACTLY — same prompt rendering, same continuation scoring, same GSM8K few-shot
# and last-number extraction — so accuracy on the SAME cached JSONL docs is a
# true apples-to-apples check of the Rust harness (and of rlx's model inference
# when run on the same weights).
#
# Usage:
#   python3 scripts/llm_bench_ref.py mmlu  --model <dir> --data <jsonl> [--mode letter|cloze] [--limit N]
#   python3 scripts/llm_bench_ref.py gsm8k --model <dir> --data <jsonl> [--limit N] [--max-new N]

import argparse
import json
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = [chr(ord("A") + i) for i in range(26)]

# Mirrors DEFAULT_GSM8K_FEWSHOT in quality/mod.rs (verbatim, incl. trailing blank line).
DEFAULT_GSM8K_FEWSHOT = (
    "Question: There are 15 trees in the grove. Grove workers will plant trees today. After they are done there will be 21 trees. How many trees did they plant?\n"
    "Answer: There were 15 trees, then 21 trees, so they planted 21 - 15 = 6 trees. The answer is 6.\n\n"
    "Question: If there are 3 cars in the parking lot and 2 more arrive, how many cars are in the parking lot?\n"
    "Answer: There are 3 cars and 2 more arrive, so 3 + 2 = 5 cars. The answer is 5.\n\n"
    "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
    "Answer: Together they had 32 + 42 = 74. After eating 35, they have 74 - 35 = 39. The answer is 39.\n\n"
    "Question: Shawn has five toys. For Christmas he got two toys each from his mom and dad. How many toys does he have now?\n"
    "Answer: He starts with 5. He gets 2 from mom and 2 from dad, so 2 + 2 = 4 more. 5 + 4 = 9. The answer is 9.\n\n"
)


def load(model_dir, dtype):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    model.eval()
    return tok, model


def encode(tok, text):
    # add_special_tokens=False to match rlx `encode(text, false)`.
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]


@torch.no_grad()
def logits_for(model, ids):
    """Full-sequence logits [seq, vocab] for a 1-D id tensor."""
    out = model(ids.unsqueeze(0))
    return out.logits[0]  # [seq, vocab]


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── MMLU (mirror render_context / choice_continuations / score_mc) ───────────

def render_context(doc, mode):
    ctx = ""
    subj = doc.get("subject")
    if subj:
        ctx += f"The following is a multiple choice question about {subj}.\n\n"
    ctx += doc["question"]
    if mode == "cloze":
        ctx += "\nAnswer:"
    else:  # letter
        ctx += "\n"
        for i, c in enumerate(doc["choices"]):
            ctx += f"{LETTERS[i] if i < len(LETTERS) else '?'}. {c}\n"
        ctx += "Answer:"
    return ctx


def continuations(doc, mode):
    if mode == "cloze":
        return [f" {c}" for c in doc["choices"]]
    return [f" {LETTERS[i]}" for i in range(len(doc["choices"]))]


def parse_answer(a):
    if isinstance(a, int):
        return a
    s = str(a).strip()
    if s.isdigit():
        return int(s)
    return ord(s[0].upper()) - ord("A")


@torch.no_grad()
def run_mmlu(tok, model, docs, mode, limit, dump=None):
    docs = docs[: (limit or len(docs))]
    n = correct = correct_norm = 0
    preds = []
    for doc in docs:
        ctx = encode(tok, render_context(doc, mode))
        gold = parse_answer(doc["answer"])
        scores, scores_norm = [], []
        for cont_txt in continuations(doc, mode):
            cont = encode(tok, cont_txt)
            seq = torch.cat([ctx, cont])
            lg = logits_for(model, seq)  # [seq, vocab]
            lp = F.log_softmax(lg.float(), dim=-1)
            # logprob of cont[i] is at position (len(ctx)+i-1) predicting it.
            total = 0.0
            start = len(ctx) - 1
            for i in range(len(cont)):
                total += lp[start + i, cont[i]].item()
            scores.append(total)
            scores_norm.append(total / len(cont))
        best = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_norm = int(max(range(len(scores_norm)), key=lambda i: scores_norm[i]))
        n += 1
        if best == gold:
            correct += 1
        if best_norm == gold:
            correct_norm += 1
        preds.append({"i": len(preds), "gold": gold, "best": best, "best_norm": best_norm})
    acc = correct / max(n, 1)
    acc_norm = correct_norm / max(n, 1)
    print(f"REF kind=mmlu n={n} acc={acc:.4f} acc_norm={acc_norm:.4f} mode={mode}")
    if dump:
        with open(dump, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"[ref] wrote {len(preds)} mmlu predictions -> {dump}", file=sys.stderr)


# ── GSM8K (mirror run_gsm8k + gsm8k.rs extraction) ───────────────────────────

def numbers(s):
    return [m.replace(",", "") for m in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", s)]


def extract_gold(answer):
    if "####" in answer:
        tail = answer.split("####", 1)[1]
        ns = numbers(tail)
        if ns:
            return ns[0]
    ns = numbers(answer)
    return ns[-1] if ns else None


def extract_pred(text):
    ns = numbers(text)
    return ns[-1] if ns else None


def answers_match(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a == b


@torch.no_grad()
def run_gsm8k(tok, model, docs, limit, max_new, dump=None):
    docs = docs[: (limit or len(docs))]
    eos = [i for i in (151643, 151645) if i < model.config.vocab_size]
    n = correct = 0
    preds = []
    for doc in docs:
        prompt = DEFAULT_GSM8K_FEWSHOT + f"Question: {doc['question']}\nAnswer:"
        ids = encode(tok, prompt).unsqueeze(0)
        out = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos or None,
            pad_token_id=eos[0] if eos else tok.eos_token_id,
        )
        gen = out[0, ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        span = text.split("Question:", 1)[0]
        pred, gold = extract_pred(span), extract_gold(doc["answer"])
        ok = pred is not None and gold is not None and answers_match(pred, gold)
        n += 1
        if ok:
            correct += 1
        preds.append({"i": len(preds), "gold": gold, "pred": pred, "correct": ok})
    acc = correct / max(n, 1)
    print(f"REF kind=gsm8k n={n} acc={acc:.4f}")
    if dump:
        with open(dump, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"[ref] wrote {len(preds)} gsm8k predictions -> {dump}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["mmlu", "gsm8k"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", default="letter", choices=["letter", "cloze"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tok, model = load(args.model, dtype)
    docs = read_jsonl(args.data)
    print(f"[ref] loaded {args.model} dtype={args.dtype} device=cpu docs={len(docs)}", file=sys.stderr)
    if args.task == "mmlu":
        run_mmlu(tok, model, docs, args.mode, args.limit, args.dump)
    else:
        run_gsm8k(tok, model, docs, args.limit, args.max_new, args.dump)


if __name__ == "__main__":
    main()
