#!/usr/bin/env python3
"""Dump an mlx-lm reference (oracle) for any mlx-community checkpoint.

Runs a full prefill through mlx-lm and writes, into the model dir:
  - oracle.json  : {prompt_ids, prefill_argmax, greedy_gen_ids, greedy_text}
  - oracle_prefill_last_logits.npy : last-token logits [vocab] (float32)

These are what rlx's generic packed decoder is validated against.

Usage:
  PYTHONPATH=/Users/Shared/mlx-lm python3 scripts/mlx_oracle_dump.py \
      <model_dir> [--prompt "The capital of France is"] [--ngen 6]
"""
import argparse, json, os, sys
import numpy as np
import mlx.core as mx
from mlx_lm import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ngen", type=int, default=6)
    ap.add_argument("--chat", action="store_true",
                    help="wrap the prompt in the tokenizer chat template")
    args = ap.parse_args()

    model, tokenizer = load(args.model_dir)

    if args.chat and tokenizer.chat_template is not None:
        msgs = [{"role": "user", "content": args.prompt}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    else:
        # Raw prompt, no special tokens added by default — mlx-lm's model
        # forward sees exactly these ids, matching what rlx feeds.
        ids = tokenizer.encode(args.prompt)
    ids = list(map(int, ids))

    x = mx.array([ids])
    logits = model(x)                    # [1, seq, vocab]
    # Cast bf16→f32 inside mlx; np.array can't read a bf16 mlx buffer directly.
    last = logits[0, -1, :].astype(mx.float32)
    mx.eval(last)
    last_np = np.array(last, dtype=np.float32)
    argmax = int(np.argmax(last_np))

    # Greedy continuation (fresh full-prefill each step — reference only).
    gen = []
    cur = list(ids)
    for _ in range(args.ngen):
        lg = model(mx.array([cur]))[0, -1, :].astype(mx.float32)
        mx.eval(lg)
        nxt = int(np.argmax(np.array(lg, dtype=np.float32)))
        gen.append(nxt)
        cur.append(nxt)

    out = {
        "prompt_ids": ids,
        "prefill_argmax": argmax,
        "greedy_gen_ids": gen,
        "greedy_text": tokenizer.decode(gen),
        "vocab": int(last_np.shape[0]),
    }
    d = args.model_dir
    with open(os.path.join(d, "oracle.json"), "w") as f:
        json.dump(out, f)
    np.save(os.path.join(d, "oracle_prefill_last_logits.npy"), last_np)
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
