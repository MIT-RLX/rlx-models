#!/usr/bin/env python3
"""Sentence-transformers reference embeddings, in f32.

Writes JSON with, per sentence, the token ids the model saw and the final
(pooled, projected, normalized) embedding, for the env-gated parity tests:

    python3 scripts/embedding_hf_reference.py weights/embed/embeddinggemma-300m \
        /tmp/embeddinggemma.ref.json --prompt "task: search result | query: "
    python3 scripts/embedding_hf_reference.py weights/embed/jina-embeddings-v5-text-nano \
        /tmp/jina-v5-nano.ref.json --task retrieval --prompt-name query

Needs `sentence-transformers` (and `peft` for jina-v5's task adapters).
"""

import argparse
import json

import torch
from sentence_transformers import SentenceTransformer

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Which planet is known as the Red Planet?",
    "Mars, often called the Red Planet, has a thin atmosphere of carbon dioxide.",
    "Rust guarantees memory safety without a garbage collector.",
    # ~700 tokens: past EmbeddingGemma's 512-token sliding window, so its
    # sliding and full-attention layers see different masks.
    " ".join(
        f"Paragraph {i}: the archive keeps letters, maps and ledgers from the harbour office, "
        "sorted by year and by the ship that carried them."
        for i in range(30)
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--prompt", default=None, help="literal prompt prefix")
    ap.add_argument("--prompt-name", default=None, help="named prompt from the model config")
    ap.add_argument("--task", default=None, help="jina-v5 task adapter")
    args = ap.parse_args()

    kwargs = {"device": "cpu", "trust_remote_code": True, "model_kwargs": {"dtype": torch.float32}}
    if args.task:
        kwargs["model_kwargs"]["default_task"] = args.task
    model = SentenceTransformer(args.model_dir, **kwargs).float().eval()

    enc_kwargs = {"convert_to_numpy": True, "batch_size": 1}
    if args.prompt is not None:
        enc_kwargs["prompt"] = args.prompt
    if args.prompt_name is not None:
        enc_kwargs["prompt_name"] = args.prompt_name
    if args.task:
        enc_kwargs["task"] = args.task

    prompt = args.prompt
    if prompt is None and args.prompt_name is not None:
        prompt = model.prompts.get(args.prompt_name, "")
    prompt = prompt or ""

    rows = []
    for s in SENTENCES:
        ids = model.tokenize([prompt + s])["input_ids"][0].tolist()
        emb = model.encode([s], **enc_kwargs)[0].tolist()
        rows.append({"text": s, "input_ids": ids, "embedding": emb})
    json.dump({"prompt": prompt, "task": args.task, "rows": rows}, open(args.out, "w"))
    print(f"wrote {args.out}: {len(rows)} rows, dim {len(rows[0]['embedding'])}, ids[0]={rows[0]['input_ids'][:12]}")


if __name__ == "__main__":
    main()
