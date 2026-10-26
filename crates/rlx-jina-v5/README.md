# rlx-jina-v5

[jinaai/jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano)
on RLX: a 12-layer EuroBERT trunk (Llama-shaped layers, no causal mask) with
one of four task LoRA adapters merged in, pooled at the **last** token and
L2-normalized. 768-wide output.

| | |
|---|---|
| Parameters | 239 M |
| Checkpoint | 424 MB bf16 safetensors + 4 × 13.6 MB adapters |
| Context | 8192 tokens |
| Tasks | `retrieval`, `text-matching`, `clustering`, `classification` |

```sh
hf download jinaai/jina-embeddings-v5-text-nano --local-dir weights/embed/jina-embeddings-v5-text-nano
cargo run -p rlx-jina-v5 --release -- --weights weights/embed/jina-embeddings-v5-text-nano \
    --task retrieval --text "Which planet is known as the Red Planet?" \
    --prompt document --text "Mars, often called the Red Planet, has a thin atmosphere."
```

## Notes

- Prompts are `"Query: "` / `"Document: "`; the tokenizer appends
  `<|end_of_text|>`, and that token's final hidden state is the embedding.
- Adapters are rank 32, alpha 32 on q/k/v/o and the three MLP projections;
  `W + (alpha / r)·B·A` is folded into the base weights at load, so the trunk
  runs as plain matmuls. Without the merge the output is off by cosine 0.94.
- Built on `rlx-llama32`'s prefill flow with each layer's attention reading a
  key-padding bias instead of a causal mask.

## Parity

`tests/hf_parity_live.rs` against sentence-transformers
(`scripts/embedding_hf_reference.py --task …`), five inputs including a long
one: max |Δ| ≤ 9.6e-7 for the `retrieval` and `text-matching` adapters on CPU,
and ≤ 8.1e-7 on Metal, MLX and wgpu.
