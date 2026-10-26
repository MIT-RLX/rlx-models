# rlx-embeddinggemma

[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
text embeddings on RLX: a Gemma 3 trunk run **bidirectionally**, mean-pooled,
projected by two bias-free dense layers and L2-normalized. 768-wide output
with Matryoshka prefixes (512 / 256 / 128).

| | |
|---|---|
| Parameters | 308 M |
| Checkpoint | 1.21 GB f32 safetensors (+ 2 × 9.4 MB dense heads) |
| Context | 2048 tokens |

The Google repo is gated; [unsloth/embeddinggemma-300m](https://huggingface.co/unsloth/embeddinggemma-300m)
carries the same files without the gate.

```sh
hf download unsloth/embeddinggemma-300m --local-dir weights/embed/embeddinggemma-300m
cargo run -p rlx-embeddinggemma --release -- --weights weights/embed/embeddinggemma-300m \
    --text "Which planet is known as the Red Planet?" \
    --prompt document --text "Mars, often called the Red Planet, has a thin atmosphere."
```

```rust,ignore
let mut m = rlx_embeddinggemma::EmbeddingGemma::open(dir, Device::Cpu)?;
let q = m.embed_text(m.prompt("query").unwrap_or(""), "which planet is red?")?;
```

## What differs from a causal Gemma 3

- Every layer's attention reads an additive bias instead of a causal mask.
  Full-attention layers see every real token; sliding layers see
  `|q − k| < sliding_window / 2 + 1` — transformers halves the configured
  512-token window on load when `use_bidirectional_attention` is set. Using the
  window as written gives cosine 0.9875 on an 899-token input.
- The head (`2_Dense` 768→3072, `3_Dense` 3072→768, identity activations, no
  bias) is one linear map, composed at load and applied once to the pooled
  vector.
- The checkpoint is a bare `Gemma3TextModel` (`layers.N…`, no `model.` prefix).

## Parity

`tests/hf_parity_live.rs` against sentence-transformers (`scripts/embedding_hf_reference.py`),
five inputs including an 899-token one: max |Δ| ≤ 4.6e-7 on CPU, Metal, MLX
and wgpu. A causal mask in place of the bidirectional one gives cosine 0.68.
