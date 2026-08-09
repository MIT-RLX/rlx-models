# rlx-carbon

[Carbon](https://huggingface.co/HuggingFaceBio/Carbon-500M) — HuggingFaceBio's
decoder-only, autoregressive **DNA** language models (500M / 3B / 8B) in RLX.

Carbon is a stock `LlamaForCausalLM` (GQA + RoPE θ=500000 + SwiGLU + RMSNorm,
tied embeddings), so the transformer backbone reuses
[`rlx-llama32`](../rlx-llama32) verbatim — Carbon's `config.json` has
`model_type = "llama"`, and `rlx-run check HuggingFaceBio/Carbon-500M` reports
`Supported { runner: "llama32" }`.

What makes Carbon a *DNA* model is its **tokenizer**, which this crate ports
natively (`tokenizer.py` → Rust):

- **Text** → Qwen3 byte-level BPE (the bundled `tokenizer.json`, ids
  `0..dna_start_id`).
- **DNA** wrapped in `<dna>…</dna>` → non-overlapping **6-mers** (≈ 6 bp each),
  each mapped to a fixed id table appended above the BPE range:

  | id | token |
  |---|---|
  | `dna_start_id + 0` | `<dna>` |
  | `dna_start_id + 1` | `</dna>` |
  | `dna_start_id + 2` | `<oov>` |
  | `dna_start_id + 3 …` | `ATCG`⁶ 6-mers (base-4 over `A,T,C,G`) |
  | … | `<unused_i>` padding (128-alignment) |

  For Carbon-500M: `k=6`, `dna_start_id=151669`, `dna_vocab_size=4107`
  (3 special + 4096 6-mers + 8 padding) → vocab **155776**.

Non-`ATCG` characters and short trailing chunks follow the reference rules: a
6-mer chunk containing a non-base → `<oov>`; a trailing partial chunk is
right-padded with `A` to width `k`, then encoded (or `<oov>`).

## CLI

```sh
# Download the model (≈1 GB bf16 safetensors).
huggingface-cli download HuggingFaceBio/Carbon-500M --local-dir weights/Carbon-500M

# Continue a raw nucleotide sequence (auto-wrapped as a <dna>…</dna> region).
cargo run -p rlx-carbon --features tokenizer --release -- \
  --model weights/Carbon-500M \
  --prompt ATGGCGACCTTTAGCGATCTGGGCAAAGAACTGCGTACCGATCTGGCAGAT \
  --device cpu --max-tokens 64
```

Flags: `--model <dir>` (or `--weights`), `--prompt <SEQ|TEXT>`, `--prompt-ids
<a,b,…>`, `--dna` / `--no-dna` (force / disable DNA wrapping), `--device`,
`--max-tokens`, `--max-seq`, `--temperature`, `--top-p`, `--packed`, `--raw`.

A bare `ACGTN` prompt is auto-treated as a DNA region; pass explicit
`<dna>…</dna>` tags to mix DNA with text metadata (e.g. species tags).

## Library

```rust
use rlx_carbon::CarbonRunner;
use rlx_runtime::Device;

let mut carbon = CarbonRunner::from_pretrained("weights/Carbon-500M", Device::Cpu)?;
let out = carbon.complete("ATCGATCGATCGATCG", 64, Some(true))?; // Some(true) = treat as DNA
println!("{}", out.text);
```

Lower-level pieces: [`HybridDnaTokenizer`] (encode/decode), [`DnaConfig`] +
`split_by_dna_tags` / `parse_dna_region` (pure DNA id math, unit-tested against
the reference table), and `CarbonRunner::generate_ids*` for the token-id path.

## Backends

Carbon is a plain Llama graph, so it runs on every backend `rlx-llama32`
supports. Backend features pass straight through: `metal`, `mlx`, `cuda`,
`rocm`, `gpu` (wgpu), `vulkan`, `coreml`, and the `all-backends` /
`apple-silicon` bundles. The `tokenizer` feature (off by default) pulls in the
`tokenizers` crate for the base BPE; without it the runner/CLI are token-ids
only.

Greedy DNA continuation is **byte-identical across backends** — verified on
Carbon-500M (safetensors):

| CPU | Metal | MLX | wgpu (`gpu`) | CoreML (ANE) | CUDA / ROCm / Vulkan |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ✅ | ✅ | ✅ | ✅ | ✅ | inherited from `rlx-llama32` (no local hw) |

Run the parameterized example on any device:

```sh
# CPU
cargo run -p rlx-carbon --example dna_generate --features tokenizer --release -- \
  --model weights/Carbon-500M --device cpu --prompt ATGGCGACCTTTAGCGATCTG --max-tokens 48

# Apple GPU / ANE (Metal | mlx | gpu | coreml)
cargo run -p rlx-carbon --example dna_generate --features tokenizer,apple-silicon --release -- \
  --model weights/Carbon-500M --device metal --prompt ATGGCGACCTTTAGCGATCTG

# NVIDIA / AMD / portable Vulkan
cargo run -p rlx-carbon --example dna_generate --features tokenizer,cuda --release -- --device cuda --model DIR
```

Or via `just`: `just features=apple-silicon carbon --model DIR --device metal --prompt ATGG…`.

Carbon-500M is intended primarily as a **draft model** for speculative decoding
against the larger Carbon variants.

## Notes

The training-time Fine-grained Nucleotide Supervision `token_mask` output of the
reference tokenizer is not ported (inference does not need it); the id stream is
identical.
