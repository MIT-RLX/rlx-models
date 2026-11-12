# rlx-inkling

[Inkling](https://huggingface.co/thinkingmachines/Inkling) ([thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling)) — natively multimodal decoder-only MoE (text / image / audio → text).

| Property | Value |
|----------|--------|
| Arch tag | `inkling_mm_model` / `InklingForConditionalGeneration` |
| Inkling (released) | ~975B total / ~41B active (66 layers, 256 routed + 2 shared, top-6) |
| Inkling-Small | ~276B / ~12B active — [thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small), BF16, 42 layers (35 sliding + 7 global), hidden 4096, 32 heads / 8 KV, 2 dense + 40 MoE layers, expert width 2048, 8 MTP heads (`InklingVariant::Small`) |
| Context | up to 1M tokens (full) |
| Attention | hybrid sliding-window + global; **relative** logits (not RoPE); short causal convs |
| MoE | sigmoid router, shared-expert sink (joint normalize with selected experts) |

## Status

Scaffold + synthetic text forward + **header-only** shape probes. Inkling-Small's
published `config.json` and full 1048-tensor manifest (names, dtypes, shapes,
including the MTP heads and both towers) are committed under
`tests/fixtures/small/` and checked in both directions by
`cargo test -p rlx-inkling --test small_manifest`; regenerate with
`python3 -I crates/rlx-inkling/scripts/fetch_small_manifest.py DIR` (safetensors
headers over HTTP Range, no weight bytes).

- HF `config.json` parse (text / vision / audio / MTP)
- Checkpoint weight-name map (HF `wq_du` / `w13_dn` layout → transformers-style names)
- Chat helpers (`reasoning_effort`, role / content special tokens)
- Tiny eager CPU text forward (dense + MoE layers) for shape / unit checks
- Safetensors **header** probing (KB of I/O — never downloads full shards)

**Compiled text graph (CPU):** `rlx_inkling::graph` builds the text-trunk prefill
as an `rlx-flow` graph — relative-position bias via `GatherElements`, additive
sliding/causal masks, sconv as pad + shifted multiplies, sigmoid router with
`TopK` + `GroupedMatMul` experts, shared-expert sink, log scaling on global
layers, last-row unembed. `forward_logits_compiled(cfg, weights, ids, device)`
runs it; `tests/graph_parity.rs` compares it with the eager forward on seeded
noise weights (4 layers, window 4, relative extent 6, 12 tokens, log scaling and
mup on, padded vocab) and fails under each of four deliberate breakages.
`weights::text_weights_from_hf` turns HF-named tensors into the eager schema,
including the interleaved gate/up de-interleave (`gate = even rows`,
`up = odd rows`), and `hf_text_plan` is checked against Small's real manifest.

**Corrected against transformers:** logits are `lm_head(hidden / mup)` cut to
`unpadded_vocab_size` (the eager path used to multiply, invisible while the
fixtures use `mup = 1`), and global layers scale queries and relative bias by
`1 + alpha·ln(max(1, (pos+1)/n_floor))` (1.0 below 128k tokens).

**Not yet:**
- Generation on real weights. Small is ~550 GB in BF16; nothing here has run on
  it. The graph is checked against the eager reference, which is checked against
  transformers only on the tiny fixture (where `mup = 1` and no log scaling).
- No KV cache or decode graph (`InklingRunner` still recomputes), no
  `auto_runner` registration, no streaming/paged experts, no quantised weights.
- The graph is verified on CPU only; `GatherElements` may not lower elsewhere.
- Vision HMLP, audio dMel encode, MTP draft heads, GGUF.

## Verification without the 1.9 TB download

| Tier | What it proves | Cost |
|------|----------------|------|
| `just test-inkling` | Config parse, renames, eager math, fixture headers | CPU seconds, no HF |
| `just test-inkling-parity` | Eager text logits vs transformers dump (tiny) | fixture in-repo; regenerate needs `transformers>=5.14` |
| `--probe-remote` (`hf-probe`) | Real Hub safetensors **shapes** vs config | config+index + a few header Range GETs |
| `--probe-gguf-remote` (`hf-probe`) | Unsloth GGUF names/dtypes/shapes for the RLX loader | meta shard (~13 MB) + weight-shard **prefix** (~tens of KB) |
| Local `--model-dir DIR --probe` | Same, against shards already on disk (headers only) | disk seeks |
| Full Unsloth GGUF on disk | Actual generation | ≥~270 GB IQ1_S — later |

Do **not** put full BF16 in CI. Even one shard is ~tens of GB; headers are enough to catch layout / dim bugs.

```bash
# Always-on gate (includes hf_tiny_parity vs checked-in transformers dump)
just test-inkling
just test-inkling-parity

# Regenerate the tiny HF dump (optional; needs transformers>=5.14 + torch)
#   python crates/rlx-inkling/scripts/dump_hf_tiny_parity.py

# Real HF shapes (network; no weight payload)
cargo run -p rlx-inkling --features hf-probe --release -- --probe-remote

# Unsloth GGUF layout for the RLX loader (Range headers only)
just inkling-probe-gguf
#   just inkling-probe-gguf -- --quant UD-IQ1_S --write-json /tmp/inkling-gguf-sniff.json

# If you already have a partial snapshot with some .safetensors files:
just inkling -- --model-dir DIR --probe
```

Inference stays on **RLX**. [unsloth/inkling-GGUF](https://huggingface.co/unsloth/inkling-GGUF) is the weight format (~270 GB 1-bit) — not a llama.cpp runtime dependency.

## CLI

```bash
just inkling -- --synth
just test-inkling
just inkling -- --model-dir /path/to/thinkingmachines/Inkling --probe
```

```bash
cargo run -p rlx-inkling --features hf-probe --release -- --probe-remote
cargo test -p rlx-inkling
```

## Public API

```rust
use rlx_inkling::{InklingConfig, chat, eager, synth};

let cfg = InklingConfig::from_json_path("config.json")?;
let prompt = chat::format_user_turn("Hello", chat::ReasoningEffort::High);
let tiny = synth::tiny_cfg();
let w = synth::synthetic_text_weights(&tiny);
let logits = eager::forward_logits(&tiny, &w, &[1, 2, 3])?;
# anyhow::Ok(())
```

## How it fits

Unique pieces vs Llama / Qwen MoE trunks (relative attention, sconv residual branches, shared-expert sink) need first-class ops before a compiled runner can match transformers / SGLang. This crate holds the config + name map + reference eager path while that lands.
