# rlx-inkling

[Inkling](https://huggingface.co/thinkingmachines/Inkling) ([thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling)) — natively multimodal decoder-only MoE (text / image / audio → text).

| Property | Value |
|----------|--------|
| Arch tag | `inkling_mm_model` / `InklingForConditionalGeneration` |
| Inkling (released) | ~975B total / ~41B active (66 layers, 256 routed + 2 shared, top-6) |
| Inkling-Small | ~276B / ~12B active — [announced preview](https://thinkingmachines.ai/news/introducing-inkling/#inkling-small); **weights not public yet** (`InklingVariant::Small`, dims TBD) |
| Context | up to 1M tokens (full) |
| Attention | hybrid sliding-window + global; **relative** logits (not RoPE); short causal convs |
| MoE | sigmoid router, shared-expert sink (joint normalize with selected experts) |

## Status

Scaffold + synthetic text forward + **header-only** shape probes:

- HF `config.json` parse (text / vision / audio / MTP)
- Checkpoint weight-name map (HF `wq_du` / `w13_dn` layout → transformers-style names)
- Chat helpers (`reasoning_effort`, role / content special tokens)
- Tiny eager CPU text forward (dense + MoE layers) for shape / unit checks
- Safetensors **header** probing (KB of I/O — never downloads full shards)

**Not yet:** compiled RLX graphs, real-weight generation, vision HMLP, audio dMel encode, MTP draft, GGUF.

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
