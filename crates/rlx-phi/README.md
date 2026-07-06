# rlx-phi

Runner for Microsoft **[Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) / Phi-4** on RLX. A thin wrapper over [`rlx-llama32`](../rlx-llama32)'s `Llama32Runner`: it validates the GGUF `general.architecture` tag (`phi3` / `phi4` — Phi-4 reuses the Phi-3 arch tag upstream, there is no separate `phi4` enum in llama.cpp), then delegates decode with Phi-specific partial RoPE and NeoX pairing.

## Quick start

```bash
just fetch-phi3-mini              # Phi-3-mini-4k-instruct Q4_K_M GGUF (~2.3 GB → /tmp/rlx-weights)

just features=tokenizer phi -- \
  --weights /tmp/rlx-weights/Phi-3-mini-4k-instruct.gguf \
  --prompt "Explain RoPE in one sentence." --max-tokens 64
```

The `rlx-phi` binary requires the `tokenizer` feature (hence `just features=tokenizer phi`). It forwards to the shared `rlx-llama32` CLI; useful flags:

| Flag | Purpose |
|------|---------|
| `--device cpu\|metal\|mlx\|cuda\|rocm\|vulkan` | Backend (default `cpu`) |
| `--prompt TEXT` / `--prompt-ids 1,2,3` | Input (text needs `tokenizer`) |
| `--max-tokens N` / `--max-seq N` | Decode length / context |
| `--temperature F` / `--top-p F` | Sampling (default greedy) |
| `--packed` / `--no-packed` | Low-memory packed-GGUF path |
| `--no-stream` | Buffer instead of streaming |

## Public API

```rust
use rlx_phi::PhiRunner;
use rlx_runtime::Device;

let mut runner = PhiRunner::builder()
    .weights("Phi-3-mini-4k-instruct.gguf")
    .device(Device::Cpu)
    .max_seq(512)
    .build()?;                                    // errors unless arch ∈ {phi3, phi4}

let prompt_ids: Vec<u32> = /* tokenize */ vec![];
let out = runner.generate(&prompt_ids, 64, |tok| print!("{tok} "))?;
let _logits = runner.predict_logits(&prompt_ids)?;   // one forward, full last-row logits
# anyhow::Ok(())
```

`PhiRunner` also exposes `generate_packed` (low-memory path), `packed_weights(bool)` on the builder, `config() -> &LlamaBaseConfig`, and `inner()` / `inner_mut()` for the underlying `Llama32Runner`. The crate re-exports `Llama32Config`, `Llama32ConfigSource`, `Llama32Runner`, `Llama32RunnerBuilder`, and `RopeStyle`.

## How it fits

All inference lives in [`rlx-llama32`](../rlx-llama32); this crate adds only the arch gate and Phi-friendly builder. Config parsing is shared via [`rlx-llama-base`](../rlx-llama-base) (`LlamaBaseConfig::from_gguf_path`), and the CLI plumbing via [`rlx-cli`](../rlx-cli).

Sibling GGUF-arch runners built the same way: [`rlx-mistral`](../rlx-mistral), [`rlx-cohere`](../rlx-cohere), [`rlx-granite`](../rlx-granite), [`rlx-glm`](../rlx-glm), [`rlx-gpt-oss`](../rlx-gpt-oss).

## Tests

```bash
just test-phi3-real            # load Phi-3-mini GGUF + parse/config checks
just test-phi3-real-inference  # real forward pass (RLX_PHI3_RUN_INFERENCE=1)
```

(Real-weights tests live in `crates/rlx-models/tests/real_weights_phi3.rs`.)
