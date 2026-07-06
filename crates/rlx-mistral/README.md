# rlx-mistral

Runner scaffold for **[Mistral](https://mistral.ai/) 3+ / Ministral** on RLX. A thin wrapper over [`rlx-llama32`](../rlx-llama32)'s `Llama32Runner` that validates the GGUF `general.architecture` tag (`mistral3` / `mistral4`) and delegates decode.

> **Status: stub (PLAN.md M4).** The underlying `rlx-llama32` builder does not yet apply Mistral 3's per-layer sliding-window mask or the Mistral-specific RoPE base, so runs emit *some* tokens but will **not** match the upstream reference until those deltas land. Mistral 1/2 ship as plain `llama` — use [`rlx-llama32`](../rlx-llama32) directly for those.

## Public API

```rust
use rlx_mistral::MistralRunner;

let mut runner = MistralRunner::builder()
    .weights("Ministral-3B.gguf")
    .max_seq(512)
    .packed_weights(true)      // low-memory packed-GGUF path
    .build()?;                 // errors unless arch ∈ {mistral3, mistral4}

let prompt_ids: Vec<u32> = /* tokenize */ vec![];
let out = runner.generate_packed(&prompt_ids, 64, |tok| print!("{tok} "))?;
# anyhow::Ok(())
```

`MistralRunner` also exposes `config() -> &LlamaBaseConfig` and `inner()` / `inner_mut()` to reach the full `Llama32Runner` API. `cli_run(&args)` re-checks the arch tag and forwards to the shared `rlx-llama32` CLI (no dedicated binary is built yet). The crate re-exports `Llama32ConfigSource`, `Llama32Runner`, and `Llama32RunnerBuilder`.

## How it fits

Inference and config parsing live in [`rlx-llama32`](../rlx-llama32) and [`rlx-llama-base`](../rlx-llama-base); this crate is just the arch gate. Sibling GGUF-arch runners built the same way: [`rlx-phi`](../rlx-phi) (working), [`rlx-cohere`](../rlx-cohere), [`rlx-granite`](../rlx-granite), [`rlx-glm`](../rlx-glm), [`rlx-gpt-oss`](../rlx-gpt-oss).
