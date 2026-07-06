# rlx-cohere

Runner scaffold for **[Cohere Command-R](https://docs.cohere.com/docs/command-r)** on RLX. A thin wrapper over [`rlx-llama32`](../rlx-llama32)'s `Llama32Runner` that validates the GGUF `general.architecture` tag (`command-r` / `cohere2`) and delegates decode.

> **Status: stub (PLAN.md M4).** Command-R's parallel-residual attention (Q/K/V and FFN added in one residual pass) and its missing embedding-output LayerNorm are not yet wired in `rlx-llama32`, so runs emit *some* tokens but will **not** match the upstream reference until those land.

## Public API

```rust
use rlx_cohere::CohereRunner;

let mut runner = CohereRunner::builder()
    .weights("c4ai-command-r.gguf")
    .max_seq(512)
    .packed_weights(true)      // low-memory packed-GGUF path
    .build()?;                 // errors unless arch ∈ {command-r, cohere2}

let prompt_ids: Vec<u32> = /* tokenize */ vec![];
let out = runner.generate_packed(&prompt_ids, 64, |tok| print!("{tok} "))?;
# anyhow::Ok(())
```

`CohereRunner` also exposes `config() -> &LlamaBaseConfig` and `inner()` / `inner_mut()` to reach the full `Llama32Runner` API. `cli_run(&args)` re-checks the arch tag and forwards to the shared `rlx-llama32` CLI (no dedicated binary is built yet). The crate re-exports `Llama32ConfigSource`, `Llama32Runner`, and `Llama32RunnerBuilder`.

## How it fits

Inference and config parsing live in [`rlx-llama32`](../rlx-llama32) and [`rlx-llama-base`](../rlx-llama-base); this crate is just the arch gate. Sibling GGUF-arch runners built the same way: [`rlx-phi`](../rlx-phi) (working), [`rlx-mistral`](../rlx-mistral), [`rlx-granite`](../rlx-granite), [`rlx-glm`](../rlx-glm), [`rlx-gpt-oss`](../rlx-gpt-oss).
