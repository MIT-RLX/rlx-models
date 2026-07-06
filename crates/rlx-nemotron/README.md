# rlx-nemotron

**NVIDIA Nemotron 3 Nano** runner for RLX — text and hybrid Mamba2/attention language models. Nemotron ships as several GGUF arch tags, handled two ways:

- `nemotron` — text-only, Llama-shaped attention stack; runs via the [`rlx_llama32::Llama32Runner`](../rlx-llama32) delegate.
- `nemotron_h` / `nemotron_h_moe` — **hybrid Mamba2 + attention**; the `NemotronHybridRunner` interleaves per-layer `Mamba2StepStage` (state-space) with stateless attention blocks.

The Omni 30B variant (vision + audio) lives in [rlx-nemotron-omni](../rlx-nemotron-omni).

## Public API

```rust
use rlx_nemotron::{NemotronHybridConfig, NemotronHybridRunner};
use rlx_runtime::Device;

// hybrid nemotron_h GGUF
let mut runner = NemotronHybridRunner::builder()
    .weights("nemotron-h.gguf")
    .device(Device::Cpu)
    .build()?;
runner.generate(&prompt_ids, 32, |tok| print!("{tok} "))?;
# anyhow::Ok(())
```

For the plain `nemotron` (attention-only) tag, `NemotronRunner` delegates to `Llama32Runner`.

## How it fits

- [rlx-ssm](../rlx-ssm) — the `Mamba2StepStage` state-space kernels used by the hybrid path.
- [rlx-llama32](../rlx-llama32) / [rlx-llama-base](../rlx-llama-base) — the attention stack + shared config.
