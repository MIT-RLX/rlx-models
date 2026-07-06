# rlx-llada2

**LLaDA2** MoE **block-diffusion** language model for RLX, plus the **TIDE** predictive expert-offload runtime. Unlike an autoregressive LM, generation runs a masked-denoising loop over fixed-size blocks; the MoE experts can be streamed on/off the GPU per step so large mixture models fit in a bounded VRAM budget.

This is a **library crate** — no CLI binary. Loading, compiling, and driving generation happen through the runner API below.

## Public API

```rust
use rlx_llada2::tide::{TideRunner, GenerateConfig};
use rlx_runtime::Device;

let mut runner = TideRunner::builder()
    .weights_path("/path/to/llada2-moe")   // dir with config.json + weights
    .device(Device::Cpu)
    .batch_seq(1, 512)                      // batch, max window
    .build()?;

let gen_cfg = GenerateConfig {
    gen_length: 128,
    block_length: 32,
    steps: 32,
    temperature: 0.0,
    ..Default::default()
};
let (tokens, step_stats) = runner.generate(&[1, 2, 3], &gen_cfg)?;
# anyhow::Ok(())
```

Both [`TideRunner`](src/tide/runner.rs) and the underlying [`LLaDA2Runner`](src/llada2/runner.rs) share the same builder ([`LLaDA2RunnerBuilder`](src/llada2/runner.rs)). Other exports:

- [`LLaDA2MoeConfig`](src/llada2/config.rs) — `from_file` / `from_json_str` HF config.
- [`BlockDenoiseLoop`] / [`run_block_diffusion`](src/tide/generate.rs) — the block-diffusion denoise driver.
- [`build_llada2_forward_graph`](src/llada2/builder.rs) — the single-block forward IR graph.
- TIDE offload — [`enable_predictive_expert_offload`](src/tide/offload.rs), [`preview_predictive_offload`](src/tide/runner.rs), [`TideOffloadStats`], [`gpu_expert_budget_from_device_memory`]. Configure via the builder's `tide_enable_predictive_expert_offload(max_per_layer, reserve_vram_gb, collect_stats, jump_steps)`.

## How it fits

- [rlx-qwen35](../rlx-qwen35) re-exports this crate's TIDE offload types for its own MoE checkpoints.

## Features

| Feature | Enables |
|---|---|
| `hf-download` | `hf-hub` weight download helpers |
| `metal` | Metal backend (`rlx-runtime/metal` + `rlx-metal`) |
| `mlx` | MLX backend (`rlx-runtime/mlx` + `rlx-mlx`) |
| `cuda`, `rocm`, `gpu`, `vulkan`, `all-backends` | forwarded to `rlx-runtime` |

GPU expert-offload is most relevant on `cuda` (VRAM-bounded); on host/unified memory the budget preview falls back to system RAM.
</content>
