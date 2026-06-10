# rlx-models-core

Shared config, weight loading, compile profiles, and packed GGUF prefill helpers for RLX model crates (published on crates.io as **`rlx-models-core`**; import as `rlx_core`).

**Workspace 0.2.5** (crates.io `rlx-models-core`; depends on upstream `rlx*` 0.2.5). Packed GGUF support (since 0.2.1):

| API | Role |
|-----|------|
| [`packed_gguf_compile_guard`](src/flow_bridge.rs) | Metal `RLX_DISABLE_MPSGRAPH`, MLX `RLX_MLX_MODE=lazy` during compile |
| [`compile_options_for_packed_gguf_prefill_with_profile`](src/flow_bridge.rs) | Fusion off on wgpu/CUDA/ROCm for `FusedResidualRmsNorm` gaps |
| [`packed_gguf_execution_device`](src/flow_bridge.rs) | Native CPU/Metal/MLX packed; wgpu/CUDA/ROCm → CPU prefill |
| [`run_packed_prefill`](src/autoregressive.rs) | Active-extent packed prefill execute (`actual_seq` inside bucket) |
| [`EmbeddedSafetensors`](src/embedded_safetensors.rs) | Parse HF safetensors from `include_bytes!` / memory; `tensor_f32(name)` |
| [`tensor_view_to_f32`](src/safetensors_checkpoint.rs) | Decode F32/F16/BF16 safetensor views to `Vec<f32>` |

Used by `rlx-llama32`, `rlx-qwen3`, `rlx-gemma`, `rlx-minicpm5`, and `rlx-vad` (embedded Silero weights).

### Embedded safetensors

For small models shipped inside the binary:

```rust
use rlx_core::embedded_safetensors::EmbeddedSafetensors;

const WEIGHTS: &[u8] = include_bytes!("../weights/model.safetensors");

let st = EmbeddedSafetensors::parse(WEIGHTS)?;
let w = st.tensor_f32("layer.weight")?;
```

Disk-backed sharded checkpoints still use [`SafetensorsCheckpoint`](src/safetensors_checkpoint.rs) (mmap + index.json).

## See also

- [README.md](../../README.md)
- [AGENTS.md](../../AGENTS.md)
