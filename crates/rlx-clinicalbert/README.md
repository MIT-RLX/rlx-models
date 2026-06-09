# rlx-clinicalbert

ClinicalBERT encoder runner (Huang / Bio_ClinicalBERT / Bio_Discharge_Summary_BERT) on top of [`rlx-bert`].

Loads HF safetensors with the `bert.` prefix, presets each variant's `(vocab_size, hidden, layers, heads)`, and exposes a high-level [`ClinicalBertRunner`] that compiles the encoder + optional heads to a single backend (CPU, Metal, MLX, CUDA, ROCm, WGPU, Vulkan).

## Quick start

```rust
use rlx_clinicalbert::{ClinicalBertRunner, MlmExecMode, Pooling};
use rlx_runtime::Device;

let mut runner = ClinicalBertRunner::builder()
    .weights("model.safetensors")
    .config_path("config.json")
    .device(Device::Cuda)
    .batch(8).max_seq(32)
    .pooling(Pooling::Cls)
    .with_pooler()
    .mlm_mode(MlmExecMode::Auto)   // pick CPU vs IR-folded per (device, batch)
    .build()?;

let hidden = runner.forward(&ids, &mask, &types, &pos)?;
let pooled = runner.pooler_output(&hidden)?;
let logits = runner.mlm_logits(&hidden)?;
```

## Heads

Two optional heads behind the `pooler` and `mlm` features (or `heads` for both):

- **Pooler** — `tanh(W·h_cls + b)`. Loads `bert.pooler.*`.
- **MLM** — `dense(H→H) + GeLU + LN + tied decoder(H→V) + bias`. Loads `cls.predictions.*`; the decoder weight is tied to the input embedding matrix when no explicit `cls.predictions.decoder.weight` is present (the common HF layout).

### MLM head placement

The MLM head runs either as a CPU post-process or as part of the compiled encoder graph. Pick via `.mlm_mode(MlmExecMode::Auto | Cpu | InGraph)` — `Auto` is the default and chooses per `(device, batch)`:

| Device                                | Batch | Pick     |
|---------------------------------------|-------|----------|
| CPU, Metal, MLX, WGPU, Vulkan, ROCm   | any   | InGraph  |
| CUDA                                  | ≤ 8   | InGraph  |
| CUDA                                  | > 8   | Cpu      |

See [`MlmExecMode`](src/runner.rs) for measured numbers (RTX 4090 + Intel x86, Bio_ClinicalBERT @ seq=32). Override only if you've measured your own `(seq_len, vocab, host BLAS)` shape — crossovers shift with all three.

## Features

- `heads` = `pooler` + `mlm`
- `tokenizer` — WordPiece via `tokenizers`
- `prepare`, `hf-download` — convert / download HF snapshots into `model.safetensors` + `tokenizer.json`
- `blas-mkl` — link `rlx-cpu` against Intel MKL instead of OpenBLAS on Linux / Windows (macOS stays on Accelerate)
- backend: `metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, plus convenience bundles (`all-backends`, `apple-silicon`, `nvidia-gpu`, `amd-gpu`, `portable-gpu`)

## Examples

```sh
# Parity vs HF reference
cargo run -p rlx-clinicalbert --release --features "heads tokenizer" \
  --example parity_check -- --weights <dir>

# IR-fold parity (encoder-graph MLM vs CPU post-process head)
cargo run -p rlx-clinicalbert --release --features "heads tokenizer cuda" \
  --example mlm_fold_parity -- --weights <dir> --device cuda

# Latency / throughput sweep over batch sizes
cargo run -p rlx-clinicalbert --release --features "heads tokenizer cuda blas-mkl" \
  --example bench_batch -- --weights <dir> --device cuda \
  --seq 32 --batches 1,4,8,16,32 --iters 20 --mlm-mode auto
```

## See also

- [README.md](../../README.md)
- [crates/rlx-bert](../rlx-bert)
