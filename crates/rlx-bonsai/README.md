# rlx-bonsai

Runner for the **Bonsai** models. "Bonsai" spans two unrelated lineages that share only the name, and this crate **dispatches on the GGUF `general.architecture` tag**:

| Lineage | GGUF arch | Engine |
|---|---|---|
| **Bonsai small-reasoning family** (1.7B / 2B / 4B / 8B) — a standard Llama-shaped decoder tuned for small-context reasoning | `llama` | [`rlx_llama32::Llama32Runner`](../rlx-llama32) (via [`BonsaiRunner`](src/lib.rs)) |
| **[`prism-ml/Bonsai-27B`](https://huggingface.co/prism-ml/Bonsai-27B-gguf)** — Qwen3.6-27B hybrid with custom 1-bit `Q1_0` weights (~1.125 bpw) | `qwen35` | [`rlx_qwen35::Qwen35Runner`](../rlx-qwen35) |
| **[`prism-ml/Ternary-Bonsai-27B`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)** — same hybrid arch with ternary `Q2_0` weights (~2.125 bpw) | `qwen35` | [`rlx_qwen35::Qwen35Runner`](../rlx-qwen35) |

## CLI

`cli_run` (registered as `rlx-run bonsai`, needs `--features bonsai`) sniffs the `--weights` GGUF header — cheaply, without slurping the multi-GB data segment — and routes to the matching runner. Every flag of the target runner works.

```console
# Bonsai 1.7B–8B (llama arch) → llama32 runner
$ rlx-run bonsai --weights bonsai-2b.gguf --prompt "…" --packed

# prism-ml/Bonsai-27B (qwen35 arch, 1-bit Q1_0) → qwen35 runner
$ rlx-run bonsai --weights Bonsai-27B-Q1_0.gguf --prompt "…" --packed

# prism-ml/Ternary-Bonsai-27B (qwen35 arch, ternary Q2_0) → qwen35 runner
$ rlx-run bonsai --weights Ternary-Bonsai-27B-Q2_0.gguf --prompt "…" --packed
```

`rlx-run inspect Bonsai-27B-Q1_0.gguf` reports `arch: qwen35`, `Q1_0: 498` tensors, and `on-disk packed ≈ 3.5 GB` vs `F32-dequant ≈ 100 GB` — use `--packed` so the 27B stays in its 1-bit form. Ternary Q2_0 is ~7.2 GB on disk.

## Library API

```rust
use rlx_bonsai::{detect_arch, BonsaiArch, BonsaiRunner};

// Classify a checkpoint by its GGUF arch (header-only read).
match detect_arch(std::path::Path::new("Bonsai-27B-Q1_0.gguf"))? {
    BonsaiArch::LlamaSmall   => { /* rlx_llama32::Llama32Runner / BonsaiRunner */ }
    BonsaiArch::Qwen35Hybrid => { /* rlx_qwen35::Qwen35Runner */ }
}

// The small (llama) family — BonsaiRunner rejects a qwen35 file with a
// pointer to the qwen35 path.
let mut runner = BonsaiRunner::builder()
    .weights("bonsai-2b.gguf")
    .max_seq(4096)
    .packed_weights(true)
    .build()?; // Err on a non-llama GGUF (incl. Bonsai-27B)
let out = runner.generate_packed(&prompt_ids, 128, |tok| { let _ = tok; })?;
# anyhow::Ok(())
```

For Bonsai-27B / Ternary-Bonsai-27B, drive [`rlx_qwen35::Qwen35Runner`](../rlx-qwen35) directly (or the CLI, which auto-dispatches).

## The 1-bit `Q1_0` and ternary `Q2_0` formats

Bonsai-27B's big projections use the PrismML `Q1_0_g128` format (ggml type **41**): one f16 group scale + 128 sign bits selecting `±d`, 18 bytes / 128 weights. Ternary-Bonsai-27B uses `Q2_0_g128` (ggml type **42**): one f16 scale + 128×2-bit codes mapped as `(q−1)·d`, 34 bytes / 128 weights. Support lives in the shared stack, not this crate:

- [`rlx_gguf::q1_dequant`](../../../rlx/crates/io/rlx-gguf) / [`rlx_gguf::q2_dequant`](../../../rlx/crates/io/rlx-gguf) — block dequant + `bytes_for` wiring.
- `QuantScheme::GgufQ1_0` / `GgufQ2_0` (`rlx-ir`) + the CPU `DequantMatMul` kernel (`rlx-cpu`) keep the weights packed.
- GPU backends with on-device dequant: **Metal** (fused GEMV/GEMM for both), plus host-dequant fallbacks on other devices.

## Features

`tokenizer` and the backend flags (`metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, `all-backends`) forward to both [rlx-llama32](../rlx-llama32) and [rlx-qwen35](../rlx-qwen35).
