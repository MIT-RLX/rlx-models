# rlx-models-core

Shared config, weight loading, compile profiles, and packed GGUF prefill helpers for RLX model crates (published on crates.io as **`rlx-models-core`**; import as `rlx_core`).

**Workspace 0.2.14** (crates.io `rlx-models-core`; depends on upstream `rlx*` 0.2.14). Packed GGUF support (since 0.2.1):

| API | Role |
|-----|------|
| [`packed_gguf_compile_guard`](src/flow_bridge.rs) | Metal `RLX_DISABLE_MPSGRAPH`, MLX `RLX_MLX_MODE=lazy` during compile |
| [`compile_options_for_packed_gguf_prefill_with_profile`](src/flow_bridge.rs) | Fusion off on wgpu/CUDA/ROCm for `FusedResidualRmsNorm` gaps |
| [`packed_gguf_execution_device`](src/flow_bridge.rs) | Native CPU/Metal/MLX/CUDA/wgpu/Vulkan/CoreML packed; `*_HOST=1` forces CPU |
| [`run_packed_prefill`](src/autoregressive.rs) | Active-extent packed prefill execute (`actual_seq` inside bucket) |
| [`EmbeddedSafetensors`](src/embedded_safetensors.rs) | Parse HF safetensors from `include_bytes!` / memory; `tensor_f32(name)` |
| [`tensor_view_to_f32`](src/safetensors_checkpoint.rs) | Decode F32/F16/BF16 safetensor views to `Vec<f32>` |
| [`weights_discover`](src/weights_discover.rs) | Scan LM Studio / Ollama / HF / Lemonade / RLX local caches; resolve short names |

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

### Local weights discovery

Walk on-disk caches used by common LLM apps (no network) and either list hits or resolve a short query to one path. Works on macOS, Linux, and Windows (`%USERPROFILE%`, `%LOCALAPPDATA%`, `%TEMP%`; `RLX_WEIGHTS_PATHS` uses `;` on Windows).

| API | Role |
|-----|------|
| [`scan_weights`](src/weights_discover.rs) | Scan default host roots |
| [`scan_weights_in_roots`](src/weights_discover.rs) | Scan caller-supplied roots only (tests / embedded) |
| [`resolve_weight_query`](src/weights_discover.rs) | Pick one path for a substring query |
| [`resolve_weight_query_in_roots`](src/weights_discover.rs) | Same, restricted roots |
| [`resolve_weights_path_or_query`](src/weights_discover.rs) | Existing path → file resolve; else short-name discovery |
| [`default_source_roots`](src/weights_discover.rs) | Existing default roots only |
| [`looks_like_filesystem_path`](src/weights_discover.rs) | Distinguishes `C:\…` / `./…` from `qwen3-0.6b` |

```rust
use rlx_core::{DiscoverOpts, WeightSourceKind, resolve_weight_query, scan_weights};

let hits = scan_weights(
    &DiscoverOpts::default()
        .with_query("qwen")
        .with_sources(vec![WeightSourceKind::LmStudio, WeightSourceKind::HuggingFace]),
)?;
let path = resolve_weight_query(
    "qwen3-0.6b",
    &DiscoverOpts::default().with_prefer_quant("Q4_K_M"),
)?;
```

**Environment overrides**

| Variable | Purpose |
|----------|---------|
| `LMS_MODELS` / `LM_STUDIO_MODELS` | LM Studio models dir |
| `OLLAMA_MODELS` | Ollama models dir |
| `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `HF_HOME` / `XDG_CACHE_HOME` | Hugging Face hub |
| `MLX_CACHE` | Extra MLX cache root |
| `VLLM_CACHE_ROOT` | Extra vLLM cache root |
| `LEMONADE_CACHE_DIR` | Lemonade cache (`config.json`, `user_models.json`) |
| `RLX_WEIGHTS_DIR` | Extra RLX local root |
| `RLX_WEIGHTS_PATHS` | Extra roots (`;` on Windows, `:` on Unix) |
| `TEMP` / `TMP` | Parent of `rlx-weights` temp dir (Windows / portable) |

**CLI:** `rlx-inspect scan` / `rlx-inspect resolve` / `just weights-scan`.  
**Example:** `cargo run -p rlx-models-core --example weights_discover -- --query qwen --json`.

Also re-exported from `rlx_models::` and `rlx_cli::`.

## MXFP4 encoder

[`mxfp4_pack`](src/mxfp4_pack.rs) is rlx's f32 → MXFP4 **encoder** — E2M1 nibbles
plus a per-group E8M0 scale. Every other MXFP4 path in the tree is consume-side,
written for checkpoints that ship already quantized (mlx-community, Kimi); this
is what lets an ordinary bf16/f32 HF checkpoint drive the same packed kernels,
for ~8x less arena.

```rust
use rlx_models_core::mxfp4_pack::{GROUP_SIZE, quantize_rows};

// A `[out, in]` HF weight — the contraction runs along the last dim.
let q = quantize_rows(&weight, out_features, in_features, GROUP_SIZE);
compiled.set_param_typed("w.codes", &q.codes, DType::U8);
compiled.set_param_typed("w.scales", q.scales_e8m0(), DType::U8);   // dense op
// ...or `q.scales_bf16()` for the grouped MoE op — see below.
```

**The two consuming ops disagree on how the scale operand is typed, and the
mismatch is silent** (both are the right byte count):

| op | scale param | contents |
|---|---|---|
| `Op::DequantMatMul` (dense) | `U8 [n, groups]` | raw E8M0 bytes |
| `Op::DequantGroupedMatMulMlx` (MoE) | `BF16 [E, n, groups]` | the *decoded* float `2^(b-127)` |

Layout for both is `[E,] N, K` row-major with the contraction along the last
dim, so a stacked expert bank needs **no transpose** — the opposite of the f32
`Op::GroupedMatMul` path.

Group exponents use the smallest `e` with `6·2^e >= amax`, which makes
saturation impossible. (OCP's `floor(log2(amax)) - 2` leaves the group max in
`[4, 8)` and clamps the top quarter of that range, costing up to 25% on the
largest weight in each group.)

`tests/mxfp4_pack_ops.rs` is the gate: it feeds packed bytes to the real ops and
compares against an f32 matmul of the dequantized weight, so a layout
misreading shared between the encoder and its own `dequantize` cannot pass.
[`examples/mxfp4_grouped_bench`](examples/mxfp4_grouped_bench.rs) times the
grouped/dense ops standalone at real MoE shapes — use it before touching any
MXFP4 kernel.

See [`rlx-ling`](../rlx-ling/README.md) for a whole model on this path.

## Distributed inference (multi-node)

Run one model split across several machines when no single host has the RAM for
the whole checkpoint. The bridge in [`distributed_bridge`](src/distributed_bridge.rs)
plugs `rlx-models-core` decoders into the model-agnostic [`rlx-distributed`](../../../rlx/crates/core/rlx-distributed)
pipeline: each node builds and serves only its own layer range, and stages
exchange just the hidden state over TCP.

| API | Role |
|-----|------|
| [`StructureLoader`](src/distributed_bridge.rs) | Build a stage from tensor **shapes only** (peak RAM = one tensor), deferring the large packed weights — no full arena load |
| [`ManifestParamSource`](src/distributed_bridge.rs) | Re-stream each stage's weight shard from the checkpoint at compile time, keyed by checkpoint name |
| [`run_decoder_pipeline_local`](src/distributed_bridge.rs) | In-process multi-stage pipeline (parity check against single-node) |

Streaming the structure first and re-fetching weights per stage avoids the 2×
arena-load peak, so a node's resident memory tracks *its layers*, not the model:
DeepSeek-V4-Flash (~111 GB on disk, 43 layers) runs across three heterogeneous
nodes at **25.5 GB total resident** (≤12 GB on any one node).

### Config-driven cluster runner

[`examples/dsv4_cluster.rs`](examples/dsv4_cluster.rs) is a full coordinator +
worker binary. One TOML ([`dsv4_cluster.toml`](examples/dsv4_cluster.toml))
describes the model and nodes; the coordinator probes each node's hardware,
plans a RAM-balanced layer split, launches the remote workers over SSH, drives a
forward pass, and prints a per-node timing/resident monitor. Each node keeps its
own `device` / `precision` / `kv_cache`.

```bash
# addresses, `~/.ssh/config` aliases, and ckpt paths are placeholders — edit dsv4_cluster.toml
cargo run --release -p rlx-models-core --example dsv4_cluster -- \
  --config dsv4_cluster.toml \
  --model-dir /path/to/DeepSeek-V4-Flash-2bit-DQ \
  --ids 0,671,6102,294,8760,344
```

**Device notes** (heterogeneous placement): a small-VRAM discrete GPU can OOM or
destabilize its driver on a large managed/oversubscribed stage — the native CPU
path is more stable there; and a backend that host-falls-back the model's
unsupported ops runs *far* slower than the native CPU executor, so prefer CPU
over a GPU that can't run the ops natively. Set each node's `device` accordingly.

## See also

- [README.md](../../README.md)
- [AGENTS.md](../../AGENTS.md)
- [rlx-distributed](../../../rlx/crates/core/rlx-distributed) — the model-agnostic multi-node pipeline
