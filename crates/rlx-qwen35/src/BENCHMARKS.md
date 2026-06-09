# Qwen3.5 performance notes

Hardware reference: **Apple M4 Pro, 64 GB** (Mac mini).

## Methodology

| Harness | Command |
|---------|---------|
| Real GGUF batch=1 (CPU/Metal/MLX) | `QWEN35_GGUF_PATH=… cargo test -p rlx-models --test qwen35_backend_gguf_bench --features "metal,mlx" --release -- --nocapture` |
| Real GGUF heterogeneous batch=2 | `QWEN35_GGUF_PATH=… cargo test -p rlx-models --test qwen35_batch_gguf_bench --features "metal,mlx" --release -- --nocapture` |
| Real GGUF batch=2 check | `cargo test -p rlx-models --test qwen35_batch_gguf_quick_check --release -- --nocapture` |
| Real GGUF VLM check | `QWEN35_GGUF_PATH=… QWEN35_MMPROJ_PATH=… cargo test -p rlx-models --test qwen35_vlm_gguf_quick_check --features qwen35-vlm --release -- --nocapture` |
| Forward CLI | `cargo run --release -p rlx-models --example qwen35_forward_bench --features "metal,mlx" -- /path/to/model.gguf --device cpu --packed --tokens 16` |
| Synthetic (6-layer toy) | `cargo bench -p rlx-models --bench qwen35_inference` |

Steady-state generate: 16 tokens after 4-token warmup, prompt `[1..=8]`, packed Q4_K_M.

### Heterogeneous batch=2

Two prompts with different lengths (8 vs 7 token ids), batch=2 runner, packed Q4_K_M:

| Metric | Meaning |
|--------|---------|
| `prefill b2` | `predict_logits_batch` steady-state for both rows |
| `decode b2 uniform agg` | 32 total tokens (16×2) / wall time |
| `decode b2 per-row-limits agg` | row0=16 tok, row1=8 tok (24 total) / wall time |
| `eff` | `agg_b2 / (2 × tok/s_b1)` — 1.0 = perfect 2× batch scaling |

Example output line:

```text
qwen35 het-batch bench Cpu: prefill b1=…ms b2=…ms | decode b1=… tok/s/stream | decode b2 uniform agg=… tok/s (…/stream, eff=…x) | …
```

## Tier C optimizations (2026-05-19)

| Item | Change |
|------|--------|
| **C.11** | Fused tiled GGUF dequant+matmul (`rlx-cpu/src/gguf_matmul.rs`) — no full F32 weight cache |
| **C.10** | GDN decode/prefill BLAS path (`sgemv`/`sger`/`sscal`, n≤128) |
| **C.9** | Parallel GDN prefill: time-outer loop, Rayon over heads |
| **C.8** | MLX default `Compiled` + `warm_compile` (override: `RLX_MLX_MODE=lazy`) |
| Metal dequant | Fused CPU path on unified memory (opt-in GPU: `RLX_METAL_DEQUANT_GPU=1`) |

## Baselines (0.8B Q4_K_M, pre-Tier C — replace after re-bench)

| Backend | Prefill steady (3 tok) | Generate tok/s |
|---------|------------------------|----------------|
| CPU | ~310 ms | ~2.35 |
| Metal | ~1225 ms | ~1.08 |
| MLX | ~3521 ms | ~0.99 |

Re-run the env-gated bench test after placing GGUF at `/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf`.

## Why it was slow (before Tier C)

1. Full-matrix GGUF dequant on every matmul (~150 packed params × 24 layers).
2. Scalar GDN inner loops (n=128) instead of BLAS.
3. MLX re-lowering every `run()` when GDN forced lazy mode.
4. Metal GPU full dequant scratch + MPS sync on decode-sized matmuls.
5. Per-`past_seq` decode graph compile + weight upload amortization.

Expected post-Tier C: **CPU 5–15 tok/s**, Metal/MLX closer to CPU on decode (still bounded by graph dispatch until fused Metal GDN lands).
