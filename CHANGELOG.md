# Changelog

## 0.2.14 — MXFP4 quantization, backend fixes & release hardening (2026-08-12)

### MXFP4 quantization (produce side)

- **`rlx_models_core::mxfp4_pack` — rlx's first f32 → MXFP4 encoder.** Every
  other MXFP4 path in the tree was consume-side, written for checkpoints that
  ship already quantized (mlx-community, Kimi). This packs E2M1 nibbles plus a
  per-group E8M0 scale, so an ordinary bf16/f32 HF checkpoint can drive the same
  packed kernels. Group exponent is the smallest `e` with `6·2^e >= amax`, which
  makes saturation impossible (OCP's `floor(log2(amax)) - 2` clamps the top
  quarter of its range). Gate is `tests/mxfp4_pack_ops.rs`, which feeds the
  packed bytes to the real ops rather than only to the encoder's own
  `dequantize` — a shared misreading of the layout cannot pass it.
- **`rlx-ling --mxfp4`** quantizes the whole model at load time: arena
  29.5 → ~4.0 GiB, steady RSS 21.6 → 8.2 GB, and Ling-3.0-tiny now **fits a
  16 GB CUDA card**, which f32 could not (`device allocation failed for
  7909017552 f32 (29.463 GiB)`). `QuantPlan` splits the LM head out because its
  4-bit error lands undiluted on the logits (3.1e-2 vs 1.9e-3 for the body);
  `--f32-head` trades 0.85 GiB for that. The token embedding stays f32 — it is
  gathered, not multiplied, and rlx has no MXFP4 gather.
- `DeepseekMoeDims::mxfp4_group` runs the routed experts as
  `Op::DequantGroupedMatMulMlx`, shared by rlx-ling / rlx-deepseek / rlx-kimi-k3
  / rlx-glm4moe.

### Backend fixes

- **The `group_limited_gate` host delegate copied the ENTIRE arena
  device→host→device** to compute a top-k over a few thousand floats. On
  Ling-3.0-tiny that was ~276 GB of PCIe traffic and **97% of CUDA prefill**
  (61.8 s of 63.5 s). It now stages only the ~70 KB it touches. The cost scaled
  with *arena size, not problem size*, so it was invisible on small models and
  worst on the ones big enough to need a GPU; every MoE crate on that op was
  paying it.
- **CUDA MXFP4 grouped matmul, 22×** (`gate_up` m=64: 10.33 → 0.46 ms, 110 GB/s):
  new split-K kernel. The old one issued one 32-bit load per *nibble* and gave
  one thread per output, so a warp's lanes read weight rows `k/2` bytes apart —
  fully uncoalesced. Also slightly *more* accurate (tree reduction).
- **CUDA dense MXFP4 GEMM, 1.4×**: it staged X through shared memory where each
  thread wrote and read back its own slot — a no-op round-trip costing 8 KB of
  occupancy-limiting shared memory and a `__syncthreads()` per K-chunk.
- Together: **CUDA Ling prefill 63.4 s → 0.266 s (238×), 1.0 → 240.9 tok/s.**
- **Metal MXFP4, 1.25×** (Ling prefill 45.4 → 56.7 tok/s): the same no-op
  threadgroup staging in both `dequant_matmul_mlx_gemm` and
  `grouped_dequant_matmul_mlx_gemm` (45.4 → 50.9), plus staging the activation as
  `half` with an f32 accumulator (50.9 → 56.7).
- **wgpu arena overrun**: a non-matmul `set_param_typed(BF16)` param was widened
  to f32 and written `ne*4` bytes into an `ne*2` slot (`plan_f32_uniform` keeps
  non-F32 *params* native), corrupting the following param and disagreeing with
  host steps that read the slot as bf16.
- Deprecated `rlx_cpu::llada2_gate::execute_gate_in_f32_arena` (removal in 0.3).
  Its whole-arena offset signature is what made the wasteful staging above the
  natural thing to write; `execute_gate_f32` takes plain slices and is the entry
  point now. It has no remaining callers, but it shipped in the published 0.2.13
  API, so it stays until a major bump.

### Tooling

- `rlx-models-core/examples/mxfp4_grouped_bench` times the grouped and dense
  MXFP4 ops standalone at real MoE shapes — seconds per kernel iteration instead
  of a whole-model prefill. Use it before touching any MXFP4 kernel.

### New model crates

- **`rlx-motif` — Motif-3** ([Motif-Technologies/Motif-3](https://huggingface.co/Motif-Technologies/Motif-3),
  `model_type = "Motif"`): 53 layers, ~314 B parameters, 262 144 context. Three
  pieces with no prior analogue in the workspace:
  - **GDLA** (grouped differential latent attention) — MLA-style low-rank Q/KV
    with one shared RoPE head, 80 heads in bundles of 5 where the last head of
    each bundle is *subtracted* with an input-dependent λ, plus an element-wise
    sigmoid output gate. 3 layers in 4 are 128-key sliding-window on their own
    RoPE base; the rest are global with YaRN and `mscale²` on the softmax scale.
  - **MHC** (manifold-constrained hyper-connections) — four parallel residual
    streams mixed per sublayer by a doubly stochastic 4×4 matrix from 20 inline
    Sinkhorn iterations.
  - **PolyNorm MoE** — a trainable polynomial activation with *per-expert*
    coefficients across 384 experts. Folding `σ(weight)`/bias-clamp host-side
    turns those into a table the graph gathers by routed expert id, so each
    top-k slot stays one `GroupedMatMul`; the reference has to fall back to an
    eager Python loop over experts for exactly this reason.

  Prefill graph, no real-weight run (629 GB / 155 shards). 30 tests — host
  references for each block plus full-graph causality — green on **all 7
  backends + CoreML** across mac / RTX 3080 Ti / MI100. Linux wgpu needs
  `RLX_ARENA_NO_REUSE=1` for the pre-existing `rlx-wgpu` slot-reuse corruption.

### Changed

- `rlx-deepseek`'s MoE emitter now builds its expert GEMMs with
  `HirGraphExt::grouped_matmul`, which derives the output shape from the
  operands instead of taking a hand-written one. That is what rejects an expert
  bank still in the checkpoint's `[E, N, K]` order — previously a silent
  partial write. Needs upstream RLX with `rlx_ir::shape::grouped_matmul_dims`.

### Release hardening & repo hygiene

Workspace `[workspace.package].version` = **0.2.14**, pinned to upstream
**`rlx*`** **0.2.14** on crates.io (`rlx-runtime`, `rlx-ir`, `rlx-flow`, …).
Requires RLX **0.2.14** published from
[MIT-RLX/rlx](https://github.com/MIT-RLX/rlx) first. Minimum supported Rust
version is **1.89**, matching upstream `rlx*` 0.2.14.

#### Notable changes

- **Release hygiene: the whole workspace is `fmt`- and `clippy`-clean.**
  `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D
  warnings` now pass across every crate. Besides formatting, this fixed ~50
  `clippy` findings (`manual_is_multiple_of`, `manual_checked_ops`,
  `unnecessary_cast`, `field_reassign_with_default`, `ptr_arg`, `needless_return`,
  `redundant_clone`, `repeat().take()` → `repeat_n`, …) and several examples/tests
  that had drifted from current APIs: six Gemma parity tests were missing newer
  `GemmaConfig` fields; the `gemma4_e2b` backend-parity test now uploads packed
  weights through the `PackedSrc` enum (`Owned`/`Borrow`/`F32`) like production;
  and the `backend_sweep` (`runner`) / `cmp_ort` (`onnx`) examples gained
  `required-features` so `--all-targets` skips them under default features instead
  of failing to compile.
- **Repo layout:** `rlx-tiny` / `rlx-tinystories` now default their trained
  `.rlxts` checkpoints under `weights/<model>/` (`weights/tinystories/…`,
  `weights/tiny/…`) rather than the repo root, and `checkpoint::save` creates the
  parent directory. Generated `memory_probe` / retention benchmark output dirs are
  consolidated under a git-ignored `bench_out/`.

## 0.2.8 — model coverage expansion (2026-06-21)

Workspace `[workspace.package].version` = **0.2.8**, pinned to upstream
**`rlx*`** **0.2.8** on crates.io (`rlx-runtime`, `rlx-ir`, `rlx-flow`, …).
Requires RLX **0.2.8** published from
[MIT-RLX/rlx](https://github.com/MIT-RLX/rlx) first.

### New model crates

Audio codecs: `rlx-snac`, `rlx-encodec`, `rlx-speechtokenizer`,
`rlx-wavtokenizer`, `rlx-xcodec`, `rlx-facodec`, `rlx-nanocodec`,
`rlx-mimi`, `rlx-dac`, `rlx-tsac`.

ASR / audio: `rlx-wav2vec2-asr`, `rlx-nemotron-asr`, `rlx-qwen3-asr`,
`rlx-funasr`, `rlx-diarize`, `rlx-aec`.

TTS / speech: `rlx-orpheus`, `rlx-kyutai-tts`, `rlx-pocket-tts`,
`rlx-inflect-nano`, `rlx-tiny-tts`, `rlx-vibevoice`, `rlx-moshi`.

Vision / VLM: `rlx-bioclip2`, `rlx-florence2`, `rlx-grounding-dino`.

LM: `rlx-eagle3`.

### Notable changes

- **Qwen3.6-27B-MTP-GGUF** (`qwen35` arch, `unsloth/Qwen3.6-27B-MTP-GGUF`) text
  generation is now coherent and matches llama.cpp. Fixed two GatedDeltaNet bugs
  in `rlx-qwen35`: (1) the decay gate applied a spurious `-exp()` to `ssm_a`,
  which the GGUF already stores as `-exp(A_log)` — collapsing the recurrent
  state; (2) the GQA q/k head expansion (16→48) used *interleave* instead of
  *tile*, flipping the sign of every middle head's output. Also fixed the Metal
  Q3_K dequant (`dequant_gguf.msl` was dropping 8 of 16 sub-block scales — fixes
  Q3_K for all models) and the qwen3vl vision `mmproj` (CLIP merger) loader.
- Removed the `rlx-tensor-host` crate (the host-kernel shim that existed only
  to dodge a crates.io name clash with the framework's `rlx-tensor`). Its host
  kernels now live in `rlx_core::host_kernels` (math unchanged). `rlx-grounding-dino`
  additionally moved its compute (Swin / text encoder / enhancer / decoder) onto
  the `rlx` graph path, with `nn.rs` rebacked on `rlx_cpu::blas`.
- `scripts/publish.sh` publish tiers regenerated from the workspace
  dependency graph to cover all publishable crates.

## 0.2.6 — RLX runtime alignment (2026-06-13)

Workspace and model runners now pin upstream **`rlx*`** **0.2.6** on crates.io
(`rlx-runtime`, `rlx-ir`, `rlx-flow`, …). Requires RLX **0.2.6** published from
[MIT-RLX/rlx](https://github.com/MIT-RLX/rlx) first.

### Model runners (dependency-only release)

Same Rust sources as **0.2.5**; `Cargo.toml` pins updated from `=0.2.5` to
`=0.2.6`:

- `rlx-neutts` 0.2.6
- `rlx-gemma` 0.2.6
- `rlx-minicpm5` 0.2.6
- `rlx-minimax` 0.2.6
- `rlx-nemotron` 0.2.6
- `rlx-models` 0.2.6 (facade; publish last)

Publish tiers 0–6 before the facade (`scripts/publish.sh --list`). After
`rlx-kittentts` **0.2.8** and the tier-5 runners above are on crates.io, Skill
can drop `[patch.crates-io]` path deps and use registry versions only.

### Also at 0.2.6+ in this workspace

- Full workspace `[workspace.package].version` = **0.2.6**
- `kitten_tts_mini_rlx` **0.2.7**, `rlx-kittentts` **0.2.8** (native RLX bundle path)
- `rlx-qwen3-tts`, `rlx-fft` at **0.2.7** where noted in `Cargo.toml`
