# rlx-qwen35

Alibaba **Qwen3.5 / Qwen3.6 / Qwen3.8** for RLX — hybrid **Gated DeltaNet** ("linear attention") + full attention every `full_attention_interval` layers, optional **MTP** head for speculative decode. Dense (`qwen35` / `qwen36`) and MoE (`qwen35moe`) GGUFs load through the shared GGUF metadata reader.

**Status:** dense and MoE GGUF prefill + bucketed decode run on CPU and on GPU backends (`metal`, `mlx`, `cuda`, …) with `--packed` for K-quants / Q1_0. **Qwen3.6-27B-MTP (Q3_K_S) text generation is coherent and matches llama.cpp** on CPU and Metal (e.g. "The capital of France is" → "Paris."). Some ops may still fall back to host on a given device; see root [README.md](../../README.md) backend matrix. Remaining gaps (MoE offload polish, VLM parity): `PLAN.md` § Qwen3.5.

### Pestle-27B-Ternary

[`Doses-AI/Pestle-27B-Ternary-GGUF`](https://huggingface.co/Doses-AI/Pestle-27B-Ternary-GGUF)
is Qwen3.6-27B (`qwen35` arch, same 64-block topology, same tokenizer, no MTP
head) with every transformer linear replaced by a **Pestle factorization** —
so it needs more than a new quant type. `just fetch-pestle-27b`, then:

```bash
just qwen35 -- --weights weights/pestle-27b-ternary-gguf/pestle-27b-ternary.gguf \
  --packed --device metal --fast --prompt "The capital of France is"
```

Each `[in → out]` projection becomes a pair of ternary matrices around a
rank-`r` waist plus three per-channel f32 scales:

```text
y = scale_post ⊙ ( U ( scale_mid ⊙ ( V ( scale_pre ⊙ x ) ) ) )
```

with tensors `blk.N.pestle.S.{v,u,scale_pre,scale_mid,scale_post}.weight`.
The factorization does **not** shrink the parameter count — `in·r + r·out ≈
in·out` at Pestle's ranks — it buys expressivity, since a product of two
ternary matrices covers far more of the original weight space than one
ternary matrix does. That is what holds a 27B together at 1.79 nominal
bits/weight (2.52 effective, 7.9 GiB on disk).

The file is **mixed** in three ways, all of which the loader handles per
tensor rather than per model:

| Part | Format | Notes |
|---|---|---|
| Blocks 0..=62 projections | `Q2_0` factor pairs (978 tensors) | already supported (Ternary Bonsai); stays packed through `DequantMatMul` |
| Block 63 | dense **BF16** (7 tensors) | "matching-parent final decoder block"; probed per layer via `blk.N.pestle.0.v.weight` |
| `token_embd` / `output` | **`G8_0`** (ggml type 143) | new — 32-element blocks, four bf16 scales (one per group of 8), 2-bit `{−1,0,+1}` codes |
| scales, norms, `ssm_*` | F32 | host-side, like the norms |

Slot numbering (which of the 8 slots is `ssm_beta` vs `ssm_alpha`, etc.) is
fixed by the `mortar.cpp` fork and is not derivable from the tensor names —
three of the pairs (`attn_k`/`attn_v`, `ssm_beta`/`ssm_alpha`,
`ffn_gate`/`ffn_up`) have identical shapes, so a swap type-checks and still
produces fluent text. `weights.rs` pins the mapping against the fork's
`build_layer_*`. Ranks are read off `scale_mid` rather than hardcoded, so a
future Pestle checkpoint with different ranks loads unchanged.

The companion `mmproj-pestle-27b-ternary.gguf` is a **stock** BF16
`clip` / `qwen3vl_merger` projector (334 tensors, BF16 + F32, no Pestle or
G8_0), i.e. the same shape as Qwen3.6/3.8's `mmproj-F16.gguf` — it needs no
new code on the vision side.

**Status: greedy-identical to the reference runtime.** Against `mortar.cpp`
(`llama-completion -st -rea off --temp 0`, thinking off, same prompt), rlx on
Metal reproduces the reference completion character-for-character over the
full overlapping span:

> Metformin primarily works by inhibiting mitochondrial complex I, which
> reduces hepatic gluconeogenesis and decreases glucose production in the
> liver. It also increases insulin sensitivity in peripheral tissues, such as
> muscle and adipose tissue, …

That one run covers what the shapes cannot: the three same-shaped slot pairs,
the factor order, the Q2_0 factor decode, the G8_0 embed/lm_head and the dense
BF16 block. `tests/pestle_projection.rs` additionally builds the same tiny
model twice — once factorized, once as its algebraically exact dense
equivalent — and requires matching logits, pinning the factor order, both
transposes, and the axis each of the three scales broadcasts along.

Measured on an M4 Pro, `--device metal --fast`, **median per-step decode**
(`RLX_QWEN35_DECODE_TRACE=1`, steps ≥ 2):

| | rlx | `mortar.cpp` |
|---|---:|---:|
| decode (steady) | **8.22 tok/s** (was 5.08) | 11.15 tok/s |
| startup | ~190–500 s compile + bucket warm | 0.5 s |
| peak footprint | 37.1 GB | — |

**Do not compute tok/s as total decode ÷ tokens for this model.** The first
decode step pays the decode-bucket compile — on a 200-token run that was 92.4 s
of a 117.1 s decode, 79% of the total, in one step, with the remaining 199 at a
median of 121.9 ms. Every "1.5 tok/s"-class figure in this crate's history came
from that division and is measuring compile amortisation, not decode.

**Decode is matmul-bound, and the lm-head was most of it.** A per-thunk profile
(`RLX_METAL_THUNK_PROFILE=1`) puts `dequant_matmul_gguf` at 73–75% — 979 calls,
978 `Q2_0` factors plus one `G8_0` lm-head — with `attention` at 1.6% and the
three Pestle scale multiplies at 3.2%. The single lm-head dominated: Metal's
fused-GEMV gate covered `Q1_0`/`Q2_0` only, so `output` (`[248320, 5120]`,
`G8_0`) fell to dequant-to-scratch, re-dequantising ~10 GiB every token. Adding
`g8_0_mv_f32_sg` is worth **+57% steady decode** (195.8 → 124.3 ms/token,
non-overlapping distributions) and **−11 GB peak**; with the `Q2_0` 16-bit
inner loop (+7.6%) the pair is **1.62×** end to end, 196.7 → 121.7 ms/token,
output byte-identical.
Methodology and the hypotheses that died on the way:
[src/BENCHMARKS.md](src/BENCHMARKS.md#measure-median-per-step-time-never-total--tokens).

Backend coverage above is at the kernel and graph level. Running the **full
27B checkpoint** is a separate question, and two backends can't hold it —
for reasons that apply to any 27B, not to Pestle:

| Backend | Full 27B checkpoint |
|---|---|
| CPU, Metal | ✅ runs; Metal is greedy-identical to `mortar.cpp` |
| CUDA | ❌ compiles (46 s) and uploads all 979 packed params, then needs a **13.2 GiB** arena — more than a 16 GiB card has once the 8.5 GB of packed weights are resident. Driven by the LM-head scratch below. |
| ROCm | ❌ `rlx-rocm: arena is 16432893472 bytes, past the 4 GiB addressable by this backend's u32 byte offsets`. A pre-existing ROCm limit on *any* large model; the backend catches it explicitly rather than computing zeros. Vulkan had the same bug and was fixed. |
| MLX, wgpu, Vulkan | not attempted at 27B |

Two memory caveats, both pre-dating Pestle:

- **The LM head is dequantized to F32 scratch** rather than fused. `output`
  is 248320 × 5120, so that slab is ~5.1 GiB on top of 8.5 GB of packed
  weights — which is what overflows a 16 GiB CUDA card. wgpu already avoids
  this for `Q4_K`/`Q6_K`/`Q1_0` via a scratch-free windowed GEMV
  (`gemv_supports_scheme`); `G8_0` wants the same treatment.
- **`token_embd` is dequantized eagerly to F32** (another 5.1 GB), as it is
  for every model here — most of why peak RSS is 21 GB for a 7.9 GiB file
  (Metal's peak *footprint*, including device allocations, reaches 47 GB).
  `rlx_gguf::g8_dequant::gather_rows_g8_0` exists for a lazy
  gather-from-packed embed path; wiring it is not done. Note
  `RLX_QWEN35_HOST_EMBED=1` does **not** help here — it moves the gather, not
  the LM-head slab above.

### Qwen3.8-27B

`unsloth/Qwen3.8-27B-GGUF` ships the **same `qwen35` architecture as Qwen3.6-27B** — identical `qwen35.*` metadata, an identical 866-tensor layout (same names *and* dtypes) and a byte-identical tokenizer — so it loads through this crate unchanged (`just fetch-qwen38-27b`). Validated on Q3_K_S: text generation coherent and factually correct on **CPU and Metal (token-identical to each other)**, `--verify-selftest` prefill↔decode PASS (maxdiff 2e-5, per-layer KV divergence ~1e-6), chat `--fast` answers cleanly and stops on `<|im_end|>`, and the `--mmproj` vision path captions correctly.

**Use `Q4_K_M`, not `Q3_K_S`, on Metal.** Despite being 36% larger it decodes
**~1.5× faster** (6.2–6.5 vs 4.0–4.3 tok/s steady-state on an M4 Pro), because
Q3_K's GEMV is dequant-ALU bound at ~48 GB/s vs Q4_K's ~185 GB/s — the Q3_K
kernel is already at its documented floor. Full numbers, the ~97 ms/token
dispatch floor that dominates once you switch, and the `--aot-cache` /
`--fast` / `--spec-decode` findings: [src/BENCHMARKS.md](src/BENCHMARKS.md#qwen38-27b-on-metal--quant-choice-dominates-decode).

Two caveats, both **pre-existing and not Qwen3.8-specific**:

- **Greedy output is not token-identical to llama.cpp.** Both agree on every high-confidence token, but diverge where the top-2 logit gap is small (measured 0.4–1.0 via `llama-completion --logit-bias` bisection) — typically `\n` vs `\n\n`. ggml quantizes activations to int8 in its matmuls; rlx dequantizes weights and computes in F32, so the two are different-but-valid decodes of the same weights. Qwen3.6 shows the same flips.
- **Qwen3.8's chat template adds a `reasoning_effort` system-prompt injection** (`xhigh` default | `medium` | `low`) that Qwen3.6 lacks. `format_chatml_with` does not emit it, so `--chat` with thinking on omits that steering line. The generation prompt itself still matches the template exactly, and `--no-think` is unaffected.
- **`<|endoftext|>` (248044) is not a stop token.** The GGUF declares `eos = 248046` (`<|im_end|>`) and rlx stops on that, but the HF `config.json` declares `eos_token_id = 248044`. Generation continues past `<|endoftext|>` if the model emits it (seen in the vision run). Same on Qwen3.6.

## Quick start

```bash
just fetch-qwen35-08b

# GGUF path (default fetched quant: Q4_K_M)
just qwen35 -- --weights weights/Qwen3.5-0.8B-gguf/Qwen3.5-0.8B-Q4_K_M.gguf \
  --packed --chat --prompt "Hello" --max-tokens 32

# Or safetensors directory from Qwen/Qwen3.5-0.8B-Base
just qwen35 -- --weights weights/Qwen3.5-0.8B-Base --chat --prompt "Hello" --max-tokens 32

# Generic local GGUF path
just qwen35 -- --weights model.gguf --prompt "Hello" --max-tokens 32
# or:
cargo run -p rlx-qwen35 --release --features apple-silicon -- \
  --weights model.gguf --packed --device metal --fast \
  --prompt "What is the capital of France?"
```

`--fast` turns on ChatML with thinking disabled, a tight `max_seq` (prompt + tokens), and a `prefill_seq` equal to the prompt length so prefill GEMMs are not padded to decode capacity.

### Useful CLI flags

| Flag | Role |
|------|------|
| `--device` | `cpu`, `metal`, `mlx`, `cuda`, … |
| `--packed` | Keep GGUF K-quants / Q1_0 packed (required for large Bonsai-class files) |
| `--prompt` / `--prompt-ids` | Text (needs `tokenizer`) or raw ids (`;` separates batch rows) |
| `--chat` / `--system` / `--messages-json` | ChatML formatting |
| `--no-think` / `--think` / `--thinking-budget N` / `--show-thinking` | Reasoning on/off and budget |
| `--fast` | Low-latency QA: `--no-think` + tight seqs + `prefill_seq` |
| `--max-seq` / `--max-tokens` | Decode capacity / generation length |
| `--mtp` / `--spec-decode` / `--spec-n` | MTP / speculative decode |
| `--mmproj` / `--image` | VLM (`qwen35-vlm` feature) |

`rlx-qwen35 --help` lists the same set.

### Env (optional)

| Variable | Effect |
|----------|--------|
| `RLX_QWEN35_BENCH=1` | Prefill/decode ms and tok/s on stderr |
| `RLX_QWEN35_DECODE_TRACE=1` | Per-token run/cache/lm timings |
| `RLX_QWEN35_WARM_DECODE=1` | Force decode-bucket warm (drops prefill first on long / non-short GPU contexts) |
| `RLX_QWEN35_KEEP_PREFILL=1\|0` | Override keep/drop of the prefill arena after seed |
| `RLX_LOW_MEM_COMPILE=1` | Stream packed uploads; skip broad warm (short Metal/MLX/CUDA still warm one decode bucket) |
| `RLX_QWEN35_HOST_EMBED=1\|0` | Force host-gathered token embeddings on/off |

Short `max_seq` (≤ 128) on Metal, MLX, and CUDA keeps the prefill graph and warms one decode bucket by default so the first generate does not pay a cold compile or a prefill rebuild.

Do not use `--dynamic-prefill` on CUDA for packed Q1/K-quant GGUFs yet — specialize-per-length paths can produce garbage; prefer static `--fast` / `prefill_seq`.

## Public API

```rust
use rlx_qwen35::Qwen35Runner;
use rlx_runtime::Device;

let mut runner = Qwen35Runner::builder()
    .weights("model.gguf")
    .device(Device::Metal)
    .max_seq(512)
    .prefill_seq(128)          // optional: compile prefill tighter than decode
    .enable_mtp(true)
    .packed_weights(true)
    .build()?;

let out = runner.generate(&[1, 2, 3], 32, |tok| {
    print!("{tok} ");
    true
})?;
# anyhow::Ok(())
```

Also exported: [`Qwen35Config`](src/config.rs), graph builders (`build_qwen35_graph_sized`, `build_qwen35_prefill_cache_graph`, `build_qwen35_decode_graph`), [`Qwen35DecodeCache`](src/cache.rs), [`Qwen35SpecRunner`](src/spec_runner.rs), MoE offload (`build_moe_offload`, `MoeOffloadState`), ChatML helpers (`format_chatml_with`, `split_thinking`, …), and multimodal prefill (`MultimodalPrompt`, `Qwen35VisionEncoder`).

## How it fits

| Crate | Relationship |
|---|---|
| [rlx-qwen3](../rlx-qwen3) | Shared sampling (`SampleOpts`, `sample_token`) |
| [rlx-llada2](../rlx-llada2) | TIDE predictive expert-offload API for MoE checkpoints |

## Features

| Feature | Enables |
|---|---|
| `tokenizer` (default) | Text `--prompt` encode/decode via `tokenizers` |
| `qwen35-vlm` | Image preprocess + `--image` multimodal prefill |
| `parity-llama` | llama.cpp reference (`llama-cpp-2`) for numeric parity tests |
| `metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, `coreml`, `all-backends` | Forwarded to `rlx-runtime` |

Parity vs llama.cpp is env-gated (`QWEN35_GGUF_PATH`, optional `parity-llama`). Performance notes: [src/BENCHMARKS.md](src/BENCHMARKS.md).
