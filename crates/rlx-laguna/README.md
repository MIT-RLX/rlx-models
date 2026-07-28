# rlx-laguna

[Poolside Laguna](https://huggingface.co/poolside/Laguna-S-2.1) MoE on RLX — hybrid full/SWA attention, softplus attention output gate, sigmoid TopK router + shared expert.

## Table of contents

- [Status](#status)
  - [Memory policy (F32 expand off by default)](#memory-policy-f32-expand-off-by-default)
- [Commands](#commands)
- [OpenAI-compatible server](#openai-compatible-server)

| Property | Laguna S 2.1 | Laguna XS 2.1 |
|----------|--------------|---------------|
| Arch tag | `laguna` / `LagunaForCausalLM` | same |
| Size | ~118B total / ~8B active | ~33B-A3B (lighter bring-up) |
| Layers / MoE | 48 · 256 routed + 1 shared · top-10 | 40 · 256 + 1 · top-8 |
| Attention | 1:3 full:SWA (window 512); per-head softplus gate; QK-norm | same recipe |
| RoPE | YaRN on full layers; plain RoPE on SWA; per-layer head counts | same |
| GGUF | [unsloth/Laguna-S-2.1-GGUF](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF) | [poolside/Laguna-XS-2.1-GGUF](https://huggingface.co/poolside/Laguna-XS-2.1-GGUF) (`general.architecture = laguna`) |

## Status

Packed production path (no quant→F32 expand):

| Capability | Notes |
|------------|--------|
| HF `config.json` / GGUF metadata | S + XS dims, SWA pattern, dual RoPE, MoE |
| GGUF header sniff | `--weights` — metadata only |
| Packed mmap load | `--packed-load` — mats stay quantized; norms/biases native F32 |
| Packed generate | KV-cached greedy decode (prefill + one-token steps) |
| Device accel | `--device metal\|mlx\|cpu` — `Op::DequantMatMul` for packed mats (Metal preferred on Apple) |
| Tokenizer + chat | `--tokenizer-dir` + `--prompt` / `--system` (Jinja; strips HF `{% generation %}`) |
| OpenAI HTTP | [`rlx-openai`](../rlx-openai) `--engine laguna` (or crate `--serve`); **greedy only** |
| YaRN @ full layers | inv-freq via `rlx_flow::rope::yarn_scaled_inv_freq` + `attention_factor` scale; SWA keeps plain RoPE |
| Synth reference | `--synth` — tiny eager CPU graph for unit tests |
| Backend spot bench | `just laguna-backend-bench` — DequantMatMul parity across Host/CPU/Metal/MLX/… |
| Compiled e2e IR | **Scaffolded** (`builder::build_prefill_graph`); generate remains packed KV + DeviceMatmul |
| DFlash draft | **Scaffolded** (`dflash`); HF drafts: [XS-DFlash](https://huggingface.co/poolside/Laguna-XS-2.1-DFlash) / [S-DFlash](https://huggingface.co/poolside/Laguna-S-2.1-DFlash) — distinct from EAGLE3/MTP |

GGUF may default `attention_factor=1.0` unless HF JSON filled those fields.

### Memory policy (F32 expand off by default)

**Production GGUFs stay packed unless you opt in.** Quant→F32 expand is
disabled by default ([`ALLOW_F32_EXPAND`](src/memory.rs) default `false`;
runtime via [`allow_f32_expand`](src/memory.rs)). Header sniff prints the
per-file estimate (`packed≈` vs `F32-expand≈`); for the XS `Q4_K_M` shard
that is typically ~20 GB packed → ~134 GB if every element is widened.

Opt in with `RLX_LAGUNA_ALLOW_F32_EXPAND=1` or `--allow-f32-expand` (enables
`GgufLoader::take` / `load_weight_map` / `LagunaRunner::try_from_gguf_f32`).
Native F32/F16/BF16 norms/biases may always copy via `take_native_f32`. Expert
mats on the packed path use fused host dequant (or Metal) — never the
process-wide F32 dequant cache unless opted in.

| Path | Loads tensor payload? | Expands quants to F32? |
|------|----------------------|-----------------|
| `just laguna -- --weights …gguf` | No (header-only) | No |
| `… --weights … --packed-load` | mmap + metadata; norms F32 | No |
| `LagunaPackedRunner::from_gguf_packed` | mmap retained | No |
| `LagunaConfig::from_gguf_path` | No | No |
| `LagunaRunner::try_from_gguf_f32` (default) | — | **Error** |
| `… --allow-f32-expand` / env=1 + `try_from_gguf_f32` | Yes | **Yes (opt-in)** |
| `rlx_core::weights::open_map` on laguna (default) | — | **Error** |
| Synth `--synth` | N/A (tiny host tensors) | N/A |

## Commands

```bash
just test-laguna

# Fetch Unsloth Laguna-S GGUF (UD-Q4_K_M shards) + Poolside tokenizer → .cache/laguna-s
# S UD-Q4_K_M ≈73 GB packed — needs a high-RAM host. On ≤64 GB use XS instead:
just fetch-laguna-xs   # ~20 GB Q4_K_M → .cache/laguna-xs
just fetch-laguna      # only if you have headroom for ~73 GB + KV/OS

# Synth (no weights)
just laguna -- --synth --prompt-ids 1,2,3 --max-tokens 4

# Header sniff (dir + nested prefer → first split shard)
just laguna -- --weights .cache/laguna-s --prefer Q4_K_M

# Packed generate (host kernels, raw ids)
just laguna -- --weights .cache/laguna-s --prefer Q4_K_M --packed-load \
  --max-tokens 32 --prompt-ids 1,2,3

# Packed generate with in-crate tokenizer + chat template
# (host fused kernels are fastest for MoE chat/decode on Apple Silicon;
#  `--device metal` is skipped when prompt_len < 128)
just features=apple-silicon laguna -- \
  --weights .cache/laguna-s --prefer Q4_K_M --packed-load \
  --device metal --tokenizer-dir .cache/laguna-s \
  --prompt "Say hello" --max-tokens 8

# Packed generate on Metal (needs apple-silicon / metal feature)
just features=apple-silicon laguna -- --weights .cache/laguna-s --prefer Q4_K_M \
  --packed-load --device metal --max-tokens 32 --prompt-ids …

# DequantMatMul backend speed + parity
just laguna-backend-bench -- --weights .cache/laguna-xs/Laguna-XS-2.1-Q4_K_M.gguf
# (XS is smaller for bench; or pass the Laguna-S first-shard path after fetch-laguna)

# Optional Hub GGUF layout probe (Range GETs only):
just laguna-probe-gguf
```

`--weights` may be a Hub checkout root: Unsloth nests quants under
`UD-Q4_K_M/` (3-way splits). `--prefer Q4_K_M` (default when unresolved)
scans one child-dir level and picks `*-00001-of-*.gguf`. Override the fetch
folder with `just fetch-laguna QUANT=UD-Q4_K_XL`.

Inference stays on **RLX**. The Unsloth / Poolside GGUF is the weight format — not a llama.cpp runtime dependency (llama.cpp [#25165](https://github.com/ggml-org/llama.cpp/pull/25165) is the reference converter).

## OpenAI-compatible server

Use the central host ([`rlx-openai`](../rlx-openai/README.md)):

```bash
just features=apple-silicon,laguna openai-serve -- \
  --engine laguna --weights .cache/laguna-s --prefer Q4_K_M \
  --tokenizer-dir .cache/laguna-s --device metal \
  --host 127.0.0.1 --port 8080 --model-id laguna
```

`LagunaEngine` is greedy-only (`temperature` / `top_p` ignored). Single-model
convenience: `just features=apple-silicon laguna-serve -- --weights … --tokenizer-dir …`.

```bash
curl -s http://127.0.0.1:8080/v1/models | jq .
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"laguna","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}' | jq .
```
