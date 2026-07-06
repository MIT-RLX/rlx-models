# rlx-qwen25-vl

Alibaba **[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)** vision-language runner for RLX: a vision encoder (windowed attention + vision mRoPE + SiLU patch merger) feeding a Qwen2.5 dense LM with multimodal RoPE, plus a native **AIF** visual-KV masking probe and an in-Rust **VLMEvalKit** eval harness. Target checkpoint is Qwen2.5-VL-7B-Instruct.

Weights ship as two GGUFs (llama.cpp `convert_hf_to_gguf.py --mmproj`): the LM GGUF and the `mmproj` vision GGUF. The LM text path reuses [`rlx-qwen3`](../rlx-qwen3); the vision tower and multimodal prompt assembly are in this crate.

## Quick start

```bash
just fetch-qwen25-vl-gguf          # LM + mmproj GGUFs into .cache/qwen25-vl/gguf

# Multimodal prefill/generate (tokenizer.json must sit beside the LM GGUF,
# or set RLX_QWEN25_VL_TOKENIZER)
cargo run -p rlx-qwen25-vl --release -- \
  --weights .cache/qwen25-vl/gguf/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --mmproj  .cache/qwen25-vl/gguf/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf \
  --image crates/rlx-locateanything/fixtures/sample.jpg \
  --prompt "What is in this image?" --vlmevalkit-prompt --max-tokens 32
```

Add `--aif` (probe-driven) or `--aif-native` (native RLX Q/K probe, no Python) to enable AIF visual-token masking during decode; `--device metal|cuda|mlx|…` selects a backend.

### CLI flags

| Flag | Purpose |
|------|---------|
| `--weights` / `--mmproj` | LM GGUF and vision `mmproj` GGUF |
| `--image PATH` + `--prompt` | Multimodal run; prompt must contain `<__media__>` (or use `--vlmevalkit-prompt`) |
| `--vlmevalkit-prompt` | Wrap a plain question in the VLMEvalKit chat template |
| `--max-tokens N` | Generate N tokens (`0` = prefill only) |
| `--aif` / `--aif-native` / `--aif-ratio F` / `--aif-dynamics MODE` | AIF visual-KV masking |
| `--prompt-ids a,b,c` | Text-only prefill/generate from raw ids |
| `--device`, `--max-seq`, `--prefer-quant` | Backend, KV budget, GGUF quant preference |

## Public API

```rust
use rlx_qwen25_vl::{Qwen25VlRunner, load_tokenizer, encode_prompt, user_turn_with_media};
use rlx_qwen25_vl::vision::load_rgb_image;
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;

let mut runner = Qwen25VlRunner::builder()
    .weights("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
    .mmproj("mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf")
    .device(Device::Cpu)
    .sample(SampleOpts::greedy())
    .build()?;

let tokenizer = load_tokenizer("tokenizer.json")?;
let mut tokenize = |t: &str| encode_prompt(&tokenizer, t);

let prompt = user_turn_with_media("Describe this image.");   // inserts <__media__>
let (rgb, w, h) = load_rgb_image("photo.jpg")?;
let ids = runner.generate_multimodal(&prompt, &rgb, w, h, /*max*/ 32, &mut tokenize, /*stop*/ None)?;
# anyhow::Ok(())
```

Other runner entry points: `predict_logits` / `generate_text` (text-only), `prefill_multimodal`, `encode_image`, `generate_multimodal_aif` and `generate_multimodal_aif_native` (AIF masking), and `probe_aif_native`.

## AIF visual-KV masking

The `aif` module scores visual KV keys and blocks the highest-entropy / lowest-scored ones during decode. Native probing (`native_prefill_probe`, `dynamics_from_graph_qk_decode_step`, `compute_dynamics_eq2_prefill`, `select_adaptive_mask_ratio`) runs entirely in RLX graphs — no Python reference needed. Decode masking uses `decode_mask_row_causal` over the `VisionKeySpan`.

## VLMEvalKit eval (in Rust)

The `eval::vlmevalkit` module loads RealWorldQA / TextVQA (TSV or JSONL), scores predictions (`normalized_exact_match`, `score_prediction`), and emits a `VlmevalkitReport`. Two examples drive it:

```bash
just eval-qwen25-vl-aif        -- --weights … --mmproj … --jsonl realworldqa.jsonl --image-root … --tokenizer …
just eval-qwen25-vl-vlmevalkit -- --weights … --mmproj … --jsonl … --image-root … --tokenizer …
```

## Tests

```bash
just test-qwen25-vl-quick-check      # graph-build + AIF decode quick checks
just test-qwen25-vl-aif-algo         # AIF paper-algorithm units
just test-qwen25-vl-parity           # HF parity (needs weights; via rlx-models)
```

## How it fits

- LM text path: [`rlx-qwen3`](../rlx-qwen3) dense runner + `SampleOpts`.
- Shared multimodal traits: [`rlx-vlm-base`](../rlx-vlm-base).
- Sibling VLM/Omni runners: [`rlx-qwen3-vl`](../rlx-qwen3-vl), [`rlx-lfm-vl`](../rlx-lfm-vl), [`rlx-nemotron-omni`](../rlx-nemotron-omni).

## Features

`tokenizer` (HF tokenizers, default), `qwen25-vl-vision` (image preprocess, default), and backend forwards `metal` / `mlx` / `cuda` / `rocm` / `gpu` / `vulkan` (plus `all-backends`, `apple-silicon`, `nvidia-gpu`, `amd-gpu`, `portable-gpu`).
