# rlx-unlimited-ocr

[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) on RLX — SAM-ViT-B + CLIP-L/14-224 deep encoder, linear `2048 → 1280` projector, and a DeepSeek-V2 MoE decoder with ring-window decode.

## Status

**Vision host + compiled MoE LM on device:** SAM/CLIP/projector stay eager host-f32. The MoE decoder runs as a compiled RLX graph. Host LM pack defaults to **`LmWeightPrecision::Auto`**: F32 when RAM allows (parity), else F16, then GGUF **Q8_0**, then **Q4_0**. Override via code or CLI `--lm-precision f32|f16|bf16|q8_0|q4_0|auto`. F16/BF16 widen to F32 IR params; **Q8_0/Q4_0 stay packed in IR** via `DequantMatMul` / `DequantGroupedMatMul` + U8 typed params (same pattern as Qwen35). For **Q4_0**, attention, `lm_head`, dense MLP, and shared experts stay **F16** (routed experts stay Q4) so full-checkpoint logits stay usable (~0.997 corr vs F32 on Metal); prefer **Q8_0** when quality matters most. On Metal + Q4, the session disables fused grouped GEMV by default (crates.io `rlx` 0.2.13 `q4_0_mv_f32` nibble order); set `RLX_METAL_GROUPED_GEMV_DISABLE=0` only after upgrading to an RLX build that includes the split-nibble GEMV fix.

Checkpoint inventory, tokenizer assembly, and greedy decode checks pass against the published HF weights (`just test-unlimited-ocr-parity`). Exact HF token/text e2e needs a Python env with `transformers≈4.46` (the checkpoint’s `transformers_version`) plus `addict` / `einops` / `easydict` / `torchvision` — set `RLX_UNLIMITED_OCR_PYTHON` to that interpreter. Newer transformers builds fail to import the custom `modeling_deepseekv2.py`.

Backend matrix: synthetic tiny MoE `compile_built` + `.run` on CPU/Metal/wgpu (`just features=apple-silicon test-unlimited-ocr-backends`), including **Q8/Q4 packed-IR** prefill+decode (Q4 soft-pack asserts F16 `lm_head` + U8 experts). CUDA/ROCm/Vulkan compile via `--features all-backends` (runtime gated by `is_available`). MLX currently skips MoE run — `GroupedMatMul` / `DequantGrouped` runtime shapes diverge on that backend. Full-checkpoint greedy token parity vs CPU is env-gated (`RLX_UNLIMITED_OCR_TOKEN_PARITY=1` / `just features=apple-silicon test-unlimited-ocr-token-parity`) and forces F32 pack.

## Architecture

| Stage | Module | Notes |
|-------|--------|-------|
| Preprocess | [`preprocess`](src/preprocess.rs) | EXIF → RGB → bicubic resize/pad → `[-1,1]` NCHW; Base 1024 / Gundam 1024+640 tiles / Multi; PDF via `pdftoppm` @ 300 DPI |
| Vision — local | [`sam_tower`](src/sam_tower.rs) | SAM-ViT-B + neck + `net_2`/`net_3` → `[B,1024,q,q]` |
| Vision — global | [`clip_tower`](src/clip_tower.rs) | CLIP-L/14 fed by SAM features as patch embeds |
| Fusion | [`deep_encoder`](src/deep_encoder.rs) | `cat(clip[:,1:], flatten(sam))` → projector; pack **local → global → separator** (HF order) |
| Projector | [`projector`](src/projector.rs) | Linear `2048 → 1280` + `image_newline` / `view_seperator` |
| Decoder (eager) | [`lm_flow`](src/lm_flow.rs) | Host-f32 reference: dense L0 + MoE L1–11 + ring KV |
| Decoder (device) | [`lm_graph`](src/lm_graph.rs), [`lm_device`](src/lm_device.rs) | Compiled MHA + TopK MoE IR; host ring between decode steps |
| Decode | [`generation`](src/generation.rs), [`ngram`](src/ngram.rs) | Greedy + `SlidingWindowNoRepeatNgramProcessor` |

Config: `hidden_size=1280`, `num_hidden_layers=12`, `num_attention_heads=10`, `n_routed_experts=64`, `n_shared_experts=2`, `num_experts_per_tok=6`, `moe_intermediate_size=896`, `first_k_dense_replace=1`, `vocab_size=129280`, `max_position_embeddings=32768`, `sliding_window=128`, `use_mla=false`. Tokens: `IMAGE_TOKEN_ID=128815`, `BOS=0`, `EOS=1`.

## Setup

```bash
just fetch-unlimited-ocr
# or: huggingface-cli download baidu/Unlimited-OCR
# or: cargo run -p rlx-unlimited-ocr --features hf-download --release -- --download
```

Weights resolve from `RLX_UNLIMITED_OCR_DIR` or the Hugging Face cache (`HF_HOME` / `~/.cache/huggingface`).

## CLI

```bash
# Single image (default Gundam mode)
just unlimited-ocr -- --image page.png --device auto

# Base mode on Metal (compiled MoE LM)
just unlimited-ocr-metal -- --image page.png --mode base --device metal --max-tokens 16

# Multi-image / PDF (ngram window 1024)
just unlimited-ocr -- --images page1.png,page2.png
just unlimited-ocr -- --pdf document.pdf --max-tokens 8192

# Dry / inventory
just unlimited-ocr -- --dry
just unlimited-ocr -- --list-keys
```

PDF pages are rasterized with `pdftoppm` (Poppler) at 300 DPI.

| Flag | Default | Notes |
|------|---------|--------|
| `--model-dir` | HF cache | Dir, `hf`, or Hub id |
| `--image` | `fixtures/sample.jpg` | Single page |
| `--images` | | Comma-separated pages |
| `--pdf` | | PDF → pages → OCR |
| `--mode` | `gundam` | `base` \| `gundam` \| `multi` |
| `--prompt` | `<image>document parsing.` | Must include `<image>` for vision |
| `--device` | `auto` | `cpu`, `metal`, `cuda`, … — selects LM compile device |
| `--lm-precision` | `auto` | `f32` (parity) \| `f16` \| `bf16` \| `q8_0` \| `q4_0` \| `auto` (F32→F16→Q8→Q4 by RAM) |
| `--max-tokens` | `4096` | New-token cap |
| `--download` / `--dry` / `--list-keys` | | |

## Bench LM precision / latency / accuracy

```bash
# Tiny synthetic MoE — memory + latency (mean/p50) + accuracy vs F32
just features=apple-silicon bench-unlimited-ocr-lm-precision --device metal

# Full HF LM — skip F32 pack in the sweep; still builds F32 ref for accuracy
just features=apple-silicon bench-unlimited-ocr-lm-precision --full --device metal \
  --precisions f16,q8_0,q4_0 --seq 8 --decode-steps 4 --greedy-steps 4

# Latency only
just features=apple-silicon bench-unlimited-ocr-lm-precision --device metal --no-accuracy
```

Columns: host pack MiB, U8 typed-param MiB, pack ms, prefill/decode mean+p50 ms, tok/s, Pearson corr vs F32, max|err|, top-1 match, top-K Jaccard %. Optional `--greedy-steps N` compares N greedy argmax tokens vs an F32 reference chain.

## Rust API

```rust
use rlx_unlimited_ocr::{
    ImageMode, InferenceOptions, LmWeightPrecision, UnlimitedOcrSession, sample_image_path,
};

let options = InferenceOptions::for_ocr()
    .mode(ImageMode::Base { size: 1024 })
    .max_new_tokens(512)
    .weight_precision(LmWeightPrecision::Auto); // or F32 / F16 / Bf16 / Q8_0 / Q4_0
let mut session = UnlimitedOcrSession::open_default()?;
let out = session.run_single(sample_image_path())?;
println!("{}", out.text);
```

## Backends / tests

```bash
cargo test -p rlx-unlimited-ocr --lib
just features=apple-silicon test-unlimited-ocr-backends        # tiny MoE compile+run
just features=apple-silicon test-unlimited-ocr-token-parity    # real weights; large RAM
cargo check -p rlx-unlimited-ocr --features all-backends
just test-unlimited-ocr-parity   # needs RLX_UNLIMITED_OCR_DIR (+ optional RLX_UNLIMITED_OCR_PYTHON)
```

## Release notes (0.2.13)

- MoE LM on RLX backends (vision host); `--lm-precision auto|f32|f16|bf16|q8_0|q4_0`
- Q8/Q4 packed IR; Q4 soft-packs attn/`lm_head`/dense/shared as F16
- Metal+Q4 disables fused grouped GEMV by default until upstream ships the split-nibble `q4_0_mv_f32` fix (set `RLX_METAL_GROUPED_GEMV_DISABLE=0` on a fixed RLX build)
- Bench: `just features=apple-silicon bench-unlimited-ocr-lm-precision --full --device metal --greedy-steps 4`

Publish: upstream `rlx*` **0.2.13** is on crates.io; this repo’s `rlx-cli` / `rlx-models-core` **0.2.13** are not yet — `./scripts/publish.sh` from tier 0 (`--list`). `rlx-unlimited-ocr` is tier 3.

HF reference helper: [`scripts/hf_reference_unlimited_ocr.py`](scripts/hf_reference_unlimited_ocr.py).

## See also

- Main repo [README](../../README.md)
- [AGENTS.md](../../AGENTS.md)
