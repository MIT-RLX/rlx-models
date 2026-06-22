# rlx-bioclip2

[BioCLIP-2](https://huggingface.co/imageomics/bioclip-2) — a biology foundation
model — implemented natively in RLX. Architecturally it is a stock OpenCLIP
**ViT-L-14** (LAION-2B lineage):

| Tower  | Layers | Width | Heads | Notes |
|--------|:------:|:-----:|:-----:|-------|
| Vision | 24     | 1024  | 16    | 224px / 14-patch, bidirectional, conv1 patch stem, `ln_post` → `visual.proj` (→768) |
| Text   | 12     | 768   | 12    | 77-token causal, learned positions, EOT pooling → `text_projection` (→768) |

Activation is exact `nn.GELU` (not QuickGELU — the `-quickgelu` variant only
applies to OpenAI weights). Both towers project into a shared 768-dim space;
zero-shot logits are `exp(logit_scale) · normalize(image) · normalize(text)ᵀ`.

## Usage

```bash
hf download imageomics/bioclip-2 --local-dir weights/bioclip-2

cargo run -p rlx-bioclip2 --release -- \
  --model-dir weights/bioclip-2 \
  --image photo.jpg \
  --labels "cat, dog, bird, a striped fish" \
  --device cpu          # cpu | metal | mlx | cuda | rocm | gpu | vulkan
```

```rust
use rlx_bioclip2::BioClip2Runner;
use rlx_runtime::Device;

let mut runner = BioClip2Runner::builder()
    .model_dir("weights/bioclip-2")
    .device(Device::Cpu)
    .batch(8)            // optional: compile towers at batch 8 for throughput
    .build()?;

let logits = runner.zeroshot(&[image], &["a photo of a cat".into()])?;
```

`.batch(n)` compiles both towers for `n` items per graph run; `encode_images_nchw`
/ `encode_texts_ids` and `zeroshot` chunk transparently (default `n = 1`, which
preserves single-item behavior). `.patch_features(true)` switches the vision
tower to dense per-patch output (`[n_patches × width]`, DINOv2-style); the
`rlx-bioclip2-batch` binary uses it for feature extraction.

Image preprocessing is pure Rust — no Python. The shared, PIL-faithful resampler
lives in `rlx_models_core::image_preprocess` (`ImagePreprocessor`) so any vision
model can reuse it; it reproduces Pillow's antialiased bicubic
coefficient scheme + 8-bit two-pass resize, matching open_clip's `pixel_values`
to ~1 LSB.

## Backends

The crate exposes the standard RLX backend features
(`metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, plus `all-backends` /
`apple-silicon` / `nvidia-gpu` / `amd-gpu` / `portable-gpu`). The towers compile
through the backend-agnostic `rlx-runtime` session, so any enabled + available
device runs the same graph.

## Parity

`tests/parity.rs` compares rlx features against an OpenCLIP reference:

```bash
pip install open_clip_torch torch numpy pillow
RLX_BIOCLIP2_FIXTURE=/tmp/bioclip2_fixture RLX_BIOCLIP2_MODEL=$PWD/weights/bioclip-2 \
  python3 scripts/bioclip2_dump_reference.py

RLX_BIOCLIP2_FIXTURE=/tmp/bioclip2_fixture RLX_BIOCLIP2_MODEL=$PWD/weights/bioclip-2 \
  cargo test -p rlx-bioclip2 --release -- --nocapture
```

Measured parity vs `open_clip` (ViT-L-14, full f32) — max\|Δ\| per output,
cosine 1.000000 for image and every text embedding on all backends:

| Backend | image | text | zero-shot logits |
|---------|-------|------|------------------|
| CPU     | 1.7e-5 | < 1e-5 | 3.4e-5 |
| Metal   | 2.0e-5 | 1.1e-5 | 5.7e-5 |
| MLX     | 3.7e-5 | 3.2e-5 | 1.4e-4 |
| wgpu    | 1.5e-5 | 8.3e-6 | 7.6e-5 |

All far inside the 1e-3 feature tolerance — effectively bit-exact. CUDA / ROCm /
Vulkan reuse the same feature-gated test and skip when the device is unavailable.
Batched encoding (`.batch(n)`) is verified to match these numbers exactly, and
the pure-Rust preprocessing matches open_clip's `pixel_values` to within one
8-bit step (max\|Δ\| 1.5e-2, mean\|Δ\| 2.2e-4).

The parity test feeds the reference's exact post-preprocess pixel tensor so the
comparison isolates network correctness (PIL bicubic resize is platform-specific
and reproduced separately for the real-world CLI path). The token-embedding
lookup is done on host (like the conv1 patch stem) so the text graph is a pure
float pipeline — no in-graph integer `Gather` — which keeps all backends,
including MLX's compiled path, on the same numerically-exact route.
