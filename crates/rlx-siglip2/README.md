# rlx-siglip2

[SigLIP 2](https://huggingface.co/blog/siglip2) — Google's sigmoid-loss
vision-language model — implemented natively in RLX. Two transformer towers
project images and text into a shared space; zero-shot classification is an
independent **per-pair sigmoid**, not a softmax over labels.

The crate covers **both** SigLIP 2 architecture families (checkpoints load with
no key remapping — tensor names match the published HuggingFace
`model.safetensors`):

| Family | HF `model_type` | Patch stem | Sequence | Example checkpoint |
| --- | --- | --- | --- | --- |
| **Fixed-resolution** | `siglip` | Conv2d(3→W, k=s=`patch`) **+ bias** | fixed `(img/patch)²`, **no CLS** | `google/siglip2-base-patch16-224` |
| **NaFlex** | `siglip2` | Linear(`C·p²`→W) on pre-unfolded patches | variable ≤ `max_num_patches`, padded + masked | `google/siglip2-base-patch16-naflex` |

Both towers are otherwise identical:

| Tower | Notes |
| --- | --- |
| Vision | pre-LN encoder, separate `q/k/v/out_proj`, **`gelu_pytorch_tanh`**, `post_layernorm`, then an **attention-pooling (MAP) head** — a learned probe cross-attends the patch sequence → pooled image embedding |
| Text | learned positions, **bidirectional** (no causal mask, no padding mask), `final_layer_norm`, pooled at the **last** position → linear `head` |

`layer_norm_eps = 1e-6`; the image/text embedding dim equals the text
`projection_size` (768 for base). Zero-shot logits are
`logit_scale.exp() · normalize(image) · normalize(text)ᵀ + logit_bias`, passed
through a sigmoid for independent match probabilities.

The fixed-resolution checkpoints are, architecturally, the *original* SigLIP
model (their `config.json` sets `model_type: "siglip"` and relies on the HF
class defaults); the SigLIP 2 improvements are in training and in the NaFlex
variant. This crate is an **inference** port — the LocCa decoder and
self-distillation training objectives are out of scope.

## Usage

```bash
hf download google/siglip2-base-patch16-224   --local-dir weights/siglip2-base-224
hf download google/siglip2-base-patch16-naflex --local-dir weights/siglip2-base-naflex

# Fixed-resolution (square resize to the model's image_size)
cargo run -p rlx-siglip2 --release -- \
  --model-dir weights/siglip2-base-224 \
  --image photo.jpg \
  --labels "a cat, a dog, a green field, the ocean" \
  --device cpu          # cpu | metal | mlx | cuda | rocm | gpu | vulkan

# NaFlex (variable resolution / native aspect ratio) — auto-detected from config.json
cargo run -p rlx-siglip2 --release -- \
  --model-dir weights/siglip2-base-naflex --image doc.png --labels "a document, a photo"
```

The CLI prints per-label sigmoid match probabilities (ranked). By default each
label is wrapped in SigLIP's caption template `"This is a photo of {label}."`;
pass `--raw-prompts` to use the labels verbatim.

| flag | meaning |
| --- | --- |
| `--model-dir <dir>` | dir with `model.safetensors`, `config.json`, `tokenizer.json` (required) |
| `--image <path>` | input image (required) |
| `--labels "a,b,c"` | comma-separated zero-shot labels; omit to print the image embedding norm |
| `--raw-prompts` | use labels verbatim (skip the caption template) |
| `--device <dev>` | `cpu` \| `metal` \| `mlx` \| `cuda` \| `rocm` \| `gpu` (wgpu) \| `vulkan` |

## Library

```rust
use rlx_siglip2::Siglip2Runner;
use rlx_runtime::Device;

let mut runner = Siglip2Runner::builder()
    .model_dir("weights/siglip2-base-224")
    .device(Device::Cpu)      // cpu | metal | mlx | gpu (wgpu) | cuda | …
    .build()?;                // variant (fixed / NaFlex) is read from config.json

// Zero-shot: logits_per_image = scale·⟨î, t̂⟩ + bias (apply sigmoid for probs).
let logits = runner.zeroshot(&[image], &["a photo of a cat".into()])?;

// Or raw (un-normalized) embeddings, matching HF pooler_output:
let img_embed: Vec<f32> = runner.encode_image(&image)?;   // [embed_dim]
let txt_embed: Vec<f32> = runner.encode_text("a photo of a cat")?;
```

`.batch(n)` compiles both towers for `n` items per graph run (fixed-resolution
only; NaFlex runs one image per call because each carries its own padding mask).

Preprocessing and tokenization are pure Rust — no Python at inference:

- **Fixed-res image**: bilinear resize to a square `image_size` (no crop,
  mean = std = 0.5), via the shared PIL-faithful `ImagePreprocessor`.
- **NaFlex image** (`src/naflex.rs`): aspect-preserving resize to fit
  `max_num_patches`, patchify, and a per-image bilinear-antialias resize of the
  position-embedding grid (matches PyTorch `F.interpolate(antialias=True)`).
- **Text**: the multilingual **Gemma** tokenizer loaded from `tokenizer.json`
  via the `tokenizers` crate — `<eos>`-appended, right-`<pad>`ded to 64.

The vision patch stem, position/token embeddings, MAP-head QKV split, text head,
and logits run on host; the 12-layer transformer + `post_layernorm` + pooling
attention run in the compiled graph — the real per-backend surface.

## Backends

Standard RLX backend features (`metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`,
plus `all-backends` / `apple-silicon` / `nvidia-gpu` / `amd-gpu` /
`portable-gpu`). The towers compile through the backend-agnostic `rlx-runtime`
session, so any enabled + available device runs the same graph. Every operation
used (LayerNorm, MatMul, Add, tanh-GELU, and cross-/key-padding-masked
attention) has full backend coverage.

## Parity

`tests/parity.rs` compares rlx embeddings/logits against a HuggingFace reference
dumped by `scripts/siglip2_hf_dump.py`:

```bash
pip install transformers torch pillow numpy
python3 scripts/siglip2_hf_dump.py \
  --model weights/siglip2-base-224 --out weights/siglip2-base-224/fixture
python3 scripts/siglip2_hf_dump.py \
  --model weights/siglip2-base-naflex --out weights/siglip2-base-naflex/fixture

RLX_SIGLIP2_MODEL=$PWD/weights/siglip2-base-224 \
RLX_SIGLIP2_NAFLEX_MODEL=$PWD/weights/siglip2-base-naflex \
  cargo test -p rlx-siglip2 --features "metal mlx gpu" -- --nocapture --test-threads=1
```

The test also checks the Gemma tokenizer ids and the image preprocessing against
the same reference. Measured parity vs HuggingFace (`siglip2-base-patch16`, f32)
— **cosine 1.000000** for the image and every text embedding on every backend,
both variants:

| Backend | fixed-res image max\|Δ\| | NaFlex image max\|Δ\| | cosine |
| --- | --- | --- | --- |
| CPU | 8.6e-6 | 3.7e-6 | 1.000000 |
| Metal | 2.9e-6 | 3.5e-6 | 1.000000 |
| MLX | 3.2e-6 | 3.1e-6 | 1.000000 |
| wgpu | 5.7e-6 | 1.0e-5 | 1.000000 |
| CUDA | 8.9e-4 | 5.4e-4 | 1.000000 |

(CUDA's slightly larger absolute delta is matmul accumulation order; direction is
identical and the normalized logits are unaffected.)

> **Masking note.** The NaFlex padding mask is a pure *key-padding* mask, so both
> the encoder and the MAP head use `MaskKind::Custom` (a binary `[B, seq]` mask,
> `1 = attend`), which is implemented on every backend. An additive
> `MaskKind::Bias` tensor was intentionally avoided: it is silently mishandled on
> Metal (the attention kernel lacks a bias arm) and wgpu (the bias buffer isn't
> bound), so it passes on CPU/MLX but corrupts Metal/wgpu.
>
> **Reference note.** HuggingFace `transformers` ≥ 5 L2-normalizes
> `output.image_embeds` / `text_embeds`. For raw-magnitude parity the dump reads
> `vision_model(...).pooler_output` / `text_model(...).pooler_output` — the
> un-normalized values `encode_image` / `encode_text` return.

## License

GPL-3.0-only (workspace). SigLIP 2 weights are Google's, released under Apache
2.0 (not gated); download them yourself.
