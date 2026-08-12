# rlx-diffusiongemma

[DiffusionGemma-26B-A4B](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
for [RLX](https://github.com/rlx-ai/rlx) — a **discrete text diffusion** LM
(25.2 B total / 3.8 B active) that denoises a whole block of tokens at once
instead of emitting them one at a time.

## Architecture

The backbone is Gemma 4's MoE stack (30 layers, 128 experts with 8 active plus
an always-on shared expert, 5:1 sliding:full attention), wired as an
**encoder/decoder pair that shares all its weights**:

| | encoder | decoder ("denoiser") |
|---|---|---|
| input | the prompt | a fixed 256-token canvas |
| attention | causal (windowed on sliding layers) | **bidirectional** over `[cache ; canvas]` |
| KV cache | writes it | reads it, never writes |
| runs | once per block | once per denoising step |

The only weight the two stacks do not share is the per-layer `layer_scalar`.

Three details are easy to miss, and each one silently corrupts output:

* **Full-attention layers have no `v_proj`.** V is the *pre-`k_norm`* K
  projection, and never gets RoPE. Those layers also use a different geometry
  from sliding layers — 16 heads × 512 with 2 KV heads, vs 16 × 256 with 8.
* **Attention `scaling` is 1.0**, not `1/sqrt(head_dim)`, because Q and K are
  RMS-normed per head. V gets a scale-free RMS norm too.
* **The FFN is a two-branch block.** Both branches read the same post-attention
  residual, but the router scores the *raw* residual while the experts consume a
  separately normalized copy (`pre_feedforward_layernorm_2`).

`rope_type: "proportional"` on the global layers is not partial-width RoPE in
the usual sense: it emits a full `head_dim`-wide table where the trailing
`head_dim/2 - int(p·head_dim//2)` angle slots have `inv_freq = 0`, so those
channels pass through unrotated.

## Generation

Each block starts from a canvas of uniform random tokens and is denoised for up
to `max_denoising_steps` (48). Every step the model rescores the whole canvas;
the sampler accepts the lowest-entropy positions whose joint mutual information
stays under `entropy_bound`,

```text
Σᵢ₌₁..ₖ Hᵢ − max(H₁..Hₖ) ≤ entropy_bound
```

re-noises the rest, and feeds the step's soft embeddings back as a
self-conditioning signal. That is what buys ~15–20 tokens per forward pass.
Denoising stops early once the draft is stable across `stability_threshold`
steps and its mean entropy is under `confidence_threshold`.

## Usage

```rust,no_run
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_diffusiongemma::{
    DiffusionGemmaConfig, EncoderCacheLens, build_decoder_flow, build_encoder_flow,
    prepare_checkpoint,
};
use rlx_runtime::Device;
# fn main() -> anyhow::Result<()> {
let dir = std::path::Path::new("/weights/diffusiongemma-26B-A4B-it");
let cfg = DiffusionGemmaConfig::from_file(dir.join("config.json"))?;
let mut wm = WeightMap::from_safetensors_dir(dir)?;
prepare_checkpoint(&cfg, &mut wm)?;

let prompt_len = 32;
let encoder = build_encoder_flow(&cfg, &wm, prompt_len)?;
let cache = EncoderCacheLens::for_prompt(&cfg.text_config, prompt_len);
let decoder = build_decoder_flow(&cfg, &wm, cfg.canvas_length, cache)?;

let mut enc = compile_built(encoder, Device::Cpu)?;
let mut dec = compile_built(decoder, Device::Cpu)?;
# let _ = (&mut enc, &mut dec);
# Ok(())
# }
```

`BlockDiffusion` in [`generate`](src/generate.rs) drives the full loop against a
`Denoiser` implementation.

## Images

```rust,ignore
// Compile/run steps elided; `?` on each builder.
let img = preprocess_image(&rgb, h, w, ImagePreprocessConfig::default())?;   // resize, patchify, pad

// The vision graph is built for the *padded* budget, so one compiled graph
// serves every image regardless of aspect ratio.
let vision = build_vision_flow(&cfg, &wm, cfg_pp.max_patches(), cfg_pp.max_soft_tokens)?;
// ... compile(vision), run with img.pixels / img.pos_x / img.pos_y /
//     img.valid / img.pool + vision_rope_tables(...) -> `soft` [max_soft, hidden]

// Keep only the rows this image actually occupies, then render a prompt with
// exactly that many image slots and splice them in.
let soft = &soft[..img.num_soft_tokens * cfg.text_config.hidden_size];
let msgs = [ChatMessage::user_with_images(1, "What is this?")];
let text = format_chat(&msgs, ChatOptions::default(), &[img.num_soft_tokens])?;
let ids = tokenizer.encode(&text)?;
let embeds = merge_multimodal_embeds(&cfg, &wm, &ids, soft)?;
let encoder = build_encoder_flow_embeds(&cfg, &wm, ids.len())?;
```

The per-image slot count is that image's own `patches / pooling_kernel²`, not
the padded budget, so a small image contributes fewer slots than a large one.
Soft tokens are spliced in *unscaled* while text rows carry the `sqrt(hidden)`
embedding scale — `merge_multimodal_embeds` handles both.

DiffusionGemma is **text + images only**: the reference raises
`NotImplementedError` for audio and video, so neither is ported.

## Performance note

`build_decoder_flow_with(.., DecoderOutputs::Reduced { seed })` reduces each
denoising step to `entropy` / `argmax` / `sampled` inside the graph, so the
`[canvas, vocab]` logit block never leaves the device — 268 MB per step at
production size, versus a few KB. Sampling there uses Gumbel-max: same
distribution as the host sampler, different RNG stream. The default
(`DecoderOutputs::Logits`) returns full logits and is what the parity tests use.

## Tests

```sh
cargo test -p rlx-diffusiongemma

# Numeric parity against a PyTorch transcription of the reference forward:
python3 scripts/diffusiongemma_reference.py .fixtures/diffusiongemma-parity
RLX_DG_PARITY_DIR=$PWD/.fixtures/diffusiongemma-parity \
    cargo test -p rlx-diffusiongemma --test parity_reference
```

| test | what it pins | needs |
|---|---|---|
| `text_flow_smoke` | both graphs compile and run; encoder causal, denoiser bidirectional | — |
| `real_checkpoint_contract` | every real tensor name + shape, from the shard headers | — |
| `parity_reference` | layer-by-layer arithmetic vs torch; vision stage by stage; resampler vs Pillow; chat template vs the shipped Jinja | torch, PIL, jinja2 |
| `real_vision`, `real_layer` | the trained weights (see below) | a fetched subset |
| `real_layer_precision` | what f16 / Q8_0 / Q4_1 / Q4_0 cost | a fetched subset |

`RLX_TEST_DEVICE` switches backends; everything above passes on CPU and Metal.

## Real weights



Safetensors headers carry per-tensor byte offsets, so single subsystems can be
pulled with HTTP Range requests instead of downloading all 51 GB:

```sh
python3 scripts/diffusiongemma_fetch_subset.py /w/dg-vision --subset vision   # 1.15 GB
python3 scripts/diffusiongemma_real_vision.py /w/dg-vision
RLX_DG_REAL_VISION_DIR=/w/dg-vision cargo test -p rlx-diffusiongemma --release --test real_vision

python3 scripts/diffusiongemma_fetch_subset.py /w/dg-layer0 --subset layer0   # 1.63 GB
python3 scripts/diffusiongemma_real_layer.py /w/dg-layer0
RLX_DG_REAL_LAYER_DIR=/w/dg-layer0 cargo test -p rlx-diffusiongemma --release --test real_layer
```

Measured against torch on the trained weights (relative *mean* error):

| subsystem | CPU cosine | CPU rel. mean | Metal cosine | Metal rel. mean |
|---|---|---|---|---|
| vision patch embed (540 patches) | 1.00000000 | 0 | 1.00000000 | 0 |
| vision encoder out (27 layers) | 0.99999184 | 6.6e-5 | 1.00000000 | 8.3e-8 |
| vision soft tokens | 0.99999473 | 2.7e-4 | 1.00000000 | 3.2e-7 |
| text layer 0 output (54/128 experts) | 1.00000000 | 9.7e-9 | 1.00000000 | 1.1e-8 |

The layer test is the one that matters most for the MoE: a random router spreads
roughly uniformly, whereas the trained one routes each token to a specific 8 of
128 banks, so it actually exercises top-k dispatch.

Note the vision column: **Metal is ~800× more accurate than CPU** on the
27-layer tower, while both are exact on a single text layer. So the CPU drift is
not depth alone — it is the CPU backend accumulating less accurately on a deep
stack whose activations reach ~2250. Worth knowing before treating a CPU
cosine of 0.99999 as the model's own noise floor.

### What each precision costs

`tests/real_layer_precision.rs` round-trips the real weights through each format
and re-runs the layer (cosine vs the f32 result on the same graph):

| format | bits/wt | experts only | all weights | model experts |
|---|---|---|---|---|
| bf16 | 16 | 1.00000000 | 1.00000000 | 45.7 GB |
| f16 | 16 | 1.00000000 | 1.00000000 | 45.7 GB |
| Q8_0 | 8.5 | 0.99999687 | 0.99904031 | 24.3 GB |
| Q4_1 | 5.0 | 0.99932107 | 0.99673516 | 14.3 GB |
| Q4_0 | 4.5 | 0.99912190 | 0.99591416 | 12.8 GB |

bf16 round-trips exactly, which is the expected result for a bf16 checkpoint and
a good check on the harness. f16 is numerically free here — the weights sit well
inside its range. Quantizing the *experts* is much cheaper than quantizing
everything (Q8_0: 0.999997 vs 0.99904), which is the usual argument for
mixed-precision schemes that keep attention wider than the expert banks.

Those figures are accuracy only: they round-trip weights through a format and
run the f32 graph, and deliberately do not exercise rlx's packed
`Op::DequantGroupedMatMul` kernels.

## Status

Both graphs, the vision tower, the image processor, the chat template, the
checkpoint adapter and the sampler are implemented and covered by tests, on CPU
and Metal.

Not done:

* **No whole-model forward pass.** Every name and shape is verified against the
  real shard headers and both subsystems match on real weights, but running all
  30 layers at once is blocked on memory: `WeightMap` is f32, so the routed
  experts alone are 91.4 GB. The sweep above says the fix is a *format*, not
  more RAM — Q4_0 experts are 12.8 GB and cost ~1e-3 of cosine — so the missing
  piece is a packed loader feeding `Op::DequantGroupedMatMul`, which rlx already
  has kernels for.
* **Batched images share one vision graph invocation each** — multiple images in
  a prompt are supported (each contributes its own soft-token count), but they
  are encoded one at a time rather than as a batch.
