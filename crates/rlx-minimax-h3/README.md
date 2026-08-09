# rlx-minimax-h3

Native RLX port of [**MiniMax-H3**](https://huggingface.co/MiniMaxAI/MiniMax-H3)
(Hailuo 3.0) — an omni-modal model that generates a video **and its synchronized
soundtrack jointly**, from one flow-matching transformer.

H3 is not a video model with audio bolted on. A single stack of 50 blocks runs
**full self-attention over one packed sequence** holding the text condition, the
conditioning media, the audio rows and the target video rows at once. There is
no cross-attention and no per-modality block weights anywhere in the model. Only
three things are modality-specific: the two input patch projections, a per-row
AdaLN tag, and the two output heads.

## Architecture

| Component | Shape | Module |
|---|---|---|
| Joint video+audio DiT | 50 blocks, hidden 5376, 56 x 128 heads (inner 7168), SwiGLU 14336 | [`transformer`](src/transformer.rs) |
| Token refiner | 2 plain pre-norm blocks over the text stream | [`transformer`](src/transformer.rs) |
| Packed layout | `(t, h, w)` rotary grid, modality tags, row indices | [`layout`](src/layout.rs) |
| Flow scheduler | rectified-flow Euler, `shift` 12 (video) / 3 (audio) | [`scheduler`](src/scheduler.rs) |
| Video VAE | 16x spatial / 4x temporal, 24 latent ch; **36-layer ViT decoder** | [`vae_video`](src/vae_video.rs) |
| Audio VAE | 32 kHz, 800-sample hop (40 latents/s), 32 latent ch; BigVGAN decoder | [`vae_audio`](src/vae_audio.rs) |
| Text conditioning | Qwen3-VL, tapped at decoder **layer 50** | [`text_encoder`](src/text_encoder.rs) |

`rlx-minimax-h3 inspect` on the released checkpoint reports **33.12B**
parameters, **13.01B** of them in the AdaLN branches — matching the model card.

### Details that are easy to get wrong

- **The velocity points at the data.** `x0 = x_t + sigma * v`, a plus, where
  diffusers' flow-match Euler subtracts. Timesteps are `t = 1 - sigma` on
  `[0, 1]` with `t = 1` meaning *clean*, consumed unscaled — there is no `* 1000`.
- **Two schedules per request**, video and audio, with different shifts. At the
  same step index the two modalities sit at different noise levels, which is
  exactly why the DiT takes a *per-row* timestep.
- **The AdaLN table is addressed by `(timestep, modality)`**, as
  `timestep_index * 3 + tag` where `0 = video, 1 = text, 2 = audio`. Those
  projections hold ~13B of the ~33B parameters and are re-read every step, so
  they dominate both footprint and bandwidth.
- **Partial RoPE.** One `inv_freq` buffer of 16 frequencies is shared by the
  three axes; the blocks are concatenated and duplicated, so **96 of the 128**
  head channels rotate and 32 pass through.
- **Keyframe rows live in the text stream but are tagged video (`0`).**
- **Conditioning rows must be written back after every step.** The DiT predicts
  a velocity for every row including the anchors; letting those drift is what
  turns a keyframe into a smear.
- **The video VAE's rotary grid is length-normalized** to `[-1, 1)` and scaled by
  `2*pi` — resolution independent, the opposite convention from the DiT's fixed
  32x grid.
- **The text tap is layer 50, not the last layer.** The final layer is post-norm
  and is not what the released weights were trained against; tapping it is a
  silent quality regression rather than a crash.
- **`rope_n` with `n_rot < head_dim` is wrong on Metal and wgpu.** This is an
  upstream backend bug, not an H3 one, and it matters here because H3 rotates 96
  of 128 channels in the DiT and 48 of 64 in the video decoder. See
  [Backends](#backends).

## Backends

| Backend | DiT | Video VAE decoder | Text encoder |
|---|---|---|---|
| CPU (reference) | — | — | — |
| Metal | cos 1.000000, rel 3.0e-7 | cos 1.000000, rel 1.8e-7 | cos 1.000000, rel 6.5e-7 |
| MLX | cos 1.000000, rel 2.3e-7 | cos 1.000000, rel 1.8e-7 | cos 1.000000, rel 8.1e-7 |
| wgpu | cos 1.000000, rel 2.3e-7 | cos 1.000000, rel 1.8e-7 | cos 1.000000, rel 6.5e-7 |

```bash
cargo test -p rlx-minimax-h3 --release --features apple-silicon --test backends -- --nocapture
```

### The partial-RoPE bug

Getting there took finding a real backend bug. `rope_n` with `n_rot < head_dim`
returns **wrong values on Metal and wgpu** — relative error of order 1, not
rounding noise — while full rotation and MLX are exact.
`examples/rope_probe.rs` isolates it:

```text
hd= 128 n_rot=128 full   :  metal=1.04e-7  mlx=0.00e0  wgpu=1.04e-7
hd= 128 n_rot= 96 partial:  metal=2.00e0   mlx=0.00e0  wgpu=2.00e0
hd=  64 n_rot= 48 partial:  metal=1.77e0   mlx=0.00e0  wgpu=1.77e0
```

The cause is the cos/sin **table row stride**: those kernels indexed it by
`head_dim/2`, but the table holds exactly `n_rot/2` angles per token, so every
position after the first read into the next token's angles. The CPU thunk had
already been fixed for this; the two GPU kernels had not.

**Fixed upstream** in `rlx-metal` (three kernels: half-precision forward, f32
forward, backward) and `rlx-wgpu` (`rope.wgsl`, both NeoX and GPT-J branches),
guarded by a new `metal_partial_rope_matches_cpu` regression test — which fails
with `max abs diff 2.515` against the old kernel. Full rotation is unaffected
(`n_rot == head_dim` makes the two strides equal), and rlx's own suites stay
green: metal 200/201, wgpu 203/203, cpu 267/267, runtime 695/697 (the three
failures are pre-existing and unrelated — MPSGraph GELU and two others, verified
identical with the change reverted).

`rope::emit_partial_rope` still routes around the op, because this workspace
pins `rlx*` at `^0.2.14` from crates.io for published and fresh-clone builds,
which carry the unfixed kernels. It slices the rotated channels of every head
into their own contiguous tensor, rotates with a **full** rope of width `n_rot`,
and concatenates the tail back — exact everywhere, CPU output unchanged bit for
bit. It can be dropped once a release with the fix is pinned.

## Tasks

| Task | Layout | DiT partition |
|---|---|---|
| `t2va` | `[text \| target audio \| target video]` | `transformer/` |
| `i2va` | one keyframe anchors an end | `transformer/` |
| `fl2va` | first *and* last keyframes anchor both ends | `transformer/` |
| `ref2va` | image / video / audio reference blocks on a shared rotary clock | `transformer_ref/` |

## Usage

```bash
# What is in this checkpoint?
cargo run -p rlx-minimax-h3 --release -- inspect --weights /path/to/MiniMax-H3

# What does a request cost? Resolves geometry and prints the packed layout.
cargo run -p rlx-minimax-h3 --release -- \
    plan --weights /path/to/MiniMax-H3 --task fl2va --steps 32

# Turn latents into media (both run on the VAEs alone — no DiT needed).
cargo run -p rlx-minimax-h3 --release -- \
    decode-video --weights /path/to/MiniMax-H3 --latents v.safetensors --out frames/
cargo run -p rlx-minimax-h3 --release -- \
    decode-audio --weights /path/to/MiniMax-H3 --latents a.safetensors --out out.wav
```

`plan` for the default 16:9 canvas at 124 frames:

```text
task              fl2va
canvas            1344x768 (multiple of 32)
frames            124 at 24 fps = 5.167 s
latents           37 frames x 48x84
audio latents     207 per channel x 2 channels

packed sequence   39790 rows
  text            64
  video           39312 (2016 conditioning)
  audio           414 (0 reference)

schedules         video 31 steps (shift 12), audio 31 steps (shift 3)
residual stream   0.80 GiB per activation (39790 rows x 5376 x f32)
```

Library:

```rust
use rlx_minimax_h3::{H3Geometry, H3Request, H3Task, H3Pipeline};

let geometry = H3Geometry::resolve(768, 1344, 124, 16, 2)?;
let request = H3Request::t2va(geometry, 32);
let layout = request.build_layout(&conditioning, [1, 2, 2])?;
let latents = pipeline.sample(&request, &layout, &conditioning, &anchors)?;
```

## Status

**Implemented and covered by tests**

- The DiT, end to end, as one compiled RLX graph: input projections, the token
  refiner, 3-axis partial RoPE, all 50 AdaLN blocks, both output heads. Runs on
  CPU with tests confirming text reaches both heads and that video and audio rows
  influence each other — the joint-generation property.
- The packed layout for **all four tasks**, including the `f64` rotary grid, the
  non-uniform `5/3 * (1, 4, 4, 4, 4)` temporal spacing, and numpy's pairwise
  summation for the `"last"` keyframe anchor.
- The scheduler, checked against the reference formulas.
- The sampling loop, with conditioning rows pinned across every step.
- **Both VAEs, in both directions**, all four paths verified on the real
  checkpoint:

  | Path | Check |
  |---|---|
  | video encode | 32x32 image → 24x2x2 latent (exact 16x compression, one latent frame) |
  | video decode | latents → pixels, finite and bounded |
  | video round trip | encode → normalize → denormalize → decode lands back in display range |
  | audio encode | 1600 samples → 32x2 latent |
  | audio decode | latents → waveform in `[-1, 1]`, non-silent |
  | audio round trip | 2400 samples in, 2400 samples out — the property that keeps A/V in sync |

- The **Qwen3-VL conditioning tap**, as a compiled graph over layers `0..50`.
  Tests confirm attention is causal, that earlier tokens reach later rows, and
  that GQA head grouping widens correctly. Tapping at layer 50 means the last 14
  layers, the final norm and the 778M-parameter `lm_head` are never loaded —
  **551 of the checkpoint's 1058 tensors**, verified against the real index.
- Parameter-key sets verified to **partition the released checkpoint exactly** —
  nothing skipped, nothing invented, no overlap between halves:

  | Component | Tensors |
  |---|---|
  | DiT | 638 / 638 |
  | Video VAE | 585 decoder + 118 encoder = 703 / 703 |
  | Audio VAE | 173 encoder + 914 decoder = 1087 / 1087 |
  | Text encoder | 551 of 1058 read (the tap skips layers 50-63, the final norm and `lm_head`) |

  This check exists because it caught a real bug: the video decoder was skipping
  `post_quant_conv` and producing perfectly finite, in-range, **wrong** pixels.
  Shape and range assertions cannot see a missing module.
- **Metal, MLX and wgpu**, all three agreeing with CPU to ~1e-7 on the DiT, the
  video decoder and the text encoder — see [Backends](#backends).
- **Temporal chunking** for full-clip decoding. `clip_length` (17) is not a
  multiple of the 4x temporal compression, so each window carries an implicit
  3-frame leading pad and consecutive windows cross-fade over 5 frames. Every
  window is the same 7 latent frames — the tail is padded by repetition — so one
  compiled graph decodes a whole clip. `chunk_geometry` round-trips the
  encoder's frame relation exactly (22, 39, 56, 124, 141, 226, 362 pixel frames
  all map back), and a full-clip decode agrees with a direct window decode on
  the body to 1e-4.
- **Tokenization** (`tokenizer` feature), checked against the shipped vocabulary:
  the chat markers cost one token each (151644 / 151645), prompts round-trip
  through decode, and padding/truncation keeps the assistant turn open.
- **Spatial tiling**, which is on by default in the released pipeline and is what
  makes real resolutions tractable: 768x1344 becomes a 4x7 grid of 256-pixel
  tiles, so each decode window is ~1.8k tokens instead of ~28k. Slack is spread
  over the overlaps in whole 16-pixel steps so every tile boundary stays
  latent-aligned, and neighbours are cross-faded on both axes. Verified on real
  weights over a 4x4 grid, and `decode-video` uses it automatically.

**Not implemented**

- **Numerical parity for the text encoder.** The graph is structurally tested but
  has never been compared against the reference — the ~60 GB encoder was not
  fetched, so a layer-50 tap could not be checked. Treat it as unverified.
- **Prompts carrying images.** The Qwen3-VL vision tower is not ported, so mRoPE's
  three axes never diverge. Text-only prompts are exact (mRoPE degenerates to
  ordinary RoPE when all three axes carry the same position); image prompts are
  rejected rather than silently conditioned on wrong angles.
- End-to-end generation on the released weights. The two DiT partitions are 28 GB
  each; only the VAEs (~10 GB) and the configs were fetched.

## Tests

```bash
cargo test -p rlx-minimax-h3                       # 179 CPU tests, no weights

# With a checkpoint (only the VAEs and configs are needed):
RLX_MINIMAX_H3=/path/to/MiniMax-H3 \
    cargo test -p rlx-minimax-h3 --release --test real_weights -- --nocapture
```

## License

The code is GPL-3.0, as the rest of this workspace. The **weights** are under the
MiniMax-H3 Community License Agreement — see the model card.
