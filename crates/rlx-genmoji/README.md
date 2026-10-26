# rlx-genmoji

[Open Genmoji](https://github.com/EvanZhouDev/open-genmoji) on RLX: Genmoji-style emoji
from a short description, fully local.

```bash
just fetch-flux1-dev                 # FLUX.1 [dev] (~36 GB)
just fetch-genmoji                   # emoji LoRAs + metaprompt -> weights/image/genmoji
just fetch-qwen35-08b-instruct       # prompt-rewrite model -> weights/lm/qwen3.5-0.8b
just fetch-upscale-esrgan            # Real-ESRGAN ×4 (BSD-3) -> weights/image/upscale
just fetch-birefnet                  # open background removal (MIT); the vendor's model is used when on macOS
just features=metal genmoji "flying pig" -- --device metal -o pig.png
just features=metal genmoji "flying pig" "rainbow popsicle" -- --device metal -o out/   # several: one load
```

## Pipeline

The author's write-up of how the vendor's Genmoji works is the design:

1. **Prompt rewrite.** The LoRA alone turns `flying pig` into a pig on a white background, not
   an emoji. A language model rewrites the description under the Open Genmoji *metaprompt*
   (plus its few-shot history) into short, explicit phrases:
   `emoji of flying pink pig. enlarged head in cartoon style. cute. white wings. head is turned
   towards viewer. 3D lighting. no cast shadows.`
   Here that model is a **local rlx LM**, Qwen3.5-0.8B by default. It is resolved through
   `rlx_core::lm_registry` from `config.json`'s `model_type`, so `--assistant-weights DIR` can
   point at any family registered in the process (this crate registers Qwen3.5 and Qwen3).
   Decoding is greedy. The model is freed before FLUX loads.

   The add-on phrases are then **assembled per subject category**, the way the vendor's
   `AugmentedPromptEmbellisher` / `PersonEmbellisher` split it. The same LM classifies the request
   as object, animal or person by scoring those three words from its next-token logits (with
   contextual calibration against a content-free subject, which removes the 0.8B model's lean
   toward "person": 14/15 correct on the test prompts, against 8/10 uncalibrated). The rewrite
   keeps its descriptive phrases, and the category supplies the rest:

   | category | add-ons |
   |---|---|
   | object | `detailed texture.` |
   | animal | `cute.` `enlarged head in cartoon style.` `head is turned towards viewer.` |
   | person | `head and shoulders portrait.` `head is turned towards viewer.` |

   All three end in `3D lighting. no cast shadows.`. Settings the rewrite invents but the user
   didn't ask for are dropped (`owl in dark forest` → `owl`, `fish swimming in the water` →
   `fish swimming`). The metaprompt forbids backgrounds, and the 0.8B model writes them anyway.
   A scene in the prompt renders a scene, and no matte turns that back into a sticker.
   `astronaut on the moon` keeps its moon. `--free-rewrite` keeps the LM's own
   add-ons instead. `--direct` skips the rewrite; `--rewrite-only` prints it and exits.
2. **FLUX.1 [dev] + emoji LoRA** ([rlx-flux1](../rlx-flux1)): `flux-dev` (platform emoji without
   skin-tone variants) or `diverse-emoji` (all of them); 160×160, 20 steps, guidance 5, seed 7.
3. **Sticker.** 160 px is the LoRA's training size, so resolution comes from a super-resolution
   network rather than a 5× Lanczos blur: Real-ESRGAN `x4plus_anime_6B` (via
   [rlx-upscale](../rlx-upscale); `--sr x4plus|animevideov3|FILE|none`) takes it to 640 px, and
   a premultiplied Lanczos resample to `--size` (800, the iOS sticker size).

   **Background removal is a learned matte**, as in the vendor's pipeline (`--matte`):
   - `auto` (the default) uses **the vendor's subject-lifting network**, read natively by
     [rlx-subject-lift](../rlx-subject-lift) from this Mac's Vision.framework. That's the model
     Genmoji's own `performBackgroundRemoval` reaches through MediaAnalysis. When it's
     absent, `auto` falls back to **BiRefNet_lite**
     ([rlx-birefnet](../rlx-birefnet), MIT, `just fetch-birefnet`).
   - The matte runs on the super-resolved image, and colours are recovered against the known
     white (`C = α·F + (1−α)·255`), so edges carry no white halo. On our emoji BiRefNet_lite
     matches the vendor's own masks at boundary F 0.9995. White wings, fur and enclosed background
     pockets come out clean, where the hand-built cut-out could not separate white from white.
   - **The white background is used as well.** Saliency mattes keep *the* salient object: a
     fireplace render kept only its flame, losing the wooden surround (Vision itself does the
     same). So when the render's border is white, every large clearly non-white region
     (`white_prior`: ≥ 24 below white in a channel, ≥ 0.2 % of the image) is added to the matte.
     Near-white pixels never count, so enclosed white background stays transparent and white
     parts of the subject still come from the learned matte. A render whose border isn't white
     (a scene) uses the learned matte alone.
   - `classic` is that hand-built cut-out: border flood fill, known-white projection, and SAM
     ViT-B (`--segment`) for enclosed pockets. `--two-pass` is an experimental exact matte
     from a second render over black.
   - `--opaque` keeps the white; `--input IMAGE` runs only this step on an existing image.

   **Framing** (`--frame FILL|none`, default 0.75) follows the vendor's `EmojiFramer`. The subject's
   bounding box is scaled so its long side is FILL of the square canvas and centred. A subject
   cut off by the render's edge stays anchored to that edge. The vendor's own stickers measure 0.67–0.81
   (median 0.75), centred within 2 %. `--strikes 40,64,96,160,320` also writes `NAME@N.png` at
   each emoji size, like the vendor's `strikeSizes`.

## Speed: `--fast` and `--serve`

The vendor's Genmoji, timed through the public `ImageCreator` API (style `emoji`) on an M4 Pro, takes
2.1–2.4 s per 768² image once its service is warm, on-device with the background already
removed. The default pipeline here takes about 2 minutes per run, most of it loading and
compiling. Two flags close most of that gap:

- `--serve` keeps every model compiled and reads one description per stdin line
  (`DESC` or `DESC<TAB>OUT.png`), printing each written path. A warm-up request compiles everything
  first.
- `--fast` (`just fetch-genmoji-fast`) stacks Hyper-SD's 8-step FLUX.1-dev LoRA (scale 0.125) on the
  emoji LoRA and runs **6 steps**, sets **T5 to 128** positions, and uses the assistant as a packed
  **Q6_K GGUF**. By eye 6 steps match 20 (some compositions improve). T5 at 128 renders the same
  images as 256, while 64 crops subjects. At 160 px a step costs the same with 64 or 128
  positions, because the matmul tiles round up. The packed assistant rewrites in ~0.8 s instead of
  ~4.5 s. Q6_K is used rather than Q8_0 because Q8_0 calls "cat astronaut" a person.

Measured per denoiser step at T5 128 (M4 Pro, isolated): 1.62 s → **1.02 s** with the
simdgroup-matrix Q8_0 GEMM in rlx-metal, against the old dequant-to-f32-scratch + MPS path.
LoRAs are merged into the weights before quantizing (`DitSource::take_packed`, one Accelerate
GEMM per projection), so the graph has no low-rank side branches: with `--fast`'s two adapters
those were ~10 % of each step. Attention (~0.22 s per step) stays on the fused kernel:
batched matmul → softmax → matmul measured slower once its output was correct. The VAE decode
and the upscaler use rlx-metal's implicit-GEMM `conv2d_sg`, which is 4–8× faster than the direct
kernel at their shapes: decode is 0.13 s, down from ~1 s.

`--t5-quant q8_0` (opt-in) also packs the T5 encoder: 2.7× less GPU time for it (~0.4 s per
request) and ~4.5 GB less memory. Its hidden states are 2.3 % off transformers' (rel error), against
1e-6 at bf16. Five of six test prompts rendered the same, but the "happy cactus" lost its face,
so `--fast` leaves it off. Rewrites reuse a snapshot of the metaprompt prefix (~880 tokens,
taken once at `--serve` warm-up), which cut rewrite time ~40 % with identical output.

A resident `--fast --direct` request takes ~8.5 s: text 0.7 s, 6 steps ~7 s, decode 0.13 s,
sticker 0.3 s. The rewrite adds ~1.5 s. Fewer steps lose prompt details: at 4 the flying pig loses its wings and the cactus its face,
and at 5 the face nearly disappears. The assistant scores categories on its prefill graph rather than
through `predict_logits`, which compiled a third graph (+2.7 GB of GPU memory; scores
identical to two decimals). Moving the LM to the CPU doesn't help: it rewrites 6× slower
there.

How the OS does it (from the symbols in this Mac's `VisualGeneration` framework):
`KeyboardEmojiGenerator.Options` carries `performBackgroundRemoval`, `strikeSizes` and a
`strikeSizesScaler` whose case is `mediaAnalysisScalingModel`, an ML upscaler run by the
MediaAnalysis service per emoji "strike" size. An `EmojiFramer` squares the subject into the
frame, and `performPromptRewrite` / `AugmentedPromptEmbellisher` is the LLM rewrite. The model
assets themselves are privacy-protected and were not read.

## Flags

`--help` lists them. The most used: `--lora flux-dev|diverse-emoji|FILE`, `--lora-scale`,
`--assistant-weights DIR`, `--assistant-device`, `--size PX`, `--sr`, `--keep-raw`, `--input`,
`--matte`, `--matte-size`, `--frame`, `--strikes`, `--free-rewrite`, `--number-format`, `--two-pass`, classic-only `--cutout-*` / `--segment`,
`--fast`, `--serve`, `--add-lora FILE[:S]` (another LoRA stacked on the emoji one), `--assistant-gguf FILE`,
plus every `rlx-flux1` model/sampling flag (`--width`, `--steps`, `--guidance`, `--seed`,
`--model-dir`, …). `--assistant-url URL` uses an already-running OpenAI-compatible server instead
of the local model.

The LoRAs and metaprompt are fetched from the upstream repository rather than vendored. It
publishes no license, so this crate ships none of its files.
