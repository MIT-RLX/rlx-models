# rlx-flux3

FLUX 3 Image Klein ([black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base),
and the distilled [flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein))
for RLX: text-to-image and editing with up to 10 reference images, plus the
prompt upsampler [flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup).

Everything runs natively:

| Component | Model | Here |
|-----------|-------|------|
| Text encoder | Qwen3-VL-4B-Instruct text tower, hidden states 4, 8, …, 32 stacked → 20480-wide tokens | [`text`](src/text.rs) |
| Denoiser | 5.5B DiT: 5 per-mode blocks (image / reference / text, each attending only within its mode), then 28 shared single-stream blocks | [`dit`](src/dit.rs) |
| Autoencoder | FLUX.2 VAE (32 ch, ×8) + 2×2 patchify + latent batch-norm = ×16 | [`vae`](src/vae.rs), graphs from `rlx-flux2` |
| Upsampler | Qwen3.5-9B fine-tune, greedy + presence penalty | [`pup`](src/pup.rs), runs on `rlx-qwen35` |

The two DiT checkpoints are told apart by their tensors: **base** (CFG,
50 steps) and the distilled **flow map** (`ref_time_in`, 16 steps, no CFG).

## Run

```bash
just fetch-flux3-klein            # ~20 GB → weights/image/flux3-klein-base
just fetch-flux3-pup              # 5.6 GB Q4_K_M + 0.9 GB vision tower → weights/lm/flux3-klein-pup
just features=metal flux3 -- --model-dir weights/image/flux3-klein-base \
    --pup weights/lm/flux3-klein-pup --prompt "a red fox asleep in fresh snow" \
    --device metal -o fox.png
```

The model takes full scene captions, not short prompts. `--pup` rewrites the
prompt first (as BFL's CLI does by default); without it the prompt goes to the
DiT as typed, which is right only for a caption another LLM already wrote.
`--caption-out FILE` keeps the caption, `--upsample-only` stops there. Edit
with `--image ref.png` (repeatable, up to 10): the upsampler sees the
references too, a single reference is padded to the 16-px grid and the output
cropped back, as in the reference.

Sampler defaults come from the package's `config.json` (base: 50 steps,
guidance 3.5, schedule shift `alpha` 6.93). BFL's Python code hard-codes
CFG 4 and alpha 16 instead; pass `--guidance` / `--alpha` to use those.

## Parity

Against BFL's own code ([flux-3-image-klein](https://github.com/black-forest-labs/flux-3-image-klein)),
in f32 on CPU from the bf16 weights:

```bash
git clone https://github.com/black-forest-labs/flux-3-image-klein /tmp/flux3-ref
python scripts/flux3_reference.py --bfl-repo /tmp/flux3-ref \
    --model-dir weights/image/flux3-klein-base --out /tmp/flux3-dumps \
    --caption "…" --size 256 --steps 4
just test-flux3-parity /tmp/flux3-dumps [DEVICE]
```

Add `--flow-map-dir weights/image/flux3-klein` (and `--skip-base` to reuse a
`text` dump) for the distilled checkpoint's case, and set
`RLX_FLUX3_FLOW_MAP_DIR` for its test. End to end, BFL's published `examples/base_vs_distilled/knitted_base.png`
(1184×896, seed 1, 50 steps, CFG 3.5, made with their bf16 CUDA code) is
reproduced at **PSNR 38 dB / pixel correlation 0.999** by
`rlx-flux3 --device mlx` with the same caption and seed — which also confirms
the package's sampler defaults are the ones behind BFL's examples.

The tests check token ids, text tokens (padding rows included), the
`torch.randn` noise and f32 schedule bit for bit, the first-step velocity of
both CFG samples, a 4-step CFG run, the decoder, the encoder, and the
first-step velocity of an edit.

## Speed

Attention dominates (~4.8k tokens per sample through 33 blocks at
1184×896). Measured on a shared M4 Pro under load, CFG step at 1184×896:
**MLX ~20 s**, **Metal ~48 s** (was ~90 s before rlx-metal's
`sdpa_prefill_fa2` kernel: 32 query rows per threadgroup with Q and O in
simdgroup-matrix registers, 2.6-3.3× the old prefill attention at this shape;
`RLX_METAL_PREFILL_FA2=0` opts out). All four parity tests pass on CPU, Metal
and MLX. `--weight-quant q8_0` did not help — weights are not the bottleneck.

The autoencoder's last level at ~1 MP is a 2.4 G-element im2col; rlx-mlx used
to hand such convs to the host's naive kernel (tens of minutes), and now runs
them on MLX in bands of output rows.

## Notes for whoever changes this

- **Padding tokens reach the DiT.** The text is right-padded to a multiple of
  80 and every position, padding included, becomes a DiT token. Under the
  reference's causal + padding mask a pad query sees all real keys and no pad
  key — not even its own — and every pad has the same id and position (1), so
  all pad rows are identical. The encoder runs one extra query row attending
  to the real keys with no mask, and copies it to every pad position.
- **CFG is one batch of two.** The empty and the real prompt pad to different
  lengths; the shorter is zero-filled and a key mask removes the filler, which
  is exact because the samples never attend to each other in the reference
  either.
- **Text and references are computed once.** Their tokens sit at flow time 0
  and their early blocks never see the image, so for the base checkpoint the
  early text/reference output is per prompt, not per step. A flow map's
  `ref_time_in` moves their modulation too, so there it is per step.
- **bf16 casts the reference makes on purpose are kept**: the timesteps are a
  bf16 tensor (so `1000·t` is rounded to a multiple of 4 near t = 1), the
  timestep embedding is cast back to bf16, text and reference tokens are bf16,
  and the noise is drawn in f32 and rounded to bf16.
- **RoPE is interleaved** (pairs `2i, 2i+1`); the q/k projection rows and QK
  norm scales are permuted per head at load so the graph uses the half-split
  RoPE op.
- The edit upsampler (`i2i`) runs the Qwen3.5 vision tower (`mmproj-*.gguf`
  next to the weights) on the references after BFL's `policy_reference`
  preparation (32-px grid, per-count pixel budget, Lanczos, JPEG q92 round
  trip), with the pixel bounds from the checkpoint's `preprocessor_config.json`;
  several references go into one prompt through `rlx_qwen35::MultiImagePrompt`.
  That preparation matches PIL's only approximately (the `image` crate's
  Lanczos3 and JPEG encoder).
- **Boxes** (`--boxes FILE`, or a prompt ending in a 0-1000 table) follow
  BFL `flux3/bbox.py` ([`bbox`](src/bbox.rs)): layout system prompts, locked
  rows re-imposed on the upsampler's answer, a finished caption sent as is.
  The table is text the DiT reads, so it is written exactly as Python's
  `json.dumps` writes it (key order, separators, float repr) — `serde_json`
  sorts keys here, hence the crate's own ordered `Json`. Checked string for
  string against the Python module on 233 cases
  (`scripts/flux3_bbox_cases.py` → `tests/data/bbox_cases.json`).
