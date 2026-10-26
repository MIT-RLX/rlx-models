# rlx-flux1

**FLUX.1 `[dev]` / `[schnell]`** (Black Forest Labs) text-to-image for RLX, all native: the
T5-XXL and CLIP-L text encoders, the 12B MMDiT rectified-flow denoiser, the 16-channel VAE
decoder, and LoRA adapters. Runs on `cpu`, `metal`, `mlx`, `cuda`, `rocm`, `gpu` (wgpu), `vulkan`.

## Quick start

```bash
just fetch-flux1-dev            # ~36 GB from ungated mirrors -> weights/image/flux1-dev
just features=metal flux1 -- --model-dir weights/image/flux1-dev \
  --prompt "a red fox in the snow" --width 1024 --height 1024 --steps 28 --guidance 3.5 \
  --device metal -o fox.png
# with a LoRA (repeatable; PATH[:SCALE]):
#   --lora my_lora.safetensors:0.8
```

`--help` lists every flag (`--seed`, `--noise FILE`, `--t5-len`, graph chunking options).
`RLX_FLUX1_VERBOSE=1` prints per-stage and per-step timings.

## Weights

| component | file | source (ungated) |
|---|---|---|
| denoiser | `flux1-dev.safetensors` (BFL single-file, bf16) | `Comfy-Org/flux1-dev` |
| T5-XXL encoder | `t5/` (HF `T5EncoderModel`, bf16) + `tokenizer.json` | `city96/t5-v1_1-xxl-encoder-bf16`, tokenizer from `google/flan-t5-xxl` (same vocabulary) |
| CLIP-L | `clip/` (full CLIP or text-only) | `openai/clip-vit-large-patch14` |
| VAE | `vae/ae.safetensors` (BFL/LDM names; a diffusers `vae/` dir also works) | `ffxvs/vae-flux` |

`[schnell]` is detected from the checkpoint (no `guidance_in`): no schedule shift, T5 length 256.
Every dimension is read from tensor shapes (`Flux1Config::from_shapes`).

## Memory

The denoiser is 12B parameters. Weights stay at their checkpoint width on the device
(`WeightStorage::Auto`: bf16 -> ~24 GB), and both the denoiser and T5 are built as a **chain of
graph chunks** (`--doubles-per-graph`, `--singles-per-graph`, `--t5-layers-per-graph`), so host
memory never holds more than one chunk at f32. One graph would hold all 48 GB of f32 parameters
before compiling. The text encoders run first and are freed before the denoiser loads
(`--keep-text-encoders` keeps them for repeated prompts).

## How it maps to the reference

- **RoPE**: FLUX rotates interleaved pairs; the rlx kernels rotate half-split. The q/k projection
  rows (and QK-norm scales) of each head are permuted at load (`weights::deinterleave_rows`), which
  leaves `q·k` unchanged, so the graph uses the standard kernel.
- **LoRA** (`lora.rs`): diffusers/PEFT, BFL and kohya key names. diffusers splits projections BFL
  fuses (`to_q/to_k/to_v` -> `img_attn.qkv`, single-block `to_q/k/v/proj_mlp` -> `linear1`), so an
  adapter becomes a row band of the fused matrix; `norm_out` swaps its `(scale, shift)` halves.
  Adapters run **unmerged** as a low-rank graph branch `y += (x Aᵀ) Bᵀ`: merging into a bf16 matrix
  and storing the sum at bf16 measured 6.6e-4 relative error, unmerged 1.4e-6.
- **Conditioning MLPs** (`time_in`, `guidance_in`, `vector_in`) run on the host. The per-block
  modulation linears (a quarter of the parameters) stay in the graph.
- **T5**: no attention mask (FLUX pads to the full length and attends to the padding), unscaled
  scores plus the bucketed relative-position bias via `MaskKind::Bias`.
- **VAE**: the rlx-flux2 decoder, with LDM names renamed to diffusers' (up blocks reversed).

## Number formats

`--number-format f32|bf16|f16|f8|f6|f4|fNeXmY|weights/activations` (the repo-wide
`rlx_ir::NumberFormat`: weights rounded on upload, every intermediate in-graph, accumulation in
f32). `examples/format_report.rs` measures each component against the f32 run:

```bash
cargo run -p rlx-flux1 --release --features metal --example format_report -- \
  --device metal --formats f32,bf16,f16,f8 --lora weights/image/genmoji/lora/flux-dev.safetensors
```

On top of the generic rules (below 8 bits the residual stream, norms and RoPE tables stay f32),
`rlx_flux1::precision` applies a measured per-component policy:

| component | policy | f16 | bf16 | f8 |
|---|---|---|---|---|
| T5-XXL | residual stream + matmul outputs in f32 | 97 % → **0.22 %** | 2.8 % → 1.2 % | 29 % → 19 % |
| denoiser (4+8 blocks) | plain `f8` = e2m5 weights / e3m4 activations | 0.088 % | 0.67 % | 6.7 % → **3.6 %** |
| CLIP / VAE | generic | 0.15 % / 0.06 % | 1.4 % / 0.49 % | 12 % / 5.7 % |

(relative L2 error vs f32, Metal.) T5 v1.1's activations exceed f16's range, which is why HF keeps
its `wo` in f32. Both its residual stream and its matmul outputs overflow, so both stay f32.
In the denoiser, 16-element block scales leave the f8 exponent bits nearly idle, so mantissa
buys more. Weights alone at f8: e4m3 2.5 %, e3m4 1.3 %, **e2m5 0.72 %**. An explicit
spelling (`f8e4m3`, `f8/f16`, …) or a user-set `RLX_NUMBER_FORMAT_KEEP*` /
`_ROUND_RESIDUAL` bypasses the policy. Weight-only f8 (`f8e2m5/f16`) is the most accurate
reduced setting for the denoiser.

## Parity

`scripts/flux1_reference.py` (diffusers / transformers) dumps references; `tests/reference_parity.rs`
compares (env-gated: `RLX_FLUX1_REF`, `RLX_FLUX1_DIR`, `RLX_FLUX1_DEVICE`):

```bash
python3 scripts/flux1_reference.py text --prompt "…" --out REF/text
python3 scripts/flux1_reference.py dit --lora weights/image/genmoji/lora/flux-dev.safetensors --out REF/dit_lora
python3 scripts/flux1_reference.py vae --out REF/vae
just test-flux1-parity REF cpu      # or: just features=metal test-flux1-parity REF metal
```

| component | CPU | Metal |
|---|---|---|
| T5 / CLIP token ids | exact | exact |
| CLIP pooled | 1.3e-6 | 2.0e-6 |
| T5-XXL hidden (512 tokens) | 1.3e-5 | 5.6e-6 |
| denoiser, 2 double + 2 single + final, with LoRA | 1.4e-6 | 1.9e-6 |
| VAE decode | 1.1e-6 | 8.1e-7 |

(relative L2 error vs f32 reference.)

`e2e` runs the complete diffusers `FluxPipeline` (bf16, MPS) on explicit noise for an image-level
comparison (`--noise FILE` feeds the same noise to `rlx-flux1`).
