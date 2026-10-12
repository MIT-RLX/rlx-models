# Qwen-Image 2.1 — port reconnaissance

Not implemented. This is step 1–3 of [adding-a-model.md](adding-a-model.md)
done and written down, so the port starts from facts instead of discovery.

Source inspected:
[`abenzerps/Qwen-Image-2.1-Uncensored-GGUF`](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF)
(`qwen-image-2.1-UC-Q4_K_M.gguf`), read via HTTP range — header only, no
download.

## What the GGUF actually contains

```
general.architecture = qwen_image21        297 tensors, 3 metadata keys
```

Three metadata keys total: architecture, file type, quantization version.
**Every hyper-parameter has to be inferred from tensor shapes** — there is no
`block_count`, no `embedding_length`, nothing. That is unusual here and is the
first thing the config reader has to cope with.

32 transformer blocks, hidden 4096. Nine tensors per block:

| pattern | count | note |
|---|---|---|
| `transformer_blocks.N.attn.{to_q,to_k,to_v}.weight` | 32 each | |
| `transformer_blocks.N.attn.to_out.0.weight` | 32 | note the `.0.` — a `nn.Sequential` |
| `transformer_blocks.N.attn.{norm_q,norm_k}.weight` | 32 each | QK-norm |
| `transformer_blocks.N.img_mlp.{gate_layer,proj,out}.weight` | 32 each | gated MLP |

Outside the blocks, eight tensors:

```
img_in.weight                                     [64, 4096]     patch in
txt_in.in_layer.weight                            [4096, 4096]   text refiner
txt_in.out_layer.weight                           [4096, 4096]
txt_in.text_norm.weight                           [4096]
time_text_embed.timestep_embedder.linear_1.weight [256, 4096]    timestep embed
time_text_embed.timestep_embedder.linear_2.weight [4096, 4096]
norm_out.linear.weight                            [4096, 4096]   output head
proj_out.weight                                   [4096, 64]
modulation.1.weight                               (single tensor)
```

## Three findings that shape the port

1. **It is a single-stream DiT, not dual-stream.** There is `img_mlp` and no
   `txt_mlp`, and one attention block per layer rather than separate image and
   text towers. Text enters through `txt_in` and is concatenated, which is a
   different graph from FLUX's double-stream blocks — do not assume
   `rlx-flux2`'s block shape carries over.
2. **Modulation is global, not per block.** `modulation.N.weight` matches
   exactly once across 297 tensors. A port that emits per-block adaLN
   modulation (the usual MMDiT arrangement) will look for 32 tensors that do
   not exist.
3. **`to_out` is `.0.`-suffixed**, i.e. the reference wraps it in a
   `nn.Sequential` whose second element is a dropout. Harmless, but the name
   has to match.

## What the GGUF does *not* contain

This file is the **denoiser only**. A working text-to-image pipeline needs
three more pieces:

| piece | where it is | status here |
|---|---|---|
| text encoder | `text_encoders/qwen3vl_8b_bf16.safetensors` in the same repo | **exists** — `rlx-qwen3-vl` |
| VAE decoder | not in this repo; ComfyUI users supply `qwen_image_vae.safetensors` separately | **missing entirely** |
| flow-matching scheduler + CFG | — | pattern exists in `rlx-flux2` (rectified flow, CFG, img2img) |

So the port is: a new denoiser crate, a VAE decoder (sourced separately), the
scheduler, and wiring `rlx-qwen3-vl` as the conditioner. The denoiser is the
tractable part; the VAE is the piece with no starting point in this repo.

## Suggested order

1. Config reader that **infers** from shapes (no metadata to read): block count
   from the `transformer_blocks.N.` maximum, hidden from `img_in`'s second dim,
   patch channels from its first.
2. Tensor-manifest test against the 297 real names, both directions — the
   manifest is in this document's table and takes a header read to regenerate.
3. One block, then the stack, against the reference `QwenImageTransformer2DModel`.
4. Scheduler + CFG, reusing `rlx-flux2`'s rectified-flow loop.
5. VAE last, and only once a latent can be produced to decode.

Reference: `diffusers`' `QwenImageTransformer2DModel` and
`QwenImagePipeline`. llama.cpp does not implement this architecture, so the
GGUF's converter conventions come from the ComfyUI-GGUF tooling that produced
it rather than from `conversion/<arch>.py`.
