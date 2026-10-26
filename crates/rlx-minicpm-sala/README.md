# rlx-minicpm-sala

OpenBMB [MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA) in RLX: a
9B hybrid of sparse attention (InfLLM-v2, 8 of 32 layers) and Lightning linear
attention (24 layers), from the [MiniCPM](https://github.com/OpenBMB/MiniCPM)
repo (`model_type = minicpm_sala`).

```sh
just fetch-minicpm-sala                   # → weights/lm/minicpm-sala (19 GB, bf16)
just minicpm-sala -- --model-dir weights/lm/minicpm-sala --prompt "What is the capital of France?"
```

Flags: `--max-tokens`, `--temperature` (the card recommends 0.9), `--raw`
(no chat template), `--chunk N` (tokens per graph run, default 16),
`--max-seq N` (default 2048), `--f32-weights`, `--device`.

## Architecture

| piece | what it is |
|---|---|
| `minicpm4` layers | GQA attention (32 q / 2 kv heads), **no RoPE**, output × σ(`o_gate`(x)) |
| `lightning-attn` layers | simple-GLA linear attention, per-head decay `e^{-slope}` with ALiBi slopes; per-head RMSNorm on q/k, RoPE, RMSNorm over the output, × σ(`z_proj`(x)) |
| muP | embeddings × `scale_emb` (12), residual branches × `scale_depth/√layers`, LM head reads `hidden / (hidden_size / dim_model_base)` |

InfLLM-v2 is ordinary dense attention below `sparse_config.dense_len` keys
(8192) and block-sparse past it ([below](#block-sparse-attention-past-8192-keys));
a `max_seq` above `dense_len` builds the block selection into the graph.

## How it runs

One step graph takes a chunk of `T` tokens (default 16); the prompt goes
through in chunks, and a short tail is padded. The matrices stay bf16 (about
18 GB) and are streamed in one tensor at a time after compile, with the
`rlx-fish` graph builder. Weights must exist once — the constraint that matters
at 9B — so decode uses the same graph with one real row, except on Metal: there
a second, one-row graph compiled in the same sharing scope maps the weights the
first placed (`with_param_sharing_scope`), and decode steps run it. Metal
widens bf16 weights to f32 for a matmul of more than four rows, so a 16-row
decode step cost about 5 s on the 9B model; the one-row graph reads bf16 in
place.

Full model on Metal (M-series, 64 GB, machine shared with other jobs): 22
prompt + 40 new tokens in 25 s, physical footprint 25.7 GB.

The host threads two kinds of state through each run:

- the attention layers' K/V cache, `[max_seq, 2 × 128]` per layer, masked by
  an additive bias that also makes the chunk causal;
- the Lightning layers' recurrent state, `[32, 128, 128]` per layer.

Lightning attention for a chunk is evaluated in closed form, with decay
factors computed on the host (`graph::DecayFeeds`):

```text
o_i = scale·q_i · ( Σ_{j≤i} e^{g(i-j)} k_jᵀ v_j  +  e^{g(i+1)} S )
S'  = e^{g n} S + Σ_{j<n} e^{g(n-1-j)} k_jᵀ v_j
```

`graph::tests::chunk_closed_form_matches_recurrence` checks this against the
token-by-token recurrence across a chunk boundary and a padded tail.

## Parity

`just test-minicpm-sala-parity [LAYERS] [DEVICE]` runs OpenBMB's
`modeling_minicpm_sala.py` with `scripts/minicpm_sala_reference.py` (needs
`.venv-sala`: transformers 4.56.2 — the remote code no longer imports on 5.x
— torch, einops). That file imports `fla` (Triton) at module scope, so the
script installs a pure-torch stand-in transcribed from fla's
`naive_recurrent_simple_gla`.

| test | CPU | Metal |
|---|---|---|
| tiny random model, 4 layers (both mixers), chunk 1 / 5 / 32 | prefill logits ≤ 3.1e-7; greedy identical up to EOS | ≤ 3.3e-7; identical |
| real weights, first 4 layers (1 attention + 3 Lightning), f32 reference | chat-template ids identical; prefill logits 2.1e-5 (values up to 4.2); 8/8 greedy tokens identical | 2.1e-5; 8/8 identical |
| real weights, all 32 layers, f32-compute reference | chat-template ids identical; prefill logits 1.0e-4 (values up to 19.5); 16/16 greedy tokens identical | 8.6e-5; 16/16 identical |

Metal at chunk 1 first came out 0.07 off: its one-query attention kernels
(`sdpa_decode_m1` and the split-K partials) had no branch for an additive
bias mask and attended to every key, cache padding included. Fixed upstream in
`rlx-metal`: those kernels now add the bias (`[B, H, 1, Lk]`), with one-query
cases added to `attention_bias_mask_parity`. (Routing bias to `sdpa_long`
instead was correct but ran one thread per head, ~50 ms per call at 2048 keys.)


MLX, wgpu and Vulkan: tiny model ≤ 3.6e-7 at every chunk size, real weights
(first 4 layers) ≤ 2.0e-5 with 8/8 greedy tokens identical. CUDA (RTX 3080 Ti
Laptop) with `RLX_CUDA_PARITY=1` (strict f32): tiny ≤ 2.8e-7, real 1.8e-5,
8/8; with the default TF32 matmuls the real logits move by 2.0e-3 and the
greedy tokens stay identical. The full model is
over wgpu's and Vulkan's 4 GiB buffers.

A layer-subset run scales its residuals by `√layers` of the subset, as the
reference does when it is truncated the same way.

An f32 copy of the full model is 36 GB, so `--widen` keeps the reference's
weights in bf16 and widens one layer at a time to f32 for its own forward
(activations stay f32). The checkpoint is bf16, so this computes exactly what
an f32 load does: on the 4-layer subset the two references agree to 0.0.
The full reference peaks at ~39 GB.

## As a general trunk

The step graph is a Llama-family decoder whichever layers are softmax
attention, so `rlx-minicpm-v` runs the language models of MiniCPM-V 2.5 / 4.0 /
4.5 on it: `SalaConfig::from_llama_value(config, "llm.")` builds an
all-attention config (RoPE on, no muP scaling, LongRoPE short factors folded
into the frequencies; `attn_qk_norm` for Qwen3), `weight_prefix` names the
tensors, and `SalaRunner::feed_embeds` / `generate_from_embeds` take prompt
rows as embeddings so image features can be spliced in. Matrices stay in the
checkpoint's 16-bit type (bf16, or f16 for 2.5).

## Block-sparse attention past 8192 keys

Each query of a `minicpm4` layer attends to 96 blocks of 64 keys: the first
block, the 33 up to its own, and the best-scoring rest. `src/sparse.rs` has
the rule; in short, the 32-key windows (stride 16) of the keys are scored per
query head with `exp(q·ck/√d − lse)`, where `lse` runs over the 128-key windows
(stride 64) — OpenBMB's kernels normalize with the coarse windows — summed
over the 16 heads of a group in f32, rounded to bf16, max-pooled to blocks,
and ranked with ties to the lower block index. The step graph computes all of
it (compare-and-sum ranking, so every backend runs it) and turns the choice
into the attention bias; the host feeds the position-only masks.

The only implementation of this is OpenBMB's `infllm_v2` CUDA kernels.
`scripts/infllm_reference_env.sh` builds them on a CUDA machine — it picks
whichever installed toolkit compiles for the GPU with an available host
compiler and has a matching PyTorch wheel, sizes the build to the free memory,
and repairs two defects in the vendored CUTLASS header that break every build
(a lost `#` and stripped `__CUDACC_VER_*__` underscores) and the C++ standard
torch's headers need. `minicpm_sala_reference.py sparse` then runs the model's
first layers on a prompt past `dense_len` and dumps every query's chosen blocks.

On a 12,696-token prompt (2 layers, RTX 3080 Ti):

| check | result |
|---|---|
| the rule (`scripts/infllm_selection_check.py`, plain torch) vs the kernels, on the kernels' own inputs | all 25,392 (head, query) choices identical |
| the step graph vs the rule, both fed f32 activations | all 25,392 identical |
| the step graph vs the reference, end to end | 0.63 % of chosen blocks differ — the reference's activations are bf16 and its scores tie at bf16 resolution; prefill logits 3.9e-2 (values up to 5.4) |

Dropping the coarse-window normalizer, the bf16 rounding or the tie order each
changes 0.5–1 % of the choices — and a cast to bf16 and back is folded away as
a no-op, so the rounding is an explicit `RoundToFormat`.

`just test-minicpm-sala-sparse REF [DEVICE]` runs the comparison.

## Not covered

- Block selection in decode steps (one query over the keys so far) follows the
  kernels' decode path in the modeling code but is not compared: the reference
  recomputes the whole sequence per generated token, which is the prefill rule.
- ROCm is wired but not run; the full 32-layer model on CUDA does not fit the
  16 GB test card.
