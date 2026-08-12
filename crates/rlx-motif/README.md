# rlx-motif — Motif-3 for RLX

Native RLX support for [Motif-Technologies/Motif-3](https://huggingface.co/Motif-Technologies/Motif-3)
(`model_type = "Motif"`, `MotifForCausalLM`) — 53 layers, ~314 B total
parameters, 4096 hidden, 220 160 vocab, 262 144 context.

## Architecture

Three pieces have no analogue elsewhere in this workspace.

### GDLA — Grouped Differential Latent Attention

MLA-style low-rank projections crossed with differential attention. Heads come
in groups of `grouped_ratio + 1`; the last head of each group is a **noise**
head whose output is subtracted from every signal head in the group with an
input-dependent λ.

| | value |
|---|---|
| heads / KV heads / noise heads | 80 / 16 / 16 (⇒ 4 signal + 1 noise per group) |
| Q rank → head split | 1024 → 80 × (128 nope + 64 rope) |
| KV rank | 512 (+ one 64-wide RoPE head shared by all KV heads) |
| V | 16 × 128 (narrower than the 192-wide scores) |
| output gate | element-wise sigmoid, `q_lora_rank → 64 × 128` |

```
q_lat = RMSNorm(x·Wq_aᵀ)
q     = q_lat·Wq_bᵀ                      → [.,80,192]
gate  = q_lat·Wq_b_gateᵀ                 → [.,64,128]
ckv   = x·Wkv_aᵀ = [kv_lat | k_pe]
kv    = RMSNorm(kv_lat)·Wkv_bᵀ           → [.,16,128+128]
o     = softmax(q·kᵀ·scale)·v            → [.,80,128]      (GQA, 5 q-heads per KV head)
o     = o[signal] − σ(x·Wλᵀ)·o[noise ⇢ signal]
out   = (o · σ(gate))·Woᵀ
```

3 layers in 4 are 128-key sliding-window with their own `swa_rope_theta` table;
the rest are global, with YaRN-interpolated frequencies and an `mscale²` factor
folded into the softmax scale (never into cos/sin).

### MHC — Manifold-constrained Hyper-Connections

There is no single residual stream. The hidden state is `[batch, seq, 4, 4096]`
— four parallel streams — and each sublayer is wrapped in a learned mixing of
them ([arXiv:2512.24880](https://arxiv.org/abs/2512.24880)):

```
x_norm         = RMSNorm_{4·4096}(flatten(x))         (eps 1e-6, not rms_norm_eps)
h_pre  [.,4]   = σ(clamp(α_pre ·W_pre (x_norm) + b_pre , ±10))
h_post [.,4]   = σ(clamp(α_post·W_post(x_norm) + b_post, ±10))
h_res  [.,4,4] = Sinkhorn₂₀(α_res·W_res(x_norm) + b_res)      ← doubly stochastic

branch_in      = Σ_e h_pre[e]·x[e]
x'             = h_res·x + h_post ⊗ branch_out
```

Two of these per layer (`mhc_attn`, `mhc_ffn`), 106 in total. The 20 Sinkhorn
iterations are emitted inline — 4×4 is small enough that 120 tiny nodes per gate
beat a custom op that every backend would have to implement.

### PolyNorm FFN + 384-expert MoE

The activation is a trainable polynomial, with **per-expert** coefficients:

```
n(y)    = y / √(mean(y²) + 1e-6)
poly(x) = w₀·n(x³) + w₁·n(x²) + w₂·n(x) + b
out     = poly(clamp(gate)) · clamp(up) · 0.5
```

The router is sigmoid top-8 of 384 with a load-balancing selection bias,
weights normalized then × `route_scale`, plus one shared expert — the same
shape as DeepSeek's, so it reuses rlx-llada2's `group_limited_gate` kernel with
a single group. Layers 0 and 1 are dense; 2–52 are MoE.

```
embed → expand ×4
  53 × [ mhc_attn → reduce → RMSNorm → GDLA → mhc combine
         mhc_ffn  → reduce → RMSNorm → FFN  → mhc combine ]
  → mean over the 4 streams → RMSNorm → lm_head   (untied)
```

## Usage

```rust
let cfg = MotifConfig::from_file(dir.join("config.json"))?;
let mut wm = WeightMap::from_safetensors_dir(&dir)?;
rlx_motif::drop_mtp_layers(&mut wm);   // the unused speculative-decode head
prepare_checkpoint(&cfg, &mut wm)?;    // folds PolyNorm coeffs, transposes expert banks

let built = build_motif_text_flow(&cfg, &mut wm, seq, true)?;
let mut compiled = compile_built(built, Device::Cpu)?;

let (cos, sin) = cfg.rope_tables(seq);              // global layers (YaRN)
let (swa_cos, swa_sin) = cfg.swa_rope_tables(seq);  // sliding-window layers (plain)
let logits = compiled.run(&[
    ("input_ids", &ids),
    ("rope_cos", &cos), ("rope_sin", &sin),
    ("swa_rope_cos", &swa_cos), ("swa_rope_sin", &swa_sin),
]);
```

## Details worth knowing

Things a reading of the config alone would get wrong:

* **`num_key_value_heads == num_noise_heads` is load-bearing.** The differential
  regroup partitions heads into `num_noise_heads` contiguous groups, and GQA
  partitions them into `num_key_value_heads`. They have to be the same partition
  or signal heads pair with the wrong noise head. `validate()` rejects configs
  where they differ rather than guessing.
* **`repeat_kv` here is `repeat_interleave`**, not tiling: KV head `g` (and noise
  head `g`) serve the *consecutive* query heads `5g … 5g+4`.
* **The two PolyNorm variants differ.** `PolyNormTorch` (dense MLP, shared
  expert) clamps only `gate` and `up`; `GroupedPolyNorm` (routed experts) also
  clamps the bias and the product. Both apply `polynorm_output_scale` last.
* **`σ(weight)` and the bias clamp are folded host-side** into one `act_fn.coeff`
  row by `prepare_checkpoint`. That is what lets the graph `Gather` per-expert
  coefficients by routed expert id — upstream has to fall back to an eager
  Python loop over all 384 experts for exactly this reason.
* **`sliding_window` is off by one between the two conventions.** The reference
  passes `sliding_window + 1` to the flash interface, which becomes
  `window_size = (w−1, 0)`; `MaskKind::SlidingWindow(w)` keeps `q−k ≤ w`. Net:
  pass `config.sliding_window` verbatim.
* **MHC's RMSNorm epsilon is hardcoded `1e-6`**, not `config.rms_norm_eps`.
* **The YaRN `mscale` never touches cos/sin.** `attention_factor` is pinned to
  1.0 upstream; `mscale²` goes on the softmax scale of global layers only.
* **`model.mtp_layers.0` is dead.** The checkpoint ships it
  (`num_nextn_predict_layers = 1`) but `modeling_motif.py` never instantiates it.
* Config keys that are read and then never used: `max_window_layers`, `k_ratio`,
  `headwise_attn_output_gate`, `attention_dropout`.

## Status

Architecture complete and checked against `modeling_motif.py`,
`configuration_motif.py` and every tensor name/shape in
`model.safetensors.index.json`.

```bash
cargo test -p rlx-motif
RLX_TEST_DEVICE=metal cargo test -p rlx-motif --features metal   # …mlx/gpu/coreml/cuda/rocm/vulkan
```

All 30 tests on all 7 backends (+ CoreML), across three hosts:

| backend | host | result |
|---|---|---|
| CPU | mac / RTX 3080 Ti / MI100 | 30/30 |
| Metal | mac | 30/30 |
| MLX | mac | 30/30 |
| CoreML | mac | 30/30 |
| CUDA | RTX 3080 Ti | 30/30 |
| Vulkan | RTX 3080 Ti / MI100 | 30/30 |
| ROCm | MI100 | 30/30 |
| wgpu | mac (Metal adapter) | 30/30 |
| wgpu | Linux (Vulkan adapter) | 30/30 **with `RLX_ARENA_NO_REUSE=1`**; 7 fail without |

The Linux-wgpu caveat is not this model: `rlx-wgpu` corrupts slot-reused buffers
on Vulkan adapters (the same bug that costs llama ~1.9e-2), and the flag clears
it on both an RTX 3080 Ti and an MI100 — the failures there are O(1)
(`gdla` reads max |Δ| 1.2 against a host reference the other backends match to
4e-7), not a tolerance question. macOS wgpu, which drives Metal, is unaffected.

One tolerance note: the differential recombination subtracts two nearly-equal
attention outputs, so `gdla_reference` is cancellation-prone by construction.
Every backend but ROCm agrees with the host reference to ≤4e-7; ROCm's
accumulation order puts it at ~1e-4, which is why that test allows 5e-4.

| test | what it pins |
|---|---|
| `config` unit tests | published config parses; MoE/SWA schedules match the checkpoint's 2 dense + 51 MoE and 14 global layers; YaRN touches only low frequencies |
| `polynorm_reference` | both PolyNorm variants vs a host reference, including the per-token coefficient gather and a clamp tight enough to bite |
| `mhc_reference` | gates vs host reference, `h_res` doubly stochastic, `apply_h_pre` and the `einsum` combine |
| `gdla_reference` | full attention block vs a host reference, global and sliding-window |
| `moe_block_reference` | router + experts + shared expert vs a host reference, fed through `prepare_checkpoint` |
| `text_flow_smoke` | whole graph compiles and runs; prefill is causal; MHC-off, SWA-off and dense-only variants build |

**A real-weight run has not been done.** The checkpoint is 629 GB of bf16 across
155 shards and `WeightMap` is f32, so the expert banks alone would be ~2.5 TB
resident. Real weights need the paged/packed expert path (as in
`rlx_kimi_k3::moe`), which is not wired here.
