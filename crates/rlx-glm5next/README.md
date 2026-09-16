# rlx-glm5next

**GLM-5.3-Flash** (`general.architecture = glm5next`, HF `model_type = glm5_next`)
for RLX — 320 B total / 18 B active, a hybrid linear/sparse-attention MoE.

* Weights: [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
* GGUF: [`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)

## Architecture

45 decoder layers + 1 MTP block, `hidden = 4096`, `vocab = 154880`.

| Piece | Where | Module |
|---|---|---|
| KDA — Kimi Delta Attention (gated delta-net linear attention) | 34 layers | [`kda`](src/kda.rs) |
| NoPE multi-head latent attention, `q_lora 1536` / `kv_lora 512` / head 256 | layers 3, 7, … 43 | [`mla`](src/mla.rs) |
| DSA lightning indexer with k-pool compression (`topk 2048`, `kpool 4`) | each MLA layer | [`indexer`](src/indexer.rs) |
| mHC — Manifold-Constrained Hyper-Connections, 4 residual streams | every layer, twice | [`mhc`](src/mhc.rs) |
| 288-expert MoE, 8 active + 1 shared, `noaux_tc` sigmoid gate, clamped SwiGLU | layers 3.. | [`moe`](src/moe.rs) |
| Incremental decode — latent KV cache + carried KDA conv/scan state | one token per `run()` | [`decode`](src/decode.rs) |

Three things about this model are easy to get wrong:

**There is no RoPE.** `qk_rope_head_dim = 0`, and the upstream config *rejects*
anything else. Position information reaches the sparse-attention layers only
through the KDA layers beneath them.

**DSA is exactly dense causal attention below 2048 tokens.** The indexer picks
`index_topk / index_kpool = 512` pools of 4 tokens from `floor(seq/4)` complete
pools, plus each query's own incomplete tail. For `seq <= 2048` the budget never
binds, so every visible token is selected. `Glm5NextConfig::dsa_is_dense` tests
this and the MLA layer then takes the fused `MaskKind::Causal` path instead of
materializing an `[s, s]` bias it already knows. This is an algebraic identity,
not an approximation — `dense_and_sparse_paths_agree_when_the_budget_is_not_binding`
pins it.

**mHC here is not the mHC in `rlx-motif`.** Both crates implement
Manifold-Constrained Hyper-Connections and they differ in three load-bearing
ways: unweighted vs. weighted input norm, softmax vs. sigmoid `comb`, and a
column-first vs. symmetric Sinkhorn schedule. Do not cross-port between them.

## Usage

```rust,no_run
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_glm5next::{Glm5NextConfig, build_glm5next_text_flow};
use rlx_runtime::Device;

# fn main() -> anyhow::Result<()> {
// Shard 1 of an unsloth split carries all 72 metadata keys and zero tensors,
// so the config parses from it alone.
let cfg = Glm5NextConfig::from_gguf_path("GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf")?;
assert_eq!(cfg.num_hidden_layers, 45);

// A GGUF loads through the format registry (`WeightMap::from_file` is the
// safetensors path). `_dequant_all` because this crate has no packed-matmul
// lowering yet, so the K-quants have to land as f32.
let mut loader = rlx_core::weight_loader::load_from_path(
    "GLM-5.3-Flash-UD-IQ1_S-00002-of-00003.gguf",
)?;
let mut weights = WeightMap::from_weight_loader_dequant_all(loader.as_mut())?;
let built = build_glm5next_text_flow(&cfg, &mut weights, 64, true)?;
let mut compiled = compile_built(built, Device::Cpu)?;
# Ok(())
# }
```

`Glm5NextConfig::from_hf_json` reads the upstream `config.json` instead; both
readers land on the same struct and `tests/config_parsing.rs` asserts they agree
on the published files.

## Decode

`build_glm5next_decode_flow` emits a one-token-per-`run()` graph.
`tests/decode_equivalence.rs` steps it over a prompt and requires the result to
match the prefill graph — for the whole model and, separately, for a KDA-only,
an MLA-only and a MoE layer, so a divergence localises itself.

Two things are worth knowing before using it:

**The KV cache stores the latent, not expanded keys and values.** 512 floats per
token per layer instead of 32768 — 46 MB rather than 2.9 GB at `cap = 2048`.
That means attention runs *absorbed* (query projected into latent space, output
projected back), which is algebraically identical to prefill's expanded form and
is why the equivalence test also validates both readings of GGUF's
`attn_k_b` / `attn_v_b`.

**The KDA scan state is the opposite problem.** `num_heads · head_dim²` is 4 MB
per KDA layer per token, 143 MB round-tripped through graph I/O every step
across 34 layers. `ScanState::InPlace` keeps it in a param the op mutates
instead — correct on CPU/Metal/wgpu, silently frozen on MLX and CoreML, which is
why `Portable` is the default and both modes are tested.

**Decode is only built where DSA selection is the identity** (`cap + 1 <=
index_topk`). Past that the builder returns an error rather than running dense
attention, which would be a different model from the trained one.

## Testing against real weights

The full checkpoint is 93 GB and its smallest *text* shard is 43.5 GB, but a
GGUF header carries every tensor's byte offset — so the two blocks worth testing
come out of one shard over HTTP range requests:

```sh
just glm5next-real     # fetches a 337 MB subset, then runs tests/real_weights.rs
```

`scripts/glm5next_subset.py` writes a valid single-file `glm5next` GGUF holding
`blk.0` in full (KDA + dense FFN + mHC) and `blk.3`'s attention, indexer and mHC
— the model's two layer *kinds*, everything but the 2.2 GB of routed expert
banks. `--blocks`, `--with-experts` and `--dry-run` are available.

On those real weights `tests/real_weights.rs` checks that every tensor name and
shape in the GGUF contract resolves against the published artifact (including
the two opposite `attn_k_b` / `attn_v_b` orientations), that both attention
blocks run with plausible magnitudes, that the *trained* mHC gates are
Sinkhorn-well-formed, that KDA and MLA decode both reproduce prefill (2.4e-6 and
4e-6 relative), and that the trained DSA indexer reproduces the causal mask
below its budget — exactly.

`tests/tensor_manifest.rs` is the complement and needs **no download at all**: a
52 KB fixture of the real tensor index covers the checkpoint contract for all 46
blocks — every name, every shape, and the layer schedule cross-checked against
the weights present rather than the metadata — plus a check that the emitters
consume exactly the predicted names.

## Packed weights

`common::linear` consults `WeightSource::take_packed`, so building through
`rlx_core::flow_bridge::PackedWeightLoaderSource` (see
`build_glm5next_text_flow_with_source`) makes every 2-D projection a fused
`Op::DequantMatMul` over the GGUF blob — nothing is dequantized to f32.
`tests/packed_weights.rs` measures it on the real subset and checks the two
paths agree:

```text
  KDA blk.0                   550.9 MB → 96.3 MB   (5.7×)   bit-identical
  MLA blk.3                   469.8 MB → 146.6 MB  (3.2×)   bit-identical
  MoE blk.3, 8 real experts   906.1 MB → 78.8 MB  (11.5×)   1.2e-6 relative
```

The routed banks go through `Op::DequantGroupedMatMul`, fed by
`WeightSource::take_packed_bank` (added upstream — `take_packed` describes a 2-D
linear and cannot carry an expert count). GGUF's `[experts, out, in]` is already
that op's slab layout, so unlike the F32 path there is no
`[E, N, K] → [E, K, N]` transpose for constant-folding to duplicate.

What stays f32: norms, `ssm_a`, `dt_bias`, `exp_probs_b` — f32 in the checkpoint
anyway — plus the depthwise `ssm_conv1d_*` kernels and MLA's per-head
`attn_k_b` / `attn_v_b`, which are 3-D and small.

## Status

The text architecture is complete against `modeling_glm5_next.py` and every
tensor shape in the published GGUF. Tested on synthetic weights: config parsing
from both sources, the mHC site against an f64 transcription of the reference,
the indexer's pool/tail visibility algebra, an end-to-end tiny-model run in both
DSA regimes, and decode reproducing prefill in both scan modes. Additionally
validated on **real published weights** for both layer kinds — see above.

Building this surfaced a silent wrong-answer bug in rlx-cpu's matmul dispatch —
a rank-2 lhs broadcast across a batched rhs computed only the rhs's first batch.
Fixed upstream; `tests/mm_broadcast.rs` pins the primitive here too.

Not done, in rough order of how much each blocks actually running the model:

* **No whole-model run.** Every weight class now has a packed path, so what is
  left is scale, not a missing capability: 46 blocks × 2.2 GB of routed banks
  needs the *paging* half of the story, as in `rlx_kimi_k3::moe`.
* **Decode is capped at `index_topk` tokens** — long-context decode needs a
  cached DSA indexer (per-layer indexer keys and gate scores, plus the
  reference's dynamic `first_key` walk), which is not emitted.
* **The DSA indexer assumes an unpadded batch-1 prefill**, which makes pool
  membership and causality compile-time constants. Padded batches and cached
  decode need the reference's dynamic `first_key` walk.
* **The MTP block** (`blk.45`, `nextn.*`) is parsed but not built;
  `with_mtp = true` is an error rather than a silent skip.
* **The vision tower** (`mmproj-*.gguf`, `clip.projector_type = glm5next`) is
  not ported — this is the text model only.

## How it fits

* [rlx-ling](../rlx-ling) — the other KDA + MLA hybrid; closest sibling.
* [rlx-motif](../rlx-motif) — the *other* mHC implementation (different variant).
* [rlx-deepseek](../rlx-deepseek) — `noaux_tc` MoE; the router op is shared via
  rlx-llada2's `group_limited_gate`.
* [rlx-glm4moe](../rlx-glm4moe) — the previous GLM MoE generation.
