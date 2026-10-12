# rlx-qwen4exp

**Qwen3.8-Flash-Next** (`general.architecture = qwen4exp`, HF
`model_type = qwen4_exp`) for RLX — the first checkpoint of the Qwen4
architecture. Its own GGUF metadata describes it as *"A Preview of the Qwen4
Architecture"*, size label `512x56B`.

* Weights: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
* GGUF: [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
  [`ggml-org/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF)

## Architecture

48 decoder layers (+ 1 MTP block, shipped as its own `mtp-*.gguf`),
`hidden = 2560`, `vocab = 248320`, `head_dim = 256`.

| Piece | Where | Module |
|---|---|---|
| Hyper-connections — 4 parallel residual streams, low-rank input mixer | every layer twice, plus the output | [`hc`](src/hc.rs) |
| GatedDeltaNet linear attention (16 k-heads / 48 v-heads × 128) | 36 layers | [`gdn`](src/gdn.rs) |
| Gated GQA attention, 24 q / 2 kv heads × 256, partial M-RoPE over 64 dims | layers 3, 7, … 47 | [`attention`](src/attention.rs) |
| QSA block indexer — top-512 of 4-key mean-pooled blocks + each query's tail | each attention layer | [`indexer`](src/indexer.rs) |
| 512-expert MoE, 10 active + 1 gated shared expert, softmax top-k | every layer | [`moe`](src/moe.rs) |
| PLE — hashed n-gram embeddings, 51 B parameters | layer 1 only | [`ple`](src/ple.rs) |
| Incremental decode and chunked prefill — KV cache + carried GatedDeltaNet / PLE state | one token or one chunk per `run()` | [`decode`](src/decode.rs) |
| The ready-to-run loop — compile once, chunked prefill into one-token decode, `LmRunner` | GGUF in, logits out | [`runner`](src/runner.rs) |
| The n-gram table read lazily — only the quantization blocks a row overlaps | 28 GB on disk, 9 rows per step | `GgufPleRows` in [`ple`](src/ple.rs) |
| Experts paged off disk — route first, then the fired slices into their arena slots; quantized banks dequantize per expert | 512 per layer, 10 fire | [`paged_moe`](src/paged_moe.rs) |
| MTP — the multi-token-prediction draft block | `blk.48`, its own `mtp-*.gguf` | [`mtp`](src/mtp.rs) |
| The vision tower — a Qwen3-VL ViT: dual patch conv, axial 2-D RoPE, merge-block projector | `mmproj-*.gguf` | [`vit`](src/vit/) |
| The multimodal seam — `inputs_embeds` splice, PLE image-token substitution, 3-D M-RoPE | host-side | [`vision`](src/vision.rs) |

Five things about this model are easy to get wrong.

**There is no `output_norm`, and no `attn_norm` / `ffn_norm` either.**
Hyper-connections replace every layer norm in the architecture. The final
mixer's grouped RMSNorm (`output_hc_norm`) *is* the output norm, and the
checkpoint carries no separate tensor for it — `tests/tensor_manifest.rs`
asserts that against the real 1224-tensor list.

**`per_layer_token_embd` is bigger than the rest of the model.**
`[160, 320001536]` — 51 billion weights, ~28 GB packed. It is addressed by a
*hash of the trailing token window*, not by token id, so this crate takes the
embeddings as a graph input and gathers rows on the host
([`prepare_ple_embeddings`](src/ple.rs)). llama.cpp reaches the same arrangement
from the other side, marking the tensor `TENSOR_READ_LAZY`.

The hash constants — three 45-bit multipliers, and a distinct prime modulus plus
table offset for each of the 16 hash heads — are *derived* from
`(seed, vocab_size)` and then checked against the values the checkpoint also
publishes. They agree exactly, including the padded row count `320001536`;
`Qwen4ExpConfig::validate` refuses a checkpoint where they do not, because the
failure mode is reading the wrong rows of a 320-million-row table with nothing
else going wrong.

**The GatedDeltaNet GQA expansion tiles; the attention GQA expansion
interleaves.** The reference does `repeat_interleave` for the linear layers,
because HF stores the 48 value heads grouped by their 16 key heads. The GGUF
converter permutes them to tiled order — `_LinearAttentionVReorderBase`, applied
to `attn_qkv`'s v rows, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`,
`ssm_dt.bias`, `ssm_conv1d`'s v channels and `ssm_out`'s columns — so a
whole-block repeat is correct and cheap. The gated-attention layers were *not*
permuted, so they keep the interleaved order. Getting the linear one backwards
agrees only for v-heads 0 and 47 and flips the sign of every head in between.

**QSA is exactly dense causal attention below 2048 tokens.** The indexer picks
`indexer_budget / indexer_compress_ratio = 512` blocks of 4 keys out of
`floor(seq/4)` complete blocks, plus each query's own incomplete tail. For
`seq / 4 <= 512` the budget never binds, so every visible block is selected, and
blocks ∪ tail is exactly `0..=q`. `Qwen4ExpConfig::qsa_is_dense` tests this and
the attention layer then takes the fused `MaskKind::Causal` path instead of
materializing an `[s, s]` bias it already knows. This is an algebraic identity,
not an approximation — `qsa_selection_is_the_identity_below_the_budget` pins it
*numerically, on real weights*, by running the full selection machinery
(`force_emit`) against the short-circuit.

**The norm gains are pre-offset, and `ssm_a` is pre-negated.** Every
`Qwen4ExpTextRMSNorm` computes `x̂ · (1 + w)`; the converter folds the `+1` into
the stored tensor, so a GGUF loads it verbatim and a HF checkpoint needs the
shift (`Qwen4ExpConfig::rms_norm_offset`). The gated norm (`ssm_norm`) is the
exception — a plain `w` in both. And `ssm_a` is already `-exp(A_log)`: applying
another `-exp()` gives a ~24× too-aggressive decay, which collapses the
recurrent state in one step and turns the layer into a memoryless prompt echo.
`converter_conventions_hold_on_the_real_tensors` checks both against the real
tensors, because both are silent when wrong.

## Usage

### Running it

```rust,no_run
use rlx_qwen4exp::runner::{Qwen4ExpRunner, RunnerOptions};
use rlx_runtime::Device;

# fn main() -> anyhow::Result<()> {
// One compile, reused for every token. `prefill_chunk` above 1 adds a second
// graph that advances several positions per `run()` — faster prompts, at the
// cost of a second resident copy of the weights.
let mut run = Qwen4ExpRunner::open(
    "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf",
    RunnerOptions::new(Device::Metal).prefill_chunk(64),
)?;
let out = run.generate_greedy(&[9707, 11], 32, |t| {
    print!("{t} ");
    true
})?;
# let _ = out;
# Ok(())
# }
```

`Qwen4ExpRunner` implements `rlx_runtime::LmRunner`, so `prefill_logits` +
`decode_logits` give a host-driven loop where the caller owns sampling. It works
on token ids — there is no tokenizer here.

`open` attaches a `GgufPleRows` over the model file when it carries
`per_layer_token_embd.weight`, which is how the 51-billion-parameter n-gram
table is read without holding it: a row costs the quantization blocks it
overlaps, and for the published IQ4_NL table that is exactly 5 blocks of 18
bytes — **90 bytes per row**, against a 28 GB tensor. The published checkpoint
is split, so point `RunnerOptions::ple_table` at the shard that holds it. With
no table reachable the runner refuses rather than reading zeros.

### Building the graphs directly

```rust,no_run
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_qwen4exp::{Qwen4ExpConfig, build_qwen4exp_text_flow};
use rlx_runtime::Device;

# fn main() -> anyhow::Result<()> {
// Shard 1 of an unsloth split carries all 67 metadata keys and zero tensors,
// so the config parses from it alone.
let cfg = Qwen4ExpConfig::from_gguf_path("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf")?;
assert_eq!(cfg.num_hidden_layers, 48);
assert_eq!(cfg.ple_layers, vec![1]);

// A GGUF loads through the format registry (`WeightMap::from_file` is the
// safetensors path). For the packed path — every projection a fused
// `DequantMatMul`, every expert bank a `DequantGroupedMatMul` — use
// `build_qwen4exp_text_flow_with_source` with
// `rlx_core::flow_bridge::PackedWeightLoaderSource` instead.
let mut loader = rlx_core::weight_loader::load_from_path(
    "Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf",
)?;
let mut weights = WeightMap::from_weight_loader_dequant_all(loader.as_mut())?;

let built = build_qwen4exp_text_flow(&cfg, &mut weights, 64, true)?;
let mut compiled = compile_built(built, Device::Cpu)?;
let logits = compiled.run(&[("input_ids", &[1.0f32; 64][..])]);
# let _ = logits;
# Ok(())
# }
```

### Decode and chunked prefill

One builder serves both: `chunk = 1` is decode, `chunk > 1` is chunked prefill.
`DecodeSession` threads the state forward.

```rust,no_run
use rlx_qwen4exp::decode::{build_qwen4exp_step_flow, DecodeNames, DecodeSession, ScanState};
use rlx_qwen4exp::{OutputKind, Qwen4ExpConfig};
# fn main() -> anyhow::Result<()> {
# let cfg = Qwen4ExpConfig::qwen38_flash_next();
# let mut weights = rlx_core::weight_map::WeightMap::from_tensors(Default::default());
# let mut src = rlx_core::flow_util::WeightMapSource(&mut weights);
let (built, layout) = build_qwen4exp_step_flow(
    &cfg, &mut src, /* cap */ 2048, /* chunk */ 1,
    OutputKind::Logits, ScanState::Portable,
)?;
let names = DecodeNames::new(&cfg);
let mut session = DecodeSession::new(&cfg, layout, 2048);
// each step: bind `input_ids`, `rope_cos`/`rope_sin`, `ple_embeddings.0`,
// then `session.inputs(&cfg, &names)`; `session.commit(&packed)` returns the
// logits and stores the new state. `session.push_token(tok)` keeps the PLE
// hash window sliding.
# let _ = (built, session.state_bytes());
# Ok(())
# }
```

Per-token state traffic is dominated by one thing — the GatedDeltaNet scan
matrix is `48 · 128²` floats *per linear layer*, 113 MB across 36 layers.
`ScanState::InPlace` keeps it in a device param instead (CPU/Metal/wgpu only;
MLX and CoreML substitute the new state into a per-evaluation env, where it
would silently freeze at zero). `DecodeSession::state_bytes` reports the total.

A short final chunk is right-padded: `set_real_len` drives a `pad_mask` and a
`conv_shift` whose only job is to stop the *exported* state advancing over the
padding — the per-token outputs never need them, because the convolution and
the scan are both causal.

### Speculative decoding

The MTP block reads the trunk's **hyper-connection residual**, not its hidden
state. Build the trunk once at `OutputKind::HcResidual`, then turn that one
residual into both the trunk's logits and the draft's:

```text
   trunk(HcResidual) ──┬── build_qwen4exp_head_flow → trunk logits
                       └── build_qwen4exp_mtp_flow  → draft logits (+ next residual)
```

`the_residual_seam_reproduces_the_trunk_logits` pins that splitting the trunk
there does not change its logits — otherwise every acceptance decision would be
made against a model the trunk is not.

### Multimodal (feature `vision`)

```rust,no_run
use rlx_qwen4exp::vision::{ImageSpan, VisionEncoder, assemble_multimodal};
# fn main() -> anyhow::Result<()> {
# let cfg = rlx_qwen4exp::Qwen4ExpConfig::qwen38_flash_next();
# let (embed_table, rgb, w, h): (Vec<f32>, Vec<u8>, usize, usize) = (vec![], vec![], 0, 0);
# let ids: Vec<u32> = vec![];
let mut encoder = VisionEncoder::open(&cfg, "mmproj-F16.gguf", rlx_runtime::Device::Cpu)?;
let out = encoder.encode_rgb(&rgb, w, h)?;          // `[n_tokens, 2560]`

// `ids` holds `out.n_tokens` `<|image_pad|>` placeholders starting at index 1.
let batch = assemble_multimodal(&cfg, &ids, &embed_table, &[ImageSpan::from_encode(1, &out)?])?;
// → batch.inputs_embeds, batch.ple_token_ids, batch.section_positions
# Ok(())
# }
```

The tower ([`vit`](src/vit/)) is this crate's own: a `ModelFlow` over a
`WeightSource`, so a quantized `mmproj-Q8_0.gguf` runs **packed** and no weight
is transposed on the host — the GGUF already reports `[out, in]`, and
`v.patch_embd.weight` already arrives as `Op::Conv`'s `[out_c, in_c, kh, kw]`.
`VisionEncoder` compiles one graph per distinct smart-resized size and reuses it.

Six things carry the tower, each with a plausible wrong answer: the patch
embedding is **two** convolutions summed (a 2-frame temporal window, a still
image replicated across both); tokens are permuted into spatial-merge blocks
*before* the transformer, which is what makes the merger's reshape a reshape —
so the position table must be permuted the same way and the RoPE coordinates
generated in that order; the learned position table is **bilinearly resized**
from its 48² grid; the axial RoPE puts `[row | column]` frequencies in the first
half of each head and duplicates them into the second; the block FFN uses the
**tanh** GELU while the merger uses the **exact erf** one; and `v.post_ln` is
the *merger's* norm, applied once, not a post-block norm.

The seam is `host_embed`: the graph takes `inputs_embeds`, and
`vision::assemble_multimodal` splices the vision rows in, substitutes
`image_token_id` into the **PLE hash ids** (a different array from the prompt
ids), and advances the M-RoPE position by `max(nx, ny)` per image — not by its
token count.

### Graph inputs

| name | shape | when |
|---|---|---|
| `input_ids` | `[1, seq]` | always |
| `inputs_embeds` | `[1, seq, hidden]` | `FlowOptions::host_embed` |
| `ple_embeddings.{i}` | `[seq, ple_embed_dim]` | one per PLE module |
| `rope_cos` / `rope_sin` | `[seq, head_dim/2]` | `FlowOptions::runtime_mrope`; always on the step flow |
| `state.*`, `pad_mask`, `conv_shift` | see [`decode`](src/decode.rs) | the step flow |
| `hc_residual` | `[1, seq, hc·hidden]` | the MTP and head flows |

Fill `ple_embeddings.0` with `prepare_ple_embeddings`:

```rust,no_run
use rlx_qwen4exp::{DensePleTable, Qwen4ExpConfig, prepare_ple_embeddings};
# fn main() -> anyhow::Result<()> {
# let cfg = Qwen4ExpConfig::qwen38_flash_next();
# let mut table = DensePleTable { data: vec![], rows: 0, head_dim: cfg.ple_head_dim };
let tokens = [9707u32, 11, 1879];
let emb = prepare_ple_embeddings(&cfg, 0, &tokens, &mut table)?;
assert_eq!(emb.len(), tokens.len() * cfg.ple_embed_dim());
# Ok(())
# }
```

`DensePleTable` holds the whole table in memory and is only usable at synthetic
sizes. For the real one, implement `PleRowSource` over a `pread` into the GGUF:
at `head_dim = 160` and `IQ4_NL` a row is 90 bytes, and a `seq`-token batch
touches `16 · seq` of 320 million rows.

## Testing

```sh
cargo test -p rlx-qwen4exp
```

117 tests (with `--features vision`) on synthetic weights and header data:
config parsing from both GGUF metadata and HF `config.json` (and that the two
land on the same model); the PLE hash constants against the published metadata;
the n-gram EOS reset and its chunk-boundary equivalence; the QSA block/tail
visibility algebra; the tensor manifest against the real 1224-tensor list in
**both directions** (names and orientations, for all 48 blocks); the MTP
contract against its own `mtp-*.gguf`; the mmproj contract against the published
`mmproj-F16.gguf`; and four equivalences that each compare two independently
spelled paths —

* **decode == prefill**, token by token;
* **chunked prefill == single-shot prefill**, including a right-padded final
  chunk;
* **a padded chunk exports the same state as an exactly-sized one**, which is
  the only test that can see `pad_mask` and `conv_shift` at all;
* **the native ViT == `rlx-qwen35`'s independent implementation** of the same
  architecture, element for element on identical weights (`rlx-qwen35` is a
  dev-dependency only — an oracle, not a dependency).

All four were checked to actually fail: sabotaging the dilated tap offset, the
scan carry, the conv shift, the pad masking, the second patch convolution or
the position-table permutation each breaks exactly one of them.
`tests/vit_parity.rs` also records the two things it is *blind* to and where
those are covered instead.

Real weights, without downloading the model:

```sh
just qwen4exp-real     # fetches a ~186 MB range-sliced subset, then tests
```

`scripts/qwen4exp_subset.py` range-fetches `blk.1` (GatedDeltaNet + the PLE
module) and `blk.3` (gated attention + QSA), `--experts N` of each block's 512
routed experts (8 by default), and the `output_hc_*` tensors into a valid
standalone GGUF. A quantized expert bank is `[in, out, experts]` with the expert
axis outermost, so each expert is a contiguous byte range — 8 of 512 is ~12 MB
rather than ~800 MB.

Because `--experts` is a *parameter*, nothing reads the count from a constant:
`moe::gguf_expert_count` takes it from a bank's own shape, which is the
authoritative source. The metadata still describes all 512 — configuring that
against sliced banks would index weights that are not there, or the wrong ones,
and still run. The expert axis is the **last** GGML dimension (`[2560, 640, 8]`
is 8 experts of a 2560×640 projection), and the function refuses when blocks
disagree rather than picking one.

`just qwen4exp-real` additionally runs, on real weights: **paging the real
quantized expert banks** (`blk.1` is IQ2_XXS gate/up and IQ4_NL down) against
the same layer with all of them resident, the converter's sign
and offset conventions, both block kinds end to end, the QSA short-circuit
against the selection machinery it skips, and **decode reproducing prefill**
(two real blocks renamed into a two-layer model, so a 186 MB file drives the
whole step flow).

`just qwen4exp-vision-real` does the same for the tower — `--mmproj 3` slices
3 of its 27 blocks plus every global (the real patch kernels, the real 2304-row
position table, the real merger) into ~172 MB instead of 904 MB.

## Status

Complete against `modular_qwen4_exp.py`, llama.cpp's
`src/models/qwen4exp.cpp` and `conversion/qwen4exp.py`, and every tensor shape
and metadata key in the published GGUF conversions. Prefill, chunked prefill,
decode, the MTP draft block and the multimodal seam all run.

Not done:

* **No end-to-end generation on the full checkpoint.** Every piece is here and
  tested against the path it replaces — [`runner`](src/runner.rs) drives the
  loop (bit-for-bit against the explicit one), `GgufPleRows` reads the 28 GB
  n-gram table 90 bytes at a time, and [`paged_moe`](src/paged_moe.rs) runs a
  layer from 10 experts off disk rather than 512 resident (against the resident
  layer). What is missing is the assembly: a runner that drives all 48 layers
  with paged MoE, and the hours of disk it would read.
* **Decode and chunked prefill stop at the QSA budget.** Past 2048 cached
  positions the indexer genuinely selects, which needs a position-ordered
  indexer-key cache; the builder refuses rather than quietly running dense
  attention.
* **Padding must be on the right, and the batch is 1.** That case is exact;
  left padding and multi-sequence packing would need the reference's per-query
  visibility walk, which [`indexer`](src/indexer.rs) does not emit.
* **The tower has no deepstack path and no video.**
  `deepstack_visual_indexes` is `[]` in the published checkpoint, so
  [`vit`](src/vit/) refuses a mmproj declaring one rather than dropping the
  features; video would need the two patch kernels fed *different* frames
  instead of the same one.
* **No tokenizer or chat template.** The GGUF carries a `qwen35`-pre BPE
  vocabulary and a chat template with this model's `reasoning_effort` system
  injection; neither is wired here.
