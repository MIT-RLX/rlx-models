# Changelog

## Unreleased

### DeepSeek-V4.1-Flash

`deepseek-ai/DeepSeek-V4.1-Flash` (`model_type: deepseek_v41`, released
2026-09-10) is a different architecture from V4, not a revision of it, and the
V4 path would have loaded it into a silently wrong model. It is now its own
port in `rlx-models-core`: `dsv41` (config/shapes), `dsv41_graph` (prefill +
pipeline stages), `dsv41_decode` (KV-cache decode), `dsv41_engram`,
`dsv41_vision`, `dsv41_dspark`, `dsv41_quant`.

What V4.1 changes:

- **CSA2 KV sharing.** `compress_ratio > 0` no longer means a layer compresses
  its own KV. Only `kv_source_layer_ids` (`[2, 8, 14, 20]`) run a compressor and
  only those own index keys; every layer up to the next source reads that cache.
  `index_source_layer_ids` splits the same way for the Indexer. Feeding V4.1 to
  the V4 builder would have built 38 compressors where the checkpoint has 4.
- **A hierarchical Indexer.** Layer 20 picks `candidate_topk_blocks` blocks of
  `candidate_block_size` compressed positions; layers 24/28/32/36 score only
  inside them.
- **Engram** — n-gram hash lookups mixed into the residual stream at layers 1
  and 14, over two 384-million-row tables. Every part of the hash has to match
  training exactly: the normalized token map, the prime-sized bucket ranges, and
  the multipliers, which come from `np.random.default_rng(10007 · layer_id)` and
  therefore need numpy's `SeedSequence` → PCG64 → Lemire chain reproduced bit for
  bit (`dsv41_engram::np_rng`). The primes summing to the checkpoint's own
  `engram_num_embeddings` is what confirms the layout.
- **Vision** — a DeepSeek-ViT with 2-D RoPE plus an unfold/MLP aligner.
- **Per-stage expert banks** — the three DSpark stages route over 128 experts
  top-3, the backbone over 384 top-6.
- **A threaded Hyper-Connection pre-mix** — each sublayer computes the mix the
  *next* one consumes, and there is no `hc_head`: the final collapse reuses the
  last block's FFN mix.
- **Three quant scale layouts** behind one `weight_block_size: [32, 32]`:
  32×32 tiles for the FP8 Linears, row-wise groups for the FP4 experts, and —
  the trap — row-wise groups for the FP8 Engram table, which a tiled reading
  would smear across 32 rows of the table at a time.

Parity: the released `inference/model.py` was run on CPU with its tilelang
kernels transliterated to torch, and every stage of the port matches it to
**2e-7** relative through the logits — engram, compressor, compressed RoPE,
candidate blocks, Indexer top-k, sink attention, o-LoRA, MoE. Decode reproduces
prefill token for token; a split pipeline stage reproduces the single-shot run;
the vision tower and the DSpark draft head (rings, logits, Markov bias,
confidence, and the greedy draft tokens) match their own fixtures. End-to-end on
real weights stays out of reach: the only checkpoint is 510 GB of fp8/fp4.

Two details worth knowing when reading the code. The reference round-trips
activations through FP8/FP4 in place (`act_quant(..., inplace=True)`); those
calls are precision simulation, not semantics, and the port computes the
F32-exact value. And `torch.topk`'s order among **equal** scores is unspecified
while the Indexer produces exact ties constantly (it rectifies its head scores,
so any position every head dislikes scores exactly zero) — the port keeps the
lowest index, matching `Op::TopK`, and the parity harness pins torch to the same
rule. A thresholding gate instead keeps *every* tied entry and quietly overruns
the `index_topk` budget.

### Fixed: DeepSeek-V4 Hyper-Connections mixed the streams transposed

`build_hc_post` contracted the Sinkhorn combination matrix on the wrong axis.
The reference writes the residual term as
`(comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)`, which aligns `comb`'s
leading `hc` with the residual's and reduces *that* one — `combᵀ·residual`. The
port summed the other index, which passes every shape check (`comb` is square)
and silently permutes how the parallel residual streams mix, in every V4 prefill,
decode, pipeline and DSpark graph. `examples/hc_probe.rs` did not catch it
because its inline reference encoded the same transpose; both are fixed, and the
probe now matches at cosine 1.0.

Alongside it, `build_hc_pre` / `build_hc_head` used `hc_eps` for the RMS
pre-norm where the reference uses `norm_eps`. For V4 those are both 1e-6 so
nothing moved, but V4.1 sets them 14 orders of magnitude apart (1e-20 vs 1e-6)
and every mixing coefficient would have been perturbed. Both now take the two
epsilons separately.

### Warnings cleared, including three classes the lint gate never saw

`scripts/rust-lint-gate.sh` runs `cargo clippy --workspace --all-targets -D
warnings` — default features, workspace members only, and no rustdoc. Each of
those is a hole, and all three had something in them.

- **Rustdoc: 20 broken intra-doc links across 11 crates.** `cargo doc --workspace
  --no-deps` under `RUSTDOCFLAGS=-D warnings` now exits 0. Most were public docs
  linking private items (`PREFILL_BUCKET`, `MASK_PENALTY`, `linear_f16`,
  `emit_router_from_logits`, `Ctx::lstm_step`, `crate::export::stage_graph`,
  `SCALE`), which rustdoc rejects because the reader cannot follow them; the rest
  were stale paths (`rlx-tada`'s `Self::embed`, a method that no longer exists
  after the refactor — the real pair is `conditioning` + `with_token`),
  cross-crate paths needing qualification, an ambiguity where upstream exports
  `split_vjp` as *both* a function and a module, and an unescaped `[8,8,4,2]`
  that rustdoc read as a link. Use `--keep-going`: rustdoc stops at the first
  failing crate, so a naive loop finds them one at a time.
- **Non-default features.** A 131-crate sweep under `apple-silicon`/`espeak`
  found `rlx-neuralhash`'s `tests/backends.rs` failing to compile with any
  backend feature on: `let all = vec![…]` followed by `#[cfg]`-gated
  `all.push(…)`. Invisible by default because every push is cfg'd out. Fixed with
  `#[allow(unused_mut)] let mut`, which is the only form correct in both
  directions — plain `let mut` warns when the features are off.
- **Workspace-`exclude`d crates, which nothing lints.** `bench_matmul_rlx` builds
  again (its stale direct `rlx-ir`/`rlx-runtime` pin was caught by the 0.2.16
  bump; it had been unbuildable). `rlx-ten-vad-mcu` reports two errors when
  linted against the host — meaningless for a `no_std` firmware crate; for its
  real target (`--target riscv32imc-unknown-none-elf`) it is clean.

### 0.2.16 — targets upstream RLX 0.2.16

**espeak-ng 0.1.3 → 0.2.0, and the local patch is gone.** `rlx-kittentts` and
`rlx-sanotts` pinned `0.1.3` while `.cargo/config.toml` patched `espeak-ng` to a
sibling checkout — which had since moved to 0.2.0. A `[patch.crates-io]` only
applies when the patched version *satisfies* the requirement, so the patch was
inert and cargo said so on every single invocation
(`patch 'espeak-ng v0.2.0' was not used in the crate graph`). The en-US
phoneme-table fix that the patch existed for shipped in 0.2.0, so the pin moves
up and the patch entry is deleted: espeak-ng now resolves from crates.io and
that warning is gone.


The workspace version moves to **0.2.16**, matching the `rlx` it is built
against, as past releases did (`rlx-models` takes the number of the upstream it
targets; 0.2.14 was never published — crates.io tops out at 0.2.11). That is 183
internal path-dep pins plus 43 crate manifests, and it caught four *upstream*
deps pinned directly rather than through the workspace table
(`bench_matmul_rlx` and `rlx-vision-bench` held `rlx-ir`/`rlx-runtime` at
0.2.14), which would have dragged a second copy of the runtime into the graph.

The 28 `rlx*` pins move from `^0.2.14` to `^0.2.16`, and the two call sites that
the newer API had already outgrown are updated:

- `rlx_distributed::ModelCost` gained `per_layer_expert_bytes`,
  `per_layer_expert_active_bytes`, `kv`, `params` and `hidden_size`.
  `rlx-models-core/examples/dsv4_cluster.rs` now declares the KV profile from the
  config's MLA fields (`kv_lora_rank + qk_rope_head_dim`, bf16) and falls back to
  `KvProfile::unknown()` rather than guessing — the planner refuses to plan on
  `Unknown`, which is the point. Its estimate reads shard sizes rather than
  tensor shapes, so routed-expert bytes are not separable and stay folded into
  `per_layer_bytes`; `ModelCost::from_tensor_index` splits them properly when an
  index is available.
- `rlx_flow::blocks::BindDecodeInputsStage` gained `kv_past_len: Option<usize>`.
  `rlx-locateanything` passes `None`: its `past_k_*`/`past_v_*` inputs are
  declared at exactly `past_seq` rows and concatenated whole, so there is no
  spare capacity to distinguish.

Note these are the same two sites this file previously said to leave alone. That
was correct while the pin said 0.2.14 — they matched the *released* field set
exactly, and "fixing" them would have broken a 0.2.14 release. Which side is the
target has to be decided before touching them.

**`cargo clippy --workspace --all-targets -- -D warnings` now exits 0** across all
195 crates — these two were the last failures.

**Both release blockers are now cleared.** Upstream published `rlx*` 0.2.16 —
including **`rlx-opscope`, which had never been published at any version** and
was the hard blocker, since it is a dev-dep of five crates and cargo resolves
dev-deps at lock time for the whole workspace. Verified: with
`.cargo/config.toml` moved aside, `cargo metadata` now resolves the entire
workspace from crates.io with **no path sources at all** — previously it failed
outright with `no matching package named 'rlx-opscope' found`.


### Kyutai TTS — fixed (fox 0/6 → 6/6)

`rlx-kyutai-tts` was tracked as producing unintelligible audio. It does not: Whisper
transcribes fluent English that ignores the script entirely ("I can't do this." on repeat).
Running the same prompt with `RLX_KYUTAI_TTS_EAGER=1` returns **"Hello World!"**. The eager
reference is correct; the **RLX temporal backbone** — which `KyutaiTtsBackend::open` selects by
default — is not.

The text head makes it obvious once you look. A DSM text stream should only ever choose between
`pad` and `new_word`, the script supplying the words, and the eager head does exactly that:
token 3 and token 0 carry the mass, every word token sits at ≈ −17. The RLX head returns 8000
logits that are all equal to within ~1e-7, with pad and new_word at ~0. The defect is upstream
of the projection: one decode step from a reset state puts the two backbones' hidden states at
**cosine −0.617** (max|Δ| 3.55).

**Why this shipped.** `rlx_backend_parity.rs` compares the RLX graph on CPU against *itself*
(`assert_logits_match_cpu(label, &cpu, &cpu)`) and then against the same graph on other
devices. That is cross-device self-consistency; it cannot fail on a graph that is uniformly
wrong. `tests/rlx_vs_eager_backbone.rs` makes the comparison that was missing — RLX vs the
eager reference, on the backbone output rather than the head, so a failure localizes to the
transformer stack. It skips without a checkpoint and fails loudly with one. It fails today;
that is the point.

**The cause: the SwiGLU width came from the config instead of the checkpoint.** `hidden_scale`
is not the hidden width — Kyutai applies the usual SwiGLU ⅔ adjustment, so the 1.6B checkpoint
stores `gating.linear_in.weight = [11264, 2048]`, hidden **5632**, while `TtsDims::from_cfg`
computed `dim_feedforward / 2 = (2048 · 4.125) / 2 = 4224`. Every gate/up slice was taken at the
wrong offset and `linear_out` was transposed against the wrong stride, in all 16 layers. The
eager path never noticed because it reads each tensor's own shape. `TtsDims::from_cfg_and_weights`
now takes `ffn` from the packed projection, and `for_each_transformer_param` *ensures* the slice
matches the stored element count rather than mis-slicing in silence.

A second defect, also fixed: eager **skips** cross-attention when no speaker is set and attends
only the real context frames when one is, while the RLX graph always attended a zero-padded
`MAX_SPEAKER_CROSS_FRAMES` buffer with `MaskKind::None`. A zero key scores 0 against any query,
so every padding slot took weight `exp(0)` and diluted the real conditioning instead of dropping
out. Cross-attention is now masked to the conditioner's real frame count (synthetic gate:
max|Δ| 5.5e-3 → 1.2e-7).

**The fixture is why no test could see it.** `synthetic_weights` built the gating projection as
`[2 · (dim · hidden_scale / 2), dim]` — the same wrong convention as the buggy `from_cfg` — so
the two agreed with each other. It now emits checkpoint-shaped weights, and
`tests/rlx_vs_eager_layer.rs` compares the RLX graph against the eager `StreamingTransformer` on
synthetic weights at 1/2/3 layers (no checkpoint, so it runs in CI), with cases pinning the
width convention and asserting the binder now fails loudly on a config-derived one.

The RLX path also gained the `RLX_KYUTAI_TTS_TRACE` logits dump the eager path already had.

Verified: the whole kyutai suite is green (including 8 cross-backend cells), real-weight backbone
parity passes, and end-to-end Whisper returns *"The quick brown fox jumps over the lazy dog."* on
both **cpu** and **metal**.

### `rlx-voxtral-tts` — two full copies of the model removed from the load path

Loading the 4B checkpoint OOMed. `CheckpointParamLoader::take` — a `&mut self` method on the
`WeightLoader` trait, called once per key by `WeightMap::drain_loader` — was implemented as
`get(key).cloned()`, so the loader kept the entire backbone alive in f32 while the `WeightMap`
filled with a second copy of it: roughly 15 GB each, on top of the graph arenas. It now
`remove`s, which is what the method name says and what makes `remaining_keys` meaningful.

`CompiledBackbone::run_prefill` / `run_decode` also did `ensure_backbone_params()?.clone()` —
cloning that same ~15 GB snapshot purely to satisfy the borrow checker, since both then touch
`self.sharded` and `self.graph_params`. A `with_backbone_params` helper moves the snapshot out
and puts it back, the same idiom the surrounding code already uses for `self.sharded`.

Two `.clone()` sites remain in the HIR-template builders, where the snapshot is consumed by the
loader; with the `remove` fix above those now drain rather than duplicate. Not measured
end-to-end — this machine did not have the headroom to load the model without thrashing.

### TTS bench — the memory column

The TTS bench recorded RTF but had no RAM column, which is half of the
`rlx_beats_onnx_criteria` acceptance test. Added `metrics::{peak_rss_mb, RssTracker,
RssMetrics}`, mirroring `rlx_llm_bench::metrics::peak_rss_mb` and
`rlx_core::asr_bench::peak_rss_mb` so all three leaderboards compute it identically.

Two things make the number mean something. The suite already runs each `(model, device)` cell
in its own worker subprocess, so `getrusage(RUSAGE_SELF)` is scoped to a single model. And
`ru_maxrss` is a monotonic high-water mark that cannot be reset between models, so the tracker
takes a baseline immediately before the adapter is constructed and reports `peak - baseline` —
without that, the Whisper scorer (loaded before the model loop) is charged to every small TTS
model. `results.jsonl` carries peak, baseline and model-attributable MB; `BACKENDS.md` gains a
"Peak RAM (MB, model only)" table; `summary.json` gains `max_model_rss_mb` — the max rather
than the median, because the criterion is "same-or-less RAM than the reference".

First measured row: **luxtts cpu = 13 380 MB at RTF 0.55**.

### LuxTTS — wgpu unblocked (all 5 Apple backends)

`rlx-luxtts` now runs on **cpu, metal, mlx, wgpu and coreml, all at cos 0.99848** vs CPU.
wgpu previously panicked before producing a sample. The tracked symptom ("remainder with
divisor of zero") had since moved to `rlx-wgpu arena: no offset for node NodeId(731)`, and
both were the same underlying thing: LuxTTS's flow decoder builds an `Expand [0,1,512]`, and
`rlx-wgpu` could not handle a **zero-element tensor**. Fixed upstream in two places — the
memory planner records no buffer for a zero-size slot, so the arena had no offset to return;
and once that compiled, ~79 zero-extent dispatch guards in the run loop skipped their step
without advancing the cursor, hanging forever. See the `../rlx` changelog for both.

### KittenTTS — native path fixed (three defects)

`rlx-kittentts`'s native graph did not run at all on CPU; the tracked symptom was narrower
than the cause, and there turned out to be three independent bugs stacked on one another.

**Waveform caps must be frame-aligned.** The vocoder divides `max_wave` two different ways —
the NSF sine chain at the `f0_upsamp` nearest ×300, and the generator AdaIN wave-frame cap at
600 samples/frame — both with `div_ceil`. A cap that is not a whole number of *both* builds the
upsampled sine source longer than the wave axis it feeds: `ceil(200_000/300)*300 = 200_100`.
MLX rejected the resulting `Reshape`; CPU and Metal accepted it and read a 100-sample-misaligned
harmonic source. The unaligned caps in play were the TTS bench's round `200_000`, the wgpu 32 k
storage-bind ceiling, and the Vulkan 80 k `maxStorageBufferRange` ceiling — each off by exactly
100 samples. `bundle_patches::align_waveform_cap` now rounds **down** to a 600 boundary (down,
because those are memory ceilings that rounding up would breach; the cost is under one frame,
25 ms at 24 kHz), applied at `compile_waveform_cap`, `device_policy::clamp_waveform`, and
`set_import_max_waveform_samples`.

**`f0_upsamp` was importing as zeros.** `rlx-onnx-import` lowered nearest `Resize` for exactly
two shapes: a 2×2 upsample, and a width-only resize gated on `h_in == h_out == 1`. KittenTTS's
`f0_upsamp` is `[1,1,1,F] → [1,1,300,F]` — a *height* upsample — so it matched neither and fell
through to the zero-filled stub. The NSF f0 source was dead on every backend. Fixed upstream
with the rank-4 case of the identity the NCDHW path already uses: `[N,C,H,1,W,1]` broadcast to
`[N,C,H,kh,W,kw]` has exactly the row-major order of `[N,C,H·kh,W·kw]`, which is ONNX's
asymmetric+floor rule. (Recent upstream turned that silent zero-fill into a hard error, which
is how this surfaced — the stub had been quietly wrong for as long as it existed.)

**The f0 repair patched the wrong node.** The importer lowers one ONNX op into a chain of HIR
nodes and stamps the ONNX node name on more than one link. `find_node_by_name` returns the
first, but consumers read the last, so `inject_f0_nearest_upsample` rewrote the head of the
chain and left the voicing-mask `Greater` reading a stale rank-4 `[1,1,300,seq]` alias — while
the same pass patched that `Greater`'s *output* to `[1,max_wave,1]`. Nothing can broadcast rank
4 down to rank 3. Added `find_last_node_by_name`.

`kitten_tts_mini_rlx` unit tests go 15/19 → **19/19**, `native_smoke` 0/2 → **2/2**, and the TTS
bench's exact load parameters `(256 tokens, 200_000 samples)` synthesize on **CPU and MLX**
(`bench_cap_regression.rs`). Whisper on the long fixture returns *"This is a longer sentence for
testing the K-10 text to speech system in."*

`native_smoke.rs` also gained the process-global compile-cap mutex that
`native_whisper_roundtrip.rs` already had: the engine's mel/wave caps are process-wide, so its
two tests raced and the long-sentence one picked up the short one's 48 k cap.

### `rlx-vibevoice-asr` — VibeVoice-ASR-Streaming-7B

Native RLX path for
[microsoft/VibeVoice-ASR-Streaming-7B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B):
BF16 safetensors load (dual ConvNeXt encoders + SpeechConnectors + Qwen2.5-7B),
GELU VAE blocks, and the official chunked KV streaming loop (stop on
`<|text_chunk_end|>`). File ASR defaults to `encode_then_split` (one VAE pass);
`--encode split_then_encode` / `RLX_VIBEVOICE_ASR_ENCODE=split` for live mic.
LM weights snapshot once in RAM; intermediate speech frames use KV-only decode
(no lm_head). VAE graphs cached by padded length. Timing:
`RLX_VIBEVOICE_ASR_TIMING=1`. CLI `--model-dir`;
`just fetch-vibevoice-asr-streaming` / `just vibevoice-asr-streaming`. Backend
matrix: `just features=all-backends test-vibevoice-asr-backends`. BitNet GGUF
path unchanged.

### `rlx-glm5next` — GLM-5.3-Flash (`glm5next`)

320 B total / 18 B active, and four architectures at once: 34 KDA linear-attention
layers, 11 NoPE latent-attention layers behind a DeepSeek sparse-attention
indexer, mHC hyper-connections wrapping every sublayer, and a 288-expert
clamped-SwiGLU MoE. Config parses from GGUF metadata (`glm5next.*`) or the
upstream `config.json`; `tests/config_parsing.rs` runs both against the published
files and asserts they agree.

Three findings worth writing down, because each is a plausible-looking wrong
answer:

- **The text model has no RoPE.** `qk_rope_head_dim = 0`, and the reference
  config *rejects* anything else. Position information reaches the sparse
  layers only through the KDA layers beneath them, so there is no rope table
  input to the graph at all.

- **DSA is exactly dense causal attention up to 2048 tokens.** The indexer picks
  `index_topk / index_kpool = 512` pools of 4 tokens out of `floor(seq/4)`
  complete pools, plus each query's own incomplete tail — and "every complete
  pool at or before `q`" ∪ "that tail" is exactly `0..=q`. Below the budget the
  selection is the identity, so the MLA layer takes the fused `MaskKind::Causal`
  path instead of materializing an `[s, s]` bias it already knows. An algebraic
  identity, not an approximation, and a test pins it by running both paths.

- **GGUF stores `attn_k_b` and `attn_v_b` in opposite orientations.** The
  converter transposes the key half so GGML contracts `qk_nope_head_dim`
  (llama.cpp absorbs the query into the latent), but leaves the value half
  contracting `kv_lora_rank`. Assuming one layout for both silently transposes
  the key projection.

mHC is implemented separately from the one in `rlx-motif` on purpose: the two
differ in the input norm (unweighted vs. weighted), in `comb` (softmax vs.
sigmoid), and in the Sinkhorn schedule (column-first, so `iters` column passes
but `iters - 1` row passes). `tests/mhc_reference.rs` checks the whole site
against an f64 transcription of `Glm5NextTextHyperConnection`.

Incremental decode is wired: a latent KV cache and the carried KDA conv/scan
state, one token per `run()`. Two notes on it —

- **The KV cache stores the latent**, 512 floats per token per layer instead of
  32768 expanded per-head keys and values: 46 MB rather than 2.9 GB at
  `cap = 2048`. Attention therefore runs *absorbed*, which is algebraically the
  same as prefill's expanded form, and the equivalence test validates both
  readings of `attn_k_b` / `attn_v_b` for free.
- **Decode refuses a capacity past `index_topk`.** Past 2048 tokens DSA stops
  being the identity; running dense attention there would be a different model
  from the trained one, so it errors instead.

**This surfaced a silent wrong-answer bug in rlx-cpu**, now fixed upstream.
`matmul_shape` broadcasts a rank-2 operand across the other's batch, so
`[M,K] @ [B,K,N]` is legal and yields `[B,M,N]` — but the thunk dispatch only
took its batched-GEMM path when *both* operands were rank ≥ 3. The rank-2-lhs
case fell through to the 2-D flatten, emitting a single `Sgemm` against the
rhs's first matrix and leaving every later output batch holding whatever was in
the arena. `BatchedSgemm` already had the `a_bcast` flag for exactly this; only
the condition that reaches it was missing. (f64 has no broadcast flags on its
batched thunk, so that case now asserts instead of returning garbage.)

The MLA prefill path expands the latent to per-head keys and values, which is
precisely a rank-2 × rank-3 product, so it was wrong on every head after head 0
— and *nothing already written caught it*: the finiteness and dense-vs-sparse
tests both ran through the same bad expansion and agreed with each other. Only
prefill-vs-decode caught it, because decode happened to spell the same maths
with the batched operand on the left. The emitter now always does that (it is
also 268 MB cheaper at `seq = 2048` than materializing the broadcast), and
`tests/mm_broadcast.rs` pins the primitive.

**Validated on real published weights, without downloading the model.** A GGUF
header carries every tensor's byte offset, so `scripts/glm5next_subset.py` pulls
`blk.0` in full and `blk.3`'s attention/indexer/mHC out of one shard over HTTP
range requests — **337 MB instead of 93 GB** — and writes a valid single-file
`glm5next` GGUF with the real metadata (`just glm5next-real`). Those two blocks
are the model's two layer kinds, so between them they cover every block the
crate emits bar the routed experts. Against them `tests/real_weights.rs` pins:

- every tensor name and shape in the GGUF contract, against the published
  artifact rather than a fixture written from the same understanding — including
  the two opposite `attn_k_b` / `attn_v_b` orientations;
- the K-quant dequant path (`Q5_K`, `Q6_K`, `Q8_0`), which no synthetic f32 test
  touches;
- both attention blocks running at plausible magnitudes (KDA rms 0.0075, MLA rms
  0.43 on unit-ish input);
- the *trained* mHC gates being Sinkhorn-well-formed — `comb` columns summing to
  1 and `post` inside `[0, 2]` are properties of the learned `scale`/`base`, not
  of the shapes;
- **decode reproducing prefill on real weights** — KDA to 2.4e-6 relative, and
  MLA to 4e-6, the latter being absorbed-vs-expanded attention agreeing across
  textually disjoint code paths;
- the trained DSA indexer reproducing the causal mask below its budget, exactly.

**Two more silent wrong-answer bugs, both found by making a vacuous test real.**

`dense_and_sparse_paths_agree_when_the_budget_is_not_binding` claimed to pin the
DSA identity. It did not: selection is the identity *exactly when* `is_dense()`
holds, so both configs it compared took the short-circuit and the indexer never
ran. `IndexerDims::force_emit` now runs the machinery in the regime where it
provably selects everything, so the two paths can actually be compared — and
that immediately failed, twice over:

- **`Op::TopK` does not filter.** It returns `select_k` indices whatever the
  scores are, so a query with fewer visible pools than the budget — every query
  early in a sequence — was handed pools from its own future, and the scatter
  made them visible. This is the reference's `selected_valid` step, which was
  missing. Without it the mask was not causal.
- **rlx-cpu's `ScatterElements` mis-strides narrow indices** (fixed upstream).
  ONNX lets `indices` be smaller than `data` along an axis, and the flat
  position then decomposes by the *indices'* strides — but the kernel was never
  given the indices' shape, so it guessed with the data's axis stride and wrote
  to the wrong rows. Correct only when the two shapes agree, which is why
  `GatherElements` (whose output *is* the indices' shape) was fine and this was
  not. `indices_shape` is now threaded through the thunk.

With both fixed the emitted mask is exactly the causal mask, and on real trained
weights the two paths agree bit-exactly (Δ = 0). `tests/scatter_gather_elements.rs`
pins the primitive.

**Coverage of all 46 blocks, for free.** `tests/tensor_manifest.rs` checks the
checkpoint contract against a 52 KB fixture of the real tensor index — every
name, every shape, the layer schedule read off the weights rather than the
metadata — and separately that the emitters consume exactly those names and no
others. The one documented exception is asserted rather than waived: below the
`index_topk` budget the DSA short-circuit means the seven indexer tensors per
MLA layer go deliberately unread.

**Packed weights: the projections no longer dequantize.** `common::linear` now
consults `WeightSource::take_packed`, so building through
`rlx_core::flow_bridge::PackedWeightLoaderSource` turns every 2-D projection
into a fused `Op::DequantMatMul` over the GGUF blob — no f32 weight is
materialized. `build_glm5next_text_flow_with_source` is the entry point.
Measured on the real subset, and the output is **bit-identical** either way:

```text
  KDA blk.0   550.9 MB dequantized → 96.3 MB packed   (5.7×)   max |Δ| = 0
  MLA blk.3   469.8 MB dequantized → 146.6 MB packed  (3.2×)   max |Δ| = 0
```

MLA gains less because its per-head `attn_k_b` / `attn_v_b` are 3-D and stay
f32, as do the norms, `ssm_a`, `dt_bias`, `exp_probs_b` and the depthwise
`ssm_conv1d_*` kernels. The routed expert banks also stay f32 — they go through
`GroupedMatMul`, which has no packed form here, and at 2.2 GB per MoE layer they
are precisely what a whole-model run still needs.

**The routed experts now run packed too — the last f32 holdout.** `Op::DequantGroupedMatMul`
already existed upstream on all five backends; what was missing was a way for a
loader to *offer* a packed expert bank. `WeightSource::take_packed` describes a
2-D linear and has nowhere to put an expert count, so a 3-D `[E, out, in]` bank
read through it would silently report `out_dim = E`. Added upstream:

- `rlx_flow::GgufPackedBank` + `WeightSource::take_packed_bank` (default `None`,
  so nothing else changes), implemented by `PackedWeightLoaderSource`;
- `Graph::dequant_grouped_matmul_packed` and `HirModule::dequant_grouped_matmul_packed`,
  mirroring the existing `dequant_matmul_packed`.

`take_packed` also now *declines* 3-D tensors rather than mis-describing them.

GGUF's `[experts, out, in]` is already the op's slab layout, so the packed path
skips the `[E, N, K] → [E, K, N]` transpose the F32 path needs — which
constant-folding would otherwise materialize as a second copy of the bank. On 8
real experts sliced out of `blk.3` (`scripts/glm5next_subset.py --experts 8`;
each expert is a contiguous byte range, so 8 of 288 is ~60 MB, not 2.2 GB):

```text
  MoE blk.3, 8 real experts   906.1 MB dequantized → 78.8 MB packed  (11.5×)
```

within 1.2e-6 relative of the dequantized result, and the first test to exercise
`IQ2_XXS` / `IQ3_XXS` dequant at all.

**Also fixed upstream: f64 batched matmul could not broadcast.** The companion
to the f32 fix above — `BatchedDgemmF64` strided both operands by their matrix
size unconditionally, so a batch-1 operand was over-read. It was previously left
as a loud `assert!`; it now carries `a_bcast` / `b_bcast` like `BatchedSgemm`,
which also fixes the pre-existing case of two rank-3 operands where one has
batch 1.

Still no whole-model run — but the reason has changed. Every weight class the
model uses now has a packed path, so what remains is scale rather than a missing
capability: 46 blocks × 2.2 GB of routed banks, which needs the *paging* half of
the story (as in `rlx_kimi_k3::moe`), not another kernel. The MTP block
and the vision tower are parsed but not built; `with_mtp` is an error rather than
a silent skip.

### One seam per number format, in `rlx-ten-vad-core`

`math` was already the single seam for floating point — every transcendental
goes through it so the `std` / `no_std` split is decided once. There was no
counterpart for fixed point, and the two integer paths had quietly drifted:

- **`fixed_math` — the integer seam.** `rsh` (round-to-nearest right shift),
  `cmul_q` (the complex butterfly), and `lut_q15` (Q15 table interpolation),
  each of which existed in two places or none. `fixed` and `fft_fixed` now
  share them.

- **The network rounded its requantisations; the transform truncated its.** An
  arithmetic shift floors toward −∞, so the error was −1..0 LSB rather than
  ±½ — a bias, and ten radix-2 stages accumulate it in one direction instead of
  cancelling. Unifying on `rsh` took the integer FFT from **13.3 to 17.6 bits**
  against the f32 transform, a 20× smaller error, for one add per butterfly
  (+0.3% on `rv32imc`).

- **That was invisible until the test was fixed.** It fed the integer transform
  a quantised signal and the f32 reference an unquantised one, so it measured
  input rounding — ~1 LSB on an amplitude of 6000 — and reported the same 13.2
  bits whether the butterflies rounded or truncated. Its bound was also set at
  the 12-bit budget, loose enough to pass either way; it now sits just above the
  measurement.

- **Q30 twiddles were built through `f32`.** A 24-bit mantissa cannot hold a
  30-bit constant, and `as i32` truncated on top of that: the table sat **218
  LSB off exact**, nearly 8 bits of garbage. `math::cos64` had existed for this
  reason since Ooura's tables needed it; this path had not been given it.
  Now 0.50 LSB — optimal. It does *not* move the end-to-end figure, because
  requantisation dominates there, so it is pinned by a test on the table itself
  rather than an accuracy claim it cannot support. Construction roughly doubles,
  once, ~21 ms at 160 MHz.

### Embedded and hardware targets for TEN-VAD

The port now has two builds. `rlx-ten-vad` is the desktop and mobile one — the
model as an rlx graph, on seven backends. Everything below is the embedded one.

- **`rlx-ten-vad-core` — `no_std` foundation.** The DSP frontend, the f32 scalar
  net, and a new integer-only net, shared by every embedded target so feature
  extraction has one implementation rather than two that drift. Builds for
  `riscv32imc`, `riscv32imafc` and `thumbv7em`. `Vad` is 5,392 B of state;
  `FixedNet` is 1,328 B and needs no allocator.
  - **`fixed` — integer-only forward.** int16 weights with a per-tensor
    power-of-two scale, Q15 activations, 48-bit accumulate, and 1025-point Q15
    sigmoid/tanh LUTs over `[0, 16]`. Against the published ONNX model over the
    250-frame reference clip: **`max|Δ| = 3.7e-4`, cosine distance `2.0e-8`,
    zero decision flips** — where the shipped Agora binary sits at `9.6e-4` and
    `9.4e-8`, so the quantised port is **4.7× closer in cosine than the vendor's
    own build**.
  - The LUT domain is load-bearing: LSTM pre-activations reach 177, so clamping
    at 8 rather than 16 costs `6.9e-3`, thirty times the total error budget.

- **`rlx-ten-vad-fpga` — RTL, exported from the rlx-ir graph.** No hand-written
  model logic: `rlx-fpga`'s new sequential target lowers the same graph the
  runtime executes. **250 frames, 0 mismatches** against `rlx_ten_vad_core::fixed`
  under Icarus Verilog. 173,140 cycles/frame, so 62.5 fps needs 10.82 MHz.
  `yosys synth_ecp5`: 4,067 LUT4, 891 FF, 74 × 18 kbit BRAM, 13 DSP — an
  LFE5U-45F or XC7A35T.

- **`rlx-ten-vad-mcu` — bare-metal RISC-V firmware.** Runs under QEMU `virt`
  (same `rv32imc` as an ESP32-C3/C6) and self-checks against host-generated
  vectors: **0 mismatches**, so the integer net is bit-exact on RISC-V too.
  Measured by instruction count, not `mcycle` — that counter is wall-clock-
  derived on the `virt` board and moves with host load, while instruction
  counts reproduce to 2e-8. Integer net 926,146 instructions/frame (11.7 per
  MAC) against 9,873,154 for the f32 net: **10.7× — because `rv32imc` has no
  FPU.** `tools/insn.c` is the QEMU plugin, and the firmware takes a phase
  selector so a per-run total becomes a per-phase figure.
  - **No allocator.** Every buffer on the inference path is a fixed-size array,
    so the firmware declares no `#[global_allocator]`; `synth` is behind an
    `alloc` feature. `Vad` is 33.6 kB inline.
  - **Fixed: the FPU build hung.** RISC-V resets with `mstatus.FS = Off`, so the
    first floating-point instruction traps as illegal, and with no handler the
    core vectors to 0. `_start` now enables the FPU unconditionally — on a core
    without `F` the field is hardwired to 0 and the write is a no-op. This is
    why the `rv32imafc` target had gone untested.
  - **Which part matters more than which datapath.** Per 16 ms hop:
    ESP32-C3 (`rv32imc`, 160 MHz) needs 14.4 M instructions = 90 ms, **564% of a
    core**; ESP32-P4 (`rv32imafc`, 400 MHz) needs 1.23 M = 3.1 ms, **19% of a
    core, real-time**. Hardware floating point is worth **11.7×** on the
    pipeline.
  - **The integer net stops paying once there is an FPU.** It is 10.7× cheaper
    than f32 on `rv32imc` and 0.9× — slightly *dearer* — on `rv32imafc`, where
    i64 accumulation on a 32-bit core buys determinism rather than speed. Run
    `Net` on an FPU part; `fixed` is for FPU-less cores and for being the FPGA's
    golden model. Either use a core with an
    FPU (ESP32-P4/S3) or port the frontend to fixed point; the network already is.

### Quantisation, measured

Post-training quantisation of this model hits a wall at int8. Over the 250-frame
clip, best scale choice per scheme:

| scheme | bits/weight | size | `max|Δ|` | decision flips |
|---|---|---|---|---|
| int16, per-tensor pow2 | 16 | 146 kB | 2.0e-4 | **0** |
| int8, block-32 | 8.5 | 78 kB | 1.2e-2 | 3 |
| int6, block-32 | 6.25 | 57 kB | 8.4e-2 | 29 |
| fp4 (E2M1) block-32 = MXFP4 | 4.25 | 39 kB | 1.6e-1 | 28 |
| int4, block-32 | 4.25 | 39 kB | 2.6e-1 | 47 |
| ternary (TWN, per-row) | ~2 | 18 kB | 1.4e-1 … 7.6e-1 | 20–114 |

FP4 genuinely beats int4 — 1.6× lower error, 40% fewer flips — because the
exponent absorbs the wide per-tensor dynamic range. It is still unusable. At
75 k already-distilled parameters there is no redundancy to spend, and mixed
precision recovers nothing: a greedy search that demotes tensors to int8 at zero
flips frees 0.0 kB, because only the bias vectors qualify. Layer 1 dominates
sensitivity — `lstm1.weight_ih` is 7× more sensitive than `lstm2.weight_hh`.

### Fixed-point frontend (in progress)

Ported the FFT; the pitch estimator is next. Both steps are driven by
measurements taken with a deterministic instruction counter, not guesses.

- **`fft_fixed` — integer 1024-point real FFT.** i32 datapath, Q30 twiddles,
  i64 products, **no per-stage scaling**: i16-valued audio is under 2^15 and a
  1024-point transform grows magnitude by at most 2^10, so the result fits in
  25 bits with six to spare. The usual halve-every-stage trick would throw away
  ten bits and miss the budget. Real input is packed into a half-length complex
  transform, which is the difference between 1.8x and **2.55x** over the `f32`
  Ooura path (732,322 -> 287,254 instructions/frame).

  It makes no claim to match `ooura` bit for bit — that path stays the
  reference — and is instead held to a measured budget (below).

- **Precision budgets, measured** (`examples/pitch_value.rs`). Perturbing
  features and counting decision flips on the reference clip:

  | | budget for zero flips |
  |---|---|
  | 40 mel features | **12 bits** (step 0.0027 of the normalised range) |
  | pitch feature | **4 bits** (16 levels), or 1% absolute error |

  The pitch feature costs 81% of the frontend and needs 4 bits. Replacing it
  outright with its mean costs only 5 flips of 250. Recomputing it every 2–8
  frames instead is *not* free (1–9 flips) — the estimator carries Viterbi
  state, so skipping breaks its tracking.

- **Where the frontend's 4.62 M instructions/frame go** on `rv32imc`: pitch
  3.74 M (81%), of which a second FFT inside its autocorrelation is 629 k;
  forward FFT 732 k (16%); mel, window, log and normalise 152 k (3%).

- **Recalibration.** The integer FFT gained 2.55x, not the 10.7x the integer
  *net* gained, because a butterfly is bound by 64-bit multiplies and memory
  traffic rather than by the soft-float calls that dominate a dot product. If
  the pitch estimator behaves the same way, the full port lands near 100% of a
  160 MHz core rather than comfortably under it — so an ESP32-C3 remains
  marginal and an ESP32-P4, which needs none of this work, remains the answer.

### Quantisation-aware distillation

`examples/qat.rs` distils the f32 model into a student whose weights are
fake-quantised in the forward pass (`--format fp4|int4|int6|int8|none`), with
gradients taken at the quantised point and applied to f32 masters. Teacher
targets are recomputed per window from the same zero state the student starts
from, so the two see identical context — using the stored per-clip
probabilities would mismatch, since the teacher's state there evolved from the
clip start.

Held-out windows (2,304 scored frames, disjoint from the windows used to select
the checkpoint), against the f32 teacher:

| format | PTQ flips | QAT flips | PTQ 1−cos | QAT 1−cos |
|---|---|---|---|---|
| none *(control)* | 0 | 1 | 0 | 7.1e-9 |
| int8, block-32 | 10 | 7 | 6.7e-5 | 4.3e-5 |
| int6, block-32 | 90 | 84 | 6.7e-3 | 3.8e-3 |
| int4, block-32 | 282 | 163 | 1.7e-2 | 1.7e-2 |
| MXFP4 | 143 | **97** | 1.1e-2 | **7.5e-3** |

MXFP4 loses a third of its errors and int4 nearly half; on the 250-frame
reference clip MXFP4 goes from 26 decision flips to 21. The `none` control
holds 0–1 flips, so the training loop is not what moves the model. It still
does not rescue 4 bits — 97 flips in 2,304 is a 4.2% disagreement rate where
int16 is 0 — so the shipped datapath stays int16.

`dump_distill_set` builds the training set in Rust: **497,775 frames (2.2 h)** of features with
teacher probabilities — 305 k speech, 193 k non-speech — from LibriSpeech
`clean/validation` and ESC-50, fetched with `hf-hub`, read with `parquet`, and
decoded with `symphonia`, plus gain, additive-noise and near-silence variants.
ESC-50 matters: a VAD trained against silence alone learns nothing about rain,
engines or machinery.

### Performance

- **The mel filterbank was dense; it is now banded.** The `[40, 513]` matrix is
  20,520 coefficients of which about a thousand are non-zero — each triangle
  touches one contiguous stretch of bins — and the frontend multiplied through
  all of them. Skipping the exact `+0.0` entries is bit-identical (the
  accumulator is non-negative, so adding `+0.0` changes nothing) and the whole
  frontend still matches the upstream C on 10,250/10,250 values.

  | | before | after |
  |---|---|---|
  | mel + window + log + normalise | 7.80 us/frame | **0.14 us** |
  | frontend | 19.2 us/frame | **11.5 us** |
  | full pipeline (CPU) | 399x realtime | **490x** |
  | MCU pipeline (`rv32imc`) | 15.3 M cycles/hop | **13.5 M** |

  The MCU gains more than the arithmetic suggests: each eliminated multiply was
  a soft-float call. The same fix went upstream as `rlx_ir::audio::MelBands`,
  where it is 10.4x on `Op::LogMel`, and `rlx-conformer-ctc` now shares it.

- **Structured pruning + QAT.** `qat.rs` gained `--prune`/`--prune-mode`.
  Dropping whole *taps* — rows of a `[taps, outputs]` weight, scored by L2 norm
  — removes MACs contiguously, with nothing to index around, unlike the
  activation-sparsity attempt below. Held-out flips of 2,304, against the f32
  teacher:

  | taps dropped | MACs removed | before fine-tuning | after QAT |
  |---|---|---|---|
  | 10% | 8.9% | 260 | **88** |
  | 20% | 18.1% | 369 | **169** |
  | 30% | 27.7% | 424 | **137** |

  QAT recovers 54–68% of the damage, and the remainder is still a 3.8–7.3%
  disagreement rate where dense is 0. Worth it only if the application can
  spend that; for a port claiming parity with the reference it is not.

- **`examples/mac_budget.rs`** accounts for the remaining 79,295 MACs per frame
  and tests the levers that would cut them. Two are dead: **0.00%** of the
  weights are exactly zero, and no matrix is low-rank enough to factor —
  `lstm1 [144, 256]` needs rank 109 to keep 99% of its energy against a
  break-even of 92, so `U·V` would be **1.18x more expensive** than the dense
  product. A third is real but unclaimed: 56% of the conv stack's output is
  exactly zero after its ReLU, worth 14.5% of the frame's MACs, but exploiting
  it by index list made things *slower* (integer net 200 k -> 222 k cycles) —
  the indirection costs more than the multiply it skips. Capturing it needs
  `W_ih` stored input-major so the skip stays contiguous, which is what the
  FPGA datapath already does.

### Changed

- Parity reporting now includes **cosine distance** alongside `max|Δ|`, mean and
  decision flips. The two catch different failures: `max` a single bad frame,
  cosine a systematic tilt. The f32 graph sits at `1 − cos = 1.3e-14` against
  the published model, i.e. the floating-point floor.
- The fixed-point artifacts are generated by
  `cargo run -p rlx-ten-vad --example gen_fixed_tables`, replacing a Python
  script. It calls `rlx_fpga::seq::quantise_pow2`, so the MCU weight blob and
  the FPGA weight image are identical integers by construction. The two
  generators previously disagreed on 20 weights that land exactly on `.5`
  (banker's rounding versus half-away), which surfaced as 1-LSB output drift.
- `rlx-ten-vad` re-exports the core modules instead of carrying its own copies.


### New model crates

- **`rlx-ten-vad` — TEN-VAD (TEN Framework / Agora) voice activity detection.**
  A full Rust port: 40 log-mel bands plus an LPC pitch estimate per 16 ms hop,
  three frames of context, a separable CNN, two 64-unit LSTMs and a dense head
  (~75 k parameters). Weights are embedded (305 KB), so there is **no ONNX
  Runtime and no `libten_vad`** at run time and nothing to download.
  - **The DSP frontend is bit-identical to the upstream C**: 30 750/30 750
    feature values on the fixture and 58 548/58 548 on real speech, `max|Δ| = 0`.
    `src/ooura.rs` is a verbatim transliteration of the reference's Ooura
    split-radix FFT (generated by `scripts/transpile_ooura.py`, since `f32`
    addition is not associative and a mathematically-equivalent FFT is not
    enough), and `src/pitch.rs` follows its `f32` arithmetic operation for
    operation — including `x / (std + eps)` rather than a precomputed reciprocal.
  - **Parity with the published model** (`ten-vad.onnx` + the DSP in `src/*.cc`):
    `2.4e-7` for the network on identical features, and the same `2.4e-7` for the
    whole pipeline since the frontend contributes exactly zero. Zero
    voice-decision flips, on cpu / metal / mlx / wgpu alike (1.8e-7–2.4e-7).
    That residual is onnxruntime's `f32` accumulation order against rlx's — the
    floor for anything that is not a copy of ORT's kernels.
  - Bit-exactness is relative to an **IEEE-strict** build: clang on arm64
    contracts `a*b + c` into `fma` by default, and `-ffp-contract=off`/`on`/`fast`
    each yield a different spectrum from the same C source (1024/1024, 369/1024,
    351/1024 bit-identical respectively). The reference is not bit-reproducible
    across compilers; the fixtures pin `off`.
  - **`TenVadBatch` scores `chunk_frames` frames per dispatch with the LSTM
    state carried across chunks** (`Op::Lstm { carry }`), so chunk size is a
    latency/throughput dial and not a correctness one — the answer is identical
    at every size, pinned by `chunk_size_does_not_change_the_answer`. Batching
    the dispatch is worth far more than any graph-level fusion here: network-only
    RTF goes 36× → **1028×** on Metal from 1 to 32 frames per dispatch (20× → 378×
    MLX, 20× → 140× wgpu), because a single 16 ms frame is pure launch latency.
    End to end, Metal batched is 485× RT against 46× per-frame. This is only
    correct because the carry write-back was fixed upstream first.
  - Porting the reference's `f32` real FFT also **made it faster** than the
    generic `f64` complex transform it replaced: CPU 262× → 347× RT batched,
    and Metal 224× → 438×, the host DSP no longer being the bottleneck.
  - **Both graph shapes compile with zero missed fusion patterns**, pinned by
    `both_graph_shapes_are_fully_fused` via rlx's `assert_fusion_clean`. Two
    shapes had to change, because `rlx-fusion`'s two bias matchers differ on
    purpose: the matmul one reads the `Add` operand's rank directly (bias must be
    **bare rank-1** — `[1, n]`, or rank-1 behind an `Expand`, reports
    `BiasRankTooHigh` and leaves the chain unfused), while the conv one peels
    wrappers and *requires* `bias[C] → Reshape([1,C,1,1]) → Expand`. The LSTM
    cell also concatenates `x`/`h` so its two gate projections become one
    matmul — as two, the pass sees `add(matmul, matmul)` and fuses neither.
    Streaming throughput: CPU 237 → **328× RT**, Metal 4.1 → 6.3×, MLX 19 → 35×;
    batched wgpu 91 → 124×.
  - **Found: `Op::Lstm { carry: true }` silently does not advance state off the
    CPU.** Its contract is an in-place `hn`/`cn` writeback, but the Metal MSL
    kernel only *reads* `h0`/`c0`, MLX routes to a host path that does the same,
    wgpu likewise, and `unfuse_lstm` documents the gap outright. Wired into the
    streaming graph it scored `max|Δ| 0.52` with **64 decision flips** on
    Metal/MLX/wgpu while CPU stayed correct. This crate threads the state as
    ordinary graph inputs/outputs instead; the upstream op is left alone.
  - **The prebuilt `libten_vad` does not run its own `ten-vad.onnx`.** Its binary
    embeds the `coeff.h` DSP tables byte-for-byte, but carries the model's
    weights in no float layout, at no byte alignment, in no order (an `int8`
    correlation scan peaks at 0.33), and links no onnxruntime. It sits `9.6e-4`
    from the model shipped beside it — ~4000× further than this port. Decisions
    still agree; `shipped_library_decisions_agree` pins the gap so a change in it
    is visible. Evidence in `crates/rlx-ten-vad/tests/fixtures/README.md`.
  - **All 7 backends**, verified identical on cpu / metal / mlx / wgpu. The
    network is an rlx HIR graph in two shapes: streaming (one frame, LSTM state
    threaded as graph inputs/outputs, so no op beyond the universally supported
    set) and batched (a whole 30 s LSTM-reset window per dispatch via
    `Op::Lstm`). The two agree to `~2e-7`.
  - The DSP frontend (pre-emphasis, Hann-768 STFT, mel, biquad, the LPCNet-derived
    pitch tracker) is recursive and stays on the host. Streaming is fastest on
    **CPU** (280× real time vs 5-10× on GPUs — a 75 k-parameter per-frame
    dispatch is pure launch latency); batched CPU 262× / Metal 224× / wgpu 100× /
    MLX 73×, where the host DSP is the bottleneck.
  - `ten_vad.h` semantics are preserved: any hop ≥ 32, `probability = -1`
    before the first internal frame, `voice = probability > threshold`, LSTM
    state reset every 1875 frames.
  - **Licensing:** upstream is Apache-2.0 *with additional conditions* (a
    non-compete field-of-use clause) and the pitch estimator descends from
    Mozilla's LPCNet. Both carry over to this crate and its embedded weights —
    see [`crates/rlx-ten-vad/NOTICE`](crates/rlx-ten-vad/NOTICE) before
    redistributing. Powered by ten-vad.

- **`rlx-fireredaudio` — FireRedAudio unified audio language model.**
  FireRedTeam's general-purpose audio LM (Qwen3.5 ~9B backbone, Whisper-style
  Conv1d understanding encoder, RedAE + DiT generation). **ASR / understand run
  end-to-end** on RLX: mel → audio encoder HIR → Qwen3.5 host-embed prefill +
  greedy decode. ChatML task prompts match training character-for-character;
  acoustic edit templates and RedAE/patch rate helpers included. TTS / edit /
  voice-design APIs are present; RedAE+DiT graphs are next. Weights:
  [FireRedTeam/FireRedAudio](https://huggingface.co/FireRedTeam/FireRedAudio).

- **`rlx-s1` — S1-mini by Superwhisper, ASR transcript text normalization.**
  Raw ASR in, clean written text out: fillers removed, false starts and
  self-corrections resolved to what the speaker landed on, punctuation and
  capitalization applied, spoken numbers/dates/times/currency/emails rendered in
  written form. The checkpoint is a `Qwen/Qwen3-0.6B` fine-tune whose
  `config.json` is byte-identical to the base model's, so the forward pass is
  stock `rlx-qwen3` and the crate contributes the **input protocol** instead —
  which is what the model card spends most of its length on, and what
  integrations get wrong.
  - **Token-identical to `transformers` greedy on all 11 cases** (the card's
    worked examples plus one per control axis), checked in three layers: prompt
    string vs `apply_chat_template(..., enable_thinking=False)` byte-for-byte,
    prompt ids vs the HF tokenizer id-for-id, and completions vs
    `generate(do_sample=False)` token-for-token. Identical on CPU, Metal and MLX.
  - Reference must be dumped in **float32**, not the checkpoint's bf16: the two
    differ on near-tie argmaxes (bf16 drops the comma in `$23,450, and it's due`
    and turns the `Structure: lists` example back into prose). f32 is what
    matches both RLX and the card's own printed outputs.
  - The protocol is made unrepresentable-if-wrong rather than documented:
    `SYSTEM_PROMPT` verbatim as a `const`, the `[Styling] [Structure] [Context]`
    control line as three enums, and the `enable_thinking=False` prefix
    `<think>\n\n</think>\n\n` always emitted — omit it and the model returns
    nothing at all. Greedy is pinned; `max_new_tokens` is sized per call as
    `1.3 × prompt + 32`; transcripts past the ~1,000-token design point chunk at
    sentence boundaries, falling back to word boundaries because raw ASR usually
    has no punctuation to break on.
  - **No prefix cache**, deliberately: reusing a KV snapshot of the fixed
    ~60-token system prefix means replaying the transcript one token at a time
    through `feed_continuation`, measured 38.5 s vs 18.8 s on CPU for the same
    10 output tokens. A single batched prefill wins at every transcript length.
  - Steady-state ~0.2 s/utterance (~50 tok/s) on Metal. The first call at a
    given prompt length pays a 5–30 s graph compile, and `rlx-qwen3`'s prefill
    compile cache keys on the **exact** `(batch, seq)` — so a dictation pipeline,
    whose transcript lengths vary continuously, used to recompile on nearly
    every utterance for a 0.2 s forward pass. Hence the new prefill bucketing
    below, which `rlx-s1` turns on by default at a 64-token grid.

- **`Qwen3Generator::with_prefill_bucket` / `Qwen3RunnerBuilder::prefill_bucket`
  — one compiled prefill graph per length *range* instead of per length.**
  Rounds the prompt up to a multiple of `step`, right-pads `input_ids`, and
  gathers the LM-head row through a `last_token_idx` input (new
  `Qwen3PrefillOpts::last_token_from_input`) instead of a baked `seq - 1`.
  Off by default; `rlx-s1` opts in.
  - Output is unchanged, and the tests say so at both ends: a synthetic
    prefill/decode comparison against the exactly-sized path, and S1-mini's
    11-case HF parity suite still token-identical with bucketing on.
  - Numerically *equivalent*, not bit-identical — the padded run is a wider
    GEMM and reduces in a different order (~1e-7 relative). Pad **columns**
    contribute exactly zero (causal mask → `exp(-inf)`), and the pad KV **rows**
    are trimmed before they reach the cache, which the test pins by comparing
    cache lengths as well as contents.

- **`rlx-neuralhash` — Apple NeuralHash perceptual image hashing.** 360×360 →
  128-float descriptor → `[96, 128]` seed projection → 96-bit hash, matching the
  [reference implementation](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX)'s
  `nnhash.py` output. The architecture is read natively from the vendor's
  Espresso container that macOS installs in `Vision.framework` — 225 layers,
  MobileNetV3-shaped with **instance** norm (Espresso spells it `batchnorm` with
  `training_instancenorm`), hard-swish written out as four elementwise ops, and
  squeeze-excite gates. `espresso` parses the container (LZFSE `pbze` decoded
  inline via `libcompression`), `spec` normalizes it to a serializable op list,
  `flow` emits rlx-ir. ONNX is a validation-only path behind `onnx-parity`;
  the default build has no ONNX dependency. No model data is shipped or
  redistributed.
  - **Bit-identical (96/96) against an independent PyTorch implementation** of
    the same container on every image tried, and across all 7 backends.
    Preprocessing is 388 729/388 800 elements bit-exact vs Pillow.
  - Four container details each produce a well-formed but *wrong* hash if
    misread, and are pinned by tests: instance-norm parameters are interleaved
    `[γ, β, mean, var]` per channel (45 bits if read as contiguous blocks),
    `training_instancenorm` means runtime statistics (28 bits), `avg_or_max: 0`
    is **average** (53 bits), and `pad_mode` overrides the explicit `pad_*`
    fields, which the shipping model writes as all-zero.
  - Emitter is 460 rlx nodes rather than the naive 1109 — instance norm as one
    `GroupNorm`, implicit broadcast instead of materialized `Expand`, and the
    hard-swish chains fused onto native `HardSwish`/`HardSigmoid`. **2.9× cpu,
    1.5× metal**, hash-identical (`--no-fuse` A/Bs it).

- **`rlx-tada` — HumeAI TADA (Text-Acoustic Dual Alignment) zero-shot voice cloning.**
  A forced aligner gives every text token exactly one 50 Hz frame, so text and
  audio ride a single autoregressive stream 1:1; a DiT-style head then integrates
  a flow-matching ODE that emits acoustic latent **and** duration jointly as one
  528-wide vector (512 acoustic ‖ 2 × 8-bit Gray-coded frame gaps). Llama-3.2
  backbone via `rlx-llama32`, DAC codec via `rlx-dac`, no ONNX Runtime anywhere.
  - Validated stage-by-stage against upstream torch on the real checkpoint:
    prompt token ids and positions identical, 26 × 512 prompt latents to 1.8e-5,
    prefill embeddings bit-exact, prefill hidden 7e-6, decode hidden 1e-7, the
    solve 4e-5.
  - **CPU, Metal, MLX, wgpu, Vulkan and CoreML all agree with CPU at cosine
    1.000**; CUDA/ROCm compile but are untested here. Bringing wgpu up re-found a
    previously fixed upstream defect — `rlx-wgpu` had lost the `!src_is_weight`
    guard on deferred host→device uploads, which is silently wrong rather than
    loud.
  - The whole ODE — every Euler step, both classifier-free-guidance branches, the
    guidance blend — lowers to **one** graph per token, since the timesteps and
    guidance scales come from the schedule rather than the data and fold into
    constants. Ten head evaluations become one `run`.
  - RTF 0.17× → 0.33× (MLX reaches 1.10× on long utterances) and peak RSS
    18.7 GB → 10.1 GB. The two dominant costs were a naive weight transpose (now
    cache-blocked and parallel in `rlx-models-core`, which helps every crate) and
    mmap-charged RSS (the checkpoint reader uses `pread`, so pages are never
    charged to the process at arena-allocation time).
  - **The `_decoder.*` weights bundled in `tada-1b` are a decoy** — 195 of their
    201 tensors differ from the published `HumeAI/tada-codec/decoder`, and
    upstream's `from_pretrained` quietly fetches the real one instead. They
    produce latents correct to 1e-5 and audio that transcribes as a single
    syllable, so loading them is an error with an explanatory message rather than
    a silent fallback.

### Voice cloning — reference hygiene, runaway guard, and a fidelity measurement

Ported from [jamiepine/voicebox](https://github.com/jamiepine/voicebox), whose
cloning pipeline has had these in front of real users.

- **`rlx_core::voice_clone` — shared reference-audio hygiene.** DC-offset
  removal, edge-silence trim, edge re-padding and a peak cap, then validation
  against duration and RMS limits. Every cloner in the workspace previously
  handed the user's clip straight to an encoder, and none of that is visible to
  a numerical parity test: the port matches the reference implementation exactly
  and still clones badly. Two constants carry their rationale — the trim runs at
  40 dB rather than librosa's 60 (40 sits below speech's ~30 dB dynamic range,
  so soft trailing syllables survive), and the edge re-pad is skipped unless
  trimming actually shortened the clip, so a clip near the duration ceiling is
  not padded over it and then rejected as too long.
- **Runaway detection.** `[speech][>1 s internal silence][more speech]` is a
  reliable signature of a model that missed its stop condition and resumed with
  hallucinated speech or codec noise. `has_tts_runaway` detects it and
  `trim_tts_output` cuts at the boundary, trims the trailing silence and applies
  a 30 ms cosine fade. `rlx-tada` now runs this by default (`--keep-runaway`
  opts out); leading silence is left to TADA's own predicted gap.
- **Band-limited reference detection.** `rlx_core::voice_clone::looks_band_limited`
  flags a reference whose energy dies below the codec's band (<1% above 6 kHz,
  measured with a dependency-free biquad rather than an FFT). This is the single
  biggest predictor of a poor TADA clone and nothing else in the pipeline sees
  it: the clip is not quiet, not clipped, not short, and aligns perfectly.
  `assets/jfk/jfk_voice_clone.wav` is a 1961 archival excerpt with 99% of its
  energy below 2.5 kHz and 0.195% above 6 kHz, against 10.6% for a studio clip.

  This corrects a claim made earlier in this file. Measuring both references
  rather than one **inverts the ranking**:

  | reference | rlx-tada | rlx-chatterbox |
  |-----------|----------|----------------|
  | studio, full-band | **0.9479** | 0.9394 |
  | archival, band-limited | 0.8621 | **0.9263** |

  TADA is slightly *ahead* on a clean reference and less robust to a narrowband
  one — it conditions on a sparse set of full-band codec latents (one frame per
  text token), where ChatterBox runs a dedicated speaker encoder over the whole
  clip.

- **Solver tuning, measured and then rejected.** Raising `--cfg` from upstream's
  1.6 to 3.0 looked like a clean win on the JFK reference — mean speaker cosine
  0.8203 → 0.8446 with the spread collapsing from 0.085 to 0.019, and Whisper
  still transcribing correctly. On a second speaker it was **worse** (0.9351 →
  0.9256), so the default stays at upstream's 1.6. `latent_noise_std` was
  checked the same way and upstream's 0.5 is genuinely optimal (0 → 0.8045,
  0.25 → 0.8141, 0.5 → 0.8203). The encoder emits a single 512-wide latent
  (`hidden_linear.weight` is `[512, 1024]`, not a mean/logvar pair), so that
  noise is an external constant and not a learned bottleneck variance.

- **`rlx-tada` now scores how well the prompt transcript matches its audio.**
  The aligner is a *forced* aligner: it places every token somewhere regardless,
  so a transcript that does not describe the recording yields an alignment that
  looks structurally perfect and means nothing. `Alignment::mean_token_logprob`
  is the mean log-softmax probability the CTC head gives each token at the frame
  it was aligned to, and it separates the cases by an order of magnitude — the
  JFK clip's true transcript scores **-0.34**, the same text with five words of
  preamble that are not in the recording **-4.66**, an unrelated sentence
  **-16.12**. `looks_mismatched()` warns below -1.5; the score is carried in the
  `.tadaprompt` (`#[serde(default)]`, so older prompts still load and report
  "not recorded") and shown by `rlx-tada info`.

  This was not hypothetical: **the repo's own harness prompt was built from the
  wrong transcript.** `assets/jfk/jfk_voice_clone.wav` is the 5.2 s excerpt
  *without* the "And so my fellow Americans," preamble — confirmed by
  whisper-base.en and whisper-small.en independently — and `tada-harness-prompt`
  had been passing it anyway since the crate was written. Recipe fixed and the
  prompt regenerated (26 tokens → 19, score -0.34).

  Fixing it is worth doing for its own sake, but it is honest to say what it did
  *not* buy: over three utterances the corrected transcript measured 0.8203 mean
  speaker cosine against the wrong one's 0.8328, ranges overlapping — noise, not
  an improvement — and Whisper transcribes both outputs correctly. TADA's timbre
  comes from acoustic latents gathered at the aligned frames, and those frames
  are the same speaker either way; the alignment matters for prosody and
  duration, not identity. The value here is the *detector*, not the delta.

- **`rlx-tada` multi-sample prompts.** `PromptBuilder::build_multi` conditions on
  several clips of one speaker — each cleaned and validated on its own, then
  concatenated with the transcripts joined in order. `rlx-tada prompt` takes
  repeated `--wav`/`--text` pairs. `--raw-reference` skips cleanup entirely, for
  reproducing a known-good prompt byte-for-byte.
- **`rlx-wespeaker --example clone_fidelity` — does the clone sound like the
  speaker?** Nothing in the workspace measured this. Parity suites answer "does
  the port match the reference" and Whisper answers "are the words right";
  neither answers the question a voice cloner exists to answer. This embeds the
  reference and each synthesis and reports the speaker cosine (> 0.7 is "same
  speaker" on the VoxCeleb convention for x-vector systems).

Measured on TADA, 3 utterances per condition, against the JFK reference clip:

| reference | raw | with hygiene |
|-----------|-----|--------------|
| clean (curated asset) | 0.853 mean (0.817–0.887) | 0.833 mean (0.813–0.844) |
| degraded (DC offset, clipped, 1.5 s room tone per edge) | 0.642 mean, **0.480 worst** | 0.726 mean, **0.653 worst** |

Hygiene is worth ~+0.08 mean on a realistic bad upload and lifts its worst case
out of "identity did not transfer"; the apparent loss on an already-clean clip
sits inside run-to-run spread. Hence on by default.

**`rlx-wespeaker` was returning a constant embedding, and this is how it was
found.** Its output was byte-identical for speech, white noise and a pure tone —
the network never saw its input — and every existing check passed, because
comparing a constant against itself across five backends is perfectly
self-consistent.

The cause was in `rlx-onnx-import`, not in the crate: `conv_output_dims` kept
only `xs.last()` as *the* spatial axis and always returned rank 3, so a genuine
2-D conv lost its height. A 3x3 stride-1 pad-1 conv on `[1, 1, 80, 148]` came
out as `[1, 32, 1, 148]` and the entire ResNet ran on a one-bin spectrogram; the
`Reshape → [1, 2560, 19]` before the stats pooling was then correctly *rejected*
(its element count no longer matched), so the pooling reduced frequency instead
of time. Only the debug-build IR verifier objected
(`MatMul: matmul K mismatch: 19 vs 5120`) — release skips it and shipped the
constant.

Fixed with a rank-4 branch that computes both spatial axes per-axis, gated on
rank-4 input *and* a rank-4 weight so the rank-3 Conv1d / BLC / NCL heuristics
are untouched, and mirroring the existing stride-1 pass-through (the
`explicit Pad → VALID conv` pattern leaves the pad out of the attrs). Verified
across all 20 importer-consuming crates: no regressions. `rlx-diarize` clusters
on this embedder and was affected too.

`tests/embedding_discriminates.rs` now asserts the property that was violated —
different audio must give different, and distant, embeddings — rather than
numbers. The shipped `graphs/wespeaker.rlxp` freezes the import and was re-packed.

Note the native graph bakes `FIXED_FRAMES = 148` (~1.5 s), so it truncates long
clips and its speaker estimates are correspondingly noisy; the table above uses
the full-clip ONNX Runtime reference (`--ort`). Both embedders agree on the
direction of every comparison.

**ChatterBox's `speech_encoder` now compiles — and is still numerically wrong.**
It previously could not be imported at all (`Binary(Sub) declares [1, 128, 128]
but its operands give [1, 128, 64]`), so native reference-audio conditioning was
dead. Two more upstream defects, both of the same family as the conv one:

- `conv_pool.rs` already had a `meta_len_stale` detector that recomputes a conv's
  output from its concrete HIR input when the recorded meta disagrees, and a
  "genuine 2D forward conv" branch that computes both spatial axes. The detector
  was gated `rank0 == 3`, so a rank-4 conv trusted a stale meta and that branch
  was never reached. A correct 742-frame mel `[1, 1, 80, 742]` went into a
  stride-1 3x3 conv and came out declared `[1, 32, 80, 128]`, collapsing time for
  the whole ResNet. Added `meta_len_stale_2d`.
- `norm.rs` — `BatchNormalization` took its output shape from the meta. It is
  elementwise, so its output shape *is* its input shape; it now reads the
  concrete HIR input.

The frame relation was measured, not guessed: exposing the intermediates and
running the real graph under ONNX Runtime at 2.00 / 3.00 / 5.20 / 7.44 s gives
198 / 298 / 518 / 742 frames — a 240-sample hop at 24 kHz, `T = n/240 - 2`. After
the fixes the conv chain reads `[1, 320, 1, 742] -> [1, 128, 1, 371]`, matching
the reference exactly, and the encoder runs in ~780 ms.

**Then two more, found by bisecting against the reference tensor by tensor**
(via the importer's own `RLX_ONNX_TAP`, which appends named ONNX tensors as
extra graph outputs). Everything matched to 1e-5 down to the CAM layers, where
the shapes diverged: native carried 200 frames where the truth was 259.

- **`ceil_mode` was never read anywhere in the importer.** ONNX `ceil_mode = 1`
  rounds the pooling window count UP. The speaker encoder pools 259 frames with
  kernel/stride 100: ceil gives 3 windows, floor gives 2, and the CAM layer then
  expands the pooled segments back by 100 into a 200-frame tensor. 52 pooling
  nodes in this one graph. Added `pool_out_len`/`pool_ceil_mode` in
  `conv_pool.rs`.
- That exposed two latent bugs in the CPU pooling kernel, both silent. Its
  no-padding fast path assumed every window is in bounds — false once a
  `ceil_mode` window overhangs the end (the third window spans 200..300 of a
  259-frame input), so it indexed past the buffer. And the mean divided by the
  nominal window size, scaling that last window by 59/100. ONNX's default is
  `count_include_pad = 0`, and an overhang is not padding: those positions do
  not exist. Fixed both; `rlx-cpu/tests/pool_ceil_mode_overhang.rs` pins it.

**`speaker_embedding()` now matches onnxruntime exactly** — cosine 1.00000000 on
both speech clips (max|Δ| 5e-6 and 2.2e-5), and the speaker relationships track
the reference:

| pair | before any fix | after | onnxruntime |
|------|---------------|-------|-------------|
| default_voice vs jfk | 0.908 | **0.4382** | 0.4382 |
| default_voice vs tone | 0.740 | **0.0481** | 0.0480 |
| jfk vs tone | 0.690 | **0.1639** | 0.1640 |

(A pure tone sits at cosine 0.99999985 rather than 1.0 — near-degenerate
activations, the usual ill-conditioned case.)

Nothing regressed: all 18 importer-consuming crates pass, plus rlx-cpu (318) and
rlx-onnx-import (45); the one `kitten_tts_mini_rlx::add1_coexec` failure predates
this work and was confirmed by reverting. WeSpeaker still scores tone 0.174 /
clone 0.819, and rlx-tada passes on all six backends.

**A sixth, in `rlx-chatterbox` itself, found while verifying the fifth.** The
on-disk AOT key was `cb_{component}_{device}_s{seq}` — but `speech_encoder` is
always compiled at `seq = 100` and takes its real extent from the reference
clip's sample count, which the key never named. Clips of different durations
therefore built different graphs and then shared one cache entry: **the first
voice cloned in a session served its compiled graph to every later one.**
Measured cold against onnxruntime, the first clip scored cosine 1.00000000 and
the next two 0.992 and 0.569 — wrong in a way that reads as a mediocre model
rather than a bug. The key now carries `max_wav` and the per-component `named`
lengths; all three clips are exact.
`tests/speaker_embedding_cache_key.rs` pins it via order-independence (embed two
different-length clips in one order, wipe the cache, embed them in the other),
which needs no reference runtime; it fails at cosine 0.639 if the key regresses.

An audit of every `compile_hir_cached` key in the workspace found this to be the
only one missing a shape-determining input — `rlx-tada` (buckets + batch +
checkpoint tag), `rlx-tiny-tts` (length + named + opt flags), `rlx-inflect-nano`,
`rlx-orpheus`, `rlx-parlertts`, `rlx-sanotts` and `rlx-soprano` all name theirs.

**ChatterBox voice cloning works end to end, verified on both axes it can fail
on.** Synthesizing from the JFK reference gives a **0.9263 speaker cosine**
against that reference — higher than TADA's 0.844 on the same clip — and
Whisper transcribes the output as exactly the requested text ("The quick brown
fox jumps over the lazy dog."). Either number alone is insufficient: a high
speaker cosine can be speaker-tinted babble, and a clean transcript can be the
wrong voice. RTF is 0.04x on CPU (3.32 s of audio in 84 s), so it is correct but
slow — the CFM solver is 10.4 s of it and the AR loop most of the rest.

Two tests now guard the encoder, and neither needs a reference runtime at test
time:
`tests/speech_encoder_parity.rs` rebuilds a deterministic probe signal (two tones
plus an integer-LCG dither — a pure tone is a bad probe, its activations are
near-degenerate) and compares against a 4 KB fixture of onnxruntime's embedding;
it fails at cosine 0.969 if a couple of components are perturbed.
`tests/speaker_embedding_cache_key.rs` covers the ordering bug above.

Worth naming the shape of this: six defects in a row, each hidden by the one in
front of it, and every single one silent. A constant embedding that no test
could fail, a shape guessed where it could have been computed, a fast path whose
comment asserted an invariant it did not check. The checks that caught them were
the ones that compared against something external — the reference runtime — not
against ourselves.

### New crates

- **`rlx-jlens` — the Jacobian lens.** `lens_l(h) = unembed(J_l·h)` with
  `J_l = E[∂h_final/∂h_l]`: reads out what an activation is disposed to make the
  model *say*, transporting it into the final-layer basis rather than decoding it
  as if the remaining layers were the identity. A native port of the reference
  for *Verbalizable Representations Form a Global Workspace in Language Models*,
  **validated against it entry-for-entry (relF 1.3–2.8e-6)** — a comparison that
  immediately caught a one-layer labelling bug invisible to every
  self-consistency check, because rlx was perfectly consistent with itself.
  - Model-agnostic behind one `LensModel` trait; four implementations spanning
    both axes it abstracts — `qwen35` (hybrid delta-net, GGUF, materialized
    weights), `qwen3` (dense attention, HF safetensors, opens its own),
    `dinov3` (ViT), `qwen25_vl` (vision-language).
  - Fits on CPU/Metal/MLX/CUDA/ROCm, all agreeing with CPU to ~1e-7; applying a
    fitted lens is a forward pass plus a `d × d` matvec, so any backend can read
    through one.
  - `examples/vl_report.rs` is one command for a vision-language model:
    per-layer attention, per-word image masks, patch segmentation and the
    transported word readout, from a single model load and fit.

### Backend fixes

- **wgpu returned all-zero logits for any F32 LM whose arena crosses 4 GiB.**
  Qwen3-0.6B safetensors emitted token id 0 (`!`) forever at prompt lengths
  ≥ 92 — the point where activations + params exceed the storage-bind cap and
  `Arena::from_plan_split` moves every param into a separate weight buffer.
  Bisected to a 91-vs-92 token boundary; the same model at 91 tokens was
  correct. Four defects, each on its own sufficient to zero the output:
  - `arena_off_in_bind_window` short-circuited on "the whole act arena fits one
    binding" **before** checking whether the tensor was in the *other* buffer.
    Those two conditions are independent, so every weight read took the
    shortcut. The offset then went through `arena_local_off_f32`, where the
    `WEIGHT_BUF_TAG` (bit 62) vanishes in the `as u32` cast and a garbage
    offset comes out looking like a small, plausible index — which is why this
    failed silently rather than panicking. That helper now asserts instead.
  - Staged weight copies were queued as *deferred* host-mirror writes. The
    staging scratch is bump-allocated and wraps, so many params share one
    destination, and the mirror is keyed by destination — deferring collapsed
    the sequence to whichever copy came last. They are now written eagerly.
  - `Op::Gather` only routed to the weight-buffer-aware `run_gather_split` on
    virtually-sharded arenas, missing the case where the embedding table alone
    was parked.
  - `from_plan_split` parked params larger than the staging reserve, which no
    generic op can ever reach (a 622 MiB tied `lm_head` vs a 64 MiB reserve).
    Such params now stay in the act arena while it still fits one bind window.

  wgpu is now token-identical to CPU on Qwen3-0.6B at every prompt length
  tried (1 … 300), and `rlx-s1`'s cross-backend test passes on wgpu alongside
  CPU / Metal / MLX / CoreML.
- **`Qwen3Runner`'s Metal tuning knobs leaked into every runner built after
  them.** The builder auto-enables five Metal-only optimizations through
  `rlx_ir::env::set`, whose overrides are process-global and were never cleared.
  One of them, `RLX_QWEN3_INPLACE_KV`, emits `Op::KvAppend` — which MLX has no
  kernel for — so building a Metal runner and then an MLX one in the same
  process failed to compile with "`Backend::supported_ops()` must include each
  kind" listing 56 `KvAppend` nodes. Found by a cross-backend test that runs
  CPU → Metal → MLX in one binary; each backend passed alone. The knobs are now
  set for Metal and unset elsewhere, tracking exactly which keys the builder set
  so a caller's own override (or a real `RLX_*` env var) is never touched.
- **`AttentionBackward` silently returned zero gradients for `head_dim > 128`**
  on CUDA and ROCm (`rlx-gpu-kernels/kernels/attention_bwd.cu`). The kernel's
  early return fired and wrote *nothing*, so training proceeded and quietly
  learned nothing from those tensors — Qwen3.5's `head_dim` 256 came back as
  relF 1.0. Only one object was actually bounded (`acc[MAX_HEAD_DIM]`, a
  per-thread accumulator in the dK/dV fast path); it is now swept in tiles, so
  `head_dim <= 128` is byte-identical to before and 256 matches CPU to ~9e-7 on
  both backends. The same return also fires for `seq > 512`, which had **no**
  host guard at all; both backends now assert that bound explicitly.
- **`rlx-qwen25-vl` mRoPE decode positions ran off the end of the prompt.**
  `decode_step` indexed the prompt's sections by `past_seq - 1`, in range only
  for the first decoded token; from the second it fell back to `past_seq + 1`.
  Image tokens occupy a *grid* — 299 image tokens span ~23 positions, not 299 —
  so multimodal generation jumped ~275 positions past the prompt and collapsed
  after one good token. Text-only was off by a constant one, which preserves the
  relative distances RoPE encodes and so looked fine, which is why this survived.
- **`rlx-qwen25-vl` dropped Q/K/V bias, mis-sized every image, and emitted
  malformed ChatML.** `attention_bias` defaulted to `false` though Qwen 2 applies
  the bias unconditionally (HF has no flag, so `config.json` says nothing while
  the checkpoint ships the tensors); `image_min_pixels` defaulted to a
  1024-*token* floor that `smart_resize` upscales *to*, turning a 299-token photo
  into 1032; and `qwen25_vl_chatml` closed turns with a bare newline instead of
  `<|im_end|>`. Text-only is now bit-exact against HF (logits cosine 1.000000)
  and an image prompt matches at 0.999357 with the same top-1. None of it was
  caught because every test in the crate was synthetic — both sides of a
  self-consistency check were built from the same wrong config.
- **`rlx-rocm` host-side arena offsets widened to 64-bit** (159 `Step` fields and
  their construction sites). The 4 GiB assert stays: 479 shared-kernel offset
  parameters are still `unsigned int` against 115 `unsigned long long`, so an op
  of the first kind truncates inside the kernel signature regardless of host
  width, and letting the safe ops through while the rest wrapped silently would
  be worse than stopping.
- Two `rlx-wgpu` bugs found by the `rlx-neuralhash` port and fixed upstream in `rlx`
  (see that repo's CHANGELOG): `Op::GroupNorm` lowered to a whole-arena
  device→host→device staging step — now a native WGSL kernel, **88× on a
  35-norm model (1096 ms → 12.4 ms)** — and whole-arena host steps never
  invalidated `HostTensorCache`, so a following cache-aware host step could
  serve a stale pre-step copy. The same whole-arena staging antipattern cost
  97% of CUDA prefill in `group_limited_gate` at 0.2.14.

### Tooling

- `scripts/publish.sh` only requires a publish tier for crates that are
  actually publishable — it now reads `publish` from `cargo metadata` instead
  of expecting every `publish = false` workspace member to be hand-listed in
  `SKIPPED`. `--list` had been failing outright on six of them.

## 0.2.14 — MXFP4 quantization, backend fixes & release hardening (2026-08-12)

### MXFP4 quantization (produce side)

- **`rlx_models_core::mxfp4_pack` — rlx's first f32 → MXFP4 encoder.** Every
  other MXFP4 path in the tree was consume-side, written for checkpoints that
  ship already quantized (mlx-community, Kimi). This packs E2M1 nibbles plus a
  per-group E8M0 scale, so an ordinary bf16/f32 HF checkpoint can drive the same
  packed kernels. Group exponent is the smallest `e` with `6·2^e >= amax`, which
  makes saturation impossible (OCP's `floor(log2(amax)) - 2` clamps the top
  quarter of its range). Gate is `tests/mxfp4_pack_ops.rs`, which feeds the
  packed bytes to the real ops rather than only to the encoder's own
  `dequantize` — a shared misreading of the layout cannot pass it.
- **`rlx-ling --mxfp4`** quantizes the whole model at load time: arena
  29.5 → ~4.0 GiB, steady RSS 21.6 → 8.2 GB, and Ling-3.0-tiny now **fits a
  16 GB CUDA card**, which f32 could not (`device allocation failed for
  7909017552 f32 (29.463 GiB)`). `QuantPlan` splits the LM head out because its
  4-bit error lands undiluted on the logits (3.1e-2 vs 1.9e-3 for the body);
  `--f32-head` trades 0.85 GiB for that. The token embedding stays f32 — it is
  gathered, not multiplied, and rlx has no MXFP4 gather.
- `DeepseekMoeDims::mxfp4_group` runs the routed experts as
  `Op::DequantGroupedMatMulMlx`, shared by rlx-ling / rlx-deepseek / rlx-kimi-k3
  / rlx-glm4moe.

### Backend fixes

- **The `group_limited_gate` host delegate copied the ENTIRE arena
  device→host→device** to compute a top-k over a few thousand floats. On
  Ling-3.0-tiny that was ~276 GB of PCIe traffic and **97% of CUDA prefill**
  (61.8 s of 63.5 s). It now stages only the ~70 KB it touches. The cost scaled
  with *arena size, not problem size*, so it was invisible on small models and
  worst on the ones big enough to need a GPU; every MoE crate on that op was
  paying it.
- **CUDA MXFP4 grouped matmul, 22×** (`gate_up` m=64: 10.33 → 0.46 ms, 110 GB/s):
  new split-K kernel. The old one issued one 32-bit load per *nibble* and gave
  one thread per output, so a warp's lanes read weight rows `k/2` bytes apart —
  fully uncoalesced. Also slightly *more* accurate (tree reduction).
- **CUDA dense MXFP4 GEMM, 1.4×**: it staged X through shared memory where each
  thread wrote and read back its own slot — a no-op round-trip costing 8 KB of
  occupancy-limiting shared memory and a `__syncthreads()` per K-chunk.
- Together: **CUDA Ling prefill 63.4 s → 0.266 s (238×), 1.0 → 240.9 tok/s.**
- **Metal MXFP4, 1.25×** (Ling prefill 45.4 → 56.7 tok/s): the same no-op
  threadgroup staging in both `dequant_matmul_mlx_gemm` and
  `grouped_dequant_matmul_mlx_gemm` (45.4 → 50.9), plus staging the activation as
  `half` with an f32 accumulator (50.9 → 56.7).
- **wgpu arena overrun**: a non-matmul `set_param_typed(BF16)` param was widened
  to f32 and written `ne*4` bytes into an `ne*2` slot (`plan_f32_uniform` keeps
  non-F32 *params* native), corrupting the following param and disagreeing with
  host steps that read the slot as bf16.
- Deprecated `rlx_cpu::llada2_gate::execute_gate_in_f32_arena` (removal in 0.3).
  Its whole-arena offset signature is what made the wasteful staging above the
  natural thing to write; `execute_gate_f32` takes plain slices and is the entry
  point now. It has no remaining callers, but it shipped in the published 0.2.13
  API, so it stays until a major bump.

### Tooling

- `rlx-models-core/examples/mxfp4_grouped_bench` times the grouped and dense
  MXFP4 ops standalone at real MoE shapes — seconds per kernel iteration instead
  of a whole-model prefill. Use it before touching any MXFP4 kernel.

### New model crates

- **`rlx-motif` — Motif-3** ([Motif-Technologies/Motif-3](https://huggingface.co/Motif-Technologies/Motif-3),
  `model_type = "Motif"`): 53 layers, ~314 B parameters, 262 144 context. Three
  pieces with no prior analogue in the workspace:
  - **GDLA** (grouped differential latent attention) — MLA-style low-rank Q/KV
    with one shared RoPE head, 80 heads in bundles of 5 where the last head of
    each bundle is *subtracted* with an input-dependent λ, plus an element-wise
    sigmoid output gate. 3 layers in 4 are 128-key sliding-window on their own
    RoPE base; the rest are global with YaRN and `mscale²` on the softmax scale.
  - **MHC** (manifold-constrained hyper-connections) — four parallel residual
    streams mixed per sublayer by a doubly stochastic 4×4 matrix from 20 inline
    Sinkhorn iterations.
  - **PolyNorm MoE** — a trainable polynomial activation with *per-expert*
    coefficients across 384 experts. Folding `σ(weight)`/bias-clamp host-side
    turns those into a table the graph gathers by routed expert id, so each
    top-k slot stays one `GroupedMatMul`; the reference has to fall back to an
    eager Python loop over experts for exactly this reason.

  Prefill graph, no real-weight run (629 GB / 155 shards). 30 tests — host
  references for each block plus full-graph causality — green on **all 7
  backends + CoreML** across mac / RTX 3080 Ti / MI100. Linux wgpu needs
  `RLX_ARENA_NO_REUSE=1` for the pre-existing `rlx-wgpu` slot-reuse corruption.

### Performance

- **The mel filterbank was dense; it is now banded.** The `[40, 513]` matrix is
  20,520 coefficients of which about a thousand are non-zero — each triangle
  touches one contiguous stretch of bins — and the frontend multiplied through
  all of them. Skipping the exact `+0.0` entries is bit-identical (the
  accumulator is non-negative, so adding `+0.0` changes nothing) and the whole
  frontend still matches the upstream C on 10,250/10,250 values.

  | | before | after |
  |---|---|---|
  | mel + window + log + normalise | 7.80 us/frame | **0.14 us** |
  | frontend | 19.2 us/frame | **11.5 us** |
  | full pipeline (CPU) | 399x realtime | **490x** |
  | MCU pipeline (`rv32imc`) | 15.3 M cycles/hop | **13.5 M** |

  The MCU gains more than the arithmetic suggests: each eliminated multiply was
  a soft-float call. The same fix went upstream as `rlx_ir::audio::MelBands`,
  where it is 10.4x on `Op::LogMel`, and `rlx-conformer-ctc` now shares it.

- **Structured pruning + QAT.** `qat.rs` gained `--prune`/`--prune-mode`.
  Dropping whole *taps* — rows of a `[taps, outputs]` weight, scored by L2 norm
  — removes MACs contiguously, with nothing to index around, unlike the
  activation-sparsity attempt below. Held-out flips of 2,304, against the f32
  teacher:

  | taps dropped | MACs removed | before fine-tuning | after QAT |
  |---|---|---|---|
  | 10% | 8.9% | 260 | **88** |
  | 20% | 18.1% | 369 | **169** |
  | 30% | 27.7% | 424 | **137** |

  QAT recovers 54–68% of the damage, and the remainder is still a 3.8–7.3%
  disagreement rate where dense is 0. Worth it only if the application can
  spend that; for a port claiming parity with the reference it is not.

- **`examples/mac_budget.rs`** accounts for the remaining 79,295 MACs per frame
  and tests the levers that would cut them. Two are dead: **0.00%** of the
  weights are exactly zero, and no matrix is low-rank enough to factor —
  `lstm1 [144, 256]` needs rank 109 to keep 99% of its energy against a
  break-even of 92, so `U·V` would be **1.18x more expensive** than the dense
  product. A third is real but unclaimed: 56% of the conv stack's output is
  exactly zero after its ReLU, worth 14.5% of the frame's MACs, but exploiting
  it by index list made things *slower* (integer net 200 k -> 222 k cycles) —
  the indirection costs more than the multiply it skips. Capturing it needs
  `W_ih` stored input-major so the skip stays contiguous, which is what the
  FPGA datapath already does.

### Changed

- `rlx-deepseek`'s MoE emitter now builds its expert GEMMs with
  `HirGraphExt::grouped_matmul`, which derives the output shape from the
  operands instead of taking a hand-written one. That is what rejects an expert
  bank still in the checkpoint's `[E, N, K]` order — previously a silent
  partial write. Needs upstream RLX with `rlx_ir::shape::grouped_matmul_dims`.

### Release hardening & repo hygiene

Workspace `[workspace.package].version` = **0.2.14**, pinned to upstream
**`rlx*`** **0.2.14** on crates.io (`rlx-runtime`, `rlx-ir`, `rlx-flow`, …).
Requires RLX **0.2.14** published from
[MIT-RLX/rlx](https://github.com/MIT-RLX/rlx) first. Minimum supported Rust
version is **1.89**, matching upstream `rlx*` 0.2.14.

#### Notable changes

- **Release hygiene: the whole workspace is `fmt`- and `clippy`-clean.**
  `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D
  warnings` now pass across every crate. Besides formatting, this fixed ~50
  `clippy` findings (`manual_is_multiple_of`, `manual_checked_ops`,
  `unnecessary_cast`, `field_reassign_with_default`, `ptr_arg`, `needless_return`,
  `redundant_clone`, `repeat().take()` → `repeat_n`, …) and several examples/tests
  that had drifted from current APIs: six Gemma parity tests were missing newer
  `GemmaConfig` fields; the `gemma4_e2b` backend-parity test now uploads packed
  weights through the `PackedSrc` enum (`Owned`/`Borrow`/`F32`) like production;
  and the `backend_sweep` (`runner`) / `cmp_ort` (`onnx`) examples gained
  `required-features` so `--all-targets` skips them under default features instead
  of failing to compile.
- **Repo layout:** `rlx-tiny` / `rlx-tinystories` now default their trained
  `.rlxts` checkpoints under `weights/<model>/` (`weights/tinystories/…`,
  `weights/tiny/…`) rather than the repo root, and `checkpoint::save` creates the
  parent directory. Generated `memory_probe` / retention benchmark output dirs are
  consolidated under a git-ignored `bench_out/`.

## 0.2.8 — model coverage expansion (2026-06-21)

Workspace `[workspace.package].version` = **0.2.8**, pinned to upstream
**`rlx*`** **0.2.8** on crates.io (`rlx-runtime`, `rlx-ir`, `rlx-flow`, …).
Requires RLX **0.2.8** published from
[MIT-RLX/rlx](https://github.com/MIT-RLX/rlx) first.

### New model crates

Audio codecs: `rlx-snac`, `rlx-encodec`, `rlx-speechtokenizer`,
`rlx-wavtokenizer`, `rlx-xcodec`, `rlx-facodec`, `rlx-nanocodec`,
`rlx-mimi`, `rlx-dac`, `rlx-tsac`.

ASR / audio: `rlx-wav2vec2-asr`, `rlx-nemotron-asr`, `rlx-qwen3-asr`,
`rlx-funasr`, `rlx-diarize`, `rlx-aec`.

TTS / speech: `rlx-orpheus`, `rlx-kyutai-tts`, `rlx-pocket-tts`,
`rlx-inflect-nano`, `rlx-tiny-tts`, `rlx-vibevoice`, `rlx-moshi`.

Vision / VLM: `rlx-bioclip2`, `rlx-florence2`, `rlx-grounding-dino`.

LM: `rlx-eagle3`.

### Notable changes

- **Qwen3.6-27B-MTP-GGUF** (`qwen35` arch, `unsloth/Qwen3.6-27B-MTP-GGUF`) text
  generation is now coherent and matches llama.cpp. Fixed two GatedDeltaNet bugs
  in `rlx-qwen35`: (1) the decay gate applied a spurious `-exp()` to `ssm_a`,
  which the GGUF already stores as `-exp(A_log)` — collapsing the recurrent
  state; (2) the GQA q/k head expansion (16→48) used *interleave* instead of
  *tile*, flipping the sign of every middle head's output. Also fixed the Metal
  Q3_K dequant (`dequant_gguf.msl` was dropping 8 of 16 sub-block scales — fixes
  Q3_K for all models) and the qwen3vl vision `mmproj` (CLIP merger) loader.
- Removed the `rlx-tensor-host` crate (the host-kernel shim that existed only
  to dodge a crates.io name clash with the framework's `rlx-tensor`). Its host
  kernels now live in `rlx_core::host_kernels` (math unchanged). `rlx-grounding-dino`
  additionally moved its compute (Swin / text encoder / enhancer / decoder) onto
  the `rlx` graph path, with `nn.rs` rebacked on `rlx_cpu::blas`.
- `scripts/publish.sh` publish tiers regenerated from the workspace
  dependency graph to cover all publishable crates.

## 0.2.6 — RLX runtime alignment (2026-06-13)

Workspace and model runners now pin upstream **`rlx*`** **0.2.6** on crates.io
(`rlx-runtime`, `rlx-ir`, `rlx-flow`, …). Requires RLX **0.2.6** published from
[MIT-RLX/rlx](https://github.com/MIT-RLX/rlx) first.

### Model runners (dependency-only release)

Same Rust sources as **0.2.5**; `Cargo.toml` pins updated from `=0.2.5` to
`=0.2.6`:

- `rlx-neutts` 0.2.6
- `rlx-gemma` 0.2.6
- `rlx-minicpm5` 0.2.6
- `rlx-minimax` 0.2.6
- `rlx-nemotron` 0.2.6
- `rlx-models` 0.2.6 (facade; publish last)

Publish tiers 0–6 before the facade (`scripts/publish.sh --list`). After
`rlx-kittentts` **0.2.8** and the tier-5 runners above are on crates.io, Skill
can drop `[patch.crates-io]` path deps and use registry versions only.

### Also at 0.2.6+ in this workspace

- Full workspace `[workspace.package].version` = **0.2.6**
- `kitten_tts_mini_rlx` **0.2.7**, `rlx-kittentts` **0.2.8** (native RLX bundle path)
- `rlx-qwen3-tts`, `rlx-fft` at **0.2.7** where noted in `Cargo.toml`
