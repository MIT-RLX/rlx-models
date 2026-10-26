# Changelog

## Unreleased

## 0.2.19 — 200 model families, one exporter, upstream rlx 0.2.19 (2026-10-25)

First release since 0.2.11 on crates.io; 0.2.14 and 0.2.16 were prepared but
never published, so their notes below ship here.

### Upstream rlx 0.2.19

- Every upstream `rlx*` pin is `^0.2.19` (was `^0.2.16`, with `rlx-search`
  at `^0.2.18` and `rlx-rng` at `^0.2.17`), including `rlx-program` in the
  standalone `program/` manifests. The workspace version follows: 0.2.19.
- All 32 pinned upstream crates resolve from crates.io, `rlx-torch-ckpt`
  included, so a fresh clone builds without a sibling `../rlx`.
- `scripts/publish.sh`: `rlx-sanotts` moved from tier 0 to tier 2 (it depends
  on `rlx-models-core`, tier 1), and `rlx-models-plugin`, previously in no
  tier and so never published, gets its own final tier after the facade.

### Renamed features and options

- Every crate in the repo is versioned 0.2.19, including the standalone
  `program/` crates and `rlx-qwen3-ps5` / `rlx-ten-vad-mcu`.

### Recursive Flint (`rlx-flint`)

- **New crate** for [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint),
  Recursive's 39.8B-total / ~2.5B-active coding MoE (`model_type = flint`, BF16
  safetensors). Follows the only published implementation, the SGLang plugin in
  the model repo: 35 sliding-window (513) + 14 full-attention layers, RoPE on
  sliding layers only, per-head QK RMSNorm, learned per-head attention sinks
  (decomposed as an extra softmax column — `Op::Attention` has no sink input),
  sigmoid top-6-of-256 routing (reusing `rlx-llada2`'s gate kernel with one
  group) plus an ungated shared expert. Chat template, tokenizer, CLI
  `rlx-flint` (`just flint`), `FlintRunner` as an `LmRunner`, registered as
  `flint` for `auto_runner` (facade feature `flint`).
- `FlintRunner` never holds the 74 GB of routed experts: each layer is a graph
  (attention, router, shared expert), the fired experts are read from the
  checkpoint into bank slots, and one weightless routed graph serves all 47
  sparse layers. Non-expert weights stay bf16-resident (lossless, ~14 GB peak
  RSS on CPU). The key/value cache lives on the host, pre-transposed, with
  sliding layers in a 512-slot ring.
- Parity vs `scripts/flint_reference.py` (a torch transcription of the SGLang
  model) on the full checkpoint: logits within 3e-4 (magnitudes to ~100) and
  top-1 identical at every position, through token-by-token decode and chunked
  prefill, on CPU and Metal; on synthetic weights in CI (`tests/fixtures`).
  ~1.5 tok/s decode on CPU.
- Tool calling: `chat::render_with_tools` writes the template's tool
  preamble, assistant `<tool_call>`s and `<|observation|>` results
  (byte-identical to `chat_template.jinja` through transformers, fixture from
  `scripts/flint_chat_reference.py`); `tools::parse_reply` reads reasoning,
  content and calls back out, typed by the tool schema. CLI `--tools FILE`.
- Experts are now picked on the host from the layer graph's router scores
  (the gate kernel's CPU body, `rlx_cpu::llada2_gate`), so the paged runner
  no longer needs the custom op: parity on the full checkpoint now also holds
  on MLX, Vulkan and wgpu.
- Routed experts now run on the host straight from the checkpoint's bf16
  bytes (`experts::add_routed`: 64-row `pread` blocks, bf16 dot products, f32
  accumulate) instead of being decoded to f32, transposed into banks and run
  through a grouped-GEMM graph. Same parity; expert time on a warm page cache
  9.4 s → ~2.2 s over the 11-token parity run, CPU decode 0.53 → 0.74 tok/s on
  a busy machine, where a long generation is now bound by the disk.
- Opt-in expert read-ahead (`--prefetch`, `prefetch` module): layer l + 1's
  experts are guessed from layer l's router input (80% of guesses fire) and
  handed to the OS as read-ahead. Off by default until a quiet-machine A/B
  shows a speedup.
- Quantized routed experts: `rlx-flint quantize --scheme q8_0|q4_0` (`just
  flint-quantize`) writes llama.cpp-style Q8_0 (40 GB) / Q4_0 (21 GB) expert
  stores, read with `--experts DIR`; NEON block dots match the bf16 dot's cost
  per weight. Teacher-forced over 158 tokens: perplexity 5.61 (bf16) → 5.86
  (Q8_0, 96% top-1 agreement) → 5.94 (Q4_0, 90%). End-to-end speed could not
  be measured on the shared machine.

### Perplexity pplx-decider (`rlx-pplx-decider`)

- **New crate** for [`perplexity-ai/pplx-decider-flash-gguf-v1.2`](https://huggingface.co/perplexity-ai/pplx-decider-flash-gguf-v1.2):
  a Qwen3.5-4B classifier answering typed questions (`choice` / `noul` /
  `score`) about a text or JSON state in one forward pass, protocol v3
  (every question shares one sequence, one `<|fim_suffix|>` readout each).
  Library (`PplxDecider::decide` returns the reference's JSON), CLI
  `rlx-pplx-decider` (`just pplx-decider`), runs the Q8_0 GGUF packed
  (6 GB peak on Metal). Images through the F16 mmproj: the processor's
  `smart_resize` + torch's int16 bicubic antialias resize (pixels bit-exact),
  Qwen3.5's 3-D M-RoPE for image tokens.
- Parity vs Perplexity's own `pplx_decider` code (`scripts/pplx_decider_reference.py`),
  on CPU, Metal and MLX: prompt byte-identical, token ids exact, logits within
  4.5e-4 (probabilities 2.5e-5) of the reference run on the dequantized GGUF,
  probabilities within 4.7e-3 of the bf16 checkpoint. Images (four requests,
  upscale / downscale / multi-image, Metal): resized pixels, expanded tokens and
  M-RoPE positions identical, vision tower 2.2e-4, logits 2.2e-4 of the
  reference on the Q8_0 + mmproj weights.
- **`rlx-qwen35`**: `Qwen35VisionEncoder::load` / `encode_nchw` — encode
  pixels a caller preprocessed itself (the built-in path resizes
  nearest-neighbour). Note for GGUF callers with image positions:
  `Qwen35Config::from_gguf` leaves `mrope_interleaved` off, and Qwen3.5 needs
  it on (a contiguous layout is invisible on text and wrong on images).
  The vision encoder now defaults MLX to fused attention like CPU (decomposed
  kept every layer's scores alive: 95 GB vs 52 GB peak, 57 s vs 35 s on a
  1408x896 image).
- **`rlx-pplx-decider` speed/memory**: CPU dequant cache off by default
  (prefill-only: 5 GB vs 19 GB, as fast or faster), finer length buckets,
  shapes that cannot share weights are dropped before the next compiles,
  and an opt-in `libjpeg-turbo` feature that decodes JPEG byte-identically to
  the reference (the pure-Rust decoder moved logits by up to 0.026).
- **Upstream `rlx` (needed for the numbers above; not in 0.2.16)**:
  `rlx-compile` CPU in-order planner drops the host-indexing output-ancestor
  pin (a single `Gather` used to pin every activation; Qwen3-VL tower 24.8 →
  4.9 GB); `rlx-mlx` runs GGUF Q8_0 as native affine 8-bit `quantized_matmul`,
  lowers per-head Gated DeltaNet to MLX's fused `fast::gated_delta_update`,
  releases lazy-graph intermediates at their last use (Qwen3.5-4B MLX prefill
  peak 21.7 → 10 GB at 1024 tokens, text suite 144 s → 11 s), and exposes the
  allocator cache (`RLX_MLX_CACHE_LIMIT`, `clear_cache`, `memory_stats`,
  `RLX_MLX_MEM_TRACE`). Opt-outs: `RLX_PIN_OUTPUT_ANCESTORS=1`,
  `RLX_MLX_Q8_0_DEQUANT=1`, `RLX_MLX_GDN_UNROLL=1`, `RLX_MLX_KEEP_ENV=1`.
- **`rlx-qwen35`**: `Qwen35Flow::bidirectional_attention(key_padding_mask)` —
  non-causal full-attention layers with an optional `attn_key_mask` input, so
  right-padded batches stay exact; `Qwen35Config::gguf_attention_causal`, and
  the generative runner now refuses a GGUF that declares
  `attention.causal = false` instead of running it causally.
- **`rlx-qwen35` fix**: the tokenizer rebuilt from GGUF metadata ignored
  `tokenizer.ggml.pre` and NFC, splitting with GPT-2's regex — `"\n\n"`
  became `198 198` instead of `271`, so every GGUF-only chat prompt was
  off-distribution. It now uses Qwen2's / Qwen3.5's split regex and NFC
  (`QWEN2_SPLIT_REGEX`, `QWEN35_SPLIT_REGEX`, `qwen_pre_tokenizer`).
- **`rlx-qwen35` fix**: a `Qwen35Flow` without an embed stage handed
  `input_ids` to the layers instead of `hidden` (the first declared input
  starts the flow).

### Fish Audio S2-Pro and Drama 3 (`rlx-fish`)

- **`rlx-fish` rewritten** from a config-only stub into working ports of
  [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) (dual-AR:
  36-layer Qwen3-shaped slow backbone + 4-layer fast decoder) and
  [`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3) (40-layer
  Qwen3.5-MoE hybrid, Gated DeltaNet + gated attention, 256 block-FP8 experts
  run on the host from the mapped shards, 13-layer codebook tower), plus the
  shared 44.1 kHz codec (`codec.pth`: causal DAC + transformer-wrapped RVQ),
  text chunking / speaker turns / voice cloning as in each reference, and the
  `rlx-fish s2|drama3|codec` CLI (`just fish-s2`, `just fish-drama3`).
- Parity vs the official PyTorch code (`scripts/fish_reference.py`): codec
  waveform 3e-6, encode 440/440 codes; S2-Pro full model 8 greedy frames
  identical; Drama 3 (4-layer cut + full tower) frames identical — on CPU,
  Metal, MLX and wgpu. Whisper reads S2, S2 voice-cloned and Drama 3 output
  back verbatim.
- **`rlx-models-core`**: `audio::resample_sinc`, the torchaudio-matching
  resampler, moved here from `rlx-tada` (which re-exports it).
- Needs upstream `rlx` past 0.2.18 (`CompiledGraph::param_already_set`,
  `rlx_runtime::with_param_sharing_scope`, `rlx_flow::storage`) and benefits
  from its new Metal bf16 decode GEMVs and contiguous concat.

### KittenTTS 2 (`rlx-kittentts2`)

- **New crate** for [`KittenML/kitten-tts-2`](https://huggingface.co/KittenML/kitten-tts-2):
  1.7B Qwen3 speech LM with ternary linears + chatterbox-turbo S3Gen (meanflow
  conformer/U-Net flow, HiFT vocoder), 123 built-in voices, expression tags,
  sentence chunking, and **voice cloning** from a reference clip (S3 tokenizer,
  pyannote x-vector, CAM++ via `rlx-funasr`, Matcha mel / Kaldi fbank). At parity with the upstream PyTorch package in f32 on CPU
  and Metal (48/48 greedy tokens; waveform |Δ| ≤ 4e-5); Whisper reads the fox
  pangram back 6/6. `just fetch-kittentts2`, `just kittentts2 "…"`.
- The shipped `kittenml-ternary-1.58` checkpoint transcodes **losslessly** to a
  `Q2_0` GGUF (each 128-wide `{-s, 0, +s}` group is one `Q2_0` block), so the
  existing packed `rlx-qwen3` path runs it.
- **`rlx-qwen3`**: `Qwen3Generator::prefill_embeds_get_last_logits` (prefill
  from caller-built embeddings, native packed), `step_cached_logits` +
  `push_token` (decode without the built-in sampler), `enable_host_embed` (host
  embedding gather on any device).
- **`rlx-qwen3`: MLX resident decode failed with `missing input 'past_k_0'` —
  FIXED.** The resident K/V handles were bound with `upper` rows while the
  decode graph declares `upper + kv_spare_rows`; Metal sized the buffer from
  the graph and hid it, MLX checks shapes. The bind now pads like the host path.
- Ported the rest of upstream: the written→spoken **text normalizer**
  (`kitten_text_processing`'s NeMo WFST grammars, embedded, walked natively;
  400/400 sentences identical), **Whisper transcription** of a clone reference
  when no transcript is given (feature `transcribe`), **reference clips ≥ 30 s**
  (windowed S3 tokenization), and the distilled **`student_w4` decoder**
  (BitNet flow, 1 step; `--decoder student_w4`).
- Backends: LM parity on CPU, Metal, MLX, wgpu (macOS), CUDA and Vulkan.
  wgpu decode 0.3 → 5–9 tok/s on macOS: `rlx-wgpu`'s fused GGUF GEMV had no
  `Q2_0` branch, so all 197 ternary projections per token went through
  dequant-to-scratch with a submit and host sync each; `Q2_0` now runs in the
  cooperative 32-weight sub-block kernel (`dequant_gemv_b32x16`), within 3e-5
  of an exact f64 reference.
  wgpu on NVIDIA too, after two upstream `rlx-wgpu` / `rlx-gpu-host` fixes:
  discrete adapters run copies on the host and defer the device write, and
  (1) the hosted `BufferCopy` branch skipped the flush before the next GPU
  section, so each layer's attention ran before its KV-append row landed;
  (2) a host step reading a range flushed only a deferred write keyed at
  exactly its start, missing a row deferred *inside* the cache. Both silently
  dropped the newest token from every decode step. Tests
  `host_buffer_copy_flush.rs`, `host_kv_append_host_reader.rs` (reproduce on
  any adapter via `RLX_WGPU_HOST_BUFFER_COPY=1` / `RLX_WGPU_FORCE_HOST=1`).
  **`rlx-qwen3`**: the native packed prefill now buckets and caches its
  compiled graph (prefill 6–9 s → ~2 s per chunk); `RLX_QWEN3_KEEP_BF16` keeps
  BF16 GGUF tensors packed, so with `RLX_PACKED_LM_HEAD` a tied BF16 embedding
  is the LM head with no f32 copy (wgpu uses this — the f32 head is 1.3 GB).
  The transcoded GGUF keeps the embedding BF16 (was F16; older files are
  re-transcoded on load). **`rlx-models-core`**: `GgufLoader::take_packed`
  honours `keep_bf16`.
- Upstream `rlx` fixes found here: MLX runs `Q2_0` through native
  `quantized_matmul` (was host dequant); Vulkan has a `Q2_0` GEMV (decode had
  drifted onto a near-tie through the host int8 path); wgpu's f16 GEMM read
  `vec4`s at misaligned offsets for odd `n`, and mirrored big f32 weights to
  f16 when they fit a binding anyway; **rlx-compile pinned the whole ancestor
  chain** of every `DequantMatMul` input (and of every hosted structural op's
  inputs) to graph end, where only the tensor and the views it aliases need
  it — a packed Qwen3-1.7B prefill planned 2.6 GiB with almost no slot reused,
  now 1.4 GiB.

### Fixes and dedup from the decode audit

- **`rlx-asr`: wrong mel filterbank — FIXED.** The port wrote each filter's
  bumped edges back for the next filter; `tools/audio_io.py` does not. 7 of 80
  filters differed from the reference (up to 1.0) in release builds, and the
  edge subtraction underflowed in debug (3 tests panicked). Pinned by
  `mel_filterbank_matches_the_python_reference`.
- **`rlx-qwen35`: sampled decode reused one random number — FIXED.** Every
  step and every batch row drew `Philox(seed)`'s first value, so each token was
  taken at the same quantile of its distribution. Rows now get their own
  stream (`seed + b·2³²`) advanced per generated token; row 0 equals
  `sample_token_at(seed, step)`.
- **`rlx-qwen3` sampling** (also used by gemma, llama32, qwen35, orpheus, …):
  the inline temperature/top-k/top-p path full-sorted the vocabulary twice
  per token; it now runs through `rlx_runtime`'s `SamplerChain`
  (`SampleOpts::into_chain` maps onto `rlx_runtime::lm::SampleOpts`). At
  temperature 0.6 / top-k 20 / top-p 0.95 on a 151,936-entry row: 2.9 ms →
  0.6 ms per token, the same token in 1500 of 1500 seeded draws.
- **`rlx-nemotron-asr`**: greedy RNN-T runs on `rlx-audio-blocks`' TDT loop
  (duration `0`), like Parakeet; `shared_loop_matches_the_previous_greedy_decode`
  pins it over 200 random nets. **Either SOS rule:** new
  `rlx_audio_blocks::decoders::tdt::SosState { Carry, Discard }` — `Carry`
  (NeMo greedy search) keeps the prediction net's post-SOS state for the first
  token, `Discard` steps it from zero. Nemotron keeps its own `Discard` by
  default (`AsrConfig::sos_state`); Parakeet keeps `Carry`
  (`TdtCore::with_sos_state` to switch).
- **`rlx-translate`: `Tuning::set` no longer writes the process environment**
  to reuse the env parsers (it raced with any other thread; the `tuning` tests
  failed in parallel). It feeds the one value to the same parser; it no longer
  re-applies unrelated `RLX_TRANSLATE_*` variables.
- `rlx-asr` / `rlx-laguna`: byte-identical in-crate softmax copies merged.
  `rlx-dac`: `decoding` gated on `export` (was dead code without it, failing
  `clippy -D warnings` for every dependent). `rlx-ocr`: collapsible-if lint.

### Decoders share `rlx-search`

Beam search, CTC decoding and forced alignment now run on `rlx-search`
(needs rlx 0.2.18; local dev picks it up through `.cargo/config.toml`).

- **`rlx-whisper`**: beam search follows `openai/whisper`'s `BeamSearchDecoder`
  (`beams + 1` proposals, finished pool, ranked by `score / len`); it used to
  keep finished hypotheses in the beam and rank by raw score, which favors
  short output. The KV search scores only kept hypotheses (`beams` decoder
  steps a token, was `beams²`). **Fixed:** batched multi-region beam search
  fed each parent's argmax instead of the chosen token and dropped the first
  generated token; `suppress_tokens` were not applied to the first step and
  `begin_suppress_tokens` never in beam mode. Greedy `argmax_next` no longer
  log-softmaxes the row (argmax is unchanged by the shift).
- **`rlx-ocr`**: **Fixed** CTC beam search keyed prefixes by label *and*
  frame and moved repeat-after-blank mass onto the wrong prefix; it disagreed
  with rten's `CtcDecoder::decode_beam` on 141 of 150 test inputs, now on 0
  (`ctc_beam_matches_reference`).
- **`rlx-translate`**: `beam::search` is a configuration of the shared engine;
  `matches_the_previous_search` pins it to the old implementation (1200
  cases, tokens identical, scores within 1e-9). `no_repeat_ngram` now masks
  before each hypothesis's top-k instead of dropping blocked proposals after
  it (off by default).
- **`rlx-asr`, `rlx-ocr2`, `rlx-funasr`, `rlx-conformer-ctc`, `rlx-citrinet`,
  `rlx-qwen3-aligner`, `rlx-qwen3`**: their CTC prefix beam (two identical
  copies), CTC greedy, Viterbi aligner and greedy argmax delegate to
  `rlx-search`; public signatures unchanged. Prefix-beam ties are broken by
  label sequence, not `HashMap` order.
- **Greedy argmax in 30 more crates** (every LM/VL/TTS/ASR runner with its own
  copy) calls `rlx_runtime::search::argmax`: lane-parallel, 46 µs vs 164 µs on a
  151,936-entry row. The copies written as `max_by(partial_cmp)`
  (`rlx-tiny`, `rlx-tinystories`, `rlx-soprano`, `rlx-qwen25-vl`,
  `rlx-metavoice`, `rlx-voxtral`, `rlx-voxtral-tts`, `rlx-qwen3-asr`,
  `rlx-fireredaudio`, `rlx-locateanything`) returned the *last* index on an
  exact tie and panicked on NaN; they now return the first, as torch and numpy
  do, and skip NaN.

### One exporter for every family: `rlx_core::export`, `rlx-export`

- **`rlx_core::export`** (`rlx-models-core`, feature `export`): what every
  exporter shares, in one place. `Checkpoint` (a Hugging Face directory or a
  GGUF: config, tokenizer, chat template, stop ids, sampling defaults, card),
  `Package` (the one writer: functions, entries, sidecars, held tensors, and
  params referenced from the checkpoint), `Referencer` over safetensors *and*
  GGUF files (`open_gguf_only`, `packed`), `ExportOptions` / `ExportReport`
  for every family, `rope::Rope` (tables from `rope_scaling`: llama3,
  linear, YaRN, LongRoPE), `decoder::{Decoder, export_decoder}` (moved from
  rlx-qwen3), `graph::export_stages` (any graphs as a `pipeline` entry),
  `check::check_generate` (a package's decode against a runner's), and the
  exporter registry (`register_exporter`, `detect`, `export`): a checkpoint
  goes to the family that claims its `model_type` / GGUF architecture, else
  to the exporter named as its runner.
- **`rlx-export`** (`rlx-models`, feature `export`): `rlx-export <checkpoint>
  -o model.rlxp [--check]`. `--check` (with `runner`) decodes the package
  against the family's `LmRunner`, prompts in the package's chat template,
  each side ending at the package's stop ids. `rlx_models::export::
  {export_checkpoint, check_package, register_builtin_exporters}`.
- **Every exporter ported**: qwen3, gemma, qwen3-vl, qwen25-vl, qwen35, embed
  (and rerankers), whisper and siglip2 write through `Package` and take the
  shared options (`texts` for an encoder's batch; `frontend_package` for
  Whisper's and SigLIP's front ends). Each crate has `export::register()`.
- **New**: Llama 3.x with scaled RoPE (`rope_scaling` was refused), Mistral
  (context stops at its sliding window), LFM2 / LFM2.5 (`rlx-lfm`, feature
  `export`: the step's mask and row write computed from `position`, every
  cache a variable, symbolic in its capacity).
- **Audio codec decoders and NomicBERT**: SNAC, NanoCodec, WavTokenizer,
  XCodec2, VibeVoice, FACodec, EnCodec, SpeechTokenizer, DAC and Mimi as
  `pipeline` packages for any number of codes, the runners' host steps in
  the graph (SNAC's codebook lookup, NanoCodec's FSQ, the Vocos inverse
  STFT — `export::audio` —, FACodec's speaker timbre, EnCodec's and
  SpeechTokenizer's residual-codebook sum and LSTM, DAC's and Mimi's
  projected codebooks, Mimi's sliding-window mask from positions), each
  checked against its runner on real weights. NomicBERT through the
  `embed` exporter, and ClinicalBERT too (checked against
  `ClinicalBertRunner`). `graph::{symbolic_in, broadcast_params,
  per_size}` (a table may follow unit axes: `[1, t, d]`); a `Checkpoint`
  may be a lone `.safetensors`.
- **Parler-TTS, MioTTS and MOSS-TTS-Nano** (`rlx-parlertts`, `rlx-miotts`,
  `rlx-moss-nano`, feature `export`): language-model TTS as packages rlx
  runs alone, their sampling loops programs. Parler's decoder keeps its
  logits host-side (a slot) and `pick` reads one step; its DAC is
  `rlx_dac::export::dac_functions`. MioTTS is Qwen3's package
  (`export_decoder_with`, `rlx_qwen3::export::export_rlxp_with`) with the
  codec's graphs, its program generating through `entry.run`. Lifts: a
  baked causal mask (`Lift::Triangle`); a length-shaped param is refused.
  `audio::istft_same` (magnitude and phase, `padding="same"`). Parler
  matches its runner at cosine 1.0, MioTTS's codec at 1.0 and its LM's
  first sampled codes exactly, MOSS's frames exactly (its runner's own codec
  decode is NaN; the package's is onnxruntime's within 8.5e-6).
- **Orpheus and Maya1** (`rlx-orpheus`, `rlx-maya1`, feature `export`):
  the GGUF Llama's language-model package
  (`rlx_llama32::export::export_gguf_with`) with SNAC's lookup and decoder
  (`rlx_snac::export::snac_functions`) and a program: the prompt, the
  language model a token at a time, its logits penalized `penalty^count`
  and sampled through the host's `sample` (rlx's `sample` host service,
  new), the seven codes of each frame, SNAC, the peak normalization.
  Maya1 is Orpheus' package (`export_flavor`) with its description prompt
  and greedy decoding. Orpheus' first 28 sampled codes are the runner's;
  SNAC on the runner's codes at cosine 0.99997. Maya1's speech, greedy end
  to end, at cosine 0.99995.
- **Soprano 1.1** (`rlx-soprano`, feature `export`): its backbone and Vocos
  decoder ONNX converted to rlx IR once (the backbone with its past caches
  empty, as the runner recomputes the whole context; symbolic in the
  tokens and the frames), the runner's host logic a program — the prompt,
  its sampler (temperature, the repetition penalty, top-k, top-p, its
  xorshift stream), the chunking, the offline crop. Its audio tokens are
  the runner's, sampled (32) and greedy (29), and its speech too (cosine
  1.0); no op in the package is an ONNX one.
- **ChatterBox** (`rlx-chatterbox`, feature `export`): its five ONNX
  graphs converted to rlx IR once, each symbolic in its lengths (the flow
  encoder in all the tokens, the prompt's tokens and its mel frames — its
  dims `2·tokens − 1` and `2·tokens − frames`, which rlx's fit now finds), T3
  as rlx-llama32's graph over `native/t3_lm.safetensors` (referenced), the
  default voice's conditioning computed by rlx at export, and the runner's
  host logic a program: the prompt embeds, T3 over the context padded as
  the runner pads it, the sampler, the CFM Euler solver with guidance,
  HiFT's inverse STFT, the onset polish. Its speech tokens, mel and speech
  are the runner's (native T3), sampled and greedy. A cloned reference is
  2 to 4 s: past 100 units (4 s) the encoder's speaker branch pads to a
  multiple of 200 frames, one graph per step of it; the runner's own import
  of `audio_features` is onnxruntime's at cosine 0.945 only (the package
  matches the runner).
- **RLX TTS** (`rlx-tts`, feature `export`): FastSpeech2's eager encoder
  and decoder as graphs (symbolic in phones and frames), the sample-level
  WaveRNN as one `Op::Scan` a mel frame (the portable path's f16-rounded
  activations, its Gumbel-max), and the crate's text frontend, durations,
  buckets and output post in a program (drawing the BNNS AES-CTR stream of
  `WaveRnnRng::Bnns`). The mel is the runner's within 1e-6; the vocoder its
  portable path's for the first ~570 samples, then the runner's fused GRU
  arithmetic and the scan's separate ops part (greedy alike); Whisper reads
  the package's speech exactly. Its macOS default (`NativeBnns`, the vendor's
  private Gumbel stream) is not reproducible off macOS, and greedy decoding
  is not intelligible ("the click brown fox").
- **Fish Audio S2-Pro** (`rlx-fish`, feature `export`; facade feature
  `fish`, new): the crate's slow, fast and codec graphs (its builders now
  return them uncompiled too), their RoPE inputs table rows, the steps'
  new keys and values written into their caches, the codec symbolic in the
  frames (its window mask pinned to the sliding window, which equals the
  causal mask below it); the conversation, sampler and dual-AR loop in a
  program. Its 24 frames are the runner's, the codec's speech on them at
  cosine 1.0.
- **Qwen3-TTS's speech decoder was 0.84 off the reference** (`rlx-qwen3-tts`):
  its ConvNeXt upsampling used `0.5·v·(1 + tanh(v·FRAC_2_SQRT_PI/2))` — the
  `2/√π` constant (not `√(2/π)`) and no cubic term — where the official
  `qwen-tts` decoder uses `nn.GELU()`, the exact erf form. Every utterance's
  waveform was affected (max |d| 0.84 at peak 0.87 on fixed codes); now
  4.8e-5 (`tests/speech_decoder_reference.rs`, fixture from
  `scripts/speech_decoder_reference.py`). The talker and code predictor
  were right: CustomVoice greedy frames are `qwen-tts`'s, 30 of 30
  (`tests/custom_voice_reference.rs`, `scripts/custom_voice_reference.py`).
- **Qwen3-TTS CustomVoice** (`rlx-qwen3-tts`, feature `export`; exporter
  `qwen3-tts`): the text/codec embedding rows, the codec head, the talker's
  prompt and stateful step, the code predictor's prompt, step and per-group
  heads and embeddings, and the 12 Hz speech decoder, over the checkpoint's
  tensors; the program assembles the CustomVoice prompt and runs the codec
  and code-predictor loops. Greedy, 30 of 30 frames `qwen-tts`'s, speech
  cosine 1.0. The talker and code predictor name their weights alike
  (`model.layers.*`): the code predictor's are `cp.`, and merging two
  graphs' params now refuses one name with two values (it had silently
  bound the code predictor's five layers into the talker).
- **Moshi** (`rlx-moshi`, feature `export`; exporter `moshi`): Moshiko
  one-way as graphs over the checkpoint's bf16 tensors by name — the
  embeddings, the temporal step, a DepFormer slice per codebook — plus
  Mimi; the program feeds the prompt a token a frame with the delayed codes
  and samples with the crate's own sampler. Greedy and sampled, its frames
  are the runner's (10 of 10). `Package::reference_named` now records each
  half-precision tensor's dtype, so `weights_as_stored` keeps them bf16.
- **Kyutai TTS** (`rlx-kyutai-tts`, feature `export`; exporter
  `kyutai-tts`): every graph reads the checkpoint's bf16 tensors by their
  own names (`Package::reference_named`, nothing copied) — the embeddings,
  the speaker conditioning, the temporal step with its cross-attention, a
  DepFormer step per codebook (its per-step weights narrowed from the
  stacked projections) — plus Mimi (`rlx_mimi::export::mimi_functions`).
  The SentencePiece model becomes a `tokenizer.json` (`rlx_core::export::spm`, the
  protobuf read directly). The program is the crate's own generation loop,
  state machine, delays and sampler built to wasm. Greedy, its frames and
  speech are the runner's (cosine 0.9999999995); sampled, 22 of 34 frames.
- **Inflect-Nano** (`rlx-inflect-nano`, feature `export`; exporter
  `inflect-nano`): MicroFastSpeech as two graphs (the encoder and prosody
  heads; the frame stack through the bidirectional GRU and postnet, as
  `Op::Gru`) and the vocoder (`graph::vocoder_hir`); the crate's own text
  frontend — CMUdict, g2p, the POS tagger, the BERT tokenizer — built to
  wasm in the program via `#[path]`. Its speech is the host reference's at
  cosine ≥ 0.99999996.
- **sanoTTS** (`rlx-sanotts`, feature `export`; exporter `sanotts`): the
  duration student and the acoustic token stack as graphs symbolic in the
  tokens (`graph::token_graph`), the frame stack and decoder symbolic in the
  frames (`graph::frame_graph`, the masks inputs of ones); espeak-ng, the
  phoneme table, the half-to-even durations and the expansion to frames in a
  program. Its speech is the host reference's within 5.1e-6.
- **Wake words** (`rlx-wake` and `rlx-openwakeword`, feature `export`;
  exporters `nanowakeword`, `porcupine`, `voxrt`, `openwakeword`): the mel
  frontend as a graph and the scoring nets (the wake CNN one graph per
  window length; openWakeWord's embedding and phrase head), one program
  streaming 80 ms pushes as the engines do, with threshold and cooldown.
  Every push's score is the engine's. The stub weights every wake crate
  ships and trains from were all negative: `(rng >> 33) / u32::MAX` spans
  [0, 0.5), so `· 0.02 − 0.01` never reached above 0 — every ReLU dead and
  every push scored sigmoid(bias). Now [0, 1) (`rlx-wake` stub and MLP
  init, `rlx-wakeword-core`, `rlx-openwakeword`); the `student_*.safetensors`
  in the crates were saved from the old stub and still score a constant.
- **Voxtral** (`rlx-voxtral`, feature `export`; exporter `voxtral`): the
  Whisper log-mel over whole 30 s windows, the encoder (its
  batch-broadcast biases one held slice, `broadcast_params`) and projector
  symbolic in the windows, the prompt's embedding, the Llama prompt graph
  and step — whose caches carry the decode's in-place append row, so the
  program's mask and caches are `cap` rows with the token's last; tekken
  decoding in the program. Its transcripts are the runner's on both JFK
  clips, and the runner's are transformers' (`scripts/voxtral_reference.py`,
  with mistral-common).
- **Qwen3-ASR** (`rlx-qwen3-asr`, feature `export`; exporter `qwen3-asr`):
  Whisper's log-mel as a graph (now `rlx_core::export::audio::whisper_log_mel`,
  shared), the audio tower symbolic in its chunks and audio rows (each row's
  attention window an input, the block-diagonal bias computed in the graph:
  `build_encoder_windowed`), the prompt's embedding with the audio's rows
  written in, the Qwen3 decoder's prompt graph and its stateful step; the
  padding, chunk layout, chat prompt, greedy decode and the text after
  `<asr_text>` in a program. Its transcripts are the runner's on both JFK
  clips, and the runner's are transformers' (`scripts/qwen3_asr_reference.py`
  on the `-hf` checkpoint).
- **Whisper's STFT padding** (`rlx-whisper`): `center=True` reflects the
  signal's first samples into the left pad; `stft_mag` copied its *last*
  ones there (a wrap, zeros for any clip padded to 30 s) and mirrored the
  right edge including the edge sample. Frame 0 was 0.39 off
  `WhisperFeatureExtractor` in the log mel, frame 1 0.027; now 1.6e-5. Every
  model on `rlx_whisper::pcm_to_log_mel` (Whisper, Voxtral, FireRedASR)
  reads it; the Whisper package's front end graph reflects too (it had zeros).
- **Paraformer** (`rlx-funasr`, exporter now `funasr`, by `config.yaml`'s
  `model`): the SAN-M encoder with the CIF predictor's alphas and the
  decoder (symbolic in the frames and the tokens) as graphs; the frontend
  and the integrate-and-fire (`cif.rs`) in the program SenseVoice shares.
  Its transcripts are the runner's, which reads FunASR's.
- **SenseVoiceSmall** (`rlx-funasr`, feature `export`): the encoder, its
  prompt rows and the CTC argmax as one graph symbolic in the frames; the
  crate's frontend, the collapse and the pieces in a program. Its
  transcripts are the runner's. The runner never applied CMVN: `am.mvn`'s
  first bracketed vector is a `<Splice>` block's `[ 0 ]`, which it took for
  the shift, and the length mismatch then skipped normalization silently
  (now the last two vectors, and a mismatch is an error); and it read
  `cmvn_file: null` as off, where FunASR's loader uses the directory's
  `am.mvn` whenever it is there. With both its logits are FunASR's within
  1.3e-4 (`scripts/funasr_reference.py`, which turns off the frontend's
  dither: FunASR dithers at inference, so its own output changes run to
  run).
- **Parakeet-TDT** (`rlx-parakeet`, feature `export`; facade `parakeet`,
  new): through Nemotron's exporter, now a FastConformer transducer
  exporter (`rlx_nemotron_asr::export::transducer_rlxp`: TDT durations, the
  start-of-sequence state kept or not, the language prompt only where the
  model reads one). Its transcript is the runner's. The runner had never
  run on real weights and transcribed nothing; checked against the Hugging
  Face port (`parakeet-tdt-0.6b-v3`), it now reads the clip as the reference
  does, punctuation included, after the encoder learned the non-streaming
  FastConformer: depthwise convs padded `(k − 1) / 2` each side
  (`conv_context_size` not `causal`), BatchNorm from running statistics
  (`conv_norm_type: batch_norm`), and per-feature normalization over the
  audio's frames alone (n − 1 dof), the frame past them zeroed after.
- **Nemotron 3.5 ASR** (`rlx-nemotron-asr`, feature `export`; facade
  `nemotron-asr`): the frontend, the FastConformer (subsampling symbolic in
  the mel frames, its layers in the encoder frames), the prompt kernel and
  joint projection, the prediction LSTMs' step and the joint as graphs; the
  greedy RNN-T search and the pieces in a program. Its transcript is the
  runner's. The runner had dropped words ("we choose to write our software
  and not because it is safe and free garbage"); against the Hugging Face
  port (`scripts/nemotron_reference.py`) three fixes made it read the clip
  as the reference does, its encoder within 2e-6: the cache-aware
  `chunked_limited` attention mask was never applied; a
  `causal_downsampling` subsampling pads both axes `(k − 1, stride − 1)`
  (a non-causal one `(k − 1) / 2` each side), where it padded time
  `(2, 0)` and frequency `(1, 2 − f mod 2)`; and the STFT is zero-padded
  (`pad_mode="constant"`), not reflected, the frame past the audio zeroed.
  `rlx_core::export::audio::nemo_features_with` frames audio so
  (`NemoPadding`).
- **HumeAI TADA** (`rlx-tada`, feature `export`; facade feature `tada`,
  new): the rows' embedding, the Llama-3.2 backbone's prompt and step
  graphs (both guidance branches), the flow-matching head's ODE solve as one
  graph, and the codec decoder (its segment mask computed in the graph) in a
  package; the prompt plan, noise, durations and trims in a program, the
  voice prompt a sidecar. Its frame gaps are the runner's, its speech at
  cosine 1.0. `just fetch-tada` passed several `--include` patterns to one
  flag, which this `hf` reads as file names: the codec encoder and
  `tokenizer.json` were never fetched.
- **Fish Audio Drama 3** (`rlx-fish`, exporter `fish-drama3`, claims
  `fish_omni`): each backbone layer's prompt and step graph (the runner's
  builders, uncompiled), the routed experts as one `experts` graph whose FP8
  weights and block scales are inputs the program has the host read for the
  run (eight experts a run, dequantized in the graph), the final norm and
  main-token head, each tower layer's prompt and step graph with its code
  embeddings gathered in the graph, and S2-Pro's codec. The 39 GB
  checkpoint is referenced by name (`Package::reference_named`), no tensor
  read at export. The prompt (`drama3_text.rs`, `prompt.rs`), routing,
  repetition-aware sampling and the tower loop in a program. Its 8 frames
  are the runner's, the codec's speech on them at cosine 1.0.
- **Voxtral-4B-TTS** (`rlx-voxtral-tts`, feature `export`): the backbone as
  a prompt and a step graph (fixed-capacity caches written in place; the
  step embeds the last frame's codes), the semantic head and flow-matching
  transformer's Euler steps as one `acoustic` graph, the audio tokenizer's
  decoder symbolic in the frames, the checkpoint's weights referenced; the
  Tekken BPE, speech prompt, the runner's noise stream and code
  quantization in a program. Its 24 frames are the runner's, the codec's
  speech on them at cosine 1.0.
- **Voxtral-4B-TTS runner fixed** (`rlx-voxtral-tts`): it spoke nothing
  intelligible (Whisper heard "the" in 16 s, no end of audio). Against
  vLLM-Omni / mistral_common: the Mistral-native q/k need GPT-J RoPE (vLLM
  permutes them to NeoX halves on load); the eager attention and the
  codec's ALiBi attention started masked scores at 0, not -inf (future and
  out-of-window keys leaked — the CPU prefill is eager, and it flipped the
  first semantic code); the flow started from noise scaled by `sigma`, not
  1; the codec decoder's window starts at 2 (16 halved by the encoder's
  downsamplings), its first conv pads by replication, its transposed convs
  trim from the right, its reflect padding excludes the edge, and its q/k
  norms use `qk_norm_eps`. It now says the text (Whisper exact); its codec
  matches a PyTorch transcription of vLLM-Omni at cosine 0.9999999992
  (`scripts/voxtral_reference.py`).
- **Pocket TTS** (`rlx-pocket-tts`, feature `export`; facade feature
  `pocket-tts`, new): the crate's FlowLM step and Mimi decoder block, their
  RoPE inputs table rows read at a position, the step's new keys and values
  written into its caches; the released voices' states and the host-side
  embeddings as files; the runner's chunking (`text.rs`), noise stream, EOS
  rule and block decoding in a program. Its speech is `PocketTts::generate`'s
  (cosine 1.0, the same sample counts), one chunk and two, sampled and at
  temperature 0.
- **Gepard** (`rlx-gepard`, feature `export`): the eager Qwen3.5 backbone
  as a prompt graph and a step graph (gated attention, `(1 + w)` norms,
  GQA, fixed-capacity caches written in place; the step embeds the last
  frame's codes), both ending in the codebook and stop heads, the
  checkpoint's weights referenced; the HiFi-GAN decoder symbolic in the
  frames; the runner's prompt, `fastrand` sampler and stop rule in a
  program. Greedy and sampled, its 40 frames are the runner's; the decoder
  on them at cosine 1.0. (`GepardSynthesizer::generate_frames` /
  `decode_frames`, new, split the eager loop from the decoder.)
- **MetaVoice-1B** (`rlx-metavoice`, feature `export`): its eager CPU
  first stage as a prompt graph and a step graph over the guidance pair,
  each layer's caches a fixed number of rows the program picks a request
  (the step writes its row in place; a cache that grew a row a step
  compiled a 1.2B-parameter graph a step), its weights referenced; the
  second stage, the speaker encoder (DFT mel, `Op::Lstm`) and EnCodec's
  functions (`rlx_encodec::export::encodec_functions`, new); the runner's
  BPE, voice partials, guided argmax / nucleus loop, codebook split and
  post-processing in a program. Greedy and sampled, its 40 tokens are the
  runner's; the fine codes, the Bria voice's embedding and the speech match
  (cosine ≥ 0.999999999).
- **Zonos v0.1** (`rlx-zonos`, feature `export`): its backbone (the crate's
  CFG prefill graph, symbolic in the context, its RoPE inputs made params),
  the prefix conditioner's projection, the frames' embeddings, the heads,
  DAC's functions and the conditioning's small tensors; the runner's loop in
  a program (espeak-ng, the Fourier and language rows, guidance, the delay
  pattern, its sampler and EOS rules). Greedy and sampled, its 24 frames are
  the runner's; DAC on them at cosine 1.0.
- **Sesame CSM-1B** (`rlx-sesame`, feature `export`): its eager CPU
  backbone and depth decoder as rlx-llama32 graphs over embedded rows (the
  checkpoint's weights referenced, renamed back from the flow's), frame
  embedding, heads and projection graphs, Mimi's functions
  (`rlx_mimi::export::mimi_functions`, new), and the runner's loop in a
  program (its `fastrand` top-k). Greedy, the first 118 of 128 codes are the
  runner's; Mimi on the runner's frames at cosine 1.0. (Its sampler's
  `select_nth_unstable` orders candidates by their exact values, so sampled
  frames part sooner.)
- **MiraTTS** (`rlx-miratts`, feature `export`): the Qwen2 LM's package,
  BiCodec's `detokenizer` and `s_encoder` as rlx IR at the runner's fixed
  sizes, the reference's mel as a graph (its DFT and filterbank matmuls, so
  the program does none of it), and a program for the prompt, the sampling
  (`penalty^count`, the host's `sample` with min-p) and the glue. A
  reference's global tokens, the LM's sampled codes and the audio are the
  runner's.
- **KittenTTS 2** (`rlx-kittentts2`, feature `export`): the `Q2_0` speech
  LM's package with a `speaker` medium — `spk_proj` and a prefill over
  embedded rows, the speaker's row at the prompt's start token
  (`rlx_qwen3::export::export_gguf_media`, new; rlx's `Model::run` now
  takes media beside `input_ids`) — S3Gen's four stages symbolic in their
  lengths (the encoder's two relative-position tables slices of one), each
  voice's conditioning, and the crate's own frontend in the program (its
  NeMo grammars, chunking, prompts, sampler) with the flow loop and HiFT's
  host DSP. The LM's prompt logits are the runner's within 8e-5, S3Gen on
  the runner's tokens exact (cosine 1.0); greedy text to speech keeps the
  runner's first 42 codec tokens, then a near tie between its packed LM and
  the package's tips.
- **FIXED: `build_qwen3_prefill_embeds_built` ignored `with_lm_head`** — it
  returned the hidden states where a caller asked for logits (the
  Qwen3-VL runner's prefill read its first token's logits from them).
- **NeuTTS** (`rlx-neutts`, feature `export`): a GGUF backbone and
  NeuCodec's decoder as one package rlx runs alone. The backbone is the
  GGUF's language-model package — Air's Qwen2 GGUF through
  `rlx_qwen3::export::export_gguf_with` (new: a Qwen2 / Qwen3 GGUF's config
  from its metadata, its tokenizer from a builder the caller passes), Nano's
  Llama GGUF through `rlx_llama32::export::export_gguf_with` (new: GPT-J
  RoPE, `rope_freqs.weight`; the decode step reads its RoPE row as inputs,
  `Llama32Flow::rope_inputs`). NeuCodec is XCodec2's graphs
  (`rlx_xcodec::export::decoder_functions`, its weights' `generator.`
  prefix and head width now read) after an FSQ lookup graph. The program
  phonemizes with espeak-ng (its English data in the package), builds the
  prompt around a reference voice (`samples/` beside the GGUF), samples
  top-k 50 / top-p 0.9 through the host's `sample` and decodes the speech
  tokens. NeuCodec matches transformers' `NeuCodecModel.decode`
  (`scripts/neucodec_reference.py`) at cosine 1.0, within 2e-6; the
  backbone's first 28 sampled codes are the runner's.
- **FIXED: NeuTTS' own NeuCodec decoder** (eager and burn) disagreed with
  transformers' (cosine −0.1): FSQ digits scaled to {−1, −⅓, ⅓, 1} rather
  than {−1, −½, 0, ½}, RoPE by time position where the reference's (by
  head index) cancels, and `center` trimming rather than Vocos' "same"
  padding. Now cosine 1.0 against `NeuCodecModel.decode`; `fsq_encode`
  inverts the new scale as transformers' quantizer does (its tanh bound,
  rounding half to even).
- **FIXED, found exporting NeuTTS:** `Qwen3Runner::predict_logits` on the
  F32 path returned a one-element placeholder (the sampled id), not logits
  — every caller sampling from it read garbage; it now returns the
  prompt's last row (`prefill_get_last_logits`). NeuTTS' backbone refused
  Qwen2 GGUFs (Air) — it runs them through `Qwen3Runner` now; it decoded
  each token skipping special tokens, so its speech tokens never reached
  the code parser; and it sampled every token from the same draw
  (`sample_token`, step 0) — now `seed + step`.
- **TinyTTS / MeloTTS** (`rlx-tiny-tts`, `rlx-melotts`, feature `export`)
  and **OpenVoice v2** (`rlx-openvoice`, feature `export`) as packages rlx
  runs alone. The English frontend is a program (rlx-inflect-nano's code;
  its WordPiece groups the host's tokenizer), its neural G2P — Rust in the
  runner — rlx graphs (`g2p_encode`, `g2p_decode`, one GRU step each).
  OpenVoice is MeloTTS' package (`export_with`) with the tone extractor,
  the converter and the spectrogram (`audio::spectrogram`) as graphs,
  cloning `options.prompt`. TinyTTS within 3e-6 of the runner, OpenVoice at
  cosine 0.999999999. `symbolic_in_lifted_many` builds at any number of
  sizes.
- **F5-TTS** (`rlx-f5tts`, feature `export`) and **ZipVoice**
  (`rlx-zipvoice`, feature `export`) as packages rlx runs alone, cloning
  `options.prompt`. F5's three graphs are symbolic in the reference's
  samples, the ids and the frames — the STFT's frames and the frames past
  the reference's quotients of those — the same samples as `F5Native`
  (off by 0). ZipVoice is LuxTTS' exporter (`export_family`) with its
  sampler's defaults (cosine 0.99999999975 against the runner).
- **Supertonic-3** (`rlx-supertonic`, feature `export`) and **LuxTTS**
  (`rlx-luxtts`, feature `export`) as packages rlx runs alone: each a
  program around rlx graphs symbolic in their lengths — Supertonic's four
  in the text and latent lengths (the same samples as `Supertonic`,
  English and Spanish); LuxTTS's encoder, vocoder head, prompt log-mel,
  ISTFT and a flow decoder per frames mod 4, cloning the voice
  `options.prompt` gives (cosine ≥ 0.999999998 against `LuxTts`).
  `rlx_core::export::audio` gains `log_mel` (an STFT as matmuls by
  `stft_bases`) and `istft_center` (`torch.istft(center=True)`), checked
  against LuxTTS' Rust DSP. `symbolic_in_lifted` lifts a constant holding
  a length (counted out, `sum(ones[n])`) and a countdown (repeats
  reversed), and `symbolic_in_lifted_sizes` takes several symbols.
- **Piper** (`rlx-piper`, feature `export`): a Piper voice as a package
  rlx runs alone, text in, speech out — its frontend a program (espeak-ng,
  the tokenizer, the runner's seeded flow noise, the length regulator), its
  encoder, stochastic duration predictor, prior and decoder rlx graphs. The
  duration predictor, Rust in the runner, is an rlx graph (`sdp_graph`):
  the rational-quadratic spline's inverse for every element, the bin a
  count of the edges below, outside the interval passed through
  (`Op::Where`) — the same durations as `Sdp::durations`. The sample
  counts of `NativeVits` and the samples within 7e-6.
  `rlx_tiny_tts::model::import_rlx_graph` imports an ONNX graph as a
  package holds it (Kokoro's exporter uses it too); `symbolic_in` and
  `symbolic_in_lifted` fit from their three sizes
  (`rlx_ir::dynamic::generalize_checked`) and name the first line a
  mismatch differs in.
- **Kokoro-82M** (`rlx-kokoro`, feature `export`): text in, speech out,
  with no rlx-models code — its frontend a program (espeak-ng, the text
  preprocessor, the vocabulary, the voices, the length regulator, the
  ISTFT normalization; `program/`, wasm32-wasip1), its encoder and
  decoder rlx graphs symbolic in the phonemes and the frames, imported
  once from the runner's ONNX at export. The same samples as
  `NativeKokoro` on two voices (off by 0, at lengths past the ones the
  graphs were generalized from). `graph::symbolic_in_lifted` makes a graph
  symbolic whose builder bakes length-dependent constants: runs, blocks
  repeated with a step (aranges, repeat indices, upsampler staircases —
  clamped too — and interpolation weights, rebuilt in the graph), index
  tables.
- **`export::program`**: build a model's program crate for wasm32 and add
  it to a package as an entry of kind `program` (rlx's WebAssembly host
  logic), for the models whose glue graphs say badly.
- **Moonshine v2 streaming** (`rlx-moonshine-streaming`, feature `export`):
  a `transcribe` package — the CMVN/`asinh` front end and its masks from
  the audio's real length, the encoder (its position rows held once), the
  decoder with no step, one function per 16-token bucket symbolic in the
  encoder's length. Tiny, small and medium transcribe the JFK clip (and a
  cut of it) word for word as `MoonshineStreaming::transcribe` does.
- **TEN-VAD** (`rlx-ten-vad`, feature `export`): a clip of PCM in, a voice
  probability per hop out, with no host DSP — the frontend, the LPC pitch
  estimate (Levinson-Durbin unrolled, summed in the runner's order) and
  its two recurrences as `Op::Scan` (the anti-alias biquad over the
  samples, the Viterbi over the frames), then the network as an `Op::Scan`
  whose body is the runner's streaming graph, reset where the runner
  resets. Within 4e-6 of `TenVad` over 2000 hops. `rlx-ten-vad-core`
  gives the graph its tables (`pitch::tables`, `frontend::mel_matrix`).
- **BioCLIP-2** (`rlx-bioclip2`, feature `export`): one package, opened
  as one model — the entries `image` (a pipeline: OpenCLIP's preprocessing
  as a graph, `export::vision::clip_resize_crop`, Pillow's antialiased
  bicubic of the shortest side with its fixed-point passes and the center
  crop) and `text` (an `embed` entry) — checked against `BioClip2Runner`
  on the real weights. `Pipeline::package` returns the package unwritten,
  so an exporter adds entries beside the pipeline (`Pipeline::functions`,
  `Package::rename_entry`); `Checkpoint` reads an OpenCLIP directory.
- **Vision transformers and text embedders**: DINOv2, DINOv3 and UNI2 as
  image-in `pipeline` packages (`export::vision`: the runners' resize, patch
  tokens and class row as graphs; `vision::export_vit`), EmbeddingGemma and
  jina-embeddings-v5 (`--variant` its task) as `embed` packages with the
  runners' host steps — embeddings, masks, pooling, dense head, last-token
  pooling — in the graph. `graph::Pipeline` for any stages; a `variant`
  export option. Checked against each runner: random weights for the ViTs,
  the real checkpoints for the embedders (within 1.5e-7).
- **Every LM runner that can be expressed**: Llama 4 and MiniMax M3
  (`rlx-llama4` / `rlx-minimax`, feature `export`) as full-sequence packages
  (`decoder::Forward` / `export_forward`: a `generate` entry with no step,
  the prompt function over the whole context each token, as their runners
  do; one function per length where the graph is not symbolic in it), and
  `export::standard`: any dense decoder `standard_decoder` builds from its
  `config.json`, the exporter `detect` falls back to. Each is checked against
  its runner on random weights its test writes (`check::write_checkpoint`).
  Not exported, with why: mllama (host image tiling), qwen4exp (host n-gram
  embeddings from an external table). `docs/exporting.md` maps the other
  crates by what each needs.
- **Checked against the runners** with `rlx-export --check`, greedy tokens
  identical and logits within 3e-6: Qwen3-0.6B (CPU, Metal), Qwen3.5-0.8B
  Q4_K_M, LFM2.5-230M, and random-weight Llama 3 (scaled RoPE) and Mistral
  checkpoints. See `docs/exporting.md`.

### Package exporters: Llama family, Gemma, rerankers, Qwen2.5-VL

- **`rlx_qwen3::export::export_decoder` + `Decoder`**: the Qwen3 exporter
  over a trait (graphs, sizes, RoPE tables, optional check, multimodal hooks),
  so other decoder families reuse it. Checkpoints sharded by
  `model.safetensors.index.json` export (one weight source per shard). Prompt
  functions at fixed batch sizes (`prefill_batches`, 2/4/8);
  `chat_template.jinja` read when `tokenizer_config.json` has none.
- **Exports within memory.** Builders take a lazy mmap loader
  (`SafetensorsMmapLoader`: each tensor read when taken) instead of the
  whole checkpoint widened to f32, and each build's params go straight to
  `package_export::Referencer`, which fingerprints the checkpoint one tensor
  at a time and keeps only the params no checkpoint tensor equals — the
  first build's params are decided before the second is built. Qwen2.5-VL-3B:
  37 GB → 25 GB peak, 332 s → 97 s.
- **Llama / Mistral / Qwen2** through it, family from `model_type`:
  `qk_norm` off for those types and Qwen2's Q/K/V bias on unless the config
  says otherwise; scaled RoPE refused. SmolLM2-135M: the same 20 greedy
  tokens as `transformers` (CPU, Metal).
- **`rlx_gemma::export`** (feature `export`): Gemma 3 text checkpoints,
  both RoPE tables (sliding and full layers'). Sliding layers mask with a
  second step input, `mask_local` (`GemmaFlow::windowed_mask`), which the
  stateful step computes for their window (`Decoder::windows`), so the
  context runs past `sliding_window`. Gemma 3 270M
  (`unsloth/gemma-3-270m-it`, the ungated mirror): `transformers`' 20
  greedy tokens, its chat, batch = alone; an 886-token prompt (window 512)
  within 3.6e-4 of `transformers`' logits and its 20 tokens (CPU, Metal).
- **Fixed: Gemma 3 / 4 decode with fed RoPE rows used the sliding table on
  full-attention layers.** With a custom mask or a dynamic past (every
  bucketed and dynamic decode, `GemmaRunner`'s included) the
  `rope_cos_global` inputs were declared but unread, so full-attention
  layers' keys took the wrong rotation: logits off by up to 7 within 18
  tokens, greedy text drifting after a dozen. Now 9e-5 from the prompt
  graph. (rlx-flow's Gemma decode layer also keeps `score_scale` and the
  logit soft-cap under a custom mask; it dropped them, which mattered for
  Gemma 2.)
- **`rlx_embed::export::export_reranker_rlxp`**: a cross-encoder's classifier
  head (BERT pooler + classifier, or RoBERTa's) appended to the encoder, a
  `score` entry. ms-marco-MiniLM-L6-v2 within 2e-6 of `transformers`.
- **`rlx_qwen3_vl::export`** (feature `export`): dense Qwen3-VL from its
  Hugging Face checkpoint — the vision tower built from IR ops, symbolic in
  the image's grid (one graph for every size): the processor's patch order
  as reshape/transpose, both frames' patch weights summed, the learned
  position grid bilinearly interpolated and the 2-D RoPE rows computed in
  the graph from the grid; LayerNorm, GELU-tanh MLPs, the merger and the
  three deepstack mergers; images of any size resized as the processor
  does, the language model's step and prompt from
  rlx-qwen3's graphs (its `model.language_model.*` tensors renamed), and a
  prefill adding the deepstack features after layers 0-2. Interleaved
  multimodal RoPE. Qwen3-VL-2B-Instruct on a 448×448 image: the same
  sentence as `transformers` (18 tokens), CPU and Metal. Video: a second
  tower for frame pairs (the frames' patch weights apart), each pair its own
  placeholder run. Qwen3-VL-MoE (`qwen3_vl_moe`, every layer sparse): the
  experts through rlx-flow's `Qwen3Moe`; a random-weight checkpoint's image
  and video prefill logits within 2e-5 of `transformers`', CPU and Metal.
- **Configs saved by transformers 5 keep their RoPE.** transformers 5 writes
  `rope_theta` and the scaling (`rope_type`, `factor`, …, `mrope_section`)
  only inside `rope_parameters`; readers of the older layout fell back to
  their default theta and no scaling, rotating every position wrong
  (Qwen3-VL-MoE logits off by 1.7; a Llama 3.2 would lose its `llama3`
  scaling). `rlx_core::config::legacy_rope_keys` / `from_hf_json` restore
  `rope_theta`, `partial_rotary_factor` and `rope_scaling` where missing;
  used by qwen3, llama32, llada2, ling, motif, locateanything, qwen3-tts,
  sesame, tada, dinov3, mimi, minimax-h3, minimax (M3), nemotron and
  unlimited-ocr.
- **`rlx_qwen25_vl::export`** (feature `export`): Qwen2.5-VL from its
  Hugging Face checkpoint alone — the vision tower from `visual.*` (no mmproj
  GGUF), symbolic in the image's grid: window attention as a bias over each
  patch's window id (the same attention as HF's reordering into windows,
  tokens left in place), RoPE rows from the grid, the pixels' normalization
  in the graph; any image size, resized as the processor does; token
  embeddings; a prefill from embeddings and multimodal RoPE rows; the step
  reading its tables at `position + rope_offset`. Video: a frame-pair tower and one
  placeholder run, pair `g` at time position `g · tokens_per_second` (the
  processor's 2 fps).

### rlx-kimi-k3: training, inspection, distributed training, sizing

- **Training** (`train`, `train_stream`, `layer_exec`): full-weight and LoRA
  next-token training. `stream` keeps one layer resident and backpropagates
  layer by layer through vector–Jacobian products, MoE layers over a compact
  bank of the experts they picked; `whole` trains the whole graph through
  `rlx_tune::Trainer` (data-parallel via `GradComm`). Streamed gradients equal
  whole-graph gradients for every trained param; whole-graph gradients match
  central finite differences. PEFT adapters, merged or full checkpoints that
  write only the changed tensors and link the rest, resumable state.
- **Canonical params** (`common::CanonicalParams`): graphs built for training
  or dumps have one param per checkpoint tensor (fusions done in-graph), mapped
  back by `export::hf_name`; the router trains through a differentiable gather.
- **Inspection** (`inspect`, CLI `weights` / `trace` / `graph`): weight
  inventory, statistics and dumps (MXFP4 dequantized); every named
  intermediate per layer, expert picks, and per-layer input gradients.
- **Distributed** (`dist`, rewritten): one worker serves prefix, decode and
  training; heartbeat frames bound every read; config and layer-range agreement
  checked before work; thread per connection; per-sequence decode state with
  pipelined `generate_batch`; GPipe-style `PipelineTrainer` matching
  single-node training step for step.
- **Decode speed**: MoE layers reuse their dense weights and compiled router /
  experts graphs across tokens instead of re-reading, re-transposing and
  recompiling them every token, plus an optional dequantized-expert LRU
  (`RLX_KIMI_EXPERT_CACHE`): 2.2× less CPU per decode token (3.1× with the
  LRU) on `examples/decode_bench`, bit-identical.
- **Resident KDA state**: each KDA layer's recurrent state stays in its cached
  decode graph between tokens instead of a host round trip per token (~0.9 GB
  per token on the real model); multiple sequences share a cache by parking.
  Bit-identical on CPU and Metal; −27% CPU per token on the CPU benchmark.
- **Config-derived dims everywhere**: the cluster examples no longer hard-code
  the model dimensions; `source::verify_flow_dims` checks `config.json` against
  the checkpoint's tensor shapes whenever a checkpoint is opened.
- **Dense residency** (`common::set_dense_residency`, `--dense f32|bf16|int8`):
  first-class instead of env-only. bf16 residency was wrong on CPU (relative
  error ~1.4 per layer) until the upstream rlx-cpu `Concat` fix for 2-byte
  types; it needs an upstream release containing that fix.
- **KDA state log** (feature `state-log`): per-token recurrence inputs plus
  periodic snapshots; every KDA state rebuilds bit-exactly.
- **Sizing** (`sizing`, CLI `size`): node counts per resource and a roofline
  speed estimate from the config alone.
- **Shared code moved upstream**: the pipeline's RPC (heartbeat-framed request /
  reply, thread-per-connection server, checked codec) and stage planning now
  live in `rlx-distributed` (`rpc`, `stages` — planning on its `LayerSplit`); the
  sizing roofline in `rlx-distributed::cluster::sizing`; the optimizer is
  `rlx-optim`'s AdamW (two groups, decayed / exempt). `rlx-kimi-k3` keeps only
  the model-specific parts and needs an upstream release containing them.
- **Synthetic checkpoints** (`synth`): the real tensor layout at toy size, so
  every checkpoint path is testable without the 1.6 TB model.
- Fixed: the `kimi_k3_cluster` example builds again (it predated the current
  `rlx-distributed` `ModelCost` fields).


### Release prep: no upstream blocker left

`docs/releasing.md` named `rlx-torch-ckpt` as the *current* hard blocker — an
upstream crate that had never been uploaded, which makes the whole workspace
unresolvable once the local path patch is removed (cargo resolves dev-deps at
lock time, so a fresh clone cannot even load it). **It shipped: 0.2.16 and
0.2.17 are both on the sparse index**, as is `rlx-opscope`, the previous
blocker. `check-publishable.py --release` now reports only the two clean-tree
problems, which are the uncommitted working tree and nothing to do with
publishability.

**34 manifests now inherit the workspace version.** They carried a literal
`version = "0.2.16"` instead of `version.workspace = true`, which is the trap
the 0.2.16 bump already paid for once: a workspace-table-only bump misses them
and the crates silently ship at the old number. All 34 read exactly the
workspace version, so the conversion is a no-op today — verified by diffing
`cargo metadata` before and after, all 212 package versions byte-identical —
and the next bump is one line instead of 35. The three crates left alone are
in `[workspace.exclude]` and have no workspace to inherit from; one of them,
`bench_matmul_rlx`, also pins `rlx-ir` and `rlx-runtime` at a literal
`0.2.16` for the same reason, which is the one place the documented
direct-upstream-pin trap still lives (`publish = false`, so it only affects
opt-in manifest-path builds).

So the gates stand as: `rust-lint-gate.sh --workspace` ok — `cargo fmt --all
-- --check` clean, `clippy --workspace --all-targets -D warnings` clean,
rustdoc `-D warnings` clean — `check-models-catalog.py` OK (184 families / 16
categories, and the hand-maintained counts in `README.md` and `MODELS.md`
agree — memory says those drift every release; they did not this time),
`check-publishable.py` OK across 200 publishable crates, and
`publish.sh --list` validating tier coverage with no gaps — including the two
new untracked crates, `rlx-fgd` (correctly `publish = false`) and
`rlx-qwen4exp`.

`lint-all` deliberately omits feature states, so `check-feature-states.py`
was run separately over the eight crates whose feature-gated code changed
this session — `rlx-models-core`, `rlx-gemma`, `rlx-llama32`, `rlx-qwen3`,
`rlx-qwen3-vl`, `rlx-mistral`, `rlx-minimax`, `rlx-qwen35` — all ok in both
states. The `--changed` sweep was abandoned rather than waited out: 149
modified files span thirty-odd crates at two full compiles each, and the
crates it would have added are ones nothing in this session touched behind a
feature gate.

**One decision deliberately left to be made rather than made implicitly.**
Upstream is **0.2.17** while this workspace is 0.2.16 and pins upstream at
`^0.2.16`, which admits 0.2.17 — and the local `../rlx` checkout *is* 0.2.17.
So the patched local build and a registry build resolve identically (no
pin-vs-checkout drift of the kind that bit the 0.2.14 cycle), but rlx-models
0.2.16 would ship built against upstream 0.2.17. Either bump both to 0.2.17 or
pin `=0.2.16` on purpose; now written into `docs/releasing.md` as a
precondition instead of being rediscovered next cycle.

Also fixed in this file: the entry recording that deduplicating the LM
builders was *not* done still sat above the entry recording that it was.

### `microsoft/AesCode-8B` / `-32B` — stock Qwen3-VL, and the wiring they needed

AesCode generates infographics (slides, posters, dashboards) as editable
HTML/CSS from a prompt plus an image-generated aesthetic reference.
Architecturally there is nothing new: **AesCode-8B matches
`Qwen/Qwen3-VL-8B-Instruct` field for field and AesCode-32B matches
`Qwen3-VL-32B-Instruct`** — text dims, `rope_theta` 5e6, `mrope_section
[24, 20, 20]` with `mrope_interleaved`, and a tower of depth 27 / hidden 1152
/ patch 16 / merge 2 / deepstack `[8, 16, 24]`. Each README names the base
model, bartowski's GGUFs record it in `general.base_model.0.name`, and the
chat template is byte-identical to stock Qwen3-VL's. So this is identify-then-
wire, per `docs/adding-a-model.md`, not a port. All of it established from
`config.json` and GGUF headers read over HTTP range requests — nothing
downloaded.

**The gap that mattered: a Qwen3-VL safetensors directory resolved to
nothing.** `rlx-qwen3-vl`'s registry entry listed four GGUF arch spellings
and an **empty** `hf_model_types`, and only the GGUF list is consulted for a
GGUF. So `AesCode-8B-Q4_K_M.gguf` (arch `qwen3vl`) found the runner while the
original safetensors checkpoint (`model_type: qwen3_vl`) found no runner at
all. Now `&["qwen3_vl", "qwen3_vl_moe"]`, with both directions asserted in
`auto_runner_coverage`.

**The vision config was reading four fields fewer than it was given.**
`spatial_merge_size`, `temporal_patch_size`, `num_position_embeddings` and
`deepstack_visual_indexes` were all silently dropped — serde ignores unknown
fields — so a real Qwen3-VL config parsed *successfully* into a geometry this
crate's graph cannot compute. That graph emits one embedding per patch from
the final layer only; a Qwen3-VL tower merges 2x2 patches and feeds three
intermediate layers to the LM as well. Run it anyway and the language model
gets **4x too many vision tokens drawn from the wrong features** — it answers
confidently about an image it effectively did not see.

The fields are read now, and `Qwen3VlVisionConfig::validate_supported`
refuses, naming the merge factor, the token multiple, and where the real
implementation is (`rlx-qwen35`'s `vision/` module, which `rlx-qwen3-vl`
already depends on). `rlx-qwen4exp`'s mmproj reader already refused deepstack
for the same reason, so this makes the two agree. A refusal is the right
answer here: the alternative is a VLM that looks like it works.

**So what runs today.** The text path: `qwen3_vl`/`qwen3vl` both resolve, and
`Qwen3Config::from_file` parses either AesCode out of its nested
`text_config` with `rope_theta` 5e6 intact (pinned by a test — a defaulted
1e6 would be a model that degrades with context rather than one that fails).
For text-only input Qwen3-VL's interleaved M-RoPE degenerates to the 1-D RoPE
`rlx-qwen3` implements, so the trunk is correct, not approximated. The vision
path refuses. Since AesCode's whole premise is the image reference, **these
models are not usefully runnable until the tower is ported** — stated plainly
in `MODELS.md` rather than implied by a green test.

**Also corrected:** `rlx-qwen3-vl` was catalogued as `STUB (PLAN.md M7)` in
both `MODELS.md` and its `Cargo.toml` description. That undersold half of it
and oversold the other: the LM integration, `LmRunner` impl,
`generate_multimodal` plumbing and image preprocessing are real, while the
tower is the incomplete part. Both now say which half is which.
`check-models-catalog.py` enforces that the two agree, which is how the stale
description surfaced.

One piece of dead code went with it: `Qwen3VlRunner::build`'s no-HF-config
fallback built a `Qwen3VlVisionConfig`, threw it away with `let _ = cfg;`, and
then built the identical struct again inline.

### The LM builders now share one definition — and the reason they could not before

`rlx-gemma`, `rlx-llama32` and `rlx-qwen3` each declared the same nine
builder fields and wrote the same setters over them. Those fields are now
`rlx_core::lm_builder::LmBuilderCommon<Cfg>` and the setters come from
`rlx_core::lm_builder_setters!(FamilyConfig)`. The public chain is unchanged —
`Qwen3Runner::builder().weights(..).device(..).max_seq(..)` still compiles
exactly as before.

**Why upstream's builder was never adopted — it is not what I first thought.**
`rlx_runtime::LmRunnerBuilder<Cfg>` has these field names already and has had
zero users in this workspace. I assumed the blocker was that it stores
resolved values (`device: Device`) where the families store `Option<_>`, and
`None` is what means "pick a device for this machine". That is true and it
matters. But the harder reason only surfaced when the macro failed to compile:
**it is typed against a different `SampleOpts` and `WeightFormat`.** This
workspace has two of each — `rlx_qwen3::SampleOpts` beside
`rlx_runtime::SampleOpts`, and `rlx_cli::WeightFormat` beside
`rlx_runtime::WeightFormat` — same names, distinct types. Every family builder
uses the first of each pair; upstream's builder uses the second. Adopting it
means converting between two identically-named types at every call site.

So `sample` and `format` stay family-local: their types live in `rlx-qwen3`
and `rlx-cli`, both of which depend on `rlx-models-core`, so naming them in
the shared struct would be a dependency cycle. **Collapsing the two
`SampleOpts` (and the two `WeightFormat`) into one type each is the
prerequisite for finishing this**, and it is a wider change than a builder
cleanup — recorded rather than started.

**Two divergences the merge exposed**, both preserved rather than quietly
changed:

- **`.build()` without `.device(..)` runs gemma on the GPU and llama32,
  qwen3 and qwen35 on the CPU.** gemma calls `pick_lm_device` and logs it;
  the other three do `unwrap_or(Device::Cpu)` silently. That reads downstream
  as "gemma is mysteriously faster". The shared type offers both policies as
  named methods — `device_or_auto(tag)` and `device_or(default)` — so each
  call site states which it uses instead of burying it in an `unwrap_or`.
  Converging them changes the default device for three crates and belongs in
  its own commit.
- **`packed_weights` means different things.** gemma and llama32 auto-enable
  it for a GGUF over 256 MiB (gemma forces it off for multimodal); qwen3 does
  too, with an extra Metal small-model exception — and qwen3's doc still
  claimed "Default false". Behaviour, not duplication, so each family keeps
  its own field, setter and doc.

One improvement folded in rather than lost: qwen3's memory-cap error carried a
second sentence telling the caller what to do about it ("raise the cap or pick
a smaller model") that gemma's and llama32's copies lacked. The shared
`check_memory_cap` keeps it, so all three now say it.

**`rlx-qwen35` is deliberately left out.** It shares five of the setters but
has no `max_memory_gb` and no `stream`, so adopting the macro would *add* a
`max_memory_gb` setter that its `build()` ignores. A memory cap that silently
does nothing is worse than the duplication it would remove.

**Guarded.** Each of the three crates gets
`every_setter_still_exists_and_lands_where_build_reads_it`, which calls the
whole chain and asserts where each value landed — plus that unset stays unset,
which is what keeps auto-device reachable. A compile cannot catch a setter
wired to the wrong field of the right type (`max_seq` into `max_memory_gb`);
sabotaging exactly that fails all three tests. `LmBuilderCommon` has five of
its own covering both device policies, the cap, and the "call `.weights(...)`"
error.

This is not a line-count win: ~100 lines of duplicated fields and setters
collapse into a 296-line module that is half documentation and tests. The win
is one definition, one set of defaults (they had already drifted — `stream`
was `true` in llama32 and `false` in the two that reached it through
`#[derive(Default)]`), and the two findings above written down where the next
person will hit them.

### Host-driven decode went from 3 families to 8

`LmRunner::{prefill_logits, decode_logits}` is the pair that hands the decode
loop to the caller — its own sampling, logit bias, log-probs and stop
detection — and it is what `rlx-serve` drives. Only **llama32, qwen3 and
qwen4exp** implemented it. Everything else fell through to `generate`, which
means the family's own `SampleOpts` instead of the caller's and no per-token
logit access at all. The `HostDriven` adapter makes that fall-back work, which
is exactly why nobody noticed it was happening.

In every case added here the capability already existed one layer down and
only the trait wiring was missing:

- **gemma** — `GemmaGenerator::prefill_get_last_logits` / `decode_get_logits`
  have been there all along (and are validated by
  `incremental_decode_logits_match_full_prefill`).
- **mistral** — wraps `Llama32Runner`, which has both halves; the wrapper
  just never exposed them.
- **lfm-vl / mistral-vl / qwen3-vl** — forward to their inner text runner.
  Forwarding is right even when an inner family lacks the pair: the trait
  default and the inner's refusal are the same "unsupported" error, so a
  probing consumer sees no difference, while a family that *does* support it
  stops being hidden behind the wrapper.

**The two left out are left out on purpose**, now said so in code.
`minimax-m3` re-forwards the whole context and keeps no KV cache, so a
one-token `decode_logits` has nothing to advance — a KV cache is the
prerequisite, not trait wiring. `qwen35-spec` *is* a decode loop: the draft
proposes, the target verifies, and the accept/reject sampler decides how many
tokens a step yields; a caller stepping one token at a time would get the
target model with none of the speculation.

**A test I wrote and then deleted.** I added a greedy end-to-end comparison —
host-driven loop versus the family's own decode — and a sabotage walked
straight through it: gemma's synthetic test config decodes a *constant*
(`[8, 8, 8, 8]`), so feeding `decode_get_logits` the wrong token changed
nothing. The pre-existing `incremental_decode_logits_match_full_prefill`
already guards the composition that matters and does not depend on the output
being varied, so the weak test is gone and that one now says what it guards.
A test a sabotage ignores is worse than no test, because it reads as coverage.

### `stream` no longer gates anything in llama32, gemma or qwen3

Three crates that share eleven identical builder setters disagreed on what
`stream` defaults to: llama32 set `true` explicitly, gemma and qwen3 got
`false` from `#[derive(Default)]`. Chasing which was right turned out to be
the wrong question — qwen3 had already retired the field as vestigial
(`#[allow(dead_code)]`, setter kept for API compatibility) when its callback
became the only token sink.

One stale doc fell out of it: qwen3's `stream()` setter promised "Default
true. Setting false makes `generate` collect all tokens before returning" —
both halves wrong, since the builder derives `Default` (so it was `false`)
and the collecting branch was already gone. All three setters now say they
are accepted and ignored. The one in-repo caller that actually wanted
streaming, gemma's multimodal CLI, was passing `.stream(true)` explicitly, so
nothing was silently broken — the knob is simply moot now.

gemma and llama32 now match. Every `stream(false)` branch was the same
collect-then-replay shape: generate all tokens, *then* replay the callback
over them. Identical ids in identical order — the only difference was that a
caller asking to be told per token was told at the end instead. It is also
why the dropped stop bool was invisible: on that branch a stop could not have
taken effect even if the bool had been read. One loop now, either way, and
the field is documented as vestigial in all three.

The builders were deduplicated after this, against the reservation recorded
here first — see *The LM builders now share one definition* above, including
the reason that reservation was aimed at the wrong obstacle.

### Five families ignored `LmRunner::generate`'s stop signal

The trait callback is `FnMut(u32) -> bool`, where `false` means stop. Fixing
qwen3's impl (which discarded it — see the entry below) raised the obvious
question, and the answer was that the same `let _ = on_token(tok)` sat in four
more places. An audit of every `impl LmRunner for` in the workspace:

| family | before | now |
| --- | --- | --- |
| `rlx-gemma` | dropped the bool, no stoppable loop existed at any level | stoppable loop in the generator, the packed session and the runner |
| `rlx-llama32` | dropped the bool — though `generate_until` was right there | trait impl routes to `generate_until` |
| `rlx-mistral` | dropped the bool (wraps llama32) | `generate_until` forwards to llama32's |
| `rlx-qwen35` (spec) | dropped the bool, deliberately | honoured; the stated reason does not hold, see below |
| `rlx-qwen3` | the MLX-packed route still dropped it | `generate_packed_stoppable` |
| lfm-vl, mistral-vl, qwen3-vl, qwen4exp | already forwarded | unchanged |
| `minimax-m3` | no `generate` impl | unchanged — the trait default honours it |

What it cost a caller: generation always ran the full `n_new`, so EOS, a user
stop string, a stray `</think>` and a degenerate-repetition guard were all
unreachable; `finish_reason` was permanently "length"; and every turn paid for
tokens the caller had already rejected.

**Who was affected, precisely.** `rlx-serve` was not: it drives
`prefill_logits` / `decode_logits` itself and owns its token loop and its stop
matching (`serve/stop.rs`). The callers that pay are the ones that hand the
loop to the family — the CLI dispatch, `rlx-llm-bench`, `rlx-eval`, the
per-crate chat examples, and out-of-tree consumers. Worth stating because
"the server is fine" is the reason this survived: the most-exercised path in
the workspace does not go through `LmRunner::generate` at all.

While checking callers, one near-miss: `rlx-dflash`'s `DflashLoop::generate`
takes `stop: FnMut(u32) -> bool` where **`true`** means stop, the opposite of
`LmRunner`'s `on_token`, and two of its examples pass `|_| false` meaning
"never stop". Different trait, so nothing broke — but the two conventions
living one crate apart is a trap, and `|_| false` means opposite things in
files a few directories from each other.

**Two findings worth keeping.** llama32's and gemma's non-streaming branches
collected every token *first* and only then replayed the callback — so a stop
could not have taken effect there even if the bool had been read. A stoppable
generate has to go through the decode loop, not around it, and both now use
one loop regardless of `stream`.

And qwen35's deferral — "speculative-decode stop semantics need to interact
with the accept/reject sampler" — does not survive reading the loop: it
*already* abandons the rest of a verified batch whenever `n_new` falls inside
one, so mid-batch truncation is a state the decoder has to tolerate either
way. Honouring the signal costs nothing that the budget path did not already
cost.

Every new loop uses the contract the trait default and llama32's
`generate_until` already documented: `false` halts *after* that token, which
is included in the result (the caller has seen it, and callers splice on what
they were handed).

**Guarded where it is cheap to guard.** `generate_cached_until_stops_when_the_callback_says_so`
in `rlx-gemma`'s generator tests runs on a tiny synthetic model — no weights,
no `#[ignore]` — with a companion test that an always-`true` callback still
spends the whole budget, so the stop is a stop rather than an off-by-one. Two
sabotages confirm it bites: dropping the bool (the original bug) and checking
it before the token instead of after. qwen3's real-weight
`lm_runner_honours_stop` test passes on a Qwen3-0.6B GGUF, and is worth noting
for what it did *not* cover: `packed_weights(true)` on a GGUF builds a native
packed *generator*, so it never reached the `PackedForward` route where the
remaining qwen3 bug was. A passing stop test is not evidence that every packed
path stops.

### Tooling — `just lint-features`, for the states no gate builds

`just lint` builds **default features only**, so code behind an optional feature
is compiled by nothing. That hides a specific, repeatable break: a field
declared `#[cfg(feature = "x")]` and used without the gate compiles for whoever
turned `x` on and fails the build for everyone else. A 2026-08 sweep found ~90
such errors; `rlx-qwen3`'s `kv_store` (fixed below) was another, and it broke
the default build of `rlx-serve` and everything downstream of it.

`scripts/check-feature-states.py` lints each package twice — no features, then
all of its own features except those needing a toolchain the host lacks (CUDA,
ROCm, Vulkan, MKL are *skipped*, not failed: a red build there means "you do not
have CUDA", which is not a finding). Three ways to call it:

    just lint-features rlx-qwen3     # named packages
    just lint-features --changed     # whatever you are editing
    just lint-features --all         # every package with features

It is deliberately **not** in `just lint`: the full sweep is minutes of
compilation, and the point is to make the check cheap to run on the crate in
front of you.

The states it checks are a crate's **default** features and its default
features **plus every optional extra**. It deliberately does not test
`--no-default-features`: most crates here declare a default their core path
requires (`rlx-whisper` is `default = ["tokenizer", "timestamps"]` with the
tokenizer behind a `dep:`), so a featureless build fails *by design* — 11
errors for whisper — and reporting that is noise, not a finding. `--bare` opts
in for a crate that genuinely claims to support it. The first version of this
script got that wrong and spent its first run excavating an unsupported
configuration.

What the supported states do catch, and the default gate cannot: 5
collapsible-`if` errors behind optional features across `rlx-qwen3`,
`rlx-qwen3-tts` and `rlx-whisper`, plus feature-gated imports that are unused
without their feature (`rlx-qwen35`'s `anyhow::Context`, four of
`rlx-whisper`'s). All fixed; those crates now pass both states.

### `rlx-qwen3` — FIXED: the default build was broken by a feature-gated field

`Qwen3Generator::kv_store` is declared `#[cfg(feature = "mmap-kv")]`, and the
cache-invariant check in the decode step read it ungated, so `cargo build -p
rlx-qwen3` failed without that feature — and with it `rlx-serve`, which depends
on it. Replaced with `kv_store_absent()`, which answers `true` where there is no
store to attach, preserving the documented intent: the strict prefix check
applies only to an unspliced cache. Verified in both feature states, which is
the half that is easy to skip — gating a use site is how the mirror-image
`unused` error gets introduced.

### `rlx-qwen3` — `synth` + the pipeline stage's first automated test

`Qwen3PipelineDecodeStage` had **no** automated coverage: five examples drive
it, every one needing real weights. That is a bad place for a distributed
stage, because its failures live at the seams — which tensor crosses a cut,
which rank owns the embedding and which the head, how the KV cache is sliced —
and each produces plausible, well-scaled, wrong logits.

`tests/pipeline_equivalence.rs` now compares a 1-, 2- and 4-way split against
the monolithic generator's logits, driving the `BlockRunner`s by hand in rank
order (no transport needed), on the glm5next pattern. Two sabotages confirm it
bites: relaying ranks low-to-high instead of high-to-low (rank 0 owns the head)
diverges by 1.0 relative, and the monolithic reference had to be sliced to its
last position — `sequence_logits` returns every position, which the first
version of the test got wrong and the length assertion caught.

That needed weights an integration test can reach, so `rlx_qwen3::synth` is a
**public** module — the same reason `rlx-qwen4exp::synth` is. It is not a move
of the existing helper: `flow.rs`'s test-private `synthetic_weights` is
single-layer and **all zeros**, which is fine for checking that a graph builds
and useless for checking that two paths agree, since with zero weights every
path produces the same constant. `synth` gives N layers of deterministic
non-zero weights, seeded per layer.

A third sabotage found a blind spot worth recording: with *identical*
per-layer weights both equivalence tests still pass, because an interchangeable
layer stack cannot reveal a layer-order mistake. The thing that catches it is
`synth`'s own `each_layer_has_its_own_weights`, so that unit test is
load-bearing for the integration test rather than a formality — stated in the
test's module docs.

### `rlx-qwen3` — the pipeline stage folded onto `FamilyPipelineStage`

llama32 reaches the shared `PipelineFamily` abstraction in a 68-line impl;
qwen3 hand-rolled 886 lines of `seed` / `decode_recompile` / `decode_bucketed`
beside it. Now gone: `Qwen3Family` is 95 lines of two graph builders, four
shapes and `decode_inputs`, and `Qwen3PipelineDecodeStage` is a type alias for
`FamilyPipelineStage<Qwen3Family>` with `stage()` / `stage_with_layers()`
constructors — the same shape `rlx-llama32` already had, so the two families
now read alike. The file is 886 → 672 lines and the shared stage is the only
copy of the KV cache, the decode step, the weight filter and the bucketed
compile cache.

**What made it look harder than it was.** qwen3's stage renumbered every
per-layer tensor to a local index (`model.layers.{g}` → `model.layers.{g-start}`),
cloning the whole block's weights per rank, because its graph builders count
layers from zero. `Qwen3DecoderStage` and `Qwen3DecodeLayerStage` turn out to
carry `layer_prefix` (weights) and `layer_idx` (cache slot) as separate public
fields, so a layer can read global weight names while keeping a local
`past_k_{i}` slot — no upstream change, and the shared stage's *default*
`block_weights` then fits qwen3 exactly.

**The guard had a hole, and the hole was the tolerance.** The equivalence test
above drives each stage once, which only ever exercises `seed` — so the two
decode paths, where a generation spends all of its time, were uncovered.
`a_decode_step_matches_prefilling_the_longer_prompt` now relays two decode
steps at world 1/2/4 through *both* the recompile and the bucketed-cache path
and compares against prefilling the longer prompt.

Measuring it first is what made it a real test. On the CPU every prefill width
here is **bit-exact** against the monolithic graph and the bucketed decode
differs by 4.7e-7, while a decode position stuck at zero every step — the
classic way to break this — moves the logits by only **1.5e-4** relative at
this size. The 2e-3 tolerance the prefill tests were written with therefore
passes a model that re-attends position zero forever: sabotage confirmed it,
twice, before the tolerance came down to 1e-5. Both sabotages (frozen position;
every layer of a block reading the block's first layer) now fail only the
decode test, which is the discrimination worth having.

### `rlx-qwen3` — Metal parity tests failed a plain `cargo test`

`tests/metal_decode_graph_parity.rs` is gated on macOS but not on the `metal`
feature, and `Session::new(Device::Metal)` *panics* on a build without the
backend rather than erroring — so 10 of its 11 tests failed `cargo test -p
rlx-qwen3` on a Mac, for want of a backend rather than a parity break. Now
skips with a message, the way the other backend-parity tests here do, and still
passes 11/11 under `--features metal`.

### `rlx-llama32` / `rlx-qwen3` / `rlx-qwen35` — one single-bucket decode cache, not seven

`BucketedCompileCache::new(device, vec![1..max + 1])` — one bucket covering
every past length — was written out at **seven** places: llama32's generator,
qwen35's runner (twice), the shared pipeline block, qwen3's pipeline stage, and
a parity test. Worse than the repetition, it had been *dealt with* five
different ways: three rewrites to dodge
`clippy::single_range_in_vec_init` (which suggests `(1..n).collect()`, a type
error here since `new` takes `Vec<Range<u64>>`) and two
`#[allow]`s with their own explanatory comments.

Now `rlx_core::flow_util::single_bucket_decode_cache(device, max_past)`, with
the `#[allow]` and its justification in the one place that constructs the
thing. Every 1-based call site routes through it.

Writing the test for it found a bug the duplication had been hiding:
`max_past = 0` produces `1..1`, an **empty** range that matches no length, so
the cache silently never hits — a performance cliff with no error anywhere.
Every existing call site happened to clamp with `.max(1)` before calling; the
helper now clamps so the next one cannot forget.

Not unified: `rlx-whisper`'s `vec![0..2]`, which seeds a power-of-two ladder in
a loop rather than being a single bucket — different lower bound, different
purpose, and its `#[allow]` is legitimate.

Also formatted `rlx-llama32` and `rlx-qwen3` (13 `cargo fmt` diffs across
`flow.rs`, `lib.rs`, `pipeline_block.rs` and a new `lm_runner_honours_stop.rs`
test). Clippy was already clean in both crates in both feature states.

### `rlx-models-core` — `lm_registry`: any crate can add a model to the generic path

The dispatch behind `auto_runner` was a hardcoded `match runner_name { .. }`
inside the `rlx-models` facade. Three costs, one of which had already been paid
twice:

* **An out-of-tree crate could not add a model.** Its options were to patch the
  facade or to tell users to call its builder directly, giving up every caller
  that goes through the generic path.
* **Every new family meant editing the facade**, and that edit is easy to
  forget: `rlx-qwen4exp` and `rlx-minimax` both shipped complete `LmRunner`
  implementations that nothing could reach, because their own test suites
  passed without the arm.
* **Using the generic path meant linking ~200 crates**, since the facade
  depends on every family it might dispatch to.

`rlx_core::lm_registry` inverts it. A constructor is a plain
`fn(&Path, &LmBuildOptions) -> Result<Box<dyn LmRunner>>`, and any crate
registers one for its own name:

```rust
register_lm_runner(LmRunnerRegistration {
    runner: "mine",
    build: |path, opts| Ok(Box::new(MyRunner::open(path, opts.device)?)),
});
```

`auto_runner` then builds it like a builtin, and `LmBuildOptions::servable()`
gives it the host-driven decode pair even if it implements only
`predict_logits`. An external crate needs `rlx-models-core` and `rlx-runtime`,
not the facade. `AutoOptions` is now an alias of `LmBuildOptions` so the type
is nameable without the facade; registration is idempotent, and a later
registration of the same name overrides a builtin, which is what a host
embedding this needs.

The eleven builtin families are registered by `register_builtin_lm_runners`
(idempotent, called from `auto_runner_with`), each as a two-line wrapper over
the existing dispatch body — so the arm logic is unchanged and the refactor is
a rewiring rather than a rewrite.

**Capabilities are declared on the registration.** `with_mmproj()` marks a
family whose constructor accepts a vision encoder; `build_lm_runner` then
refuses an mmproj for the others centrally, appending that family's
`mmproj_note` to say what to do instead. This replaced **four** hand-written
`if mmproj.is_some() { bail!(..) }` checks inside the dispatch — and the
scattered version had a worse failure than duplication: a newly registered
family that forgot the check silently *ignored* an mmproj the caller asked for,
which looks like success. `lm_runner_supports_mmproj(name)` exposes the same
answer without building anything, and a test pins the declarations against
which arms actually thread an mmproj (verified: dropping `with_mmproj()` from
`qwen35` fails it).

`LmRunnerRegistration` is `#[non_exhaustive]` with a `new()` + builder-method
constructor, so the next capability field does not break every registrant. That
has teeth already: the coverage test is a separate crate and could no longer
use a struct literal, which is the intended protection.

The payoff in tests: the set is **enumerable**, so
`every_registered_constructor_is_reachable_by_a_sniff` replaces two
hand-maintained lists with a real question — can anything actually select each
family. It catches a mis-registered name (verified: renaming one registration
to `minimax-m3-TYPO` fails it), which a list of expected names never could.

### TPU accepted for causal LMs (`tpu` feature)

`Device::Tpu` was rejected for every LM — `validate_lm_device` allowed only the
seven standard backends plus `Device::Ane` under `coreml`, so `--device tpu`
failed at validation even though `rlx-tpu` (5.6k lines, libtpu via PJRT) exists
upstream and `rlx-runtime` already exposes it as `tpu = ["cpu", "dep:rlx-tpu"]`.
SAM has accepted it all along via `validate_sam_device`.

Now gated exactly like CoreML/ANE: `validate_lm_device` accepts `Device::Tpu`
when the crate is built with `tpu`, and without it says which feature to
enable rather than calling the device unsupported. The distinction matters —
`OpenGl` and `Hexagon` have no backend at all and stay refused unconditionally,
while TPU is a backend this build simply did not compile in. `is_lm_device`
(the device *class*, used for help strings and menus) accepts TPU regardless of
features, because hiding an enableable backend from a `--device` list is the
wrong answer; `LM_DEVICE_NAMES` lists it.

Forwarding features added to `rlx-models-core` and `rlx-qwen4exp`, and the
facade's pre-existing `tpu` feature extended to reach them. Deliberately narrow
rather than enumerating every model crate.

`rlx_qwen4exp::runner::scan_state_for` needed no change and is right by
construction: it promotes only CPU/Metal/wgpu to `ScanState::InPlace`, so TPU
gets `Portable` — the correct conservative default for a backend whose
in-place carry semantics are unverified.

**Untested on hardware.** There is no TPU here, so what is verified is that the
device is accepted, routed and refused correctly in both feature states —
`tests/auto_runner_coverage.rs` asserts opposite outcomes with and without the
feature, and `device_capabilities`' own tests cover the class/validation split.
Whether a graph actually executes on libtpu is unproven by anything in this
repo.

### `rlx-models` — the universal runner could not reach `qwen4exp`, and returned runners a server cannot drive

`rlx_models::run::auto_runner(path)` sniffs a checkpoint and builds the right
family's runner. Two gaps made "run anything" less than true.

**`qwen4exp` was unreachable.** The crate was written, tested and documented
without ever being registered: `general.architecture = qwen4exp` sniffed to
nothing, so the one entry point that is supposed to run any checkpoint could
not run the model whose own suite has 117 tests. Now registered (arch
`qwen4exp`, HF `qwen4_exp` / `qwen4_exp_text`), added as an optional dependency
of the facade behind feature `qwen4exp`, and given a dispatch arm.
`tests/auto_runner_coverage.rs` asserts that **every** dispatch arm has a
registry entry pointing at it, so the next crate cannot be forgotten the same
way — the crate's own tests all pass without it, which is why this went
unnoticed.

**A returned runner may not be drivable.** `auto_runner` hands back the
family's own runner, whose host-driven pair (`prefill_logits` /
`decode_logits`) defaults to *unsupported*. New `auto_runner_servable` wraps
whatever was built in `HostDriven`, so one call goes from a path to something
`rlx_serve::run_generation` can drive whatever the family implements.

**Every runner ran on the CPU.** The dispatch never called the builders'
`.device()`, and `auto_runner`'s signature did not mention a device, so there
was no way to ask for one — on a Metal machine every family built through the
multiplexer ran on the CPU, silently. New `AutoOptions { device, mmproj,
servable }` and `auto_runner_with(path, &opts)` is now the one real entry point;
the four existing `auto_runner*` functions are thin wrappers over it, so the
next knob does not mean a fifth function. The device is threaded through all
eleven arms, and the only `Device::Cpu` left in the module is the documented
default.

The device is also **validated in the dispatch**, before the config is read:
an exotic request (`Device::Tpu`, `OpenGl`, `Hexagon`) now fails naming the
device and the sniffed family rather than being discovered as a shape error
three layers down, uniformly for every family. That is what makes the plumbing
observable without weights — if `AutoOptions::device` were dropped, `Tpu` would
be indistinguishable from `Cpu` and both would report the same missing metadata
key. What remains unproven by any test is that no individual arm *omits*
`.device(device)`; the compiler confirms each call site exists, but a future arm
could forget one.

**`minimax-m3` had an `LmRunner` impl and no dispatch arm**, so the multiplexer
could not build it. Added, with the same `feature`-gated shape as the others.
`tests/auto_runner_coverage.rs` now asserts the invariant in **both**
directions: every dispatch arm has a registry entry (or the arm is dead), and
every family with an `LmRunner` impl has one (or it is unreachable). Both halves
were verified to fail when a registration is removed.

Also: an unreachable-but-known family now names the crate that handles it,
taken from the registry hint — and the hint is looked up by the **architecture**
rather than the runner name, since `phi3` → runner `phi` is not an arch.
`hint_for_hf_model_type` added as the `model_type` counterpart.

Fixing this surfaced a test file that had been **uncompilable** under the
`runner` feature: `tests/qwen35_forward_check.rs` called
`build_qwen35_{prefill_cache,decode}_hir_dynamic_ext` with 9 and 7 arguments
against 8-parameter signatures. No gate builds that feature, so nothing caught
it. Eight more collapsible-`if` and `needless_borrow` errors in the same feature
state are fixed too.

### `rlx-serve` / `rlx-models-core` — eight of ten model families could not be served

`rlx_serve::run_generation` drives a model through the **host-driven** pair,
`prefill_logits` then `decode_logits` per token, because sampling, logit bias,
log-probs and stop handling all belong above the model. `LmRunner`'s default for
that pair returns *unsupported*, and it is called with `?` — so a family that
has not implemented it does not run slowly, it fails on the first request.

A survey of the ten model families implementing `LmRunner` found that **two**
have the pair (`rlx-qwen3`, `rlx-qwen4exp`) and **one** has session
export/restore (`rlx-qwen3`), which `RunnerBatchRunner::new` requires and bails
without. Gemma, Llama-3.2, Mistral, the Qwen3.5 speculative runner and all four
VLMs override only `predict_logits`.

The pair cannot be defaulted in the trait: `decode_logits(token)` is stateful
and a trait default has nowhere to keep the context. `rlx_core::lm_host_driven::HostDriven`
keeps it — it tracks the tokens seen and answers each step by re-running
`predict_logits` over the whole context — and `rlx_serve::host_drivable` wraps a
runner in it. The result is correct and **quadratic**, which is the same bargain
the trait already strikes for `generate`'s default, and it is the difference
between a family being slow and being unavailable. A family that implements the
pair natively is detected on the first prefill and passed through untouched.

Three details worth stating:

- **The capability probe is lazy.** Deciding at construction would mean calling
  into a model before the caller has asked for anything; instead the first real
  `prefill_logits` is tried on the inner runner, and only an *unsupported* error
  falls back. A real failure (out of memory, a bad prompt) is returned as-is
  rather than retried down a different path, where it would surface as a
  confusing error or a wrong answer.
- **It takes `Box<dyn LmRunner>`**, because that is what a caller holds and
  `Box<dyn LmRunner>` does not itself implement `LmRunner` — an adapter generic
  over a sized `R` would have been unusable by the one caller it exists for. The
  crate's doctest is what caught that.
- The test asserts the wrapped loop reproduces the model's own `generate` chain
  exactly: the adapter may change the cost, never the tokens.

### `rlx-qwen4exp` — Qwen3.8-Flash-Next (the Qwen4 architecture)

New crate for `general.architecture = qwen4exp` / HF
`Qwen4ExpForConditionalGeneration`, the first published checkpoint of the Qwen4
architecture (`512x56B`). 48 layers: 36 Qwen3.5-style GatedDeltaNet linear
layers, 12 gated GQA layers behind a QSA block indexer, a 512-expert MoE at
every layer, and one PLE n-gram embedding module. Text prefill runs end to end.

Four parts of this architecture are not variations on anything already in the
workspace, and each has a failure mode that produces plausible output rather
than an error:

- **Hyper-connections replace every layer norm.** Four parallel residual streams,
  a grouped RMSNorm plus low-rank gated mixer at each of three sites per layer —
  and no `attn_norm`, `ffn_norm` or `output_norm` tensor anywhere in the
  checkpoint. The final mixer *is* the output norm. The scatter back into the
  streams is the identity weighted per stream, **not** the Sinkhorn-projected
  `comb` matrix of [`rlx-glm5next`]'s mHC or [`rlx-motif`]'s; the three
  implementations differ in the input norm, the gate shape and the scatter, and
  must not be cross-ported.

- **PLE is 51 billion hashed parameters.** `per_layer_token_embd` is
  `[160, 320001536]` — larger than the rest of the model — and is addressed by a
  hash of the trailing token window rather than by token id. The crate takes the
  embeddings as a graph input and gathers rows on the host behind a
  `PleRowSource` trait, so the table never enters an arena. The hash constants
  (three 45-bit SplitMix64-derived multipliers; a distinct prime modulus and
  table offset per hash head) are **derived** from `(seed, vocab_size)` and then
  checked against the values the checkpoint publishes: they agree exactly,
  including the padded row count. `validate` refuses a checkpoint where they do
  not, because the failure is reading the wrong rows of a 320-million-row table
  with no other symptom.

- **The two GQA expansions go opposite ways.** The GGUF converter permutes the
  linear layers' 48 value heads from grouped to tiled order (and with them
  `attn_gate`, `ssm_alpha`/`ssm_beta`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`'s v
  channels and `ssm_out`'s columns), so those take a whole-block repeat; the
  attention layers were not permuted and keep `repeat_kv`'s interleave. The
  tiled case agrees with the interleaved one only for v-heads 0 and 47 and flips
  the sign of every head in between.

- **QSA is exactly dense causal attention below 2048 tokens.** 512 blocks of 4
  mean-pooled keys plus each query's incomplete tail reconstruct `0..=q`
  exactly, so the attention layer drops the indexer and takes the fused causal
  path. Pinned numerically on real weights by running the full selection
  machinery against the short-circuit — an equivalence between two independently
  spelled paths, which a dense-config-vs-sparse-config test cannot be.

Validated on the **published weights** without downloading the model:
`scripts/qwen4exp_subset.py` range-fetches `blk.1` (GatedDeltaNet + PLE) and
`blk.3` (gated attention + QSA) with 8 of each block's 512 experts into a valid
186 MB standalone GGUF (`just qwen4exp-real`). That covers both layer kinds, the
converter's sign and offset conventions (`ssm_a` is pre-negated `-exp(A_log)`;
the norm gains are pre-offset by 1 — both silent when read the wrong way), and
the QSA equivalence. Separately, the tensor manifest is checked against the real
1224-tensor list for all 48 blocks in both directions — names *and* orientations
— from header data alone.

**Incremental decode, chunked prefill, the MTP draft block and the multimodal
seam are all wired too**, and `decode == prefill` is checked on the real
weights:

- **One builder serves decode and chunked prefill** (`chunk = 1` vs `chunk > 1`),
  because every stateful piece is spelled differently from the stateless
  prefill — a grouped `Op::Conv` versus a sum of shifted copies, `Op::GatedDeltaNet`
  versus `…_carry`, a synthesized causal mask versus a `[chunk, cap+chunk]`
  bias, the whole token array versus a two-token host-side hash window. Sharing
  the code means one equivalence test covers both. The per-token state is 96%
  one tensor: the scan matrix is `48 · 128²` floats per linear layer, 113 MB
  across 36 layers, which `ScanState::InPlace` keeps on-device where the backend
  allows it.
- **A PLE layer keeps three cache rows, not two** — the reference's
  `number_of_conv_states = 3`. The third is the trailing `ngram_size − 1` token
  *ids*, which is not a tensor: it lives on the host, because the hash does.
  Forgetting it makes every n-gram at a chunk boundary hash as if the sequence
  had just started.
- **Right padding is exact, and only the exported state needs help.** The
  convolution and the scan are both causal, so a real position never reads a
  pad; what padding breaks is the state the *next* chunk resumes from. A
  `pad_mask` zeroes the scan's gate and beta (making a step an exact no-op) and
  a `conv_shift` gathers the convolution window ending at the last real token.
  A test compares the exported state of a padded chunk against an
  exactly-sized one — the only test that can see either mechanism.
- **The MTP block reads the hyper-connection residual, not the hidden state**,
  and `eh_proj` is `cat([fc_embedding, fc_hidden], dim=1)` applied *per stream*
  on the feature axis. So the trunk is built once at `OutputKind::HcResidual`
  and that one residual feeds both the draft block and a separate head flow;
  splitting the trunk there is checked not to change its logits, because
  otherwise every acceptance decision would be made against a model the trunk
  is not.
- **The vision tower is a native port** (`vit`), not a call into another crate.
  The architecture is Qwen3-VL's — llama.cpp's converter says so outright
  (`class Qwen4ExpVisionModel(Qwen3VLVisionModel)`) — but the implementation is
  a `ModelFlow` over a `WeightSource` like the rest of this crate, so a
  quantized `mmproj-Q8_0.gguf` runs **packed** and nothing is transposed on the
  host: the GGUF already reports `[out, in]`, and `v.patch_embd.weight` already
  arrives as `Op::Conv`'s `[out_c, in_c, kh, kw]`. Six details carry it, and
  each has a plausible wrong answer: the patch embedding is **two**
  convolutions summed (a 2-frame temporal window with a still image in both);
  tokens are permuted into spatial-merge blocks *before* the transformer, which
  is the only reason the merger's reshape is a reshape — so the position table
  must be permuted identically and the RoPE coordinates generated in that
  order; the learned position table is **bilinearly resized** from its 48²
  grid; the axial RoPE puts `[row | column]` frequencies in the first half of
  each head and duplicates them into the second, which is what makes
  `rotate_half` pair row with row; the block FFN is the **tanh** GELU while the
  merger is the **exact erf** one; and `v.post_ln` is the *merger's* norm,
  applied once, not a post-block norm.

  `rlx-qwen35` ships the same tower and its captions are known-good on real
  Qwen3.6-27B weights, so it is kept as a **dev-dependency oracle**:
  `tests/vit_parity.rs` compares the two element for element on identical
  weights. Nothing in `src/` touches it. (That required one additive change
  upstream — `rlx-qwen35` re-exported `MmProjWeights` but not its `blocks`
  field type, so the bundle could be read and never constructed.)

- **The multimodal seam** (`vision`) supplies what is qwen4exp-specific rather
  than Qwen3-VL-generic: a host-assembled `inputs_embeds` (`host_embed`, which
  also keeps a 1.2 GB `token_embd` off the device), the `image_token_id`
  substitution the **PLE hash** needs at vision positions — a *different* array
  from the prompt ids — and the M-RoPE advance of `max(nx, ny)` per image
  rather than its token count.

- **A runner, so the model can be run rather than assembled** (`runner`).
  Everything else in the crate hands back a `BuiltModel` and leaves the driving
  to the caller — compile it, derive the M-RoPE tables, gather the n-gram rows
  against the right history, thread 36 layers of recurrent state, pad the chunk,
  re-read the exported state. `Qwen4ExpRunner::open` does that once and
  implements `rlx_runtime::LmRunner`, so `prefill_logits` + `decode_logits` give
  a host-driven loop the HTTP server shape can use and `generate_greedy` gives
  the simple one. Three things make it fast rather than merely convenient: the
  step graph is **compiled once** and reused (a sibling port paid 50x for
  compiling per step); `prefill_chunk` above 1 advances several positions per
  `run()` through a second graph, with the cost stated — the weights are then
  resident twice, so the default is 1; and `RunnerOptions::single_sequence`
  moves the GatedDeltaNet scan into a device-resident param, which is 96% of
  this model's per-token host traffic (113 MB across 36 layers), on exactly the
  backends where that survives to the next `run()` (`scan_state_for`: CPU,
  Metal, wgpu — MLX and CoreML freeze it at zero, which is a wrong answer, not
  a slow one).

  Chunked prefill flowing into one-token decode needed one addition in `decode`:
  `DecodeSession::set_chunk`. The two graphs pack their output differently but
  the state they thread does not, so the session is re-pointed at the other
  layout with its cache, convolution windows and n-gram history intact. It
  validates the layout against the chunk rather than trusting the pairing,
  because a mismatch would parse the next `commit` at the wrong offsets and
  silently corrupt the cache.

  `tests/runner.rs` pins the runner against the explicit loop it replaces
  **bit for bit** — same graph, same inputs, so any difference is a threading
  mistake rather than arithmetic, and a one-position M-RoPE slip (2.1e-3
  relative) would scrape through the 2e-3 tolerance a looser test would use.
  The chunked-into-decode test sits at 1e-5: 200x the legitimate chunk-3 vs
  chunk-1 reduction-order noise (4.9e-8), 50x under the state loss it exists to
  catch (5.2e-4).

- **The n-gram table read lazily, which is what makes the real checkpoint
  runnable** (`ple::GgufPleRows`). `per_layer_token_embd` is 51 billion
  parameters — `[160, 320001536]`, ~28 GB as the published IQ4_NL — and a decode
  step wants **nine** of its 320 million rows. Holding it is not an option, and
  neither is mapping it: borrowing tensor bytes from an mmap makes every page it
  touches resident, and on macOS that memory cannot be returned without
  `munmap`. So this seeks and reads the quantization blocks a row overlaps and
  nothing else. For the published type that is exactly 5 IQ4_NL blocks of 18
  bytes — **90 bytes per row** — and `tests/ple_rows.rs` pins that geometry
  against the real checkpoint's header. The block arithmetic is *derived* from
  `rlx_gguf::bytes_for_public` rather than tabulated, so an f32, k-quant or
  i-quant table all work and a row straddling a block boundary just reads both.
  `Qwen4ExpRunner::open` attaches one automatically when the file carries the
  table; `RunnerOptions::ple_table` points at the right shard of a split
  checkpoint, and an explicit path that fails to open is an error rather than a
  silent fallback to zeros.

  The oracle is the crate's own whole-tensor dequantization — a lazily read row
  has to equal that slice **exactly**, since both end up in the same
  `dequant_typed`, so only the index-to-offset arithmetic is under test. The
  straddling fixture earns its place: dropping the within-block offset is
  invisible to a 160-wide row (160 is 5 whole blocks) and only a width that does
  not divide the block catches it.

- **Experts paged off disk** (`paged_moe::PagedMoeLayer`), the last piece
  between the port and the full checkpoint. The model is `512 × 56B`: every one
  of its 48 layers carries 512 experts across three banks and a token fires
  **10**. Reading only what fires costs ~49 MB per token per layer against the
  2.5 GB a layer's resident banks need. This rides on the model-agnostic
  `rlx_distributed::ExpertPager` built for GLM-5.3, so what is new here is the
  qwen4exp half: `MoeDims::paged`, which makes the graph's expert index a
  **constant** (`arange(seq) * top_k + k`) so the gather can happen outside the
  graph, and `moe::emit_qwen4exp_router`, the routing half on its own.

  The router is extracted rather than reimplemented on the host, and runs as its
  own compiled graph before the paged one. Chicken and egg: you cannot gather
  the experts a token fires until you know which they are, and the graph that
  would tell you is the graph you are trying to feed. Deriving top-k on the host
  instead would rest on the host and the graph accumulating a 512-wide dot
  product identically — which a BLAS threshold or a device change moves
  silently, and a flipped pick weights the token by another expert's matrix and
  looks entirely normal.

  `tests/paged_moe.rs` runs the whole loop against the same layer with all 512
  experts resident, because each piece is wrong in a way that still produces
  finite, well-scaled output. Both failure modes were induced to check the test
  sees them: reversing slot order (the mistake that pairs a token with another
  token's gate) diverges 6.5e-3 relative against a 1e-5 tolerance, and ignoring
  the `paged` flag — so the graph indexes slot-width banks by global expert id —
  fails in the backend rather than silently.

`PagedMoeLayer::from_gguf` pages the checkpoint in **the only form it ships
in**: quantized. `register_gguf_expert_banks` (upstream, header-only) already
served quantized expert slices, but the layer declared f32 slots and its
geometry check refused the mismatch — correctly, since reading quantized bytes
into an f32 slot produces garbage weights. Now each fired expert is
dequantized on its way into its slot, and the dtype is read **per bank per
layer** out of the header rather than assumed, because it is not uniform:
`blk.0`'s gate and up banks are IQ1_S where `blk.1`'s are IQ2_XXS, and every
`ffn_down_exps` is IQ4_NL. Validated on the published subset — `blk.1` paged
from its real IQ2_XXS/IQ4_NL banks agrees with the same layer over the whole
dequantized banks. Feeding the quantized bytes to the graph *directly*, as
`DequantGroupedMatMul` over packed slot banks, would avoid both the host
dequantization and the f32 arena; that is an optimization left on the table,
not a correctness gap.

Each dimension bundle is now derived in exactly one place —
`HcDims::from_config`, `GdnDims`, `AttnDims`, `QsaDims`, `MoeDims`,
`PleDims` — and prefill, chunked prefill, the decode step, the MTP block and
the real-weight tests all call it. There were **four** copies of those seven
derivations, one of them in the tests, which is the arrangement where a test
quietly measures a slightly different model than the builder constructs. The
two genuine knobs stay explicit at the call site (`force_emit` for the QSA
indexer, `paged` for the expert banks), as does decode's `cap + chunk` — the
indexer sees the whole cache plus the step, not just the step.

Every test in the crate runs over the one fixture in `synth`, whose shapes
`tests/tensor_manifest.rs` checks against the real 1224-tensor list — so no
test carries a second copy of the tensor contract to drift out of date.

Each of the four equivalences was checked to actually fail: sabotaging the
dilated tap offset, the scan carry, the conv shift, the pad masking, the second
patch convolution or the position-table permutation breaks exactly one of them.
`tests/vit_parity.rs` also records the two things it is *blind* to — the
cos/sin table generation (covered by a closed-form unit test instead) and which
GELU is which (the two agree to ~3e-4, so no end-to-end comparison resolves it)
— rather than implying coverage it does not have.

`scripts/qwen4exp_subset.py --mmproj 3` applies the same range-slicing trick to
the tower: 3 of its 27 blocks plus every global, ~172 MB instead of 904 MB, with
`clip.vision.block_count` rewritten to match (`just qwen4exp-vision-real`).

Not done: end-to-end generation on the full 48-layer checkpoint (expert paging
at scale, not a missing capability); decode past the 2048-position QSA budget,
which needs a position-ordered indexer-key cache and is *refused* rather than
silently run dense; left padding and multi-sequence packing; the tokenizer /
chat template; and the tower's deepstack path and video (both refused rather
than approximated — `deepstack_visual_indexes` is `[]` in the published
checkpoint, and video would need the two patch kernels fed *different* frames).

### `rlx-fgd` — Section 4.3: radiance fields

Inverse rendering, the paper's fourth and largest experiment: recover a scene's
density and view-dependent colour from posed images by descending the
photometric loss of Equation 11 in function space. Adds a volumetric renderer
(DDA voxel traversal with exact per-voxel integration — `σ` is constant inside a
voxel, so `T·(1−e^{−σΔt})·c` is the integral, not an approximation of it), the
NeRF camera model, spherical harmonics with their RKHS kernel, a
`VoxelGrid` representation, and the NeRF-Synthetic loader. Runs natively and in
parallel rather than through an RLX graph: the work is pointer-chasing ray
marching, not dense linear algebra.

**Proposition A.3's gradient is checked against finite differences of the loss**
— the only check that distinguishes a correct Jacobian from a plausible one.
Both sides are Monte-Carlo, so it is a convergence claim: 10% disagreement at
256 rays/pixel and 4096 samples/voxel, 3% at 2048/32768, and **0.4%** at
16384/262144, for density and colour alike.

Four departures from the reference implementation:

- **The colour gradient is closed-form, not fitted.** `∇_{c_k}L(x, ·)` is a sum
  of kernels `K(ω_i, ·)`, and `K(ω,ω') = Σ_j Y_j(ω)Y_j(ω')` is the reproducing
  kernel of the harmonics' span — so expanding it gives the coefficients
  directly. The representation holds the colour gradient *exactly* in the
  direction variable; only the spatial part is ever approximated. The reference
  fits the harmonics by least squares over sampled directions instead, which
  agrees when that system has full rank and silently drops components when it
  does not. This removes the direction sampling, the pseudo-inverse and that
  failure mode at once.

- **Refinement has to re-impose the visual hull**, which is what the new
  `Functional::after_refine` hook is for. A coarse voxel straddling the
  silhouette counts as occupied; its children outside the hull inherit its
  density, and the gradient is identically zero there, so nothing ever takes it
  away. Measured on the test scene: the run stalls at `1.08e-3` / 30.5 dB with
  the halo against `1.09e-4` / 37.3 dB once it is removed — a permanent blocky
  shell around the object. Found by noticing the adaptive run plateaued while
  the same grid trained from scratch did not.

- **Directions are normalised everywhere.** The reference's RKHS kernel
  normalises its arguments; its renderer evaluates the harmonics at the raw ray
  direction, whose length varies across the image plane. The harmonics are only
  orthonormal on the sphere.

- **The density and colour error terms are weighted consistently.** The
  reference's density residual carries a ½ that its colour residual and both
  norms do not, understating the density half of `RelErr`'s numerator by √2
  against its own denominator; its two density branches also disagree with each
  other about it.

`tests/radiance.rs` adds 13 tests in all. Besides the finite-difference check:
harmonics orthonormal by quadrature and against the addition theorem, the
renderer against the closed form for a uniform medium, `Camera::project` as the
exact inverse of `Camera::ray_through`, the mid-march stop against a separate
march, and an end-to-end reconstruction scored on viewpoints never trained on.

On the synthetic scene, adaptive beats every fixed grid whose cost fits its
budget (`2.2×` below fixed 32³ at ε = 5.0, the next rung up costing 50% more
than it spent) but does not dominate the *interpolated* fixed frontier — it
sits 13–20% above it at equal work, consistently across ε. The frontier is not
purchasable (fixed grids come in powers of two) and the scene is two smooth
blobs that 32³ nearly resolves, so there is little for a coarse phase to
exploit; the paper's Ficus scene has structure at scales a coarse grid cannot
see.

### `rlx-fgd` — new: functional gradient descent with adaptive representations

Rust port of [arXiv:2606.16926](https://arxiv.org/abs/2606.16926) (Csillag et
al., NeurIPS 2026): Algorithm 1, the convergence theory of Section 3.1, and the
paper's `L²` fitting, RKHS regression and wave-equation tasks. `Refinement::Fixed`
turns the same driver into the paper's fixed-representation baseline, so the
central comparison runs on one code path. Section 4.3 (radiance fields) is not
implemented in that first pass; it landed in the entry above.

Three things the port had to pin down that the reference code leaves loose:

- **The `L²` error estimator understated the quantity Algorithm 1 tests.** The
  fitted constant on a cell is the *sample* mean, so it is itself off by
  `O(σ/√N)`; the unbiased estimate of `‖g − ∇L(f)‖²` there is `s²(1 + 1/N)` with
  `s²` Bessel-corrected. The reference divides by `N`, which is low by
  `(N−1)/(N+1)` — 29% at its `N = 8`. With the biased form, Lemma 3.5's bound was
  violated at steps of a real run (`L = 1.3231e-3` against a bound of
  `1.3202e-3`); with the corrected one it holds with ≥4.5% margin at every step,
  at every sample count tried. `tests/theory.rs` checks this against a run rather
  than re-deriving it.

- **Losses have to be reported on a quadrature pinned for the whole run.** A
  quadrature read off the iterate's own grid changes scale exactly when the
  representation refines, so each refinement becomes a step change in the
  reported loss that has nothing to do with the iterate. That makes the
  adaptive-versus-fixed comparison — the entire point of Figure 1 — meaningless.
  It showed up first as the wave loss *rising* on the first step at every step
  size, including arbitrarily small ones, which is not something a step size can
  cause.

- **Two different tolerances.** Algorithm 1's `ε` is not the `RelErr` bound in
  Propositions 3.6/3.8: running at `ε` guarantees `RelErr ≤ ε/(1+ε)`
  (Theorem 3.10(i)), and the theorem is stated for `ε ∈ (0,1)`, which is exactly
  what keeps that under the `1/2` Lemma 3.5 needs. The authors' own `ε = 3`
  teaser and `ε = 50` wave runs sit outside it. `Report::guarantees_hold` reports
  which case a run is in.

Verified: Lemma 3.5 and Proposition 3.8's rate against real runs; the `L^∞`
Lipschitz bound against the actual sup-norm error at non-centre points;
Proposition A.2's `α`/`β` against Assumptions 3.3/3.4 numerically; `h` and `ĥ`
confirmed a Fourier pair by direct quadrature; the wave solve recovers its
initial condition at `+0.976` correlation. Section 4.1's gradient kernel is
compiled to RLX and matches the `f64` host reference on CPU/Metal/MLX/wgpu (2e-6),
including making the same refinement decisions.

### `rlx-kev` — FIXED: a warm-started run exported a checkpoint missing the model

`--init-from` folded the checkpoint's adapter into the base weights and fitted a
*fresh* rank-16 adapter on top. That trains a correct model — and then exports
only the new delta. Loading the result the normal way (`base + adapter`, which
is what `--run <dir> --base <dir>` does) silently dropped everything the
checkpoint knew.

Measured on real weights, a two-step delta run from kev-0.8b:

    kev-0.8b source adapter   total ||B|| = 86.8524
    delta-run output          total ||B|| = 0.3469     ratio 0.0040

The output was, to three significant figures, the bare base model. Since the
released recipe **is** a continuation round (`init_from:
jaredpalmer/kev-0.8b`), reproducing it this way was impossible.

Composing on export is not available: a rank-16 adapter cannot represent the
sum of two rank-16 deltas. The fix is the thing peft and kev already do —
**continue the same adapter**. The base now stays bare for a warm start, and
`train::restore_adapter` seeds the checkpoint's `A`/`B` into the trainable
parameters. After the fix, the same run:

    delta-run output (fixed)  total ||B|| = 86.8586    ratio 1.0001

i.e. the whole source adapter plus two steps of training, and step 1 opens at
loss 0.0038 against a chance floor of 0.2878.

`restore_adapter` inverts `export_adapter`'s transpose exactly (peft `A [r, in]`
/ `B [out, r]`; graph `A [in, r]` / `B [r, out]`). Neither side bakes in
`alpha/rank` — the graph applies it as an op and `merge_into` applies it on load
— so this is a pure re-layout with no rescaling. Rank mismatch is refused with
the `--lora-rank` to rerun with, rather than restored wrong.

Pinned as a **round trip** — restore an adapter, export it again, compare tensor
for tensor — because that is the property that makes an export self-contained,
and a transpose dropped on either side leaves both files the same size and shape.
The test also asserts the fresh graph's `B` is zero first, so it cannot pass
vacuously.

This was found by checking a claim rather than trusting it: the previous entry
added a README line pointing users at `--init-from` for the published recipe,
which was worth verifying before shipping.

### `rlx-kev` — FIXED: `--init-from` threw the trained pointer head away

`--init-from` is documented as "warm-start the adapter **+ head** from this
checkpoint". It restored the adapter and silently discarded the head: nothing in
the training path ever read `head.safetensors` back, and `build_train_graph`
unconditionally seeds the head from a hashed PRNG.

So a warm start merged the backbone correctly and then bolted a *random readout*
onto it. The model scored at chance at step 0 and had to relearn the head from
scratch — which defeats the point of warm-starting, and makes the released
recipe unreproducible, because that recipe **is** a continuation round
(`init_from: jaredpalmer/kev-0.8b`).

Measured end-to-end after the fix, warm-starting from kev-0.8b:

    start: warm from /Volumes/FOUR/kev/kev-0.8b
    restored the pointer head from /Volumes/FOUR/kev/kev-0.8b
    epoch 0 step 1/3: loss 0.0000 (chance 0.3385) grad_norm 0.00

Loss 0.0000 against a chance floor of 0.3385 — the model is immediately correct,
because it is kev-0.8b. With a random head it could only have scored at chance:
a random readout cannot rank options.

`train::restore_head` inverts the transpose `Trainer::pointer_head` applies (the
graph stores `[d, dp]`, `PointerHead` holds `nn.Linear`'s `[dp, d]`) and refuses
a head whose dimensions disagree rather than restoring a mismatched one. It runs
only on the first graph build; the 27 later shape rebuilds already carry every
parameter across.

The test does not settle for "the parameters changed" — a dropped transpose
changes them too and is still wrong. It asserts the restored head reads back
*equal to the trained one*, and separately that the fresh head genuinely differed
first, so it cannot pass vacuously. Mutation-tested by replacing the transpose
with an identity copy: it fails.

### Release note — `rlx-kev` Metal *training* needs upstream fixes not in rlx 0.2.16

Checked against the actual published crates rather than the local `../rlx`
checkout, which is dirty and masks this.

**The API surface is clean.** Nothing in rlx-models' compiled code calls an
upstream symbol missing from crates.io 0.2.16. `ArenaWidthPolicy` is used only
as `Native` / `Hybrid`, both published; the `NativeHalfWidened` /
`NativeBf16Widened` variants added during this work appear *only in a doc
comment*. `WeightSource::take_packed_bank` is present in published `rlx-flow`
0.2.16 at the same line. So rlx-models 0.2.16 builds from the registry.

**The behaviour is not.** The Metal backward fixes that `rlx-kev --train
--device metal` depends on are absent from published `rlx-metal` 0.2.16 —
verified by unpacking the crate: no `uses_at` guard in `src/thunk/mod.rs`, no
`round_bf16_f32` in `src/kernels.rs`. Against the registry build, Metal returns
NaN or zero for most Qwen3.5 trunk parameter gradients while computing the
forward bit-identically, which is the bug that cost a 2-hour training run.

It fails *loudly* rather than silently: `Trainer::optimizer_step` refuses a
non-finite gradient norm and names the worst parameter. Forward and decode are
unaffected on every backend, as is CPU training.

So the choice before publishing is either to ship 0.2.16 with that caveat
documented, or to hold `rlx-kev`'s `train` feature until an upstream release
carries the Metal fixes. That is a judgement call about the release, not
something to decide silently in a changelog.

### Release prep — catalog, publish tiers, and two retracted claims in the docs

`just lint-all`'s gates found three publish blockers introduced by the new
`rlx-kev` crate, all of the classes the playbook in `docs/releasing.md` exists
to catch:

* **Four `include_str!` targets were gitignored.** `crates/*/tests/fixtures/*.json`
  is a repo-wide net, and kev's four golden fixtures (890 B - 124 KB) fell under
  it. They are compile-time inputs, so a fresh clone could not build those test
  targets at all. Added to the existing negation list alongside rlx-tada,
  rlx-glm5next and rlx-chatterbox, which exist for exactly this reason.
* **`rlx-kev` was in no publish tier**, so `publish.sh` would silently never
  ship it. Placed in tier 6: it depends on `rlx-qwen35` (tier 5),
  `rlx-models-core` and `rlx-tune` (both tier 1). The validator checks tier
  *membership* but never *ordering*, so ordering was verified separately against
  `cargo metadata` — 0 violations across all 199 publishable crates.
* **`rlx-kev` was filed under "Vision-language & multimodal"**, which it is not —
  it is a text-only decision model with no vision and no generation. Moved to
  "Language models (text LLMs & reasoning)". This also explains the catalog
  check's complaint that the VLM row said 12 with 13 rows present: the move
  fixes that count without touching it. Totals corrected 182 -> 183 in MODELS.md
  and the three places README.md repeats them.

A fourth problem was found by hand rather than by a gate, and the gate now
covers it: the `rlx-kev` row contained `max |Δp| 1.4e-5`, whose **unescaped
pipes split the markdown table**, giving that row 5 cells where the header has
3. GitHub renders it wrong — but worse, `check-models-catalog.py` reads the
backend cell as `cells[2]`, so it had been validating a prefix truncated at the
first stray pipe. It was the only such row in all 183. Reworded to `max abs(Δp)`,
and `parse_rows` now reports the cell count so a row whose width disagrees with
its header is an error that explains the truncation. Mutation-tested by putting
the pipes back: the check fires and names the crate.

Two claims in `crates/rlx-kev/README.md` were **retracted**, since shipping docs
should not carry numbers that later measurement contradicted:

* "The fix costs ~2x on a Metal training step" — the fusion it blames fires on
  0 of 34 candidate anchors in that graph, and the 2x was machine contention
  (identical work timed 492 s vs 1062 s). A step is compute-bound: 10.2 s
  measured against ~6.2 s of f32 FLOPs.
* The 154-step run's **0.3389** was quoted against the broken run's 0.2845
  without its chance floor, which is **0.3059**. It is at chance. The README now
  says so.

The divergence write-up was also superseded: the Adam-moment carry across shape
rebuilds attenuated it, but the mechanism was `rlx-tune`'s shared bias-correction
counter (see above).

### `rlx-kev` — the documented reproduction command had the wrong learning rate

`kev-0.8b`'s own `training_config.json` says `lr: 2e-05` and `head_lr: 0.0`
(which means "use `lr`"). The README's "Reproducing the released recipe" block
said `--lr 1e-4 --head-lr 1e-3` — 5x and 50x the reference. Corrected, and the
README now states what that config actually describes: `init_from:
"jaredpalmer/kev-0.8b"` with `replay: 6000`, i.e. the released checkpoint is the
output of a **continuation round** (round 15), not of one cold run. `0.8136` is
an iteratively-refined model's score, and a cold single-epoch run should not be
expected to match it.

### `rlx-tune` — FIXED: Adam's bias correction was shared across parameters

`Optimizer::step` is called once per *parameter*, and `Adam::step` incremented a
step counter stored on the optimizer. So `t` advanced by the number of
trainables on every optimizer step, not by one — with rank-16 LoRA on all
targets of a 24-layer model that is ~340 per step.

The smallest demonstration: two parameters, same initial value, same gradient,
same optimizer, same step. Adam is elementwise and carries no cross-parameter
state, so they must move identically. They did not — `-1.0` and `-0.744`,
decided purely by iteration order.

The consequence is not a small numerical wobble. After the very first optimizer
step `t >= 340`, so `1 - beta^t` is 1 to machine precision and bias correction
is **off for the rest of training**. Uncorrected, `v` is badly underestimated
while it warms up (`beta2 = 0.999`), so `m / sqrt(v)` is inflated — the
effective learning rate for a constant gradient is

    step     1     5    10    25    50   100   153   400
    mult  1.70  5.24  6.42  5.91  4.50  3.24  2.65  1.74

peaking at **6.5x** around step 13 and still **2.65x** at step 153 — which is
exactly the OneCycle peak of the kev run (`pct_start 0.1` of 1529 steps). A
recipe asking for `1e-4` was really training at several times that through the
entire ramp. That is the divergence this port has been chasing: loss 1.06 at
step 50 -> 4.33 at step 100, above every `ln(n_options)` chance floor.

Fixed by keeping the step count in the per-parameter state entry — `(m, v, t)`
— which is what PyTorch's `state['step']` is and why it is per parameter there.

Three tests pin it, and the middle one matters: two identical parameters also
agree with each other when bias correction is disabled *for everyone*, so
`the_first_adam_update_has_magnitude_lr` separately pins that the correction is
actually applied (Adam's first update is `lr` regardless of gradient scale).
The third checks a parameter stepped n times alongside another still warms up
like one stepped n times alone — the bug is invisible to any single-tensor
smoke test, which is how it survived.

`FlatAdam` was checked and is correct: it steps the whole parameter bucket at
once, so its counter already advances once per optimizer step.

### `rlx-kev` — a long training run now survives being killed

The suite-training path exported the adapter and head exactly once, after the
final optimizer step. A full run is 1529 steps at ~27 s/step — 8-14 hours — so a
run that is killed, OOMs, or hits a thermal stall at hour 13 left nothing behind.
One already had.

`--checkpoint-every <N>` (default 100, 0 disables) now exports mid-run. The
write costs 0.1 s, so at the default it is under a thousandth of the run.

The write is staged rather than in-place, which is the part that matters. A
periodic checkpoint written directly into the output directory is *worse* than
no checkpoint during its write window: the process dies mid-file and now the new
checkpoint is torn and the previous one is already gone. So the full set is
built in `<out>/.staging` and each file is `rename`d into place — atomic within
a filesystem, and staging is a child of the output directory, so a kill at any
instant leaves either the old file or the new one, never half of either.

The tokenizer is now copied before the first step rather than after the last, so
a checkpoint written mid-run is a complete, loadable directory on its own.

Verified by actually doing it: a run was `kill -9`ed immediately after its second
checkpoint, and the output directory held a complete, loadable checkpoint
(`adapter_config.json`, `adapter_model.safetensors`, `head.json`,
`head.safetensors`, `tokenizer.json`) with no `.staging` residue.
`a_periodic_checkpoint_overwrites_in_place_and_stays_loadable` pins the
properties in CI: staging never outlives a successful write, every round loads
through `LoraAdapter::open`, and — so the test cannot pass on three identical
copies of an untrained export — the merged weights must differ after two further
optimizer steps.

### `rlx-kev` — the shape-padding speed lever is measured and closed

`examples/pad_audit` runs the real `shape_batches` packer over the real corpus
on CPU with no model weights, so the answer covers all 12,576 records instead of
whatever a short GPU run happens to sample. That distinction mattered: the
"~32% padding" figure quoted earlier in this work came from a 120-record run and
is **wrong at full scale**. The corpus numbers are

    3057 batches over 15378 examples, budget 2048, max_rows 8
    slot 4019456 positions: real 79.9%, unfilled rows 4.3%, seq padding 15.8%
      seq padding splits: bucket rounding 5.7%, within-batch spread 10.1%

The small run over-reported because the packer pads each `(seq, opts)` group's
tail batch to a constant row count; with 19 shapes over 120 records nearly every
batch *is* a tail batch (22% unfilled rows), while at full scale groups are deep
and that falls to 4.3%. Both measurements are consistent once group depth is
accounted for.

So the ceiling on this lever is 1/0.799 = **1.25x**, and the dominant single
component is within-batch length spread — rows of one record are a shared state
prefix plus option branches of genuinely different lengths, which no regrouping
removes.

The two knobs that look like they control this do not. Sweeping
`KEV_TRAIN_TOKEN_BUDGET` over 1024..8192 against `max_rows` 8 and 16 — eight
configurations — moves total computed positions from 4.10M to 3.91M, under 3%:

    budget 1024 rows  8: real 78.4%  idle 6.1%  seq 15.5%   slots 4099008
    budget 2048 rows  8: real 79.9%  idle 4.3%  seq 15.8%   slots 4019456  (default)
    budget 4096 rows 16: real 82.1%  idle 1.7%  seq 16.2%   slots 3912832  (best)
    budget 8192 rows 16: real 81.6%  idle 2.3%  seq 16.1%   slots 3937856

Raising the budget trades unfilled rows against sequence padding almost
one-for-one. The comment at the knob claiming it "trades these two against each
other" as a tuning opportunity has been corrected to say it is an arena-headroom
setting, not a speed setting.

Taken with the earlier finding that a step is compute-bound (10.2 s/step
measured against ~6.2 s of f32 FLOPs), and that identical work has timed 492 s
against 1062 s purely from machine contention, every remaining padding gain is
smaller than the measurement noise. This lever is closed; the training log keeps
reporting the split so a future change to the corpus is visible without rerunning
the audit.

### `rlx-kev` — scores and losses are reported against their chance floor

The training log now prints the chance floor of the batches each window came
from, and marks `AT CHANCE` when the loss is not beating it. The floor moves with
the shape mix — `ln 2` on a yes/no bucket, `ln 128` on banking77 — so a bare loss
is unreadable: 0.709 looks excellent and is exactly chance at two options.

Evaluation gained the same thing: `eval::Metrics` carries `chance_accuracy`
(`mean(1/n_options)`) and `chance_nll` (`mean(ln n_options)`), prints both
inline, and adds `at_chance()` plus an
`<== AT CHANCE` marker. Neither is a constant on this suite: `decision-v7` mixes
2-option yes/no with banking77's 77 intents, so the floor moves with the sample.

This is the diagnostic that was missing. Every failure this crate has hit —
a backend returning NaN gradients, a learning rate diverging — presents as
"plausible loss, chance accuracy", and `ln 2 = 0.693` reads like a *good* loss if
the comparison in your head is zero. It is what let a 1529-step run complete,
report a falling loss, and score 0.2845 against a published 0.8136.

`eval::score` also visits records in **shape order** rather than suite order. One
graph is compiled per `(rows, seq)` bucket, and a suite is ordered by source, so
consecutive records asked for unrelated shapes and the model recompiled on almost
every one. Metrics are order-independent, which a new test pins so a future
running-statistic cannot quietly make them otherwise.

The training loop reports what its shape bucketing costs, in both directions:
graph-rebuild count and time, and **shape utilisation** — `real_tokens /
(rows × seq)`, the fraction of each step's compute that is not padding.
`KEV_TRAIN_TOKEN_BUDGET` trades one against the other, and neither was visible
before.

Measured on a 25-step run: 19 rebuilds cost 236 s of 492 s (48%) at ~12.4 s each
— of which ~2-4 s is loading the base, 3-21 s building the graph and 10-21 s
Metal compiling it — and utilisation was **68%**. So rebuilds dominate a *short*
run but are ~6 min of a full 1529-step one, while the 32% padding is the cost that
scales. Worth knowing before optimizing either: the intuition that reloading the
1.7 GB base per rebuild was the problem was wrong, and measuring said so.

### `rlx-tune` — Adam state survives a graph rebuild

`Adam` grew `take_state` / `restore_state` (and `AdamState`), because a caller
that has to recompile — a new input shape, say — must not silently get a fresh
optimizer. Dropping `m`/`v` restarts the bias correction, so the step right after
each rebuild is effectively un-normalized.

`rlx-kev`'s recipe hit exactly that: it compiles one graph per
(rows, seq, max_options) bucket and rebuilds the trainer on every shape change —
28 times in a full `decision-v7` epoch. It carried the adapter and head weights
across but not the moments, and a comment claimed `Trainer::rebuild_at` handled
it, a function that never existed. With 28 restarts during the OneCycle ramp the
run diverged: loss 1.06 at step 50 → 4.33 at step 100 with a pre-clip grad norm
of 1294 → 9.41 by step 150, above every `ln(n_options)` chance floor. That was
invisible until Metal's backward was fixed, because before that the gradients
were NaN or zero and nothing moved at all.

`Adam::step` now also re-initialises a restored moment pair whose length no longer
matches its parameter, instead of indexing past the end.

### `rlx-upscale` — single-image super-resolution (15 architectures)

New crate covering both tiers of the current super-resolution landscape, native
Rust end to end:

- **Tier 1, efficient CNNs**: **`ESRGAN`/RRDBNet**, `Compact` (SRVGGNet),
  `SPAN`, **`SPANV2`** (team
  XiaomiMM's NTIRE 2026 Efficient SR challenge winner), `PLKSR` (all three
  channel mixers), `RealPLKSR` (GroupNorm and channel-first LayerNorm variants,
  `PixelShuffle` and `DySample` heads).
  Plus **`SAFMN`** (receptive field from pooling rather than attention or large
  kernels) and **`RealCUGAN`** (`upcunet_v3` — two *valid-padded* U-Nets in
  series; the anime upscaler, and the only network here whose convolutions
  shrink the feature map).
- **Tier 2, window-attention transformers**: `SwinIR`, **`Swin2SR`**, `HAT`,
  `DRCT`, `DAT`, `MambaIRv2`, plus **`OmniSR`** (MaxViT-shaped: block
  attention, *grid* attention and channel attention interleaved).

`Swin2SR` is SwinIR rebuilt on Swin **V2** blocks — cosine attention, a learned
per-head temperature, an MLP-generated position bias and res-post-norm — so it
is a flag on the existing module rather than a second copy of it. Both frozen
pieces (the temperature's `clamp(…).exp()` and the whole
`16·sigmoid(cpb_mlp(table))` bias) fold to host constants, leaving a graph the
same shape as V1's. Its `PatchEmbed` also carries a real 1×1 convolution that
SwinIR's does not, which is invisible until a checkpoint reports unread tensors.

Legacy `torch.save` containers (pre-1.6, five concatenated pickles and no ZIP)
now load, which unlocks the ESRGAN-era back-catalogue — `4x-UltraSharp` among
them. Writing a synthetic container for the test exposed a defect in the pickle
VM: `read_long` shifted past the register width for integers wider than 8 bytes,
and torch's own file magic is ten bytes, so **every** legacy file hit it — a
panic in debug, silent corruption of the low bytes in release.

Checkpoints that carry both `params` and `params_ema` (BasicSR's default) now
resolve to the EMA copy instead of failing every wrapper's unanimity test, and
`thop` profiler buffers are stripped — 545 of OmniSR's 728 tensors are
`total_ops`/`total_params`, which would otherwise defeat the guard that reports
tensors a build never read.

Three variants are **refused with their reason** rather than guessed at, because
each shares key names with something supported and would have produced a
plausible, wrong image: the NTIRE-2023 SAFMN (global response norm, no trunk
skip, biasless convolutions and pooling by `2^(i+1)` — even its tile multiple
differs), Swin2SR's `pixelshuffle_aux` (takes a second bicubic input), and
OmniSR with `pe=False` (the window size is then stated nowhere in the weights).

**MambaIRv2 is deterministic here and the reference is not.** Upstream routes
tokens through `F.gumbel_softmax(..., hard=True)` with **no `self.training`
guard anywhere in the file**, so the released model draws fresh Gumbel noise on
every forward — the token ordering, the selective scan and therefore the image
differ run to run — and `torch.sort(..., stable=False)` over `num_tokens`
distinct values adds tie-breaking on top. This port takes the `argmax` of the
routing logits with a stable sort: a nondeterministic upscaler is not useful,
and matching PyTorch's RNG stream *and* its unstable sort is unachievable
regardless. Two exact simplifications follow — the trailing `LogSoftmax` cannot
change an argmax, and the prototype table is a product of two frozen parameters,
so it folds to a host constant and the prompt becomes one `Gather`. The scan
maps onto `Op::SelectiveScan` with the reference's `D·x` skip added outside.

Each is a transliteration of its reference module, verified against the upstream
source rather than from memory. Supporting machinery:

- **Architecture detection from a bare state dict.** These checkpoints carry no
  config, so every hyperparameter is solved from key names and tensor shapes —
  window size from the `(2W−1)²`-row bias table, HAT's overlap ratio from its
  second table, DySample's group count from the offset conv's width. The three
  parameters that genuinely leave no trace are documented as assumptions instead
  of guessed silently. A checkpoint tensor the graph never reads is a hard
  error, since it almost always means a mis-detection.
- **Tiled inference.** Graphs are shape-static, so the network compiles once for
  one tile and every tile — edge tiles included — is that shape, edge-replicated.
  Peak memory follows the tile, not the image. A test pins that tiled and
  untiled runs agree.
- **Compile-time folding.** SPAN's `Conv3XC` reparameterization, the Swin
  lineage's relative-position bias, DAT's `DynamicPosBias` MLP and every
  `BatchNorm2d` are resolved while building. Keeping the Swin mask at
  `[1, nW, 1, N, N]` rather than the `[B·nW, nH, N, N]` a fused attention op
  wants is 590 KB instead of 340 MB on a 192² tile.

Two upstream findings worth recording:

- **HAT's overlapping-cross-attention index is reproduced bug-compatibly.**
  Upstream shifts by `ws − ows + 1` where centring needs `(ows − ws)/2 + ws − 1`,
  so the index goes negative and PyTorch wraps it from the end of the table. The
  released weights were *trained* that way; centring it would silently permute
  the learned bias.
- **A release SPAN checkpoint's stored `eval_conv` is 11% stale.** The reference
  recomputes it on every forward, so the training branches are authoritative —
  the fold is recomputed here and matches the unfused branches to 5e-7.

Validated on **38 release checkpoints** spanning all fifteen architectures,
scales ×1 / ×2 / ×3 / ×4, and **every upsampler tail** — `pixelshuffle` (single,
×2×2 chained, and ×3), `pixelshuffledirect`, `nearest+conv`, `DySample`, the
`residual` denoising tail, and Real-CUGAN's transposed-convolution and
valid-U-Net tails. Correlation runs 0.9496 (anime restoration, which is
*supposed* to change the picture) to 0.9999.

PLKSR was the last architecture without real weights — the community moved
wholesale to RealPLKSR, so nothing is on HuggingFace or in any model zoo. The
author's official DF2K checkpoints are on Google Drive; all three (×2, ×4 and
tiny ×4) load with zero unread tensors and round-trip at r=0.9997.

The ×1 case previously had no real-weight coverage at all: the round-trip test
skipped anything below ×2, so the restoration path was only ever exercised
structurally. A ×1 model is now compared to the input directly.

Real weights found three things the synthetic suite could not:

- **The lightweight SwinIR tail was misdetected.** All three tails carry
  `upsample.0.weight`, so matching on it first read `UpsampleOneStep` as
  classical and then failed looking for a `conv_last` it never had. Ordering the
  checks by `conv_up1` → `conv_before_upsample` → `upsample.0` fixes it.
- **DAT's `qkv.bias` lives under `attn.`**, so `qkv_bias` came back false and
  every attention bias was silently skipped. Caught by the unread-tensor guard,
  which is exactly what it is there for.
- **Tile defaults cannot be a per-family constant** — cost varies by two orders
  of magnitude *within* a family. Measured peak RSS per input pixel: SPANV2
  0.0027 MB, Compact 0.020, RealPLKSR 0.25, because im2col row width is
  `C_in·k²` and RealPLKSR's whole idea is a 17×17 kernel. The flat transformer
  default likewise had HAT-L OOM-killed while SwinIR-M was fine at the same
  size. `ModelConfig::peak_working_set_bytes` now estimates the convolution and
  attention terms with multipliers **fitted to measured RSS**, and
  `default_tile` shrinks along the window grid to a 2 GiB budget: tiles come out
  394 / 278 / 98 / 80 for SPANV2 / Compact / RealPLKSR / HAT-L, landing peak RSS
  between 0.5 and 2.5 GB where RealPLKSR previously reached **16.6 GB**.

Two memory fixes found by measuring rather than reasoning:

- **MambaIRv2 was costed as a Swin-family transformer and is not one.** Its
  `ConvFFN` runs a depthwise 5×5 at twice the embedding width in *every* layer —
  108 of them in MambaIRv2-L — so its scratch behaves like a conv net's. The
  estimate was 5× low and a real checkpoint peaked at **10.6 GB**; costed with
  the conv-net live-buffer count it takes tile 64 and 2.26 GB, and the three
  measured tiles fit `0.391 MB/px² · tile² + 663 MB` to within 0.5%.
- **`Upscaler` no longer holds a second copy of the weights.** It handed them to
  the graph and then kept the whole `Checkpoint` alive so a later recompile
  could use it — 231 MB on DRCT-L, 158 MB on HAT-L, for an event that usually
  never comes. It now remembers the path and re-reads if asked; an `Upscaler`
  built from in-memory tensors still keeps them, since there is nothing to
  re-read from. `holds_weights()` reports which.

**Transparency is preserved.** Every path previously went through
`image::open(..).to_rgb8()`, which flattens a logo, icon or sprite sheet onto an
opaque rectangle and reports success — a silent data loss for exactly the assets
people upscale most. `Upscaler::upscale_rgba8` splits the mask off and
`AlphaMode` chooses its treatment: `upscale` (default) runs it through the same
network so mask and colour edges agree, `resize` bicubic-resamples it for
roughly free, `discard` returns RGB. A fully opaque image skips the second pass.
None premultiply, and the docs say why.

**ESRGAN / RRDBNet.** By volume the most deployed super-resolution architecture
there is — `4x-UltraSharp`, `4x_foolhardy_Remacri`, BSRGAN, RealSR and most of
OpenModelDB are this network. It ships under **three** naming schemes (old
`model.N…`, Real-ESRGAN `conv_first`/`body.i.rdbj`, BSRGAN
`RRDB_trunk.i.RDBj`), so `weights::esrgan_to_old_arch` normalizes all three on
load, and a test asserts the three layouts detect to an identical config.
Covers ESRGAN+ (the 1×1 side paths) and the `pixel_unshuffle` variants
Real-ESRGAN uses at ×1/×2. Validated at r=0.9909 / 0.9906 / 0.9671.

Writing it surfaced two bugs, both caught by tests before a real model hit them:

- **`esrgan_to_old_arch` triggered on `conv_first` alone**, which SwinIR, DRCT,
  DAT and MambaIRv2 all have — it would have quietly rewritten every
  Swin-family stem into `model.0` and made them undetectable. Now requires an
  actual residual-dense key.
- **The trainer-wrapper stripper ate ESRGAN's own prefix.** Its old layout puts
  the whole network under `model.`, so the unanimity check fired and left
  `0.weight`, `1.sub.…`. A wrapper contains *named* submodules; a flattened
  `nn.Sequential` starts with a bare index, so that case is now refused.

**SwinIR's ×1 denoising / JPEG-artifact variant now loads.** Its tail is a bare
`conv_last` whose output is *added to the input* — no upsampler at all — and
since every other tail carries `conv_last` too, the check has to come last.
Released checkpoints of this shape (`005_colorDN_DFWB_…`) previously failed
outright with "no recognizable upsampler tail". Added as `Upsampler::Residual`,
which asserts ×1 rather than silently producing a size mismatch.

Five runnable examples, each answering a question you have when you download an
upscaler: `identify` (walk a folder of unlabelled checkpoints and say what each
one is, without loading a weight into a graph), `upscale` (the whole API end to
end), `tile_sweep` (measure the real memory/speed trade-off on this machine,
each configuration in a fresh process so peak RSS is not a stale high-water
mark), `compare_models` (run them all over one image with fidelity and cost),
and `gallery` (a labelled side-by-side sheet, captioned with a 5×7 bitmap font
so it needs no typeface dependency; `--crop` runs the models only on the region
the sheet shows). The sweep makes the estimator's calibration visible: it reads 3–4×
conservative on a shallow net like SPANV2, whose live-buffer count was fitted to
deeper models.

The tile budget is overridable with the workspace-wide `RLX_MAX_RAM_BYTES`,
because a smaller tile is not free: only the tile's interior survives the crop,
so HAT-L on a 512×384 input took 539 s at tile 176 but **1028 s at tile 128** —
nearly twice as long for 40% less memory. A 2 / 4 / 8 / 16 GiB budget gives
HAT-L a tile of 80 / 112 / 160 / 224. `--inspect` and the runner resolve the
same way, so what is reported is what runs, and each run now prints its halo
overhead.

The tier-2 halo is now one window rather than two. Tiling is exact for a conv
net, whose receptive field a halo can cover; it cannot be for a twelve-group
transformer, where shifted windows propagate half a window per block. One window
keeps seams invisible and is honest about not being exact — and it made the
tier-2 validation suite about twice as fast with no measurable quality change
(HAT-L 47.2 → 47.5 dB).

DAT's shift schedule is independently confirmed by a release checkpoint: it
registers `attn_mask_0`/`_1` for precisely the blocks `is_shifted` predicts
(`layers.0.blocks.2`, `layers.1.blocks.0`, `layers.1.blocks.4`), now pinned as a
test.

#### Backends: all fifteen on everything testable here

`coreml` is now a feature, so the **Neural Engine** is reachable — it was
not merely unwired, it was unreachable: no feature existed, `parse_standard_device`
rejects `Device::Ane` outright, MIL caps tensor rank at **5** while pixel shuffle
and window partition are naturally rank-6, and four activations (`Mish`,
`HardSwish`, `HardSigmoid`, `LogSigmoid`) hit an `unreachable!()` inside the
backend. `Mish` is RealPLKSR's activation, so every RealPLKSR checkpoint crashed
CoreML. All 24 rank-6 sites are now written with the batch axis elided — it is
structurally 1, and `Builder::rank5` checks that rather than assuming it.

Verified numerically against CPU across all fifteen architectures: Metal 4.2e-7,
MLX 4.8e-7, wgpu 1.8e-7, Vulkan 1.8e-7, ANE 3.6e-7. CUDA and ROCm are **not**
verified — no such hardware here, and the test reports them skipped rather than
passing them silently.

`scripts/upscale_backend_matrix.py` decides the rest statically, over every
backend crate in the RLX tree. Reading `SUPPORTED_OPS` alone is not enough: an
op is reachable by three routes, and two are invisible in the claim set — the
legalize loop rewrites what a target does not claim (this is why `Pad` runs
everywhere despite only Metal and CUDA having a kernel), and `OpCaps::FUSED`
kinds decompose. CoreML leaves `SelectiveScan` *deliberately* unclaimed so the
rewriter emits the compact `Op::Scan` form; a claim-only matrix calls that a gap.
`--check` is a regression guard, not a demand that every cell be green: gaps on
a backend with no FP32 datapath are expected, anything else fails.

Ten backends cover all fifteen architectures; WebGL covers fourteen (MambaIRv2
needs `ArgSort`). QNN and XDNA fall short but are int8/fp16 NPUs — XDNA's own
source notes it has no FP32 datapath.

#### OmniSR: grid channel-attention had the wrong shape

Block attention batches over tiles and contracts positions; **grid does the
opposite**. One shape was written for both. It is invisible to a value check —
a reshape only relabels, so the flat buffer is identical either way — and what
sees it is the *next* op, the attention matmul, which contracted `nw` where it
should have contracted `ws²`. The first test passed against the buggy code for
exactly that reason; asserting the declared shape makes it fail `[6,2,3,4]` vs
`[4,2,3,6]`. The einops patterns were then re-derived independently in numpy to
confirm the oracle. OmniSR ×3 51.5 → **53.8 dB**, ×1 DeJPG 38.3 → **41.1 dB**.

**38 release checkpoints** now validate by round-trip correlation, covering all
fifteen architectures, every scale and every upsampler tail.

### Release prep (0.2.16)

Worked the documented preflight. Findings, in order of how badly they would have
bitten:

**Two hard blockers remain, neither fixable from this repo alone.**

1. **`rlx-torch-ckpt` has never been published** (404 on the sparse index). It is
   a *normal* dependency of `rlx-upscale`, so the whole workspace fails to
   resolve from crates.io — `cargo metadata` dies outright with the local
   `.cargo/config.toml` patch moved aside. This is the `rlx-opscope` situation
   recurring with a new crate. The crate exists upstream at
   `../rlx/crates/io/rlx-torch-ckpt` and carries no `publish = false`; it simply
   was never uploaded. Publish it with the next upstream release and this
   clears. Every *other* upstream pin was checked against the index
   individually: all 28 resolve at 0.2.16.
2. **54 untracked files, including two whole crates** (`rlx-dsv41`,
   `rlx-jina-ocr`) and **10 `src/` modules of `rlx-models-core` that are
   `mod`-declared in `lib.rs`**. `cargo publish` refuses a dirty tree, and
   `--allow-dirty` would exclude untracked files — so `rlx-models-core` would
   ship as a crate that does not compile. These need committing before a
   release run.

**Fixed here:**

- **`publish.sh` TIERS: 3 crates missing** (`rlx-dsv41`, `rlx-upscale` → tier 3;
  `rlx-jina-ocr` → tier 4). A crate absent from every tier is silently never
  published. Recomputed from `cargo metadata` over all deps incl. dev and
  optional: now 198/198 covered with **0 ordering violations** (the previous
  release's ordering fix held).
- **The `include_str!` / `.gitignore` trap, worse than the documented case.**
  Three baked assets were ignored — and three of them feed `include_bytes!` in
  *library* `src/`, not just tests, so `rlx-ten-vad`, `rlx-ten-vad-core` and the
  whisper DTW test do not build at all on a fresh clone. These were caught by
  *directory* rules (`crates/*/weights/`, `crates/rlx-whisper/fixtures/`) rather
  than the file globs the existing negations handle, and git will not descend
  into an excluded directory — so the usual `!path/to/file` negation is inert.
  Fixed with re-include-dir → re-exclude-contents → allow-one-file, and verified
  both that the three are now trackable and that a stray blob dropped in either
  directory is still ignored. 305 KB / 150 KB / 256 B, far under the 10 MiB cap.
- **Two `include_*!` calls reached outside their own package** — a strictly
  worse form of the trap above, and invisible to `.gitignore` fixes. Found by
  running `cargo package` *with* build verification rather than `--no-verify`:

  - `rlx-ten-vad-core/src/weights.rs` did
    `include_bytes!("../../rlx-ten-vad/weights/ten_vad.safetensors")`. `cargo
    package` ships only files under the crate root, so the published crate could
    never compile — `cargo package -p rlx-ten-vad-core` failed outright.
    Dependency direction is rlx-ten-vad → rlx-ten-vad-core, so the blob was
    moved *down* into core (which already owns the tensor layout) and exported
    as `weights::SAFETENSORS`; `rlx-ten-vad` now re-uses that constant instead
    of embedding a second 305 KB copy. `cargo package` on core now build-verifies.
  - `rlx-models/tests/whisper_word_dtw.rs` pulled its fixture from
    `../../rlx-whisper/fixtures/`. Nothing in rlx-whisper used it, so it moved to
    `rlx-models/tests/fixtures/` where its only consumer lives.

  A repo-wide scan now reports **0** cross-package includes. Note the library
  build passing is *not* sufficient evidence here: the stale path left behind in
  core's `#[cfg(test)]` block still compiled the lib fine and only surfaced when
  the tests were actually run.
- **`cargo fmt --all`**: 155 files were unformatted (the documented inter-release
  drift). Now clean.
- **Catalog drift, and a checker so it stops recurring.** MODELS.md says it is
  "generated from each crate's `Cargo.toml`" — there is no generator; it is
  hand-maintained and stale in four independent ways:

  - **Five model crates had no row at all**: `rlx-dsv41`, `rlx-jina-ocr`,
    `rlx-upscale`, `rlx-wespeaker`, `rlx-translate`. The last one hid from a
    substring search because `rlx-translategemma` contains its name — only an
    exact row-name comparison found it. The other ten absentees are
    infrastructure (`rlx-ssm`, `rlx-llama-base`, `rlx-vlm-base`, …) and are
    correctly excluded.
  - **Six backend cells were wrong**, including one that over-claimed:
    `rlx-mamba` advertised wgpu with no `gpu` feature. Four understated —
    `rlx-minimax-h3` (listed 4 backends, actually all 7), `rlx-diarize` (listed
    CPU, actually Metal/MLX/CUDA/wgpu/CoreML) — and `rlx-ten-vad`,
    `rlx-voice-gender`, `rlx-wespeaker` omitted CoreML.
  - **The category table was independently stale**: it read 45 for TTS against
    46 actual rows, which is why the headline said 176 while the tables summed
    to 177.
  - Counts recomputed from the rows: **182 families across 16 categories**,
    updated in MODELS.md (headline, category table, total) and the three places
    README repeats it.

  Added `scripts/check-models-catalog.py`, wired into the fmt-clippy workflow
  and `just lint-all` (metadata only — no build). It checks all seven failure
  modes above: missing crates, rows naming crates that do not exist, cells
  claiming a backend the crate has no feature for, cells omitting one it does
  have (unless the cell carries a prose status caveat), category counts, the
  total, the headline counts in both files, and `STUB` markers against the crate
  description. Mutation-tested — deleting a row, renaming one to a nonexistent
  crate, over-claiming a backend, understating one, and corrupting either count
  each produce a finding, so it is not vacuous.
- **149 GB of stale `target/debug`** — the known `cargo test --workspace` blowup
  — left the volume 98% full with 40 GiB free, too little to run the clippy
  preflight at all. Removed; 169 GiB free.

**Release documentation.** The process lived in `publish.sh`'s header and in
nobody's notes, so each release rediscovered the same traps. Added
`docs/releasing.md`: preconditions, the gate commands, and the five trap classes
with what each actually broke. Linked from AGENTS.md.

**Stale instruction docs corrected.** AGENTS.md and TODO.md still told readers
the workspace pins `^0.2.14` — it has been `^0.2.16` since the last bump — in
six places, and two were wrong in a second way: `rlx-models-core` was described
as tier 0 (it is tier 1) and `kitten_tts_mini_rlx` as tier 2 (tier 0, ahead of
`rlx-kittentts` in tier 7). AGENTS.md also had `rlx-opscope` at upstream tier 9;
it is tier 8. Recomputed all of these from `publish.sh` rather than editing by
hand.

**Tier 0 verified end to end.** All **28** tier-0 crates `cargo package` *with*
build verification (not `--no-verify`, which is what missed the include trap) —
28 ok, 0 failures, largest 783.9 KiB. Tier 0 is the only tier verifiable ahead
of a release; every later tier depends on crates not yet on the index, so their
packaging failure is expected rather than a defect.

**The upstream blocker is fully characterized.** `rlx-torch-ckpt` packages and
build-verifies cleanly in `../rlx` (74 KB, complete metadata, no
`publish = false`) and depends only on third-party crates — anyhow, flate2,
half, plus a tempfile dev-dep. Nothing gates its upload; it sits in upstream
tier 0 and was simply missed. One `cargo publish` clears this repo's blocker.

**New: `scripts/check-publishable.py`.** Every blocker found this release was
caught by hand, with throwaway scans — nothing would stop any of them recurring.
This script is those scans made permanent. It checks the classes that no cargo
command catches, because they only surface from `cargo publish` or on a *fresh
clone* that lacks files the developing machine happened to have:

1. `include_str!`/`include_bytes!` escaping its own package (the published crate
   cannot compile; `cargo package --no-verify` misses it too);
2. an include target that git ignores (compile-time input, so a fresh clone
   cannot build the target at all);
3. a path dependency with no version — self dev-deps excluded, since cargo
   strips those;
4. a packaged crate over the 10 MiB crates.io cap;
5. `publish.sh` TIERS coverage, **ordering**, and crates listed in more than one
   tier (last one silently wins).

With `--release` it also checks that every upstream `rlx*` pin resolves on the
crates.io index, and that the tree is clean — both of which fail every ordinary
development day, which is why they are opt-in. Running it today reports exactly
the two known blockers and nothing else.

Wired into the fmt-clippy workflow and `just lint-all` (offline, fast); `just
release-preflight` runs the full set.

All nine checks are mutation-tested rather than trusted because they went green:
each was made to fire by breaking the thing it guards. Two attempts initially
produced *zero* findings and both turned out to be bad tests, not weak checks —
the tier-ordering mutation added a crate to a second tier without removing it
from the first, which last-wins made a no-op (that near-miss is why the
duplicate-tier check now exists), and an earlier over-claimed-backend mutation
edited the description cell instead of the backend cell.

**Checked, no action needed:** five crates carry a self path dev-dep with no
version (`rlx-orpheus`, `rlx-moshi`, `rlx-mimi`, `rlx-neutts`, `rlx-kyutai-tts`
each depend on themselves to enable test features), which looks like the
publish-rejecting class of bug — it is not: a scratch-crate probe shows cargo
strips self dev-deps entirely, emitting an empty `[dev-dependencies]`. No
publishable crate has a genuine versionless path dep. Packaged size is far under
the 10 MiB cap — largest is `rlx-qwen3-tts` at 1.73 MiB compressed, and
`rlx-assets` dry-run packages at 93.3 KiB. Also: all package versions agree at 0.2.16 (the one
crate at 0.1.0 is `publish = false`); the two bench crates that pin `rlx-ir` /
`rlx-runtime` literally instead of through `[workspace.dependencies]` are
currently consistent, so they are noted rather than churned.

### rlx-qwen35 prefill/decode equivalence

`rlx-qwen35` had no test that decode, stepped one token at a time, reproduces
what prefill computes for the same context (`rlx-glm5next` has one). Building
the equivalent — `tests/decode_equivalence.rs` — turned up three real bugs.
All are fixed; greedy generation on `Ternary-Bonsai-2-27B` now matches the
reference runtime character-for-character over 40 tokens instead of the first
6, and plain `Qwen3.8-27B` is fixed with it.

The two state bugs share one cause: **the prefill-cache graph is built once at
`max_seq` and every shorter prompt is padded up to it.** Attention is immune —
it masks padding causally and has its padded KV scrubbed afterwards
(`zero_prompt_padding_kv`) — but the recurrent layers carry state forward, and
nothing masked it.

- **FIXED: padding advanced the GatedDeltaNet scan.** Every padded position
  still updated the state, so the exported state described the prompt *plus* a
  run of pad tokens, and the amount depended on `max_seq`. The scan now takes a
  per-position pad mask and zeroes `g` and `β` there: by the recurrence
  `S *= exp(g); S += k ⊗ ((v − Sᵀk)·β)` a step is a no-op only when **both**
  are zero — zeroing `β` alone still decays the state.
- **FIXED: the short-conv window was exported from the padding.**
  `narrow(padded, 1, seq, k-1)` takes the last `k-1` positions, which are pad
  tokens unless the prompt fills the compiled extent. It now gathers the window
  ending at the last real token.
- **FIXED: `predict_logits` returned position 0.** With
  `last_logits_only == false` the logits are `[batch, max_seq, vocab]` but the
  row was sliced at `b * vocab` — always the first prompt position, so every
  call returned the same logits however the prompt grew.

Both graph inputs default-degrade: `gdn_pad_l{il}` is "pad" polarity and
`gdn_conv_shift_l{il}` is an offset, so an unbound input reads zeros and
reproduces the previous behaviour rather than corrupting the scan.

Also added `cpu_gated_delta_net_carry_splits_without_changing_the_scan` in
`rlx-runtime`, which pins the contract the whole hybrid decode path rests on:
splitting a scan into `s-1` + `1` must be exact. It passes — the op was never
at fault, which is what let the search converge on the wiring.

### Lazy `token_embd` (rlx-qwen35) — 24.0 → 18.9 GB

Ternary Bonsai 2's embedding table is `[248320, 5120]`: 278 MB packed, 4.74 GiB
expanded to F32, and decode reads one row per token. It now stays packed and
rows are gathered on demand, with the `prism.hadamard` inverse applied per row
rather than to all 248320 at load. Weight load drops 4.69 s → 11 ms.

Gated on the packed table being readable (`token_embd_lm`) and host-gather
being on, so neither the lookup nor the LM head needs the F32 copy — **tied
heads included**, since both the host and graph LM-head paths prefer
`token_embd_lm` and only fall back to the F32 table when it is absent. Every
packed qwen35 model benefits:

| model | RSS dense | RSS lazy |
|---|---:|---:|
| Ternary Bonsai 2 (untied) | 24.05 GB | 18.78 GB |
| Bonsai 1 Q1_0 (tied) | 16.51 GB | 11.61 GB |
| Qwen3.8-27B-Q3_K_S (tied) | 32.24 GB | 20.19 GB |

`RLX_QWEN35_NO_LAZY_EMBED=1` opts out, and
`lazy_embed_rows_match_the_materialized_table` checks the gather row-for-row
against the dense table on real weights.

`Qwen35Weights::token_embd` is now `pub(crate)` behind an accessor that
**panics** under a lazy table instead of handing back an empty slice — callers
index it directly, so empty reads as zeros and the model would emit plausible
tokens from nothing. That turned ~45 sites across five crates into compile
errors, each resolved deliberately: metadata to `embd_elems()`, gathers to
`embed_row_into`, excluded paths left to panic. `from_dense_parts` is the
constructor for adapters that build a bundle outside the loader.

Also added `rlx_gguf::dequant_typed`, split out of `GgufFile::dequant_f32`, so
a caller can decode a slice — one embedding row — without materializing the
whole tensor.

### Stream packed weight uploads (rlx-qwen3)

The same mmap-borrow upload as the qwen35 fix below, in two places:
`high_level_runner.rs` borrowed the blob and handed it to `set_param_typed`
(which copies through it, faulting in the whole checkpoint), and
`generator.rs` additionally `to_vec()`d from that borrow, paying the fault-in
*and* the copy. Both now `pread`, falling back to the borrow when the loader
has no streaming backing. Output-identical on Qwen3-0.6B-Q4_K_M; only 50 MB
there because the checkpoint is 400 MB, but it scales — the same change was
worth 4.7-19.6 GB on qwen35's 5.5/13 GB checkpoints. Revert with
`RLX_QWEN3_MMAP_UPLOAD=1`.

### Stream packed weight uploads (rlx-dflash, rlx-gemma, rlx-lfm)

The remaining three crates sharing the pattern, each now verified against real
weights rather than taken on faith. All are token-identical to the mmap arm
across interleaved runs, with the anonymous footprint flat and the whole saving
in file-backed pages — the expected signature.

| crate | checkpoint | RSS | revert |
|---|---|---|---|
| rlx-gemma | translategemma-4b Q4_K_M, 2.49 GB | 24.08 → 22.10 GB (−1.98) | `RLX_GEMMA_MMAP_UPLOAD=1` |
| rlx-dflash | Qwen3.8-27B-DFlash2 Q4_K_M, 1.14 GB | 22.29 → 21.31 GB (−0.98) | `RLX_DFLASH_MMAP_UPLOAD=1` |
| rlx-lfm | LFM2-350M Q4_K_M, 229 MB | 1.28 → 1.11 GB (−0.17) | `RLX_LFM_MMAP_UPLOAD=1` |

Two of the three needed more than the upload site. `rlx-dflash` and `rlx-lfm`
reach the checkpoint through wrapper loaders (`ReplayLoader`, and `GgufNameShim`
which remaps HF names to GGUF ones); both only forwarded
`tensor_bytes_borrowed`, so without also forwarding `read_tensor_bytes_into` the
trait default returns `false` and the change is a silent no-op that still
borrows. Instrumenting `bind` confirmed the live path: 57 packed tensors /
1.09 GB, all streamed, and 0 streamed under the revert flag. `rlx-dflash`'s
`dflash_forward` example carries its own copy of `bind` — fixed too, or the
example would keep demonstrating the pattern the library no longer uses.

Gemma's fused-component case concatenates several tensors into one param; it
streams each into a second scratch and extends, so a fused upload still costs
one component rather than the whole mmap.

### LFM2 decode was silently non-exact — `RLX_Q4K_FUSED_MIN_N` override removed

`build_decode_session` forced `RLX_Q4K_FUSED_MIN_N=2048` on CPU, citing a "~2×
decode win". That is what broke `decode_parity_live` (below), and re-measuring
on real checkpoints does not support the claim.

The fused kernel quantizes the *activation* to Q8_K. Upstream rlx defaults it
off — "rlx keeps decode on f32 for fidelity (and decode↔prefill parity)" — and
this override silently reversed that policy process-wide, via an `env::set`
from inside a builder (so it leaked into every other model in the process).
Once the threshold is low enough to pull in the FFN matmuls, the error
compounds through every remaining layer; the LM head alone is terminal and
cannot compound.

Agreement with the f32 prefill reference, 8 random-id prompts × 8 tokens
(deterministic, so these numbers are exact):

| `RLX_Q4K_FUSED_MIN_N` | 350M | 2.6B |
|---|---|---|
| unset (off) | 100% | 100% |
| 65536 (LM head only) | 100% | 100% |
| 4096 | 75.9% | 90.6% |
| 2048 (was the default) | 72.2% | 92.2% |

Throughput, arms alternated **inside one process**, min-of-5. Sequential
per-config runs were worthless here — two arms executing identical code
differed by 28% on machine drift, and a first pass that way produced a
confident "2048 halves 2.6B throughput" that is simply not true:

| arm | 350M | 2.6B |
|---|---|---|
| off | 33.7 tok/s | 8.4 tok/s |
| 65536 | +1.4% | −0.4% |
| 2048 | −14.1% | +11.1% |

The threshold also decides peak RSS, because the exact path dequantizes Q4_K
into an f32 cache while the fused kernel reads packed bytes in place — on 2.6B,
**14.45 GB exact vs 5.17 GB fused**, stable to ±0.02 GB over three interleaved
rounds (8.7× vs 3.1× amplification of the 1.67 GB checkpoint). An earlier draft
of this entry quoted 13.39/4.57 from single unrepeated runs; the direction was
right but the figures were not reproducible.

So the old default was strictly worse on 350M (slower *and* lossy), while on
2.6B it bought ~11% decode and ~8.8 GB for non-bit-exact output. That is a real
tradeoff — and on a memory-constrained box the RSS saving dwarfs the throughput
delta — but not one to make silently on a crate whose own test asserts
exactness. Default is now upstream's (off, exact); opt in with
`RLX_Q4K_FUSED_MIN_N=2048`. Cache-thrash protection is untouched: rlx-cpu's
`prefer_cached_blas` routes to the fused kernel on its own when the f32 cache
would thrash, independent of this threshold.

`decode_parity_live` now passes on both LFM2-350M and LFM2.5-2.6B.

### New: exact Q4_K decode GEMV that never materializes f32 (`RLX_Q4K_EXACT_GEMV`)

Removing the override above left a gap. For a Q4_K decode GEMV there were only
two strategies, and neither is exact-and-cheap:

- **cached-f32-BLAS** — exact, but materializes the matrix as f32 (8.7× the
  checkpoint on 2.6B);
- **packed Q8_K** — reads packed bytes in place, but quantizes the activation.

`prefer_cached_blas` switches to the second on its own once `cache_thrashing()`
trips, so on a machine where the f32 cache does not fit, decode numerics
silently degrade. `RLX_DEQUANT_CACHE=0` is exact but re-dequantizes the whole
weight per call.

Added a third arm: dequantize each Q4_K super-block to f32 on the fly and dot it
against the *unquantized* activation, row-parallel over the output. The
primitive for this (`rlx_gguf::q4_k_dot_f32`) already existed but was referenced
only by tests — no GEMV was built on it. Opt in with `RLX_Q4K_EXACT_GEMV=1`.

Throughput against the other two *exact* options (arms alternated in one
process, min-of-4):

| exact path | 350M | 2.6B |
|---|---|---|
| cached-f32-BLAS | 36.0 tok/s | 9.0 tok/s |
| `RLX_DEQUANT_CACHE=0` | 12.0 | 1.8 |
| this kernel | 23.8 | 5.1 |

It does not beat the f32 cache when the cache fits; it is **2.8× the throughput
of the only other exact option when it does not**. `decode_parity_live` passes
on both LFM2 checkpoints with this arm enabled and fails with the Q8_K arm, and
a unit test pins it within 1e-4 of f32 BLAS while asserting it is strictly
closer than the Q8_K arm (so the test keeps its teeth if either kernel is
retuned).

**Not established: its effect on RSS.** Routing is adaptive — `cache_thrashing()`
is accumulated runtime state — so per-arm peak RSS is path-dependent and did not
hold still across flag combinations. One oddity is worth flagging because it
reproduces *without* this kernel: lowering `RLX_Q4K_FUSED_MIN_N` from 2048 to 1
sends strictly more matmuls to the packed path, yet *raised* 2.6B RSS from
5.17 GB to 14.46 GB. That is backwards and is left open.

### rlx-gguf streaming reads: drop a checkpoint-sized memset and a per-tensor `open`

Two pieces of waste in `read_tensor_bytes_into`, the shared path behind every
crate's streaming weight upload:

- `buf.clear()` before `buf.resize(nbytes, 0)` forced the resize to zero-fill
  all `nbytes` immediately before `read_exact_at` overwrote every one of them —
  a memset of the entire checkpoint per load. `Vec::resize` only initializes
  elements it *adds*, so dropping the `clear()` zeroes just the growth delta: a
  reused scratch now pays at most the largest single tensor, once.
- `File::open` ran per tensor (hundreds per model) for a full path walk and an
  `open`/`close` pair. `pread` needs no per-call seek state, so one lazily
  opened `OnceLock<File>` serves every tensor.

**No measurable end-to-end win**, and the honest reason is worth recording: an
interleaved A/B on translategemma-4b (2.49 GB) put best-of-5 at 5.72 s new vs
5.43 s old — load is disk-bound, and ~2.5 GB of memset is ~0.12 s of it. An
earlier sequential reading suggested 9.7 s → 5.2 s; that was page-cache state,
not the change. Both fixes are kept as removal of plain waste, not as a
speedup. Verified unchanged: rlx-gguf/rlx-cpu suites (292 + 98 tests), the
Bonsai-2-27B real-weight suite on the 5.95 GB PTQ1_0 checkpoint, and
token-identical output from gemma, dflash and lfm.

#### How the LFM2 parity failure was tracked down

`decode_parity_live` is gated on `RLX_LFM_WEIGHTS`, and with no LFM2 checkpoint
on this machine it had never run. Pulling LFM2-350M to verify the upload change
ran it for the first time: it **fails**, byte-identically with and without the
change (`RLX_LFM_MMAP_UPLOAD=1` reproduces the same two token lists), so it is
pre-existing and unrelated. `warm_cache_speedup_live` passes.

The trail is worth keeping, because two plausible readings were both wrong.
Trailing pad in `generate_prefill` was ruled out first (its first token is
invariant across `n_new` 1→24). "Near-tie greedy flip" was ruled out next, and
decisively: at length 6 decode picked a token ranked **16th** in the prefill
logits, 1.272 below the top — a tie flip would be rank 1 at ~1e-6. Dumping full
logit vectors then showed decode's `sum|logit|` running 1–6% above prefill's at
*every* length, including lengths where the argmax happened to agree, so the
"sporadic" divergence was really a systematic one that only surfaced when the
top-2 gap was narrow. Finally, comparing at length 1 — one token, zero history,
empty KV — still disagreed, which ruled out the KV/conv-state advance and
pointed at the graph itself. The `env::set` in `build_decode_session` was two
lines up from there.

### Stream packed weight uploads (rlx-qwen35) — RSS −4.7 to −19.6 GB

`upload_packed_opt`'s low-mem path borrowed each packed tensor from the mmap.
The arena copy reads through that borrow, so it faulted in every page of the
checkpoint and left a second full-size copy resident; on macOS that is
unreclaimable short of `munmap`, since `release_mapped_pages` is a Linux-only
win. Now it `pread`s into one reused scratch, capping resident cost at the
largest single tensor.

| | RSS before | RSS after |
|---|---:|---:|
| Ternary Bonsai 2 (5.5 GB weights) | 28.6 GB | 23.9 GB |
| Qwen3.8-27B-Q3_K_S (13 GB weights) | 41.7 GB | 22.1 GB |

Reproducible to 0.1 GB, benefits every qwen35 model, costs ~400 ms of one-time
upload. Falls back to borrowing when the loader has no streaming backing;
revert with `RLX_QWEN35_MMAP_UPLOAD=1`. rlx-llama32 already made this trade and
measured 5.93 GB vs 0.07 GB on a 6 GB checkpoint.

### PTQ1_0 arena scratch (rlx-metal) — peak memory 32.2 → 23.1 GB

Adding the fused PTQ1_0 GEMV updated the run-time dispatch but not the
compile-time arena sizing: `dequant_gguf_scratch_bytes` had skip arms for
Q1_0, Q2_0 and G8_0 and none for PTQ1_0, so every graph still reserved an f32
dequant slab the fused kernel never writes. Ternary Bonsai 2's
`[248320, 5120]` LM head is ~5 GiB of that on its own.

**Peak memory footprint 32.2 → 23.1 GB (−28%)**, reproducible to 0.1 GB across
repeats. The skip mirrors the dispatch exactly (`m == 1`, `k % 128`, `n % 8`,
same off-switch), because the two disagreeing in either direction is a bug: one
way reserves a slab nothing writes, the other way the encoder reaches for one
that was never allocated.

Note for anyone measuring this: quote `peak memory footprint` from
`/usr/bin/time -l`, not `maximum resident set size` — RSS moved by several GB
between identical runs here while the footprint was stable to a tenth.

### Small-m square GEMMs to MPS (rlx-metal cost model)

`prism.hadamard`'s rotation reshapes to `[-1, 1024]`, so it runs as
`m = 5/6/17, k = n = 1024` about 190 times per decoded token. `m` is not a
multiple of 8, so `Simd` is ineligible and these land on `SimdPadded`, which
pads `m` up to a simdgroup and wastes most of the tile. Routing them to MPS
instead takes Ternary Bonsai 2 from 111.3 → 103.2 ms/token, winning every
interleaved round.

The rule is deliberately narrow — `2 <= m < 32`, `m % 8 != 0`, `k == n >= 256`.
`k == n` is the discriminator and is not arbitrary: a square operator is a
rotation or basis change, never a transformer projection. An earlier version
without it also caught rectangular shapes and cost `Qwen3.8-27B-Q3_K_S` 1.6%.
`RLX_METAL_SMALL_M_MPS_TRACE=1` confirms the final rule fires on **no** shape
in either Qwen3.8-27B or Bonsai-1, so it cannot regress them. Opt out with
`RLX_METAL_NO_SMALL_M_MPS=1`.

Also: `RLX_QWEN35_GPU_KV=1` (in-place KV append instead of a concat) was
measured and showed no gain, so it stays off.

### Fused PTQ1_0 decode GEMV (Metal)

`Ternary-Bonsai-2-27B` decode was spending **89.8%** of its GPU time in
`dequant_matmul_gguf` (`RLX_METAL_THUNK_PROFILE=1`): with no fused kernel,
every `PTQ1_0` weight was re-expanded to an f32 scratch on every token — for a
5.9 GB model, tens of GB of dequant traffic per token.

`ptq1_0_mv_f32_sg` reads the packed 28-byte blocks straight out of the arena,
simdgroup-cooperative, modelled on `q1_0_mv_f32_sg`. Two details specific to
this format: `(b · 3ⁿ) mod 256` collapses to a single multiply because every
`3ⁿ` for `n < 5` is already < 256, and the staged element → (byte, digit) map
depends only on the lane, so it is resolved once per lane instead of once per
row.

**0.61 → 4.34 tok/s (7.1×)** (median over decode steps ≥ 2, min of 4 trials),
output character-for-character unchanged.
Decode is now 74% GEMV, i.e. mostly irreducible weight streaming. Off-switch
`RLX_METAL_PTQ1_0_FUSED_DISABLE=1`; `m > 1` (prefill) still takes the scratch
path, and `dequant_gguf_scratch_bytes` mirrors that. Also wired `PTQ1_0` into
`rlx_gguf::quantize` so the parity tests can build packed fixtures.

A second kernel, `ptq1_0_mv_f32_sg_fp`, is now the default and takes decode to
**8.1 tok/s / 124 ms**, and with the cost-model rule above **9.7 tok/s /
103 ms** — **~16× the 0.61 tok/s** this started at. Against the reference the
interleaved ratio is a stable **~2.0×**; absolute ms/token on this machine
swings by 2× with background load and should not be quoted without min-of-N. Ported from the fork's own Metal GEMV: the base-3 digit extraction
moves entirely into the float pipe — `digit_n = floor(3^(n+1)·u) − 3·floor(3^n·u)`
for `u = b/256`, telescoped so each trit costs one floor and one fma — because
this ISA cannot co-issue integer and floating-point work; and each thread owns
whole *bytes*, so a block's bytes are read once instead of five times.
`RLX_METAL_PTQ1_0_INT_PIPE=1` selects the integer kernel, which is kept as the
A/B reference and stays covered by its own parity test.

With it the profile is flat — `dequant_matmul_gguf` 28%, `sgemm` 22%, `concat`
17% — so no single term dominates any more.

**Measurement note.** Run sequentially, the float-pipe kernel looked 40%
*slower*; run interleaved against the integer one it won 5 rounds out of 5.
The difference was drift on a machine that was also running other jobs. GPU
A/Bs here have to alternate the variants, not follow one with the other, and
`ps -Ao pcpu,comm | sort -rn | head` is worth a look before trusting a number.

**Left on the table, measured but not taken.** The Hadamard rotation is
`m=5, k=1024, n=1024` — 5 MFLOP, below `mps_threshold_flop`, so it takes
rlx-metal's `m < 32` MSL cascade. Forcing MPS (`RLX_METAL_SGEMM_MPS=1`) won 5 of
6 interleaved rounds (~6%), and the cost model already special-cases
`m == 1 && n < 64` → MPS for the same occupancy-starvation reason. Not changed:
that cascade is globally tuned with its own sweep test, and 6% from one noisy
A/B on one model does not justify moving it. Recorded in PARITY.md for a proper
multi-model sweep.

Two other attempts measured slower and were reverted, with the numbers recorded
in-place: 16 rows per simdgroup instead of 8 (`x` is already cached), and
factoring the rotation as `H_1024 = H_32 ⊗ H_32` (exactly equal, 4 KB instead of
4 MB, but the matrix stays cached and doubling the dispatch count costs more
than the traffic saved).

### Ternary-Bonsai-2-27B

`prism-ml/Ternary-Bonsai-2-27B-gguf` runs through `rlx-qwen35`. Its
architecture is unchanged from the base Qwen3.8-27B — same 64 blocks, same
hybrid 3:1 linear/full attention, byte-for-byte the hparams the crate already
ran — so the port is entirely about how the weights are *stored*.

- **Two new GGUF quant types.** `PQ2_0` (on-disk type 142) is byte-identical
  to the existing `GgufQ2_0` codec at a distinct type id, so it is a pure
  remap. `PTQ1_0` (type 143) is new: base-3 trits, five per byte, 28 bytes per
  128 weights (1.75 bpw), with the f16 scale *last* in the block. Its
  element → (byte, digit) map is staged rather than sequential, so a
  sequential walk still decodes to a valid ternary tensor — just a permuted
  one. `rlx_gguf::ptq1_dequant` is checked against the fork's own CUDA
  element-map reference, and the Metal and WGSL kernels against the CPU codec.
- **⚠ Type id 143 collides.** Doses AI's `mortar.cpp` uses it for `G8_0`
  (Pestle) and PrismML's llama.cpp for `PTQ1_0`; nothing in the id
  distinguishes them. `rlx_gguf::TypeDialect` picks per file from the
  `prism.*` metadata, leaving the historical `G8_0` reading untouched.
- **The weights are in a rotated basis.** Each matrix is folded by a
  blockwise normalized Sylvester–Walsh Hadamard rotation (block 1024) with a
  per-width sign flip, so a matmul against them is only correct if the
  *activation* gets the matching transform first. `rlx_qwen35::prism_hadamard`
  parses the `prism.hadamard.*` contract and `emit_linear` applies
  permute → signs → rotation ahead of each of the 401 folded weights;
  `token_embd` stores rotated rows and is un-rotated once at load. The GDN
  `ssm_out` path additionally needs a tiled → grouped head reorder.
  Skipping any of this does not fail — the rotation is orthogonal, so the
  model still runs and still emits fluent text, just not this model's — so
  the transform lives on the weight (`Proj::Folded`), `Proj::dense()` refuses
  it rather than letting a fusion fast-path take the raw matrix, and an
  unrecognized metadata variant is an error rather than a fallback.
- **The bug this actually shipped with, caught on real weights.** The
  embedding table's inverse is `h = s ⊙ (H z)` — rotation *then* sign flip,
  because the forward transform is `H(s ⊙ x)` and the two do not commute. The
  port applied only the rotation, which flipped the signs of the wrong 5120
  channels: the model loaded, ran, and emitted EOS as its first token. Every
  synthetic test passed, because they were self-consistent about an ordering
  that was wrong on both sides. What found it was dumping the reference's own
  graph (`llama-eval-callback`), where the `MUL(..., prism.hadamard.signs.5120)`
  right after the embedding rotation is plainly visible.
- **Real-weight parity.** Prefill argmax is token-exact with the reference —
  `"The capital of France is"` → `"Paris.\nThe capital of Germany"` — and
  ChatML answers `"Paris"`. See PARITY.md, including a pre-existing
  `rlx-qwen35` prefill-vs-decode drift (reproduced on plain Qwen3.8 with no
  Hadamard) and the current 0.61 tok/s decode.
- **Tests.** `tests/prism_hadamard_projection.rs` builds the same tiny model
  twice — folded and dense — and requires matching logits;
  `tests/bonsai2_manifest.rs` checks the port against the real published
  header (175 KB fixture, `scripts/bonsai2_manifest.py`): 402 ternary tensors,
  401 folded names all resolving to real tensors, sign vectors splitting at
  5120/6144/17408, and 48 permuted `ssm_out` weights at (128, 16, 3).

### Jina-OCR-v1

`jinaai/jina-ocr-v1` is a DeepSeek-OCR derivative, and `baidu/Unlimited-OCR`
is the same architecture under the same 2 722 tensor names — so the new
`rlx-jina-ocr` crate reuses `rlx-unlimited-ocr`'s SAM+CLIP DeepEncoder, linear
projector, expert packing and compiled MoE decoder outright, and contributes
only what actually differs. Every difference fails silently, which is why each
has its own test:

- **No sliding window.** jina's `config.json` has no `sliding_window` key and
  `modeling_deepseekv2.py` reads it as `getattr(self, "sliding_window", None)`
  — plain causal attention over the whole history. Unlimited-OCR's card sets
  `128`, so inheriting that fallback would have clamped attention to the last
  128 tokens and still produced fluent text.
- **`rope_theta = 1e6`**, where Unlimited-OCR omits the key and falls back to
  `10_000`.
- **No BOS.** The processor calls `text_encode(..., bos=False)` and
  `tokenizer_config.json` sets `add_bos_token: false`; the id stream starts at
  the `<|User|>:` chunk. Unlimited-OCR's own assembly prepends BOS.
- **`dynamic_preprocess(max_num=9)`** vs 32, and the tiling threshold is
  `image_size` rather than a constant.
- **n-gram guard 35 / 1024** with `<td>` / `</td>` whitelisted, so long tables
  are not truncated by the repeat blocker.

Also new: `postprocess` (ports `decode_ocr`, `parse_refs` and
`extract_markdown_and_crops` — the two reference regexes are hand-scanned so
the crate stays dependency-free) and `mtp`, the FastMTP draft head.

**Verified on the real checkpoint (CPU).** Running it end to end surfaced two
pre-existing bugs in the shared compiled decoder — see below — and after fixing
them rlx reproduces the reference implementation's greedy output **token for
token** (first-token logit 14.877 vs 14.875) and transcribes the bundled page
correctly. `tests/checkpoint_inventory.rs` additionally checks every tensor name
and shape against the published safetensors *headers*, read over HTTP range
requests and baked into a 16 KB fixture; `tests/prompt_ids.rs` checks the chat
template and prompt-id splice against the checkpoint's own `tokenizer.json`.

**All five available backends now agree.** Fed the same `inputs_embeds`, CPU,
Metal, MLX, wgpu and Vulkan return token-identical greedy output, matching the
reference. Getting there took two more backend fixes (below). CUDA/ROCm are not
present on this machine and were not checked.

| precision | cpu | metal | mlx | wgpu | vulkan |
|---|---|---|---|---|---|
| f32 / f16 | exact | exact | exact | capacity | capacity |
| q8_0 / q4_0 | exact | exact | exact | exact | exact |

`capacity` is a loud refusal, not a wrong answer: wgpu declines to stripe a
20 GiB activation arena across 4 GiB buffers (striping silently corrupts), and
Vulkan reports the 10.3 GiB weight prefix exceeding `maxStorageBufferRange`.
Both run fine at `--lm-precision q8_0`.

**FastMTP is now wired into decode.** The multi-token-with-past graph it
needed (`seq = K+1`, `MaskKind::Bias`) is the chunked verify pass added to
`rlx-unlimited-ocr` (see below); `JinaMtp` loads the draft head and the
target tensors it shares, and `JinaOcrRunner::generate_with_mtp` runs it.
On the real checkpoint the output is **byte-identical** to plain decode, which
is the property that matters — the verify pass keeps a draft token only when
it equals the target's own greedy pick.

It is **opt-in**, and measures as follows on the bundled page (Metal, q8_0,
927-token prompt). Two defects were found and fixed getting here: the graph
tapped the *post*-norm hidden state where FastMTP wants the pre-norm one
(`pre_norm_hidden_states` in `modeling_deepseekv2.py`), and the draft block —
a transformer layer with its own KV cache — was never primed over the prompt.

`examples/mtp_probe.rs` scores the head directly, and prints a control first:
the target's own tapped hidden, through the shared norm and host LM head, must
reproduce the tokens the target actually generated. It does, at **47/47**, so
the draft numbers under it mean something. That control is not ceremony — an
earlier version of this probe lacked it and spent a run producing draft scores
that could not be interpreted.

- **Concat order settled by measurement**, since the checkpoint ships the draft
  weights but not the module that runs them (the card points at a vLLM
  plugin): `eh_proj([enorm(e); hnorm(h)])` scores **25.5%** top-1 agreement
  with the target against **0.0%** reversed. That matches DeepSeek-V3's *code*
  — not its paper, which writes `M[RMSNorm(h_i); RMSNorm(Emb(t_{i+1}))]` with
  the operands the other way round.
- **`K = 3` from the config is the wrong depth.** The draft-depth curve shows
  step 1 accepting 25.5%, step 2 6.4%, and **step 3 exactly 0.0%** — a third of
  the host draft cost buying nothing. `K = 1` gets 1.26 tokens/round of the
  1.32 available at `K = 2`; `MtpHead::set_steps` overrides it.
- **1.30 tokens/round end to end**, which independently matches the probe's
  curve.

**And it is a net loss on this model — the draft head is not why.**
Drafting costs ~10 ms of a ~1250 ms round, under 2%. What kills it is the
verify forward: `examples/mtp_cost.rs` times a plain decode step against a
chunked one *interleaved in one process*, at a real 927-token context and
after a warm-up so no timing pays for a graph compile, and the chunk costs
**1.6x-2.6x a single-token step**. At 1.26-1.41 tokens/round that comes out at
**0.47x-0.83x** — slower than plain decode, which the 192-token end-to-end run
independently reproduces at 0.62x.

The chunk cost is also nearly flat in `n` (1247 / 1253 / 1217 ms for
`n = 2 / 3 / 4`), which says it is not the extra tokens but the `n > 1` path
itself taking a slower route than the single-token one. That is the thing to
fix: if a chunked forward reached parity with a decode step, `K = 2` at 1.32
tokens/round would be a ~1.3x win. Until then `generate_with_mtp` stays
opt-in.

Measuring this correctly took three tries, and the wrong ways are worth
recording. This box is shared: identical work returned 88 s, 129 s and 255 s
depending on what else was running, so **timing the two arms sequentially
compares load, not code** — that is what produced an apparent 1.05x speedup
that a later run flatly contradicted. Deriving the plain per-token cost by
subtracting an estimated prefill was no better, because on a 48-token
transcription the vision encode plus the 927-token prefill is ~101 s of ~131 s
and swamps what is being measured. Only interleaved timings of the three terms
in one process reproduced.

Also note the draft head needs its own host-resident f32 copy of `lm_head`
(129280x1280, 662 MB) because the draft hidden never enters the compiled
graph; `JinaMtp::host_bytes` reports it. The embedding table is *not*
duplicated — draft steps look rows up through the already-packed weights.

### rlx-unlimited-ocr

**Two silent correctness bugs in the compiled decoder.** Both predate this work
and made the compiled path wrong for *every* checkpoint this crate serves,
including `baidu/Unlimited-OCR` itself. Neither was caught because
`backend_quick_check` only asserts that logits are finite, and the end-to-end
parity test needs a checkpoint that was never downloaded. The eager
[`lm_flow`] path was correct throughout, which is what made the split
diagnosable: fed the reference implementation's own `inputs_embeds`, eager
returned the published logits and the compiled graph did not.

- **`Op::Rope` on rank-4 BHSD input wrote almost nothing** (fixed upstream in
  `rlx-cpu`'s `thunk::ops::attention::compile_rope`). It read the shape as
  `[batch, seq, hidden]` positionally, so for the `[B, H, S, D]` tensor that
  `apply_rope_bhsd` produces it took `seq = H` and `hidden = S`, sized the
  output as `B*H*S` instead of `B*H*S*D`, and left the rest of the destination
  zero. Attention over a near-zero K is uniform, so the model emitted confident
  nonsense rather than crashing. `executor.rs`'s RoPE handles rank 4 correctly;
  the thunk path shadows it. New `tests/rope_lowering.rs` pins the rotation
  (with a control proving the reshape/transpose round-trip is identity, so the
  failure cannot be a harness artifact).
- **The MoE router gathered expert weights with ONNX `Gather` instead of
  `take_along_axis`.** `[rows, experts]` × `[rows, k]` came back
  `[rows, rows, k]`, and the following `reshape([rows, 1])` silently
  reinterpreted a `rows*k` buffer as `rows` — so each token was weighted by an
  arbitrary other token's routing probability. The expert *indices* were right,
  which is why every layer stayed plausible while drifting further from the
  reference. Now `Op::GatherElements { axis: 1 }`; new
  `tests/moe_router_lowering.rs` checks selection and that the weights are raw
  softmax probabilities that do **not** sum to 1 (`norm_topk_prob=false`).

**Four more silent bugs, found by running the op suite on every backend rather
than only CPU, and across op *variants* rather than one shape.** `tests/` now carries a `common` device
enumerator, and the RoPE, GroupedMatMul, MoE-router, packed-quant,
view-readback and bucketed-decode checks all run on every backend compiled in
and present, reporting every failing backend at once. The RoPE suite covers both
pairing conventions, partial rotation, GQA head counts, decode-width sequences,
multi-batch and `heads == 1` — the last of which is where the rank-4 bug was
invisible.

- **Metal and Vulkan had the same rank-4 BHSD `Op::Rope` defect as CPU** — both
  read the shape as `[batch, seq, hidden]` positionally. Fixed the same way
  (fold leading axes into `batch`). Metal already carried a comment about
  fixing this exact class of bug for the *rank-2* case; rank 4 was missed.
- **Metal's fused per-row grouped dequant read `expert_idx` on the host without
  syncing first.** Its own doc says the indices "must already be resident"; the
  slower grouped path below it calls `sync_gpu!()` for precisely that reason,
  the fast path did not. With routing produced by a `TopK` earlier in the same
  command buffer — i.e. every MoE decoder — it read stale bytes and routed every
  row to whatever the buffer held (usually expert 0). In range, so the
  `debug_assert` passed. Fixed by syncing before the encode.

- **rlx-mlx read strided views linearly.** `to_f32` / `to_bytes` in the C++ shim
  tested `flags().row_contiguous` **before** `eval()`, and on a lazily built
  graph those flags describe nothing yet — so the contiguity check passed and
  the `memcpy` walked the base buffer, turning a sliced column into the first N
  elements of its parent. Every host-lowered op that reads a view was affected;
  it surfaced as the MoE router sending each token to the wrong expert (ids
  stayed in range, so nothing complained). Fixed with
  `materialize_row_contiguous`: evaluate, *then* test, then materialize.
  `tests/view_readback_parity.rs` pins it by reading the same view through two
  different consumers, across slice / transpose / double-narrow shapes.
- **wgpu indexed the RoPE tables by `rot_half` instead of the table's own row
  width.** Correct only when `n_rot == head_dim`; under partial rotation it read
  the wrong table row for every position past the first, while still producing a
  correctly-normed rotation — so it looked right. The CPU kernel already carried
  a `cos_row_stride` field for exactly this reason (and a comment describing the
  trap); wgpu's shader never got it, and its own Rust-side doc claimed a stride
  the shader did not use. `RopeParams` now carries `cos_row_stride`.

- **`Op::RopeBackward` carried the same rank-4 defect in *seven* backends**
  (cpu, metal, vulkan, wgpu, cuda, rocm, oneapi) — the forward fix had not been
  mirrored onto the gradient. With BHSD input it wrote `B*H*S` of `B*H*S*D`
  elements, so **100% of the gradient came back zero** in every case the new
  test covers, `heads == 1` included (where the forward bug was a no-op).
  Nothing here trains a BHSD-RoPE model, so it degraded training silently
  rather than failing. Fixed in all seven; cuda/rocm/oneapi are **untested
  here** (no device on this machine) but the change is mechanical and provably
  a no-op at rank 3. `rope_lowering.rs` now checks the backward pass against the
  negated forward rotation on every available backend.
- **A second pre-`eval()` `flags()` check in the MLX shim** (`rlx_mlx_row_bytes`)
  had the same ordering bug; there a wrong-direction flag would corrupt an
  in-place row write rather than a read. Reordered.

**No backend is clamped.** `device_supports_packed_quant` is kept as the hook
for "this backend computes the wrong answer, downgrade rather than trust the
user's `--lm-precision`", but currently returns true for everything.

Support for checkpoints with no sliding window, which the crate could not run:

- `sliding_window == 0` previously computed `window = usize::MAX` and then
  `prefill_len + window`, which overflows. Both the compiled path and the eager
  host flow now carry the window as `Option<usize>`.
- Full-causal decode grows its KV every token, so the exact-shape
  `MaskKind::Causal` graph would be recompiled once per generated token. New
  `build_unlimited_ocr_decode_built{,_from_pack}_ext` take a `use_custom_mask`
  flag that adds a `[batch, past_seq + 1]` keep-mask input and switches
  attention to `MaskKind::Custom`; `CompiledLm` pads the past to a 256-row
  bucket and masks the padding, compiling once per bucket.
  `tests/full_causal_decode.rs` asserts bucketed decode matches the exact-shape
  graph (with a negative control that unmasked padding does change the logits),
  and that one bucket compiles one graph.
- `validate()` accepts `deepseek_vl_v2` alongside `unlimited-ocr`; `SampleOpts`
  gained `ngram_whitelist`; `UnlimitedOcrRunner` gained `open_with_config` and
  `generate_from_ids` so a derivative crate can supply its own resolved config
  and prompt-id layout. Windowed behaviour is unchanged.

### FIXED: MLX returned all-zero qwen35 logits — a redundant gather, found by bisect

The one finding from the test sweep that was recorded rather than fixed. The
graph bisect it needed — compile each node as the sole output on CPU and MLX and
walk for the first divergence — puts it at node 221 of 223, and the cause is
plain once seen:

`emit_qwen35_prefill_tail` gathers the last token **out of the logits**, while
every caller that passes it a `last_token_idx` has *already* narrowed the hidden
(the flow runs `gather_last_token_dynamic` before the tail). Its own
`logit_rows = if last_token_idx.is_some() { 1 }` assumes exactly that. So the
projection is `[batch, 1, vocab]` and the trailing gather asks for index
`seq - 1` along an axis of length **1**. CPU clamps an out-of-range index and
returned the right row by luck; MLX returns zeros. Every qwen35 prefill on MLX
produced all-zero logits unless the last token happened to sit at index 0 —
which is why a 1-token prompt looked fine.

Why the earlier hunt missed it: the bisect skipped non-F32 nodes, and the
suspicion had landed on the gather's *index* path. Every isolated reproduction —
gather alone, gather + matmul, gather + RMSNorm + matmul, two I32 inputs — was
correct, because none of them reproduced the one thing that mattered: a gather
applied to an axis that was already length 1.

**The self-cleaning exclusions did their job.** Three tests carried
`assert_matches_cpu_except(.., &[("mlx", ..)], ..)` for this bug. Two of them
failed the moment it was fixed, with "excused as a known failure … but now
MATCHES CPU — drop the exclusion". That is the whole point of running an excused
backend anyway rather than skipping it.

The third kept failing, and for a different reason: **`prefill_seed_from_hidden`
never fed the GDN pad masks.** The text prefill path feeds them; the multimodal
one built its own feed list and did not, so MLX refused the graph with
`missing input 'gdn_pad_l0'` and every other backend silently scanned the
zero-padded tail as though it were prompt. Same class as the `last_token_idx`
omissions above — an unbound input that CPU tolerates.

Also fixed while clearing the last failures:

- **Two kimi-k3 decode tests had not compiled in a long time.**
  `decode_full` and `mla_decode` both sized their V cache at
  `num_heads * qk()`, but the vdim path — the default — stores V at
  `num_heads * v_head_dim`, so `concat` refused to join a `[1, s_past, 12]`
  cache to a `[1, 1, 8]` new row. Both are real decode-vs-prefill parity tests;
  the crate went from 5 passing to 34. `mla::mla_vdim` is now public, because it
  changes a shape the caller has to allocate.
- **`glare_smoke` ran CUDA on a machine with no CUDA.** Its cases gate on
  `#[cfg(feature = ...)]`, which says the backend was compiled in, not that a
  device exists. Now checks `is_available` and announces the skip.

### The rest of the finite-only tests — and four more bugs

Finishing the sweep. Roughly 26 more tests converted, and the classification
itself needed correcting: counting *files* over-reported, and several tests
flagged by a keyword scan already had real oracles. The ones deliberately left
alone are listed at the end, with the reason checked rather than assumed.

- **A test fixture mis-sized every indexer and attention-output weight in
  `rlx-glm5next`, and the suite passed anyway.** `tensor_manifest`'s shape
  dispatcher matches on name suffixes in order, and
  `blk.N.attn_output.weight` ends with `output.weight` — so it hit the LM-head
  arm first and got `[vocab, hidden]` instead of `[hidden, proj]`. The
  `indexer.*` arms sat *after* the generic `attn_k` / `attn_q_b` ones, so every
  indexer weight was sized as the attention block's. The graph was malformed in
  four places; the reshapes downstream silently fabricated or dropped elements
  to fit, and the test went green. The reshape element-count guard is what
  surfaced it.
- **`rlx-hoct` computed the wrong segment-to-segment distance for parallel
  segments.** The parallel branch pins `sc = 0` and projects onto the other
  segment, but skips the clamp-back step the non-parallel branch does — so two
  *touching* collinear segments reported the distance between their start
  points (1.0 for a unit pair) instead of 0. The test asserted only that the
  distance was finite, which the wrong answer is. It now checks three cases
  against the closed form: touching-collinear (0), parallel-offset (2), and
  skew (5).
- **Two more test files had not compiled in a long time.**
  `llama32_metal_mlx_parity.rs` and `llama32_gpu_backend_parity.rs` — the actual
  CPU-vs-GPU comparisons — were missing three `Llama32Config` fields. Being
  feature-gated, nothing noticed. They run now.
- **`glass_posterior_runs` could not fail.** `sample_posterior` writes into an
  `out_z` the caller hands in, and the test pre-filled it with zeros before
  asserting every element was finite — so it passed whether or not the function
  wrote anything. It now checks that the buffer was written, that the result is
  deterministic, that identical inputs give position-independent output, and
  that scaling the noise changes the answer.
- **`decoder_rlx_matches_eager` never compared anything to eager.** It checked
  the backend *name*, the length, and finiteness. A real comparison is not
  reachable from an integration test (`decode_forward` is `pub(crate)`) and
  would be tautological today anyway, since `decoder::rlx::decode` delegates
  straight to it. Renamed to what it verifies, with the reasoning recorded and
  a pointer to where the parity test belongs once the rlx path has its own
  implementation.

Converted to cross-backend parity: FunASR (SenseVoice, Paraformer, FSMN-VAD,
CAM++ — all four already looped every device and compared none of them),
GLM-5.3 sparse-DSA and ragged-tail prefill, Ling's softplus-gate and ungated-MLA
variants, Motif's no-sliding-window and dense-only variants, MiniMax M3 text
flow and projector, Kimi-K3's vision tower under arena reuse, VibeVoice's two
VAE encoders, TimesFM3's device sweep, and the Qwen2.5-VL / Qwen3.5-VL
multimodal prefill+decode paths. Converted to determinism / closed-form /
non-degeneracy where a device does not change the computation: TimesFM3 synth
forward, the DIAMOND posterior and re-noise, `early_stop_ddpm` (linearity), the
wake-word trainer, and the NeuTTS decoders.

**Left alone, having checked why.** The `cpu_reference_logits_finite` family
across the `*_backend_parity.rs` files is *not* vacuous in context: its siblings
call `assert_logits_match_cpu(Device::Metal, ...)` and do the real comparison,
so asserting the reference is finite is exactly the right scope for it.
`vit_parity::metal_full_output_finite` is a Metal-only guard for a specific
LayerNorm-clamp NaN, with `forward_parity_*` siblings doing the comparison.
`jlens::mlx_status` is `#[ignore = "diagnostic"]` and documented as a reporting
tool. `qwen35_forward_check`'s four tests assert real cache state
(`cache.past_seq`, output counts); the `is_finite` in them is incidental.

### Finite-only tests converted to real oracles — and the six bugs that fell out

~43 test files asserted only that their output was *finite*. That is the
weakest useful property: an all-zero result passes it, so does a tensor with one
head's worth of real values and the rest zero, and so does a router that sent
every token to the wrong expert. Converting them to compare against CPU — or,
where a device does not change the computation, to assert determinism and
framing — immediately surfaced defects that had been sitting in a green suite.

`rlx_models_core::backend_matrix` grew `assert_matches_cpu_on_all` for the
common case, plus `assert_matches_cpu_except` for backends excused by a known
bug. Exclusions are **self-cleaning**: the excused backend still runs, and if it
*passes* the test fails and tells you to drop the exclusion. An exclusion that
silently outlives its bug is how a suite stops testing the thing it was written
for.

**A harness bug first.** `candidate_devices()` was cfg-gated on *this* crate's
backend features, but model crates forward theirs to `rlx-runtime`, not here —
so the sweep would have reported "cpu only" in precisely the crates that needed
it, without even announcing a skip. It now lists every device and lets
`rlx_runtime::is_available` decide.

- **The qwen35 MoE router weighted every token by another token's routing
  probability.** `gather_(probs, top_idx, 1)` is ONNX Gather, which applies each
  row's index list to *every* row: `[rows, experts]` x `[rows, k]` comes back
  `[rows, rows, k]`, and the following `reshape([rows, 1])` silently
  reinterpreted it. The expert *indices* were right, so every MoE layer stayed
  finite and plausible. This is the same defect this workspace already fixed in
  `rlx-unlimited-ocr`'s router; it was caught here by the new reshape
  element-count guard, which turned the silent truncation into an error.
- **`enable_mtp_head = true` panicked for everyone.** `lower_qwen35_mtp_head`
  gathered the last token with a rank-2 index, which ONNX Gather turns into rank
  4, and then asserted against a rank-3 `out_shape`. Nobody hit it because the
  test that covers it needs `qwen35` *and* a backend feature together, which CI
  does not build.
- **Five llama32 backend checks had not compiled in a long time.**
  `Llama32Config` gained `sliding_window`, `sliding_window_pattern` and
  `final_logit_softcap`; the tests were never updated, and being feature-gated,
  nothing noticed. Same story for `Qwen35Weights::output_fold` in
  `qwen35_vlm_quick_check`.
- **Two tests never fed `last_token_idx`.** The graph declares it under
  `last_logits_only`; CPU ran anyway with the input unbound, so the omission was
  invisible until MLX refused. Same class as the next one.
- **qwen35 dynamic prefill was broken on MLX.** `prompt_pad_mask_feeds`
  deliberately returned an empty vec when nothing was padded, documented as "the
  caller can skip the feed, the graph input defaults to all-zero". MLX does not
  default unbound inputs, so it failed with `missing input 'gdn_pad_l0'`. It now
  always emits the mask; the buffer is `batch * seq` floats per GDN layer.
- **A nemotron-ASR fixture sized every strided conv as the wrong kind.** Its
  shape heuristic had depthwise and pointwise inverted, handing the encoder 64
  elements where it declares `[c, 1, 3, 3]` = 72. CPU ran on the short buffer;
  MLX rejected it.

Two findings are recorded rather than fixed, because both are honest behaviour
or need work beyond this pass:

- **MLX returns all-zero qwen35 logits** whenever `last_logits_only` is set and
  `last_token_idx != 0`. Metal and wgpu are fine. Not the gather, the RMSNorm,
  the matmul, the index dtype, the GDN layers, run order, MLX compile mode or
  the fusion profile — each reproduced standalone on MLX and each is correct.
  **Since fixed** — see the MLX entry above; the graph bisect found a redundant
  gather over an axis of length 1, and the `#[ignore]` and all three exclusions
  are gone.
- **wgpu's qwen35 logits are CPU's times 1.0146..1.0161** — a *uniform gain*,
  spread 1.5e-3, so the predicted token is unchanged. That signature is a
  lower-precision `inversesqrt` in a normalization, not a lowering that
  misplaced values, so the test asserts the gain property rather than widening a
  tolerance to 2e-2 and calling it agreement.

Vulkan's refusal to run MLA (`asymmetric v_head_dim not yet supported`) is
excused rather than silenced — it is the right behaviour, and the exclusion
drops out automatically if Vulkan gains the feature.

### rlx-unlimited-ocr — speculative decoding, and four bugs the suite could not see

Adding a draft-head seam to the shared decoder, plus the correctness work that
fell out of testing it. Every bug below was silent: the model kept emitting
fluent, in-range output.

- **Chunked verify pass.** `build_unlimited_ocr_decode_chunk_built` scores
  `n` tokens against a bucketed past in one forward, using `MaskKind::Bias`
  for the additive per-query mask that plain decode never exercises (one
  query only needs the cheaper binary keep-mask). `CompiledLm::decode_chunk`
  drives it; `DeviceKvCache::rollback_to` drops a rejected draft.
- **`decode_chunk` with `n == 1` fed the wrong mask.** The graph picks its
  mask *kind* from `seq`, so a single-query chunk compiles the `Custom`
  keep-mask path while `chunk_bias_mask` always built the `[1, heads, n, k]`
  bias — a leaf-shape mismatch (`host len 1028 != shape [1, 257]`). It only
  fires when a drafter proposes nothing, which the first adversarial test did.
- **`speculative.rs`: greedy speculation that is provably lossless.** A
  `Drafter` trait so the property is testable without any particular draft
  head, and `generate_speculative` accepts a draft token only when it equals
  the target's own greedy pick. `tests/speculative_equivalence.rs` runs four
  adversarial drafters (empty, constant, always-wrong, oracle) on every
  backend and requires byte-identical output.
- **That test was vacuous about the KV rollback.** Deleting
  `lm.rollback(...)` entirely — leaving every rejected draft token in the
  cache for later queries to attend to — changed no emitted token, because on
  a model this small the perturbation never moves an argmax. Fixed by
  asserting on cache state directly (`CompiledLm::layer_kv`): with the
  rollback removed the cache holds 47 rows where plain decode leaves 17.
  Both this and the accept condition are now sabotage-verified.
- **The KV cache row width is `num_kv_heads * head_dim`, not
  `hidden_size`.** `lm_device.rs` used `hidden_size` as the stride in all
  three paths. The two coincide for every config the suite had — jina-ocr-v1
  included, at 10 query heads and 10 K/V heads — so nothing covered GQA at
  all. New `UnlimitedOcrConfig::kv_hidden()` and
  `tests/gqa_cache_stride.rs`, whose oracle needs no reference: prefilling
  `n` tokens and prefilling `n-1` then decoding the last one must agree, and
  only the second route touches the cache.
- **The hidden tap published the post-norm state; the draft head wants
  pre-norm.** `DeepseekV2Model.forward` keeps `pre_norm_hidden_states =
  hidden_states` *before* `self.norm(...)` and hands that to FastMTP, which
  applies its own `hnorm`. Feeding the normed tensor is not a crash — the
  draft head still emits valid tokens, just uninformed ones, so the only
  symptom is a low acceptance rate. Prefill now also taps *every* position
  rather than the last, since a draft head that is itself a transformer layer
  has to be primed over the prompt.
- **And that test was covering one of four tap sites.** The tap is wired
  separately in each of {packed, raw} x {prefill, decode}; the old test only
  built the raw prefill graph, so reverting the packed branch — the one the
  real checkpoint runs — left it green. Now swept over both LM-head
  lowerings and both graph kinds, and each of the four sites individually
  sabotage-checked.

`rlx_models_core::backend_matrix` is the shared harness behind these:
`available_devices()` announces skips rather than hiding them, and `Failures`
accumulates so one run reports every failing (backend, variant) pair instead
of stopping at the first.

### Upstream `../rlx`: reshape stopped silently discarding elements

`Shape::reshape` accepted any target shape and truncated or zero-filled to fit.
Two real bugs were hiding behind that, both found the moment it became an
error:

- **`splat_common` summed a gradient with `keep_dim = false`**, producing
  `[count]` where `[count, 1]` was needed; the reshape quietly papered over the
  rank change.
- **`build_kv_compressor_pool` relied on the truncation** to trim a padded
  window, which now trims explicitly with `narrow_`.

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

Real weights are reachable without the download, and four tests do it.

A safetensors header gives every tensor's byte range, so `scripts/dsv41_ref/`
range-fetches only what a test touches — ~300 MB of the 510 GB:

- `real_layer_attention_matches_reference` runs the port's own `DsV41Loader` and
  attention over real fp8 bytes at 160 tokens (past `sliding_window`, so the
  window evicts), for layer 0 **and** layer 2. Layer 0 is sliding-window only;
  layer 2 is a KV *and* index source, so the compressor's gated pooling, the
  index keys and the YaRN-scaled compressed RoPE all run on trained weights.
  Agreement is **7.5e-7** relative.
- `dequant_matches_reference_on_real_bytes` decodes one real tensor per scale
  layout **exactly** — FP8 tiles, FP4 nibble pairs, and 64 rows of the
  384-million-row Engram table, which is FP8 but row-wise scaled. The Engram
  slice costs 17 KB of a 98 GB tensor.
- `engram_token_map_matches_the_real_tokenizer` runs the normalization over the
  real 129,280-token vocab and must land on exactly **99092** — the constant
  every hash multiplier is derived from — and fingerprints every individual
  merge, because two different normalizations can share a bucket count.
- `tensor_manifest_matches_the_real_checkpoint` needs no download at all: a
  15 KB inventory distilled from the 48 shard headers pins all **96,085**
  tensors against `DeepseekV41Spec::expected_tensors`, both directions, so a
  subsystem the port forgot shows up as an unexplained tensor rather than as
  silence. The only ones it knowingly skips are the DSpark stages'
  `gate.bias_vl` (drafts are text).

Each was mutation-checked. Swapping the FP4 nibble order, reading the Engram
table as tiled, mapping a compressed latent to position `j` instead of
`j · ratio`, dropping the inverse output RoPE, or dropping accent stripping are
all caught — and the compressed-latent one is caught by layer 2 while layer 0
still passes, which is the evidence that the second layer buys real coverage.

Two details worth knowing when reading the code. The reference round-trips
activations through FP8/FP4 in place (`act_quant(..., inplace=True)`); those
calls are precision simulation, not semantics, and the port computes the
F32-exact value. And `torch.topk`'s order among **equal** scores is unspecified
while the Indexer produces exact ties constantly (it rectifies its head scores,
so any position every head dislikes scores exactly zero) — the port keeps the
lowest index, matching `Op::TopK`, and the parity harness pins torch to the same
rule. A thresholding gate instead keeps *every* tied entry and quietly overruns
the `index_topk` budget.

### DeepSeek-V4.1: one block builder instead of three

Prefill, decode and the DSpark draft head were three ways of walking the same
block, and each carried its own copy of the Hyper-Connection wrapper — the part
of V4.1 that is easiest to get subtly wrong, because the pre-mix is *lagged*
(the mix a sublayer computes is consumed by the next one). Three copies of that
is three chances to restart the chain in the wrong place, and no test would
notice until the logits moved.

There is now one copy, in a new `dsv41_block` module: `hc_sublayer` takes the
body as a closure, and the three builders supply only what actually differs —
how they assemble keys. `dsv41_moe` moved out of the prefill module too, since
all three route through it; `dsv41_decode` used to import ten items from
`dsv41_graph`, which had the layering backwards.

Along with it:

- **`Ctx`** bundles the graph, its parameters, the packed side table, the
  checkpoint and the spec, which is what made the helpers take a handful of
  meaningful arguments instead of a dozen positional ones.
- **`AttnProj`** separates loading the attention weights from applying them.
  `WeightLoader::take` is destructive and DSpark needs two KV latents from the
  same `wkv` — one from the main model's stream, one from its draft block — so a
  load-and-apply helper could not serve both.
- **`StageSpan`** replaces `(layers, first, last)`; `..., true, false, ...` at a
  call site said nothing. It also checks the split: every `kv_source_layer` has
  to sit in the same stage as the layers reading it, or the consumers find an
  empty compressed cache.
- **`Tap`** replaces re-reading the environment at each of eight tap sites. It
  reads once, so a concurrently-running test cannot change the tap mid-build, and
  **a tap that never fires is an error** — previously, asking for a stage a layer
  does not have silently compared logits instead and read as a failure of that
  stage. That check immediately caught a second bug: the taps are process-global
  and `cargo test` is multi-threaded, so graph-building tests now hold a shared
  lock and set the tap through an RAII guard.

Behaviour is unchanged: all 14 parity tests still pass, on CPU, Metal and MLX,
including the real-weight runs.

### Known: CPU arena reuse miscompiles a real-scale V4.1 attention graph

Compiling and running the same graph with the same inputs on the same device
returns a different answer on roughly **one attempt in ten**, and the wrong
answer is badly wrong — a contiguous band of query rows, every column, max |Δ|
≈ 3.5 against activations averaging 0.5.

`RLX_ARENA_NO_REUSE=1` fixes it completely, so it is slot reuse rather than a
kernel. It reproduces single-threaded (not a race), with a fresh session per run
and with one reused session. `RLX_MEM_VERIFY=1` reports no overlap, no
read-after-death and no view-past-root — so the verifier's liveness model does
not describe whatever is going on, which is a finding in itself. Disabling
fusion or shared-input matmul only lowers the rate. `build_v4_sink_attention`
alone, at the same shape, is deterministic over hundreds of runs, so it needs the
surrounding layer.

It is captured as `arena_reuse_is_deterministic` (`#[ignore]`d, run with
`--ignored`) with `examples/dsv41_determinism_probe` as a standalone reproducer;
`real_layer_attention_matches_reference` pins the arena so that it measures the
port rather than re-discovering this. Until it is fixed, **any CPU number from a
real-scale graph is a coin flip unless the arena is pinned** — which also means
this is not specific to V4.1.

### Fixed: two backends broke DeepSeek-V4/V4.1 attention, upstream in `../rlx`

The V4.1 port had only ever run on CPU. Running the parity suite on Metal and
MLX found one bug each, both in shared code rather than anything V4.1-specific,
and both of a kind that CPU cannot see.

**Metal: a rank-2 RoPE input was read with heads and tokens transposed.** A
partial (tail) RoPE feeds `[tokens, heads · head_dim]` — tokens outermost, heads
striding *within* a row — which is what DeepSeek-V4 and V4.1 do to rotate the
last `rope_head_dim` dims of every head. `rlx-metal`'s forward RoPE derived
`(batch, seq, hidden) = (total / (s · head_dim), s, head_dim)` from that shape,
inventing a batch of `heads` and then indexing it as `(b · seq + s)`. It is a
no-op when `heads == 1`, which is why it survived; with more heads it silently
rotates the wrong elements. `RopeBackward` in the same file, and the CPU thunk,
already used the right derivation.

It hid well: the forward `q`/`k` errors largely cancel in `q·kᵀ` (RoPE's whole
point is relative position), so attention output was only ~0.4% off, while the
*inverse* rope on the attention output — which has no cancelling partner — was
91% off. **V4 is affected too**, and so is any model that rotates a multi-head
rank-2 tensor on Metal.

**MLX: `Op::TopK` broke ties by the largest index, not the smallest.** The op
documents "ties broken by smaller index", CPU and Metal implement it by repeated
argmax with a strict `>`, and `rlx-mlx` used `argpartition`, which picks an
arbitrary member of a tied group — the lowering's own comment conceded the point.
On an all-tied row it returned the highest indices. That is not a corner case
here: the Indexer rectifies its head scores, so any compressed position every
head dislikes scores *exactly* zero, and the tie rule alone decides which
positions the model attends to. It now selects the set arithmetically —
strictly-greater entries always win, and the group equal to the threshold is
filled lowest-index-first — then converts that to indices via a key with no ties
left, so no sort-stability assumption is involved. `rlx-mlx` also refused the
rank-2 multi-head shape outright (`Cannot reshape array of size 384 into shape
(12,8,2)`); its split/transpose path is rank-agnostic and was simply gated on
rank ≥ 3.

With both fixed, every stage of the V4.1 prefill agrees across CPU, Metal and
MLX to ~3e-7, and `prefill_matches_reference_on_all_backends` covers it. The
upstream suites stay green (`rlx-metal` 325, `rlx-mlx` 195).

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
- **Non-default features.** A 131-crate sweep under `macos-gpu`/`espeak`
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

### LuxTTS — wgpu unblocked (all 5 macOS backends)

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

- **`rlx-neuralhash` — NeuralHash perceptual image hashing.** 360×360 →
  128-float descriptor → `[96, 128]` seed projection → 96-bit hash, matching the
  [reference implementation](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX)'s
  `nnhash.py` output. The architecture is read natively from the vendor's
  container that installs in the framework — 225 layers,
  MobileNetV3-shaped with **instance** norm (spells it `batchnorm` with
  `training_instancenorm`), hard-swish written out as four elementwise ops, and
  squeeze-excite gates. Then it parses the container (LZFSE `pbze` decoded
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
