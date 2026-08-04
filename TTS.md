# RLX TTS models & voice chat

Index of the text-to-speech models ported to RLX, their reported throughput, and
the **Gemma 3 270M + Inflect-Nano** local voice-chat pairing built on top of them.

> **Unified harness:** [`crates/rlx-tts-bench`](crates/rlx-tts-bench/README.md)
> (`just tts-bench-apple run …`) produces `results.jsonl`, `report.html`, and
> `BACKENDS.md` (model × device RTF / cosine / Whisper). Cross-backend notes and
> historical matrices are in the [Cross-backend parity & benchmarks](#cross-backend-parity-benchmarks--matrices)
> section below.

> **Backend status (2026-08, NVIDIA CUDA):** a cross-backend matrix (RTX 3080 Ti)
> caught a **silent-audio regression** on the split enc-on-CPU + dec-on-CUDA models
> (piper / kokoro / styletts2 / zipvoice) — root-caused to `rlx-cuda`
> `clone_for_cache` dropping `set_param` weights on `compile_named().clone()` and
> **fixed** (D2D param copy). **piper CUDA is bit-exact (cos 1.000)**; the other
> three are un-silenced with a residual sine-source parity drift still open.
> Also landed: fused-kernel activation completeness (ops 12–28). Vulkan is the
> strongest GPU path (all TTS 0.975–1.000); Apple backends (Metal/MLX/CoreML/wgpu)
> remain bit-exact where noted. Full detail + repro in the
> [Cross-backend parity & benchmarks](#cross-backend-parity-benchmarks--matrices)
> section below and `PARITY.md`.

## Models

| model | crate | ~params | best reported RTF | backend | notes |
|-------|-------|---------|-------------------|---------|-------|
| 🥇 **Inflect-Nano** | `rlx-inflect-nano` | ~4.6M | **48×** (reported) / ~10–13× *(verified here, Metal)* | MLX / Metal | FastSpeech-style acoustic + Snake HiFi-GAN vocoder, 24 kHz; fastest by far |
| **Pocket-TTS** | `rlx-pocket-tts` | ~100M | "faster than realtime" | CPU (Accelerate) | Kyutai FlowLM + Mimi codec |
| **Orpheus** | `rlx-orpheus` | 3B | ~22× warm (decode-bucket reuse) | Metal (native decode) | LLM-style; only whisper-OK Metal config is documented (see notes) |
| **Qwen3-TTS** | `rlx-qwen3-tts` | 0.6B | 1.05–1.7× | Metal | best speed/quality of the larger models |
| **Kitten-TTS** | `rlx-kittentts` / `kitten_tts_mini_rlx` | ~15M | "faster than ONNX on CPU" | CPU | tiny |
| **Tiny-TTS (VITS2/MeloTTS)** | `rlx-tiny-tts` | — | — | CPU/MLX/wgpu/CoreML | bit-exact across all 4 backends |
| **NeuTTS** | `rlx-neutts` | Nano/Air | — | — | no reported RTF |
| **Voxtral-TTS** | `rlx-voxtral-tts` | 4B | — | — | no reported RTF |
| **Kyutai-TTS** | `rlx-kyutai-tts` | 1.6B | — | — | see crate README |
| **Gepard** | `rlx-gepard` | ~556M | — | Metal→MLX AR + NanoCodec | fox 6/6; see cross-backend section below |
| **ChatterBox / Supertonic / LuxTTS / Piper / …** | see below | — | — | multi-backend | unified matrix via `rlx-tts-bench`; cross-backend section below |

## Leaderboard

| metric | winner | value |
|--------|--------|-------|
| ⚡ **Fastest** | **Inflect-Nano @ MLX** | **~48× realtime** (reported); ~10–13× on Metal *(verified here)* |
| 🪶 **Smallest** | **Inflect-Nano** | ~4.6M params |
| 🗣️ **Fastest at "large-model" quality** | **Qwen3-TTS @ Metal** | ~1.05–1.7× realtime (0.6B, production-wired) |

Inflect-Nano wins on raw speed and size (it's tiny); the larger models
(Qwen3-TTS 0.6B, Orpheus 3B, Voxtral-TTS 4B) trade speed for naturalness.

---

## Voice chat: Gemma 3 270M → Inflect-Nano (`rlx-gemma-inflect-nano`)

A fully-local, streaming **voice chat**: you type, `gemma3-270m` generates a reply
on the GPU, and `inflect-nano` speaks it back — the LLM and TTS both run on Metal.

- Crate + full docs: [`crates/rlx-gemma-inflect-nano/README.md`](crates/rlx-gemma-inflect-nano/README.md)
- Interactive REPL example: [`crates/rlx-gemma-inflect-nano/examples/chat.rs`](crates/rlx-gemma-inflect-nano/examples/chat.rs)
- One-shot (prompt → WAV) example: `examples/speak.rs`

### Run

```bash
cd /Users/Shared/rlx-models
cargo run --release --features metal -p rlx-gemma-inflect-nano --example chat -- \
  --device metal --tts-device metal
# then type; /reset clears history, /quit exits.
```

Defaults resolve `RLX_GEMMA3_GGUF` (else `/tmp/rlx-weights/gemma-3-270m.gguf`) and
`RLX_INFLECT_NANO_DATA` (else `weights/inflect-nano-rlx`). See the crate README for
all flags (`--speed`, `--sentence-pause`, `--prime-secs`, `--first-sentence`,
`--no-audio`, `--temp`, `--system`, …).

### Design

- **Pipelined**: sentences are split as the LLM streams them (on `.`/`!`/`?`/newline)
  and each is vocoded + queued for playback *immediately* — playback starts once
  ~4s is buffered and overlaps ongoing generation. You hear sentence 1 while Gemma
  writes sentence 2.
- **Pacing** (1.5× slower default) is applied at **synthesis** (Inflect `InferOpts::with_speed`,
  natural pitch), not at playback; the audio path only resamples 24 kHz → device rate.
- Live playback via `cpal` (streaming ring buffer, adapted from `rlx-moshi`), with a
  macOS `afplay` fallback when no output device is available.

### Verified performance (Metal, release, this session)

| stage | number | how |
|-------|--------|-----|
| gemma3-270m decode (warm) | **~32 tok/s**, bit-exact vs CPU (logit_cos 1.0) | `rlx-gemma` `examples/gemma_bench.rs` |
| gemma3-270m decode (chat, short reply) | ~9–17 tok/s (incl. per-turn prefill + incremental decode) | `chat --no-audio` |
| Inflect-Nano TTS (Metal, cached graph) | ~10–13× realtime | `chat` |
| first reply of a session | one-time prefill graph compile (a few seconds) | — |

### Engineering findings / fixes (this session)

1. **Metal multi-turn prefill NaN → garbage (FIXED).** Prompts past the first
   prefill bucket (≳30 tokens of history) padded to a power-of-two bucket, which set
   prefill **active-extent**, forcing Metal off the validated MPSGraph-hybrid path
   onto the per-op MSL thunk path — where a Gemma 3 Q4 `Op::Attention`→`o_proj`
   dequant **arena-aliasing** defect (task #50) zeroes attention output → all-NaN
   logits → silent CPU fallback → garbage. Short prompts (exact bucket) never hit it,
   so the 21-token parity test missed it. **Fix:** drop `Device::Metal` from
   `packed_prefill_active_extent_enabled` (`crates/rlx-models-core/src/autoregressive.rs`)
   so Metal prefill stays on the good path; keep pow2 bucketing in
   `crates/rlx-gemma/src/packed_session.rs::prefill_bucket_len_device` for graph reuse.
   Correct because logits are read from `last_token_idx = n−1` and KV is truncated to `n`.
2. **`*_auto` tokenizer reload gotcha (FIXED, ~5–7× chat speedup).** The `chat`
   streaming callback called `rlx_gemma::decode_token_auto` **per generated token**,
   and the `*_auto` helpers (`decode_ids_auto`/`encode_prompt_auto`) **reload
   `tokenizer.json` from disk on every call** (~370 ms/token). That — not the model —
   was the apparent "~2 tok/s"; real warmed decode is ~32 tok/s. **Fix:** load
   `tokenizers::Tokenizer::from_file` once and decode incrementally.
3. **Vocoder graph cache (~3× TTS throughput on streamed replies).** New
   `InflectNano::synthesize_on_cached` (`crates/rlx-inflect-nano/src/lib.rs`) buckets
   frame counts (mel padded to a multiple of 64, waveform trimmed) and reuses compiled
   vocoder graphs across sentences/turns instead of recompiling per length
   (~3.2× → ~10× realtime). `synthesize_on` stays bit-exact for existing callers.
4. **Pipelined streaming + 4 s prime buffer** to keep playback smooth (replaces
   the earlier per-comma micro-segmentation that caused choppiness).

### References

- Memory notes: `gemma3_270m_metal_prefill_nan.md`, `rlx_tokenizer_auto_reload_gotcha.md`,
  `inflect_nano_crate.md`, `orpheus_tts_perf.md`, `tiny_tts_backends.md`.
- Related backend gotchas: `gemma4_metal_q4_bugs.md`, `gemma_cuda_attention_scale.md` (task #50 family).
- Bench tool: `cargo run --release -p rlx-gemma --features metal --example gemma_bench`
  (per-backend prefill/decode timing + CPU parity).

---

## Cross-backend parity, benchmarks & matrices

*(Merged from the former `TTS_BACKENDS.md`.)* Apple measurements on this
Apple-Silicon Mac; NVIDIA CUDA/Vulkan via the remote host (`RLX_CUDA_HOST` / the
`scripts/matrix` harness). Harness: `crates/rlx-<model>/examples/backend_matrix.rs`
(loops `Device` variants via `load_on`, median of 3 timed iters after 1 warmup;
parity = cosine of the waveform vs the CPU output; whisper = word coverage of the
reference sentence via `.cache/whisper-tiny` / `whisper-base.en`). "Original" =
the model's own **ONNX Runtime** path (`synthesize_ort`, CPU EP) run in-process.

### Verified local bench (2026-08, Apple cpu/metal/mlx) — `rlx-tts-bench`

Measured live via `rlx-tts-bench run --models all --devices cpu,metal,mlx
--phrases short --whisper --spectral --noise --clone --iters 1` on this Mac.
All 23 adapters now have weights (orpheus/qwen3-tts/kittentts/kyutai/pocket-tts/
voxtral-tts downloaded + voice/tokenizer data prepared this session). RTF = audio÷wall (>1 = faster than
real-time). "GPU cos vs cpu" = **min** waveform cosine across metal/mlx (precision
integrity). Whisper = word coverage of "The quick brown fox …" via whisper-base.en.

| model | RTF cpu | RTF metal | RTF mlx | GPU cos vs cpu | Whisper cov (fox) |
|---|---|---|---|---|---|
| piper | 10.56× | 3.95× | 7.94× | 1.000 | 0.77 (4/6) |
| rlx-tts | 6.01× | (cpu-only) | (cpu-only) | — | 1.00 (6/6) |
| supertonic | 2.07× | 1.10× | 1.79× | 1.000 | 1.00 (6/6) |
| melotts | 3.30× | 1.06× | 3.63× | 1.000 | 0.92 (5/6) |
| styletts2 (kokoro) | 1.62× | 0.84× | 1.95× | **0.9925 (FIXED, mlx fastest)** | 1.00 (6/6 all) |
| soprano | 0.49× | 1.25× | 1.04× | 1.000 | 1.00 (6/6) |
| moss-nano | 0.25× | 0.24× | 0.67× | 1.000 | 0.92 (6/6) |
| luxtts | 0.23× | 0.16× | 0.45× | **0.006 (metal broken)** | 0.69 (4/6) |
| zonos | 0.08× | 0.12× | 0.14× | 1.000 | 0.85 (6/6) |
| f5tts | 0.13× | 0.12× | 0.40× | 0.567 (waveform; stft 0.98) | 1.00 (6/6) |
| metavoice | 0.05× | 0.03× | 0.05× | 1.000 | 0.31 (1/6) |
| gepard | 0.25× | ~0.4× | ~0.4× | 1.000 | **1.00 (6/6, greedy)** |
| parlertts | timeout | timeout | 0.05× | — | 1.00 (6/6 mlx) |
| chatterbox | timeout | timeout | 0.09× | — | 1.00 (6/6 mlx) |
| miotts | timeout (>180 s) | timeout | timeout | — | — |
| miratts | fail (f32 gen n/a) | — | — | — | — |
| sesame | fail (mimi weights missing) | — | — | — | — |

**Reading it:**
- **Fastest (RTF > 1):** piper (10.6× cpu), rlx-tts (6×), melotts (3.3×), supertonic (2.1×), styletts2 (1.6× cpu / 3.7× mlx). Everything below moss-nano is **slower than real-time** on this Mac (large AR/diffusion models).
- **Precision (cos vs cpu = 1.000)** holds for piper/supertonic/melotts/soprano/moss-nano/zonos/metavoice. **styletts2/mlx now FIXED** (0.9925 — the grouped-ConvTranspose2d fix above). **Still-open GPU cells:** luxtts **metal** (cos ≈0 at long frames — root: the onnx-imported **fm_decoder has a malformed matmul at large num_frames** (the same op that panics on CUDA "unsupported shapes"; Metal silently garbles). Short phrases are fine (cos 0.99); it's **num_frames-dependent, driven by the clone-prompt length** — so the adapter's char-based chunking can't fix it (a 52-char chunk with a long clone ref still overruns). Needs the fm_decoder matmul shape fixed in the graph import). f5tts GPU (0.567 waveform / 0.98 spectral — DiT phase drift, still intelligible).
- **Whisper (intelligibility):** perfect 6/6 for rlx-tts, supertonic, styletts2 (all 3 backends now), soprano, f5tts, parlertts, chatterbox, qwen3-tts; strong melotts/moss-nano/zonos (0.85–0.92); weaker piper 0.77, luxtts 0.69; **gepard 6/6 (greedy — FIXED)**; **poor kyutai 0/6 & metavoice 0.31** (kyutai=silence bug; metavoice=bench max-token cap truncation, greedy already).
- **Not benchable / failed:** kyutai runs and produces **audible** speech (peak 0.95) but the words are **unrelated to the transcript** (whisper "I'm going to make a video about this." for the fox prompt) — the Moshi multi-stream text-forcing injects tokens (`RLX_KYUTAI_TTS_TRACE` shows word + multiplexed 2-speaker ids) but the LM doesn't follow them; greedy doesn't help. Deep/incomplete-port conditioning bug, not silence or sampling. voxtral-tts (4B) OOMs on load after its voice-embedding data fix; kittentts/mlx panics (Reshape 200100-vs-200000 length inconsistency in the code-gen graph, off-by-one frame); parlertts/MLX **FIXED** (was a stale AOT cache-key collision, not shape strictness — key now folds pt/et).

Full per-cell JSON + HTML: `/tmp/tts-bench/{results.json,report.html}`.

### 2026-08 "fix broken ones" pass — bugs found & fixed

Full 23-model deterministic run (all downloads present) + targeted follow-ups. Key
results this session (verified via `rlx-tts-bench` + Whisper):

| model / cell | was | now | fix |
|---|---|---|---|
| **styletts2 / mlx** | cos −0.008, fox **0/6** (garbage) | **cos 0.9925, fox 6/6, rtf 1.95× (fastest)** | **real rlx-mlx bug:** native grouped/depthwise `ConvTranspose2d` mixed channels across groups (ISTFTNet upsampler g=512 → output ~25× too large). Host-eval `groups>1` (mirrors CT3d). +regression test. |
| **chatterbox / metal** | cos 0.028 flagged as "Metal bug" | **not a bug** — correct speech (whisper 6/6, logmel 0.97) | single near-tie greedy-argmax flip (rep-penalty tie + ~1e-4 GPU noise); bench now decodes **deterministically** (`SynthRequest.deterministic`) so waveform-cosine parity is meaningful for AR-TTS. |
| **voxtral-tts** | FAIL "no .f32 embedding" | data-unblocked (4B then OOMs on load — separate) | converted 20 voice `.pt`→`.f32` (`--convert-voices`). |
| **qwen3-tts** | FAIL "missing tokenizer.json" | **cpu fox 6/6, rtf 0.25×** | built `tokenizer.json` from `vocab.json`+`merges.txt` (Base repo ships none). |
| **kyutai** | FAIL "missing voice embedding" | runs (data-unblocked) but **fox 0/6** | downloaded voice from `kyutai/tts-voices`; output quality is a separate open bug. |
| **gepard** | metal/mlx crash + fox **0/6** (babble) | **cpu 6/6, metal 6/6** | two fixes: (1) crash — qwen35 `clear_host_dense_projections` skips release when `weights_path` empty (inline_weights; helps any inline-weights qwen35 on Apple GPU); (2) **0/6 was sampling free-running into coherent-but-wrong words** — greedy is faithful (whisper "The quick brown fox…"), adapter now honors `deterministic`→greedy. Same class as chatterbox. |
| **parlertts / MLX** (+cpu greedy) | crashed on MLX for any transcript ≠19 tokens (`from_bytes` nbytes mismatch); cpu greedy errored "seq 439≠433" | **runs on all backends for any text** | two fixes: (1) cpu greedy — derive prompt-prefix from returned `seq` not `pt`; (2) **the MLX crash was a stale AOT cache-key collision** — `$TMPDIR/rlx_parler_native` keyed decoder graphs by `seq` only (not `pt`/`et`), so a graph compiled for one transcript length reloaded for another → baked `prompt_input_ids [1,19]` → MLX strict panic (CPU tolerated). Fix: fold all named dims into the cache key. NB parlertts *greedy* is degenerate (collapses to "the"); bench uses sampling (fox 6/6), GPU cosine gap is benign AR divergence. |

Waveform-cosine is the **wrong parity metric for AR/LM TTS** (chatterbox, orpheus,
sesame, parlertts…): one flipped/sampled token desyncs phase for the rest of the
utterance while the words stay correct. Judge those by greedy token-prefix + Whisper
coverage + log-mel cosine, not raw waveform cosine.

### 2026-07 parity pass (Whisper-gated)

| Model | Issue | Fix | Gate |
|-------|-------|-----|------|
| ChatterBox | Fox 0/6 false negative | `rlx-whisper` skips lang/task tokens on `.en` checkpoints | Apple Whisper 6/6 |
| StyleTTS2 / Kokoro native | Decoder garbage (historical) | Remove Cin↔Cout transpose on ONNX `ConvTranspose` (`rlx-onnx-import`). Native default: ORT CPU encoder + RLX decoder (or `RLX_KOKORO_NATIVE_ENC=1`); fox **6/6** CPU/Metal/MLX/wgpu | fox 6/6 Apple |
| Soprano | Harness + MLX peak | Default greedy; coverage `len≥2`; Vocos on CPU when backbone is MLX | CPU/Metal/MLX Whisper 1.00 |
| Piper | Stochastic ORT / GPU enc | Native path; CPU-pinned `enc_p`; `RLX_PIPER_DETERMINISTIC=1` | Apple cos 1.0; NVIDIA CUDA cos≈1.0 |
| Gepard | Bucketed decode mask/KV slice drift | Fix `slice_kv_from_bucket` + dense `generated` for custom-mask AR; bucketed default | Apple fox 6/6 + long 15/15 |
| MeloTTS / tiny-tts | 512-sample silence | Same CT fix + `ensure_audible` | Local CPU/wgpu/Metal cos 1.0 |
| MetaVoice | Whisper ~67% | Greedy defaults, speaker required, PCM postprocess | Fox 6/6 Apple |
| F5-TTS | DiT GPU mismatch | Metal/MLX on-device fox 6/6; ScatterNd pin over MPS cliff. True wgpu (`RLX_F5_WGPU_DIT=1`): NFE=32 fox **6/6** after (1) unsharded `>bind` arenas → dedicated scratch/virtual stripes (`RLX_WGPU_LARGE_BUFFERS=1`); (2) `Transpose(Param)` binds the act-output stripe + stages the weight | fox 6/6 Metal + true wgpu |
| Supertonic | Vulkan cos≈0.03 | Host non-last Reduce; true Vulkan cos 1.0 | NVIDIA cos 1.0 |

### 2026-08 CUDA silent-audio regression FIXED (piper/kokoro/styletts2/zipvoice)

A cross-backend matrix run (msi RTX 3080 Ti) found piper/kokoro/styletts2/zipvoice
**silent (peak=0.00)** on CUDA — a regression from the 2026-07 CUDA-green state.

**Root cause (rlx-cuda):** `CudaExecutable::clone_for_cache()` — reached via
`CompiledGraph::clone` → `clone_box` — recompiles into a fresh zeroed arena and
re-bakes Constants but **never copies the `set_param`-uploaded `Op::Param`
weights.** So every `TinyModel::compile_named().clone()` consumer (the split
enc-on-CPU + dec-on-CUDA models: piper `flow_dec`, kokoro/styletts2 decoder,
zipvoice) ran the clone with **all weights zero** → all-zero graph → silence.
Models driving the graph **in place** via `run_named` (supertonic/luxtts/F5 ODE
loops) never clone — unaffected, as were single-graph melotts/tiny-tts.

**Fix:** `clone_for_cache` calls `copy_params_from(self)` — D2D-copies every param
slot (byte_size/4 f32; covers packed U8/I8) from the source arena into the clone
(`crates/backends/rlx-cuda/src/backend/mod.rs`). Diagnostics: `RLX_CUDA_INPUT_DIAG`
(input/param upload + silent skips), `RLX_CUDA_DUMP_IO=1` (Input/Param/Constant
slots), `RLX_DUMP_KERNELS=<dir>` (exact NVRTC source per kernel).

| model | cuda before | cuda after fix |
|---|---|---|
| piper | silent (0.00) | **PASS cos 1.000** (bit-exact) |
| kokoro | silent | cos 0.263 (un-silenced) |
| zipvoice | silent | cos 0.469 (un-silenced) |
| styletts2 | silent | cos 0.263 (un-silenced) |

**Also landed — fused-kernel activation completeness:** `fused_binary_unary.cu`
(relu-first, no `default`) and `batch_elementwise_region.cu` /
`elementwise_region.cu` (gelu-first) implemented only activation opcodes **0–11**,
so a fused `Sin`/`Cos`/`Round`/… fell through to **identity**. Added cases 12–28
(exprs copied from the codegen'd `unary.cu`). Benefits any model fusing `Mul→Sin`.
*Build gotcha:* those `.cu` are `include_str!`'d and `cargo clean -p
rlx-gpu-kernels` does **not** force a rebuild — use
`rm -rf target/release/build/rlx-gpu-kernels-* .fingerprint/rlx-gpu-kernels-*`.

**Residual drift — localized, open.** kokoro/styletts2 (0.263, shared StyleTTS2
decoder) + zipvoice (0.469) trace to the sine source: *some* standalone
`Activation(Sin)` outputs read identity though `unary.cu case 13u=sinf`,
`Step::Unary(op=13)`, and the launch args are all verified correct. Leading
hypothesis: an arena slot-offset mismatch. See memory `cuda_silent_split_graph_input`.

### Supertonic-3 (CFM, 4 subgraphs) — "The quick brown fox …", 4.30 s @ 44.1 kHz

**rlx native, cross-backend**

| backend | RTF | median ms | cosine vs CPU | whisper |
|---|---|---|---|---|
| CPU | 1.0× | 4346 | 1.00000 | 1.00 |
| Metal | 1.8× | 2359 | 1.00000 | 1.00 |
| MLX | 13.5× | 318 | 1.00000 | 1.00 |
| wgpu | 1.5× | 2856 | 1.00000 | 1.00 |
| CoreML | 1.3× | 3399 | 1.00000 | 1.00 |
| **CUDA** | **2.4×** | **1782** | 0.96463 | 1.00 |

Perfect on Apple backends (bit-identical, cos 1.00000); CUDA matches whisper
coverage at cos ≈0.965 (TF32 / kernel order).

**vs onnxruntime (original, CPU EP) — 848 ms (5.1× RT):** rlx beats ort-CPU **only
on MLX** (318 ms, 2.67× faster). rlx CPU is ~5× *slower* (4346 ms); Metal/wgpu/CoreML
also trail ort-CPU. rlx's wins are being native + portable + bit-exact, not raw speed.

### LuxTTS (ZipVoice-distill CFM, 3 subgraphs, voice cloning) — 2.39 s @ 24 kHz

| backend | RTF | median ms | cosine vs CPU | whisper |
|---|---|---|---|---|
| CPU | 0.3× | 7556 | 1.00000 | 0.85 |
| Metal | 0.4× | 6573 | 0.99992 | 0.85 |
| MLX | 1.7× | 1383 | 1.00000 | 0.85 |
| wgpu | 0.3× | 7423 | 1.00000 | 0.85 |
| **CUDA** | 1.4× | 1664 | 0.99979 | 0.85 |

Parity holds (whisper 0.85 = luxtts's known espeak-vowel coverage). **CoreML** runs
all three subgraphs on `Device::Ane` only with `RLX_COREML_UNITS=gpu` (auto-set by
TinyModel; default Neural-Engine units SIGSEGV in BNNS). vs ort-CPU (772 ms), rlx
loses on every backend (MLX 1383 ms = 0.56×, CPU 7556 ms = 0.10×).

### rlx-cpu `BinaryFull` optimization (profiler-driven)

`RLX_PROFILE_THUNKS` showed `BinaryFull` = 88% of a supertonic subgraph (single-
threaded, per-element modulo + per-call alloc). Fix: fuse index math, hoist the
op-match, rayon (f32+i64), bit-exact → Supertonic CPU 4346→**2756 ms (1.58×)**,
cos 1.00000 unchanged. Gap vs ort-CPU narrowed 5×→3.2×.

### Takeaways

- **Native-backend parity is real** — cos ≈ 1.0 + stable whisper on CPU/Metal/MLX/wgpu.
  CoreML (`Device::Ane`) pins `RLX_COREML_UNITS=gpu` (Neural-Engine BNNS SIGSEGVs on
  large MIL). **Vulkan (2026-07-19):** piper/sesame/kokoro/styletts2/supertonic/
  melotts/moss true-Vulkan green after Cast-copy + Reduce-host + arena/barrier fixes.
- **rlx is generally slower than ONNX Runtime** — only supertonic@MLX beat ort-CPU;
  rlx's CPU path is 5–10× behind. The wins are native/portable/bit-exact, not speed.

### 2026-07-18/19 NVIDIA CUDA + Vulkan + Mac CoreML matrix

**CUDA green:** melotts/tiny-tts ≈1.000 · piper ≈1.000 (`RLX_PIPER_DETERMINISTIC=1`) ·
kokoro/styletts2 ≈0.998 · supertonic ≈0.994 · sesame 1.000 · gepard 1.000 (AR on CPU) ·
f5tts/chatterbox PASS (audible). *(Note: piper/kokoro/styletts2 later regressed to
silent — see the 2026-08 fix above.)* CUDA fixes 2026-07-18: luxtts (empty-tensor
arena slots + skip zero-size Expand), moss-nano (bucketed prefill).

**Vulkan (NVIDIA) true-Vulkan pass — root causes fixed in `rlx-vulkan`:** (1) arena
> `maxStorageBufferRange` (~4 GiB) all-zero bindings → `plan_memory_f32_uniform`
liveness reuse + param/act split + activation striping; (2) barrier elision
(range-based RAW/WAW/WAR); (3) bool→f32 Cast no-op left F32 slot zeroed → identity
copy when src≠dst; (4) non-last/multi-axis Reduce silently wrong → host every Reduce;
(5) host-fallback I64; (6) versioned `RLXLIR01` AOT header. Result: piper/sesame
cos 1.0, kokoro/styletts2 ≈0.995, supertonic 1.0, melotts/tiny-tts 1.0, moss-nano 1.0,
f5tts full striped DiT, gepard NanoCodec on-device (eager AR on CPU), chatterbox
after AOT schema bump.

**CoreML (Mac):** melotts/tiny-tts/piper/moss-nano 1.000 · kokoro/styletts2 ≈0.994 ·
f5tts ≈0.866 (pass @ 0.85) · supertonic flaky (MIL parse) · chatterbox omitted
(MIL KernelChannels mismatch).
