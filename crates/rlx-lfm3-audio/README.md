# rlx-lfm3-audio

LiquidAI **LFM3-Audio-2B** from its GGUF release
([`LiquidAI/LFM3-Audio-2B-GGUF`](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF)):
speech recognition, speech synthesis and spoken chat as rlx graphs.

```sh
just fetch-lfm3-audio                     # Q4_K_M LM + F16 mmproj, ~1.7 GB
just lfm3-audio asr  --wav assets/harness/jfk_16k.wav
just lfm3-audio tts  --text "Hello from Liquid." --voice male -o hello.wav
just lfm3-audio chat --wav question.wav --speak -o reply.wav
just features=metal lfm3-audio asr --wav clip.wav --device metal
```

`asr` takes `--language English` (omit it to let the model pick); `tts` takes a
language code from the GGUF's template (`en`, `en-gb`, `fr`, `ja`, …) and
`--voice female|male`; `chat --speak` answers with interleaved text and speech
(`<|mixed_start|>`), without `--speak` in text only. Speech sampling defaults
to the card's (temperature 1.0, top-k 128, repetition penalty 1.1 over 4096
frames); `--audio-temperature 0` is greedy.

From Rust:

```rust,ignore
let mut m = Lfm3Audio::open_dir(dir, Some("Q4_K_M"), Some("Q8_0"), Device::Metal)?;
m.set_low_memory(true);                       // optional: ~2 GiB peak
let tok = Tokenizer::new(m.lm_path());
let text = m.transcribe(&tok, &pcm_16k, Some("English"), 256)?;
let pcm_24k = m.speak(&tok, "Hello.", "en", "female", 0, 512)?;
```

`Lfm3Audio::generate` with the `prompt` builders covers spoken chat and the
sampling options; `set_timing(true)` prints compiles and steps with the
process footprint.

## What runs

| stage | tensors | graph |
|---|---|---|
| log-mel | — | host: 16 kHz, 25/10 ms Hann, 128 Slaney mels, `(log10 ∨ −12 + 4) / 4` |
| speech encoder | `a.conv1d.*`, `a.pre_encode.out`, `a.blk.*`, `a.post_ln` | 3 stride-2 convs (8× in time) → 17-layer causal LFM2, sliding window 32 |
| projector | `mm.a.mlp.*` | LayerNorm → Linear → GELU → Linear, one LM embedding per 80 ms |
| LM | `blk.*` (main GGUF, `lfm3`) | 24 layers: width-7 gated short conv / GQA attention (full, half-rotary θ 1e6; sliding 256, θ 1e4), per-head sigmoid output gate, sandwich norms |
| depth decoder | `a.gen.code.*` | 3-layer bidirectional LFM3 over the 12 codebook slots; codebooks revealed in 3 MaskGIT passes |
| frame feedback | `a.gen.code.out_embd` | a frame enters the LM as the sum of its 12 codebook rows |
| codec | `a.gen.wav.*` | FSQ codes → 8-layer LFM3 at 75 Hz (sliding 30) → log-magnitude + phase → iSTFT (host), 1920 samples / frame at 24 kHz |

All four stacks are one builder (`trunk.rs`). The GGUF stores partial RoPE as
llama.cpp frequency divisors (`rope_freqs.weight`: 1 for 32 frequencies, 1e30
for the other 32); the builder reads that back as "rotate the first half, keep
the zero-tail progression of the full head", as `Lfm3RotaryEmbedding` does.
The codec decoder's 30-step window has no metadata key and is fixed.

The builder is public for other LFM3 checkpoints: `graph::Gb` takes any
`rlx_core::weight_loader::WeightLoader` (the GGUF loader here, safetensors in
[`rlx-liquid-codec`](../rlx-liquid-codec/README.md)), `trunk::sequence` builds
a whole-sequence stack and `trunk::step` a KV-cached one, and `Istft` is the
codec's inverse STFT fed a piece at a time (`istft()` is the one-call form).

The LM advances against fixed-size caches (one compile per 256 slots): a
32-row graph for prompt chunks (masked matmul attention) and a 1-row graph for
decoding. Quantized projections stay packed (`DequantMatMul`), the tied LM head
included; the Q6_K token table stays packed on the host and decodes rows on
demand.

## Parity

`scripts/lfm3_audio_reference.py` is a float32 torch port of the released
modeling code (`modeling_lfm3*.py`, `modeling_liquid_codec.py`, the feature
extractor) over the bf16 safetensors checkpoint — the files themselves need an
unreleased transformers. It writes `tests/fixtures/reference.json` for the JFK
clip; `just test-lfm3-audio-parity [DEVICE]` checks each stage in pipeline
order against it on the F16 LM.

```sh
just fetch-lfm3-audio-hf    # only to regenerate the reference
python3 scripts/lfm3_audio_reference.py
```

Measured (JFK clip, F16 LM, CPU and Metal give the same figures):

| stage | vs reference |
|---|---|
| log-mel | max abs(Δ) < 1e-4 |
| encoder + projector | cosine 1.000000, max abs(Δ) 7e-7 |
| LM (text prompt) | hidden cosine 1.000000, top-5 ids identical, top logit Δ 5e-5 |
| greedy ASR | token-identical ("ask not what your country can do for you …") |
| depth decoder | max abs(Δlogit) 7e-5, argmax identical |
| greedy TTS | 18 / 18 frames identical |
| codec | max abs(Δ) 7e-9 |

On Q4_K_M the TTS output transcribes back to its input text exactly.

Checked independently with Whisper large-v3 (rlx-whisper, Metal): 8 English
sentences × both voices, seed 0, in two setups — F16 LM + F16 mmproj, and the
low-memory Q4_K_M + Q8_0 pair. Every one of the 32 clips was transcribed as
the text it was given; the only differences were Whisper spelling "river
bank" as "riverbank" (WER 2.4 % on plain word matching, 0 % with that
compound joined), identical across both setups and voices.

### Backends

`--features all-backends` (metal, mlx, cuda, rocm, gpu, vulkan); `coreml`,
`tpu` and `webgl` are forwarded too but opt-in. Parity test results:

| device | LM | result |
|---|---|---|
| CPU (Apple arm64) | F16 | every stage as above |
| CPU (x86-64 Linux) | Q8_0 | every stage; ASR token-identical |
| Metal | F16, Q8_0 | every stage as above |
| MLX | F16 | every stage as above |
| ROCm (integrated RDNA 3) | F16, Q8_0 | every stage as above |
| CUDA | Q8_0 | every stage; depth logits within 6e-3 (TF32) |
| Vulkan | Q8_0 | every stage |
| wgpu (Metal) | Q8_0 | every stage |
| CoreML (`--device ane`) | Q8_0 | every stage (CPU and CPU + Neural Engine units) |
| WebGL (`--device opengl`, planner on the host) | Q8_0 | every stage; 18 GB peak, 23 min — dense weights |
| TPU | — | builds; not run (needs a TPU host or a PJRT plugin) |

With a packed LM (Q8_0, Q4_K_M) the LM-side bounds are wider (top logit
within 0.5, hidden cosine > 0.999), and greedy TTS keeps its first frame:
quantization moves near-tied argmaxes, and every backend lands on the same
frames for the same file. The F16 LM is ~7 GB of f32 per graph, beyond wgpu's
4 GiB buffers; use a packed quant there.

Per-device adjustments: on CoreML and WebGL the tied LM head (128000 × 2048)
is scored on the host — CoreML will not bake a weight past 64M elements and
WebGL cannot hold it in a texture. WebGL takes no packed parameters, so its
graphs get every quant widened to f32. CoreML defaults `RLX_COREML_UNITS` to
`ane` (CPU + Neural Engine): its CPU + GPU default aborts inside Apple's MPS
compiler on the speech encoder (`XPC_ERROR_CONNECTION_INTERRUPTED`). The graphs
are fp32, so CoreML schedules them on its CPU; an f16 graph would be needed to
put work on the Neural Engine itself.

Dense weights default to f32. `RLX_WEIGHT_STORAGE=auto` keeps F16 tensors at
2 B/elem (lossless, half the memory; verified on CUDA and Vulkan) but is
opt-in: the x86 CPU f16-weight matmul (`rlx_cpu::blas::sgemm_half_rhs_widened`)
segfaults inside OpenBLAS, and ROCm's hipRTC fails to compile its
`cast_f32_to_half` kernel on an integrated RDNA 3 GPU.

## Speed

Apple M-series, Q4_K_M, Metal: ~16 ms per decode step (~60 tok/s), prompt
prefill ~6 ms per position in 32-row chunks, ~38 ms per speech frame (12.5
frames are one second of audio, so ~2× faster than real time). Each process
pays ~7 s compiling the two LM graphs plus a one-time first-run warm-up of a
few seconds. CPU runs every stage; no CPU timings were measured on an idle
machine, so none are given here. `--timing` prints compiles and per-step runs.

## Memory

Peak process footprint (`phys_footprint`, what iOS's limit counts), Metal,
Q4_K_M LM, JFK ASR / a 3.4 s TTS sentence:

| mmproj | mode | ASR | TTS |
|---|---|---|---|
| F16 | default | 4.06 GiB | 4.60 GiB |
| F16 | `--low-memory` | 2.14 GiB | 3.13 GiB |
| Q8_0 | `--low-memory` | **2.03 GiB** | **2.21 GiB** |

(The first version peaked at 6.6 / 7.0 GiB.) What it took:

- **One LM graph** (`--low-memory`, `Lfm3Audio::set_low_memory`, default on
  iOS): no 32-row prefill graph, so prompts are fed a position at a time
  (~16 ms each on Metal); the encoder and codec graphs are dropped once used.
  Each Metal graph keeps its own copy of the packed weights inside its arena,
  so the second graph was a second copy of the model.
- **The tied head in slices of 8192 rows**: Metal reserves a dequant scratch
  as large as the largest packed weight in f32; the 128000-row head made that
  ~1 GB per graph. Quant rows are contiguous, so each slice is a byte range
  of `token_embd` uploaded straight from the file mapping. The depth decoder's
  12288-row input projection is sliced the same way.
- **`mmproj-…-Q8_0`** (`--mmproj-prefer Q8_0`): the speech stacks stay packed
  instead of the F16 file widening to f32; the depth decoder's tied heads run
  per codebook over row ranges of the packed table. Encoder cosine 0.99998,
  greedy ASR unchanged, codec 0.99995; the Q4_K_M mmproj is not recommended
  (encoder cosine 0.96 on the last frame).
- **Host tables at their stored width** (quants packed, f16 as f16) and the
  GGUF tokenizer built from the header only — it used to read the whole
  checkpoint (`rlx_core::tokenizer`, ~1 GB).

`--timing` prints each graph's build / compile / upload with the footprint.

## iOS

`crates/rlx-lfm3-audio-ffi` is a C ABI (`include/rlx_lfm3_audio.h`) over this
crate, and `crates/rlx-lfm3-audio-ffi/ios` a SwiftUI app (XcodeGen) on it:
Load, Transcribe (bundled JFK clip), Speak. The weights go into the app's
Documents (Files app / Finder / `devicectl`). The C API, build settings and
signing are in [its README](../rlx-lfm3-audio-ffi/README.md).

```sh
just fetch-lfm3-audio Q4_K_M Q8_0       # the low-memory pair
just lfm3-audio-ios-sim                 # build, install, self-test on the booted simulator
just lfm3-audio-ios DEVICE TEAM         # the same on a paired, unlocked device
```

`--selftest` runs ASR on the JFK clip, TTS, and the TTS transcribed back,
printing timings and footprint. On the iOS simulator (Metal, Q4_K_M +
Q8_0 mmproj, low memory) the JFK transcript is correct, the spoken sentence
transcribes back word for word and the footprint stays ≤ 1.7 GiB; timings
vary from run to run on the simulator (ASR 3–9 s, TTS 2.5–7 s for ~2 s of
speech). Not yet run on a device.
