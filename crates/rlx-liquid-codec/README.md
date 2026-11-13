# rlx-liquid-codec

[LiquidAI LiquidCodec](https://huggingface.co/LiquidAI/Liquid-Codec): 24 kHz
mono speech ↔ 12 tokens per 80 ms frame (12.5 Hz, codebooks of 2016,
≈1.65 kbps). Encoder and decoder are rlx graphs and run on every backend.
Both can be streamed: codes and samples match the whole-clip calls. A
10-minute clip encodes in under 1 GB.

```bash
just fetch-liquid-codec                       # weights/audio/liquid-codec, 147 MB bf16
just liquid-codec roundtrip --wav clip.wav -o out.wav
just liquid-codec encode --wav clip.wav -o codes.json [--stream]
just liquid-codec decode --codes codes.json -o out.wav [--stream]
just liquid-codec roundtrip --wav clip.wav -o out.wav --weight-storage auto   # bf16 graph weights
just features=metal liquid-codec roundtrip --wav clip.wav -o out.wav --device metal
```

`--weights` also takes an LFM3-Audio checkpoint directory (it reads that
directory's `audio_tokenizer/`), which holds the same weights under the older
names. `codes.json` is `{"audio_codes": [[…] × 12]}`, codebook-major, the
layout of `LiquidCodecModel.encode(...).audio_codes[0]`.

```rust
use rlx_liquid_codec::{AudioCodec, LiquidCodec};
let codec = LiquidCodec::open("weights/audio/liquid-codec".as_ref(), rlx_runtime::Device::Cpu)?;
let codes = codec.encode(&pcm_24k)?;      // [frames][12]
let audio = codec.decode(&codes)?;        // 1920 samples per frame
let mut e = codec.encode_stream();        // samples in as they arrive
let mut live = e.push(&pcm_24k[..4000])?; // frames completed so far
live.extend(e.finish()?);                 // == codec.encode of the same samples
let mut d = codec.decode_stream();        // frames in, samples out
let mut out = d.push(&codes[..1])?;
out.extend(d.finish());
```

It also implements `rlx_core::AudioCodec`. Every frame needs all 12 groups
(grouped FSQ, not RVQ), so `encode_pcm(_, Some(n))` rejects any `n` other
than 12.

## Architecture

| stage | where | what |
|---|---|---|
| STFT | host (`stft.rs`) | reflect-pad 480, `n_fft` 1280, hop 320, periodic Hann; per bin `log1p\|S\|`, `Re S/\|S\|`, `Im S/\|S\|` → 1923 features at 75 Hz |
| front end | graph | 1×1 conv → SiLU → causal width-3 conv → SiLU → kernel = stride 6 conv → 512 at 12.5 Hz |
| encoder | graph | 8-layer causal LFM3 (gated short conv width 7; full attention with per-head sigmoid gate on layers 2/4/6; sandwich norms; RoPE θ 1e6) |
| to quantizer | graph | 512 → 256 → 48 (`project_in`) |
| GFSQ | host (`fsq.rs`) | 12 groups × 4 dims, levels `[8, 7, 6, 6]`; `tanh` bound, round-half-even, mixed radix |
| decoder | graph | 48 → 256 → 512, each frame repeated 6×, 8-layer causal LFM3 with a 30-step sliding window, linear to 1282 = log-magnitude + phase |
| iSTFT | host (`rlx_lfm3_audio::Istft`) | `irfft` × Hann, overlap-add, ÷ Σ window², trim 480 at both stream ends |

The LFM3 stacks come from `rlx_lfm3_audio::trunk`, the same builder that
runs this decoder from LFM3-Audio's GGUF mmproj. `weights.rs` maps the
safetensors names onto it. Every stage is causal, so a graph built for a
padded bucket gives the same rows for the real frames.

- **Whole-clip decode** (`decode`): a decoder row reads at most 117 STFT
  steps back (3 attention layers × 29 plus 5 conv layers × 6), so clips
  longer than 250 frames decode in 250-frame pieces with 20 frames of
  history.
- **Streaming decode** (`DecodeStream`): one frame (6 STFT steps) per graph
  run, through `trunk::step`. Attention keys and values sit in a 36-slot
  ring (the 29 past steps the window can see, plus the frame's own 6). The
  short-convolution states carry over between runs, as in
  `LiquidCodecStreamingCache`.
- **Whole-clip encode** (`encode`): one graph up to 375 frames (30 s;
  `set_max_whole_frames` / `--max-whole-frames`). Longer clips run in
  64-frame chunks against a growing KV cache: the encoder attends causally,
  so this is the same computation. Features are also computed a chunk at a
  time.
- **Streaming encode** (`EncodeStream`): an STFT frame is computed once its
  window's samples are in, and a codec frame is encoded (one row of the step
  graph, carrying the front end's convolution history and the KV cache) once
  its 6 STFT frames exist. That is 480 samples (20 ms) after the frame ends.
  Only the last frame sees the zero padding and right-edge reflection, which
  `finish` applies, so the codes equal `encode` of the same samples. The
  reference has no streaming encoder; this one follows from the encoder
  being causal.

### Memory

Peak process footprint, CPU, measured with `/usr/bin/time -l`:

| | before | now (f32 graph weights) | `--weight-storage auto` |
|---|---|---|---|
| encode 20 s | 0.80 GB | 0.50 GB | 0.36 GB |
| encode 10 min | 3.66 GB (one graph) | 0.74 GB | 0.64 GB |
| stream-encode 10 min | 4.37 GB | 0.99 GB | — |
| roundtrip 20 s | — | 0.77 GB | 0.56 GB |

What changed:
- **Chunked encoding.** A clip encoded in one graph holds attention scores
  for every pair of frames.
- **Host weights stay bf16.** The checkpoint is kept as stored (147 MB) and
  each tensor is widened only while a graph is built.
- **Outgrown graphs are dropped.** A grown encoder cache drops the smaller
  step graphs.
- **rlx-cpu scratch fix** (upstream, `rlx-cpu/src/thunk/exec_dispatch.rs`).
  Attention scratch was sized `heads × max(q, k)²` rather than
  `heads × q × k`, so one decode query over 8192 cached keys allocated and
  zeroed 2 GiB per run.

`--weight-storage auto` keeps the graph weights at the checkpoint's bf16. It
is lossless, and it passes parity on CPU (arm64, x86), CUDA and Vulkan.

## Parity

`scripts/liquid_codec_reference.py` runs LiquidAI's own
`modeling_liquid_codec.py` / `modeling_lfm3.py` from the checkpoint in
float32. The installed transformers is older than those files expect, so the
script shims a few symbols that this path never uses and builds the attention
masks directly. That shimmed run matches an independent torch port
(`scripts/lfm3_audio_reference.py`) to 2.8e-6.

```bash
python3 -I scripts/liquid_codec_reference.py   # writes tests/fixtures/{clip_24k.wav,reference.safetensors}
just test-liquid-codec-parity [DEVICE]
```

The clip is 4 s of `assets/harness/jfk_16k.wav` at 24 kHz (50 frames):

Seven tests: encoder latents and codes, chunked encode and streaming encode
(codes, and frames arriving exactly 480 samples after they end), decoded
audio, streaming decode vs whole, the trait, and the LFM3-Audio GGUF decoder.
Chunked and streaming encode give 600/600 codes on every row below.

| device | codes | latents max \|Δ\| | decode max \|Δ\| | stream vs whole |
|---|---|---|---|---|
| CPU, arm64 (Mac) | 600/600 | 1.6e-3 (cos 1.0000000) | 4.9e-6 | 3.2e-6 |
| CPU, x86 (Linux) | 600/600 | 1.6e-3 | 4.2e-6 | 3.2e-6 |
| Metal | 600/600 | 1.6e-3 | 4.4e-6 | 5.4e-6 |
| MLX | 600/600 | 1.6e-3 | 4.3e-6 | 4.3e-6 |
| wgpu (Mac / RTX 3080 Ti) | 600/600 | 1.6e-3 | 4.6e-6 / 3.4e-6 | 3.8e-6 |
| CoreML | 600/600 | 1.6e-3 | 3.9e-6 | 3.2e-6 |
| CUDA (RTX 3080 Ti) | 600/600 | 1.6e-3 | 4.7e-6 | 3.4e-6 |
| ROCm (Radeon 780M) | 600/600 | 1.6e-3 | 6.6e-6 | 2.7e-6 |
| CPU, x86 (780M host) | 600/600 | 1.6e-3 | 4.2e-6 | 5.1e-6 |
| Vulkan (MoltenVK / RTX 3080 Ti / 780M) | 600/600 | 1.6e-3 | 6.2e-6 / 5.2e-6 / 8.3e-6 | ≤5.0e-6 |

CUDA's default TF32 matmuls move the decoded audio by about 2e-3 (cosine
0.9999992), and the codes stay identical. The test turns TF32 off
(`RLX_CUDA_NO_TF32=1`) unless that variable is already set.

The latent difference is the same on every device. It comes from the host
STFT (f64 here, f32 in torch) amplified by the unit-phase features, not from
the graphs. The LFM3-Audio GGUF decoder (F16 mmproj) decodes the same codes
to within 6.1e-6. On 31 s of audio, the 250-frame chunked decode and
per-frame streaming differ by at most one int16 step. On 10 minutes, one
graph, 64-frame chunks and streaming encode give identical codes
(90000/90000).

`rlx-lfm3-audio`'s own parity (`just test-lfm3-audio-parity`, F16 LM,
x86 CPU) still passes every stage after its iSTFT and loader changes.

## Speed

`cargo run -p rlx-liquid-codec --release --example stream_latency -- --device D`
reports warm whole-clip encode and decode of the 4 s clip and the
per-frame cost of streaming after the first push (which includes the
compile):

| device | whole encode / decode, 50 frames | stream encode, per 80 ms | stream decode, per 80 ms (median / p90) |
|---|---|---|---|
| CPU, x86 laptop (idle) | 27 / 206 ms | 7.4 ms | 5.8 / 5.9 ms |
| CUDA, RTX 3080 Ti | 5 / 8 ms | 1.6 ms | 1.5 / 1.5 ms |
| ROCm, Radeon 780M | 30 / 54 ms | 5.3 ms | 5.6 / 8.5 ms |
| Vulkan, RTX 3080 Ti | 42 / 112 ms | 24 ms | 22 / 22 ms |
| Metal (Mac under load) | — / 13 ms | — | 7.0 / 10 ms |

Compiling each graph takes 120–350 ms. A streamed encoder step attends
over everything so far, so its cost grows with the stream. Over a
10-minute stream on a loaded Mac it averaged 14–30 ms per frame.
