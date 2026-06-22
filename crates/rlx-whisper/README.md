# rlx-whisper

OpenAI [Whisper](https://github.com/openai/whisper) ASR for RLX: compiled mel encoder, bucketed autoregressive decoder, and a native Rust **subtitles pipeline** (segment timestamps, word alignment, VAD chunking, diarization) — no Python runtime.

## Quick start

```bash
just fetch-whisper fetch-whisper-bench

# Plain greedy transcript
cargo run -p rlx-whisper --features tokenizer --release -- \
  --weights .cache/whisper-tiny/model.safetensors \
  --wav .cache/whisper-bench/jfk_16k.wav --lang en

# Subtitles (SRT + DTW word times + Silero VAD)
just whisper-subtitles
```

Weights: HuggingFace `openai/whisper-tiny` (or any compatible `model.safetensors` + `config.json` + `tokenizer.json`).

## Subtitles pipeline

| Stage | API / module |
|-------|----------------|
| ASR + segment times | [`WhisperPipeline`](src/pipeline.rs), [`timestamp_parse`](src/timestamp_parse.rs) |
| Word alignment (DTW) | [`WordAlignMode::Dtw`](src/transcript.rs), [`cross_attn_align`](src/cross_attn_align.rs) |
| Word alignment (CTC) | [`WordAlignMode::Wav2Vec2`](src/transcript.rs) + `rlx-wav2vec2-asr` |
| Silero VAD chunking | [`silero_vad`](src/silero_vad.rs) |
| Speaker labels | [`diarize`](src/diarize.rs) + `rlx-diarize` |
| Export | [`subtitles`](src/subtitles.rs) — SRT, VTT, TSV, JSON |

```rust
use rlx_whisper::{WhisperPipeline, WhisperPipelineOpts, WordAlignMode, WhisperRunner};

let runner = WhisperRunner::builder()
    .weights(".cache/whisper-tiny/model.safetensors")
    .device(rlx_runtime::Device::Metal)
    .timestamps(true)
    .build()?;

let mut pipeline = WhisperPipeline::new(runner, WhisperPipelineOpts {
    word_align: WordAlignMode::Dtw,
    use_silero_vad: true,
    max_region_batch: 4,
    ..Default::default()
});

let pcm = rlx_whisper::load_wav_mono_f32("audio16k.wav")?;
let transcript = pipeline.run(&pcm)?;
println!("{}", rlx_whisper::to_srt(&transcript));
```

### CLI flags

| Flag | Purpose |
|------|---------|
| `--timestamps` | Structured [`WhisperTranscript`](src/transcript.rs) |
| `--word-align dtw\|wav2vec2` | Word-level times |
| `--silero-vad` | Long-audio chunking via `rlx-vad` Silero |
| `--diarize` | Speaker labels (`diarize` feature) |
| `--max-region-batch N` | Batched region encode width |
| `--output PATH` | SRT / VTT / TSV / JSON (from extension or `--output-format`) |

## Device routing

When `--device metal` (or CUDA for encoder):

| Stage | Typical device | Notes |
|-------|----------------|-------|
| Mel encoder | Metal / CUDA | [`whisper_encoder_device`](src/backend.rs) |
| Cross, prefill, decode | CPU | [`whisper_decoder_device`](src/backend.rs) — parity gate |
| DTW align-hidden | Same as encoder | Compiled graph through alignment layer |

Call [`WhisperRunner::prepare_pcm_geometry`](src/runner.rs) (or `mel_frames_for_pcm` on the builder) so encoder graphs match utterance length instead of padding to 30 s.

## Performance (JFK, whisper-tiny, ~11 s)

Approximate `timestamps+dtw` on Apple Silicon (1 warmup run):

| Backend | ASR ms | Align ms | Total ms | RTF |
|---------|--------|----------|----------|-----|
| Metal | ~3200–3400 | ~120–130 | ~3300–3600 | ~0.30–0.32 |
| CPU | ~3850 | ~585 | ~4400 | ~0.40 |

GPU DTW align-hidden replaces ~800 ms CPU decoder forward with ~120 ms Metal graph + batched QK.

```bash
just bench-whisper-subtitles -- --device metal --modes timestamps+dtw
just bench-whisper-subtitles-all-backends -- --modes timestamps+dtw
```

## Batched region decode

- **Energy VAD** (`transcribe_structured_vad`): full batched cross + prefill + decode per chunk.
- **Silero VAD**: batched encode; decode is serial when a chunk has **multiple** regions (correctness); single-region chunks use full batch decode.
- Batched greedy decode indexes logits with `dec_seq = prompt.len()` on prefill, then `dec_seq = 1` on each decode step ([`batched_logits_row`](src/decode.rs)).

## Features

| Feature | Enables |
|---------|---------|
| `tokenizer` | Greedy/beam decode, CLI `--wav` |
| `timestamps` (default) | Segment parse, pipeline, subtitle export |
| `word-dtw` | Cross-attention + DTW word alignment |
| `word-w2v` | Wav2Vec2 CTC forced alignment |
| `silero-vad` | Silero VAD chunking |
| `diarize` | ECAPA-TDNN speaker labels via `rlx-diarize` |
| `metal`, `cuda`, … | Forwarded to `rlx-runtime` |

Facade feature `whisper-subtitles` on `rlx-models` enables the full stack for examples and tests.

## Tests

```bash
just test-whisper-timestamps          # segment parse, DTW units, wav2vec2/diarize crates
just test-whisper-e2e                 # greedy decode vs reference (needs weights)
cargo test -p rlx-whisper --features timestamps --release
```

## Related crates

- `rlx-wav2vec2-asr` — CTC forced alignment (WhisperX-style)
- `rlx-diarize` — speaker embedding + clustering
- `rlx-vad` — Silero VAD used by `--silero-vad`
