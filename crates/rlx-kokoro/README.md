# rlx-kokoro

Kokoro-82M text-to-speech for RLX — a StyleTTS2 acoustic model with an
ISTFTNet vocoder (82M parameters, Apache-2.0, 24 kHz output, 8 languages /
50+ voices).

Kokoro is in the same StyleTTS2 family as [`rlx-kittentts`](../rlx-kittentts),
so it reuses that crate's espeak-ng → IPA phonemizer, text preprocessor and
ONNX Runtime execution-provider selector. The Kokoro-specific pieces here are
the misaki phoneme vocabulary (loaded from `tokenizer.json`), the `[510, 256]`
voice-style packs, and the `input_ids` / `style` / `speed` → `waveform` runner.

## Quick start

```bash
# Download the ONNX bundle + English voices (needs the hf-download feature)
cargo run -p rlx-kokoro --features hf-download --bin rlx-kokoro -- --download

# Synthesize
cargo run -p rlx-kokoro --bin rlx-kokoro -- \
    --text "Hello from Kokoro." --voice af_heart --out kokoro.wav

# List available voices
cargo run -p rlx-kokoro --bin rlx-kokoro -- --list-voices

# Synthesize directly from phonemes (skip G2P)
cargo run -p rlx-kokoro --bin rlx-kokoro -- --ipa "həlˈoʊ" --voice am_michael
```

Set `RLX_KOKORO_DIR` to point at a model directory, or pass `--data <dir>`.
The default directory is `.cache/kokoro-82m`. The default **native** CLI path needs
`onnx/rlx-split/` (one-time: `python crates/rlx-kokoro/scripts/split_kokoro.py …`).

## Model directory layout

```text
config.json
tokenizer.json
onnx/model.onnx          # or model_fp16.onnx, model_q8f16.onnx, …
voices/af_heart.bin      # one raw-f32 [510, 256] pack per voice
```

Weights: [`onnx-community/Kokoro-82M-v1.0-ONNX`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX).

## Backends

| Feature | Path | Devices |
|---------|------|---------|
| **`native`** (default) | **native RLX (ort-free)** | **cpu / metal / mlx / wgpu / coreml** |
| `onnx` | ONNX Runtime (optional) | CPU |
| `metal` / `mlx`  | ONNX Runtime | CoreML execution provider (macOS) |
| `cuda`           | ONNX Runtime | CUDA execution provider |
| `gpu`            | ONNX Runtime | DirectML / CUDA / CoreML |

For the ort path, select the device with `--device cpu|metal|mlx|cuda|gpu`; the
requested EP is tried first with a CPU fallback.

### Native / hybrid multi-backend (`native`)

The `native` feature runs the graph-split decoder on the RLX compiler. With
`onnx` (default), the duration/prosody **encoder** prefers onnxruntime on CPU
(fast path); set `RLX_KOKORO_NATIVE_ENC=1` for a fully native RLX encoder
(also Whisper fox 6/6). The monolithic graph has a data-dependent length
regulator and an ISTFT (`NonZero`/`ScatterND` overlap-add) that don't fit one
static-shape compile, so it is **graph-split** into two fixed-shape subgraphs
with the dynamic pieces in Rust — verified **bit-exact** (cosine 1.0, max_abs 0)
against the monolithic model, and whisper-validated end-to-end (coverage 1.00
on CPU):

```text
encoder.onnx   [input_ids, style, speed] → prosody[1,640,seq], text[1,512,seq], dur[1,seq]
  ── Rust length regulator: repeat_interleave columns by dur → en[1,640,F], asr[1,512,F] ──
decoder_raw.onnx  [en, asr, style] → raw waveform
  ── Rust ISTFT overlap-add normalization (window_sum, ×n_fft/hop, crop n_fft/2) ──
                → 24 kHz waveform
```

Both subgraphs import through `rlx-onnx-import` → rlx-ir (StyleTTS2's LSTM/STFT
and the ALBERT duration predictor are all covered) and run on any RLX backend.
Produce the split bundle once, then synthesize:

```bash
# one-time: split the monolithic model into the native bundle (onnx/rlx-split/)
python crates/rlx-kokoro/scripts/split_kokoro.py \
  weights/tts/kokoro-82m/onnx/model.onnx weights/tts/kokoro-82m/onnx/rlx-split

cargo run -p rlx-kokoro --no-default-features --features native,espeak \
  --example native_synthesize -- \
  --model weights/tts/kokoro-82m --voice af_heart \
  --text "Hello from native Kokoro." --out out.wav [--device cpu|metal|mlx|gpu|coreml|vulkan]
```

CoreML (`--device coreml` / `ane`) auto-sets `RLX_COREML_UNITS=gpu` via
`rlx_tiny_tts::resolve_tts_device` — the default Neural-Engine path crashes BNNS
on these graphs. Vulkan is available with `--features vulkan` / `all-backends`.
`NativeKokoro::load(model_dir, device)` is a drop-in for the ort `Kokoro` with
the same `generate_from_text` / `infer_phonemes` entry points.

## Voices & languages

Voice names encode language and gender: `af_/am_` American English, `bf_/bm_`
British English, and `e*/f*/h*/i*/j*/p*/z*` for Spanish / French / Hindi /
Italian / Japanese / Portuguese / Mandarin. The bundled espeak-ng data covers
English; non-English voices require the corresponding espeak language data.

## API

```rust
use rlx_kokoro::Kokoro;

let tts = Kokoro::load_from_dir(std::path::Path::new(".cache/kokoro-82m"))?;
let audio = tts.generate_from_text("Hello from Kokoro.", "af_heart", 1.0)?;
tts.write_wav(&audio, std::path::Path::new("out.wav"))?;
```
