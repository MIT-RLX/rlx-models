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
The default directory is `.cache/kokoro-82m`.

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
| `onnx` (default) | ONNX Runtime | CPU |
| `metal` / `mlx`  | ONNX Runtime | CoreML execution provider (macOS) |
| `cuda`           | ONNX Runtime | CUDA execution provider |
| `gpu`            | ONNX Runtime | DirectML / CUDA / CoreML |

Select the device with `--device cpu|metal|mlx|cuda|gpu`; the requested EP is
tried first with a CPU fallback.

### Native rlx-ir multi-backend (planned)

A fully native RLX graph path (Metal / MLX / wgpu via the RLX compiler) is
planned. StyleTTS2 uses LSTM and STFT operators that `rlx-onnx-import` does not
yet cover, so — like `kitten_tts_mini_rlx` for KittenTTS — it needs a
hand-decomposed rlx-ir graph. Until then the ONNX Runtime path above is the
supported route.

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
