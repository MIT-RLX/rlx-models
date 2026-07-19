# rlx-openvoice

[OpenVoice v2](https://github.com/myshell-ai/OpenVoice) — MyShell's zero-shot
voice-cloning TTS for RLX (**MIT**, 22.05 kHz).

Pipeline (all **native RLX** — no ONNX Runtime at runtime):
1. **MeloTTS base** via `rlx-tiny-tts` speaks the text.
2. A magnitude **spectrogram** is computed in Rust.
3. Native **`tone_extract.onnx`** (rlx-ir) pulls a 256-d speaker embedding from
   the base audio and the reference clip.
4. Native **`tone_color.onnx`** transfers the reference timbre onto the base
   audio → cloned speech.

Runs on every RLX backend (`cpu` / `metal` / `mlx` / `wgpu` / `cuda` / …).

## Setup

```bash
# MeloTTS base bundle (tiny-tts engine) → weights/tiny-tts-rlx/
# OpenVoice v2 ONNX (converter + extractor):
huggingface-cli download Hinotsuba/OpenVoice-ONNX-v2 --local-dir weights/tts/openvoice
```

## Usage

```bash
cargo run -p rlx-openvoice --bin rlx-openvoice --features apple-silicon -- \
    --ref-wav reference.wav \
    --text "The quick brown fox jumps over the lazy dog." --out out.wav
```

`--tau` (flow temperature, default 0.3), `--melo-dir`, `--data`, `--device`.

Whisper round-trip: ~0.89 coverage (`tests/whisper_roundtrip.rs`). Tone-color
conversion trades a little intelligibility for timbre transfer.
