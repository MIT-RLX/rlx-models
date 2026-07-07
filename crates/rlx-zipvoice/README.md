# rlx-zipvoice

[ZipVoice](https://github.com/k2-fsa/ZipVoice) flow-matching voice-cloning TTS
for RLX (k2-fsa, **Apache-2.0**, 24 kHz).

ZipVoice and LuxTTS are the same architecture — LuxTTS is ZipVoice-distill
fine-tuned — with byte-identical ONNX interfaces. This crate **reuses the
`rlx-luxtts` runner + DSP wholesale**; it only changes the inference defaults
(4-step anchor-ODE sampler, `speed_mult = 1.0` — no LuxTTS `×1.3` bump).

## Setup

Download the **`zipvoice_distill/`** subdir of
[`k2-fsa/ZipVoice`](https://huggingface.co/k2-fsa/ZipVoice) into
`weights/tts/zipvoice-distill/`: `text_encoder.onnx`, `fm_decoder.onnx`,
`tokens.txt`. Then export the Vocos vocoder head once (venv with `vocos onnxscript`):

```bash
python crates/rlx-zipvoice/scripts/export_vocoder.py \
    weights/tts/zipvoice-distill/onnx/vocoder_spec.onnx
```

## Usage

```bash
cargo run -p rlx-zipvoice --bin rlx-zipvoice -- \
    --prompt-wav reference.wav \
    --prompt-text "transcript of the reference audio" \
    --text "Text to speak in the cloned voice." --out out.wav
```

`--steps` (default 4), `--speed` (1.0), `--device cpu|metal|mlx|cuda|gpu`.

## Note

The full (non-distilled) `zipvoice/` model uses a different (Euler) sampler than
the distill's anchor-ODE and is not covered by this crate's defaults. English
uses the bundled espeak-ng (a few vowels shift vs the reference build); output
stays intelligible (~0.85 whisper coverage).
