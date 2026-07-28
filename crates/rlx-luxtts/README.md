# rlx-luxtts

LuxTTS voice-cloning text-to-speech for RLX — a **ZipVoice-distill**
flow-matching model (123M, Apache-2.0, 24 kHz) that clones a voice from a short
reference clip + its transcript.

Three ONNX subgraphs are chained with Rust glue mirroring the reference
`zipvoice` ONNX inference:

1. `text_encoder(tokens, prompt_tokens, prompt_len, speed)` → `text_condition`
2. reference wav → `VocosFbank` log-mel (× 0.1) → `speech_condition`
3. **4-step anchor-ODE flow-matching** loop: `fm_decoder(t, x, text_cond, speech_cond, guidance)` → velocity `v`
4. `vocoder_spec` (Vocos backbone + spectral head) → STFT real/imag → **Rust ISTFT** → 24 kHz audio

The DSP (`VocosFbank` log-mel + ISTFT) is validated **bit-close to
torch/torchaudio** in `tests/dsp_parity.rs` (cosine 1.000000). The tokenizer is
char-level espeak-IPA (reusing KittenTTS's bundled espeak-ng).

## Setup

The Vocos vocoder ships as raw weights (`vocoder/vocos.bin`), not ONNX — and its
ISTFT can't be represented in ONNX. Export the spectral head once (the ISTFT is
done in Rust):

```bash
# in a venv:  pip install vocos onnxscript
python crates/rlx-luxtts/scripts/export_vocoder.py \
    weights/tts/luxtts/vocoder/vocos.bin weights/tts/luxtts/onnx/vocoder_spec.onnx
```

Model directory (`weights/tts/luxtts/`): `text_encoder.onnx`, `fm_decoder.onnx`,
`onnx/vocoder_spec.onnx`, `tokens.txt`, `vocoder/vocos.bin`. Weights:
[`YatharthS/LuxTTS`](https://huggingface.co/YatharthS/LuxTTS).

## Usage

```bash
cargo run -p rlx-luxtts --bin rlx-luxtts -- \
    --prompt-wav reference.wav \
    --prompt-text "transcript of the reference audio" \
    --text "Text to speak in the cloned voice." \
    --out out.wav
```

`--steps` (default 4), `--guidance` (3.0), `--speed` (1.0), `--seed`.

## Backends

**Default path is native RLX** (`encoder_body` + `fm_decoder` + `vocoder_spec` via
`rlx-tiny-tts`). Mel + ISTFT stay in Rust. Optional `--features onnx` keeps an
ORT reference path for parity.

`Device::Ane` / CoreML runs **end-to-end** on all three subgraphs. Upstream
`rlx-coreml` defaults **fp32** graphs to CPU+GPU compute units (Neural-Engine
BNNS AOT SIGSEGVs on these CFM graphs). TinyModel also pins
`RLX_COREML_UNITS=gpu` when unset so f16 edges stay off ANE. Override with
`RLX_COREML_UNITS=all|cpu|ane` if needed. Metal / MLX / wgpu / CUDA remain
available for non-CoreML GPU.

**CUDA (NVIDIA):** RTF ≈1.4×, cos **0.99979** vs CPU, whisper **0.85** (same known
espeak coverage as Apple backends).

## Known limitation

The tokenizer uses espeak-ng, which differs slightly from the reference
`piper_phonemize`, so a few vowels can shift (e.g. `fox`→`fix`). Output stays
intelligible (~0.85 whisper word coverage); matching `piper_phonemize` exactly
is a refinement.
