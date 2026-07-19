# rlx-miotts — MioTTS-0.6B

[MioTTS-0.6B](https://huggingface.co/Aratako/MioTTS-0.6B) (Apache-2.0) on RLX:
Qwen3-0.6B speech LM + [MioCodec-25Hz-24kHz](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz)
decode → 24 kHz PCM. Voice via preset global embeddings (`en_female`, …).

## Status

| Stage | Path |
|-------|------|
| LM (Qwen3-0.6B) | Eager native via `rlx-qwen3` (CPU) |
| Codec body | `decoder_body.onnx` via native RLX (`rlx-tiny-tts`) |
| ISTFT | Host `rlx_xcodec::istft_same` |

## Fetch + export codec

```bash
just fetch-miotts
# one-time: create .venv-miotts and export decoder_body.onnx
uv venv .venv-miotts --python 3.12
uv pip install --python .venv-miotts/bin/python torch soundfile onnx onnxruntime \
  'transformers<5' 'miocodec @ git+https://github.com/Aratako/MioCodec@main'
just export-miocodec
```

Presets (`en_female.f32`, …) are written next to the HF `.pt` files under
`weights/tts/miotts/presets/`.

## Run

```bash
just miotts
just miotts-whisper
just miotts-backends   # LM once → codec per Device → Whisper fox
```

CLI:

```bash
cargo run -p rlx-miotts --release --features apple-silicon -- \
  --text "The quick brown fox jumps over the lazy dog." \
  --preset en_female --device metal --seed 42 --output /tmp/miotts.wav
```

`--device` selects the RLX backend for MioCodec (`decoder_body.onnx`). The LM
stays on CPU.

## Pipeline

```text
user text
  → chat template → Qwen3 AR → <|s_N|> content codes (0..12799)
  → MioCodec native (codes + preset global emb → mag/phase)
  → ISTFT → 24 kHz mono
```
