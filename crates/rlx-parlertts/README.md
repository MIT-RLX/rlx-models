# rlx-parlertts — Parler-TTS Mini v1

Voice-description TTS ([`parler-tts/parler-tts-mini-v1`](https://huggingface.co/parler-tts/parler-tts-mini-v1), ~878M)
on RLX backends. **Native / ort-free**: T5 encoder + 9-codebook delay-pattern decoder
via `rlx-onnx-import`, Descript DAC via [`rlx-dac`](../rlx-dac).

## Weights layout

```text
weights/tts/parlertts/
  onnx/text_encoder.onnx
  onnx/decoder.onnx
  tokenizer.json
  config.json
weights/tts/parler-dac/
  config.json
  model.safetensors
```

Export ONNX from the HF checkpoint:

```bash
python3 crates/rlx-parlertts/scripts/export_onnx.py \
  weights/tts/parlertts weights/tts/parlertts/onnx
```

Fetch DAC (44.1 kHz / 9 codebooks):

```bash
huggingface-cli download parler-tts/dac_44khz \
  --local-dir weights/tts/parler-dac
```

## Run

```bash
cargo run -p rlx-parlertts --release -- \
  --text "Hello from Parler." \
  --voice "A clear female voice speaks slowly." \
  --device cpu \
  --output /tmp/parler.wav
```

Example (same pipeline):

```bash
cargo run -p rlx-parlertts --example synthesize --release -- "Hello world." /tmp/out.wav
```

## Notes

- The current ONNX decoder export has **no** `prompt_input_ids` input. True Parler
  routes the **voice description** through T5 and the **transcript** as a prompt
  embedding prefix (`embed_prompts`). Until a re-export lands that path, this crate
  feeds the **transcript** into the T5 encoder so the AR loop has content to speak;
  `--voice` still mixes into the sampling seed.
- ort is a **dev-only** dependency (`examples/native_parity.rs`). Runtime has zero
  ONNX Runtime.

## Tests

```bash
cargo test -p rlx-parlertts --lib --test parlertts_parity
# heavy (needs weights + Whisper):
cargo test -p rlx-parlertts --test native_whisper_roundtrip --release -- --nocapture
```
