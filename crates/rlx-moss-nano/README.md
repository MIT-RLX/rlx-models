# rlx-moss-nano

[MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano) — OpenMOSS's
0.1B multilingual **hierarchical autoregressive** codec-LM TTS for RLX
(**Apache-2.0**, 48 kHz stereo, en/zh/ja).

A pure "audio-tokenizer + LLM" pipeline, run fully on ONNX Runtime:

- a **global** 12-layer transformer (`prefill` + KV-cached `decode_step`),
- a fused **local** graph (`fixed_sampled_frame`) that samples the 16
  audio-codebook tokens per frame (sampling done inside ONNX via random-uniform
  inputs + a repetition-seen mask),
- the separate **MOSS-Audio-Tokenizer** (`decode_full`) that turns codes into
  48 kHz stereo audio.

Voice cloning uses 18 builtin voices (pre-computed reference codes ship in the
manifest — no reference audio needed).

## Setup

Download two repos into `weights/tts/moss-nano/`:

```bash
# LM (prefill / decode / local + external .data + tokenizer.model + manifest)
huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX --local-dir weights/tts/moss-nano
# codec → weights/tts/moss-nano/codec/
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX --local-dir weights/tts/moss-nano/codec
# convert the BPE tokenizer.model → tokenizer.json (pure-Rust loadable; see note)
python crates/rlx-moss-nano/scripts/convert_tokenizer.py weights/tts/moss-nano
```

## Usage

```bash
cargo run -p rlx-moss-nano --bin rlx-moss-nano -- \
    --text "The quick brown fox jumps over the lazy dog." --voice Trump --out out.wav
cargo run -p rlx-moss-nano --bin rlx-moss-nano -- --list-voices
```

`--seed`, `--max-frames`, `--device cpu|metal|mlx|cuda|gpu`.

## Notes

- **Tokenizer**: we convert `tokenizer.model` (a BPE SentencePiece model) to a
  `tokenizer.json` loaded by the pure-Rust `tokenizers` crate. We must **not**
  link the C++ `sentencepiece` crate: it statically bundles its own protobuf,
  which clashes (ODR) with ONNX Runtime's protobuf and breaks ORT model loading.
- Whisper round-trip: 1.00 coverage on English (`tests/whisper_roundtrip.rs`).
