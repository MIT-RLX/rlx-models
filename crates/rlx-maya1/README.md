# rlx-maya1

[Maya1](https://huggingface.co/maya-research/maya1) — Maya Research's 3B
expressive **voice-design** TTS for RLX (**Apache-2.0**, 24 kHz).

Maya1 is Orpheus-family: a Llama-3.2-3B `LlamaForCausalLM` that emits SNAC
codec tokens, with a **byte-identical SNAC token layout** to Orpheus. So this
crate reuses [`rlx-orpheus`]'s GGUF backbone + SNAC decoder + prompt builder
wholesale — the only difference is Maya1's body format:
`<description="<voice design>"> <text>`.

Voice is a natural-language description (age, gender, accent, pitch, timbre,
pacing, emotion), not a preset. Inline emotion tags (`<laugh>`, `<whisper>`,
`<sigh>`, `<cry>`, …) are supported in the text.

## Setup

```bash
# quantized GGUF (Q4_K_M ~2GB; Q8_0 for higher quality)
huggingface-cli download mradermacher/maya1-GGUF maya1.Q4_K_M.gguf --local-dir weights/tts/maya1
# SNAC decoder (shared with Orpheus) — export once, then set the env var
python scripts/export_snac_decoder.py --repo hubertsiuzdak/snac_24khz --out weights/tts/snac_24khz
export ORPHEUS_SNAC_PATH=$PWD/weights/tts/snac_24khz/snac_24khz_decoder.safetensors
```

## Usage

```bash
cargo run -p rlx-maya1 --bin rlx-maya1 -- \
    --description "Realistic female voice in her 20s with a British accent. Warm timbre, conversational pacing." \
    --text "The quick brown fox jumps over the lazy dog." --out out.wav
```

`--gguf`, `--snac`, `--seed`, `--device cpu|metal|mlx|cuda|gpu`.

## Notes

- 3B model — Q4_K_M keeps it ~2 GB / RAM-friendly. CPU decode is slow; use a GPU
  backend (`--device metal|cuda`) for realtime.
- Whisper round-trip validated on English (`tests/whisper_roundtrip.rs`).
