# rlx-metavoice — MetaVoice-1B

Zero-shot voice-cloning TTS ([`metavoiceio/metavoice-1B-v0.1`](https://huggingface.co/metavoiceio/metavoice-1B-v0.1),
Apache-2.0, ~1.2B) on RLX.

## Status

| stage | status |
|-------|--------|
| Weights `weights/tts/metavoice/` | ✅ `.pt` → safetensors |
| Custom BPE tokenizer | ✅ |
| First-stage GPT (24×2048, CFG, KV cache) | ✅ eager native (greedy default) |
| Second-stage fine codebooks (6×384) | ✅ eager native |
| EnCodec 24 kHz decode | ✅ via `rlx-encodec` (cpu/metal/mlx/wgpu/coreml/cuda) |
| Speaker LSTM from reference wav | ✅ (required; default `bria_16k.wav`) |
| PCM postprocess | ✅ silence trim + peak normalize |

## Convert + run

```bash
python3 crates/rlx-metavoice/scripts/convert_pt_to_safetensors.py weights/tts/metavoice
# EnCodec (once):
mkdir -p weights/tts/encodec24
cp crates/rlx-encodec/tests/fixtures/encodec24.safetensors weights/tts/encodec24/model.safetensors

cargo run -p rlx-metavoice --release -- \
  --text "The quick brown fox jumps over the lazy dog." \
  --max-tokens 864 --device metal --output /tmp/metavoice.wav
```

Defaults: **greedy** argmax, `--max-tokens 864`, `--seed 1337`, speaker from
`bria_16k.wav`. Use `--sample` for top-p. First-stage is CPU-eager (~8–10 min for
864 steps release). EnCodec is the device path: `--device metal|mlx|wgpu|coreml|cuda|cpu`.

## Whisper + backend matrix

```bash
# Apple Silicon
cargo run -p rlx-metavoice --release --example backend_matrix \
  --features "metal,mlx,gpu,coreml"

# CUDA host (set RLX_CUDA_HOST)
cargo run -p rlx-metavoice --release --example backend_matrix --features cuda,gpu
```

Env: `RLX_TEXT`, `RLX_MAX_TOKENS` (default 864), `RLX_DEVICES=cpu,metal,mlx,wgpu,coreml,cuda`,
`RLX_CODES_CACHE=/tmp/metavoice_codes.json` (reuse first/second-stage codes),
`RLX_WHISPER_DIR=.cache/whisper-tiny`, `RLX_SAMPLE=1` for top-p, `RLX_FORCE_RESYNTH=1`
to rebuild codes.

Validated sentence (Whisper-tiny, **≥5/6** fox words `quick/brown/fox/jumps/lazy/dog`,
cos=1.0 vs CPU EnCodec):

| host | backends |
|------|----------|
| Mac | CPU, Metal, MLX, wgpu, CoreML |
| NVIDIA (RTX 3080 Ti) | CPU, CUDA, wgpu |

**Requires** speaker conditioning (`bria_16k.wav` or `--reference`). Zero emb or a
short max-token budget without a speaker is what produced the old 67% transcript
(“To quick brain fox straps…”).
