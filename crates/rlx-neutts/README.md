# rlx-neutts

On-device voice-cloning TTS for RLX: **NeuTTS** (Neuphonic [neutts-nano](https://huggingface.co/neuphonic)) — a GGUF Llama-shaped backbone that generates speech tokens from IPA + reference codes, followed by an eager **NeuCodec** decoder that turns those tokens into a 24 kHz waveform. No Python at inference.

## How it fits

| Stage | Feature | Stack |
|-------|---------|-------|
| Speech-token backbone | `llama` (default) | [`rlx_llama32::Llama32Runner`](../rlx-llama32) + GGUF vocab encode/decode from [rlx-qwen35](../rlx-qwen35) |
| NeuCodec decoder | `codec` (default) | eager ndarray decoder → 24 kHz mono f32 |
| NeuCodec encoder | `codec` + optional `w2v-bert` | eager CodecEnc + SemanticEncoder + FSQ; Wav2Vec2-BERT layer-16 via [`rlx-wav2vec2-bert`](../rlx-wav2vec2-bert) |
| Backbone device | `metal`/`mlx`/`cuda`/`rocm`/`gpu`/`vulkan` | forwarded to `rlx-llama32`; `NeuTTS::load` auto-picks the fastest available |
| Parity | `parity-llama-cpp` | `llama-cpp-2` side-by-side (tests only) |
| Alt codec | `burn` / `wgpu` | Burn GPU/CPU NeuCodec (orthogonal to the RLX backbone) |

The backbone runs on GPU by default: `NeuTTS::load` honours `RLX_DEVICE`, else falls back to Metal → MLX → CUDA → ROCm → wgpu → CPU.

## Quick start

The `synth_metal` example synthesizes from a preset `.npy` of reference codes:

```bash
REF_NPY=/path/to/reference/jo.npy \
NEUTTS_DECODER_PATH=/path/to/neucodec_decoder.safetensors \
BACKBONE_GGUF=/path/to/neutts-nano-Q4_0.gguf \
cargo run -p rlx-neutts --release --example synth_metal \
  --features llama,codec,rlx,metal
```

Reference/input phonemes default to a built-in IPA pair; override with `REF_IPA` / `INPUT_IPA`.

## Public API

```rust
use rlx_neutts::NeuTTS;

// Load the GGUF backbone (device auto-selected) + NeuCodec decoder.
let model = NeuTTS::load(backbone_gguf_path, "en")?;

// IPA in, 24 kHz mono f32 out (reference codes clone the voice).
let audio: Vec<f32> = model.infer_from_ipa(input_ipa, &ref_codes, ref_ipa)?;

// Or decode pre-generated speech token ids directly.
let audio = model.decode_tokens(&speech_ids)?;
```

Sample rate is [`rlx_neutts::SAMPLE_RATE`] (24 kHz). Also public: [`GenerationConfig`],
`tokens::{build_prompt, extract_ids, ids_to_token_str}`, and per-backend
`*_feature_enabled()` reporting helpers.

## NeuCodec encoder (reference → speech codes)

```bash
# Acoustic path only (zeros semantic features):
NEUTTS_ENCODER_PATH=weights/tts/neutts/neucodec_encoder.safetensors \
NEUTTS_ENCODER_STUB_SEMANTIC=1 \
  cargo test -p rlx-neutts --test encoder_eager encoder_encode_stub --release

# Full path (Wav2Vec2-BERT layer 16):
#   huggingface-cli download facebook/w2v-bert-2.0 --local-dir weights/w2v-bert-2.0
NEUTTS_ENCODER_PATH=weights/tts/neutts/neucodec_encoder.safetensors \
RLX_W2V_BERT_DIR=weights/w2v-bert-2.0 \
  cargo test -p rlx-neutts --features w2v-bert --test encoder_eager \
  encoder_encode_w2v_silence --release -- --nocapture
```
