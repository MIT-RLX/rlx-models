# rlx-voxtral

Mistral **[Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)** speech LM for RLX: a Whisper-style audio encoder + 4× projector + Llama text decoder, with a fully native Rust/RLX frontend (Whisper log-mel + Mistral transcription template) — no Python runtime. Audio and text embeddings are fused additively at `audio_token_id` placeholders before the Llama trunk runs.

## Quick start

```bash
# Native transcription: 16 kHz mono WAV → text (log-mel + prompt built in Rust)
cargo run -p rlx-voxtral --release -- \
  --weights /path/to/Voxtral-Mini-3B-2507/model.safetensors \
  --wav audio16k.wav --transcribe --lang en --device cpu

# or via the just recipe
just voxtral --weights .../model.safetensors --wav audio16k.wav --transcribe
```

Weights: HuggingFace `mistralai/Voxtral-Mini-3B-2507` (safetensors with `audio_tower.*`, `multi_modal_projector.*`, `language_model.*` tensors) plus its `config.json` and tokenizer.

### CLI flags (`src/cli.rs`)

| Flag | Purpose |
|------|---------|
| `--weights PATH` | Model safetensors (required) |
| `--wav PATH` | 16 kHz mono WAV (required unless `--dry`) |
| `--transcribe` | Native log-mel + Mistral transcription prompt |
| `--prompt-ids 1,24,…` | Explicit prompt token ids (when not `--transcribe`) |
| `--lang` / `--language` | Transcription language |
| `--config PATH` | Override `config.json` |
| `--device cpu\|metal\|mlx\|cuda\|…` | Backend |
| `--max-tokens N` | Decode cap (default 256) |
| `--dry` | Load + validate config, skip forward |

## Public API

```rust
use rlx_voxtral::{VoxtralRunner, decode_token_ids};
use rlx_runtime::Device;

let runner = VoxtralRunner::builder()
    .weights("/path/to/Voxtral-Mini-3B-2507/model.safetensors")
    .device(Device::Cpu)
    .max_new_tokens(256)
    .build()?;

// End-to-end native transcription (WAV → token ids)
let tokens = runner.transcribe_wav("audio16k.wav".as_ref(), Some("en"))?;
let text = decode_token_ids(Some(runner.model_dir()), &tokens)?;
# anyhow::Ok(())
```

Lower-level entry points are also public: `runner.encode_audio(&mel)` (encoder + projector → audio embeds), `runner.generate(prompt_ids, &mel)` (fused prefill + bucketed KV decode), and the graph builders `build_voxtral_encoder_built`, `build_voxtral_projector_built`, `build_voxtral_prefill_built`, `build_voxtral_decode_built`. Audio helpers: `pcm_to_mel`, `mel_from_flat`, `MelSpectrogram`, `fuse_inputs_embeds`, `transcription_prompt_ids`.

## How it fits

| Piece | Crate |
|---|---|
| Whisper-style audio encoder | [`rlx-whisper`](../rlx-whisper) |
| Llama text decoder trunk | [`rlx-llama32`](../rlx-llama32) |
| Compile / device routing | [`rlx-runtime`](../rlx-runtime) |

The decoder compiles the prefill and a single fixed-cap decode graph once, then serves every `past_len` by padding the KV cache to the bucket and masking padded slots (`bucket_decode_mask` + `compact_bucketed_kv_buffer`) — no per-token recompile.

## Tests

```bash
cargo test -p rlx-voxtral            # synthetic encoder → projector → prefill graph runs on CPU
```
