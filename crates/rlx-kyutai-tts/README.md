# rlx-kyutai-tts

Native Rust inference for [Kyutai TTS](https://huggingface.co/kyutai/tts-1.6b-en_fr) — a 1.6B-parameter depth-multiplexed text-to-speech model from Kyutai Labs. Pairs the static architecture preset, weight fetching, and a CLI with [`rlx-mimi`](../rlx-mimi) for the Mimi neural codec.

## Audio examples

Two pre-generated samples ship with the crate under [`examples/audio/`](examples/audio/) — they exercise the real audio output pipeline (Mimi codec @ 24 kHz / 12.5 Hz, 8 codebooks) without requiring the 4 GB LM weight download.

### Voice cloning via Mimi codec round-trip

[`examples/audio/voice_clone_mimi_roundtrip.m4a`](examples/audio/voice_clone_mimi_roundtrip.m4a) — 12.08 s, ~100 KB, AAC 64 kbps.

A reference speech WAV (JFK's *"Ask not what your country..."*) is encoded by Mimi to 8 codebooks × 151 frames, then decoded back. This is the same codec the Kyutai TTS LM emits into — re-synthesising a voice from its Mimi tokens is the voice-identity backbone that the cross-attention `speaker_wavs` conditioner extends into prompted generation.

```bash
cargo run --release --example generate_wav -p rlx-kyutai-tts -- \
    --reference assets/jfk/jfk_rust_speech.wav \
    --out /tmp/voice-clone.wav
```

### TTS codec output (synthetic codes smoke test)

[`examples/audio/tts_synthetic_codes.m4a`](examples/audio/tts_synthetic_codes.m4a) — 5.12 s, ~42 KB, AAC 64 kbps.

A deterministic codes pattern (LCG, 64 frames × 8 codebooks, valid `card=2048` range) is decoded by Mimi to PCM. Not speech — this exercises the path Kyutai TTS's depth transformer emits codes into, so you can hear what valid-but-random codebook output sounds like before the LM is wired.

```bash
cargo run --release --example generate_wav -p rlx-kyutai-tts -- \
    --frames 64 --out /tmp/synthetic.wav
```

> **Status:** end-to-end TTS works — `KyutaiTtsSession::generate` runs the DSM
> loop (Helium temporal LM on RLX backends + eager DepFormer) and Mimi decode
> to 24 kHz PCM. Whisper E2E: `RLX_KYUTAI_TTS_E2E=1 cargo test -p rlx-kyutai-tts --test whisper_validate`.
>
> The upstream Kyutai `moshi` 0.6.4 + Candle crates are dev-only (parity / validation tests; see `tests/whisper_validate.rs`) and are NOT in the runtime dep graph.

## Architecture (mirrors `kyutai/tts-1.6b-en_fr/config.json`)

| Block | Spec |
|-------|------|
| **Backbone temporal LM** | 1B Helium-style, 16 layers × 16 heads, `d_model=2048`, `context=500`, `hidden_scale=4.125`, RoPE, RMSNorm (f32), SwiGLU/SiLU |
| **DepFormer (depth)** | 600M, 4 layers × 16 heads, `d_model=1024`, `dim_ff=3072`, **per-step weights** (33-entry sharing schedule → 11 unique heads), low-rank codebook embeddings (rank 128), multi-linear |
| **Codebooks** | 32 generated (`n_q = dep_q = 32`), `card=2048`, delays `[0, 0, 2, …, 2]` |
| **Text vocab** | 8000 SPM tokens (en/fr + audio control), padding id 3 |
| **Conditioners** | `speaker_wavs` (512-D tensor, cross-attn) + `cfg` (LUT, 7 bins 1.0–4.0) + `control` (LUT, 1 bin) |
| **Streams** | Demuxed second stream, audio shifted 1.28 s (16 frames @ 12.5 Hz) behind text |
| **Sampling** | Distilled CFG (single-pass — no batch doubling), `temp = text_temp = 0.6` |
| **Codec** | Mimi @ 12.5 Hz, 24 kHz mono, f32 PCM |

## Native module index (all pure Rust, no candle / moshi at runtime)

| Module | Role | Unit tests |
|--------|------|:----------:|
| [`nn`](src/nn.rs) | `linear`, `rms_norm`, `silu`, `swiglu_mlp`, `softmax_last_dim`, `rope_tables`, `apply_rope_vec`, `sin_pos_embed` | 8 |
| [`low_rank_embedding`](src/low_rank_embedding.rs) | Factorized codebook embedding `E ≈ A · B` (rank 128) | 3 |
| [`conditioner`](src/conditioner.rs) | `LutConditioner` (cfg, control), `TensorConditioner` (speaker_wavs) | 6 |
| [`fuser`](src/fuser.rs) | `sum` / `prepend` / `cross` routing into the backbone | 5 |
| [`cross_attention`](src/cross_attention.rs) | Multi-head cross-attn for speaker conditioning, sinusoidal pos-emb | 4 |
| [`transformer`](src/transformer.rs) | Streaming temporal backbone with optional cross-attn per layer, KV cache ring buffer | 5 |
| [`depformer`](src/depformer.rs) | Per-step DepFormer (head selection via schedule, low-rank input embeddings) | 5 |
| [`sampling`](src/sampling.rs) | Temperature + top-k `LogitsProcessor` + distilled `CfgSampler` | 5 |
| [`delays`](src/delays.rs) | `StreamLayout` — per-codebook delay padding + demuxed-stream offsets | 5 |
| [`weights`](src/weights.rs) | Native safetensors loader (F32 / F16 / BF16 → f32) + expected-key inventory | 6 |
| [`session`](src/session.rs) | `KyutaiTtsSession` — high-level fetch / load / generate entry point | — |
| [`config`](src/config.rs) | `KyutaiTtsConfig::v1_6b_en_fr()` matching the published `config.json` | 4 |

**Total: 56 native tests** (50 lib + 4 config_load + 2 whisper_validate).

## Files in `kyutai/tts-1.6b-en_fr`

| File | Size | Role |
|------|------|------|
| `config.json` | 2.4 kB | Static architecture |
| `dsm_tts_1e68beda@240.safetensors` | 3.68 GB | Backbone + DepFormer weights |
| `tokenizer-e351c8d8-checkpoint125.safetensors` | 385 MB | Mimi codec sidecar (same as `kyutai/moshiko-*`) |
| `tokenizer_spm_8k_en_fr_audio.model` | 120 kB | SentencePiece text tokenizer |

The Mimi sidecar matches the file shipped by `kyutai/moshiko-*` — [`rlx_mimi::resolve_candle_weights`](../rlx-mimi/src/download.rs) picks it up out of the TTS dir.

## Quick start

```bash
# Fetch weights (~4 GB) — LM + Mimi sidecar + SPM tokenizer
cargo run -p rlx-kyutai-tts --features hf-download -- --fetch

# Print the resolved architecture preset
cargo run -p rlx-kyutai-tts -- --info --model-dir .cache/kyutai-tts-1.6b-en_fr

# Generate audio (uses path 2: Mimi codec round-trip on a reference WAV)
cargo run --release --example generate_wav -p rlx-kyutai-tts -- \
    --reference assets/jfk/jfk_rust_speech.wav \
    --out /tmp/kyutai-tts.wav

# Native TTS synthesis (`KyutaiTtsSession::generate`)
cargo run -p rlx-kyutai-tts --release --features "hf-download,apple-silicon" -- \
    --prompt "Bonjour, comment ça va ?" \
    --voice expresso/ex03-ex01_happy_001 \
    --out-wav /tmp/kyutai.wav \
    --device metal

# Or: just kyutai-tts / just kyutai-tts-e2e
```

## Benches

Native kernel timings at production sizes (Apple Silicon, single-threaded eager ndarray):

```bash
cargo bench -p rlx-kyutai-tts --bench kernels
cargo bench -p rlx-kyutai-tts --bench backbone_step
```

| What | Median |
|------|-------:|
| `rms_norm` backbone step `[1, 2048]` | 1.42 µs |
| `apply_rope_vec` single head `[128]` | 23 ns |
| `softmax_last_dim` over `card=2048` | 3.07 µs |
| `LogitsProcessor.sample` top-k=256 | 3.91 µs |
| `linear` self-attn QKV step `[1, 2048] @ [2048, 2048]ᵀ` | 1.34 ms |
| `linear` SwiGLU-in step `[1, 2048] @ [16896, 2048]ᵀ` | 14.80 ms |
| `swiglu_mlp` backbone full block | 21.4 ms |
| `cross_attention.forward_step` `t_kv=16` | 3.43 ms |
| `low_rank_embedding.forward_one` (rank=128, dim=2048) | 317 µs |
| **One backbone step**, warm KV, no cross-attn | **470 ms** |
| **One backbone step**, warm KV + cross-attn (16-frame ctx) | **523 ms** |

(Full results in [the bench-suite report](../../README.md).)

## Validation harness

`tests/whisper_validate.rs` wires any Kyutai TTS-synthesised WAV through `rlx-whisper` for ASR-based ground truth, opt-in via env:

```bash
RLX_KYUTAI_TTS_VALIDATE_WAV=/tmp/kyutai-tts.wav \
RLX_KYUTAI_TTS_VALIDATE_PROMPT="ask not what your country can do for you" \
RLX_WHISPER_DIR=.cache/whisper-base.en \
    cargo test -p rlx-kyutai-tts --test whisper_validate -- --nocapture
```

Skips silently when env / weights are missing — CI stays green.

## Env

| Var | Default | Notes |
|-----|---------|-------|
| `RLX_KYUTAI_TTS_DIR` | `.cache/kyutai-tts-1.6b-en_fr` | Model directory override |
| `RLX_KYUTAI_TTS_CHECKPOINT` | `1.6b-en_fr` | Preset name |
| `RLX_MIMI_DIR` | `.cache/mimi` | Mimi cache (shared with `rlx-mimi`) |
| `RLX_KYUTAI_TTS_DEVICES` | `cpu` | Test-time device list (`all` to expand) |
| `RLX_WHISPER_DIR` | `.cache/whisper-base.en` | Whisper weights for validation tests |

## Voices

Voice conditioning uses pre-computed 512-D embeddings from [`kyutai/tts-voices`](https://huggingface.co/kyutai/tts-voices) — see [`KyutaiTtsVoice`](src/checkpoint.rs). Voices are not separate checkpoints; they fill the `speaker_wavs` cross-attention slot at inference time.

[`KyutaiTtsSession`](src/session.rs) defaults to `alba-mackenna/casual.wav` (same as the CLI). Unconditional generation is quieter / less intelligible on short prompts — call `set_voice(KyutaiTtsVoice::unconditional())` to opt out.

## Runtime dep graph (verified)

```
rlx-kyutai-tts
├── anyhow, half, ndarray, rand, safetensors, serde, serde_json
├── rlx-cli, rlx-mimi, rlx-models-core, rlx-runtime
```

No `candle`, no upstream `moshi`, no `tokio` in the runtime graph. Dev-only deps (parity / Whisper validation) are isolated under `[dev-dependencies]`.

## Upstream

- Model: <https://huggingface.co/kyutai/tts-1.6b-en_fr>
- License: CC-BY 4.0 (model), GPL-3.0-only (this crate)
- Reference implementation: <https://github.com/kyutai-labs/delayed-streams-modeling>
