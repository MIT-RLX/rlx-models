# rlx-voxtral-tts

**Voxtral-4B-TTS** (Mistral) on RLX — a native Rust port of vLLM-Omni's `VoxtralTTSAudioGeneration` (no Python at inference). The pipeline is a Ministral LM backbone → acoustic flow-matching head → codec decode to waveform.

## Quick start

```bash
cargo run -p rlx-voxtral-tts --bin rlx-voxtral-tts --release -- --help

# Stage timing on a checkpoint:
just bench-voxtral-tts
```

Set the checkpoint dir via `RLX_VOXTRAL_TTS_DIR`.

## Modules

- `backbone` — the Ministral LM trunk (via [rlx-llama32](../rlx-llama32)).
- `acoustic` / `acoustic_flow` / `acoustic_engine` — flow-matching acoustic head.
- `codec` — neural codec decoder → audio.
- `cli` / `bench` — command-line entry + stage timing.

## How it fits

- [rlx-llama32](../rlx-llama32) — the Ministral LM backbone.
- [rlx-voxtral](../rlx-voxtral) — the Voxtral ASR/audio-understanding sibling.
- [rlx-voxtral-tts-train](../rlx-voxtral-tts-train) — voice-cloning fine-tuning for this model.
