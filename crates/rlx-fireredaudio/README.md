# rlx-fireredaudio

**FireRedAudio** ([FireRedTeam](https://github.com/FireRedTeam/FireRedAudio)) on
RLX — a general-purpose **audio language model** with a shared **Qwen3.5 (~9B)**
backbone and **decoupled continuous representations**: a Whisper-style Audio
Encoder for understanding, and a **RedAE → Patch Encoder → DiT** pathway for
speech generation.

| Path | Status |
|------|--------|
| ASR / understand | **End-to-end** — mel → Conv1d encoder → Qwen3.5 `inputs_embeds` → greedy text |
| TTS / edit / voice design | API + ChatML prompts; RedAE+DiT hybrid AR graphs next |

| Component | Reuse |
|-----------|-------|
| Backbone | `rlx-qwen35` (`hidden_prefill` + `force_host_embed`) |
| Mel frontend | Whisper FE (128 bins @ 16 kHz) |
| Audio encoder | Native HIR (Conv1d-via-Conv2d + adapter) |
| Generation | planned RedAE / DiT (24 kHz, 25 Hz latents) |

Weights: [FireRedTeam/FireRedAudio](https://huggingface.co/FireRedTeam/FireRedAudio)
(`FireRedAudio/` safetensors + `RedAE_decoder/model.pt` for generation).
Apache-2.0.

## Quick start

```bash
just fetch-fireredaudio          # full ~30GB; or ONLY_META=1 for config/tokenizer
just fireredaudio -- --weights .cache/fireredaudio --task asr --audio clip16k.wav
just fireredaudio -- --device metal --weights .cache/fireredaudio --task asr --audio clip16k.wav
just fireredaudio -- --show-prompt --task understand --prompt "how many speakers?"
```

`--device` matches Qwen3.5 LM backends: `auto|cpu|metal|mlx|cuda|rocm|gpu|vulkan|coreml`.

Synth audio-encoder on every available backend (no HF weights):

```bash
just features=all-backends test-fireredaudio-backends
```

Library: `FireRedRunner::builder().weights(dir).device(device).build()?` then
`asr_wav` / `understand_wav`.
