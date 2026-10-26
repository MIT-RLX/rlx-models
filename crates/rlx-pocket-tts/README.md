# rlx-pocket-tts

Native Rust port of [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
([model card](https://huggingface.co/kyutai/pocket-tts)): a small text-to-speech
model (a causal FlowLM over continuous Mimi latents) that streams 24 kHz audio.
`PocketTts` runs it as rlx graphs on every RLX backend.

## Models

All 27 upstream configs are bundled (`rlx-pocket-tts languages` lists them);
`--language` takes the upstream name:

| Config | FlowLM | Notes |
|---|---|---|
| `english` (= `english_2026-10`), `english_2026-09[_2]`, `english_2026-04`, `english_2026-01` | 6 × 1024 | `english_2026-01` is the January checkpoint (`tts_b6369a24`, `audio_prompt` voices) |
| `english_2026-04_24l`, `english_2026-09_24l` | 24 × 1024 | |
| `english_drifting_26-09` | 6 × 1024 | one-step drifting sampler head |
| `english_18m/7m/2m_2026-10` (+ `_refl`) | 8–12 layers, GQA, low-rank text table | pointwise head, own small Mimi decoder |
| `french`, `german`, `italian`, `portuguese`, `spanish`, `dutch` (+ `_24l`) | 6 / 24 × 1024 | |

## Weights

```bash
just fetch-pocket-tts english alba        # → weights/tts/pocket-tts-kyutai
just fetch-pocket-tts french estelle
```

The ungated [`kyutai/pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning)
has every language and the released voices. Its Mimi **encoder** is zeroed, so
cloning from a WAV needs the gated [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts)
(accept the terms, `hf auth login`); `PocketTts` refuses a zeroed encoder with
that message rather than producing garbage. The drifting model reuses
`languages/english_2026-09/tokenizer.*`.

`--root` accepts a checkout of either repo, a `languages/<name>` directory, or a
flat directory with one `*.safetensors` and `tokenizer.model` (the January mirror).

## Usage

```bash
rlx-pocket-tts say "Hello world." --root weights/tts/pocket-tts-kyutai --voice alba -o hello.wav
rlx-pocket-tts say "Bonjour." --root … --language french --voice estelle --device metal
rlx-pocket-tts export-voice me.wav --root <gated checkout> -o me.safetensors   # then --voice me.safetensors
```

```rust
use rlx_pocket_tts::PocketTts;
use rlx_runtime::Device;

let mut tts = PocketTts::open("weights/tts/pocket-tts-kyutai", "english", Device::Metal)?;
let voice = tts.load_voice("alba")?;            // or tts.voice_from_wav("me.wav", false)?
tts.generate("Hello, world.", &voice)?.write_wav("out.wav")?;
```

`tts.options` holds temperature, EOS threshold, sampler steps, noise clamp,
frames after EOS and seed (defaults from the config). `RLX_POCKET_TTS_TIMING=1`
prints per-chunk stage timings.

## How it runs

- **FlowLM step** (`graph::flow_lm`): one row through the backbone, then the EOS
  head and the sampler head (LSD / flow matching unrolled over its steps,
  drifting, pointwise). The KV cache is held on the host and fed as inputs of a
  bucketed capacity (multiples of 64), so one compiled graph serves a whole
  chunk. Prompt tokens go through the same graph one row at a time.
- **Mimi decoder** (`graph::mimi`): a streaming block graph (8 frames per call)
  with every causal conv's left context, every transposed conv's overlap tail
  and the sliding-window KV as explicit state, so one graph decodes any length.
  The encoder (voice cloning) is built the same way.
- **Voices** (`voice_state`): the released voices are the FlowLM KV cache after
  the voice prompt (`…self_attn/cache`, `…/offset`); `save_voice` writes the
  same format, which upstream `pocket_tts` loads too.
- **Text** (`text`): upstream's text preparation and token-level sentence
  chunking (50 tokens per chunk).

## Parity

`tests/reference_parity.rs` replays a dump from
`scripts/pocket_tts_reference.py` (same text, voice and per-frame sampler noise)
twice: **teacher-forced** (each frame fed the reference's previous latent — the
strict per-frame check) and **free-running** (frame count and audio).
`tests/clone_parity.rs` checks cloning on a checkpoint whose zeroed encoder was
filled with seeded random weights (`scripts/pocket_tts_clone_reference.py`).

| Model | Teacher-forced max\|Δ\| (backbone / latent) | Free-running | Devices |
|---|---|---|---|
| `english_2026-10` / alba | 3e-6 / 1e-5 | 72/72 frames, audio cos 0.99999 | CPU, Metal, MLX, wgpu |
| `english_7m_2026-10`, `english_2m_2026-10` / alba | ≤ 4e-6 | same frames, cos 1.000000 | CPU |
| `english_drifting_26-09` / alba | 2e-6 / 2e-5 | same frames, cos 0.95 | CPU |
| `french` / estelle | 4e-6 / 1e-5 | same frames, cos 0.9999 | CPU |
| Mimi decoder on reference latents | 5e-6 (cos 1.0000000) | | CPU, Metal, MLX, wgpu |
| Voice cloning (random encoder) | voice state rel 3e-6 | | CPU, Metal, MLX, wgpu |

Free running, the sampler feeds its own latents back and amplifies float noise:
the drifting head (one step, guidance baked in) turns a 2e-3 backbone difference
into a different but equally valid latent around frame 53, which is why the
strict check is the teacher-forced one.

```bash
.venv-pocket/bin/python -I scripts/pocket_tts_reference.py \
    --lang-dir weights/tts/pocket-tts-kyutai/languages/english_2026-10 \
    --language english_2026-10 --voice alba \
    --out weights/tts/pocket-tts-kyutai/reference/english_2026-10-alba
just test-pocket-tts-parity weights/tts/pocket-tts-kyutai/reference/english_2026-10-alba metal
```

Not yet run: CUDA, ROCm, Vulkan, CoreML (built, untested); cloning on the real
gated encoder weights; the 24-layer and 18M models (same code paths as the
tested ones).

## Legacy eager path

`TtsModel` (ndarray, Accelerate on macOS) is the earlier CPU-only port of the
January checkpoint, kept for its examples (`examples/generate.rs`,
`examples/whisper_check.rs`). `PocketTts` loads that checkpoint too
(`--language english_2026-01`).
