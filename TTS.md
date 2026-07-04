# RLX TTS models & voice chat

Index of the text-to-speech models ported to RLX, their reported throughput, and
the **Gemma 3 270M + Inflect-Nano** local voice-chat pairing built on top of them.

> Unlike [ASR.md](ASR.md), there is **no unified TTS benchmark harness** yet, so
> per-model numbers below are **as reported by each crate's own README/notes**
> (backend + hardware vary). Numbers explicitly marked *(verified here)* were
> measured this session on Apple Silicon / Metal against the local
> `gemma-3-270m` GGUF + `weights/inflect-nano-rlx` bundle.

## Models

| model | crate | ~params | best reported RTF | backend | notes |
|-------|-------|---------|-------------------|---------|-------|
| 🥇 **Inflect-Nano** | `rlx-inflect-nano` | ~4.6M | **48×** (reported) / ~10–13× *(verified here, Metal)* | MLX / Metal | FastSpeech-style acoustic + Snake HiFi-GAN vocoder, 24 kHz; fastest by far |
| **Pocket-TTS** | `rlx-pocket-tts` | ~100M | "faster than realtime" | CPU (Accelerate) | Kyutai FlowLM + Mimi codec |
| **Orpheus** | `rlx-orpheus` | 3B | ~22× warm (decode-bucket reuse) | Metal (native decode) | LLM-style; only whisper-OK Metal config is documented (see notes) |
| **Qwen3-TTS** | `rlx-qwen3-tts` | 0.6B | 1.05–1.7× | Metal | best speed/quality of the larger models |
| **Kitten-TTS** | `rlx-kittentts` / `kitten_tts_mini_rlx` | ~15M | "faster than ONNX on CPU" | CPU | tiny |
| **Tiny-TTS (VITS2/MeloTTS)** | `rlx-tiny-tts` | — | — | CPU/MLX/wgpu/CoreML | bit-exact across all 4 backends |
| **NeuTTS** | `rlx-neutts` | Nano/Air | — | — | no reported RTF |
| **Voxtral-TTS** | `rlx-voxtral-tts` | 4B | — | — | no reported RTF |
| **Kyutai-TTS** | `rlx-kyutai-tts` | 1.6B | — (scaffolding) | — | generation loop not wired yet |

## Leaderboard

| metric | winner | value |
|--------|--------|-------|
| ⚡ **Fastest** | **Inflect-Nano @ MLX** | **~48× realtime** (reported); ~10–13× on Metal *(verified here)* |
| 🪶 **Smallest** | **Inflect-Nano** | ~4.6M params |
| 🗣️ **Fastest at "large-model" quality** | **Qwen3-TTS @ Metal** | ~1.05–1.7× realtime (0.6B, production-wired) |

Inflect-Nano wins on raw speed and size (it's tiny); the larger models
(Qwen3-TTS 0.6B, Orpheus 3B, Voxtral-TTS 4B) trade speed for naturalness.

---

## Voice chat: Gemma 3 270M → Inflect-Nano (`rlx-gemma-inflect-nano`)

A fully-local, streaming **voice chat**: you type, `gemma3-270m` generates a reply
on the GPU, and `inflect-nano` speaks it back — the LLM and TTS both run on Metal.

- Crate + full docs: [`crates/rlx-gemma-inflect-nano/README.md`](crates/rlx-gemma-inflect-nano/README.md)
- Interactive REPL example: [`crates/rlx-gemma-inflect-nano/examples/chat.rs`](crates/rlx-gemma-inflect-nano/examples/chat.rs)
- One-shot (prompt → WAV) example: `examples/speak.rs`

### Run

```bash
cd /Users/Shared/rlx-models
cargo run --release --features metal -p rlx-gemma-inflect-nano --example chat -- \
  --device metal --tts-device metal
# then type; /reset clears history, /quit exits.
```

Defaults resolve `RLX_GEMMA3_GGUF` (else `/tmp/rlx-weights/gemma-3-270m.gguf`) and
`RLX_INFLECT_NANO_DATA` (else `weights/inflect-nano-rlx`). See the crate README for
all flags (`--speed`, `--sentence-pause`, `--prime-secs`, `--first-sentence`,
`--no-audio`, `--temp`, `--system`, …).

### Design

- **Pipelined**: sentences are split as the LLM streams them (on `.`/`!`/`?`/newline)
  and each is vocoded + queued for playback *immediately* — playback starts once
  ~4s is buffered and overlaps ongoing generation. You hear sentence 1 while Gemma
  writes sentence 2.
- **Pacing** (1.5× slower default) is applied at **synthesis** (Inflect `InferOpts::with_speed`,
  natural pitch), not at playback; the audio path only resamples 24 kHz → device rate.
- Live playback via `cpal` (streaming ring buffer, adapted from `rlx-moshi`), with a
  macOS `afplay` fallback when no output device is available.

### Verified performance (Metal, release, this session)

| stage | number | how |
|-------|--------|-----|
| gemma3-270m decode (warm) | **~32 tok/s**, bit-exact vs CPU (logit_cos 1.0) | `rlx-gemma` `examples/gemma_bench.rs` |
| gemma3-270m decode (chat, short reply) | ~9–17 tok/s (incl. per-turn prefill + incremental decode) | `chat --no-audio` |
| Inflect-Nano TTS (Metal, cached graph) | ~10–13× realtime | `chat` |
| first reply of a session | one-time prefill graph compile (a few seconds) | — |

### Engineering findings / fixes (this session)

1. **Metal multi-turn prefill NaN → garbage (FIXED).** Prompts past the first
   prefill bucket (≳30 tokens of history) padded to a power-of-two bucket, which set
   prefill **active-extent**, forcing Metal off the validated MPSGraph-hybrid path
   onto the per-op MSL thunk path — where a Gemma 3 Q4 `Op::Attention`→`o_proj`
   dequant **arena-aliasing** defect (task #50) zeroes attention output → all-NaN
   logits → silent CPU fallback → garbage. Short prompts (exact bucket) never hit it,
   so the 21-token parity test missed it. **Fix:** drop `Device::Metal` from
   `packed_prefill_active_extent_enabled` (`crates/rlx-models-core/src/autoregressive.rs`)
   so Metal prefill stays on the good path; keep pow2 bucketing in
   `crates/rlx-gemma/src/packed_session.rs::prefill_bucket_len_device` for graph reuse.
   Correct because logits are read from `last_token_idx = n−1` and KV is truncated to `n`.
2. **`*_auto` tokenizer reload gotcha (FIXED, ~5–7× chat speedup).** The `chat`
   streaming callback called `rlx_gemma::decode_token_auto` **per generated token**,
   and the `*_auto` helpers (`decode_ids_auto`/`encode_prompt_auto`) **reload
   `tokenizer.json` from disk on every call** (~370 ms/token). That — not the model —
   was the apparent "~2 tok/s"; real warmed decode is ~32 tok/s. **Fix:** load
   `tokenizers::Tokenizer::from_file` once and decode incrementally.
3. **Vocoder graph cache (~3× TTS throughput on streamed replies).** New
   `InflectNano::synthesize_on_cached` (`crates/rlx-inflect-nano/src/lib.rs`) buckets
   frame counts (mel padded to a multiple of 64, waveform trimmed) and reuses compiled
   vocoder graphs across sentences/turns instead of recompiling per length
   (~3.2× → ~10× realtime). `synthesize_on` stays bit-exact for existing callers.
4. **Pipelined streaming + 4 s prime buffer** to keep playback smooth (replaces
   the earlier per-comma micro-segmentation that caused choppiness).

### References

- Memory notes: `gemma3_270m_metal_prefill_nan.md`, `rlx_tokenizer_auto_reload_gotcha.md`,
  `inflect_nano_crate.md`, `orpheus_tts_perf.md`, `tiny_tts_backends.md`.
- Related backend gotchas: `gemma4_metal_q4_bugs.md`, `gemma_cuda_attention_scale.md` (task #50 family).
- Bench tool: `cargo run --release -p rlx-gemma --features metal --example gemma_bench`
  (per-backend prefill/decode timing + CPU parity).
