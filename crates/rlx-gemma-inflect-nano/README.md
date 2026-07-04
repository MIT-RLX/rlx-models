# rlx-gemma-inflect-nano

Unpublished workspace crate (`publish = false`) that pairs **Gemma 3 270M IT** (packed GGUF chat) with **Inflect-Nano** English TTS.

Pipeline: user text → Gemma greedy decode → spoken WAV (24 kHz).

## Prerequisites

```bash
just fetch-gemma3-270m
# One-time Inflect export (see crates/rlx-inflect-nano/README.md):
python3 scripts/export_inflect_nano.py --repo /tmp/inflect-nano --out weights/inflect-nano-rlx
```

Override paths with `RLX_GEMMA3_GGUF` and `RLX_INFLECT_NANO_DATA`.

## Run

```bash
just gemma-inflect-speak -- --user "What is the capital of France?"
```

Or directly:

```bash
cargo run --release -p rlx-gemma-inflect-nano --features apple-silicon \
  --example speak -- \
  --device metal --tts-device auto \
  --user "Hello from Gemma and Inflect." \
  --out /tmp/gemma-inflect-speak.wav
```

By default only the **first sentence** of the model reply is spoken (`--full-reply` for the entire answer).

## Chat (interactive voice REPL)

The `chat` example is a live loop: type a message, watch Gemma stream its reply,
then hear it spoken back through your speakers. Conversation history is kept
across turns, so Gemma remembers what was said.

Playback is **pipelined**: as the LLM produces each sentence (split on `.`/`!`/`?`
or newline), that sentence is vocoded on the GPU and pushed into a live output
buffer *immediately* — it does not wait for the full reply. Playback starts once
~4s of audio is queued, then keeps draining as later sentences arrive. A short
silence follows each sentence so the speech breathes. Speaking rate defaults to
`0.667` (≈1.5× slower) and is applied at **synthesis** (acoustic durations,
natural pitch); playback does no rate change, only resampling to the device rate.

```bash
cargo run --release -p rlx-gemma-inflect-nano --features metal \
  --example chat -- \
  --device metal --tts-device metal \
  --system "You are a concise, friendly assistant."
```

```
you> give me three short tips for staying healthy.
gemma> Here are three short tips for staying healthy:
1.  Eat a balanced diet — fruits, vegetables, lean proteins, whole grains.
2.  Get enough sleep — aim for 7–9 hours a night.
3.  Stay hydrated — drink plenty of water.
[gemma] 72 tok in 7.16s (10.1 tok/s)     # each sentence is spoken as it completes
you> /quit
```

(Metal, release, warm. First reply of a session also pays a one-time graph
compile; TTS runs at ~10–13× realtime and overlaps generation, so you hear
sentence 1 while Gemma is still writing sentence 2.)

In-session commands: `/reset` clears the conversation, `/quit` (or Ctrl-D) exits.
Live playback uses `cpal` (falls back to macOS `afplay` if no output device).
Useful flags:

| flag | default | effect |
|------|---------|--------|
| `--speed F` | `0.667` | speaking rate, applied at synthesis (`<1` slower, `>1` faster) |
| `--sentence-pause F` | `0.45` | silence after each spoken sentence (seconds) |
| `--prime-secs F` | `4.0` | audio buffered before playback starts |
| `--first-sentence` | off | speak only the first sentence of each reply |
| `--no-audio` | off | print replies without synthesis/playback |
| `--temp F` | `0` | sampling temperature (`0` = greedy) |

### Notes

- **Multi-turn on Metal**: earlier, prompts past the first prefill bucket
  (≳30 tokens of history) hit a Metal prefill-NaN and silently fell back to CPU
  garbage. Fixed in `rlx-models-core` (`autoregressive.rs`) by dropping `Device::Metal`
  from packed-prefill **active-extent** — that keeps Metal prefill on the validated
  MPSGraph-hybrid path instead of the per-op MSL thunk path that trips a Gemma 3 Q4
  `Op::Attention`→`o_proj` dequant arena-aliasing defect (task #50). `rlx-gemma`
  keeps power-of-two prefill bucketing (`packed_session.rs::prefill_bucket_len_device`)
  so compiled prefill graphs are reused across a prompt-length range.
- **Vocoder graph cache**: `chat` synthesizes each segment via
  `InflectNano::synthesize_on_cached`, which buckets frame counts and reuses
  compiled vocoder graphs across segments/turns — recovering ~3× TTS throughput
  versus recompiling per segment (e.g. 3.2× → ~10× realtime on a multi-clause reply).
- **LM decode speed**: gemma3-270m packed decode on Metal is ~32 tok/s warmed
  (bench); the chat sees ~9–17 tok/s for short replies once per-turn prefill and
  incremental decode are included. The earlier "~2 tok/s" was **not** the model —
  the streaming callback was calling `decode_token_auto` per token, and the `_auto`
  helpers reload `tokenizer.json` from disk every call (~370ms/token). `chat` now
  loads the tokenizer once (`tokenizers::Tokenizer::from_file`) and decodes
  incrementally, a ~5–7× speedup. First use of a new pow2 prompt-length bucket
  still pays a one-time prefill compile (~a few seconds); same-bucket turns reuse it.
