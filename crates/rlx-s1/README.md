# rlx-s1

**S1-mini** by [Superwhisper](https://superwhisper.com) ([`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini)) for RLX — a 0.6 B text normalizer that turns a raw ASR transcript into clean written text: fillers removed, false starts and self-corrections resolved to whatever the speaker landed on, punctuation and capitalization applied, and spoken numbers, dates, times, currency and email addresses rendered in written form.

```text
audio ──▶ ASR (Whisper, Parakeet, …) ──▶ S1-mini ──▶ clean text
```

S1-mini is a fine-tune of `Qwen/Qwen3-0.6B` and its `config.json` is byte-identical to the base model's — 28 layers, 16 Q / 8 KV heads, hidden 1024, RoPE θ 1e6, tied embeddings. **There is no new architecture here:** the forward pass is [`rlx-qwen3`](../rlx-qwen3), and every backend that already runs Qwen3 runs this one.

What this crate contributes is the input protocol, which is the part the model card spends most of its length on and the part that integrations get wrong:

| | |
|---|---|
| `prompt::SYSTEM_PROMPT` | The exact system prompt, verbatim. Re-wording it degrades the output. |
| `Controls` | The `[Styling: …] [Structure: …] [Context: …]` control line as three enums, so an untrained axis value is unrepresentable. |
| `EMPTY_THINK_BLOCK` | The `enable_thinking=False` assistant prefix `<think>\n\n</think>\n\n`. Omit it and the model emits an empty think block and stops — the single most common way to get a blank result. |
| Greedy decoding | Pinned, not configurable. `generation_config.json` ships `do_sample: false`. |
| `max_new_tokens` | Sized per call as `1.3 × prompt + 32` rather than a flat 1024, stopping on `<\|im_end\|>` / `<\|endoftext\|>`. |
| Chunking | Sentence-boundary splitting past the ~1,000-token design point, with a word-boundary fallback — raw ASR usually has no punctuation to break on. |

## Quick start

```bash
hf download superwhisper/s1-mini --local-dir ./s1-mini

cargo run --release -p rlx-s1 -- \
    --weights ./s1-mini \
    --transcript "so um i need to like send the the report by uh friday no wait make that thursday"
# So I need to send the report by Thursday.
```

```bash
# Steer with the control line; all 4 × 2 × 2 combinations were trained.
rlx-s1 --weights ./s1-mini --stdin --styling formal --structure lists --context email < notes.txt

# Inspect the exact wire format without loading any weights.
rlx-s1 --transcript "hello there" --show-prompt
```

GGUF builds from [`superwhisper/s1-mini-GGUF`](https://huggingface.co/superwhisper/s1-mini-GGUF) work too — pass the file or the directory (`--prefer-quant` picks the quant, default `q4_k_m`); the tokenizer falls back to the GGUF-embedded vocab when no `tokenizer.json` sits next to it.

## API

```rust
use rlx_s1::{Controls, S1Runner, Styling, Structure, Context};
use rlx_runtime::Device;

let mut s1 = S1Runner::builder()
    .weights("./s1-mini")       // HF dir, .safetensors, or .gguf
    .device(Device::Metal)
    .build()?;

let raw = "so um i need to like send the the report by uh friday no wait make that thursday";
assert_eq!(s1.normalize(raw)?, "I need to send the report by Thursday.");

let email = Controls::new()
    .styling(Styling::Formal)
    .structure(Structure::Prose)
    .context(Context::Email);
let _ = s1.normalize_with(raw, email)?;
# anyhow::Ok(())
```

- `normalize` / `normalize_with` — chunk if needed, normalize, join.
- `normalize_chunks_with` — one cleaned string per chunk, join it yourself.
- `normalize_pass` — a single pass with a per-token callback, no chunking.
- `encode_prompt` / `chunk` / `count_tokens` — the pieces, for pipelines that want to batch or budget themselves.

An **empty result is a valid answer**: filler-only input ("um") normalizes to the empty string by design, and a pipeline should treat that as success.

## Parity

`tests/hf_parity.rs` checks three layers against `transformers`, using the dump from `scripts/s1_hf_reference.py`:

1. **Prompt string** vs `apply_chat_template(..., enable_thinking=False)` — byte-for-byte.
2. **Prompt ids** vs the HF tokenizer — id-for-id.
3. **Greedy completions** vs `generate(do_sample=False)` — token-for-token.

All 11 cases (the model card's worked examples plus one per control axis) pass at all three layers.

```bash
python3 scripts/s1_hf_reference.py --weights ./s1-mini --dtype float32 --out /tmp/s1_reference.json
S1_REFERENCE=/tmp/s1_reference.json S1_WEIGHTS=./s1-mini \
    cargo test --release -p rlx-s1 --test hf_parity -- --nocapture
```

> Generate the reference in **float32**, not the checkpoint's bf16. RLX computes in f32, and the two differ on near-tie argmaxes — bf16 drops the comma in `The invoice came to $23,450, and it's due on March 3, 2026.` and turns the `Structure: lists` example back into prose. The f32 reference is the one that matches both RLX and the model card's own printed outputs.

`tests/backends.rs` re-runs four pinned cases on every enabled backend. **CPU, Metal, MLX, wgpu and CoreML/ANE all agree**, in a single process (which is how two backend bugs surfaced: `rlx-qwen3` leaking Metal-only env flags into the MLX runner, and wgpu reading zeros for any F32 LM whose arena crosses the 4 GiB storage-bind cap — both fixed, see the CHANGELOG).

## Performance

Four utterances of *different* lengths, each run twice, on an M4 Pro / Metal with the BF16 safetensors (`cargo run --release -p rlx-s1 --features metal --example s1_bench`):

| prompt tokens | bucketing **off** (1st / 2nd) | bucketing **on** (1st / 2nd) |
|--:|---|---|
| 97 | 7.31 s / 0.20 s | 8.98 s / 0.21 s |
| 89 | 3.09 s / 0.17 s | **0.17 s** / 0.17 s |
| 94 | 3.20 s / 0.19 s | **0.20 s** / 0.20 s |
| 88 | 2.94 s / 0.17 s | **0.19 s** / 0.19 s |

Steady state is ~0.2 s per utterance either way (~50 tok/s); the difference is what a *new* prompt length costs. `rlx-qwen3`'s prefill compile cache keys on the **exact** `(batch, seq)`, so with bucketing off every fresh length pays a ~3 s graph compile for a 0.2 s forward pass — and a dictation pipeline sees a fresh length almost every time.

**Prefill bucketing** (on by default, `--prefill-bucket`, default 64) rounds the prompt up to the next multiple of 64, right-pads `input_ids`, and gathers the LM-head row through a `last_token_idx` input instead of a baked `seq - 1`, so one compiled graph covers 64 consecutive lengths — above, all four utterances land in the same bucket and only the first compiles. Causal masking keeps the pad columns out of every real row and the pad KV rows are trimmed before they reach the cache; the 11-case HF parity suite is token-identical with bucketing on. It is numerically *equivalent*, not bit-identical — the padded run is a wider GEMM and reduces in a different order (~1e-7 relative). The cost is at most 63 padded positions of prefill (~30 ms here). `--prefill-bucket 0` restores exact lengths.

**No prefix cache.** Reusing a KV snapshot of the fixed ~60-token system prefix looks appealing, but the suffix it has to replay *is the transcript*, and the reuse path folds a suffix in one token at a time — measured 38.5 s vs 18.8 s on CPU for the same 10 output tokens. A single batched prefill wins for every transcript length, so this crate always does that.

## Model card notes worth keeping

- Send the system prompt **and** the control line, exactly as documented. They are the only steering mechanism and the model was never trained without them.
- Keep single passes under roughly 1,000 tokens; chunk longer transcripts at sentence boundaries (`S1Runner` does this for you).
- English only in v1.

## License

This crate is GPL-3.0-only like the rest of the workspace. The **weights** are not: S1-mini is Apache 2.0 (inherited from Qwen3-0.6B) plus one additional term — wherever it is used it must keep its name, "S1-mini" by "Superwhisper", with that exact capitalization. Read the upstream `LICENSE` before shipping it.
