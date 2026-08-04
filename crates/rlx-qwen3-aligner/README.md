# rlx-qwen3-aligner

**Qwen3 forced aligner** on RLX, native Rust. Given audio + a *known* transcript,
produce per-token timestamps: the acoustic encoder emits per-frame token
log-probabilities, and a monotonic **Viterbi forced alignment** maps the target
token sequence onto frames.

| Component | Reuse |
|-----------|-------|
| Acoustic encoder | `rlx-qwen3-asr` (already multi-backend) |
| Alignment DP | this crate |

## What's here (checkpoint-free, tested)

- `Qwen3AlignerConfig` — encoder dims + vocab + `frame_rate` (+ `frame_to_seconds`).
- `forced_align(log_probs, targets)` — monotonic Viterbi: each token consumes ≥1
  consecutive frame, path advances through tokens in order, every frame assigned;
  returns each token's `[start_frame, end_frame)` span.

4 CPU smoke tests: frame→seconds, monotonic alignment, contiguous full-coverage
spans, and rejects (more tokens than frames / empty / out-of-vocab).

## Next step

Wire the Qwen3 encoder + token-emission head (reuse `rlx-qwen3-asr`) to produce the
`log_probs`, then per-backend parity. Needs a Qwen3-aligner checkpoint.
