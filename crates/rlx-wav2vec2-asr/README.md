# rlx-wav2vec2-asr

Classic **Wav2Vec2 + CTC forced alignment** for RLX — a WhisperX-style port that aligns known transcript text to 16 kHz mono PCM and returns per-word start/end times. This is the CTC forced-alignment path (distinct from the Conformer encoder in [`rlx-wav2vec2-bert`](../rlx-wav2vec2-bert)); it is consumed by [`rlx-whisper`](../rlx-whisper)'s `word-w2v` word-alignment mode.

This crate has no rlx-* dependencies (just `anyhow` + `serde`): it holds the CTC trellis, word aggregation, and the language→model registry. The neural acoustic forward is a scaffold — `AlignSession` accepts a `weights_path` but the current trellis is driven by a monotonic synthetic-CTC assignment.

## Public API

```rust
use rlx_wav2vec2_asr::{AlignSession, AlignedWord, align_model_for_language};

let mut session = AlignSession::new();
let words: Vec<AlignedWord> = session.align_text(&pcm_16k, "hello world", "en")?;
for w in &words {
    // w.text, w.start (s), w.end (s), w.score
    println!("{:>8.2}–{:<8.2} {}", w.start, w.end, w.text);
}

// Language code → HuggingFace Wav2Vec2 CTC checkpoint (WhisperX registry subset)
if let Some(spec) = align_model_for_language("en") {
    println!("{} / {}", spec.hf_repo, spec.config_file);
}
# anyhow::Ok(())
```

- `AlignSession::new()` / `AlignSession::with_weights(path)` — build an aligner; `align_text(pcm, text, language)` returns `Vec<AlignedWord>` (times relative to the PCM slice start).
- `AlignedWord { text, start, end, score }` — word span in seconds.
- `align_model_for_language(lang)` — maps `en`/`de`/`fr`/`es` to a HF Wav2Vec2 CTC repo (`Wav2Vec2AsrConfig` / `AlignModelSpec`).

## How it fits

`rlx-whisper` calls this crate for WhisperX-style word timestamps (`WordAlignMode::Wav2Vec2`), as an alternative to its own cross-attention DTW alignment.

## Tests

```bash
cargo test -p rlx-wav2vec2-asr --release
```
