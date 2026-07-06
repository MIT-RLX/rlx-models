# rlx-diarize

Native RLX **speaker diarization**: slide a window over mono PCM, embed each window, then agglomeratively cluster the embeddings into speaker turns. Pure Rust (only `anyhow` + `serde`), no external model runtime — used by [`rlx-whisper`](../rlx-whisper)'s `diarize` feature to attach speaker labels to a transcript.

## Public API

```rust
use rlx_diarize::{DiarizeSession, DiarizeConfig, SpeakerTurn};

let mut session = DiarizeSession::new(DiarizeConfig::default());
let turns: Vec<SpeakerTurn> = session.diarize(&pcm_16k)?;   // 16 kHz mono
for t in &turns {
    // t.speaker_id, t.start (s), t.end (s)
    println!("speaker {} : {:.2}–{:.2}", t.speaker_id, t.start, t.end);
}
# anyhow::Ok(())
```

- `DiarizeConfig { window_sec, hop_sec, cluster_threshold }` — defaults `1.5 s` / `0.75 s` / `0.25`.
- `DiarizeSession::new(cfg).diarize(pcm)` — sliding-window embeddings (`embed` module) → agglomerative clustering (`cluster` module) → merged contiguous `SpeakerTurn`s.
- `SpeakerTurn { speaker_id, start, end }` — a contiguous span for one speaker (serde-serializable).

Very short inputs (below half a window) collapse to a single turn covering the whole clip.

## How it fits

`rlx-whisper` calls `DiarizeSession::diarize` to label its segment/word timeline with speakers. See the `diarize` stage in [`rlx-whisper/src/diarize.rs`](../rlx-whisper/src/diarize.rs).

## Tests

```bash
cargo test -p rlx-diarize --release
```
