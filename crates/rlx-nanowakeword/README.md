# rlx-nanowakeword

Native RLX CNN wake-word detector. **Train custom words with `rlx-wake-train`** (no Python trainer).

```bash
cargo run -p rlx-wake --bin rlx-wake-train --release -- cnn \
  --pos positives --neg negatives --keyword "hey nano" \
  --out crates/rlx-nanowakeword/weights/hey_nano.safetensors

just nanowakeword-demo -- --wav clip.wav \
  --weights crates/rlx-nanowakeword/weights/hey_nano.safetensors --lite
```
