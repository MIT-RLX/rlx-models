# rlx-openwakeword

Native RLX port of the openWakeWord streaming pipeline + **RLX-only phrase-head training**.

## Custom wake word (RLX train)

```bash
# positives/*.wav  negatives/*.wav  (≥ ~1.5s each recommended)
cargo run -p rlx-openwakeword --bin rlx-openwakeword-train --release -- \
  --pos positives --neg negatives --keyword "hey rlx" \
  --out-dir crates/rlx-openwakeword/weights --epochs 40

just openwakeword-demo -- --wav clip.wav \
  --weights crates/rlx-openwakeword/weights --keyword "hey rlx"
```

Embedding stays frozen; only the phrase DNN is trained (SGD on `rlx-cpu`).

## Inference

```bash
just fetch-openwakeword
just openwakeword-demo -- --wav clip.wav --device cpu
cargo test -p rlx-openwakeword --release
```

Optional `--features onnx` is for parity against upstream `.onnx` only — not for training.
