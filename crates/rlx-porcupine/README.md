# rlx-porcupine

Porcupine-style wake detector on RLX. Train custom phrases in RLX:

```bash
just wake-train-cnn -- --pos positives --neg negatives --keyword porcupine \
  --out crates/rlx-porcupine/weights/model.safetensors

just porcupine-demo -- --wav clip.wav \
  --weights crates/rlx-porcupine/weights/model.safetensors
```
