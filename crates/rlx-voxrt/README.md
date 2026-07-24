# rlx-voxrt

VoxRT-style wake detector on RLX. Train custom phrases in RLX:

```bash
just wake-train-cnn -- --pos positives --neg negatives --keyword "hey assistant" \
  --out crates/rlx-voxrt/weights/model.safetensors

just voxrt-demo -- --wav clip.wav \
  --weights crates/rlx-voxrt/weights/model.safetensors
```
