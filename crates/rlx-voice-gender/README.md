# rlx-voice-gender

On-device speech gender for TTS bank selection.

| Mode | Feature | Weights |
|------|---------|---------|
| ACF F0 bands (default) | `f0` | none — [`rlx-f0`](../rlx-f0) |
| ECAPA ONNX | `onnx` | JaesungHuh / Alice-Sabrina-Ivy q8 ONNX via `TinyModel` |

```rust
use rlx_voice_gender::{GenderEstimator, GenderEstimate};

let est = GenderEstimator::acf_default();
let g = est.estimate(&pcm, 16_000);
```

SwiftF0 (neural pitch) is **not** bundled — use [`rlx-f0`](../rlx-f0) for the iOS ship path; optional SwiftF0 ONNX can wrap the same `estimate` API later.
