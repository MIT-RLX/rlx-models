# rlx-f0

Short-time autocorrelation fundamental frequency (F0) for on-device speech — no model weights.

Lifted from the translator auto-voice path for reuse (gender banding, TEN-VAD secondary signals, etc.).

```rust
use rlx_f0::{estimate_f0_hz, gender_from_f0, SpeechGender};

let f0 = estimate_f0_hz(&pcm, 16_000);
let g = gender_from_f0(f0, 165.0, 145.0);
```

Backend: pure Rust CPU (no RLX graph). Suitable for iOS.

Neural pitch (SwiftF0 / CREPE-class ONNX) is intentionally out of scope here —
wrap weights later behind the same `estimate_f0_hz` surface if needed.
