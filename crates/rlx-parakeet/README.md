# rlx-parakeet

NVIDIA **Parakeet-TDT** on RLX — a FastConformer acoustic encoder feeding a
**Token-and-Duration Transducer** (TDT). Parakeet-TDT shares nearly its whole
stack with Nemotron ASR; this crate is mostly composition:

| Component | Source |
|-----------|--------|
| FastConformer encoder | reused from `rlx-nemotron-asr::encoder` |
| LSTM prediction network | reused from `rlx-nemotron-asr::decoder` (`PredictionNet`, `LstmCell`) |
| TDT greedy decode loop | reused from `rlx-audio-blocks::decoders::tdt` |
| **TDT joint** (token + duration heads) | **new** — [`joint::TdtJoint`] |
| **TDT core** (predictor ↔ joint binding) | **new** — [`transducer::TdtCore`] |

The only real difference from RNN-T is the joint's second **duration head**: the
classifier emits `n_classes + num_durations` logits, and decoding advances the
time index by `durations[argmax(duration_head)]` instead of always by one frame.

```rust
use rlx_parakeet::{TdtJoint, tdt_greedy_decode};

// enc: [frames, d_model] acoustic encoder output; durations e.g. [0,1,2,3,4]
let out = tdt_greedy_decode(&pred, &joint, &enc, &durations, max_symbols_per_step)?;
// out.token_ids / out.token_timestamps / out.token_durations
```

## Status

Arch + CPU smoke. Transducer head, TDT core, and greedy decode are implemented and
unit-tested against a synthetic model (skip-by-duration, blank consumption,
zero-duration symbol cap, shape checks).

**Next:** wire the real FastConformer encoder graph (builder already lives in
`rlx-nemotron-asr`) and `.nemo` weight loading for an end-to-end `transcribe`, then
real-weight parity against NeMo.
