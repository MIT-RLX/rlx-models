# rlx-fish

**Fish-Speech** (Fish Audio) TTS / voice cloning on RLX — native Rust. A **dual-AR**
model: a slow Llama-style backbone emits one semantic step per audio frame, and a
small **fast (depth) transformer** autoregressively emits that frame's
`num_codebooks` acoustic codes; a **Firefly-GAN** codec decodes the
`[frames, num_codebooks]` code matrix to a waveform.

| Component | Reuse |
|-----------|-------|
| Slow backbone + fast transformer | both Llama-style → `rlx-llama32` |
| Firefly codec (VQ + GAN) | conv/GAN vocoder stack (`rlx-neutts` BigVGAN / `rlx-dac`) |

## What's here (checkpoint-free, tested)

- `FishConfig` / `FireflyConfig` — dual-AR backbone + fast transformer dims + codec
  (num_codebooks 8, codebook_size 1024, hop 512, 44.1 kHz).
- `codebook_matrix` / `flatten_codebook_matrix` — the bridge between the fast
  transformer's flat frame-major stream and the codec's per-frame codebook rows.
- `validate_codes` — width + range checks on a code matrix.

5 CPU smoke tests: config/validate, frame rate, codebook-matrix round-trip, ragged
stream rejection, code validation.

## Next step

Wire the slow backbone + fast/depth transformer (dual-AR sampling loop over
`rlx-llama32`) and the Firefly decoder for end-to-end synthesis, then per-backend
parity. Needs a Fish-Speech checkpoint.
