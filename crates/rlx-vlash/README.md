# rlx-vlash

Native [RLX](https://crates.io/crates/rlx-runtime) port of the
[VLASH](https://github.com/mit-han-lab/vlash) **π₀** and **π₀.₅**
Vision-Language-Action robot policies.

Both policies pair a **PaliGemma** backbone (SigLIP-So400m/14 @224 vision tower
+ a Gemma-2B text model) with a **Gemma-300M action expert**. The two Gemma
stacks run through 18 *joint* transformer layers that share one attention over
the concatenated `[image ++ text ++ suffix]` sequence, and actions are produced
by **flow matching** (a short Euler integration of a learned velocity field).

- **π₀** — state is a suffix token; time is fused into the action embeddings;
  standard Gemma RMSNorm.
- **π₀.₅** — action-only suffix; state + time drive **adaptive RMSNorm**
  (adaRMS) in the action expert.

Weights load from the published `lerobot/pi0_base` / `lerobot/pi05_base`
checkpoints (bf16 `model.safetensors`, OpenPI naming; remapped by
[`weights::canonical_key`]).

## Status

Arch + weight loader + host preprocessing/normalization/tokenization landed;
CPU parity against the Python reference is validated stage-by-stage via
`tests/parity.rs` (fixtures produced by `scripts/vlash_ref_dump.py`).

## License

GPL-3.0-only, matching the RLX workspace. The upstream VLASH project is
Apache-2.0.
