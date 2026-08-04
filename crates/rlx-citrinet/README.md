# rlx-citrinet

**Citrinet** ASR (NVIDIA NeMo) on RLX, native Rust. A 1D **time-channel separable
convolution** network with **squeeze-and-excitation** blocks over mel features,
trained with **CTC**. Decoding is CTC greedy: argmax per frame, collapse repeats,
drop blanks.

| Component | Reuse |
|-----------|-------|
| TCS-conv + SE stack | `rlx-conformer-ctc` conv modules |
| `.nemo` loader | `rlx-nemo` |
| CTC decode | this crate (reusable by any CTC model) |

## What's here (checkpoint-free, tested)

- `CitrinetConfig` — feature dim + channels (512/1024) + blocks + SE reduction +
  subsampling + vocab + blank id; `num_classes` (vocab + 1).
- `ctc_greedy_decode` / `ctc_greedy_decode_logits` — CTC greedy: collapse consecutive
  repeats, drop the blank (from ids or straight from logits via argmax).

3 CPU smoke tests: config/classes, CTC collapse+blank rules, logits→argmax→decode.

## Next step

Wire the TCS-conv + SE encoder + CTC head (reuse `rlx-conformer-ctc`) and the
`rlx-nemo` weight loader, then per-backend parity. Needs a Citrinet checkpoint.
