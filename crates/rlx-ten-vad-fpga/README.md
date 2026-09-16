# rlx-ten-vad-fpga

> Cross-target performance for every deployment of this model — MCU, FPGA and
> the accuracy each costs — is consolidated in
> [`docs/ten-vad-embedded.md`](../../docs/ten-vad-embedded.md). This file covers
> the RTL export itself.

TEN-VAD as an FPGA datapath, exported from the same rlx-ir graph the runtime
executes.

There is no hand-written model logic here. `rlx-fpga`'s sequential target
lowers the graph to a descriptor ROM plus two fixed SystemVerilog modules, so
retraining the network re-exports the hardware with no Verilog edits:

```bash
cargo run -p rlx-ten-vad --example export_rtl      # -> rtl/
```

## Verification

The RTL is checked bit-for-bit against `rlx_ten_vad_core::fixed`, which the
parity suite in turn ties to the published ONNX model. So "the hardware is
correct" reduces to a chain of equalities rather than a tolerance:

```bash
cargo run -p rlx-ten-vad --example dump_fpga_golden   # golden vectors
iverilog -g2012 -I rtl_ir -o sim rtl/tv_lut.sv rtl/tv_core.sv tb/tv_tb.sv
(cd rtl_ir && vvp ../sim +frames=250)
```

```
frames        250
mismatches    0
cycles/frame  173140
RESULT PASS — RTL is bit-exact against the Rust fixed-point net
```

## Architecture

One MAC, one activation LUT, one descriptor ROM. Every stage of this network —
convolution, pooling, matmul, LSTM gating — is "bias, then a dot product over
an address pattern, then requantise and activate", so a single datapath walks
all of them. Recurrence needs no special support: LSTM state is a tensor in the
activation RAM like any other, and the engine updates it in place.

Two consequences worth knowing:

* Addresses come from running accumulators, not multipliers, so the address
  path is adders only.
* Padding, flattening and transposition cost nothing. The graph expresses
  padding as `Concat` with zero params, which becomes a reserved range in a
  zero-initialised RAM; the flatten before the first LSTM is folded into the
  *producing* stage's write strides.

At 79,295 MACs per 16 ms frame the design is deliberately serial — 2 cycles per
MAC, so 62.5 fps needs **10.82 MHz**. One MAC per cycle is a mechanical change
(present tap *i+1*'s address while accumulating tap *i*) if the headroom is ever
needed.

### Resources

`yosys synth_ecp5`:

| | count |
|---|---|
| LUT4 | 4,168 |
| TRELLIS_FF | 893 |
| DP16KD (18 kbit BRAM) | 70 → 1.26 Mbit |
| MULT18X18D | 13 |

Re-measured after the banded activation-RAM allocation; an earlier revision of
this table read 4,067 / 891 / 74, from before that change.

Fits an LFE5U-45F or an XC7A35T. The weight ROM is 1.2 Mbit of the 1.26, so
weight width is the only thing standing between this and a much smaller part.

## Numeric format

Chosen by measurement, not taste.

| | format | why |
|---|---|---|
| weights | `i16`, per-tensor power-of-two scale | see the table below |
| activations | `i32`, Q15 | `z1` reaches 177, so 8 integer bits are live |
| accumulator | 48-bit | 144 terms of Q15×Q15 needs 44 |
| sigmoid/tanh | 1025-point Q15 LUT over [0, 16] | clamping at 8 costs 6.9e-3 |

Against the published ONNX model over the 250-frame reference clip:
**max |Δ| 3.7e-4, cosine distance 2.0e-8, zero decision flips**.

The shipped Agora binary sits at max |Δ| 9.6e-4 and cosine distance 9.4e-8 from
that same model — so this quantised datapath is **4.7× closer to the reference
in cosine than the vendor's own build**, despite running int16 weights on one
MAC. Both metrics are reported because they fail differently: `max` catches a
single bad frame, cosine catches a systematic tilt that small per-frame
differences would hide.

### Why not fewer bits

Post-training quantisation hits a wall at int8 on this network. Measured on the
same clip, best scale choice per scheme:

| scheme | bits/weight | size | max \|Δ\| | decision flips |
|---|---|---|---|---|
| int16, per-tensor pow2 | 16 | 146 kB | 2.0e-4 | **0** |
| int8, block-32 float scale | 8.5 | 78 kB | 1.2e-2 | 3 |
| int8, per-row | 8 | 73 kB | 1.1e-2 | 5 |
| int6, block-32 | 6.25 | 57 kB | 8.4e-2 | 29 |
| **fp4 (E2M1), block-32 = MXFP4** | 4.25 | 39 kB | 1.6e-1 | 28 |
| int4, block-32 | 4.25 | 39 kB | 2.6e-1 | 47 |
| ternary (TWN, per-row) | ~2 | 18 kB | 1.4e-1 … 7.6e-1 | 20–114 |

FP4 genuinely beats int4 — 1.6× lower error and 40% fewer flips, because the
exponent absorbs this model's wide per-tensor dynamic range. It is still
unusable. TEN-VAD is 75 k parameters of already-distilled network with no
redundancy to spend, which is the opposite of the regime where 1-bit and
ternary work.

Getting below 8 bits therefore needs quantisation-*aware* training, not a
better rounding rule.

### What QAT recovers

`cargo run -p rlx-ten-vad --release --example qat -- --format fp4` distils the
f32 model into a student whose weights are fake-quantised in the forward pass,
with gradients taken at the quantised point and applied to f32 masters. Teacher
targets are recomputed per window from the same zero state the student starts
from, so the two see identical context.

Held-out windows (2,304 scored frames, disjoint from the windows used to pick
the checkpoint), against the f32 teacher:

| format | PTQ flips | QAT flips | PTQ 1−cos | QAT 1−cos |
|---|---|---|---|---|
| none *(control)* | 0 | 1 | 0 | 7.1e-9 |
| int8, block-32 | 10 | 7 | 6.7e-5 | 4.3e-5 |
| int6, block-32 | 90 | 84 | 6.7e-3 | 3.8e-3 |
| int4, block-32 | 282 | 163 | 1.7e-2 | 1.7e-2 |
| **MXFP4** | **143** | **97** | **1.1e-2** | **7.5e-3** |

On the 250-frame reference clip, MXFP4 goes from 26 decision flips to **21**,
and cosine distance from 3.9e-3 to 2.4e-3.

Three things this says:

* **QAT is real here** — MXFP4 loses a third of its errors, and int4 nearly
  half. The `none` control stays at 0–1 flips, so the training loop is not what
  moves the model.
* **It does not rescue 4 bits.** 97 flips out of 2,304 is a 4.2% disagreement
  rate where int16 is 0. A VAD that changes its mind on one frame in 24 is not
  the same product.
* **FP4 beats int4 after training too**, and by more than before it (97 vs 163).
  The exponent is doing real work on this weight distribution, not just
  flattering the post-training number.

So the int16 datapath above stands. The honest use for this result is the
opposite direction: it says how much headroom a *larger* VAD would have, and it
gives a working QAT harness for the next model that is big enough to spend it.

### Layer sensitivity

If you do retrain, this is where the bits matter. Quantising one tensor to int8
and leaving the rest in f32:

| most sensitive | max \|Δ\| | least sensitive | max \|Δ\| |
|---|---|---|---|
| `lstm1.weight_ih` | 2.4e-2 | `dense1.bias` | 4.4e-4 |
| `sep1.pointwise.weight` | 2.2e-2 | `conv0.bias` | 1.4e-3 |
| `lstm1.weight_hh` | 1.4e-2 | `dense2.weight` | 1.9e-3 |
| `sep1.depthwise.weight` | 1.1e-2 | `lstm2.weight_hh` | 3.4e-3 |

Layer 1 dominates: `lstm1.weight_ih` is 7× more sensitive than `lstm2.weight_hh`
despite being the same kind of tensor. Mixed precision does not help as a
post-training move — a greedy search that demotes tensors to int8 while holding
zero flips recovers 0.0 kB, because only the bias vectors qualify.

## Files

| | |
|---|---|
| `rtl/` | generated: descriptor ROM, `.mem` images, defs, and the two fixed modules |
| `tb/tv_tb.sv` | self-checking testbench |
| `tb/golden_*.mem` | vectors from `rlx_ten_vad_core::fixed` |
