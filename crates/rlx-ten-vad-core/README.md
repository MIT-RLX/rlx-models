# rlx-ten-vad-core

The embedded half of the TEN-VAD port: `no_std`, no allocator required for
inference, no floating point on the integer path.

`rlx-ten-vad` is the desktop and mobile build — it runs the model as an rlx
graph across seven backends. This crate is what goes on a microcontroller or
becomes an FPGA datapath. Both share this crate's DSP frontend, so there is one
implementation of the feature extraction rather than two that drift.

## What's here

| module | |
|---|---|
| `frontend`, `ooura`, `pitch`, `biquad` | the DSP: 40 log-mel bands + an LPC pitch feature per 16 ms hop |
| `net` | f32 scalar forward, allocation-free |
| `fixed` | integer-only forward — int16 weights, Q15 activations, no FP at all |
| `synth` | test-signal generation |

## Targets

Builds for bare metal with `--no-default-features`:

```bash
cargo build -p rlx-ten-vad-core --no-default-features \
  --target riscv32imc-unknown-none-elf      # ESP32-C3 / C6
cargo build -p rlx-ten-vad-core --no-default-features \
  --target thumbv7em-none-eabihf            # Cortex-M4F / M7
```

## No allocator

Nothing on the inference path allocates. `Frontend`, `Net` and `FixedNet` hold
fixed-size arrays, so `--no-default-features` builds with no `alloc` at all —
no global allocator to provide, nothing to fragment, and no allocation that can
fail mid-frame. Only `synth` (test-signal generation) needs one, and it is
behind the `alloc` feature.

The structs are correspondingly larger than when the same buffers lived in
`Vec`s. That is the same memory, moved: total usage is lower, because there is
no allocator overhead and no leaked reallocation. Put `Vad` in a `static` —
33 kB is more than a default MCU task stack.

With `std` (the default) the transcendentals go through `std`'s, which keeps
the frontend bit-identical to the reference DSP on arm64 macOS. Without it they
go through `libm`, which is portable but not bit-identical — see
`math.rs` for why that trade is deliberate.

## Footprint

| | |
|---|---|
| `Vad` (f32 path) | 33,632 B, all inline |
| `FixedNet` | 1,648 B, all inline |
| f32 weights | 305 kB flash |
| int16 weights | 146.5 kB flash |

## The integer path

`fixed::FixedNet` exists for two reasons: FPU-less MCUs, and to be the bit-exact
golden model the FPGA RTL is checked against.

Measured on an `rv32imc` core (QEMU `virt`, same ISA as an ESP32-C3), counting
instructions rather than `mcycle` — that counter is wall-clock-derived and
moves with host load, while instruction counts reproduce to 2e-8. See
`rlx-ten-vad-mcu/README.md` for the recipe.

| | `rv32imc` (no FPU) | `rv32imafc` (FPU) |
|---|---|---|
| `fixed::FixedNet` | 926,150 | 926,146 |
| `net::Net` (f32) | 9,873,148 | **820,360** |

Without an FPU the integer path is **10.7× cheaper**. With one it is slightly
*dearer* — `fixed` pays for i64 accumulation on a 32-bit core. Use `Net` on an
FPU part; `FixedNet` is for FPU-less cores and for being the FPGA's golden
model.

Accuracy against the published ONNX model over the 250-frame reference clip:
**max \|Δ\| 3.7e-4, cosine distance 2.0e-8, zero decision flips**. The shipped
Agora binary sits at 9.6e-4 and 9.4e-8 — so the integer path is 4.7× closer to
the reference in cosine than the vendor's own build.

Regenerate the integer artifacts after changing weights or format:

```bash
cargo run -p rlx-ten-vad --release --example gen_fixed_tables
```

That uses `rlx_fpga::seq::quantise_pow2` — the same quantiser the RTL exporter
uses — so the MCU blob and the FPGA weight image are identical integers by
construction.

## Related

* `rlx-ten-vad` — desktop/mobile, rlx graph, seven backends
* `rlx-ten-vad-mcu` — bare-metal firmware built on this crate
* `rlx-ten-vad-fpga` — RTL exported from the same rlx-ir graph
