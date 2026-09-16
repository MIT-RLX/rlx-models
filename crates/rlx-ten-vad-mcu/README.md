# rlx-ten-vad-mcu

> Cross-target performance for every deployment of this model — MCU, FPGA and
> the accuracy each costs — is consolidated in
> [`docs/ten-vad-embedded.md`](../../docs/ten-vad-embedded.md). This file covers
> the firmware itself.

Bare-metal RISC-V firmware running TEN-VAD on `rlx-ten-vad-core`.

Deliberately outside the workspace: it builds for a bare-metal target with its
own linker script and profile, and pulling it in would force those on every
host build.

```bash
cargo run --release                                    # rv32imc, under QEMU
cargo build --release --target riscv32imafc-unknown-none-elf   # FPU variant
```

QEMU's `virt` board is the same `rv32imc` ISA as an ESP32-C3/C6, so the compute
path is what the chip executes. Porting to real silicon means swapping `link.x`
for esp-hal's and pointing the UART at `0x6000_0000`.

## No allocator

There isn't one. Every buffer in `Frontend`, `Net` and `FixedNet` is a
fixed-size array, so the firmware declares no `#[global_allocator]` at all —
nothing to size, nothing to fragment, and no allocation that can fail
mid-frame. `Vad` is 33.6 kB, which goes in a `static`: it is more than a
default task stack.

## What it reports

The firmware self-checks against vectors produced on the host by
`rlx_ten_vad_core::fixed` — **zero mismatches**, so the integer net is
bit-exact on RISC-V, not merely close.

It also prints `mcycle` timings, but **do not trust them**. On the `virt` board
that counter is wall-clock-derived: identical runs vary by ~8% and the number
moves with host load. Use instruction counts instead, which reproduce to 2e-8:

```bash
cc -shared -fPIC -Wl,-undefined,dynamic_lookup \
   $(pkg-config --cflags glib-2.0) -I/opt/homebrew/include -o libinsn.dylib tools/insn.c

K=target/riscv32imc-unknown-none-elf/release/ten-vad-mcu
for p in 4 1 2 3; do            # 4 = baseline, 1 = int net, 2 = f32 net, 3 = full
  qemu-system-riscv32 -machine virt -bios none -nographic -plugin ./libinsn.dylib \
    -device loader,addr=0x80700000,data=$p,data-len=4 -kernel $K 2>&1 | grep INSNS
done
```

Per frame, baseline subtracted, for both ISAs:

| | `rv32imc` (C3/C6, no FPU) | `rv32imafc` (P4, FPU) |
|---|---|---|
| integer net | 926,186 | 926,146 |
| f32 net | 9,873,151 | **820,368** |
| full pipeline (per 16 ms hop) | 14,420,372 | **1,212,327** |
| f32 ÷ integer | **10.7x** | **0.9x** |

Frontend breakdown, same units:

| | `rv32imc` | `rv32imafc` |
|---|---|---|
| f32 FFT (`ooura`, phase 5) | 732,333 | 66,510 |
| integer FFT (`fft_fixed`, phase 9) | **284,797** | 264,469 |
| pitch estimator (phase 6) | 3,741,103 | 347,958 |
| whole frontend (phase 7) | 4,625,311 | 437,349 |
| mel + rest (7 − 5 − 6) | 151,875 | 22,881 |

The integer FFT is **2.57x cheaper than f32 without an FPU and 4.0x more
expensive with one** — the same shape as the net, and for the same reason.

**`fft_fixed` is not wired into the pipeline.** `frontend.rs` still calls
`ooura::power_spectrum`, so phase 9 measures a transform the `full` row does
not use, and the 2.57x is available but unrealized. Switching it changes the
spectrum feeding mel, so it needs parity against the f32 path and the published
ONNX before it can be the default — that validation is the work, not the swap.

Two things fall out of that last row.

Without an FPU the integer net is **10.7x cheaper** than the f32 one — every
multiply-add is a soft-float call — which is why `fixed` exists.

**With** an FPU it is *slightly more expensive* than plain f32 (0.9x), because
`fixed` pays for i64 accumulation on a 32-bit core to buy determinism it no
longer needs for speed. On an FPU part, run `Net`. `FixedNet` earns its keep on
FPU-less cores, and as the bit-exact golden model for the FPGA.

## Which part

| | pipeline | at clock | verdict |
|---|---|---|---|
| ESP32-C3/C6 `rv32imc` | **3.48 M** insn/hop | 21.7 ms @ 160 MHz | **135% of a core** |
| ESP32-P4 `rv32imafc` | 1.14 M insn/hop | 2.8 ms @ 400 MHz | **18% of a core — real-time** |

The C3 row was 14.4 M / 90.3 ms / **564%** before the integer net and the pitch
work above — **4.1x** faster, and still 35% short.

**135% is what this crate builds today.** A 114% figure appears elsewhere in
this repo's history; it assumes `fft_fixed` is wired into `frontend.rs`, which
it is not — the frontend still calls `ooura::power_spectrum`, so the f32 FFT
(732,324) is in the measurement and the integer one (191,555) is not. Treat
114% as a projection of a one-line change, not a measurement.

Part of an earlier step from 118% to 114% was a measurement fix rather than a speedup:
phases that construct an object inside the measured run were charging its
one-time cost to 16 frames. Phases 15/16/17 isolate construction so it can be
subtracted; the integer FFT's steady-state cost is 192 k, not the 285 k that
included a share of building its twiddle table.

Hardware floating point is worth **11.7x** on the whole pipeline. The FPU build
needs one thing the soft-float one does not: `mstatus.FS` set at startup.
RISC-V resets with the FPU off and every float instruction traps as illegal;
with no handler installed the core vectors to 0 and hangs. `_start` sets it
unconditionally — on a core without `F` the field is hardwired to 0 and the
write is a no-op.

## Read the last line

**TEN-VAD is comfortably real-time on an ESP32-P4, and still ~18% over budget
on an ESP32-C3.** At one instruction per cycle the C3 needs 18.9 ms per 16 ms
hop; the P4 needs 2.8 ms. Real silicon is slower than that ideal, so the C3 gap
is wider in practice than 35%.

Where the C3's 3.48 M goes — note that **73% of it is floating point on a core
with no FPU**, which is the entire reason for the gap:

| | insn/hop | share | arithmetic |
|---|---|---|---|
| `ooura` FFT (f32) | 732,324 | 21% | **f32** |
| `update_excitation` | 720,764 | 21% | **f32** |
| `update_xcorr` | 326,716 | 9% | **f32** |
| `estimate` (Viterbi) | 254,271 | 7% | **f32** |
| bands + DCT | 219,255 | 6% | **f32** |
| `update_lpc` | 177,221 | 5% | **f32** |
| mel + rest | ~120,000 | 3% | **f32** |
| integer net | 924,202 | 27% | integer |

(`fft_fixed` would replace the first row with 191,555 once wired in.)

**`update_excitation` and `update_xcorr` are what is left**, and neither has any
hoistable work in it — both are flat multiply-accumulate loops (an order-16 LPC
inverse filter and a 5-section biquad over 256 samples; a 64x32 correlation
twice). At ~60 instructions per soft-float multiply-add and ~16,000 of them per
frame, that is the floor for floating point on this core.

Porting those two to fixed point is the remaining step. The net went 10.7x that
way and the FFT 2.6x; even 7x here would take 1.05 M down to ~150 k and the
pipeline to **~2.1 M = 81% of a core**. Unlike the four changes above, it is a
*behavioural* change rather than a rounding one — the Viterbi picks an argmax
over correlation peaks, so it has to be validated on decision flips, not on
|Δ|.

So the options are, in order of effort:

1. **Use a core with an FPU.** ESP32-P4 (`rv32imafc`, 400 MHz) or ESP32-S3
   (Xtensa LX7). The firmware builds and **runs** for `riscv32imafc` — the FPU
   variant hung until `_start` set `mstatus.FS`, since RISC-V resets with the
   FPU off and the first float instruction traps into a vector table that is
   not there.
2. **Finish the fixed-point frontend.** `update_excitation` and `update_xcorr`
   are the only float hot spots left, together 35% of the C3 pipeline. This is
   the real fix for a C3-class part and the larger job — the frontend's
   bit-exactness fixtures are tied to f32 arithmetic, so it needs its own
   decision-level validation.
3. Run the VAD on a host and use the MCU only for capture.

There is no heap: see **No allocator** above. Every buffer is a fixed-size
array, `Vad` is 33.6 kB, and the firmware declares no `#[global_allocator]`.
