# TEN-VAD on embedded targets — how it runs, and how fast

Every number here was measured, and the method is stated next to it. Where a
figure is a projection rather than a measurement it says so; those are the ones
most likely to be wrong.

## The short version

| target | per 16 ms hop | verdict | basis |
|---|---|---|---|
| **ESP32-P4** `rv32imafc` @400 MHz | 2.84 ms | **17% of a core — real-time** | measured |
| **FPGA** ECP5 @50 MHz (network only) | 3.5 ms | **4.6x margin — real-time** | measured |
| **ESP32-S3** Xtensa LX7 @240 MHz | ~4.7 ms | ~30% of one core — expect real-time | projected |
| **ESP32-C3/C6** `rv32imc` @160 MHz | 21.7 ms | **135% — not real-time** | measured |

One sentence explains the whole table: **the C3 has no FPU and 73% of this
pipeline is floating-point arithmetic.** Every other target either has hardware
float or does not execute instructions at all.

## Why the C3 is slow

`rv32imc` has no `F` extension, so every float operation is a soft-float library
call — about 60-85 instructions per multiply-add against 7-8 for the integer
equivalent. Three independent measurements of the same effect:

| | soft-float | integer | ratio |
|---|---|---|---|
| the network | 9,873,158 | 924,206 | **10.7x** |
| the 1024-point FFT | 732,331 | 191,563 | **3.8x** |

And the cleanest one, because nothing changes but the hardware: the *same f32
network source* costs 9,873,158 instructions on `rv32imc` and **820,373** on
`rv32imafc`. Twelve times, from one missing unit.

Where the C3's 3,475,421 instructions go:

| | insn/hop | share | arithmetic |
|---|---|---|---|
| `ooura` FFT | 732,331 | 21% | f32 |
| `update_excitation` | 716,747 | 21% | f32 |
| `update_xcorr` | 326,766 | 9% | f32 |
| `estimate` (Viterbi) | 254,282 | 7% | f32 |
| bands + DCT | 219,294 | 6% | f32 |
| `update_lpc` | 177,268 | 5% | f32 |
| mel + rest | 124,527 | 4% | f32 |
| **integer net** | **924,206** | **27%** | integer |

The model is 27% of the frame. **The DSP frontend is 73%**, and it is the part
still in floating point. No amount of model work fixes a C3; finishing the
fixed-point frontend does.

## How it was measured

`mcycle` on QEMU's `virt` board is wall-clock-derived — identical runs vary ~8%
and the number moves with host load, which is enough to invent or hide a 10%
change. Everything above instead uses a QEMU TCG plugin counting retired
instructions, reproducible to 2e-8:

```bash
cc -shared -fPIC -Wl,-undefined,dynamic_lookup \
   $(pkg-config --cflags glib-2.0) -I/opt/homebrew/include -o libinsn.dylib tools/insn.c

K=target/riscv32imc-unknown-none-elf/release/ten-vad-mcu
qemu-system-riscv32 -machine virt -bios none -nographic -plugin ./libinsn.dylib \
  -device loader,addr=0x80700000,data=$PHASE,data-len=4 -kernel $K
```

A plugin reports one total per run, so `data=$PHASE` selects which stage runs:
4 is a bare harness to subtract, 1/2/3 the integer net / f32 net / full
pipeline, 5..9 parts of the frontend, 10..14 stages inside the pitch estimator,
and **15/16/17 isolate object construction** so it can be subtracted from the
others. That last group matters: phases that build an estimator inside the
measured run were charging its one-time cost to 16 frames, which made the
integer FFT read 285 k instead of its steady-state 192 k.

## What made it 4.1x faster

564% -> 135% of a C3 core. In order of size:

| change | effect | bit-exact? |
|---|---|---|
| integer network (`fixed`) | 9,873,158 -> 924,206 | no — a separate datapath |
| `update_lpc`: 18x17 matrix for a 1024-point inverse FFT | 1,703,431 -> 177,268 | no (`fast-pitch`) |
| Viterbi transition penalty tabled | 751,799 -> 254,282 | **yes** |
| band geometry tabled | 339,695 -> 219,294 | **yes** |
| LPC history: circular buffer, not a shift register | -61,000 | **yes** |

Three of the five change no bits, so the frontend's "bit-identical to the
upstream C DSP" test still guards them unchanged.

The `update_lpc` one is the interesting case. It built a 513-bin gain spectrum
and ran a 1024-point inverse FFT to keep 17 autocorrelation lags — and every
step of that chain is linear, so the whole thing collapses to a fixed 18x17
matrix: 306 multiply-adds. It sums in a different order than the FFT, so it sits
behind `fast-pitch`; measured effect on the published fixture is
**max|Δ| 2.09e-6 with zero decision flips**.

## Accuracy

Against **the model TEN-framework publishes** (`ten-vad.onnx` driven by the DSP
in `src/*.cc`), over the 250-frame reference clip:

| | max\|Δ\| | 1-cos | flips |
|---|---|---|---|
| this port, f32 | 5.1e-7 | 1.3e-14 | 0/250 |
| this port, int16 net | **3.7e-4** | 2.0e-8 | 0/250 |
| TEN-framework's own prebuilt `libten_vad` | 9.6e-4 | 9.4e-8 | — |

The quantised port is **2.6x closer in max|Δ| than the vendor's shipped
binary**. That is the property to protect, and it is why the faster
`narrow-acc` datapath below is off by default.

## Optional: `narrow-acc`

The LSTM gate accumulator is `i64` because measured partial sums need 36 bits
and `i32` allows 31. The width comes from the *product*: the conv output reaches
18.8 in Q15 units, so it occupies ~20 bits, and 20 + 16 weight bits is already
36. Q15 is simply the wrong scale for that tensor — it reserves 15 bits below
1.0 and then the values run past it.

Shifting the LSTM input down 6 bits fixes that:

| | net insn | max\|Δ\| vs published | flips / 497,775 |
|---|---|---|---|
| default, i64 | 924,206 | **3.7e-4** | 107 |
| `narrow-acc`, i32 | **646,886** | 1.6e-3 | 122 |

30% off the network. It is opt-in because 1.6e-3 is worse than the vendor's
9.6e-4, and being better than that binary is a claim this port is made on.
Decisions are unaffected on the reference clip; over 497,775 frames they move
0.021% -> 0.025%.

Asymmetric shifts were tried — giving `h` and the conv output different budgets
— and are worse in both directions (x>>8/h>>4 costs 232 flips, x>>4/h>>10 costs
2417, against 122 for a uniform 6). The conv output is the operand that needs
the bits.

After changing this, regenerate the firmware's golden vectors:
`cargo run -p rlx-ten-vad --example gen_mcu_vectors`.

## FPGA

`rlx-ten-vad-fpga` exports RTL from the rlx-ir graph — a microcoded single-MAC
datapath with a descriptor ROM and liveness-allocated activation RAM.

| | |
|---|---|
| frames checked | 250 |
| **mismatches** | **0** — bit-exact against the Rust fixed-point net |
| cycles/frame | 173,140 |
| minimum clock for 62.5 fps | **10.82 MHz** |

`yosys synth_ecp5`: 4,168 LUT4, 893 TRELLIS_FF, 70 DP16KD (1.26 Mbit),
13 MULT18X18D. Fits an LFE5U-45F or an XC7A35T. The weight ROM is 1.2 Mbit of
the 1.26, so weight width is the only thing between this and a smaller part.

At a typical 50 MHz that is 3.5 ms per 16 ms frame — **4.6x margin**, 9.2x at
100 MHz. The design is deliberately serial at 2 cycles/MAC; one MAC per cycle is
a mechanical change that halves it again.

**The RTL implements the network only.** The DSP frontend — 73% of the work on
the MCU — is not in RTL at all. A self-contained FPGA device needs it as a
second block or on a host.

```bash
iverilog -g2012 -I rtl -o /tmp/tv_tb tb/tv_tb.sv rtl/tv_core.sv rtl/tv_lut.sv
cd rtl && vvp /tmp/tv_tb
yosys -p "read_verilog -sv -I rtl rtl/tv_core.sv rtl/tv_lut.sv; synth_ecp5 -top tv_core"
```

## Footprint

`Vad` is **33,632 B, entirely inline** — the firmware declares no
`#[global_allocator]` at all, so there is nothing to size, nothing to fragment,
and no allocation that can fail mid-frame. Put it in a `static`: 33 kB is more
than a default task stack. Weights are 305,080 B of flash.

About 8.7 kB of that RAM is tables built at startup that are pure functions of
constants — `FixedFft` twiddles (4,096 B, 3.4 M instructions) and the pitch
estimator's matrix, DCT and band tables (4,580 B, 5.3 M). Generating them into
flash `const`s would free 25% of `Vad` and ~54 ms of boot. It would not change
the frame budget by one instruction.

## Portability

`rlx-ten-vad-core` is `no_std` with exactly one dependency (`libm`) and builds
for `riscv32imc`, `riscv32imafc` and `thumbv7em-none-eabihf`. The compute is
portable; the firmware harness is not — linker script, QEMU `virt` board, and a
RISC-V `mstatus.FS` startup. An ESP32-S3 port is a toolchain exercise, not DSP
work.

The FPU build needs one thing the soft-float one does not: `mstatus.FS` set at
startup. RISC-V resets with the FPU off and the first float instruction traps as
illegal; with no handler the core vectors to 0 and hangs. `_start` sets it
unconditionally — on a core without `F` the field is hardwired to 0 and the
write is a no-op.

## What remains for a C3

`update_excitation` (716,747) and `update_xcorr` (326,766) are **30% of the
frame and still floating point** — an order-16 LPC inverse filter plus a
5-section biquad over 256 samples, and a 64x32 correlation twice. Neither has
any hoistable work left; both are flat multiply-accumulate loops at the
soft-float floor.

Porting them at the ratio the other integer ports achieved (net 10.7x, FFT 3.8x)
projects to **~68% of a core** — real-time with the margin real silicon needs,
since none of these figures account for flash wait states or cache misses at
less than one instruction per cycle.

Unlike the bit-exact pitch work, that is a *behavioural* change: the Viterbi
takes an argmax over correlation peaks, so it must be validated on decision
flips rather than |Δ|, and the biquad is IIR — five cascaded sections whose
state needs enough fractional bits to avoid limit cycles.

## An architecture alternative, measured

The LSTM's recurrent term is a matmul (`h @ W_hh`, 32,768 MACs — 41% of the
model). An SRU's is elementwise, so all its matmuls depend only on `x`.

Trained as distillation students from scratch, equal budget, 3 seeds each:

| | mean flips / 2304 | MACs | net insn | % of C3 core | FPGA cycles |
|---|---|---|---|---|---|
| LSTM H=64 | 585.0 | 79,295 | 924,128 | 114%* | 173,070 |
| **SRU H=64** | 602.3 | **37,311** | 434,833 | 95%* | **81,435** |

The difference is **0.54 sigma** — indistinguishable — at **47% of the MACs**.
SRU also has a property that matters more than the MAC count: `c = f*c_prev +
(1-f)*g` is a convex combination, so |c| <= 1 by construction, where the LSTM's
independent gates let the cell grow (measured max 326.9) and forced the 36-bit
accumulator in the first place.

Both students are far from teacher parity (26% flips at 2000 steps ~ 0.5
epochs), so this shows equal *learning per step at equal budget*, not that
either reaches the teacher. `*` these percentages assume the integer FFT wired
in, and the SRU row is a projection from MAC counts — no SRU integer kernel
exists.

Reproduce: `cargo run --release -p rlx-ten-vad --example arch_compare -- --arch sru --hidden 64 --steps 2000`
