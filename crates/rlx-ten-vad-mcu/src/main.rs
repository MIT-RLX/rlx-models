// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Bare-metal RISC-V firmware: TEN-VAD on a core with no FPU and no OS.
//!
//! Runs on QEMU's `virt` board, which is the same `rv32imc` ISA as an
//! ESP32-C3/C6 — so the compute path here is what the chip executes; only the
//! linker script and UART address differ. Building for real silicon means
//! swapping `link.x` for esp-hal's and pointing [`uart`] at `0x6000_0000`.
//!
//! It self-checks against vectors produced by `rlx_ten_vad_core::fixed` on the
//! host, then reports cycle counts for three paths so the cost of each is
//! visible:
//!
//! * the integer net — no floating point at all,
//! * the f32 scalar net — every MAC through soft-float,
//! * the full pipeline — DSP frontend plus net, from PCM.

#![no_std]
#![no_main]

use core::panic::PanicInfo;
use rlx_ten_vad_core::fixed::{FixedNet, ONE};
use rlx_ten_vad_core::{CONTEXT_FRAMES, FEATURE_LEN, HOP_SIZE, Vad, net::Net};

core::arch::global_asm!(
    r#"
    .section .init
    .globl _start
_start:
    la   sp, _stack_top
    // Enable the FPU before any Rust runs. RISC-V resets with mstatus.FS = Off,
    // and every floating-point instruction then traps as illegal; with no
    // handler installed the core vectors to 0 and hangs. On a core without `F`
    // the field is hardwired to 0 and this csrs is a no-op, so it is
    // unconditional.
    li   t0, 0x2000                 // mstatus.FS = Initial
    csrs mstatus, t0
    la   t0, __bss_start
    la   t1, __bss_end
1:  bgeu t0, t1, 2f
    sw   zero, 0(t0)
    addi t0, t0, 4
    j    1b
2:  call rust_main
3:  j    3b
"#
);

// ---- platform ------------------------------------------------------------

/// QEMU `virt` NS16550A transmit register. ESP32-C3 puts its UART at
/// `0x6000_0000`.
const UART: *mut u8 = 0x1000_0000 as *mut u8;
/// QEMU `virt` SiFive test finisher, so the run terminates instead of spinning.
const FINISHER: *mut u32 = 0x0010_0000 as *mut u32;

fn putc(c: u8) {
    unsafe { UART.write_volatile(c) }
}

fn print(s: &str) {
    for b in s.bytes() {
        putc(b);
    }
}

fn print_u32(mut v: u32) {
    let mut buf = [0u8; 10];
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    for &b in &buf[i..] {
        putc(b);
    }
}

/// Print a fixed-point value as `int.frac` with three decimals.
fn print_q15(v: i32) {
    if v < 0 {
        putc(b'-');
    }
    let a = v.unsigned_abs();
    print_u32(a >> 15);
    putc(b'.');
    let frac = (a & 0x7fff) as u64 * 1000 / 32768;
    let mut d = frac as u32;
    if d < 100 {
        putc(b'0');
    }
    if d < 10 {
        putc(b'0');
    }
    if d == 0 {
        d = 0;
    }
    print_u32(d);
}

/// Low half of `mcycle`. Deltas are taken with `wrapping_sub`, so a 32-bit
/// counter is enough for anything shorter than 2^32 cycles (~27 s at 160 MHz).
/// Which phase to run, poked into RAM by QEMU before boot.
///
/// `mcycle` on the `virt` board is wall-clock-derived: identical runs vary by
/// ~8% and the number moves with host load, so it cannot separate a real
/// change from noise. An instruction count can — it is reproducible to 2e-8 —
/// but a plugin only reports a total for the run. Selecting one phase per run
/// turns that total into a per-phase figure:
///
/// ```text
/// cc -shared -fPIC -Wl,-undefined,dynamic_lookup \
///    $(pkg-config --cflags glib-2.0) -I/opt/homebrew/include -o libinsn.dylib insn.c
/// qemu-system-riscv32 -machine virt -bios none -nographic \
///   -plugin ./libinsn.dylib \
///   -device loader,addr=0x80700000,data=1,data-len=4 \
///   -kernel target/riscv32imc-unknown-none-elf/release/ten-vad-mcu
/// ```
///
/// Phase 0 runs everything (and prints); 4 runs nothing, giving the harness
/// baseline to subtract. 5/6/7 isolate parts of the DSP frontend: the FFT, the
/// pitch estimator, and the whole frontend — mel and the rest are 7 - 5 - 6.
const PHASE_ADDR: *const u32 = 0x8070_0000 as *const u32;

fn phase() -> u32 {
    unsafe { PHASE_ADDR.read_volatile() }
}

fn cycles() -> u32 {
    let lo: u32;
    unsafe { core::arch::asm!("csrr {}, mcycle", out(reg) lo, options(nomem, nostack)) };
    lo
}

fn exit(code: u16) -> ! {
    unsafe { FINISHER.write_volatile(if code == 0 { 0x5555 } else { 0x3333 | (u32::from(code) << 16) }) }
    loop {
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    print("PANIC\n");
    exit(2)
}

// The VAD needs no allocator. Every buffer in `Frontend`, `Net` and
// `FixedNet` is a fixed-size array, so there is no heap here at all — nothing
// to size, nothing to fragment, and no allocation that can fail mid-frame.

// ---- vectors -------------------------------------------------------------

const FRAMES: usize = 16;
const STACK: usize = CONTEXT_FRAMES * FEATURE_LEN;

static FEATS: [u8; FRAMES * STACK * 4] = *include_bytes!("../data/feats_q15.bin");
static PROBS: [u8; FRAMES * 4] = *include_bytes!("../data/probs_q15.bin");
static PCM: [u8; FRAMES * HOP_SIZE * 2] = *include_bytes!("../data/pcm_i16.bin");

fn feat_q15(frame: usize, i: usize) -> i32 {
    let o = (frame * STACK + i) * 4;
    i32::from_le_bytes([FEATS[o], FEATS[o + 1], FEATS[o + 2], FEATS[o + 3]])
}

fn prob_q15(frame: usize) -> i32 {
    let o = frame * 4;
    i32::from_le_bytes([PROBS[o], PROBS[o + 1], PROBS[o + 2], PROBS[o + 3]])
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_main() -> ! {
    print("rlx-ten-vad on rv32imc (no FPU)\n\n");

    // ---- integer net: must be bit-exact against the host ----------------
    let mut net = FixedNet::embedded();
    let mut stack = [0i32; STACK];
    // Correctness first, from a known state — the golden vectors are a single
    // continuous run, so a repeat would score against the wrong LSTM state.
    let mut bad = 0u32;
    for f in 0..FRAMES {
        for (i, slot) in stack.iter_mut().enumerate() {
            *slot = feat_q15(f, i);
        }
        if net.forward(&stack) != prob_q15(f) {
            bad += 1;
        }
    }
    // Then timing. Under a phase selector this is the only work in the run,
    // so the plugin's instruction total is this phase's cost.
    let sel = phase();
    let mut int_cy = 0;
    if sel == 0 || sel == 1 {
        net.reset();
        let t0 = cycles();
        for f in 0..FRAMES {
            for (i, slot) in stack.iter_mut().enumerate() {
                *slot = feat_q15(f, i);
            }
            core::hint::black_box(net.forward(&stack));
        }
        int_cy = cycles().wrapping_sub(t0) / FRAMES as u32;
    }

    print("integer net    ");
    print_u32(int_cy);
    print(" cycles/frame   mismatches ");
    print_u32(bad);
    putc(b'\n');

    // ---- f32 scalar net: the same maths through soft-float --------------
    let mut fnet = Net::new(rlx_ten_vad_core::weights::embedded_net());
    let mut fstack = [0.0f32; STACK];
    let mut f32_cy = 0;
    if sel == 0 || sel == 2 {
        let t0 = cycles();
        for f in 0..FRAMES {
            for (i, slot) in fstack.iter_mut().enumerate() {
                *slot = feat_q15(f, i) as f32 / ONE as f32;
            }
            core::hint::black_box(fnet.forward(&fstack));
        }
        f32_cy = cycles().wrapping_sub(t0) / FRAMES as u32;
    }
    print("f32 net        ");
    print_u32(f32_cy);
    print(" cycles/frame\n");

    // ---- full pipeline from PCM ----------------------------------------
    // Static rather than a local: 34 kB is more than a default task stack.
    static mut VAD: Option<Vad> = None;
    let vad = unsafe {
        let slot = &raw mut VAD;
        (*slot) = Some(Vad::new());
        (*slot).as_mut().unwrap()
    };
    let mut hop = [0i16; HOP_SIZE];
    let mut last = 0.0f32;
    let mut full_cy = 0;
    if sel == 0 || sel == 3 {
        vad.reset();
        let t0 = cycles();
        for h in 0..FRAMES {
            for (i, slot) in hop.iter_mut().enumerate() {
                let o = (h * HOP_SIZE + i) * 2;
                *slot = i16::from_le_bytes([PCM[o], PCM[o + 1]]);
            }
            last = vad.process_i16(&hop);
        }
        full_cy = cycles().wrapping_sub(t0) / FRAMES as u32;
    }
    print("full pipeline  ");
    print_u32(full_cy);
    print(" cycles/hop     last p=");
    print_q15((last * ONE as f32) as i32);
    putc(b'\n');

    // 16 ms of audio per hop, so this is the real-time budget.
    print("\nat 160 MHz (ESP32-C3): frontend+net ");
    print_u32(full_cy / 160);
    print(" us per 16000 us frame = ");
    print_u32(full_cy / 160 * 100 / 16000);
    print("% of one core\n");

    // ---- frontend breakdown, for the phase selector ----------------------
    if sel >= 5 {
        let mut window = [0.0f32; rlx_ten_vad_core::WINDOW_SIZE];
        for (i, w) in window.iter_mut().enumerate() {
            let o = (i % (FRAMES * HOP_SIZE)) * 2;
            *w = f32::from(i16::from_le_bytes([PCM[o], PCM[o + 1]]));
        }
        let mut spectrum = [0.0f32; rlx_ten_vad_core::SPECTRUM_BINS];
        if sel == 5 {
            for _ in 0..FRAMES {
                rlx_ten_vad_core::ooura::power_spectrum(&window, &mut spectrum);
                core::hint::black_box(&spectrum);
            }
        }
        if sel == 6 {
            rlx_ten_vad_core::ooura::power_spectrum(&window, &mut spectrum);
            let mut pe = rlx_ten_vad_core::pitch::PitchEstimator::new();
            let mut sig = [0.0f32; HOP_SIZE];
            for h in 0..FRAMES {
                for (i, v) in sig.iter_mut().enumerate() {
                    let o = ((h * HOP_SIZE + i) % (FRAMES * HOP_SIZE)) * 2;
                    *v = f32::from(i16::from_le_bytes([PCM[o], PCM[o + 1]]));
                }
                core::hint::black_box(pe.process(&sig, &spectrum));
            }
        }
        if sel == 9 {
            // The same transform in integer arithmetic.
            let qi: [i32; rlx_ten_vad_core::WINDOW_SIZE] =
                core::array::from_fn(|i| window[i] as i32);
            let fft = rlx_ten_vad_core::fft_fixed::FixedFft::new();
            let mut pw = [0i64; rlx_ten_vad_core::fft_fixed::BINS];
            for _ in 0..FRAMES {
                fft.power_spectrum(&qi, &mut pw);
                core::hint::black_box(&pw);
            }
        }
        if sel == 8 {
            // The inverse transform inside update_lpc: a second FFT per frame.
            let mut ac = [0.0f32; 17];
            for _ in 0..FRAMES {
                rlx_ten_vad_core::ooura::real_spectrum_to_autocorrelation(&spectrum, &mut ac);
                core::hint::black_box(&ac);
            }
        }
        // Pitch internals: 10 bands+DCT, 11 update_lpc, 12 update_excitation,
        // 13 update_xcorr, 14 estimate. Their sum should account for phase 6.
        // 15 = construction alone. The pitch phases below build an estimator
        // inside the measured run, so its cost lands in every one of them;
        // `fast-pitch` made that construction expensive enough to swamp the
        // differences it was meant to show. Subtract this from 10..=14.
        if sel == 15 {
            core::hint::black_box(rlx_ten_vad_core::pitch::PitchEstimator::new());
        }
        // 17 = FixedFft construction alone (phase 9 builds one inside its run).
        if sel == 17 {
            core::hint::black_box(rlx_ten_vad_core::fft_fixed::FixedFft::new());
        }
        // 16 = Frontend construction alone, the same subtraction for phase 7.
        if sel == 16 {
            core::hint::black_box(rlx_ten_vad_core::frontend::Frontend::new(
                rlx_ten_vad_core::weights::embedded(),
            ));
        }
        if (10..=14).contains(&sel) {
            let mut pe = rlx_ten_vad_core::pitch::PitchEstimator::new();
            let mut sig = [0.0f32; HOP_SIZE];
            for (i, v) in sig.iter_mut().enumerate() {
                let o = (i % (FRAMES * HOP_SIZE)) * 2;
                *v = f32::from(i16::from_le_bytes([PCM[o], PCM[o + 1]]));
            }
            let cep = pe.bench_bands_dct(&spectrum);
            for _ in 0..FRAMES {
                match sel {
                    10 => {
                        core::hint::black_box(pe.bench_bands_dct(&spectrum));
                    }
                    11 => pe.bench_update_lpc(&cep),
                    12 => pe.bench_update_excitation(&sig),
                    13 => pe.bench_update_xcorr(),
                    _ => {
                        core::hint::black_box(pe.bench_estimate());
                    }
                }
            }
        }
        if sel == 7 {
            let mut fe = rlx_ten_vad_core::frontend::Frontend::new(
                rlx_ten_vad_core::weights::embedded(),
            );
            let mut prev = 0.0f32;
            let mut emph = [0.0f32; HOP_SIZE];
            let mut raw = [0.0f32; HOP_SIZE];
            for h in 0..FRAMES {
                for (i, v) in raw.iter_mut().enumerate() {
                    let o = (h * HOP_SIZE + i) * 2;
                    *v = f32::from(i16::from_le_bytes([PCM[o], PCM[o + 1]]));
                }
                rlx_ten_vad_core::frontend::pre_emphasis(&raw, &mut prev, &mut emph);
                fe.push(&raw, &emph);
                core::hint::black_box(fe.context());
            }
        }
    }

    if bad == 0 {
        print("\nRESULT PASS\n");
        exit(0)
    } else {
        print("\nRESULT FAIL\n");
        exit(1)
    }
}
