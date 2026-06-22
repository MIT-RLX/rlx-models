// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Shared helpers for the per-crate codec micro-benchmarks (`examples/bench.rs`
//! in each `rlx-*` audio codec crate). The goal is one uniform, machine-parseable
//! result line per `(crate, backend, op)` so a top-level matrix builder can scrape
//! decode/encode throughput across every rlx backend.
//!
//! Each example owns its own `bench_devices()` (it must `#[cfg]`-gate on the
//! features *that* crate compiled), then calls [`time_median_ms`] / [`report`]
//! here so timing and formatting stay identical everywhere.

use rlx_runtime::Device;
use std::time::Instant;

/// Tiny deterministic PRNG (SplitMix64-ish) so benches synthesize identical
/// inputs across backends without pulling in `rand`. Decode/encode cost depends
/// on sequence length, not code values, so random-but-valid inputs are fine for
/// timing.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `0..n` (n>=1).
    #[inline]
    pub fn idx(&mut self, n: usize) -> u32 {
        (self.next_u64() % n.max(1) as u64) as u32
    }

    /// Uniform f32 in `[-1, 1)`.
    #[inline]
    pub fn f(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Quantizer-major random codes `[n_q][t]`, each in `0..codebook_size`.
pub fn synth_codes_qmajor(n_q: usize, t: usize, codebook_size: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut r = Lcg::new(seed);
    (0..n_q)
        .map(|_| (0..t).map(|_| r.idx(codebook_size)).collect())
        .collect()
}

/// Same as [`synth_codes_qmajor`] but `i64` (Group-FSQ style decoders).
pub fn synth_codes_i64(
    n_groups: usize,
    t: usize,
    codebook_size: usize,
    seed: u64,
) -> Vec<Vec<i64>> {
    let mut r = Lcg::new(seed);
    (0..n_groups)
        .map(|_| (0..t).map(|_| r.idx(codebook_size) as i64).collect())
        .collect()
}

/// Random PCM in `[-amp, amp)`.
pub fn synth_pcm(n: usize, amp: f32, seed: u64) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| r.f() * amp).collect()
}

/// Random latent/embedding buffer of length `n` in `[-1, 1)`.
pub fn synth_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| r.f()).collect()
}

/// Run `f` `warmup` times untimed, then `iters` times timed; return the median
/// wall time in milliseconds. Propagates the first error.
pub fn time_median_ms<T>(
    warmup: usize,
    iters: usize,
    mut f: impl FnMut() -> anyhow::Result<T>,
) -> anyhow::Result<f64> {
    for _ in 0..warmup {
        f()?;
    }
    let mut samples = Vec::with_capacity(iters.max(1));
    for _ in 0..iters.max(1) {
        let t0 = Instant::now();
        f()?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples[samples.len() / 2])
}

/// Time a single call (e.g. one-shot graph compile) in milliseconds.
pub fn time_once_ms<T>(mut f: impl FnMut() -> anyhow::Result<T>) -> anyhow::Result<(T, f64)> {
    let t0 = Instant::now();
    let out = f()?;
    Ok((out, t0.elapsed().as_secs_f64() * 1000.0))
}

/// Emit the uniform success line. `rtf` = audio seconds / wall seconds (higher is
/// faster than real time). `compile_ms` is one-time graph build (`0.0` if N/A).
pub fn report(
    krate: &str,
    backend: Device,
    op: &str,
    dur_s: f64,
    t_frames: usize,
    compile_ms: f64,
    run_ms: f64,
) {
    let rtf = if run_ms > 0.0 {
        dur_s / (run_ms / 1000.0)
    } else {
        0.0
    };
    println!(
        "BENCH crate={krate} backend={backend:?} op={op} dur_s={dur_s:.3} t={t_frames} \
         compile_ms={compile_ms:.2} run_ms={run_ms:.2} rtf={rtf:.1} ok=true"
    );
}

/// Emit the uniform failure/skip line so the matrix can mark a `(crate, backend)`
/// cell as a gap with a reason.
pub fn report_fail(krate: &str, backend: Device, op: &str, reason: &str) {
    let reason = reason.replace('\n', " ");
    println!("BENCH crate={krate} backend={backend:?} op={op} ok=false reason=\"{reason}\"");
}

/// Parse the common `--dur <secs>` / `--iters <n>` args shared by every codec
/// bench example. Defaults: 4.0 s, 5 timed iterations.
pub fn parse_dur_iters() -> (f64, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dur = 4.0f64;
    let mut iters = 5usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dur" | "--seconds" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    dur = v;
                }
            }
            "--iters" | "-n" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    iters = v;
                }
            }
            _ => {}
        }
        i += 1;
    }
    (dur, iters)
}

/// Standard backend list for a bench example. Callers pass the devices their own
/// crate `#[cfg]`-compiled; this only filters to what the hardware reports
/// available, always keeping CPU.
pub fn available(candidates: &[Device]) -> Vec<Device> {
    let mut v = vec![Device::Cpu];
    for &d in candidates {
        if d != Device::Cpu && rlx_runtime::is_available(d) {
            v.push(d);
        }
    }
    v
}
