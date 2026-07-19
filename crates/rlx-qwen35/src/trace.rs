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

//! Decode tracing and Metal↔CUDA tap fingerprints.
//!
//! Env (all off by default):
//! - `RLX_QWEN35_BENCH=1` — end-of-run prefill/decode ms and tok/s (runner)
//! - `RLX_QWEN35_DECODE_TRACE=1` — per-token run/cache/lm/total ms
//! - `RLX_QWEN35_TAP=1` — JSONL fingerprints (hidden/logits top-k + checksum)
//! - `RLX_QWEN35_TAP_STEPS=N` — max decode steps to tap (default 8)
//! - `RLX_QWEN35_TAP_PATH=path` — write taps to file (else stderr)

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rlx_runtime::Device;

pub(crate) fn decode_trace_enabled() -> bool {
    rlx_ir::env::flag("RLX_QWEN35_DECODE_TRACE")
}

pub(crate) fn tap_enabled() -> bool {
    rlx_ir::env::flag("RLX_QWEN35_TAP")
}

fn tap_steps_limit() -> usize {
    rlx_ir::env::parse_or("RLX_QWEN35_TAP_STEPS", 8usize)
}

static TAP_STEP: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn reset_tap_step() {
    TAP_STEP.store(0, Ordering::Relaxed);
}

pub(crate) fn log_generate_header(
    device: Device,
    fast_greedy_lm_head: bool,
    host_embed: bool,
    packed: bool,
    max_seq: usize,
    vocab: usize,
    n_new: usize,
) {
    if !decode_trace_enabled() && !tap_enabled() {
        return;
    }
    let lm = if fast_greedy_lm_head {
        "host_greedy"
    } else {
        "graph"
    };
    eprintln!(
        "[qwen35][decode-trace] device={device:?} lm_head={lm} host_embed={host_embed} \
         packed={packed} max_seq={max_seq} vocab={vocab} n_new={n_new}"
    );
}

pub(crate) struct StepTimer {
    t0: Instant,
    run: Duration,
    cache: Duration,
    lm: Duration,
}

impl StepTimer {
    pub fn start() -> Self {
        Self {
            t0: Instant::now(),
            run: Duration::ZERO,
            cache: Duration::ZERO,
            lm: Duration::ZERO,
        }
    }

    pub fn time_run<R>(&mut self, f: impl FnOnce() -> R) -> R {
        let t = Instant::now();
        let out = f();
        self.run += t.elapsed();
        out
    }

    pub fn time_cache<R>(&mut self, f: impl FnOnce() -> R) -> R {
        let t = Instant::now();
        let out = f();
        self.cache += t.elapsed();
        out
    }

    pub fn time_lm<R>(&mut self, f: impl FnOnce() -> R) -> R {
        let t = Instant::now();
        let out = f();
        self.lm += t.elapsed();
        out
    }

    pub fn finish(self, step: usize, token: u32) {
        if !decode_trace_enabled() {
            return;
        }
        let total = self.t0.elapsed();
        eprintln!(
            "[qwen35][decode-trace] step={step} token={token} run_ms={:.2} cache_ms={:.2} \
             lm_ms={:.2} total_ms={:.2}",
            self.run.as_secs_f64() * 1e3,
            self.cache.as_secs_f64() * 1e3,
            self.lm.as_secs_f64() * 1e3,
            total.as_secs_f64() * 1e3,
        );
    }
}

/// Stats + top-k + FNV-1a checksum for cross-device diffs.
pub(crate) fn fingerprint(values: &[f32], top_k: usize) -> Fingerprint {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nan = 0usize;
    let mut nnz = 0usize;
    let mut hash: u64 = 0xcbf29ce484222325;
    for &x in values {
        // bit-stable checksum (NaNs → sentinel)
        let bits = if x.is_nan() {
            0x7fc0_0000u32
        } else {
            x.to_bits()
        };
        hash ^= bits as u64;
        hash = hash.wrapping_mul(0x100000001b3);

        if x.is_nan() {
            nan += 1;
            continue;
        }
        sum += x as f64;
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        if x != 0.0 {
            nnz += 1;
        }
    }
    let mean = if values.is_empty() {
        0.0
    } else {
        sum / values.len() as f64
    };
    if !min.is_finite() {
        min = 0.0;
    }
    if !max.is_finite() {
        max = 0.0;
    }

    let k = top_k.min(values.len());
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let top_ids: Vec<u32> = idx.iter().take(k).map(|&i| i as u32).collect();
    let top_vals: Vec<f32> = idx.iter().take(k).map(|&i| values[i]).collect();

    Fingerprint {
        len: values.len(),
        nnz,
        nan,
        min,
        max,
        mean,
        checksum: hash,
        top_ids,
        top_vals,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Fingerprint {
    pub len: usize,
    pub nnz: usize,
    pub nan: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub checksum: u64,
    pub top_ids: Vec<u32>,
    pub top_vals: Vec<f32>,
}

pub(crate) fn emit_tap(
    phase: &str,
    step: Option<usize>,
    token: Option<u32>,
    kind: &str,
    fp: &Fingerprint,
) {
    if !tap_enabled() {
        return;
    }
    let step_i = step.unwrap_or_else(|| TAP_STEP.fetch_add(1, Ordering::Relaxed));
    if step_i > tap_steps_limit() && phase != "prefill" && phase != "decode" {
        return;
    }
    let top_ids = fp
        .top_ids
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let top_vals = fp
        .top_vals
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let line = format!(
        "{{\"phase\":\"{phase}\",\"step\":{step_i},\"token\":{},\"kind\":\"{kind}\",\
         \"len\":{},\"nnz\":{},\"nan\":{},\"min\":{:.6},\"max\":{:.6},\"mean\":{:.6},\
         \"checksum\":\"{:016x}\",\"top_ids\":[{top_ids}],\"top_vals\":[{top_vals}]}}",
        token
            .map(|t| t.to_string())
            .unwrap_or_else(|| "null".into()),
        fp.len,
        fp.nnz,
        fp.nan,
        fp.min,
        fp.max,
        fp.mean,
        fp.checksum,
    );
    if let Some(path) = rlx_ir::env::var("RLX_QWEN35_TAP_PATH") {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{line}");
            return;
        }
    }
    eprintln!("[qwen35][tap] {line}");
}

pub(crate) fn log_lm_head_path(path: &str) {
    if !(decode_trace_enabled() || tap_enabled()) {
        return;
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if LOGGED.swap(true, Ordering::Relaxed) {
        return;
    }
    eprintln!("[qwen35][decode-trace] lm_head_path={path}");
}

/// Fingerprint host-side decode cache after prefill (KV + recurrent).
/// Env: `RLX_QWEN35_CACHE_TAP=1` (also implied by `RLX_QWEN35_TAP=1`).
pub(crate) fn emit_cache_tap(cache: &crate::cache::Qwen35DecodeCache) {
    if !tap_enabled() && !rlx_ir::env::flag("RLX_QWEN35_CACHE_TAP") {
        return;
    }
    let mut k_fp = None;
    let mut v_fp = None;
    let mut conv_fp = None;
    let mut ssm_fp = None;
    let mut n_full = 0usize;
    let mut n_linear = 0usize;
    for layer in &cache.layers {
        match layer {
            crate::cache::Qwen35LayerState::FullAttn { past_k, past_v, .. } => {
                n_full += 1;
                if k_fp.is_none() {
                    k_fp = Some(fingerprint(past_k, 8));
                    v_fp = Some(fingerprint(past_v, 8));
                }
            }
            crate::cache::Qwen35LayerState::Linear {
                conv_state,
                ssm_state,
            } => {
                n_linear += 1;
                if conv_fp.is_none() {
                    conv_fp = Some(fingerprint(conv_state, 8));
                    ssm_fp = Some(fingerprint(ssm_state, 8));
                }
            }
        }
    }
    let mut all_k = Vec::new();
    let mut all_ssm = Vec::new();
    for layer in &cache.layers {
        match layer {
            crate::cache::Qwen35LayerState::FullAttn { past_k, .. } => {
                all_k.extend_from_slice(past_k);
            }
            crate::cache::Qwen35LayerState::Linear { ssm_state, .. } => {
                all_ssm.extend_from_slice(ssm_state);
            }
        }
    }
    let all_k_fp = fingerprint(&all_k, 4);
    let all_ssm_fp = fingerprint(&all_ssm, 4);
    eprintln!(
        "[qwen35][cache-tap] past_seq={} batch={} full={} linear={} \
         k0_checksum={:016x} k0_nnz={} k0_min={:.6} k0_max={:.6} \
         v0_checksum={:016x} \
         conv0_checksum={:016x} conv0_nnz={} conv0_max={:.6} \
         ssm0_checksum={:016x} ssm0_nnz={} ssm0_min={:.6} ssm0_max={:.6} \
         all_k_checksum={:016x} all_k_nnz={} \
         all_ssm_checksum={:016x} all_ssm_nnz={} all_ssm_max={:.6}",
        cache.past_seq,
        cache.batch,
        n_full,
        n_linear,
        k_fp.as_ref().map(|f| f.checksum).unwrap_or(0),
        k_fp.as_ref().map(|f| f.nnz).unwrap_or(0),
        k_fp.as_ref().map(|f| f.min).unwrap_or(0.0),
        k_fp.as_ref().map(|f| f.max).unwrap_or(0.0),
        v_fp.as_ref().map(|f| f.checksum).unwrap_or(0),
        conv_fp.as_ref().map(|f| f.checksum).unwrap_or(0),
        conv_fp.as_ref().map(|f| f.nnz).unwrap_or(0),
        conv_fp.as_ref().map(|f| f.max).unwrap_or(0.0),
        ssm_fp.as_ref().map(|f| f.checksum).unwrap_or(0),
        ssm_fp.as_ref().map(|f| f.nnz).unwrap_or(0),
        ssm_fp.as_ref().map(|f| f.min).unwrap_or(0.0),
        ssm_fp.as_ref().map(|f| f.max).unwrap_or(0.0),
        all_k_fp.checksum,
        all_k_fp.nnz,
        all_ssm_fp.checksum,
        all_ssm_fp.nnz,
        all_ssm_fp.max,
    );
    if let Some(fp) = k_fp {
        emit_tap("cache", Some(0), None, "past_k_l0", &fp);
    }
    if let Some(fp) = ssm_fp {
        emit_tap("cache", Some(0), None, "ssm_state_l0", &fp);
    }
    emit_tap("cache", Some(0), None, "all_k", &all_k_fp);
    emit_tap("cache", Some(0), None, "all_ssm", &all_ssm_fp);
}
