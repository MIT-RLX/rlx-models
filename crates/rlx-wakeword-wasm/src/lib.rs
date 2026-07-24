// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! WASM entry for wakeword core — Node, browser main thread, or dedicated Web Worker.
//!
//! Timing uses `Date.now()` (no `window` / DOM). Drive [`WakeSession`] from a module
//! worker via `postMessage` (see `web/wake_worker.js`).

use rlx_wakeword_core::{
    MelConfig, MelFrontend, SAMPLE_RATE_16K, TernaryOpts, WakeCnn, WakeCnnConfig, WakeCnnWeights,
    pack_trits,
};
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

/// Installs panic → console (works in window and dedicated workers).
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(target_arch = "wasm32")]
    {
        use std::sync::Once;
        static HOOK: Once = Once::new();
        HOOK.call_once(|| {
            std::panic::set_hook(Box::new(|info| {
                let msg = format!("rlx-wakeword-wasm panic: {info}");
                web_sys::console::error_1(&JsValue::from_str(&msg));
            }));
        });
    }
}

/// True: no DOM / `window` required (Web Worker safe).
#[wasm_bindgen]
pub fn worker_safe() -> bool {
    true
}

#[wasm_bindgen]
pub fn sample_rate() -> u32 {
    SAMPLE_RATE_16K as u32
}

#[derive(Clone, Copy)]
enum WeightMode {
    F32,
    TernaryFc,
    TernaryAll,
}

impl WeightMode {
    fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::TernaryFc => "tern-fc",
            Self::TernaryAll => "tern-all",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "tern-fc" | "fc" => Self::TernaryFc,
            "tern-all" | "all" => Self::TernaryAll,
            _ => Self::F32,
        }
    }
}

fn tone(seconds: f32, freq_hz: f32, amp: f32) -> Vec<f32> {
    let n = (seconds * SAMPLE_RATE_16K as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE_16K as f32;
            (t * freq_hz * core::f32::consts::TAU).sin() * amp
        })
        .collect()
}

fn bias_bytes(w: &WakeCnnWeights) -> usize {
    (w.conv1_b.len()
        + w.conv2_b.len()
        + w.conv3_b.len()
        + w.fc1_b.len()
        + w.fc2_b.len())
        * 4
}

fn weight_storage_bytes(mode: WeightMode) -> usize {
    let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
    match mode {
        WeightMode::F32 => {
            (w.conv1_w.len()
                + w.conv1_b.len()
                + w.conv2_w.len()
                + w.conv2_b.len()
                + w.conv3_w.len()
                + w.conv3_b.len()
                + w.fc1_w.len()
                + w.fc1_b.len()
                + w.fc2_w.len()
                + w.fc2_b.len())
                * 4
        }
        WeightMode::TernaryFc => {
            w.ternarize(TernaryOpts::fc_only());
            pack_trits(&w.fc1_w).len()
                + pack_trits(&w.fc2_w).len()
                + (w.conv1_w.len() + w.conv2_w.len() + w.conv3_w.len()) * 4
                + bias_bytes(&w)
        }
        WeightMode::TernaryAll => {
            w.ternarize(TernaryOpts::all_weights());
            pack_trits(&w.conv1_w).len()
                + pack_trits(&w.conv2_w).len()
                + pack_trits(&w.conv3_w).len()
                + pack_trits(&w.fc1_w).len()
                + pack_trits(&w.fc2_w).len()
                + bias_bytes(&w)
        }
    }
}

fn make_heads(n: usize, mode: WeightMode, window_frames: usize) -> Vec<WakeCnn> {
    (0..n)
        .map(|_| {
            let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
            match mode {
                WeightMode::F32 => {}
                WeightMode::TernaryFc => {
                    w.ternarize(TernaryOpts::fc_only());
                }
                WeightMode::TernaryAll => {
                    w.ternarize(TernaryOpts::all_weights());
                }
            }
            WakeCnn::new(w).with_window_frames(window_frames)
        })
        .collect()
}

fn hop_samples(hop_ms: u32) -> usize {
    ((hop_ms as u64 * SAMPLE_RATE_16K as u64) / 1000) as usize
}

fn window_frames_for(context_ms: f32) -> usize {
    let samples = (context_ms / 1000.0 * SAMPLE_RATE_16K as f32) as usize;
    (samples / MelConfig::default().hop_length.max(1)).max(8)
}

fn run_stream(heads: &mut [WakeCnn], pcm: &[f32], hop: usize) -> usize {
    let mut mel = MelFrontend::new(MelConfig::default());
    let mut hops = 0usize;
    let mut off = 0usize;
    while off + hop <= pcm.len() {
        let chunk = &pcm[off..off + hop];
        off += hop;
        let frames = mel.push(chunk);
        for h in heads.iter_mut() {
            let _ = h.push_mel_frames(&frames);
        }
        hops += 1;
    }
    hops
}

/// Streaming multi-phrase session (Web Worker / main thread).
#[wasm_bindgen]
pub struct WakeSession {
    heads: Vec<WakeCnn>,
    mel: MelFrontend,
    hop: usize,
    pcm_buf: Vec<f32>,
    mode: String,
}

#[wasm_bindgen]
impl WakeSession {
    /// `mode`: `f32` | `tern-fc` | `tern-all`.
    #[wasm_bindgen(constructor)]
    pub fn new(n_phrases: u32, hop_ms: u32, mode: &str) -> WakeSession {
        let hop_ms = match hop_ms {
            20 | 32 | 40 | 80 => hop_ms,
            _ => 40,
        };
        let n = (n_phrases as usize).clamp(1, 32);
        let mode_e = WeightMode::from_str(mode);
        let heads = make_heads(n, mode_e, window_frames_for(1200.0));
        WakeSession {
            heads,
            mel: MelFrontend::new(MelConfig::default()),
            hop: hop_samples(hop_ms).max(1),
            pcm_buf: Vec::new(),
            mode: mode_e.label().into(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn phrase_count(&self) -> u32 {
        self.heads.len() as u32
    }

    #[wasm_bindgen(getter)]
    pub fn hop_samples(&self) -> u32 {
        self.hop as u32
    }

    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> String {
        self.mode.clone()
    }

    pub fn reset(&mut self) {
        self.mel.reset();
        self.pcm_buf.clear();
        for h in &mut self.heads {
            h.reset();
        }
    }

    /// Push 16 kHz mono PCM. Returns flat scores: for each completed hop,
    /// `phrase_count` floats (one score per head), row-major.
    pub fn push(&mut self, pcm: &[f32]) -> Vec<f32> {
        self.pcm_buf.extend_from_slice(pcm);
        let mut out = Vec::new();
        let n = self.heads.len();
        while self.pcm_buf.len() >= self.hop {
            let chunk: Vec<f32> = self.pcm_buf.drain(..self.hop).collect();
            let frames = self.mel.push(&chunk);
            for h in &mut self.heads {
                out.push(h.push_mel_frames(&frames));
            }
            debug_assert!(out.len().is_multiple_of(n));
        }
        out
    }

    /// Like [`Self::push`], but only the max score per phrase across hops in this call.
    pub fn push_peak(&mut self, pcm: &[f32]) -> Vec<f32> {
        let n = self.heads.len();
        let mut peak = vec![0.0f32; n];
        let flat = self.push(pcm);
        for (i, &s) in flat.iter().enumerate() {
            let p = i % n;
            if s > peak[p] {
                peak[p] = s;
            }
        }
        peak
    }
}

/// Finite score on a synth tone (preflight / self-test).
#[wasm_bindgen]
pub fn smoke_score() -> f32 {
    let mut cnn = WakeCnn::new(WakeCnnWeights::stub(WakeCnnConfig::lite())).with_window_frames(40);
    let mut mel = MelFrontend::new(MelConfig::default());
    let pcm = tone(1.2, 440.0, 0.25);
    let frames = mel.push(&pcm);
    cnn.push_mel_frames(&frames)
}

/// Multi-phrase f32 / tern-fc / tern-all bench table (Node / browser / worker).
///
/// `modes` is comma-separated: `f32,tern-fc,tern-all` (default all three).
#[wasm_bindgen]
pub fn bench_multi_phrase(hop_ms: u32, n_min: u32, n_max: u32, modes: &str) -> String {
    let hop_ms = match hop_ms {
        20 | 32 | 40 | 80 => hop_ms,
        _ => 40,
    };
    let n_min = n_min.clamp(1, 16) as usize;
    let n_max = n_max.clamp(n_min as u32, 16) as usize;
    let hop = hop_samples(hop_ms);
    let window_frames = window_frames_for(1200.0);

    let mut pcm = Vec::new();
    for (i, f) in [220.0, 330.0, 440.0, 550.0, 660.0].iter().enumerate() {
        pcm.extend(tone(1.2, *f, 0.25 + 0.02 * i as f32));
        pcm.extend(vec![0.0f32; (0.4 * SAMPLE_RATE_16K as f32) as usize]);
    }
    let audio_s = pcm.len() as f64 / SAMPLE_RATE_16K as f64;

    let mode_list: Vec<WeightMode> = if modes.trim().is_empty() {
        vec![
            WeightMode::F32,
            WeightMode::TernaryFc,
            WeightMode::TernaryAll,
        ]
    } else {
        modes
            .split(',')
            .map(|s| WeightMode::from_str(s.trim()))
            .collect()
    };

    let mut out = String::new();
    out.push_str(&format!(
        "rlx-wakeword WASM bench (hop={hop_ms} ms, audio={audio_s:.2}s, wasm32)\n\n"
    ));
    out.push_str(&format!(
        "{:>8}  {:>3}  {:>6}  {:>10}  {:>9}  {:>8}  {:>11}\n",
        "mode", "N", "hops", "mean_hop_us", "wall_ms", "RTF", "weights_KiB"
    ));
    out.push_str(&"-".repeat(78));
    out.push('\n');

    let warmup = 1usize;
    let rounds = 3usize;

    for mode in mode_list {
        for n in n_min..=n_max {
            let mut heads = make_heads(n, mode, window_frames);
            for _ in 0..warmup {
                for h in &mut heads {
                    h.reset();
                }
                let _ = run_stream(&mut heads, &pcm, hop);
            }
            let mut total_ms = 0.0f64;
            let mut hops = 0usize;
            for _ in 0..rounds {
                for h in &mut heads {
                    h.reset();
                }
                let t0 = now_ms();
                hops = run_stream(&mut heads, &pcm, hop);
                total_ms += now_ms() - t0;
            }
            let wall_ms = total_ms / rounds as f64;
            let mean_hop_us = (wall_ms * 1000.0) / hops.max(1) as f64;
            let rtf = (wall_ms / 1000.0) / audio_s;
            let w_kib = (weight_storage_bytes(mode) * n) as f64 / 1024.0;
            out.push_str(&format!(
                "{:>8}  {:>3}  {:>6}  {:>10.1}  {:>9.2}  {:>8.4}  {:>11.1}\n",
                mode.label(),
                n,
                hops,
                mean_hop_us,
                wall_ms,
                rtf,
                w_kib
            ));
        }
        out.push('\n');
    }

    out.push_str("── size @ N (packed storage KiB) ──\n");
    out.push_str(&format!(
        "{:>3}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}\n",
        "N", "f32", "tern-fc", "tern-all", "fc÷f32", "all÷f32"
    ));
    out.push_str(&"-".repeat(62));
    out.push('\n');
    for n in n_min..=n_max {
        let f32b = weight_storage_bytes(WeightMode::F32) * n;
        let fcb = weight_storage_bytes(WeightMode::TernaryFc) * n;
        let allb = weight_storage_bytes(WeightMode::TernaryAll) * n;
        out.push_str(&format!(
            "{:>3}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.2}  {:>10.2}\n",
            n,
            f32b as f64 / 1024.0,
            fcb as f64 / 1024.0,
            allb as f64 / 1024.0,
            fcb as f64 / f32b as f64,
            allb as f64 / f32b as f64
        ));
    }
    out
}
