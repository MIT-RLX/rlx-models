// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! MetaVoice speaker encoder: 40-mel → 3-layer LSTM → L2-normalized 256-d emb
//! (matches `fam.quantiser.audio.speaker_encoder`).

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};

const SR: u32 = 16_000;
const N_FFT: usize = 400; // 25 ms
const HOP: usize = 160; // 10 ms
const N_MELS: usize = 40;
const N_FREQ: usize = N_FFT / 2 + 1;
const HIDDEN: usize = 256;
const PARTIAL_FRAMES: usize = 160;
const LAYERS: usize = 3;

pub struct SpeakerEncoder {
    // Per layer: weight_ih [4H, in], weight_hh [4H, H], bias [4H]
    layers: Vec<LstmLayer>,
    linear_w: Vec<f32>, // [256, 256] row-major out×in
    linear_b: Vec<f32>,
    window: Vec<f32>,
    fb: Vec<f32>, // [N_MELS, N_FREQ]
    fft: Arc<dyn Fft<f32>>,
}

struct LstmLayer {
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
    in_dim: usize,
}

impl SpeakerEncoder {
    pub fn from_weights(w: &HashMap<String, Vec<f32>>) -> Result<Self> {
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let in_dim = if i == 0 { N_MELS } else { HIDDEN };
            layers.push(LstmLayer {
                w_ih: take(w, &format!("lstm.weight_ih_l{i}"), 4 * HIDDEN * in_dim)?,
                w_hh: take(w, &format!("lstm.weight_hh_l{i}"), 4 * HIDDEN * HIDDEN)?,
                b_ih: take(w, &format!("lstm.bias_ih_l{i}"), 4 * HIDDEN)?,
                b_hh: take(w, &format!("lstm.bias_hh_l{i}"), 4 * HIDDEN)?,
                in_dim,
            });
        }
        Ok(Self {
            layers,
            linear_w: take(w, "linear.weight", HIDDEN * HIDDEN)?,
            linear_b: take(w, "linear.bias", HIDDEN)?,
            window: hann_periodic(N_FFT),
            fb: mel_filterbank(),
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        })
    }

    /// Embed mono PCM (any rate) → 256-d L2-normalized speaker vector.
    pub fn embed_wav(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let mut wav = if sample_rate == SR {
            pcm.to_vec()
        } else {
            resample_linear(pcm, sample_rate, SR)
        };
        trim_top_db(&mut wav, 20.0);
        if wav.len() < HOP * 8 {
            return Err(anyhow!("reference wav too short after trim"));
        }
        let mel = self.power_mel(&wav); // [T, 40]
        self.embed_mel_partials(&mel)
    }

    pub fn embed_wav_path(&self, path: &Path) -> Result<Vec<f32>> {
        let (pcm, sr) = read_wav_mono(path)?;
        self.embed_wav(&pcm, sr)
    }

    pub fn power_mel_for_test(&self, wav: &[f32]) -> Vec<Vec<f32>> {
        self.power_mel(wav)
    }

    pub fn mel_filterbank_for_test() -> Vec<f32> {
        mel_filterbank()
    }

    fn power_mel(&self, wav: &[f32]) -> Vec<Vec<f32>> {
        // librosa default: center=True reflect pad
        let pad = N_FFT / 2;
        let padded = reflect_pad(wav, pad);
        let n_frames = if padded.len() >= N_FFT {
            1 + (padded.len() - N_FFT) / HOP
        } else {
            0
        };
        let mut out = Vec::with_capacity(n_frames);
        let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut power = vec![0f32; N_FREQ];
        for t in 0..n_frames {
            let start = t * HOP;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for (k, p) in power.iter_mut().enumerate() {
                let n = buf[k].norm();
                *p = n * n; // power=2 (librosa default)
            }
            let mut row = vec![0f32; N_MELS];
            for m in 0..N_MELS {
                let mut s = 0.0f32;
                for k in 0..N_FREQ {
                    s += self.fb[m * N_FREQ + k] * power[k];
                }
                row[m] = s;
            }
            out.push(row);
        }
        out
    }

    fn embed_mel_partials(&self, mel: &[Vec<f32>]) -> Result<Vec<f32>> {
        let rate = 1.3f32;
        let min_coverage = 0.75f32;
        let samples_per_frame = HOP;
        let n_samples = mel.len() * samples_per_frame;
        let n_frames = mel.len();
        let frame_step = ((SR as f32 / rate) / samples_per_frame as f32).round() as usize;
        let frame_step = frame_step.max(1);
        let steps = (n_frames + frame_step)
            .saturating_sub(PARTIAL_FRAMES)
            .max(1);
        let mut slices = Vec::new();
        let mut i = 0;
        while i < steps {
            slices.push(i..i + PARTIAL_FRAMES);
            i += frame_step;
        }
        if let Some(last) = slices.last() {
            let wav_start = last.start * samples_per_frame;
            let wav_stop = last.end * samples_per_frame;
            let coverage =
                (n_samples.saturating_sub(wav_start)) as f32 / (wav_stop - wav_start) as f32;
            if coverage < min_coverage && slices.len() > 1 {
                slices.pop();
            }
        }
        anyhow::ensure!(!slices.is_empty(), "no speaker partials");

        let need = slices.last().unwrap().end;
        let mut mel_pad = mel.to_vec();
        while mel_pad.len() < need {
            mel_pad.push(vec![0.0; N_MELS]);
        }

        let mut acc = vec![0.0f32; HIDDEN];
        for sl in &slices {
            let chunk: Vec<f32> = mel_pad[sl.clone()].iter().flatten().copied().collect();
            let emb = self.forward_lstm(&chunk, PARTIAL_FRAMES)?;
            for d in 0..HIDDEN {
                acc[d] += emb[d];
            }
        }
        let n = slices.len() as f32;
        for a in &mut acc {
            *a /= n;
        }
        l2_normalize(&mut acc);
        Ok(acc)
    }

    fn forward_lstm(&self, mel_flat: &[f32], t: usize) -> Result<Vec<f32>> {
        // mel_flat: [T * 40]
        let mut h_prev = vec![vec![0.0f32; HIDDEN]; LAYERS];
        let mut c_prev = vec![vec![0.0f32; HIDDEN]; LAYERS];
        for ti in 0..t {
            let mut x: Vec<f32> = mel_flat[ti * N_MELS..(ti + 1) * N_MELS].to_vec();
            for li in 0..LAYERS {
                let (h, c) = lstm_step(&self.layers[li], &x, &h_prev[li], &c_prev[li]);
                h_prev[li] = h;
                c_prev[li] = c;
                x = h_prev[li].clone();
            }
        }
        // linear + relu + L2 (per-partial; outer mean re-normalizes)
        let h = &h_prev[LAYERS - 1];
        let mut out = vec![0.0f32; HIDDEN];
        for o in 0..HIDDEN {
            let mut s = self.linear_b[o];
            for i in 0..HIDDEN {
                s += self.linear_w[o * HIDDEN + i] * h[i];
            }
            out[o] = s.max(0.0); // ReLU
        }
        l2_normalize(&mut out);
        Ok(out)
    }
}

fn lstm_step(layer: &LstmLayer, x: &[f32], h: &[f32], c: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let hs = HIDDEN;
    let mut gates = vec![0.0f32; 4 * hs];
    for g in 0..4 * hs {
        let mut s = layer.b_ih[g] + layer.b_hh[g];
        for i in 0..layer.in_dim {
            s += layer.w_ih[g * layer.in_dim + i] * x[i];
        }
        for i in 0..hs {
            s += layer.w_hh[g * hs + i] * h[i];
        }
        gates[g] = s;
    }
    let mut h_out = vec![0.0f32; hs];
    let mut c_out = vec![0.0f32; hs];
    for i in 0..hs {
        let ii = sigmoid(gates[i]);
        let ff = sigmoid(gates[hs + i]);
        let gg = gates[2 * hs + i].tanh();
        let oo = sigmoid(gates[3 * hs + i]);
        c_out[i] = ff * c[i] + ii * gg;
        h_out[i] = oo * c_out[i].tanh();
    }
    (h_out, c_out)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v {
        *x /= n;
    }
}

fn take(w: &HashMap<String, Vec<f32>>, name: &str, n: usize) -> Result<Vec<f32>> {
    let v = w.get(name).with_context(|| format!("missing {name}"))?;
    anyhow::ensure!(v.len() == n, "{name}: len {} != {n}", v.len());
    Ok(v.clone())
}

fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

fn hz_to_mel_slaney(f: f64) -> f64 {
    // librosa hz_to_mel(htk=False)
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f64 = 0.068_751_777; // ln(6.4)/27 ≈ log(6.4)/27 * ln(10)? librosa: np.log(6.4)/27
    if f < MIN_LOG_HZ {
        f / F_SP
    } else {
        MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

fn mel_to_hz_slaney(m: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f64 = 0.068_751_777;
    if m < MIN_LOG_MEL {
        F_SP * m
    } else {
        MIN_LOG_HZ * (LOGSTEP * (m - MIN_LOG_MEL)).exp()
    }
}

/// librosa default mel filterbank: Slaney scale + Slaney area norm.
fn mel_filterbank() -> Vec<f32> {
    let f_max = SR as f64 / 2.0;
    let all_freqs: Vec<f64> = (0..N_FREQ)
        .map(|k| k as f64 * f_max / (N_FREQ - 1) as f64)
        .collect();
    let m_min = hz_to_mel_slaney(0.0);
    let m_max = hz_to_mel_slaney(f_max);
    let f_pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz_slaney(m_min + (m_max - m_min) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut fb = vec![0f32; N_MELS * N_FREQ];
    for m in 0..N_MELS {
        let (lo, ctr, hi) = (f_pts[m], f_pts[m + 1], f_pts[m + 2]);
        for (k, &f) in all_freqs.iter().enumerate() {
            let down = (f - lo) / (ctr - lo);
            let up = (hi - f) / (hi - ctr);
            fb[m * N_FREQ + k] = down.min(up).max(0.0) as f32;
        }
        // Slaney area norm: divide by bandwidth in Hz (from librosa.filters.mel).
        let enorm = 2.0 / (f_pts[m + 2] - f_pts[m]);
        for k in 0..N_FREQ {
            fb[m * N_FREQ + k] *= enorm as f32;
        }
    }
    fb
}

fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    if x.is_empty() {
        return vec![0.0; pad * 2];
    }
    let mut out = Vec::with_capacity(x.len() + 2 * pad);
    for i in 0..pad {
        let idx = (pad - i).min(x.len() - 1);
        out.push(x[idx]);
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        let idx = x.len().saturating_sub(2 + i);
        out.push(x[idx]);
    }
    out
}

fn resample_linear(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = x[idx.min(x.len() - 1)];
            let b = x[(idx + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

fn trim_top_db(wav: &mut Vec<f32>, top_db: f32) {
    if wav.is_empty() {
        return;
    }
    let frame = 480usize; // 30 ms @ 16 kHz
    let hop = 160usize;
    let mut energies = Vec::new();
    let mut i = 0;
    while i + frame <= wav.len() {
        let e: f32 = wav[i..i + frame].iter().map(|x| x * x).sum::<f32>() / frame as f32;
        energies.push(e.max(1e-12).log10() * 10.0);
        i += hop;
    }
    if energies.is_empty() {
        return;
    }
    let peak = energies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let thr = peak - top_db;
    let first = energies.iter().position(|&e| e >= thr).unwrap_or(0);
    let last = energies
        .iter()
        .rposition(|&e| e >= thr)
        .unwrap_or(energies.len() - 1);
    let start = first * hop;
    let end = ((last + 1) * hop + frame).min(wav.len());
    *wav = wav[start..end].to_vec();
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path)
        .with_context(|| format!("open reference {}", path.display()))?;
    let spec = r.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    };
    let ch = spec.channels as usize;
    let mono = if ch <= 1 {
        raw
    } else {
        raw.chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_filterbank_shape_and_range() {
        let fb = mel_filterbank();
        assert_eq!(fb.len(), N_MELS * N_FREQ);
        let peak = fb.iter().cloned().fold(0.0f32, f32::max);
        // Slaney-normalized peaks are ≪ 1.
        assert!(peak > 0.001 && peak < 0.1, "peak={peak}");
    }
}
