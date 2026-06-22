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

//! FunASR `WavFrontend`: Kaldi-compatible log-mel **fbank** (matching
//! `torchaudio.compliance.kaldi.fbank` with FunASR's defaults), then **LFR**
//! (low-frame-rate stacking, `m`/`n`) and **CMVN** (`(x + neg_mean) * inv_std`
//! from a Kaldi-nnet `am.mvn`).
//!
//! The fbank pipeline per frame, in Kaldi order: remove DC offset → 0.97
//! pre-emphasis (replicate-padded) → Hamming window → zero-pad to the next
//! power of two → power spectrum → Kaldi triangular mel banks (mel =
//! `1127·ln(1 + f/700)`, no Slaney normalization) → `ln(max(e, ε))`.
//! `dither` is fixed at 0 for deterministic inference.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rustfft::num_complex::Complex;

/// Frontend hyper-parameters (FunASR `frontend_conf`).
#[derive(Debug, Clone)]
pub struct FrontendConfig {
    /// Audio sample rate (Hz).
    pub sample_rate: usize,
    /// Number of mel filterbank bins.
    pub n_mels: usize,
    /// Analysis window length (ms).
    pub frame_length_ms: f32,
    /// Frame hop (ms).
    pub frame_shift_ms: f32,
    /// Mel low cutoff (Hz).
    pub low_freq: f32,
    /// Mel high cutoff (Hz); 0 ⇒ Nyquist.
    pub high_freq: f32,
    /// Pre-emphasis coefficient.
    pub preemphasis: f32,
    /// LFR frame-stacking count.
    pub lfr_m: usize,
    /// LFR frame hop.
    pub lfr_n: usize,
}

impl Default for FrontendConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_mels: 80,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            low_freq: 20.0,
            high_freq: 0.0,
            preemphasis: 0.97,
            lfr_m: 7,
            lfr_n: 6,
        }
    }
}

/// Computed fbank/LFR feature matrix, row-major `[n_frames, feat_dim]`.
#[derive(Debug, Clone)]
pub struct Fbank {
    /// Number of frames (rows).
    pub n_frames: usize,
    /// Feature dimension (columns).
    pub feat_dim: usize,
    /// Row-major `[n_frames, feat_dim]` data.
    pub data: Vec<f32>,
}

/// Cepstral mean/variance normalization stats parsed from a Kaldi `am.mvn`.
#[derive(Debug, Clone)]
pub struct Cmvn {
    /// Additive shift (negated means), length = feature dim.
    pub neg_mean: Vec<f32>,
    /// Multiplicative rescale (inverse std), length = feature dim.
    pub inv_std: Vec<f32>,
}

impl Cmvn {
    /// Parse the Kaldi-nnet text `am.mvn` (an `<AddShift>` vector followed by a
    /// `<Rescale>` vector, each wrapped in `[ ... ]`).
    pub fn parse(text: &str) -> Result<Self> {
        let vecs = parse_bracket_vectors(text);
        if vecs.len() < 2 {
            bail!(
                "am.mvn: expected 2 bracketed vectors (AddShift, Rescale), got {}",
                vecs.len()
            );
        }
        Ok(Self {
            neg_mean: vecs[0].clone(),
            inv_std: vecs[1].clone(),
        })
    }

    /// Parse a Kaldi `am.mvn` file.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::parse(&text)
    }

    /// Apply `(x + neg_mean) * inv_std` over each `feat_dim`-wide row in place.
    pub fn apply(&self, fb: &mut Fbank) {
        let d = fb.feat_dim;
        if self.neg_mean.len() != d || self.inv_std.len() != d {
            return; // shape mismatch: leave features untouched
        }
        for f in 0..fb.n_frames {
            let row = &mut fb.data[f * d..(f + 1) * d];
            for (i, v) in row.iter_mut().enumerate() {
                *v = (*v + self.neg_mean[i]) * self.inv_std[i];
            }
        }
    }
}

/// Load `am.mvn` only when `config.yaml`'s `frontend_conf.cmvn_file` is set to a
/// real path (Paraformer); returns `None` when it is `null` (SenseVoice ships an
/// unused `am.mvn` but disables CMVN).
pub fn load_configured_cmvn(dir: &Path) -> Option<Cmvn> {
    let text = std::fs::read_to_string(dir.join("config.yaml")).ok()?;
    let mut enabled = false;
    for line in text.lines() {
        if let Some(v) = line.trim().strip_prefix("cmvn_file:") {
            let v = v.trim();
            enabled = !(v.is_empty()
                || v.eq_ignore_ascii_case("null")
                || v.eq_ignore_ascii_case("none")
                || v == "~");
        }
    }
    if !enabled {
        return None;
    }
    let p = dir.join("am.mvn");
    if p.is_file() {
        Cmvn::load(&p).ok()
    } else {
        None
    }
}

fn parse_bracket_vectors(text: &str) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    let mut cur: Option<Vec<f32>> = None;
    for tok in text.split_whitespace() {
        match tok {
            "[" => cur = Some(Vec::new()),
            "]" => {
                if let Some(v) = cur.take() {
                    out.push(v);
                }
            }
            _ => {
                if let Some(v) = cur.as_mut() {
                    if let Ok(x) = tok.parse::<f32>() {
                        v.push(x);
                    }
                }
            }
        }
    }
    out
}

/// The complete WavFrontend: fbank → LFR → optional CMVN.
pub struct WavFrontend {
    cfg: FrontendConfig,
    cmvn: Option<Cmvn>,
    window: Vec<f32>,
    filters: Vec<f32>, // [n_mels, n_freq] row-major
    n_fft: usize,
    n_freq: usize,
    frame_len: usize,
    frame_shift: usize,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl WavFrontend {
    /// Build a frontend from its config and optional CMVN stats.
    pub fn new(cfg: FrontendConfig, cmvn: Option<Cmvn>) -> Self {
        let frame_len = (cfg.frame_length_ms * cfg.sample_rate as f32 / 1000.0).round() as usize;
        let frame_shift = (cfg.frame_shift_ms * cfg.sample_rate as f32 / 1000.0).round() as usize;
        let n_fft = frame_len.next_power_of_two();
        let n_freq = n_fft / 2 + 1;
        let window = hamming(frame_len);
        let high = if cfg.high_freq > 0.0 {
            cfg.high_freq
        } else {
            cfg.sample_rate as f32 / 2.0
        };
        let filters = kaldi_mel_banks(
            cfg.sample_rate as f32,
            n_fft,
            cfg.n_mels,
            cfg.low_freq,
            high,
        );
        let fft = rustfft::FftPlanner::new().plan_fft_forward(n_fft);
        Self {
            cfg,
            cmvn,
            window,
            filters,
            n_fft,
            n_freq,
            frame_len,
            frame_shift,
            fft,
        }
    }

    /// The frontend configuration.
    pub fn config(&self) -> &FrontendConfig {
        &self.cfg
    }

    /// Feature dimension after LFR (`n_mels * lfr_m`).
    pub fn feat_dim(&self) -> usize {
        self.cfg.n_mels * self.cfg.lfr_m
    }

    /// Compute the full feature matrix from mono PCM at the configured rate.
    pub fn extract(&self, pcm: &[f32]) -> Fbank {
        let fbank = self.fbank(pcm);
        let lfr = apply_lfr(&fbank, self.cfg.n_mels, self.cfg.lfr_m, self.cfg.lfr_n);
        let mut out = lfr;
        if let Some(cmvn) = &self.cmvn {
            cmvn.apply(&mut out);
        }
        out
    }

    /// Raw 80-bin Kaldi log-mel fbank `[n_frames, n_mels]`.
    pub fn fbank(&self, pcm: &[f32]) -> Fbank {
        let n_mels = self.cfg.n_mels;
        // snip_edges = True
        let n_frames = if pcm.len() >= self.frame_len {
            1 + (pcm.len() - self.frame_len) / self.frame_shift
        } else {
            0
        };
        let mut data = vec![0.0f32; n_frames * n_mels];
        let mut frame = vec![0.0f32; self.frame_len];
        let mut buf: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; self.n_fft];
        let eps = f32::EPSILON;

        for fi in 0..n_frames {
            let start = fi * self.frame_shift;
            frame.copy_from_slice(&pcm[start..start + self.frame_len]);
            // FunASR/Kaldi scale the [-1,1] waveform to the int16 range before
            // fbank; this keeps the mel-energy log floor (FLT_EPSILON) behaving
            // identically to torchaudio.compliance.kaldi.
            for v in frame.iter_mut() {
                *v *= 32768.0;
            }

            // remove DC offset (subtract mean)
            let mean = frame.iter().sum::<f32>() / self.frame_len as f32;
            for v in frame.iter_mut() {
                *v -= mean;
            }
            // pre-emphasis with replicate left-pad: new[0] = x0 - p*x0
            let p = self.cfg.preemphasis;
            if p != 0.0 {
                for j in (1..self.frame_len).rev() {
                    frame[j] -= p * frame[j - 1];
                }
                frame[0] -= p * frame[0];
            }
            // window + zero-pad to n_fft
            for (j, b) in buf.iter_mut().enumerate() {
                let s = if j < self.frame_len {
                    frame[j] * self.window[j]
                } else {
                    0.0
                };
                *b = Complex { re: s, im: 0.0 };
            }
            self.fft.process(&mut buf);
            // mel banks over power spectrum, log floor
            for mi in 0..n_mels {
                let row = &self.filters[mi * self.n_freq..(mi + 1) * self.n_freq];
                let mut acc = 0.0f32;
                for (bin, &w) in row.iter().enumerate() {
                    if w != 0.0 {
                        let c = buf[bin];
                        acc += w * (c.re * c.re + c.im * c.im);
                    }
                }
                data[fi * n_mels + mi] = acc.max(eps).ln();
            }
        }
        Fbank {
            n_frames,
            feat_dim: n_mels,
            data,
        }
    }
}

/// Kaldi Hamming window: `0.54 - 0.46·cos(2πn/(N-1))`.
fn hamming(n: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n];
    if n <= 1 {
        return vec![1.0; n.max(1)];
    }
    let a = 2.0 * std::f64::consts::PI / (n as f64 - 1.0);
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = (0.54 - 0.46 * (a * i as f64).cos()) as f32;
    }
    w
}

fn hz_to_mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// Kaldi triangular mel filterbank `[n_mels, n_freq]` (no normalization).
fn kaldi_mel_banks(sample_rate: f32, n_fft: usize, n_mels: usize, low: f32, high: f32) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;
    let fft_bin_width = sample_rate / n_fft as f32;
    let mel_low = hz_to_mel(low);
    let mel_high = hz_to_mel(high);
    let delta = (mel_high - mel_low) / (n_mels as f32 + 1.0);
    let mut fb = vec![0.0f32; n_mels * n_freq];
    for bin in 0..n_mels {
        let left = mel_low + bin as f32 * delta;
        let center = mel_low + (bin as f32 + 1.0) * delta;
        let right = mel_low + (bin as f32 + 2.0) * delta;
        for i in 0..n_freq {
            let freq = fft_bin_width * i as f32;
            let mel = hz_to_mel(freq);
            if mel > left && mel < right {
                let w = if mel <= center {
                    (mel - left) / (center - left)
                } else {
                    (right - mel) / (right - center)
                };
                fb[bin * n_freq + i] = w;
            }
        }
    }
    fb
}

/// Low-frame-rate stacking: stack `m` consecutive frames, hop `n`; replicate
/// the first frame `(m-1)/2` times on the left and the last frame to fill the
/// final window. Output `[T_lfr, m*feat_dim]`.
pub fn apply_lfr(fb: &Fbank, feat_dim: usize, m: usize, n: usize) -> Fbank {
    let t = fb.n_frames;
    if t == 0 {
        return Fbank {
            n_frames: 0,
            feat_dim: feat_dim * m,
            data: Vec::new(),
        };
    }
    let left = (m - 1) / 2;
    // padded frame view: `left` copies of frame 0, then the real frames.
    let frame = |idx: i64| -> &[f32] {
        let i = (idx - left as i64).clamp(0, t as i64 - 1) as usize;
        &fb.data[i * feat_dim..(i + 1) * feat_dim]
    };
    // FunASR computes T_lfr from the *original* frame count, then left-pads.
    let t_lfr = t.div_ceil(n);
    let out_dim = feat_dim * m;
    let mut data = vec![0.0f32; t_lfr * out_dim];
    for i in 0..t_lfr {
        let base = i * n; // index into the padded view
        for j in 0..m {
            let src = frame((base + j) as i64);
            let dst = &mut data[i * out_dim + j * feat_dim..i * out_dim + (j + 1) * feat_dim];
            dst.copy_from_slice(src);
        }
    }
    Fbank {
        n_frames: t_lfr,
        feat_dim: out_dim,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbank_shape_and_finite() {
        let fe = WavFrontend::new(FrontendConfig::default(), None);
        let pcm: Vec<f32> = (0..16_000)
            .map(|n| (2.0 * std::f32::consts::PI * 220.0 * n as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let fb = fe.fbank(&pcm);
        // snip_edges: 1 + (16000-400)/160 = 98 frames.
        assert_eq!(fb.n_frames, 98);
        assert_eq!(fb.feat_dim, 80);
        assert!(fb.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn lfr_dim_is_m_times_feat() {
        let fe = WavFrontend::new(FrontendConfig::default(), None);
        let pcm: Vec<f32> = (0..16_000).map(|n| (n as f32 * 0.01).sin() * 0.3).collect();
        let feats = fe.extract(&pcm);
        assert_eq!(feats.feat_dim, 560);
        assert!(feats.n_frames > 0);
        assert!(feats.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn cmvn_parse_and_apply() {
        let txt = "<Nnet> <AddShift> 4 4 <LearnRateCoef> 0 [ -1 -2 -3 -4 ] <Rescale> 4 4 <LearnRateCoef> 0 [ 2 2 2 2 ] </Nnet>";
        let c = Cmvn::parse(txt).unwrap();
        assert_eq!(c.neg_mean, vec![-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(c.inv_std, vec![2.0, 2.0, 2.0, 2.0]);
        let mut fb = Fbank {
            n_frames: 1,
            feat_dim: 4,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        c.apply(&mut fb);
        // (x - mean)*std with neg_mean as additive: (1-1)*2=0, (2-2)*2=0 ...
        assert_eq!(fb.data, vec![0.0, 0.0, 0.0, 0.0]);
    }
}
