// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! 80-bin log-mel frontend.
//!
//! Matches `tools/audio_io.py` defaults:
//! 25 ms / 10 ms frames @ 16 kHz, deterministic dither + pre-emphasis, povey window.

use crate::spec::MEL_BINS;
use anyhow::{Result, bail};

/// Frame shift / length at 16 kHz (10 ms / 25 ms) — Kaldi-style fbank defaults from mini.json.
pub const FRAME_SHIFT: usize = 160;
pub const FRAME_LENGTH: usize = 400;
pub const SAMPLE_RATE: u32 = 16_000;
/// Kaldi dither on int16-scale samples (see `fbank-with-audio-analytics.dither`).
pub const DITHER: f32 = 1.0;
pub const PREEMPH: f32 = 0.97;

/// Compute `[n_frames, MEL_BINS]` log-mel filterbank from mono PCM f32 in [-1, 1].
pub fn log_mel_fbank(pcm: &[f32], sample_rate: u32) -> Result<Vec<Vec<f32>>> {
    log_mel_fbank_inner(pcm, sample_rate, true)
}

/// ASR alias — same as [`log_mel_fbank`] (seed-0 dither enabled).
pub fn log_mel_fbank_asr(pcm: &[f32], sample_rate: u32) -> Result<Vec<Vec<f32>>> {
    log_mel_fbank_inner(pcm, sample_rate, true)
}

fn log_mel_fbank_inner(pcm: &[f32], sample_rate: u32, dither: bool) -> Result<Vec<Vec<f32>>> {
    if sample_rate != 8_000 && sample_rate != 16_000 {
        bail!("unsupported sample rate {sample_rate}");
    }
    let pcm = if sample_rate == 16_000 {
        pcm.to_vec()
    } else {
        upsample_2x(pcm)
    };
    if pcm.len() < FRAME_LENGTH {
        return Ok(Vec::new());
    }
    let mut pcm = pcm;
    for x in pcm.iter_mut() {
        *x *= 32768.0;
    }
    if dither {
        // Match tools/audio_io.py: Gaussian dither (deterministic LCG → Box–Muller).
        let mut state = 0x1234_5678u32;
        for x in pcm.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u1 = ((state >> 8) as f32 / 16_777_216.0).clamp(1e-7, 1.0 - 1e-7);
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = (state >> 8) as f32 / 16_777_216.0;
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            *x += DITHER * z;
        }
    }
    let mean = pcm.iter().sum::<f32>() / pcm.len() as f32;
    for x in pcm.iter_mut() {
        *x -= mean;
    }
    if pcm.len() > 1 {
        for i in (1..pcm.len()).rev() {
            pcm[i] -= PREEMPH * pcm[i - 1];
        }
    }
    let n_fft = 512;
    let filters = mel_filterbank(SAMPLE_RATE, n_fft, MEL_BINS);
    let window: Vec<f32> = (0..FRAME_LENGTH)
        .map(|i| {
            // numpy.hanning / Kaldi povey: 0.5 - 0.5*cos(2π n/(N-1)), then ^0.85
            let x = 2.0 * std::f32::consts::PI * i as f32 / (FRAME_LENGTH as f32 - 1.0);
            let hann = 0.5 - 0.5 * x.cos();
            hann.powf(0.85)
        })
        .collect();
    let n_frames = 1 + (pcm.len() - FRAME_LENGTH) / FRAME_SHIFT;
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * FRAME_SHIFT;
        let mut frame = vec![0f32; n_fft];
        for i in 0..FRAME_LENGTH {
            frame[i] = pcm[start + i] * window[i];
        }
        let power = rfft_power(&frame);
        let mut mel = vec![0f32; MEL_BINS];
        for (b, filt) in filters.iter().enumerate() {
            let mut e = 0f32;
            for (k, &w) in filt.iter().enumerate() {
                e += w * power[k];
            }
            mel[b] = (e.max(1e-10)).ln();
        }
        out.push(mel);
    }
    Ok(out)
}

fn upsample_2x(pcm: &[f32]) -> Vec<f32> {
    let mut o = Vec::with_capacity(pcm.len() * 2);
    for i in 0..pcm.len() {
        o.push(pcm[i]);
        let n = if i + 1 < pcm.len() {
            pcm[i + 1]
        } else {
            pcm[i]
        };
        o.push(0.5 * (pcm[i] + n));
    }
    o
}

fn mel_filterbank(sr: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let fmin = 20.0;
    let fmax = (sr as f32) / 2.0;
    let n_freqs = n_fft / 2 + 1;
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |m: f32| 700.0 * (10f32.powf(m / 2595.0) - 1.0);
    let mmin = hz_to_mel(fmin);
    let mmax = hz_to_mel(fmax);
    let mut mels = Vec::with_capacity(n_mels + 2);
    for i in 0..n_mels + 2 {
        mels.push(mmin + (mmax - mmin) * i as f32 / (n_mels as f32 + 1.0));
    }
    let hz: Vec<f32> = mels.iter().map(|&m| mel_to_hz(m)).collect();
    let mut bins: Vec<usize> = hz
        .iter()
        .map(|&f| {
            let b = ((n_fft as f32 + 1.0) * f / sr as f32).floor() as isize;
            b.clamp(0, (n_fft / 2) as isize) as usize
        })
        .collect();
    let mut filters = vec![vec![0f32; n_freqs]; n_mels];
    for m in 0..n_mels {
        let left = bins[m];
        let mut center = bins[m + 1];
        let mut right = bins[m + 2];
        // Match tools/audio_io.py::_mel_filterbank width bump.
        if center == left {
            center += 1;
        }
        if right == center {
            right += 1;
        }
        let rise = center - left;
        for (i, k) in (left..center).enumerate() {
            if k < n_freqs && rise > 0 {
                filters[m][k] = i as f32 / rise as f32;
            }
        }
        let fall = right - center;
        for (i, k) in (center..right).enumerate() {
            if k < n_freqs && fall > 0 {
                filters[m][k] = 1.0 - i as f32 / fall as f32;
            }
        }
        bins[m + 1] = center;
        bins[m + 2] = right;
    }
    filters
}

fn rfft_power(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let n_freqs = n / 2 + 1;
    let mut out = vec![0f32; n_freqs];
    for k in 0..n_freqs {
        let mut re = 0f32;
        let mut im = 0f32;
        for (t, &x) in frame.iter().enumerate() {
            let ang = -2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
            re += x * ang.cos();
            im += x * ang.sin();
        }
        out[k] = re * re + im * im;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbank_shape() {
        let pcm = vec![0.1f32; 16_000];
        let m = log_mel_fbank(&pcm, 16_000).unwrap();
        assert!(!m.is_empty());
        assert_eq!(m[0].len(), MEL_BINS);
    }

    #[test]
    fn dither_changes_fbank() {
        let pcm = vec![0.01f32; 16_000];
        let a = log_mel_fbank_inner(&pcm, 16_000, false).unwrap();
        let b = log_mel_fbank_inner(&pcm, 16_000, true).unwrap();
        assert_eq!(a.len(), b.len());
        let da: f32 = a
            .iter()
            .flat_map(|f| f.iter())
            .zip(b.iter().flat_map(|f| f.iter()))
            .map(|(&x, &y)| (x - y).abs())
            .sum();
        assert!(da > 0.1, "dither should change mel energy, da={da}");
    }

    #[test]
    fn dump_ask_not_mel_for_parity() {
        let wav = std::path::Path::new("/tmp/ask_not_16k.wav");
        if !wav.is_file() {
            return;
        }
        let (pcm, sr) = read_wav_mono_pcm(wav);
        let mel = log_mel_fbank(&pcm, sr).unwrap();
        let flat: Vec<u8> = mel.iter().flatten().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write("/tmp/rust_ask_mel.bin", flat).unwrap();
        std::fs::write(
            "/tmp/rust_ask_mel_shape.txt",
            format!(
                "{} {}\n",
                mel.len(),
                mel.first().map(|r| r.len()).unwrap_or(0)
            ),
        )
        .unwrap();
    }

    #[test]
    fn dump_paired_wav_mels() {
        let out = std::path::Path::new("/tmp/rust_asr_mels");
        let _ = std::fs::create_dir_all(out);
        let wavs = [
            "/tmp/ask_not_16k.wav",
            "/tmp/moon_16k.wav",
            "/tmp/rlx_intro_16k.wav",
            "/tmp/voice_chat_question_16k.wav",
            "/tmp/voice_chat_reply_16k.wav",
            "/tmp/jfk_16k.wav",
            "/tmp/jfk_voice_clone_16k.wav",
            "/tmp/jfk_rust_speech_16k.wav",
            "/tmp/prompt_16k.wav",
            "/tmp/motor_60s_16k.wav",
            "/tmp/motor_30s_16k.wav",
        ];
        for w in wavs {
            let path = std::path::Path::new(w);
            if !path.is_file() {
                continue;
            }
            let (pcm, sr) = read_wav_mono_pcm(path);
            let mel = match log_mel_fbank(&pcm, sr) {
                Ok(m) if !m.is_empty() => m,
                _ => continue,
            };
            // Keep full stem (e.g. ask_not_16k) so dumps match CLI wav names.
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let flat: Vec<u8> = mel.iter().flatten().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(out.join(format!("{stem}.bin")), flat).unwrap();
            std::fs::write(
                out.join(format!("{stem}.shape")),
                format!("{} {}\n", mel.len(), mel[0].len()),
            )
            .unwrap();
        }
    }

    fn read_wav_mono_pcm(wav: &std::path::Path) -> (Vec<f32>, u32) {
        let bytes = std::fs::read(wav).unwrap();
        let mut off = 12usize;
        let mut sr = 16_000u32;
        let mut ch = 1u16;
        let mut data_off = None;
        let mut data_sz = 0usize;
        while off + 8 <= bytes.len() {
            let id = &bytes[off..off + 4];
            let sz = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
            off += 8;
            if id == b"fmt " && sz >= 16 {
                ch = u16::from_le_bytes(bytes[off + 2..off + 4].try_into().unwrap());
                sr = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            } else if id == b"data" {
                data_off = Some(off);
                data_sz = sz;
                break;
            }
            off += sz;
        }
        let off = data_off.expect("data chunk");
        let samples: Vec<i16> = bytes[off..off + data_sz]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let pcm: Vec<f32> = if ch <= 1 {
            samples.iter().map(|&s| s as f32 / 32768.0).collect()
        } else {
            samples
                .chunks_exact(ch as usize)
                .map(|f| f.iter().map(|&s| s as f32).sum::<f32>() / (ch as f32 * 32768.0))
                .collect()
        };
        (pcm, sr)
    }
}
