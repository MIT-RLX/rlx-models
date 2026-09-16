// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Lightweight autocorrelation F0 for on-device speech (no neural weights).

/// Coarse perceived gender from source speech pitch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechGender {
    Female,
    Male,
    Unknown,
}

/// Estimate median F0 (Hz) for a mono PCM slice, or `None` if unvoiced / too short.
pub fn estimate_f0_hz(pcm: &[f32], sample_rate: u32) -> Option<f32> {
    if pcm.len() < sample_rate as usize / 5 {
        return None;
    }
    let sr = sample_rate as usize;
    let frame = (sr * 40 / 1000).max(256);
    let hop = (sr * 10 / 1000).max(64);
    let min_period = (sr as f32 / 300.0).floor() as usize;
    let max_period = (sr as f32 / 80.0).ceil() as usize;
    if max_period <= min_period + 2 || frame <= max_period {
        return None;
    }

    let energy_thresh = {
        let mut sum = 0.0f32;
        for &s in pcm.iter().take(pcm.len().min(sr * 2)) {
            sum += s * s;
        }
        let n = pcm.len().min(sr * 2).max(1) as f32;
        (sum / n) * 0.15
    };

    let mut f0s: Vec<f32> = Vec::new();
    let mut i = 0usize;
    while i + frame <= pcm.len() {
        let win = &pcm[i..i + frame];
        let mut e = 0.0f32;
        for &s in win {
            e += s * s;
        }
        e /= frame as f32;
        if e >= energy_thresh
            && let Some(f0) = frame_f0_acf(win, sample_rate, min_period, max_period)
        {
            f0s.push(f0);
        }
        i += hop;
        if f0s.len() >= 80 {
            break;
        }
    }

    if f0s.len() < 3 {
        return None;
    }
    f0s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(f0s[f0s.len() / 2])
}

fn frame_f0_acf(
    frame: &[f32],
    sample_rate: u32,
    min_period: usize,
    max_period: usize,
) -> Option<f32> {
    let n = frame.len();
    let mut best_corr = 0.0f32;
    let mut r0 = 0.0f32;
    for &s in frame {
        r0 += s * s;
    }
    if r0 < 1e-8 {
        return None;
    }

    let mut peaks: Vec<(usize, f32)> = Vec::new();
    for lag in min_period..=max_period.min(n / 2) {
        let mut corr = 0.0f32;
        for i in 0..(n - lag) {
            corr += frame[i] * frame[i + lag];
        }
        corr /= r0;
        if corr > best_corr {
            best_corr = corr;
        }
        if corr >= 0.35 {
            peaks.push((lag, corr));
        }
    }

    if best_corr < 0.35 || peaks.is_empty() {
        return None;
    }

    // Among strong peaks, take the *shortest* lag near the max (fundamental).
    // Pure tones also peak at 2×/3× period; preferring max-corr alone often
    // picks a subharmonic and reads female speech as male.
    let thresh = best_corr * 0.90;
    let best_lag = peaks
        .iter()
        .filter(|(_, c)| *c >= thresh)
        .map(|(l, _)| *l)
        .min()?;

    Some(sample_rate as f32 / best_lag as f32)
}

/// Map F0 to coarse gender bands (`female_hz` / `male_hz` thresholds).
pub fn gender_from_f0(f0_hz: f32, female_f0_hz: f32, male_f0_hz: f32) -> SpeechGender {
    if f0_hz >= female_f0_hz {
        SpeechGender::Female
    } else if f0_hz <= male_f0_hz {
        SpeechGender::Male
    } else {
        SpeechGender::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / sr as f32).sin() * 0.5)
            .collect()
    }

    #[test]
    fn low_pitch_male_band() {
        let f0 = estimate_f0_hz(&tone(120.0, 16_000, 0.8), 16_000).unwrap();
        assert!(f0 > 100.0 && f0 < 140.0, "f0={f0}");
        assert_eq!(gender_from_f0(f0, 165.0, 145.0), SpeechGender::Male);
    }

    #[test]
    fn high_pitch_female_band() {
        let f0 = estimate_f0_hz(&tone(210.0, 16_000, 0.8), 16_000).unwrap();
        assert!(f0 > 180.0 && f0 < 240.0, "f0={f0}");
        assert_eq!(gender_from_f0(f0, 165.0, 145.0), SpeechGender::Female);
    }
}
