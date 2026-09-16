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

//! Reference-audio hygiene and generated-audio sanity checks for the voice
//! cloning crates.
//!
//! Every zero-shot cloner in this workspace takes a user-supplied reference
//! clip and hands it straight to an encoder. That is fine for the curated clips
//! the parity suites use and bad for real recordings, which arrive with DC
//! offset, half a second of room tone at each end, and peaks just over full
//! scale. None of that is visible to a numerical parity test — the port matches
//! the reference implementation exactly and still clones badly — so the checks
//! live here, shared, rather than in whichever crate noticed first.
//!
//! The thresholds follow jamiepine/voicebox, which has had them in front of
//! real users; where a constant looks arbitrary the reason it is that value is
//! written down next to it.
//!
//! Dependency-free, like [`crate::audio`].

/// Knobs for [`preprocess_reference`].
#[derive(Debug, Clone, Copy)]
pub struct ReferencePrep {
    /// Peak amplitude cap, applied only when the input exceeds it. Scaling a
    /// hot recording down keeps it from being rejected as clipping; it cannot
    /// repair clipping that is already baked into the samples.
    pub peak_target: f32,
    /// Edge-silence threshold, in dB below the loudest frame. 40 dB sits below
    /// normal speech dynamic range (~30 dB), so soft trailing syllables
    /// survive while obvious lead-in/lead-out silence goes. librosa's own
    /// default of 60 is considerably more permissive.
    pub trim_top_db: f32,
    /// Silence re-added at each edge *after* trimming, so the encoder has
    /// something to anchor on. Applied only when trimming actually shortened
    /// the clip, and never past the original length — otherwise a clip near
    /// the duration ceiling gets padded over it and rejected as too long.
    pub edge_padding_ms: u32,
}

impl Default for ReferencePrep {
    fn default() -> Self {
        Self {
            peak_target: 0.95,
            trim_top_db: 40.0,
            edge_padding_ms: 100,
        }
    }
}

/// Duration and level limits for [`validate_reference`].
#[derive(Debug, Clone, Copy)]
pub struct ReferenceLimits {
    pub min_seconds: f32,
    pub max_seconds: f32,
    /// Full-clip RMS floor. Catches clips that are silence, near-silence, or
    /// recorded with the wrong input device.
    pub min_rms: f32,
}

impl Default for ReferenceLimits {
    fn default() -> Self {
        Self {
            min_seconds: 2.0,
            max_seconds: 30.0,
            min_rms: 0.01,
        }
    }
}

/// Why a reference clip is unusable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReferenceReject {
    Empty,
    TooShort { seconds: f32, min_seconds: f32 },
    TooLong { seconds: f32, max_seconds: f32 },
    TooQuiet { rms: f32, min_rms: f32 },
}

impl std::fmt::Display for ReferenceReject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "reference audio is empty"),
            Self::TooShort {
                seconds,
                min_seconds,
            } => write!(
                f,
                "reference audio is {seconds:.2}s; at least {min_seconds:.1}s is needed to \
                 characterize a voice"
            ),
            Self::TooLong {
                seconds,
                max_seconds,
            } => write!(
                f,
                "reference audio is {seconds:.2}s; the ceiling is {max_seconds:.1}s"
            ),
            Self::TooQuiet { rms, min_rms } => write!(
                f,
                "reference audio is too quiet (rms {rms:.4} < {min_rms:.4}) — it may be \
                 silent, or recorded from the wrong input"
            ),
        }
    }
}

impl std::error::Error for ReferenceReject {}

/// Root-mean-square level of a whole buffer.
pub fn rms(pcm: &[f32]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt()
}

/// Frame-wise RMS over non-overlapping windows of `frame_len` samples.
fn frame_rms(pcm: &[f32], frame_len: usize) -> Vec<f32> {
    if frame_len == 0 {
        return Vec::new();
    }
    pcm.chunks_exact(frame_len).map(rms).collect()
}

/// First and last sample of the non-silent span, where "silent" means a frame
/// more than `top_db` below the loudest one.
///
/// This is the intent of `librosa.effects.trim` — frame RMS, compared in dB
/// against the clip maximum — using librosa's 2048/512 framing. It is not
/// bit-identical to librosa: the frames here are not centre-padded.
fn nonsilent_span(pcm: &[f32], top_db: f32) -> Option<(usize, usize)> {
    const FRAME: usize = 2048;
    const HOP: usize = 512;
    if pcm.len() < FRAME {
        return if rms(pcm) > 0.0 {
            Some((0, pcm.len()))
        } else {
            None
        };
    }
    let frames: Vec<f32> = (0..=(pcm.len() - FRAME) / HOP)
        .map(|i| rms(&pcm[i * HOP..i * HOP + FRAME]))
        .collect();
    let peak = frames.iter().copied().fold(0f32, f32::max);
    if peak <= 0.0 {
        return None;
    }
    // amplitude_to_db(x, ref=peak) > -top_db  <=>  x > peak * 10^(-top_db/20)
    let floor = peak * 10f32.powf(-top_db / 20.0);
    let first = frames.iter().position(|&r| r > floor)?;
    let last = frames.iter().rposition(|&r| r > floor)?;
    Some((first * HOP, (last * HOP + FRAME).min(pcm.len())))
}

/// Clean a reference clip before it is encoded: drop DC offset, trim edge
/// silence, re-pad the edges, and cap the peak.
///
/// Deliberately conservative — the goal is to accept reasonable real-world
/// recordings, not to rescue badly distorted ones.
pub fn preprocess_reference(pcm: &[f32], sample_rate: u32, prep: &ReferencePrep) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let mean = pcm.iter().sum::<f32>() / pcm.len() as f32;
    let mut out: Vec<f32> = pcm.iter().map(|x| x - mean).collect();

    if let Some((start, end)) = nonsilent_span(&out, prep.trim_top_db)
        && end - start < out.len()
    {
        let pad_each = (sample_rate as usize * prep.edge_padding_ms as usize) / 1000;
        // Never pad past the original length.
        let headroom = (out.len() - (end - start)) / 2;
        let pad = pad_each.min(headroom);
        let mut padded = vec![0f32; pad];
        padded.extend_from_slice(&out[start..end]);
        padded.extend(std::iter::repeat_n(0f32, pad));
        out = padded;
    }

    let peak = out.iter().fold(0f32, |m, x| m.max(x.abs()));
    if peak > prep.peak_target && peak > 0.0 {
        let g = prep.peak_target / peak;
        for x in &mut out {
            *x *= g;
        }
    }
    out
}

/// Check a (preferably already preprocessed) reference clip against `limits`.
pub fn validate_reference(
    pcm: &[f32],
    sample_rate: u32,
    limits: &ReferenceLimits,
) -> Result<(), ReferenceReject> {
    if pcm.is_empty() || sample_rate == 0 {
        return Err(ReferenceReject::Empty);
    }
    let seconds = pcm.len() as f32 / sample_rate as f32;
    if seconds < limits.min_seconds {
        return Err(ReferenceReject::TooShort {
            seconds,
            min_seconds: limits.min_seconds,
        });
    }
    if seconds > limits.max_seconds {
        return Err(ReferenceReject::TooLong {
            seconds,
            max_seconds: limits.max_seconds,
        });
    }
    let level = rms(pcm);
    if level < limits.min_rms {
        return Err(ReferenceReject::TooQuiet {
            rms: level,
            min_rms: limits.min_rms,
        });
    }
    Ok(())
}

/// Share of a clip's energy above `cutoff_hz`, via a 2nd-order Butterworth
/// high-pass. Returns 0 for an empty or silent clip.
///
/// A full FFT would be tidier but this module is deliberately dependency-free,
/// and a biquad is plenty for a smoke alarm.
pub fn high_band_energy_ratio(pcm: &[f32], sample_rate: u32, cutoff_hz: f32) -> f32 {
    if pcm.len() < 4 || sample_rate == 0 || cutoff_hz <= 0.0 {
        return 0.0;
    }
    let nyquist = sample_rate as f32 / 2.0;
    if cutoff_hz >= nyquist {
        return 0.0;
    }
    // RBJ cookbook high-pass, Q = 1/sqrt(2) (Butterworth).
    let w0 = std::f32::consts::TAU * cutoff_hz / sample_rate as f32;
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / std::f32::consts::SQRT_2;
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 + cos_w0) / 2.0) / a0;
    let b1 = (-(1.0 + cos_w0)) / a0;
    let b2 = b0;
    let a1 = (-2.0 * cos_w0) / a0;
    let a2 = (1.0 - alpha) / a0;

    let (mut x1, mut x2, mut y1, mut y2) = (0f32, 0f32, 0f32, 0f32);
    let mut hi = 0f64;
    let mut all = 0f64;
    for &x in pcm {
        let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
        hi += (y as f64) * (y as f64);
        all += (x as f64) * (x as f64);
    }
    if all <= 0.0 { 0.0 } else { (hi / all) as f32 }
}

/// Cutoff the band-limit check measures above, in Hz. A quarter of the 24 kHz
/// codec rate — speech has real energy here (fricatives, breath) and a
/// telephone- or archival-bandwidth recording has almost none.
pub const BAND_LIMIT_CUTOFF_HZ: f32 = 6_000.0;

/// Does this reference look band-limited for a full-band (24 kHz) codec?
///
/// A narrowband recording encodes to latents that describe a muffled voice, and
/// the clone inherits that — quietly. Measured share of energy above 6 kHz: a
/// 24 kHz studio clip 10.6%, a 1961 archival speech excerpt 0.195%, whose clone
/// scores ~0.82 speaker cosine against it where the studio clip reaches ~0.94.
/// Nothing else in the pipeline notices: the clip is not quiet, not clipped, not
/// short, and aligns perfectly.
pub fn looks_band_limited(pcm: &[f32], sample_rate: u32) -> bool {
    sample_rate as f32 / 2.0 > BAND_LIMIT_CUTOFF_HZ
        && high_band_energy_ratio(pcm, sample_rate, BAND_LIMIT_CUTOFF_HZ) < 0.01
}

/// Settings for the generated-audio checks.
#[derive(Debug, Clone, Copy)]
pub struct OutputTrim {
    pub frame_ms: u32,
    /// Absolute frame-RMS floor in dBFS below which a frame counts as silence.
    /// Absolute, not relative to the clip peak as [`ReferencePrep::trim_top_db`]
    /// is — here the question is "did the model stop talking", and a clip that
    /// is quiet throughout must not have its own noise floor promoted to speech.
    pub silence_threshold_db: f32,
    /// An internal silence longer than this ends the utterance. Anything after
    /// it is treated as post-EOS hallucination.
    pub max_internal_silence_ms: u32,
    /// Trailing silence kept after the last speech frame.
    pub trailing_silence_ms: u32,
    /// Cosine fade-out applied to the end, so a mid-waveform cut does not click.
    pub fade_ms: u32,
    /// Also drop leading silence. Off for models that derive their own leading
    /// gap (TADA predicts it), on for models that do not.
    pub trim_leading: bool,
}

impl Default for OutputTrim {
    fn default() -> Self {
        Self {
            frame_ms: 20,
            silence_threshold_db: -40.0,
            max_internal_silence_ms: 1000,
            trailing_silence_ms: 200,
            fade_ms: 30,
            trim_leading: true,
        }
    }
}

/// Per-frame speech mask plus the frame length used to compute it.
fn speech_mask(
    pcm: &[f32],
    sample_rate: u32,
    frame_ms: u32,
    threshold_db: f32,
) -> (Vec<bool>, usize) {
    let frame_len = (sample_rate as usize * frame_ms as usize) / 1000;
    if frame_len == 0 || pcm.len() < frame_len {
        return (Vec::new(), frame_len);
    }
    let threshold = 10f32.powf(threshold_db / 20.0);
    let mask = frame_rms(pcm, frame_len)
        .into_iter()
        .map(|r| r >= threshold)
        .collect();
    (mask, frame_len)
}

/// Does this clip look like `[speech] [long silence] [more speech]`?
///
/// That shape is a reliable signature of a model that missed its end-of-speech
/// token and resumed with hallucinated speech or codec noise. Leading and
/// trailing silence do not count, because they are not bounded by speech on
/// both sides.
pub fn has_tts_runaway(pcm: &[f32], sample_rate: u32, trim: &OutputTrim) -> bool {
    let (mask, _) = speech_mask(pcm, sample_rate, trim.frame_ms, trim.silence_threshold_db);
    if mask.is_empty() {
        return false;
    }
    let max_gap = (trim.max_internal_silence_ms / trim.frame_ms.max(1)) as usize;
    let mut seen_speech = false;
    let mut gap = 0usize;
    for is_speech in mask {
        if is_speech {
            if seen_speech && gap >= max_gap {
                return true;
            }
            seen_speech = true;
            gap = 0;
        } else if seen_speech {
            gap += 1;
        }
    }
    false
}

/// Cut the clip at the first over-long internal silence, drop the trailing
/// silence beyond `trailing_silence_ms`, and fade the end out.
///
/// Returns the input unchanged when it contains no speech at all — an empty
/// return would turn "the model produced nothing useful" into "the model
/// produced nothing", which is harder to diagnose.
pub fn trim_tts_output(pcm: &[f32], sample_rate: u32, trim: &OutputTrim) -> Vec<f32> {
    let (mask, frame_len) = speech_mask(pcm, sample_rate, trim.frame_ms, trim.silence_threshold_db);
    if mask.is_empty() {
        return pcm.to_vec();
    }
    let Some(first_speech) = mask.iter().position(|&s| s) else {
        return pcm.to_vec();
    };
    let start_frame = if trim.trim_leading {
        first_speech.saturating_sub(1) // keep one frame of run-up
    } else {
        0
    };

    let max_gap = (trim.max_internal_silence_ms / trim.frame_ms.max(1)) as usize;
    let mut cut_frame = mask.len();
    let mut gap = 0usize;
    for i in first_speech..mask.len() {
        if mask[i] {
            gap = 0;
        } else {
            gap += 1;
            if gap >= max_gap {
                cut_frame = i - gap + 1;
                break;
            }
        }
    }

    let mut end_frame = cut_frame;
    while end_frame > first_speech && !mask[end_frame - 1] {
        end_frame -= 1;
    }
    let keep = (trim.trailing_silence_ms / trim.frame_ms.max(1)) as usize;
    end_frame = (end_frame + keep).min(cut_frame);

    let start = start_frame * frame_len;
    let end = (end_frame * frame_len).min(pcm.len());
    if end <= start {
        return pcm.to_vec();
    }
    let mut out = pcm[start..end].to_vec();

    let fade = (sample_rate as usize * trim.fade_ms as usize) / 1000;
    if fade > 0 && out.len() > fade {
        let n = out.len();
        for (i, x) in out[n - fade..].iter_mut().enumerate() {
            let t = i as f32 / fade as f32 * std::f32::consts::FRAC_PI_2;
            *x *= t.cos().powi(2);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 24_000;

    fn tone(seconds: f32, amp: f32) -> Vec<f32> {
        let n = (SR as f32 * seconds) as usize;
        (0..n)
            .map(|i| amp * (i as f32 * 440.0 * std::f32::consts::TAU / SR as f32).sin())
            .collect()
    }

    fn silence(seconds: f32) -> Vec<f32> {
        vec![0f32; (SR as f32 * seconds) as usize]
    }

    #[test]
    fn band_limited_references_are_flagged() {
        // Wideband: a click train has energy everywhere.
        let wide: Vec<f32> = (0..SR as usize * 3)
            .map(|i| if i % 64 == 0 { 0.8 } else { 0.0 })
            .collect();
        // Narrowband: a 500 Hz tone has essentially nothing above 6 kHz.
        let narrow = tone(3.0, 0.4);

        let hi_wide = high_band_energy_ratio(&wide, SR, BAND_LIMIT_CUTOFF_HZ);
        let hi_narrow = high_band_energy_ratio(&narrow, SR, BAND_LIMIT_CUTOFF_HZ);
        assert!(hi_wide > 0.05, "wideband share {hi_wide}");
        assert!(hi_narrow < 0.01, "narrowband share {hi_narrow}");
        assert!(!looks_band_limited(&wide, SR));
        assert!(looks_band_limited(&narrow, SR));
    }

    #[test]
    fn band_limit_check_is_skipped_when_nyquist_is_too_low() {
        // At 8 kHz there is no band above 6 kHz to measure, so the check must
        // not fire on every clip.
        let x = tone(3.0, 0.4);
        assert!(!looks_band_limited(&x, 8_000));
    }

    #[test]
    fn preprocess_removes_dc_offset() {
        let mut x = tone(3.0, 0.3);
        for v in &mut x {
            *v += 0.2;
        }
        let y = preprocess_reference(&x, SR, &ReferencePrep::default());
        let mean = y.iter().sum::<f32>() / y.len() as f32;
        assert!(mean.abs() < 1e-3, "residual DC {mean}");
    }

    #[test]
    fn preprocess_trims_edge_silence_but_keeps_padding() {
        let mut x = silence(1.0);
        x.extend(tone(2.0, 0.4));
        x.extend(silence(1.0));
        let y = preprocess_reference(&x, SR, &ReferencePrep::default());
        // 2 s of speech + 100 ms padding per edge, well under the 4 s input.
        let secs = y.len() as f32 / SR as f32;
        assert!(secs > 2.0 && secs < 2.6, "got {secs}s");
    }

    #[test]
    fn preprocess_caps_a_hot_peak_and_leaves_a_quiet_one() {
        let hot = preprocess_reference(&tone(3.0, 1.4), SR, &ReferencePrep::default());
        let peak = hot.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!((peak - 0.95).abs() < 1e-3, "peak {peak}");

        let quiet_in = tone(3.0, 0.2);
        let quiet = preprocess_reference(&quiet_in, SR, &ReferencePrep::default());
        let peak = quiet.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(peak < 0.25, "a quiet clip must not be boosted, got {peak}");
    }

    #[test]
    fn padding_never_extends_past_the_original() {
        // Almost entirely speech: there is no headroom to pad back into.
        let x = tone(4.0, 0.4);
        let y = preprocess_reference(&x, SR, &ReferencePrep::default());
        assert!(y.len() <= x.len(), "{} > {}", y.len(), x.len());
    }

    #[test]
    fn validate_rejects_short_long_and_quiet() {
        let lim = ReferenceLimits::default();
        assert_eq!(
            validate_reference(&[], SR, &lim),
            Err(ReferenceReject::Empty)
        );
        assert!(matches!(
            validate_reference(&tone(1.0, 0.3), SR, &lim),
            Err(ReferenceReject::TooShort { .. })
        ));
        assert!(matches!(
            validate_reference(&tone(31.0, 0.3), SR, &lim),
            Err(ReferenceReject::TooLong { .. })
        ));
        assert!(matches!(
            validate_reference(&tone(5.0, 0.001), SR, &lim),
            Err(ReferenceReject::TooQuiet { .. })
        ));
        assert!(validate_reference(&tone(5.0, 0.3), SR, &lim).is_ok());
    }

    #[test]
    fn runaway_needs_speech_on_both_sides_of_the_gap() {
        let t = OutputTrim::default();
        let mut runaway = tone(1.0, 0.3);
        runaway.extend(silence(1.5));
        runaway.extend(tone(0.5, 0.3));
        assert!(has_tts_runaway(&runaway, SR, &t));

        // The same gap at the end is just trailing silence.
        let mut trailing = tone(1.0, 0.3);
        trailing.extend(silence(1.5));
        assert!(!has_tts_runaway(&trailing, SR, &t));

        // And a short internal pause is normal speech rhythm.
        let mut pause = tone(1.0, 0.3);
        pause.extend(silence(0.3));
        pause.extend(tone(1.0, 0.3));
        assert!(!has_tts_runaway(&pause, SR, &t));
    }

    #[test]
    fn trim_cuts_the_hallucinated_tail() {
        let t = OutputTrim::default();
        let mut x = tone(1.0, 0.3);
        x.extend(silence(1.5));
        x.extend(tone(2.0, 0.3)); // hallucination
        let y = trim_tts_output(&x, SR, &t);
        let secs = y.len() as f32 / SR as f32;
        assert!(secs < 1.5, "expected ~1s kept, got {secs}s");
        assert!(secs > 0.9, "cut too aggressively: {secs}s");
    }

    #[test]
    fn trim_fades_the_end_out() {
        let t = OutputTrim::default();
        let y = trim_tts_output(&tone(2.0, 0.5), SR, &t);
        assert!(y.last().unwrap().abs() < 1e-3, "no fade: {:?}", y.last());
    }

    #[test]
    fn trim_leaves_a_clean_utterance_alone() {
        let t = OutputTrim::default();
        let x = tone(2.0, 0.3);
        let y = trim_tts_output(&x, SR, &t);
        let ratio = y.len() as f32 / x.len() as f32;
        assert!(ratio > 0.95, "clean audio was shortened to {ratio}");
    }

    #[test]
    fn silent_input_is_returned_unchanged() {
        let t = OutputTrim::default();
        let x = silence(1.0);
        assert_eq!(trim_tts_output(&x, SR, &t).len(), x.len());
        assert!(!has_tts_runaway(&x, SR, &t));
    }

    #[test]
    fn leading_trim_can_be_disabled() {
        let mut x = silence(0.5);
        x.extend(tone(1.0, 0.3));
        let keep = OutputTrim {
            trim_leading: false,
            ..OutputTrim::default()
        };
        let drop = OutputTrim::default();
        assert!(trim_tts_output(&x, SR, &keep).len() > trim_tts_output(&x, SR, &drop).len());
    }
}
