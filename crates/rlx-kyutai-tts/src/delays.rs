//! Per-codebook delay bookkeeping for Kyutai TTS.
//!
//! Kyutai TTS's `delays` array has `n_q + 1` entries — one per stream:
//!
//! ```text
//!     [0]      → text stream (delay 0)
//!     [1]      → first audio codebook (delay 0)
//!     [2..32]  → remaining audio codebooks (delay 2)
//! ```
//!
//! At inference time, generated tokens for stream `i` arrive `delays[i]` frames
//! later than the canonical "logical step". The generation loop needs to:
//!
//! 1. **Pad** the early frames of every delayed stream with an audio pad token
//!    (`KyutaiTtsConfig::audio_pad_token() = card`).
//! 2. **Track** which logical frame each emitted token corresponds to so the
//!    Mimi codec sees a consistent `[t, n_q]` matrix.
//! 3. **Honour** the demuxed second-stream lead (`second_stream_ahead = 2` frames)
//!    when present.
//!
//! [`StreamLayout`] encapsulates that bookkeeping. It is pure index arithmetic
//! — no model calls — so it can be unit-tested independently.

use crate::config::KyutaiTtsConfig;

/// Resolved per-stream offsets for one generation run.
#[derive(Debug, Clone)]
pub struct StreamLayout {
    /// `[n_q + 1]` per-stream emission delay in frames (mirrors `cfg.delays`).
    pub delays: Vec<i32>,
    /// Audio pad token id (= `card`).
    pub audio_pad: u32,
    /// Maximum delay across all streams (longest pad prefix).
    pub max_delay: i32,
    /// Lead of the demuxed second stream over the primary (Kyutai: 2 frames).
    pub second_stream_ahead: usize,
}

impl StreamLayout {
    /// Resolve a layout for a given config.
    pub fn from_config(cfg: &KyutaiTtsConfig) -> Self {
        let max_delay = cfg.delays.iter().copied().max().unwrap_or(0);
        Self {
            delays: cfg.delays.clone(),
            audio_pad: cfg.audio_pad_token(),
            max_delay,
            second_stream_ahead: cfg.tts_config.second_stream_ahead,
        }
    }

    /// Number of streams (1 text + N audio codebooks).
    pub fn num_streams(&self) -> usize {
        self.delays.len()
    }

    /// Number of audio codebooks (`delays.len() - 1`).
    pub fn num_audio_codebooks(&self) -> usize {
        self.delays.len().saturating_sub(1)
    }

    /// Delay for an audio codebook by codebook index `q` (0-based).
    pub fn audio_delay(&self, q: usize) -> i32 {
        self.delays.get(q + 1).copied().unwrap_or(0)
    }

    /// Delay for the text stream.
    pub fn text_delay(&self) -> i32 {
        self.delays.first().copied().unwrap_or(0)
    }

    /// True if codebook `q` is still in its pad-prefix at logical frame `t`.
    pub fn is_pad_for_audio(&self, q: usize, t: usize) -> bool {
        (t as i32) < self.audio_delay(q)
    }

    /// Convert a logical frame `t` (0-based) → the slot the model writes into
    /// for codebook `q`. Returns `None` if the slot is still padded.
    pub fn slot_for_audio(&self, q: usize, t: usize) -> Option<usize> {
        let d = self.audio_delay(q);
        if (t as i32) < d {
            None
        } else {
            Some((t as i32 - d) as usize)
        }
    }

    /// Logical-frame count needed to fully emit `audio_frames` worth of audio
    /// across every codebook (includes the pad prefix).
    pub fn total_steps_for(&self, audio_frames: usize) -> usize {
        audio_frames + self.max_delay.max(0) as usize
    }

    /// Build the pad-prefix matrix for the first `max_delay` frames.
    ///
    /// Returns `[max_delay, n_q]` filled with `audio_pad`. Codebooks whose
    /// delay is shorter than `max_delay` keep the pad for their own
    /// delay rows and would be overwritten by real tokens afterwards.
    pub fn pad_prefix(&self) -> Vec<Vec<u32>> {
        let n = self.num_audio_codebooks();
        let rows = self.max_delay.max(0) as usize;
        vec![vec![self.audio_pad; n]; rows]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KyutaiTtsConfig;

    #[test]
    fn layout_from_published_config() {
        let cfg = KyutaiTtsConfig::v1_6b_en_fr();
        let lay = StreamLayout::from_config(&cfg);
        assert_eq!(lay.num_streams(), 33);
        assert_eq!(lay.num_audio_codebooks(), 32);
        assert_eq!(lay.audio_pad, cfg.card as u32);
        assert_eq!(lay.text_delay(), 0);
        assert_eq!(lay.audio_delay(0), 0); // first codebook
        assert_eq!(lay.audio_delay(1), 2); // rest
        assert_eq!(lay.audio_delay(31), 2);
        assert_eq!(lay.max_delay, 2);
        assert_eq!(lay.second_stream_ahead, 2);
    }

    #[test]
    fn pad_detection_respects_per_codebook_delay() {
        let cfg = KyutaiTtsConfig::v1_6b_en_fr();
        let lay = StreamLayout::from_config(&cfg);
        // Codebook 0 has delay 0 → never padded.
        assert!(!lay.is_pad_for_audio(0, 0));
        // Codebook 1 has delay 2 → padded at t=0, t=1, not at t=2.
        assert!(lay.is_pad_for_audio(1, 0));
        assert!(lay.is_pad_for_audio(1, 1));
        assert!(!lay.is_pad_for_audio(1, 2));
    }

    #[test]
    fn slot_for_audio_skips_pad_rows() {
        let cfg = KyutaiTtsConfig::v1_6b_en_fr();
        let lay = StreamLayout::from_config(&cfg);
        // Delayed codebook returns None during pad, then 0..N after.
        assert_eq!(lay.slot_for_audio(5, 0), None);
        assert_eq!(lay.slot_for_audio(5, 1), None);
        assert_eq!(lay.slot_for_audio(5, 2), Some(0));
        assert_eq!(lay.slot_for_audio(5, 3), Some(1));
    }

    #[test]
    fn total_steps_includes_pad_prefix() {
        let cfg = KyutaiTtsConfig::v1_6b_en_fr();
        let lay = StreamLayout::from_config(&cfg);
        assert_eq!(lay.total_steps_for(0), 2);
        assert_eq!(lay.total_steps_for(10), 12);
    }

    #[test]
    fn pad_prefix_is_max_delay_rows_of_card() {
        let cfg = KyutaiTtsConfig::v1_6b_en_fr();
        let lay = StreamLayout::from_config(&cfg);
        let pad = lay.pad_prefix();
        assert_eq!(pad.len(), 2);
        for row in &pad {
            assert_eq!(row.len(), 32);
            for v in row {
                assert_eq!(*v, lay.audio_pad);
            }
        }
    }
}
