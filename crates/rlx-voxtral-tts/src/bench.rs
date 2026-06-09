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

//! Stage timing for native TTS (LM prefill / decode, acoustic, codec).

use rlx_runtime::Device;

/// Per-stage milliseconds from one [`super::backbone::NativeTtsEngine::synthesize_profiled`] run.
#[derive(Debug, Clone, Copy)]
pub struct VoxtralTtsBenchReport {
    pub device: Device,
    pub eager_lm: bool,
    pub eager_acoustic: bool,
    /// Token + voice embedding assembly (host).
    pub embed_ms: f64,
    /// First LM forward (prompt prefill).
    pub lm_prefill_ms: f64,
    /// Sum of per-frame LM decode forwards.
    pub lm_decode_ms: f64,
    /// Sum of per-frame acoustic (flow-matching) work.
    pub acoustic_ms: f64,
    /// Codec decode of all frames.
    pub codec_ms: f64,
    /// `lm_prefill + lm_decode + acoustic + codec` (excludes embed).
    pub synthesis_ms: f64,
    pub audio_frames: usize,
    pub prompt_tokens: usize,
    pub pcm_samples: usize,
    /// Flow-matching Euler steps per frame (from model config).
    pub euler_steps_per_frame: usize,
    /// Compiled acoustic velocity `run()` count (= frames × euler × 2 CFG passes).
    pub acoustic_velocity_runs: u64,
}

impl VoxtralTtsBenchReport {
    pub fn sample_rate_hz() -> u32 {
        24_000
    }

    pub fn audio_duration_ms(&self) -> f64 {
        if self.pcm_samples == 0 {
            return 0.0;
        }
        self.pcm_samples as f64 / Self::sample_rate_hz() as f64 * 1000.0
    }

    pub fn rtf(&self) -> f64 {
        let dur = self.audio_duration_ms();
        if dur <= 0.0 {
            return 0.0;
        }
        self.synthesis_ms / dur
    }

    pub fn lm_total_ms(&self) -> f64 {
        self.lm_prefill_ms + self.lm_decode_ms
    }

    pub fn stage_share(&self, ms: f64) -> f64 {
        let total = self.synthesis_ms;
        if total <= 0.0 {
            0.0
        } else {
            100.0 * ms / total
        }
    }

    pub fn label(&self) -> String {
        format!(
            "lm={} acoustic={}",
            if self.eager_lm { "eager" } else { "compiled" },
            if self.eager_acoustic {
                "eager"
            } else {
                "compiled"
            }
        )
    }

    pub fn print_line(&self) {
        let dur = self.audio_duration_ms();
        let rtf = self.rtf();
        let lm = self.lm_total_ms();
        println!(
            "config={} device={:?} frames={} euler={} rtf={:.3} synthesis_ms={:.2} \
             lm_prefill_ms={:.2} lm_decode_ms={:.2} acoustic_ms={:.2} codec_ms={:.2} \
             lm_share={:.0}% acoustic_share={:.0}% codec_share={:.0}% \
             lm_ms_per_frame={:.3} acoustic_ms_per_frame={:.3} velocity_runs={}",
            self.label(),
            self.device,
            self.audio_frames,
            self.euler_steps_per_frame,
            rtf,
            self.synthesis_ms,
            self.lm_prefill_ms,
            self.lm_decode_ms,
            self.acoustic_ms,
            self.codec_ms,
            self.stage_share(lm),
            self.stage_share(self.acoustic_ms),
            self.stage_share(self.codec_ms),
            if self.audio_frames > 0 {
                lm / self.audio_frames as f64
            } else {
                0.0
            },
            if self.audio_frames > 0 {
                self.acoustic_ms / self.audio_frames as f64
            } else {
                0.0
            },
            self.acoustic_velocity_runs,
        );
        println!(
            "  audio_ms={:.2} prompt_tokens={} pcm_samples={} embed_ms={:.2}",
            dur, self.prompt_tokens, self.pcm_samples, self.embed_ms,
        );
    }
}
