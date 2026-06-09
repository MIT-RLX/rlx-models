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

//! Stage timing (talker / code_predictor / vocoder).

use crate::options::Qwen3TtsOptions;
use rlx_runtime::Device;

#[derive(Debug, Clone, Copy)]
pub struct Qwen3TtsBenchReport {
    pub device: Device,
    pub eager_talker: bool,
    pub talker_prefill_ms: f64,
    pub talker_decode_ms: f64,
    pub code_predictor_ms: f64,
    pub vocoder_ms: f64,
    pub synthesis_ms: f64,
    pub codec_frames: usize,
    pub pcm_samples: usize,
    pub talker_decode_steps: usize,
}

impl Qwen3TtsBenchReport {
    pub fn audio_duration_ms(&self) -> f64 {
        if self.pcm_samples == 0 {
            0.0
        } else {
            self.pcm_samples as f64 / crate::tokens::SAMPLE_RATE_HZ as f64 * 1000.0
        }
    }

    pub fn rtf(&self) -> f64 {
        let d = self.audio_duration_ms();
        if d <= 0.0 { 0.0 } else { self.synthesis_ms / d }
    }

    pub fn talker_total_ms(&self) -> f64 {
        self.talker_prefill_ms + self.talker_decode_ms
    }

    pub fn print_line(&self) {
        println!(
            "config=talker={} device={:?} rtf={:.3} synthesis_ms={:.2} \
             talker_prefill_ms={:.2} talker_decode_ms={:.2} code_predictor_ms={:.2} vocoder_ms={:.2} \
             frames={} decode_steps={}",
            if self.eager_talker {
                "eager"
            } else {
                "compiled"
            },
            self.device,
            self.rtf(),
            self.synthesis_ms,
            self.talker_prefill_ms,
            self.talker_decode_ms,
            self.code_predictor_ms,
            self.vocoder_ms,
            self.codec_frames,
            self.talker_decode_steps,
        );
    }
}

pub fn options_from_report(r: &Qwen3TtsBenchReport) -> Qwen3TtsOptions {
    Qwen3TtsOptions {
        device: r.device,
        eager_talker: r.eager_talker,
        max_frames: 0,
    }
}
