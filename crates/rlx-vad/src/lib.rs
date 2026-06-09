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

//! Voice activity detection on RLX (Earshot + Silero architectures).
//!
//! Enable backends via features: `earshot`, `silero` (both on by default).

pub mod audio;
pub mod device;
pub mod ops;
pub mod segments;

#[cfg(feature = "earshot")]
pub mod earshot;

#[cfg(feature = "silero")]
pub mod silero;

pub use audio::{
    SAMPLE_RATE_16K, SpeechSegment, load_wav_mono_f32, parse_wav_mono_f32, resample_linear,
};
pub mod cli;

pub use device::{
    available_device_labels, available_devices, bench_device_label, ensure_backend_ready,
    parse_device_list, resolve_device, streaming_execution_device,
};
pub use segments::SegmentParams;
#[cfg(feature = "earshot")]
pub use segments::speech_segments_earshot;
#[cfg(feature = "silero")]
pub use segments::speech_segments_silero;

#[cfg(feature = "earshot")]
pub use earshot::Detector as EarshotDetector;

#[cfg(feature = "silero")]
pub use silero::{SileroConfig, SileroSession, SileroWeights};

/// Backends compiled into this crate (from Cargo features).
pub fn enabled_backends() -> &'static [&'static str] {
    &[
        #[cfg(feature = "earshot")]
        "earshot",
        #[cfg(feature = "silero")]
        "silero",
    ]
}

/// Default `--backend` for the CLI (first enabled backend).
pub fn default_backend() -> &'static str {
    #[cfg(feature = "earshot")]
    {
        "earshot"
    }
    #[cfg(all(not(feature = "earshot"), feature = "silero"))]
    {
        "silero"
    }
    #[cfg(not(any(feature = "earshot", feature = "silero")))]
    {
        "none"
    }
}

/// Supported PCM sample rates for Silero.
#[cfg(feature = "silero")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleRate {
    Hz8000,
    Hz16000,
}

#[cfg(feature = "silero")]
impl SampleRate {
    pub const fn as_usize(self) -> usize {
        match self {
            Self::Hz8000 => 8_000,
            Self::Hz16000 => 16_000,
        }
    }

    pub fn from_hz(hz: usize) -> anyhow::Result<Self> {
        match hz {
            8_000 => Ok(Self::Hz8000),
            16_000 => Ok(Self::Hz16000),
            _ => anyhow::bail!("unsupported sample rate {hz} (Silero supports 8000 and 16000)"),
        }
    }
}
