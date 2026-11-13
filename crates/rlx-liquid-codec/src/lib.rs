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

//! LiquidAI **LiquidCodec**
//! ([`LiquidAI/Liquid-Codec`](https://huggingface.co/LiquidAI/Liquid-Codec)):
//! 24 kHz mono speech ↔ 12 tokens per 80 ms frame (12.5 Hz, ≈1.65 kbps),
//! as rlx graphs on every backend.
//!
//! * **encode** — STFT (`n_fft` 1280, hop 320) → `log1p` magnitude and unit
//!   phase per bin ([`stft`]) → 1×1 and causal width-3 convolutions → a
//!   stride-6 convolution to 12.5 Hz → an 8-layer causal LFM3 (short
//!   convolutions, full attention on layers 2/4/6) → 48 latents → grouped
//!   FSQ, levels `[8, 7, 6, 6]` per token ([`fsq`]);
//! * **decode** — FSQ latents → each frame repeated 6× → an 8-layer causal
//!   LFM3 with a 30-step sliding window at 75 Hz → log-magnitude + phase per
//!   STFT frame → inverse STFT ([`rlx_lfm3_audio::Istft`]).
//!
//! The LFM3 stacks are [`rlx_lfm3_audio::trunk`], the builder that already
//! runs this decoder inside LFM3-Audio's GGUF; [`weights`] maps the
//! safetensors names onto it. Both stacks look only backwards, so
//! [`LiquidCodec::encode_stream`] and [`LiquidCodec::decode_stream`] run
//! frame by frame (KV-cached step graphs) to the same codes and audio as one
//! call, and long clips encode in chunks with memory linear in their length.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rlx_liquid_codec::LiquidCodec;
//! let codec = LiquidCodec::open("weights/audio/liquid-codec".as_ref(), rlx_runtime::Device::Cpu)?;
//! let pcm = rlx_liquid_codec::read_24k("clip.wav".as_ref())?;
//! let codes = codec.encode(&pcm)?; // [frames][12]
//! let audio = codec.decode(&codes)?; // 1920 samples per frame
//!
//! let mut enc = codec.encode_stream();
//! let mut live = Vec::new();
//! for piece in pcm.chunks(1920) {
//!     live.extend(enc.push(piece)?); // frames out 20 ms after they end
//! }
//! live.extend(enc.finish()?);
//! assert_eq!(live, codes);
//!
//! let mut stream = codec.decode_stream();
//! let mut out = Vec::new();
//! for frame in &codes {
//!     out.extend(stream.push(std::slice::from_ref(frame))?);
//! }
//! out.extend(stream.finish());
//! assert_eq!(out.len(), audio.len());
//! # Ok(()) }
//! ```

pub mod cli;
pub mod codec;
pub mod config;
pub mod fsq;
pub mod graphs;
pub mod stft;
pub mod weights;

use std::path::Path;

use anyhow::{Context, Result};

pub use codec::{DecodeStream, EncodeStream, LiquidCodec};
pub use config::LiquidCodecConfig;
pub use rlx_core::audio_codec::{AudioCodec, CodecInfo, RvqCodes};
pub use rlx_lfm3_audio::wav::write_mono_16;

/// Mono samples of a WAV file at the codec's 24 kHz.
pub fn read_24k(path: &Path) -> Result<Vec<f32>> {
    let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(path)
        .with_context(|| format!("read {}", path.display()))?;
    Ok(if rate == 24_000 {
        pcm
    } else {
        rlx_core::audio::resample_sinc(&pcm, rate as usize, 24_000)
    })
}
