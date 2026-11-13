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

//! LiquidAI **LFM3-Audio-2B** from its GGUF release
//! ([`LiquidAI/LFM3-Audio-2B-GGUF`](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF)):
//! speech recognition, speech synthesis and spoken chat, all as rlx graphs.
//!
//! Two files: the LM (`LFM3-Audio-2B-<quant>.gguf`, architecture `lfm3`) and
//! the `mmproj` holding everything audio:
//!
//! * **speech in** — 16 kHz log-mel ([`mel`]) → three stride-2 convolutions
//!   (8× in time) → a 17-layer causal LFM2 with 32-step sliding attention →
//!   LayerNorm/GELU projector, one LM embedding per 80 ms;
//! * **LM** — the 24-layer LFM3 hybrid (width-7 gated short convolutions,
//!   full and 256-step sliding GQA attention with per-head sigmoid gates,
//!   sandwich norms, half-rotary RoPE on the full layers);
//! * **speech out** — each frame's 12 codebooks are sampled from the LM's
//!   hidden state by a 3-layer bidirectional LFM3 depth decoder over three
//!   MaskGIT passes, and the frame is fed back as the sum of its codebook
//!   embeddings;
//! * **codec** — FSQ codes → 8-layer LFM3 at 75 Hz (sliding window 30) →
//!   log-magnitude + phase → inverse STFT, 1920 samples of 24 kHz per frame.
//!
//! All four stacks share one builder ([`trunk`]). Quantized projections stay
//! packed (`DequantMatMul`); the token embedding table stays packed on the
//! host and decodes rows on demand.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rlx_lfm3_audio::{Lfm3Audio, text::Tokenizer, wav};
//! let mut m = Lfm3Audio::open_dir("weights/lm/lfm3-audio-2b-gguf".as_ref(), Some("Q4_K_M"),
//!                                 None, rlx_runtime::Device::Cpu)?;
//! let tok = Tokenizer::new(m.lm_path());
//! let pcm = wav::read_16k("clip.wav".as_ref())?;
//! println!("{}", m.transcribe(&tok, &pcm, Some("English"), 256)?);
//! let _speech = m.speak(&tok, "Hello.", "en", "female", 0, 512)?; // 24 kHz
//! # Ok(()) }
//! ```
//!
//! [`Lfm3Audio::generate`] with the [`prompt`] builders covers the rest
//! (spoken chat, interleaved replies, sampling options).

pub mod cli;
pub mod config;
pub mod diag;
pub mod graph;
pub mod istft;
pub mod mel;
pub mod model;
pub mod prompt;
pub mod sampling;
mod stages;
#[cfg(feature = "tokenizer")]
pub mod text;
pub mod trunk;
pub mod wav;

pub use config::{Lfm3AudioConfig, SpeechSampling, Tokens};
pub use diag::{footprint, set_timing};
pub use istft::{Istft, istft};
pub use model::{GenerateOptions, Generated, Lfm3Audio, Mode, RevealOrder, Segment, resolve};
pub use rlx_core::weight_storage::WeightStorage;
pub use sampling::{Rng, Sampling};
