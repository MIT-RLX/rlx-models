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

//! NVIDIA **Conformer-CTC** ASR on RLX.
//!
//! Loads classic NeMo EncDecCTC Conformer checkpoints (e.g.
//! [`nvidia/stt_en_conformer_ctc_small`](https://huggingface.co/nvidia/stt_en_conformer_ctc_small))
//! **natively from the distributed `.nemo`** via [`rlx_nemo`] — no ONNX and no
//! Python at runtime.
//!
//! # Pipeline
//!
//! 1. Log-mel frontend ([`mel`]) — NeMo `AudioToMelSpectrogramPreprocessor`
//! 2. Conformer encoder graph ([`encoder`]) — `striding`×4 subsample + N blocks
//! 3. CTC linear head + greedy decode ([`ctc`])
//! 4. SentencePiece detokenize ([`tokenizer`])
//!
//! The encoder is compiled once per mel-length bucket and reused via
//! [`rlx_runtime::compile_cache::CompileCache`] (see [`ConformerCtc::warm`]).
//!
//! # Quick start
//!
//! ```rust,no_run
//! use rlx_conformer_ctc::{ConformerCtc, wav};
//! use rlx_runtime::Device;
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut asr = ConformerCtc::open(
//!     "stt_en_conformer_ctc_small.nemo".as_ref(),
//!     Device::Cpu,
//! )?;
//! let bytes = std::fs::read("clip.wav")?;
//! let w = wav::parse(&bytes)?;
//! let pcm = wav::resample(&w.samples, w.sample_rate, asr.config().sample_rate as u32);
//! let text = asr.transcribe(&pcm)?;
//! println!("{text}");
//! # Ok(())
//! # }
//! ```
//!
//! FastConformer (`dw_striding`) + RNN-T checkpoints belong in the
//! `rlx-nemotron-asr` crate.

pub mod cli;
pub mod config;
pub mod ctc;
pub mod encoder;
pub mod mel;
pub mod runner;
pub mod tokenizer;
pub mod wav;
pub mod weights;

pub use config::AsrConfig;
pub use mel::{MelSpectrogram, bucket_mel_frames, pad_mel_to_frames};
pub use runner::ConformerCtc;
pub use wav::Wav;

/// Model family identifier for CLI / facade registry dispatch.
pub const FAMILY: &str = "conformer-ctc";
