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

//! **FunASR** (ModelScope) ported to Rust on RLX — runs on every RLX backend
//! (cpu / metal / mlx / cuda / rocm / wgpu).
//!
//! The full production pipeline is covered:
//!
//! * [`paraformer`] — non-autoregressive ASR: SAN-M encoder → CIF predictor
//!   ([`cif`]) → SAN-M decoder. The flagship model.
//! * [`sensevoice`] — SenseVoiceSmall, an encoder-only multilingual CTC model
//!   with language / event / emotion tags.
//! * [`vad`] — FSMN voice-activity detection (DFSMN graph + host state machine).
//! * [`punc`] — CT-Transformer punctuation restoration.
//! * [`speaker`] — CAM++ speaker embedding (192-d).
//! * [`pipeline`] — the chained VAD → ASR → punctuation → speaker workflow.
//!
//! Shared infrastructure: the Kaldi-fbank + LFR + CMVN [`frontend`], the
//! [`sanm`] SAN-M / FSMN HIR building blocks, [`weights`] loading from native
//! PyTorch `model.pt` ([`pt`]) or `safetensors`, [`config`] (`config.yaml`),
//! and [`tokenizer`] (char / SentencePiece).
//!
//! Heavy compute (encoders, decoders, classifiers) is compiled to an RLX graph
//! and runs on the selected device; the inherently sequential pieces (CIF
//! integrate-and-fire, CTC collapse, the VAD state machine, beam-free argmax)
//! run on the host, exactly as the other RLX ASR crates do.

#![warn(missing_docs)]

pub mod audio;
pub mod cache;
pub mod cif;
pub mod cli;
pub mod config;
pub mod frontend;
pub mod paraformer;
pub mod pipeline;
pub mod pt;
pub mod punc;
pub mod runner;
pub mod sanm;
pub mod sensevoice;
pub mod speaker;
pub mod streaming;
pub mod tokenizer;
pub mod vad;
pub mod wav;
pub mod weights;

pub use config::{
    CamPlusConfig, CtTransformerConfig, FsmnVadConfig, ModelKind, ParaformerConfig,
    SanmEncoderConfig, SenseVoiceConfig,
};
pub use frontend::{Fbank, WavFrontend};
pub use paraformer::Paraformer;
pub use pipeline::{FunPipeline, PipelineResult, Segment};
pub use sensevoice::SenseVoice;
pub use streaming::StreamingRecognizer;
pub use vad::FsmnVad;

/// Model family identifier for CLI / registry dispatch.
pub const FAMILY: &str = "funasr";
