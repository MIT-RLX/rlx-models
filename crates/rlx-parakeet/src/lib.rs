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

//! # rlx-parakeet
//!
//! NVIDIA **Parakeet-TDT** on RLX — a FastConformer acoustic encoder feeding a
//! **Token-and-Duration Transducer** (TDT). Parakeet-TDT shares almost its entire
//! stack with Nemotron ASR (same FastConformer encoder, same LSTM prediction
//! network) and differs only in the joint network, which grows a second *duration*
//! head, and in the decode loop, which skips a learned number of encoder frames
//! per emitted token.
//!
//! This crate is therefore mostly composition:
//!
//! - **Acoustic encoder + prediction net** — reused from
//!   [`rlx_nemotron_asr`] (`encoder`, `decoder::{PredictionNet, LstmCell}`).
//! - **Decode loop** — reused from
//!   [`rlx_audio_blocks::decoders::tdt`] (`run_tdt_greedy_duration_loop`).
//! - **New here** — [`joint::TdtJoint`] (token + duration heads) and
//!   [`transducer::TdtCore`] (the [`TdtDecoderCore`](rlx_audio_blocks::decoders::TdtDecoderCore)
//!   implementation that binds the prediction net to the joint) plus
//!   [`transducer::tdt_greedy_decode`], the host-side transducer search.
//!
//! - **End-to-end runner** — [`runner::Parakeet`] loads a `.nemo` checkpoint
//!   (via `rlx-nemo`), builds the FastConformer encoder + mel frontend + LSTM
//!   prediction net (reused from `rlx-nemotron-asr`) + the [`TdtJoint`], and
//!   exposes [`Parakeet::transcribe`] (`pcm → text`) driving the TDT decode.
//!
//! Status: arch + CPU smoke + **e2e wired**. The `.nemo` → log-mel → encoder →
//! TDT greedy decode → SentencePiece text path is implemented in
//! [`runner::Parakeet`]; real-weight transcription parity is gated on a local
//! Parakeet-TDT `.nemo` checkpoint (not yet present).

pub mod joint;
pub mod runner;
pub mod transducer;

pub use joint::TdtJoint;
pub use runner::Parakeet;
pub use transducer::{TdtCore, tdt_greedy_decode};

// Re-export the shared decode result types so callers need only this crate.
pub use rlx_audio_blocks::decoders::tdt::{TdtDecodeResult, TdtJointStep};
