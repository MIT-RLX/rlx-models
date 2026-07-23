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

//! Native RLX streaming Conformer ASR.
//!
//! ```text
//! audio → 80-mel frontend → energy VAD → encoder → CTC beam → AED → text
//! ```
//!
//! **Weights:** single file `weights/asr/model.gguf` (`RLX_ASR_DIR`,
//! `just asr-pack-gguf`). Units, silence fbank, etiquette, TP FSTs, AED, and
//! folded encoder tensors live inside the GGUF.
//!
//! The Rust CLI uses a stub encoder until the folded Conformer graph is wired;
//! Python `tools/e2e_native_whole.py` runs the folded CTC path from the same GGUF.

pub mod beam;
pub mod effective_decoder;
pub mod encoder;
pub mod env;
pub mod frontend;
pub mod gguf_io;
pub mod k_codebook;
pub mod ls_projections;
pub(crate) mod npy_io;
pub mod pipeline;
pub mod search;
pub mod spec;
pub mod textproc;
pub mod units;
pub mod vad;
pub(crate) mod weights;

pub use beam::StreamingCtcBeam;
pub use effective_decoder::EffectiveStep1;
pub use env::{asr_dir, asr_dir_env, timing, AsrPaths};
pub use gguf_io::{pack_asr_gguf, resolve_gguf_path, AsrGguf, DEFAULT_GGUF_NAME};
pub use k_codebook::{affine_group, KCodebook, TextKLayer, TEXT_K_LAYERS};
pub use ls_projections::{LsProjections, LsVLayer, ATT_V_HEAD_DIM, ATT_V_OUT};
pub use pipeline::{AsrSession, StreamingAsr, Transcript};
pub use units::Units;
pub use spec::{BEAM, BLANK, EOS, MEL_BINS, SOS, VOCAB};
