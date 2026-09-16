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

//! the OS's on-device machine translation, natively on RLX.
//!
//! macOS 15+ ships a `Translation.framework` whose low-latency path is a
//! classic NMT stack: a sed-script normalizer, SentencePiece, a multilingual
//! Espresso encoder-decoder driven by beam search, and a stack of pre/post
//! processing blocks. All of it is described by a JSON **Quasar** config that
//! the OS installs separately from the weights.
//!
//! ```text
//!   text ──normalizer (sed script)──▶ tokenizer ──▶ phrasebook lookup
//!        ──SentencePiece encode────▶ [control tokens] + ids
//!        ──Espresso NMT, beam 3────▶ target ids
//!        ──SentencePiece decode────▶ structured prediction ──▶ case map ──▶ text
//! ```
//!
//! # What it does
//!
//! Translates, in every direction the machine has installed, and reproduces the
//! shipped framework closely enough to be checked against it sentence by
//! sentence: **653 of 855 outputs byte-identical over 57 directions**, and
//! within **0.002 chrF** of the OS when both are scored against human
//! references (4175 sentences of FLORES-200 and Tatoeba). It is ahead of the
//! OS in roughly 40% of directions. A wider sweep covers **378** directions.
//!
//! - [`quasar`] parses the shipped configs — all 416 of them — and [`pdec`]
//!   gives typed decode parameters.
//! - [`pipeline`] resolves a pair's stage DAG into an ordered plan, and
//!   [`execute`] runs it: phrasebook, SentencePiece, translator, quality
//!   estimator, case map, do-not-translate.
//! - [`espresso`] parses `pyespresso.mdl.bin`, which is a *manifest* rather
//!   than a weight container: it names a shared encoder/embedding/readout plus
//!   per-language decoder, handover and input graphs, in the classic
//!   `.espresso.{net,shape,weights}` triple. [`net`] reads those and [`exec`]
//!   evaluates them.
//! - [`decode`] drives the NMT — encode, beam search, readout — and [`beam`]
//!   is the search itself, against any [`beam::NmtDecoder`].
//! - [`convert`] exports every installed bundle to safetensors, GGUF and
//!   `.rlxp`.
//!
//! Four blocks are parsed and planned but not executed, and pass their input
//! through: `PDecForceAlign`, `StructuredPrediction`, `AlignmentProcessor` and
//! `LinkAlternatives`. All four are annotators that add metadata rather than
//! change text. `rlx-translate plan <pair>` marks them `todo`.
//!
//! # Getting the model
//!
//! This crate ships no model data; everything is read from what the OS
//! installed. The weights are a second download, separate from the config
//! asset. Install a pair from System Settings → General → Language & Region →
//! *Translation Languages* (or drive the same daemon API — see the README),
//! then [`assets::Assets::availability`] reports the files as present.
//!
//! # Example
//!
//! ```no_run
//! use rlx_translate::{assets::Assets, decode::Nmt, pdec::PDecParams,
//!                     quasar::LangPair, spm::Vocab};
//!
//! let assets = Assets::discover();
//! let pair = LangPair::parse("en_US-fr_FR")?;
//! let (_, config) = assets.best_config(&pair)?;
//! let params = PDecParams::from_block(config.mt_app()?.translator(&pair)?)?;
//! let home = assets.model_home(&pair).expect("weights installed");
//! let vocab = Vocab::load(home.join("spm.model"))?;
//!
//! // `input_<lang>` follows the *source* language, the other two the target.
//! let nmt = Nmt::load_for_pair(&home, &[home.as_path()], "en", "fr",
//!                              &params.shortlist.lang_pair)?;
//! let best = nmt.translate_nbest(&vocab, &params, "the sea is warm", 1)?;
//! println!("{}", best[0].text);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! For the whole pipeline rather than the model alone — phrasebook, casing,
//! do-not-translate — build a [`pipeline::TranslationPlan`] and run it with
//! [`execute::run`]; `src/bin/rlx_translate.rs` does exactly that.
//!
//! # Tuning
//!
//! Everything that changes what the search does lives in [`tuning::Tuning`],
//! settable from the environment as `RLX_TRANSLATE_*` or as `key=value`
//! arguments to any subcommand. `rlx-translate tune` prints what is in effect.

pub mod assets;
pub mod beam;
pub mod casemap;
pub mod convert;
pub mod decode;
pub mod dnt;
pub mod espresso;
pub mod exec;
pub mod execute;
pub mod export;
pub mod net;
pub mod normalizer;
pub mod pdec;
pub mod phrasebook;
pub mod pipeline;
pub mod postproc;
pub mod profile;
pub mod quality;
pub mod quasar;
pub mod score;
pub mod shortlist;
pub mod spm;
pub mod tensor;
pub mod tuning;

pub use assets::{Assets, Availability};
pub use beam::{Hypothesis, NmtDecoder, SearchOptions};
pub use casemap::sentence_case;
pub use espresso::Manifest;
pub use net::{Graph, Shape, Weights};
pub use normalizer::Normalizer;
pub use pdec::PDecParams;
pub use phrasebook::Phrasebook;
pub use pipeline::{Stage, StageKind, Support, TranslationPlan};
pub use postproc::normalize_punctuation;
pub use quasar::{BlockKind, LangPair, QuasarConfig, TASK_MT_APP};
pub use shortlist::Shortlist;
pub use spm::Vocab;
pub use tensor::Tensor;
