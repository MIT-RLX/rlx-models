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

//! **Kev** — small Jev-like *System One* decision models, native in RLX.
//!
//! A decision model takes one document (the **state**) plus a set of typed
//! questions and returns a calibrated probability distribution for each, in a
//! single forward pass. It does not generate text. Ported from
//! [jaredpalmer/kev](https://github.com/jaredpalmer/kev), which reimplements
//! the architecture reverse-engineered from TypeSafe's hosted Jev.
//!
//! ```text
//! state  ──► Qwen3.5 backbone (+ merged rank-16 LoRA) ──► hidden
//!                                                          │
//! question k:  <q> instr <opt> a </opt> <opt> b </opt> <decide>
//!                              │                 │          │
//!                              └──── pointer head ──────────┘  ──► p(a), p(b)
//! ```
//!
//! # What the pieces are
//!
//! | module | role |
//! |---|---|
//! | [`api`] | `POST /v1/systemone` shapes, `render`, typed answers |
//! | [`encode`] | packing a record into tokens; the delimiter protocol |
//! | [`mod@pointer`] | the readout head and its calibration temperature |
//! | [`lora`] | reading a peft adapter and folding it into base weights |
//! | [`weights`] | HF Qwen3.5 base + merged adapter → `Qwen35Weights` |
//! | [`checkpoint`] | a run directory: `head.pt`, adapter, tokenizer |
//! | [`model`] | the forward pass and the compiled-shape policy |
//!
//! # Scope of this port
//!
//! Implemented: the **row form** — one causal row per question, which is what
//! kev itself uses on Qwen3.5 because the Gated DeltaNet layers ignore
//! attention masks. Isolation is exact by construction.
//!
//! Not implemented: the **packed form** (all questions in one sequence under
//! an additive block-causal mask). It is only reachable on attention-only
//! Qwen3-generation bases, and it needs `MaskKind::Bias` threaded through the
//! `rlx-qwen35` prefill graph, which currently exposes a mask input on the
//! cache/verify path only. [`encode`] already produces the `seg` / `opt`
//! vectors it would need.
//!
//! Weights are loaded as dense f32. Kev-0.8B is comfortable on a laptop;
//! 4B and 9B want a large-memory host until a packed or bf16 path exists.

pub mod api;
pub mod checkpoint;
pub mod cli;
pub mod encode;
#[cfg(feature = "train")]
pub mod eval;
#[cfg(feature = "export")]
pub mod export;
pub mod json;
pub mod lora;
pub mod model;
pub mod pointer;
#[cfg(feature = "prefix-cache")]
pub mod prefix;
#[cfg(feature = "train")]
pub mod pyrandom;
#[cfg(feature = "train")]
pub mod recipe;
#[cfg(feature = "serve")]
pub mod serve;
#[cfg(feature = "train")]
pub mod train;
pub mod weights;

pub use api::{Answer, Question, Record, RecordQuestion, SystemOneRequest, to_answers, to_record};
pub use checkpoint::{Checkpoint, Meta};
pub use encode::{Encoder, Encoding, Row, TokenizerLike, rows_of};
pub use lora::LoraAdapter;
pub use model::{DecisionModel, RowHidden, RowReadout, ShapePolicy, Usage};
pub use pointer::PointerHead;
pub use weights::{HfQwen35Fixups, KevWeights, LoraMergingLoader, load_merged};

#[cfg(feature = "tokenizer")]
pub use encode::HfTokenizer;

#[cfg(feature = "tokenizer")]
mod load {
    use std::path::Path;

    use anyhow::{Context, Result};
    use rlx_runtime::Device;

    use crate::checkpoint::Checkpoint;
    use crate::encode::{Encoder, HfTokenizer};
    use crate::lora::LoraAdapter;
    use crate::model::{DecisionModel, ShapePolicy};

    /// Load a kev run directory against its base model.
    ///
    /// `run_dir` is a checkpoint (`head.pt`, `adapter_model.safetensors`,
    /// `tokenizer.json`); `base_dir` is the matching `Qwen/Qwen3.5-*-Base`
    /// snapshot. The base is *not* resolved from `head.pt`'s `base` field
    /// automatically — that field names a Hub id, and this crate does not
    /// download.
    pub fn load(
        run_dir: &Path,
        base_dir: &Path,
        device: Device,
        policy: ShapePolicy,
    ) -> Result<DecisionModel<HfTokenizer>> {
        let ckpt = Checkpoint::open(run_dir)?;
        let head = ckpt.pointer_head()?;
        let adapter = LoraAdapter::open(run_dir)?;
        let merged = crate::weights::load_merged(base_dir, adapter)?;

        let tok_path = ckpt.tokenizer_path();
        let tok = HfTokenizer::from_file(&tok_path)
            .with_context(|| format!("loading {}", tok_path.display()))?;
        let encoder = Encoder::new(tok)?;

        // Let the model release the ~3.5 GB bundle after each build; a
        // reshape re-reads it rather than keeping it resident forever.
        let (base_owned, run_owned) = (base_dir.to_path_buf(), run_dir.to_path_buf());
        let reload: crate::model::WeightsReloader = Box::new(move || {
            let adapter = LoraAdapter::open(&run_owned)?;
            Ok(crate::weights::load_merged(&base_owned, adapter)?.weights)
        });

        Ok(
            DecisionModel::new(encoder, head, merged.cfg, merged.weights, device, policy)?
                .with_reloader(reload),
        )
    }
}

#[cfg(feature = "tokenizer")]
pub use load::load;
