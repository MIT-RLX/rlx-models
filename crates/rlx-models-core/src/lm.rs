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

//! Shared causal-LM flow helpers — re-export tier-0 surface for model authors.
//!
//! Model authors should use [`rlx_flow::prelude`] and arch recipes (`llama32::Llama32Flow`).
//! Import HIR/Graph only via [`into_compile_parts`] at the compile boundary.

pub use rlx_flow::prelude::*;

pub trait FlowBuildExt {
    fn with_lm_head(self, vocab: usize, hidden: usize, tied: bool) -> Self;
    fn logits(self) -> Self;
}

impl FlowBuildExt for ModelFlow {
    fn with_lm_head(self, vocab: usize, hidden: usize, tied: bool) -> Self {
        self.lm_head(vocab, hidden, tied)
    }

    fn logits(self) -> Self {
        self.output("logits")
    }
}

/// Split built flow for compile — no Graph/HIR imports needed at call site.
pub fn into_compile_parts(
    built: BuiltModel,
) -> anyhow::Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    built.into_parts()
}
