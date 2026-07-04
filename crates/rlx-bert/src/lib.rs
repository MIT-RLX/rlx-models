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

//! BERT encoder graph builder for RLX.
//!
//! Lowers a [`rlx_core::config::BertConfig`] + weights into an RLX IR graph via
//! the shared `rlx_flow::ModelFlow` DSL. This crate only *builds* the graph;
//! compiling and running it (tokenization, device dispatch, pooling) lives in
//! consumer crates such as `rlx-embed` and `rlx-clinicalbert`. See the README.

pub mod bert;
pub mod flow;

#[cfg(test)]
mod test_support;

pub use bert::build_bert_graph_sized;
pub use flow::{BertFlow, build_bert_built};
