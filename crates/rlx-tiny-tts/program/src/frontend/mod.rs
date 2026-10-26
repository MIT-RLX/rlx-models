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

//! The English frontend: rlx-inflect-nano's modules, its neural G2P and its
//! WordPiece segmentation this program's own (the G2P's GRU an rlx graph, the
//! tokenizer the host's).

#[path = "../../../../rlx-inflect-nano/src/frontend/clean.rs"]
pub mod clean;
#[path = "../../../../rlx-inflect-nano/src/frontend/cmudict.rs"]
pub mod cmudict;
#[path = "../../../../rlx-inflect-nano/src/frontend/english.rs"]
pub mod english;
pub mod g2p;
#[path = "../../../../rlx-inflect-nano/src/frontend/normalize.rs"]
pub mod normalize;
#[path = "../../../../rlx-inflect-nano/src/frontend/numbers.rs"]
pub mod numbers;
#[path = "../../../../rlx-inflect-nano/src/frontend/pos.rs"]
pub mod pos;
#[path = "../../../../rlx-inflect-nano/src/frontend/symbols.rs"]
pub mod symbols;
pub mod tokenize_bert;

pub use english::English;
