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

//! spaCy 3.8 English pipeline inference in pure Rust, as far as sentence
//! boundaries need it (`doc.sents` of `en_core_web_lg`): the rule-based
//! tokenizer, the `tok2vec` embedding (hash embeddings + static vectors,
//! maxout window CNN) and the transition-based dependency parser.

pub mod parser;
pub mod tok2vec;
pub mod tokenizer;

pub use parser::{Parse, Parser};
pub use tok2vec::Tok2Vec;
pub use tokenizer::{Token, Tokenizer};
