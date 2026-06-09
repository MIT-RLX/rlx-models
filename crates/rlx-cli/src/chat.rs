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

//! Chat-template engine for RLX runners.
//!
//! The implementation moved into the published `rlx-text` crate so
//! downstream tools (servers, web playground, training scripts) can
//! depend on the chat plumbing without pulling in this CLI helper crate.
//! This module is now a re-export shim — `use rlx_cli::ChatTemplate`
//! and `use rlx_text::ChatTemplate` resolve to the same type.

pub use rlx_text::chat::{ChatMessage, ChatTemplate, ChatTemplateSource, auto_chat_template};
