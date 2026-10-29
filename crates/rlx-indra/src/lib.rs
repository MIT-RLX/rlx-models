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

//! Indra OCR ([datalab-to/indra](https://github.com/datalab-to/indra)) on RLX.
//!
//! Every network (line detector, four line recognizers, script router,
//! orientation, reading-order / layout transformer) is a native RLX graph
//! ([`graph`]) over safetensors weights; the page pipeline around them —
//! deskew, line crops, CTC + n-gram beam search, reading order, markdown — is
//! Rust ([`pipeline`]). ONNX Runtime appears only in the parity tests.
//!
//! ```ignore
//! use rlx_indra::{Indra, Options, load_gray};
//! let ocr = Indra::load(".cache/indra", Options::default())?;
//! let page = ocr.read(&load_gray("scan.png".as_ref())?, true)?;
//! println!("{}", page.markdown.unwrap_or_else(|| page.text()));
//! # anyhow::Ok(())
//! ```

#[cfg(feature = "convert")]
pub mod convert;
pub mod graph;
pub mod imgproc;
pub mod layout;
pub mod lines;
pub mod lm;
pub mod order;
pub mod pipeline;
pub mod text;

pub use imgproc::Gray;
pub use pipeline::{Indra, Line, Options, Page, load_gray};
