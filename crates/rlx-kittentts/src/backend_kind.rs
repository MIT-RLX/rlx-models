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

//! Inference backend selection (native RLX).

#[cfg(feature = "native")]
use crate::native::NativeEngine;

/// How [`crate::KittenTTS`] runs the acoustic model.
pub enum BackendKind {
    #[cfg(feature = "native")]
    Native(NativeEngine),
}

impl BackendKind {
    pub fn backend_label(&self) -> &str {
        // Pure-frontend build (no `native` backend): `BackendKind` is an
        // uninhabited enum — a consumer that reuses only the espeak phonemizer
        // (e.g. rlx-kokoro's native path) can then depend on this crate without
        // pulling an inference backend.
        #[cfg(not(feature = "native"))]
        match *self {}
        #[cfg(feature = "native")]
        match self {
            Self::Native(_) => "rlx/native",
        }
    }
}
