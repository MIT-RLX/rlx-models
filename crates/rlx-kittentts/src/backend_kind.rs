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

//! Inference backend selection (ONNX Runtime vs native RLX).

#[cfg(feature = "onnx")]
use std::sync::Mutex;

#[cfg(feature = "onnx")]
use ort::session::Session;

#[cfg(feature = "native")]
use crate::native::NativeEngine;

/// How [`crate::KittenTTS`] runs the acoustic model.
pub enum BackendKind {
    #[cfg(feature = "onnx")]
    Onnx {
        session: Mutex<Session>,
        ort_ep: String,
    },
    #[cfg(feature = "native")]
    Native(NativeEngine),
}

impl BackendKind {
    pub fn backend_label(&self) -> &str {
        // Pure-frontend build (neither `onnx` nor `native`): `BackendKind` is an
        // uninhabited enum — a consumer that reuses only the espeak phonemizer
        // (e.g. rlx-kokoro's native path) can then depend on this crate without
        // pulling an inference backend.
        #[cfg(not(any(feature = "onnx", feature = "native")))]
        match *self {}
        #[cfg(any(feature = "onnx", feature = "native"))]
        match self {
            #[cfg(feature = "onnx")]
            Self::Onnx { ort_ep, .. } => ort_ep,
            #[cfg(feature = "native")]
            Self::Native(_) => "rlx/native",
        }
    }
}
