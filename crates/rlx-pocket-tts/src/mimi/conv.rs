// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Conv1d/ConvTranspose1d helpers specific to the Mimi codec. The core
//! computations live in [`crate::ops`]; this module is reserved for streaming
//! state types (currently unused — one-shot decode does not need them).

#[derive(Debug, Clone, Default)]
pub struct StreamingConvState {
    /// Previous receptive-field buffer (Conv1d) or partial overlap-add tail
    /// (ConvTranspose1d). Empty for a one-shot decode.
    pub previous: ndarray::Array2<f32>,
}
