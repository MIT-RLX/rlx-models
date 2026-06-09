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

//! Parse HuggingFace safetensors blobs from memory (`include_bytes!`, etc.).
//!
//! Used by the `rlx-vad` crate to embed `silero_vad_16k.safetensors` without
//! mmap or disk I/O. For sharded on-disk checkpoints see
//! [`SafetensorsCheckpoint`](crate::safetensors_checkpoint::SafetensorsCheckpoint).

use anyhow::{Context, Result};
use safetensors::SafeTensors;

use crate::safetensors_checkpoint::tensor_view_to_f32;

/// Zero-copy view over an in-memory safetensors file.
pub struct EmbeddedSafetensors<'a> {
    inner: SafeTensors<'a>,
}

impl<'a> EmbeddedSafetensors<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let inner = SafeTensors::deserialize(bytes).context("parse embedded safetensors")?;
        Ok(Self { inner })
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let view = self
            .inner
            .tensor(name)
            .with_context(|| format!("tensor {name}"))?;
        tensor_view_to_f32(name, view)
    }
}
