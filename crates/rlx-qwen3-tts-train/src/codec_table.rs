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

//! Flat codec embedding table (mmap row loads, no full-vocab clone).

use anyhow::{Context, Result, ensure};
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;

pub struct CodecEmbeddingTable {
    pub hidden: usize,
    data: Vec<f32>,
}

impl CodecEmbeddingTable {
    pub fn open(store: &Qwen3TtsWeightStore) -> Result<Self> {
        let name = "talker.model.codec_embedding.weight";
        let snap = store.tensor_snapshot(&[name])?;
        let (flat, shape) = snap.get(name).context(name)?;
        ensure!(shape.len() == 2);
        Ok(Self {
            hidden: shape[1],
            data: flat.clone(),
        })
    }

    #[inline]
    pub fn row(&self, id: u32) -> &[f32] {
        let off = id as usize * self.hidden;
        &self.data[off..off + self.hidden]
    }

    pub fn copy_row(&self, id: u32, out: &mut [f32]) {
        let row = self.row(id);
        let n = out.len().min(row.len());
        out[..n].copy_from_slice(&row[..n]);
    }

    pub fn add_row_into(&self, id: u32, acc: &mut [f32]) {
        let row = self.row(id);
        for (a, &v) in acc.iter_mut().zip(row.iter()) {
            *a += v;
        }
    }
}
