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

//! Token embeddings gathered on the host from the mmapped GGUF, for the
//! native packed CPU path (`RLX_QWEN3_HOST_EMBED`).
//!
//! The in-graph embedding is a gather over an F32 copy of `token_embd`
//! (593 MB for Qwen3-0.6B), and every compiled graph — the prefill seed and
//! each decode bucket — holds its own copy to look up one row per decoded
//! token. Here the row is dequantized straight from the packed bytes instead,
//! and the graph takes it as `inputs_embeds`. Same dequantization as the
//! F32 table, so the values are identical.

use anyhow::{Context, Result, ensure};
use rlx_gguf::{GgufFile, GgufTensor};

const TOKEN_EMBD: &str = "token_embd.weight";

pub(crate) struct HostEmbed {
    file: GgufFile,
    tensor: GgufTensor,
    hidden: usize,
    vocab: usize,
    row_bytes: usize,
}

impl HostEmbed {
    /// `None` when the file has no `[vocab, hidden]` `token_embd` this can
    /// slice by row; the caller keeps the in-graph gather then.
    pub(crate) fn open(gguf: &str, hidden: usize) -> Result<Option<Self>> {
        let file = GgufFile::from_path_mmap(gguf).with_context(|| format!("mmap {gguf}"))?;
        let Some(tensor) = file.get(TOKEN_EMBD).cloned() else {
            return Ok(None);
        };
        // GGML order: innermost first.
        let (&[h, vocab], Ok(bytes)) = (tensor.shape.as_slice(), file.tensor_bytes(&tensor)) else {
            return Ok(None);
        };
        if h != hidden || vocab == 0 || !bytes.len().is_multiple_of(vocab) {
            return Ok(None);
        }
        let row_bytes = bytes.len() / vocab;
        Ok(Some(Self {
            file,
            tensor,
            hidden,
            vocab,
            row_bytes,
        }))
    }

    /// Rows for `ids` (token ids as f32, the generator's input encoding),
    /// concatenated: `[ids.len(), hidden]`.
    pub(crate) fn rows(&self, ids: &[f32]) -> Result<Vec<f32>> {
        let bytes = self.file.tensor_bytes(&self.tensor)?;
        let mut out = Vec::with_capacity(ids.len() * self.hidden);
        for &id in ids {
            let id = id as usize;
            ensure!(
                id < self.vocab,
                "token id {id} outside vocab {}",
                self.vocab
            );
            let row = &bytes[id * self.row_bytes..(id + 1) * self.row_bytes];
            out.extend(rlx_gguf::dequant_typed(
                self.tensor.dtype,
                row,
                self.hidden,
                TOKEN_EMBD,
            )?);
        }
        Ok(out)
    }
}

impl std::fmt::Debug for HostEmbed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostEmbed")
            .field("dtype", &self.tensor.dtype)
            .field("vocab", &self.vocab)
            .field("hidden", &self.hidden)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_gguf::{GgmlType, GgufWriter};

    /// Rows gathered from the packed bytes must equal the same rows of the
    /// whole-table dequant the in-graph gather reads — exactly, since both
    /// run the same block decoder.
    #[test]
    fn host_rows_match_full_table_dequant() {
        let (hidden, vocab) = (512, 7);
        let table: Vec<f32> = (0..hidden * vocab)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect();
        let mut w = GgufWriter::new();
        w.add_tensor_bytes(
            TOKEN_EMBD,
            vec![hidden, vocab],
            GgmlType::Q4K,
            rlx_gguf::quantize(&table, GgmlType::Q4K).unwrap(),
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("rlx-host-embed-{}.gguf", std::process::id()));
        w.write_to_path(&path).unwrap();
        let path = path.to_str().unwrap();

        let he = HostEmbed::open(path, hidden)
            .unwrap()
            .expect("sliceable token_embd");
        let (full, _) = GgufFile::from_path_mmap(path)
            .unwrap()
            .dequant_f32(TOKEN_EMBD)
            .unwrap();
        let got = he.rows(&[3.0, 0.0, 6.0]).unwrap();
        for (i, id) in [3usize, 0, 6].into_iter().enumerate() {
            assert_eq!(
                got[i * hidden..(i + 1) * hidden],
                full[id * hidden..(id + 1) * hidden],
                "row {id}"
            );
        }
        assert!(he.rows(&[vocab as f32]).is_err(), "id past the vocab");
        assert!(
            HostEmbed::open(path, 256).unwrap().is_none(),
            "wrong hidden"
        );
        let _ = std::fs::remove_file(path);
    }
}
