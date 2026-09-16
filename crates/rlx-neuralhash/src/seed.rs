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

//! The `neuralhash_128x96_seed*.dat` output projection.
//!
//! The file is a 128-byte header followed by a row-major `[96, 128]` f32
//! little-endian matrix — exactly what the reference slices off:
//!
//! ```python
//! seed1 = open(sys.argv[2], 'rb').read()[128:]
//! seed1 = np.frombuffer(seed1, dtype=np.float32).reshape([96, 128])
//! ```
//!
//! [`SeedMatrix::project`] is the `seed1.dot(embedding)` step: 96 dot products
//! against the 128-float model output, whose signs become the hash bits.

use anyhow::{Context, Result, bail, ensure};
use std::path::Path;

use crate::hash::HASH_BITS;

/// Width of the model embedding the seed matrix consumes.
pub const EMBED_DIM: usize = 128;
/// Bytes of leading header skipped before the matrix payload.
pub const SEED_HEADER_BYTES: usize = 128;
/// Bytes of f32 matrix payload (`96 * 128 * 4`).
pub const SEED_MATRIX_BYTES: usize = HASH_BITS * EMBED_DIM * 4;
/// Total expected size of a `neuralhash_128x96_seed*.dat` file.
pub const SEED_FILE_BYTES: usize = SEED_HEADER_BYTES + SEED_MATRIX_BYTES;

/// The `[96, 128]` output projection, row-major.
#[derive(Clone)]
pub struct SeedMatrix {
    rows: Vec<f32>,
}

impl SeedMatrix {
    /// Parse a whole `neuralhash_128x96_seed*.dat` file image.
    pub fn from_file_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < SEED_HEADER_BYTES {
            bail!(
                "neuralhash seed: file is {} bytes, shorter than the {SEED_HEADER_BYTES}-byte header",
                bytes.len()
            );
        }
        let payload = &bytes[SEED_HEADER_BYTES..];
        ensure!(
            payload.len() == SEED_MATRIX_BYTES,
            "neuralhash seed: expected {SEED_FILE_BYTES} bytes \
             ({SEED_HEADER_BYTES}-byte header + [{HASH_BITS}, {EMBED_DIM}] f32), got {}",
            bytes.len()
        );
        let rows = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();
        Ok(Self { rows })
    }

    /// Read the seed matrix from disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading neuralhash seed {}", path.display()))?;
        Self::from_file_bytes(&bytes).with_context(|| format!("parsing {}", path.display()))
    }

    /// Build directly from a row-major `[96, 128]` slice (tests / synthetic seeds).
    pub fn from_rows(rows: Vec<f32>) -> Result<Self> {
        ensure!(
            rows.len() == HASH_BITS * EMBED_DIM,
            "neuralhash seed: expected {} values, got {}",
            HASH_BITS * EMBED_DIM,
            rows.len()
        );
        Ok(Self { rows })
    }

    /// Row-major `[96, 128]` view.
    pub fn as_slice(&self) -> &[f32] {
        &self.rows
    }

    /// `seed.dot(embedding)` → the 96 pre-threshold scores.
    ///
    /// Accumulates in f64 so the sign near a near-tie does not depend on
    /// summation order across backends; the reference runs the dot product in
    /// NumPy (f32 pairwise), and near-zero scores are exactly where the
    /// upstream "a few bits off" caveat bites.
    pub fn project(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            embedding.len() == EMBED_DIM,
            "neuralhash seed: expected a {EMBED_DIM}-float embedding, got {}",
            embedding.len()
        );
        let mut out = vec![0f32; HASH_BITS];
        for (r, o) in out.iter_mut().enumerate() {
            let row = &self.rows[r * EMBED_DIM..(r + 1) * EMBED_DIM];
            let mut acc = 0f64;
            for (w, x) in row.iter().zip(embedding.iter()) {
                acc += *w as f64 * *x as f64;
            }
            *o = acc as f32;
        }
        Ok(out)
    }
}

impl std::fmt::Debug for SeedMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SeedMatrix([{HASH_BITS}, {EMBED_DIM}])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_file(rows: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0u8; SEED_HEADER_BYTES];
        // A non-zero header proves the first 128 bytes really are skipped.
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        bytes.extend(rows.iter().flat_map(|v| v.to_le_bytes()));
        bytes
    }

    #[test]
    fn skips_the_header_and_reads_row_major() {
        let rows: Vec<f32> = (0..HASH_BITS * EMBED_DIM).map(|i| i as f32).collect();
        let seed = SeedMatrix::from_file_bytes(&synthetic_file(&rows)).unwrap();
        assert_eq!(seed.as_slice(), rows.as_slice());
        // Row r starts at r * 128.
        assert_eq!(seed.as_slice()[EMBED_DIM], EMBED_DIM as f32);
    }

    #[test]
    fn project_is_a_row_major_matvec() {
        // Row r selects element r of the embedding (identity in the first 96).
        let mut rows = vec![0f32; HASH_BITS * EMBED_DIM];
        for r in 0..HASH_BITS {
            rows[r * EMBED_DIM + r] = 1.0;
        }
        let seed = SeedMatrix::from_rows(rows).unwrap();
        let emb: Vec<f32> = (0..EMBED_DIM).map(|i| i as f32 - 40.0).collect();
        let scores = seed.project(&emb).unwrap();
        assert_eq!(scores.len(), HASH_BITS);
        for (r, s) in scores.iter().enumerate() {
            assert_eq!(*s, emb[r], "row {r}");
        }
    }

    #[test]
    fn size_mismatches_are_errors() {
        assert!(SeedMatrix::from_file_bytes(&[0u8; 64]).is_err());
        assert!(SeedMatrix::from_file_bytes(&[0u8; SEED_HEADER_BYTES]).is_err());
        // Truncated payload.
        assert!(SeedMatrix::from_file_bytes(&vec![0u8; SEED_FILE_BYTES - 4]).is_err());
        // Trailing garbage — `np.frombuffer(...).reshape([96,128])` would also fail.
        assert!(SeedMatrix::from_file_bytes(&vec![0u8; SEED_FILE_BYTES + 4]).is_err());
        assert!(SeedMatrix::from_file_bytes(&vec![0u8; SEED_FILE_BYTES]).is_ok());
    }

    #[test]
    fn project_rejects_wrong_embedding_width() {
        let seed = SeedMatrix::from_rows(vec![0f32; HASH_BITS * EMBED_DIM]).unwrap();
        assert!(seed.project(&[0.0; 96]).is_err());
        assert!(seed.project(&[0.0; EMBED_DIM]).is_ok());
    }
}
