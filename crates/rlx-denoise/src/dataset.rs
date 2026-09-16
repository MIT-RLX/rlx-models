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

//! Training tiles on disk.
//!
//! ```text
//! magic   "RLXDN002"   8 bytes
//! tile    u32          side length in pixels, a multiple of 8
//! count   u32          number of tiles
//! inputs  u32          input planes per tile
//! outputs u32          target planes per tile
//! tiles   f32 x count x (inputs + outputs) x tile x tile   little-endian, planar
//! ```
//!
//! The plane counts are read from the file rather than assumed, so a set built
//! for one network shape is rejected by another instead of being silently
//! reinterpreted. `RLXDN001` predates the fields and is read as 9 and 3.
//!
//! Planar rather than interleaved, because that is the NCHW layout the network
//! wants — a batch is then a contiguous copy rather than a transpose.
//!
//! # What the planes mean is not recorded here
//!
//! The header says how many input planes a tile has, not what they contain.
//! Nine planes are unambiguous — colour, albedo, normal — but an eleventh has
//! meant two different things ([`crate::model::Guides`]), and pairing weights
//! with a set that disagrees produces no error and a slightly wrong image.
//!
//! [`crate::checkpoint`] records the encoding on the weights, and the renderer
//! checks it there, which is where the pairing actually happens at run time. A
//! training run pairing a checkpoint with a dataset is the caller's
//! responsibility. This is why the shipped configuration is nine planes: the
//! ambiguity cannot arise.
//!
//! Radiance is unbounded and a convolution stack is not, so colour and target
//! arrive compressed by `x / (1 + x)`: bounded in `[0, 1)`, monotone, and
//! invertible as `x' / (1 - x')`. Albedo is already a reflectance and normals
//! are already unit vectors, so both are stored as they are.

use anyhow::{Result, bail, ensure};
use std::path::Path;

const MAGIC: &[u8; 8] = b"RLXDN002";
/// The original format: no plane counts, so colour/albedo/normal and a colour.
const MAGIC_V1: &[u8; 8] = b"RLXDN001";

/// A set of training tiles, held in memory.
pub struct Dataset {
    tile: usize,
    count: usize,
    inputs: usize,
    outputs: usize,
    /// `count` tiles, each `(inputs + outputs) * tile * tile` floats.
    data: Vec<f32>,
}

impl Dataset {
    /// Read a dataset file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("denoise: cannot read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= 16,
            "dataset is {} bytes, too short for a header",
            bytes.len()
        );
        let v1 = &bytes[..8] == MAGIC_V1;
        if &bytes[..8] != MAGIC && !v1 {
            bail!(
                "dataset does not start with {}",
                String::from_utf8_lossy(MAGIC)
            );
        }
        let tile = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
        let count = u32::from_le_bytes(bytes[12..16].try_into()?) as usize;
        ensure!(
            tile > 0 && tile.is_multiple_of(8),
            "tile {tile} must be a positive multiple of 8"
        );

        // `RLXDN001` predates the plane counts, and had exactly one shape.
        let (inputs, outputs, header) = if v1 {
            (9usize, 3usize, 16usize)
        } else {
            ensure!(bytes.len() >= 24, "dataset header is truncated");
            (
                u32::from_le_bytes(bytes[16..20].try_into()?) as usize,
                u32::from_le_bytes(bytes[20..24].try_into()?) as usize,
                24usize,
            )
        };
        ensure!(
            inputs > 0 && outputs > 0,
            "a tile cannot have {inputs}+{outputs} planes"
        );

        let per_tile = (inputs + outputs) * tile * tile;
        let want = count * per_tile;
        let floats = (bytes.len() - header) / 4;
        ensure!(
            floats == want,
            "dataset holds {floats} floats, {count} tiles of {tile} with \
             {inputs}+{outputs} planes need {want}"
        );

        let mut data = Vec::with_capacity(want);
        for chunk in bytes[header..].chunks_exact(4) {
            data.push(f32::from_le_bytes(chunk.try_into()?));
        }

        // A NaN anywhere in an input plane propagates through every convolution
        // that touches it and comes out as a loss of NaN, at which point Adam
        // writes NaN into every weight and the run is over — with nothing in the
        // logs naming the plane that did it. Cheaper to refuse the file.
        if let Some(bad) = data.iter().position(|v| !v.is_finite()) {
            let per_tile = (inputs + outputs) * tile * tile;
            let plane = (bad % per_tile) / (tile * tile);
            bail!(
                "dataset holds {} at tile {}, plane {plane} — an input plane is not finite",
                data[bad],
                bad / per_tile
            );
        }

        Ok(Self {
            tile,
            count,
            inputs,
            outputs,
            data,
        })
    }

    /// Input planes per tile, as the file declares them.
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// Target planes per tile.
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// Check this set matches the network about to be trained on it.
    ///
    /// A mismatch is otherwise invisible: the tile count and the byte length
    /// both work out, and the network trains happily on planes that are not
    /// what it thinks they are.
    pub fn check_shape(&self, inputs: usize, outputs: usize) -> Result<()> {
        ensure!(
            self.inputs == inputs && self.outputs == outputs,
            "dataset holds {}+{} planes, this network wants {inputs}+{outputs}",
            self.inputs,
            self.outputs
        );
        Ok(())
    }

    pub fn tile(&self) -> usize {
        self.tile
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Gather `indices` into contiguous `input` and `target` buffers, the
    /// layout [`crate::Batch`] expects.
    pub fn gather(&self, indices: &[usize]) -> Result<(Vec<f32>, Vec<f32>)> {
        let pixels = self.tile * self.tile;
        let per_tile = (self.inputs + self.outputs) * pixels;
        let mut input = Vec::with_capacity(indices.len() * self.inputs * pixels);
        let mut target = Vec::with_capacity(indices.len() * self.outputs * pixels);
        for &i in indices {
            ensure!(
                i < self.count,
                "tile {i} is past the end of a {} tile set",
                self.count
            );
            let base = i * per_tile;
            input.extend_from_slice(&self.data[base..base + self.inputs * pixels]);
            target.extend_from_slice(&self.data[base + self.inputs * pixels..base + per_tile]);
        }
        Ok((input, target))
    }

    /// Relative-L2 error of the *unfiltered* colour against the target, which
    /// is the number a trained network has to beat.
    pub fn baseline_error(&self, epsilon: f32) -> f32 {
        let pixels = self.tile * self.tile;
        let per_tile = (self.inputs + self.outputs) * pixels;
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for i in 0..self.count {
            let base = i * per_tile;
            for c in 0..self.outputs {
                for p in 0..pixels {
                    let y = self.data[base + c * pixels + p] as f64;
                    let t = self.data[base + self.inputs * pixels + c * pixels + p] as f64;
                    let d = y - t;
                    sum += d * d / (t * t + epsilon as f64);
                    n += 1;
                }
            }
        }
        if n == 0 {
            0.0
        } else {
            (sum / n as f64).sqrt() as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{IN_CHANNELS, OUT_CHANNELS};

    fn synthetic(tile: usize, count: usize) -> Vec<u8> {
        let per_tile = (IN_CHANNELS + OUT_CHANNELS) * tile * tile;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(tile as u32).to_le_bytes());
        bytes.extend_from_slice(&(count as u32).to_le_bytes());
        bytes.extend_from_slice(&(IN_CHANNELS as u32).to_le_bytes());
        bytes.extend_from_slice(&(OUT_CHANNELS as u32).to_le_bytes());
        for i in 0..count * per_tile {
            bytes.extend_from_slice(&(i as f32 * 1e-3).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn a_dataset_round_trips() {
        let ds = Dataset::from_bytes(&synthetic(8, 3)).expect("parse");
        assert_eq!(ds.tile(), 8);
        assert_eq!(ds.len(), 3);
        let (input, target) = ds.gather(&[0, 2]).expect("gather");
        assert_eq!(input.len(), 2 * IN_CHANNELS * 64);
        assert_eq!(target.len(), 2 * OUT_CHANNELS * 64);
        // Tile 2's inputs start where tile 2 starts.
        let per_tile = (IN_CHANNELS + OUT_CHANNELS) * 64;
        assert_eq!(input[IN_CHANNELS * 64], (2 * per_tile) as f32 * 1e-3);
    }

    #[test]
    fn a_truncated_file_is_rejected() {
        let mut bytes = synthetic(8, 2);
        bytes.truncate(bytes.len() - 4);
        assert!(
            Dataset::from_bytes(&bytes).is_err(),
            "a short file must not parse"
        );
    }

    /// A file built for another network shape must be refused rather than
    /// reinterpreted — the byte count works out either way.
    #[test]
    fn a_set_of_the_wrong_shape_is_caught() {
        let ds = Dataset::from_bytes(&synthetic(8, 2)).expect("parse");
        assert_eq!(ds.inputs(), IN_CHANNELS);
        assert_eq!(ds.outputs(), OUT_CHANNELS);
        assert!(ds.check_shape(IN_CHANNELS, OUT_CHANNELS).is_ok());
        assert!(
            ds.check_shape(9, 3).is_err(),
            "a 9-plane network must be refused"
        );
    }

    /// The original format has no plane counts and exactly one shape.
    #[test]
    fn a_v1_file_reads_as_nine_and_three() {
        let tile = 8usize;
        let count = 2usize;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RLXDN001");
        bytes.extend_from_slice(&(tile as u32).to_le_bytes());
        bytes.extend_from_slice(&(count as u32).to_le_bytes());
        for i in 0..count * (9 + 3) * tile * tile {
            bytes.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let ds = Dataset::from_bytes(&bytes).expect("parse v1");
        assert_eq!(ds.inputs(), 9);
        assert_eq!(ds.outputs(), 3);
        assert_eq!(ds.len(), count);
    }

    /// A non-finite value has to be caught at load, naming the plane. Left
    /// alone it surfaces hours later as a loss of NaN with nothing to point at.
    #[test]
    fn a_non_finite_value_is_refused_with_its_plane() {
        let tile = 8usize;
        let mut bytes = synthetic(tile, 2);
        let per_tile = (IN_CHANNELS + OUT_CHANNELS) * tile * tile;
        // Plane 10 of tile 1 — the standard-error plane, which is exactly where
        // an infinite variance used to arrive.
        let index = per_tile + 10 * tile * tile + 3;
        let at = 24 + index * 4;
        bytes[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());

        let text = match Dataset::from_bytes(&bytes) {
            Ok(_) => panic!("NaN must not load"),
            Err(e) => e.to_string(),
        };
        assert!(text.contains("plane 10"), "should name the plane: {text}");
        assert!(text.contains("tile 1"), "should name the tile: {text}");
    }

    #[test]
    fn a_foreign_file_is_rejected() {
        let mut bytes = synthetic(8, 1);
        bytes[..8].copy_from_slice(b"NOTMINE!");
        assert!(Dataset::from_bytes(&bytes).is_err());
    }

    #[test]
    fn tiles_must_survive_three_halvings() {
        let mut bytes = synthetic(8, 1);
        bytes[8..12].copy_from_slice(&12u32.to_le_bytes());
        assert!(
            Dataset::from_bytes(&bytes).is_err(),
            "12 is not a multiple of 8"
        );
    }

    #[test]
    fn gathering_past_the_end_is_an_error() {
        let ds = Dataset::from_bytes(&synthetic(8, 2)).expect("parse");
        assert!(ds.gather(&[0, 5]).is_err());
    }
}
