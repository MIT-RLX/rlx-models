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

//! Dense f32 tensors in Espresso's layout.
//!
//! Espresso describes every blob as `(n, k, h, w)` with **`w` innermost**, and
//! `.espresso.shape` reports a `_rank` saying how many of those are meaningful.
//! A rank-2 activation is therefore `[h, w]` — `h` rows of `w` features — which
//! is the transpose of how a transformer is usually written down, so the layout
//! is kept explicit rather than being flattened into "rows and columns".
//!
//! Shapes here are stored outermost-first ([`Tensor::dims`]) because that is
//! the order every op reasons in; [`crate::net::Shape`] converts.

use anyhow::{Result, ensure};

/// A dense row-major f32 tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Outermost-first, innermost last.
    dims: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Wraps `data` with `dims`, checking the element count.
    pub fn new(dims: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let want: usize = dims.iter().product();
        ensure!(
            want == data.len(),
            "shape {dims:?} needs {want} elements, got {}",
            data.len()
        );
        Ok(Self { dims, data })
    }

    /// Zero tensor.
    pub fn zeros(dims: Vec<usize>) -> Self {
        let n = dims.iter().product();
        Self {
            dims,
            data: vec![0.0; n],
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Innermost dimension — the feature width for an activation.
    pub fn width(&self) -> usize {
        self.dims.last().copied().unwrap_or(1)
    }

    /// Product of all but the innermost dimension: how many rows of
    /// [`Tensor::width`] the tensor holds.
    pub fn rows(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims[..self.dims.len() - 1].iter().product()
        }
    }

    /// One innermost row.
    pub fn row(&self, i: usize) -> &[f32] {
        let w = self.width();
        &self.data[i * w..(i + 1) * w]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let w = self.width();
        &mut self.data[i * w..(i + 1) * w]
    }

    /// Reinterprets the buffer under new dimensions. One dimension may be `-1`
    /// (passed as `None`) and is solved from the element count, matching
    /// Espresso's `dst_*` of `-1`.
    pub fn reshape(&self, dims: &[Option<usize>]) -> Result<Self> {
        let known: usize = dims.iter().flatten().product();
        let holes = dims.iter().filter(|d| d.is_none()).count();
        ensure!(
            holes <= 1,
            "reshape has {holes} unknown dimensions, at most 1 allowed"
        );
        let solved: Vec<usize> = if holes == 1 {
            ensure!(
                known > 0 && self.len().is_multiple_of(known),
                "cannot solve reshape {dims:?} for {} elements",
                self.len()
            );
            let fill = self.len() / known;
            dims.iter().map(|d| d.unwrap_or(fill)).collect()
        } else {
            dims.iter().map(|d| d.expect("no holes")).collect()
        };
        Self::new(solved, self.data.clone())
    }

    /// Permutes axes. `perm[i]` is the source axis that becomes axis `i`.
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        ensure!(
            perm.len() == self.rank(),
            "permutation {perm:?} does not match rank {}",
            self.rank()
        );
        let mut seen = vec![false; perm.len()];
        for &p in perm {
            ensure!(p < perm.len(), "permutation axis {p} out of range");
            ensure!(!seen[p], "permutation {perm:?} repeats axis {p}");
            seen[p] = true;
        }
        let out_dims: Vec<usize> = perm.iter().map(|&p| self.dims[p]).collect();
        let src_strides = strides(&self.dims);
        let mut out = vec![0.0f32; self.len()];
        let mut idx = vec![0usize; self.rank()];
        for (flat, slot) in out.iter_mut().enumerate() {
            // Decompose `flat` in the *output* index space.
            let mut rem = flat;
            for (i, d) in out_dims.iter().enumerate().rev() {
                idx[i] = rem % d;
                rem /= d;
            }
            // Map back through the permutation to the source offset.
            let mut src = 0usize;
            for (i, &p) in perm.iter().enumerate() {
                src += idx[i] * src_strides[p];
            }
            *slot = self.data[src];
        }
        Self::new(out_dims, out)
    }
}

/// Row-major strides for `dims`.
pub fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_checks_the_element_count() {
        assert!(Tensor::new(vec![2, 3], vec![0.0; 6]).is_ok());
        assert!(Tensor::new(vec![2, 3], vec![0.0; 5]).is_err());
    }

    #[test]
    fn rows_and_width_follow_the_innermost_axis() {
        let t = Tensor::new(vec![2, 8, 64], vec![0.0; 1024]).expect("build");
        assert_eq!(t.width(), 64);
        assert_eq!(t.rows(), 16);
        assert_eq!(t.row(0).len(), 64);
    }

    #[test]
    fn strides_are_row_major() {
        assert_eq!(strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides(&[5]), vec![1]);
    }

    #[test]
    fn reshape_solves_a_single_hole() {
        let t = Tensor::new(vec![4, 6], (0..24).map(|v| v as f32).collect()).expect("build");
        let r = t.reshape(&[Some(3), None, Some(2)]).expect("reshape");
        assert_eq!(r.dims(), &[3, 4, 2]);
        assert_eq!(r.data(), t.data());
        assert!(t.reshape(&[None, None]).is_err(), "two holes must fail");
        assert!(
            t.reshape(&[Some(5), None]).is_err(),
            "24 is not divisible by 5"
        );
    }

    #[test]
    fn permute_moves_elements_not_just_labels() {
        // [2,3] laid out 0..5 -> transpose is [3,2] = 0,3, 1,4, 2,5
        let t = Tensor::new(vec![2, 3], (0..6).map(|v| v as f32).collect()).expect("build");
        let p = t.permute(&[1, 0]).expect("permute");
        assert_eq!(p.dims(), &[3, 2]);
        assert_eq!(p.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn permute_round_trips_on_rank_3() {
        let t = Tensor::new(vec![2, 3, 4], (0..24).map(|v| v as f32).collect()).expect("build");
        let once = t.permute(&[1, 2, 0]).expect("permute");
        assert_eq!(once.dims(), &[3, 4, 2]);
        // Inverse of [1,2,0] is [2,0,1].
        let back = once.permute(&[2, 0, 1]).expect("permute");
        assert_eq!(back.dims(), t.dims());
        assert_eq!(back.data(), t.data());
    }

    #[test]
    fn permute_rejects_bad_permutations() {
        let t = Tensor::zeros(vec![2, 3]);
        assert!(t.permute(&[0]).is_err(), "wrong length");
        assert!(t.permute(&[0, 0]).is_err(), "repeated axis");
        assert!(t.permute(&[0, 5]).is_err(), "out of range");
    }
}
