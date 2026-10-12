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

//! CART-style decision trees — the `L^∞` representation of Section 4.1.
//!
//! Section 4.1 approximates gradients in `B = L^∞` rather than in the RKHS `H_K`
//! itself, because `H_K` is too constrained to hold a piecewise-constant
//! function. A [`RegressionTree`] is a complete binary tree over an axis-aligned
//! box: every leaf splits its widest side at the midpoint, so the leaves stay a
//! partition into boxes of controlled diameter. That diameter is what turns a
//! Lipschitz constant for the kernel into the `U_t` of Equation 5 — see
//! [`crate::tasks::rkhs`].
//!
//! The tree is *complete* (every leaf at the same depth), matching the paper's
//! reference implementation. Splitting on the widest side keeps leaves from
//! degenerating into slabs, which matters because the error bound scales with
//! the leaf diagonal, not with its volume.

/// A complete binary space-partitioning tree with one constant per leaf.
#[derive(Debug, Clone, PartialEq)]
pub struct RegressionTree {
    dim: usize,
    depth: u32,
    /// Split `(axis, threshold)` per internal node, in complete-tree order:
    /// node `i` has children `2i+1` (left, `x[axis] ≤ threshold`) and `2i+2`.
    splits: Vec<(u32, f64)>,
    /// One value per leaf; leaf `k` is complete-tree node `k + 2^depth − 1`.
    values: Vec<f64>,
    /// Leaf bounding boxes, flattened `dim` at a time.
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl RegressionTree {
    /// The trivial tree: one leaf holding `0` over the whole box.
    pub fn root(lower: &[f64], upper: &[f64]) -> Self {
        assert_eq!(lower.len(), upper.len(), "bounds must agree in dimension");
        assert!(!lower.is_empty(), "dimension must be positive");
        Self {
            dim: lower.len(),
            depth: 0,
            splits: Vec::new(),
            values: vec![0.0],
            lower: lower.to_vec(),
            upper: upper.to_vec(),
        }
    }

    /// The root over the unit box `[0, 1]^dim`, already refined `depth` times.
    pub fn unit(dim: usize, depth: u32) -> Self {
        let mut tree = Self::root(&vec![0.0; dim], &vec![1.0; dim]);
        for _ in 0..depth {
            tree = tree.refined();
        }
        tree
    }

    /// Input dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Tree depth; the leaf count is `2^depth`.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Number of leaves.
    pub fn leaves(&self) -> usize {
        self.values.len()
    }

    /// Leaf values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutable leaf values.
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// The bounding box of leaf `k`.
    pub fn leaf_bounds(&self, k: usize) -> (&[f64], &[f64]) {
        let range = k * self.dim..(k + 1) * self.dim;
        (&self.lower[range.clone()], &self.upper[range])
    }

    /// The centre of leaf `k`.
    pub fn leaf_center(&self, k: usize) -> Vec<f64> {
        let (lo, hi) = self.leaf_bounds(k);
        lo.iter().zip(hi).map(|(a, b)| 0.5 * (a + b)).collect()
    }

    /// Half the diagonal of leaf `k` — the radius `r` in a Lipschitz bound
    /// `sup_leaf |g − ∇L(f)| ≤ Lip · r`.
    pub fn leaf_radius(&self, k: usize) -> f64 {
        let (lo, hi) = self.leaf_bounds(k);
        0.5 * lo
            .iter()
            .zip(hi)
            .map(|(a, b)| (b - a) * (b - a))
            .sum::<f64>()
            .sqrt()
    }

    /// The largest leaf radius in the tree.
    pub fn max_leaf_radius(&self) -> f64 {
        (0..self.leaves())
            .map(|k| self.leaf_radius(k))
            .fold(0.0f64, f64::max)
    }

    /// The index of the leaf containing `x`.
    pub fn leaf_of(&self, x: &[f64]) -> usize {
        debug_assert_eq!(x.len(), self.dim);
        let mut node = 0usize;
        for _ in 0..self.depth {
            let (axis, threshold) = self.splits[node];
            node = 2 * node + if x[axis as usize] <= threshold { 1 } else { 2 };
        }
        node - (self.values.len() - 1)
    }

    /// The value of the encoded function at `x`.
    pub fn eval(&self, x: &[f64]) -> f64 {
        self.values[self.leaf_of(x)]
    }

    /// `‖f‖_{L^∞}` — the largest absolute leaf value.
    ///
    /// Exact for this representation: the function really is constant on each
    /// leaf, so its supremum is attained.
    pub fn sup_norm(&self) -> f64 {
        self.values.iter().fold(0.0f64, |a, v| a.max(v.abs()))
    }

    /// Split every leaf across its widest side at the midpoint, copying the
    /// parent value into both children. The encoded function is unchanged.
    pub fn refined(&self) -> Self {
        let n = self.leaves();
        let mut splits = self.splits.clone();
        splits.reserve(n);
        let mut values = Vec::with_capacity(2 * n);
        let mut lower = Vec::with_capacity(2 * n * self.dim);
        let mut upper = Vec::with_capacity(2 * n * self.dim);

        for k in 0..n {
            let (lo, hi) = self.leaf_bounds(k);
            let axis = (0..self.dim)
                .max_by(|&a, &b| {
                    (hi[a] - lo[a])
                        .partial_cmp(&(hi[b] - lo[b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("dimension is positive");
            let threshold = lo[axis] + 0.5 * (hi[axis] - lo[axis]);
            splits.push((axis as u32, threshold));

            // Left child: x[axis] ≤ threshold.
            values.push(self.values[k]);
            lower.extend_from_slice(lo);
            upper.extend_from_slice(hi);
            upper[k * 2 * self.dim + axis] = threshold;

            // Right child.
            values.push(self.values[k]);
            lower.extend_from_slice(lo);
            upper.extend_from_slice(hi);
            lower[(k * 2 + 1) * self.dim + axis] = threshold;
        }

        Self {
            dim: self.dim,
            depth: self.depth + 1,
            splits,
            values,
            lower,
            upper,
        }
    }

    /// `self ← self − η·gradient`.
    pub fn descend_by(&mut self, learning_rate: f64, gradient: &Self) {
        assert_eq!(
            self.values.len(),
            gradient.values.len(),
            "gradient must be at the iterate's refinement level"
        );
        for (value, g) in self.values.iter_mut().zip(&gradient.values) {
            *value -= learning_rate * g;
        }
    }
}

impl crate::algorithm::Representation for RegressionTree {
    fn dofs(&self) -> usize {
        self.values.len()
    }

    fn level(&self) -> u32 {
        self.depth
    }

    fn refine(&self) -> Self {
        self.refined()
    }

    fn descend(&mut self, learning_rate: f64, gradient: &Self) {
        self.descend_by(learning_rate, gradient);
    }
}
