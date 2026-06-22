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

//! Agglomerative clustering on speaker embeddings.

use crate::embed::cosine;

pub fn cluster_embeddings(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let mut labels = vec![0usize; n];
    let mut next_cluster = 1usize;
    for i in 1..n {
        let sim = cosine(&embeddings[i], &embeddings[i - 1]);
        if sim >= 1.0 - threshold {
            labels[i] = labels[i - 1];
        } else {
            labels[i] = next_cluster;
            next_cluster += 1;
        }
    }
    labels
}
