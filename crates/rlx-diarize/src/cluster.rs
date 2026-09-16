// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Centroid-based agglomerative clustering on speaker embeddings.

use crate::embed::cosine;

/// Cluster embeddings online: assign to nearest centroid if cosine ≥ `1 - threshold`,
/// else start a new cluster. Stronger than adjacent-only linking for multi-speaker audio.
pub fn cluster_embeddings(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let min_sim = 1.0 - threshold;
    let mut labels = vec![0usize; n];
    let mut centroids: Vec<Vec<f32>> = vec![embeddings[0].clone()];
    let mut counts: Vec<usize> = vec![1];

    for i in 1..n {
        let mut best = None::<(usize, f32)>;
        for (c, cent) in centroids.iter().enumerate() {
            let sim = cosine(&embeddings[i], cent);
            if best.map(|(_, s)| sim > s).unwrap_or(true) {
                best = Some((c, sim));
            }
        }
        if let Some((c, sim)) = best
            && sim >= min_sim
        {
            labels[i] = c;
            let cnt = counts[c] as f32;
            for (a, &b) in centroids[c].iter_mut().zip(embeddings[i].iter()) {
                *a = (*a * cnt + b) / (cnt + 1.0);
            }
            // re-normalize centroid
            let nrm: f32 = centroids[c].iter().map(|x| x * x).sum::<f32>().sqrt();
            if nrm > 1e-8 {
                for x in &mut centroids[c] {
                    *x /= nrm;
                }
            }
            counts[c] += 1;
            continue;
        }
        let id = centroids.len();
        labels[i] = id;
        centroids.push(embeddings[i].clone());
        counts.push(1);
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_vectors_one_cluster() {
        let e = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.99, 0.01]];
        let labels = cluster_embeddings(&e, 0.25);
        assert_eq!(labels, vec![0, 0, 0]);
    }

    #[test]
    fn orthogonal_two_clusters() {
        let e = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let labels = cluster_embeddings(&e, 0.25);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
    }
}
