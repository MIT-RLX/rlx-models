//! Product quantization (k-means) for **data-optimal** codebook-synthesis init.
//!
//! The runtime model (`crate::model::synth_linear`) replaces each weight
//! `W [k,n]` (consumed `x·W`) with a fixed u8 index table `[n, k/ED]` + a trained
//! codebook `[NE, ED]`; `Op::SynthMatMul` reconstructs `Wᵀ[n,k]` inside the
//! matmul (`out[i,j] = Σ_p x[i,p]·Wᵀ[j,p]`, so `Wᵀ[j,p] = W[p,j]`). By default
//! the indices are a *random frozen* SplitMix assignment and the codebook starts
//! at `N(0,0.02)` — low effective DOF, quality plateau.
//!
//! This module derives BOTH the indices and the codebook from a **trained dense
//! weight** by product quantization: reshape `Wᵀ` into `n·(k/ED)` blocks of
//! length `ED`, run k-means (`NE` centroids) → the centroids are the codebook and
//! each block's nearest-centroid id is its index. Same u8 index bytes, same
//! `[NE,ED]` codebook — but now the codebook *encodes real weights*, so the
//! model starts far closer to the dense reference (measured before any
//! fine-tuning). Multi-stage VQ initializes stage `s+1` from the *residual*
//! `W − Ŵ` (residual k-means); the optional LoRA correction is initialized from a
//! truncated SVD of the leftover residual.

/// SplitMix64 → the deterministic RNG that seeds k-means++ and the SVD power
/// iteration, so `pq_quantize` is a pure function of its inputs (reproducible).
struct SplitMix(u64);

impl SplitMix {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    #[inline]
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
    /// Standard normal via Box–Muller (for SVD power-iteration seeds).
    #[inline]
    fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Fixed seed — `pq_quantize` takes no seed argument (its output must match the
/// baked index constant deterministically), so the k-means RNG is seeded here.
const PQ_SEED: u64 = 0x1234_5678_9ABC_DEF0;
/// k-means Lloyd iterations. Cheap at `ED=4`; plenty for a stable codebook.
const KMEANS_ITERS: usize = 16;
/// Joint additive-VQ (AQ) refinement sweeps applied AFTER greedy residual VQ.
/// Each sweep is monotone (ICM + codebook re-estimation are coordinate descent on
/// the same `‖W − Σ_s C_s[idx_s]‖²` objective), so more never hurts; it saturates
/// fast. `0` reproduces the old greedy-only init exactly.
const REFINE_ITERS: usize = 6;

#[inline]
fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

/// Product-quantize a weight `w` (stored `[k,n]` row-major, consumed `x·W`) into
/// `indices [n, k/entry_dim]` (u8) + `codebook [num_entries, entry_dim]` (f32),
/// exactly the two tensors `synth_linear` bakes/trains. Reconstruction is
/// `Ŵᵀ[j, kb·ED+t] = codebook[indices[j·nb+kb]]·[t]` (i.e. `Ŵ[kb·ED+t, j]`).
///
/// Runs k-means (k-means++ init, `KMEANS_ITERS` Lloyd steps) over the
/// `n·(k/entry_dim)` length-`entry_dim` blocks of `Wᵀ`. `num_entries ≤ 256`.
pub fn pq_quantize(
    w: &[f32],
    k: usize,
    n: usize,
    entry_dim: usize,
    num_entries: usize,
) -> (Vec<u8>, Vec<f32>) {
    assert!(entry_dim >= 1, "entry_dim must be >= 1");
    assert_eq!(
        k % entry_dim,
        0,
        "entry_dim ({entry_dim}) must divide k ({k})"
    );
    assert!(
        num_entries <= 256,
        "num_entries ({num_entries}) must fit a u8 index"
    );
    assert_eq!(w.len(), k * n, "w must be [k,n] = {k}x{n}");

    let ed = entry_dim;
    let nb = k / ed; // codebook blocks per output column
    let num_blocks = n * nb;

    // Gather the blocks of Wᵀ. Block `bi = j*nb + kb` holds `Wᵀ[j, kb·ED .. ]`,
    // i.e. `W[kb·ED+t, j] = w[(kb·ED+t)*n + j]` — a strided read of the [k,n] buffer.
    let mut blocks = vec![0f32; num_blocks * ed];
    for j in 0..n {
        for kb in 0..nb {
            let dst = (j * nb + kb) * ed;
            for t in 0..ed {
                blocks[dst + t] = w[(kb * ed + t) * n + j];
            }
        }
    }
    let block = |bi: usize| &blocks[bi * ed..bi * ed + ed];

    let mut rng = SplitMix::new(PQ_SEED);
    let mut codebook = vec![0f32; num_entries * ed];

    // ── k-means++ seeding: spread the initial centroids by D² sampling. ──────
    {
        let c0 = rng.below(num_blocks.max(1));
        codebook[0..ed].copy_from_slice(block(c0));
        let mut d2: Vec<f32> = (0..num_blocks)
            .map(|bi| dist2(block(bi), &codebook[0..ed]))
            .collect();
        for c in 1..num_entries {
            let sum: f64 = d2.iter().map(|&x| x as f64).sum();
            let pick = if sum <= 0.0 {
                rng.below(num_blocks.max(1)) // all remaining blocks coincide
            } else {
                let target = rng.unit() as f64 * sum;
                let mut acc = 0.0;
                let mut idx = num_blocks - 1;
                for (bi, &x) in d2.iter().enumerate() {
                    acc += x as f64;
                    if acc >= target {
                        idx = bi;
                        break;
                    }
                }
                idx
            };
            codebook[c * ed..c * ed + ed].copy_from_slice(block(pick));
            for (bi, dd) in d2.iter_mut().enumerate() {
                *dd = dd.min(dist2(block(bi), &codebook[c * ed..c * ed + ed]));
            }
        }
    }

    // ── Lloyd iterations. ────────────────────────────────────────────────────
    let mut assign = vec![0u8; num_blocks];
    for _ in 0..KMEANS_ITERS {
        // Assign every block to its nearest centroid.
        for bi in 0..num_blocks {
            let b = block(bi);
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..num_entries {
                let d = dist2(b, &codebook[c * ed..c * ed + ed]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assign[bi] = best as u8;
        }
        // Recompute centroids as the mean of their members.
        let mut sums = vec![0f64; num_entries * ed];
        let mut counts = vec![0usize; num_entries];
        for bi in 0..num_blocks {
            let c = assign[bi] as usize;
            counts[c] += 1;
            let b = block(bi);
            for t in 0..ed {
                sums[c * ed + t] += b[t] as f64;
            }
        }
        for c in 0..num_entries {
            if counts[c] > 0 {
                for t in 0..ed {
                    codebook[c * ed + t] = (sums[c * ed + t] / counts[c] as f64) as f32;
                }
            } else {
                // Empty cluster: reseed to a random block so the code stays useful.
                let bi = rng.below(num_blocks.max(1));
                codebook[c * ed..c * ed + ed].copy_from_slice(block(bi));
            }
        }
    }

    // Final assignment against the converged centroids.
    let mut indices = vec![0u8; num_blocks];
    for bi in 0..num_blocks {
        let b = block(bi);
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..num_entries {
            let d = dist2(b, &codebook[c * ed..c * ed + ed]);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        indices[bi] = best as u8;
    }
    (indices, codebook)
}

/// Reconstruct the dense weight `Ŵ [k,n]` (row-major, same layout as the input
/// to [`pq_quantize`]) from `indices [n, k/ED]` + `codebook [NE, ED]`:
/// `Ŵ[kb·ED+t, j] = codebook[indices[j·nb+kb]]·[t]`. Used to form residuals for
/// multi-stage VQ and to measure reconstruction error.
pub fn reconstruct(
    indices: &[u8],
    codebook: &[f32],
    k: usize,
    n: usize,
    entry_dim: usize,
) -> Vec<f32> {
    let ed = entry_dim;
    let nb = k / ed;
    let mut out = vec![0f32; k * n];
    for j in 0..n {
        for kb in 0..nb {
            let c = indices[j * nb + kb] as usize * ed;
            for t in 0..ed {
                out[(kb * ed + t) * n + j] = codebook[c + t];
            }
        }
    }
    out
}

/// Relative Frobenius reconstruction error `‖Ŵ − W‖ / ‖W‖` for a PQ of `w`.
/// A convenience for tests / the `--init-from` report.
pub fn reconstruction_error(
    w: &[f32],
    k: usize,
    n: usize,
    entry_dim: usize,
    num_entries: usize,
) -> f32 {
    let (idx, cb) = pq_quantize(w, k, n, entry_dim, num_entries);
    let approx = reconstruct(&idx, &cb, k, n, entry_dim);
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in approx.iter().zip(w) {
        let d = (*a - *b) as f64;
        num += d * d;
        den += (*b as f64) * (*b as f64);
    }
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt() as f32
    }
}

/// The per-weight PQ init for a full synth layer: `stages` rounds of residual
/// k-means (stage `s+1` quantizes `W − Σ_{≤s} Ŵ`), then a rank-`lora_rank`
/// truncated-SVD fit of whatever residual is left (in the dense `[k,n]` space,
/// so it drops straight into the `A·Bᵀ` LoRA correction).
///
/// Returns `(stage_indices, stage_codebooks, lora_a [k·r], lora_b [n·r])`.
/// `lora_a`/`lora_b` are empty when `lora_rank == 0`.
#[allow(clippy::type_complexity)]
pub fn pq_multistage(
    w: &[f32],
    k: usize,
    n: usize,
    entry_dim: usize,
    num_entries: usize,
    stages: usize,
    lora_rank: usize,
) -> (Vec<Vec<u8>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let mut residual = w.to_vec(); // [k,n], updated in place as stages peel off
    let mut stage_idx = Vec::with_capacity(stages);
    let mut stage_cb = Vec::with_capacity(stages);
    for _ in 0..stages {
        let (idx, cb) = pq_quantize(&residual, k, n, entry_dim, num_entries);
        let approx = reconstruct(&idx, &cb, k, n, entry_dim);
        for (r, a) in residual.iter_mut().zip(&approx) {
            *r -= *a;
        }
        stage_idx.push(idx);
        stage_cb.push(cb);
    }
    // Joint AQ refinement: greedy residual VQ picks each stage against the PREVIOUS
    // stages' residual only, so the codebooks aren't jointly optimal. Re-optimize
    // indices + codebooks against the FULL additive sum — strictly lowers the
    // reconstruction error at the same tensors (free at inference). The dense
    // `residual` for the LoRA fit is then recomputed from the refined stages.
    refine_additive_vq(
        w,
        k,
        n,
        entry_dim,
        num_entries,
        &mut stage_idx,
        &mut stage_cb,
        REFINE_ITERS,
    );
    residual.copy_from_slice(w);
    for st in 0..stages {
        let approx = reconstruct(&stage_idx[st], &stage_cb[st], k, n, entry_dim);
        for (r, a) in residual.iter_mut().zip(&approx) {
            *r -= *a;
        }
    }
    let (lora_a, lora_b) = if lora_rank > 0 {
        low_rank_approx(&residual, k, n, lora_rank)
    } else {
        (Vec::new(), Vec::new())
    };
    (stage_idx, stage_cb, lora_a, lora_b)
}

/// Joint additive-VQ (AQ) refinement of the residual stages — the free,
/// architecture-preserving stand-in for OPQ (whose `k×k` rotation can't fold
/// through this model's LayerNorm / residual / attention neighbors, so it would
/// add an inference-time matmul; the AQ objective lowers the same error with the
/// SAME codebook + index tensors, zero inference cost).
///
/// Minimizes the joint reconstruction `Σ_bi ‖t_bi − Σ_s C_s[idx_s[bi]]‖²` over the
/// `ED`-length blocks `t_bi` of `Wᵀ`, alternating two monotone coordinate-descent
/// sweeps:
///   • **ICM index update** — for each block, each stage, reassign `idx_s[bi]` to
///     the entry that best fits the residual left by the *other* stages;
///   • **codebook re-estimation** — each entry ← mean of the residual over the
///     blocks currently assigned to it (optimal given the other stages fixed).
/// Both never increase the objective, so the result is ≥ as good as greedy init.
#[allow(clippy::too_many_arguments)]
fn refine_additive_vq(
    w: &[f32],
    k: usize,
    n: usize,
    ed: usize,
    num_entries: usize,
    stage_idx: &mut [Vec<u8>],
    stage_cb: &mut [Vec<f32>],
    iters: usize,
) {
    let stages = stage_idx.len();
    if stages == 0 || iters == 0 {
        return;
    }
    let nb = k / ed;
    let num_blocks = n * nb;
    // Block targets `t_bi = Wᵀ[j, kb·ED..]` — the same strided gather `pq_quantize`
    // forms (block `bi = j·nb + kb`).
    let mut tgt = vec![0f32; num_blocks * ed];
    for j in 0..n {
        for kb in 0..nb {
            let dst = (j * nb + kb) * ed;
            for t in 0..ed {
                tgt[dst + t] = w[(kb * ed + t) * n + j];
            }
        }
    }
    let mut resid = vec![0f32; ed]; // scratch: target minus the OTHER stages
    for _ in 0..iters {
        // ── ICM: reassign every (block, stage) index against the other stages. ──
        for bi in 0..num_blocks {
            for s in 0..stages {
                resid.copy_from_slice(&tgt[bi * ed..bi * ed + ed]);
                for (s2, cb2) in stage_cb.iter().enumerate() {
                    if s2 == s {
                        continue;
                    }
                    let c = stage_idx[s2][bi] as usize * ed;
                    for t in 0..ed {
                        resid[t] -= cb2[c + t];
                    }
                }
                let cb = &stage_cb[s];
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for e in 0..num_entries {
                    let c = e * ed;
                    let mut d = 0f32;
                    for t in 0..ed {
                        let df = resid[t] - cb[c + t];
                        d += df * df;
                    }
                    if d < best_d {
                        best_d = d;
                        best = e;
                    }
                }
                stage_idx[s][bi] = best as u8;
            }
        }
        // ── Codebook re-estimation: each entry ← mean residual of its blocks. ──
        for s in 0..stages {
            let mut sums = vec![0f64; num_entries * ed];
            let mut counts = vec![0usize; num_entries];
            for bi in 0..num_blocks {
                resid.copy_from_slice(&tgt[bi * ed..bi * ed + ed]);
                for (s2, cb2) in stage_cb.iter().enumerate() {
                    if s2 == s {
                        continue;
                    }
                    let c = stage_idx[s2][bi] as usize * ed;
                    for t in 0..ed {
                        resid[t] -= cb2[c + t];
                    }
                }
                let e = stage_idx[s][bi] as usize;
                counts[e] += 1;
                for t in 0..ed {
                    sums[e * ed + t] += resid[t] as f64;
                }
            }
            let cb = &mut stage_cb[s];
            for e in 0..num_entries {
                if counts[e] > 0 {
                    for t in 0..ed {
                        cb[e * ed + t] = (sums[e * ed + t] / counts[e] as f64) as f32;
                    }
                }
            }
        }
    }
}

/// Rank-`r` truncated SVD of `R [k,n]` (row-major) via power iteration with
/// deflation, returned as `A [k,r]` and `B [n,r]` (row-major) with `R ≈ A·Bᵀ` —
/// exactly the `synth_linear` LoRA factors (`lora = (x·A)·Bᵀ`). Column `c` is
/// `σ_c·u_c` / `v_c` for the c-th singular triple.
fn low_rank_approx(r: &[f32], k: usize, n: usize, rank: usize) -> (Vec<f32>, Vec<f32>) {
    const POWER_ITERS: usize = 64;
    let mut res = r.to_vec(); // deflated copy
    let mut a = vec![0f32; k * rank];
    let mut b = vec![0f32; n * rank];
    let mut rng = SplitMix::new(PQ_SEED ^ 0xA5A5_A5A5);
    let mut u = vec![0f32; k];
    let mut v = vec![0f32; n];
    for c in 0..rank {
        for vj in v.iter_mut() {
            *vj = rng.normal();
        }
        normalize(&mut v);
        let mut sigma = 0f32;
        for _ in 0..POWER_ITERS {
            // u = R v
            for (p, up) in u.iter_mut().enumerate() {
                let row = &res[p * n..p * n + n];
                *up = row.iter().zip(&v).map(|(&rr, &vv)| rr * vv).sum();
            }
            if normalize(&mut u) == 0.0 {
                break;
            }
            // v = Rᵀ u
            for (j, vj) in v.iter_mut().enumerate() {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += res[p * n + j] * u[p];
                }
                *vj = acc;
            }
            sigma = norm(&v);
            if sigma == 0.0 {
                break;
            }
            for vj in v.iter_mut() {
                *vj /= sigma;
            }
        }
        if sigma == 0.0 {
            break; // residual exhausted; leave remaining columns at zero
        }
        for p in 0..k {
            a[p * rank + c] = sigma * u[p];
        }
        for j in 0..n {
            b[j * rank + c] = v[j];
        }
        // Deflate: R -= σ · u vᵀ.
        for p in 0..k {
            let up = sigma * u[p];
            let row = &mut res[p * n..p * n + n];
            for (j, rr) in row.iter_mut().enumerate() {
                *rr -= up * v[j];
            }
        }
    }
    (a, b)
}

#[inline]
fn norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// Normalize in place, returning the original norm (0 ⇒ left untouched).
#[inline]
fn normalize(x: &mut [f32]) -> f32 {
    let nrm = norm(x);
    if nrm > 0.0 {
        for v in x.iter_mut() {
            *v /= nrm;
        }
    }
    nrm
}

#[cfg(test)]
mod tests {
    use super::*;

    // A deterministic pseudo-random [k,n] weight.
    fn rand_weight(k: usize, n: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix::new(seed);
        (0..k * n).map(|_| rng.normal() * 0.02).collect()
    }

    #[test]
    fn pq_reconstruction_error_is_small_and_shrinks_with_entries() {
        let (k, n, ed) = (32usize, 48usize, 4usize);
        let w = rand_weight(k, n, 7);

        // Shapes are exactly what synth_linear consumes.
        let (idx, cb) = pq_quantize(&w, k, n, ed, 256);
        assert_eq!(idx.len(), n * (k / ed));
        assert_eq!(cb.len(), 256 * ed);

        // Reconstruction error drops monotonically as the codebook grows.
        let e16 = reconstruction_error(&w, k, n, ed, 16);
        let e64 = reconstruction_error(&w, k, n, ed, 64);
        let e256 = reconstruction_error(&w, k, n, ed, 256);
        assert!(
            e16 > e64 && e64 > e256,
            "error should shrink with num_entries: {e16} {e64} {e256}"
        );
        // 256 four-dim centroids over a random weight — comfortably under 40%.
        assert!(e256 < 0.4, "256-entry PQ error too high: {e256}");
    }

    #[test]
    fn residual_stages_reduce_error() {
        let (k, n, ed) = (48usize, 32usize, 4usize);
        let w = rand_weight(k, n, 11);

        // One stage vs two stages (no LoRA): the second stage must cut the error.
        let (idx1, cb1, _, _) = pq_multistage(&w, k, n, ed, 256, 1, 0);
        let approx1 = reconstruct(&idx1[0], &cb1[0], k, n, ed);
        let err1 = rel_err(&approx1, &w);

        let (idx2, cb2, _, _) = pq_multistage(&w, k, n, ed, 256, 2, 0);
        let mut sum = vec![0f32; k * n];
        for st in 0..2 {
            let a = reconstruct(&idx2[st], &cb2[st], k, n, ed);
            for (s, v) in sum.iter_mut().zip(&a) {
                *s += *v;
            }
        }
        let err2 = rel_err(&sum, &w);
        assert!(err2 < err1, "two stages should beat one: {err1} -> {err2}");
    }

    #[test]
    fn lora_residual_lowers_error_further() {
        // More blocks (n·k/ED = 1024) than centroids (32), so single-stage PQ
        // leaves a real residual for the LoRA SVD to eat into.
        let (k, n, ed, ne, r) = (64usize, 64usize, 4usize, 32usize, 8usize);
        let w = rand_weight(k, n, 5);
        let (idx, cb, a, b) = pq_multistage(&w, k, n, ed, ne, 1, r);
        assert_eq!(a.len(), k * r);
        assert_eq!(b.len(), n * r);

        // Stage reconstruction alone.
        let mut approx = reconstruct(&idx[0], &cb[0], k, n, ed);
        let err_stage = rel_err(&approx, &w);
        assert!(err_stage > 0.0, "expected a nonzero residual to correct");
        // Add the LoRA correction A·Bᵀ (dense [k,n]).
        for p in 0..k {
            for j in 0..n {
                let mut acc = 0f32;
                for c in 0..r {
                    acc += a[p * r + c] * b[j * r + c];
                }
                approx[p * n + j] += acc;
            }
        }
        let err_lora = rel_err(&approx, &w);
        assert!(
            err_lora < err_stage,
            "LoRA residual should lower error: {err_stage} -> {err_lora}"
        );
    }

    fn rel_err(approx: &[f32], w: &[f32]) -> f32 {
        let mut num = 0f64;
        let mut den = 0f64;
        for (a, b) in approx.iter().zip(w) {
            let d = (*a - *b) as f64;
            num += d * d;
            den += (*b as f64) * (*b as f64);
        }
        (num / den).sqrt() as f32
    }

    /// Sum the additive reconstruction `Σ_s C_s[idx_s]` of a multistage result.
    fn reconstruct_stages(
        idx: &[Vec<u8>],
        cb: &[Vec<f32>],
        k: usize,
        n: usize,
        ed: usize,
    ) -> Vec<f32> {
        let mut sum = vec![0f32; k * n];
        for st in 0..idx.len() {
            let a = reconstruct(&idx[st], &cb[st], k, n, ed);
            for (s, v) in sum.iter_mut().zip(&a) {
                *s += *v;
            }
        }
        sum
    }

    #[test]
    fn additive_refinement_beats_greedy_residual() {
        // Fewer centroids than blocks so greedy residual VQ leaves a real,
        // jointly-improvable error (n·k/ED = 512 blocks, NE = 64).
        let (k, n, ed, ne, stages) = (32usize, 64usize, 4usize, 64usize, 2usize);
        let w = rand_weight(k, n, 23);

        // Greedy-only baseline: quantize each stage against the running residual,
        // exactly what pq_multistage did before AQ refinement (REFINE_ITERS = 0).
        let mut resid = w.clone();
        let mut gidx = Vec::new();
        let mut gcb = Vec::new();
        for _ in 0..stages {
            let (idx, cb) = pq_quantize(&resid, k, n, ed, ne);
            let a = reconstruct(&idx, &cb, k, n, ed);
            for (r, x) in resid.iter_mut().zip(&a) {
                *r -= *x;
            }
            gidx.push(idx);
            gcb.push(cb);
        }
        let greedy_err = rel_err(&reconstruct_stages(&gidx, &gcb, k, n, ed), &w);

        // Refined: pq_multistage now runs REFINE_ITERS of joint AQ on top.
        let (ridx, rcb, _, _) = pq_multistage(&w, k, n, ed, ne, stages, 0);
        let refined_err = rel_err(&reconstruct_stages(&ridx, &rcb, k, n, ed), &w);

        assert!(
            refined_err < greedy_err,
            "AQ refinement should lower error: greedy {greedy_err} -> refined {refined_err}"
        );
        // Same tensor shapes — the win is free at inference.
        assert_eq!(ridx.len(), stages);
        assert_eq!(rcb[0].len(), ne * ed);
    }
}
