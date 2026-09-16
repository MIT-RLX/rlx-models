//! Fitting `J_l` over a corpus.
//!
//! A single prompt gives a noisy `J_l`. The lens is defined as an *expectation*
//! over prompts, positions and targets, and it is the averaging that makes it
//! prompt-independent — fit once, apply to anything. The reference uses 1000
//! sequences of 128 tokens and reports quality saturating around 100.
//!
//! Two things follow from the fit being long. It needs to survive a crash, so
//! [`CorpusFit`] checkpoints the running sum atomically and resumes from it. And
//! it needs to say when it has converged, so each prompt reports the diagnostics
//! the reference logs: the prompt's own `‖J‖/√d`, which flags heavy-tailed
//! outliers, and the relative shift of the running mean, which falls like `1/n`
//! once the estimate has settled.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use crate::fit::Jacobian;
use crate::lens::JacobianLens;

/// What one prompt did to the running estimate.
#[derive(Debug, Clone, Copy)]
pub struct FitProgress {
    /// Prompts folded in so far.
    pub n_done: usize,
    /// `max_l ‖J_l‖_F / √d` for *this* prompt. A prompt far above the others is
    /// an outlier worth looking at rather than averaging in blind.
    pub scaled_norm: f32,
    /// How far this prompt moved the running mean, relative to the mean's own
    /// size. Falls like `1/n` once converged; `NaN` for the first prompt, which
    /// has no mean to move.
    pub mean_rel_change: f32,
}

/// A running mean of per-layer Jacobians, checkpointable mid-corpus.
#[derive(Debug, Clone)]
pub struct CorpusFit {
    /// Running **sum**, not mean — dividing only at the end avoids rescaling
    /// every entry on every prompt.
    sums: BTreeMap<usize, Jacobian>,
    d_model: usize,
    target_layer: usize,
    /// Prompts successfully folded in.
    n_done: usize,
    /// Index of the next prompt to process. Tracked apart from `n_done` so a
    /// prompt that was *skipped* (too short, say) is not retried on resume.
    next_idx: usize,
}

impl CorpusFit {
    pub fn new(layers: &[usize], d_model: usize, target_layer: usize) -> Result<Self> {
        ensure!(!layers.is_empty(), "a fit needs at least one layer");
        ensure!(d_model > 0, "d_model must be non-zero");
        Ok(Self {
            sums: layers
                .iter()
                .map(|&l| (l, Jacobian::zeros(d_model)))
                .collect(),
            d_model,
            target_layer,
            n_done: 0,
            next_idx: 0,
        })
    }

    pub fn layers(&self) -> Vec<usize> {
        self.sums.keys().copied().collect()
    }

    pub fn n_done(&self) -> usize {
        self.n_done
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// The layer these Jacobians transport *into*.
    pub fn target_layer(&self) -> usize {
        self.target_layer
    }

    /// Index of the next prompt to process — where a resumed run picks up.
    pub fn next_idx(&self) -> usize {
        self.next_idx
    }

    /// Note that the prompt at `next_idx` was skipped, without folding anything
    /// in. Advances past it so a resume does not retry it.
    pub fn skip(&mut self) {
        self.next_idx += 1;
    }

    /// Fold one prompt's Jacobians into the running mean.
    ///
    /// `jacobians` must be in [`Self::layers`] order — the order `StackLens`
    /// returns them.
    pub fn observe(&mut self, jacobians: &[Jacobian]) -> Result<FitProgress> {
        let layers = self.layers();
        ensure!(
            jacobians.len() == layers.len(),
            "got {} Jacobians for {} layers",
            jacobians.len(),
            layers.len()
        );

        let sqrt_d = (self.d_model as f32).sqrt();
        let mut scaled_norm = 0.0f32;
        let mut mean_rel_change = if self.n_done == 0 { f32::NAN } else { 0.0 };

        for (layer, j) in layers.iter().zip(jacobians) {
            ensure!(
                j.d_model == self.d_model,
                "layer {layer} is {}-wide, fit is {}-wide",
                j.d_model,
                self.d_model
            );
            scaled_norm = scaled_norm.max(frob(&j.values) / sqrt_d);

            let sum = self.sums.get_mut(layer).expect("layer present");
            if self.n_done > 0 {
                // ‖J_prompt − mean‖ / ((n+1)·‖mean‖): the shift this prompt
                // would induce, relative to where the mean already is.
                let n = self.n_done as f32;
                let mut diff_sq = 0.0f64;
                let mut mean_sq = 0.0f64;
                for (s, v) in sum.values.iter().zip(&j.values) {
                    let mean = s / n;
                    diff_sq += ((v - mean) as f64).powi(2);
                    mean_sq += (mean as f64).powi(2);
                }
                let denom = (n + 1.0) as f64 * mean_sq.sqrt().max(1e-12);
                mean_rel_change = mean_rel_change.max((diff_sq.sqrt() / denom) as f32);
            }
            for (s, v) in sum.values.iter_mut().zip(&j.values) {
                *s += v;
            }
        }

        self.n_done += 1;
        self.next_idx += 1;
        Ok(FitProgress {
            n_done: self.n_done,
            scaled_norm,
            mean_rel_change,
        })
    }

    /// Turn the running sum into the fitted lens.
    pub fn finish(self) -> Result<JacobianLens> {
        ensure!(
            self.n_done > 0,
            "no prompts were folded in; every one was skipped or the corpus was empty"
        );
        let n = self.n_done as f32;
        let jacobians = self
            .sums
            .into_iter()
            .map(|(l, mut j)| {
                j.scale(n);
                (l, j)
            })
            .collect();
        JacobianLens::new(jacobians, self.n_done, self.target_layer)
    }

    /// Snapshot the running **sum** so a killed fit resumes rather than restarts.
    ///
    /// Stored as f32, unlike the finished lens: a partial sum is rescaled by
    /// `1/n` at the end, and f16 rounding compounding across a thousand prompts
    /// is a real loss where it is negligible in the final artifact.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut out = Vec::new();
        out.extend_from_slice(b"RLXJFIT1");
        out.extend_from_slice(&(self.d_model as u64).to_le_bytes());
        out.extend_from_slice(&(self.target_layer as u64).to_le_bytes());
        out.extend_from_slice(&(self.n_done as u64).to_le_bytes());
        out.extend_from_slice(&(self.next_idx as u64).to_le_bytes());
        out.extend_from_slice(&(self.sums.len() as u64).to_le_bytes());
        for (layer, j) in &self.sums {
            out.extend_from_slice(&(*layer as u64).to_le_bytes());
            for v in &j.values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, &out).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        ensure!(
            b.len() >= 48 && &b[..8] == b"RLXJFIT1",
            "not an rlx-jlens checkpoint"
        );
        let u64_at = |o: usize| -> usize {
            u64::from_le_bytes(b[o..o + 8].try_into().expect("bounds checked")) as usize
        };
        let d_model = u64_at(8);
        let target_layer = u64_at(16);
        let n_done = u64_at(24);
        let next_idx = u64_at(32);
        let n_layers = u64_at(40);

        let mut sums = BTreeMap::new();
        let mut off = 48;
        let per = d_model * d_model;
        for _ in 0..n_layers {
            ensure!(off + 8 + per * 4 <= b.len(), "checkpoint is truncated");
            let layer = u64_at(off);
            off += 8;
            let values: Vec<f32> = b[off..off + per * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            off += per * 4;
            sums.insert(layer, Jacobian { values, d_model });
        }
        if sums.is_empty() {
            bail!("checkpoint holds no layers");
        }
        Ok(Self {
            sums,
            d_model,
            target_layer,
            n_done,
            next_idx,
        })
    }
}

fn frob(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt() as f32
}

// ── Corpus text ─────────────────────────────────────────────────────────────

/// Split `text` into paragraph-ish chunks of at least `min_chars`.
///
/// Blank-line separated, mirroring the reference's WikiText loader, which keeps
/// records of at least 600 characters. Short fragments make poor prompts: the
/// estimator drops leading sink positions and the final one, so a chunk has to
/// be long enough to leave positions worth averaging over.
pub fn paragraphs(text: &str, min_chars: usize) -> Vec<&str> {
    text.split("\n\n")
        .map(str::trim)
        .filter(|p| p.len() >= min_chars)
        .collect()
}

/// Tokenize chunks into fixed-length prompts.
///
/// Chunks tokenizing to fewer than `seq` tokens are dropped rather than padded —
/// padding would average the Jacobian over positions the model never really
/// saw. Longer ones are truncated. Returns token ids as `f32`, the form the
/// graphs take.
pub fn to_prompts(
    chunks: &[&str],
    seq: usize,
    max_prompts: usize,
    mut tokenize: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<Vec<Vec<f32>>> {
    ensure!(seq > 0, "seq must be non-zero");
    let mut out = Vec::new();
    for chunk in chunks {
        if out.len() >= max_prompts {
            break;
        }
        let ids = tokenize(chunk)?;
        if ids.len() < seq {
            continue;
        }
        out.push(ids[..seq].iter().map(|&t| t as f32).collect());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(d: usize, v: f32) -> Jacobian {
        let mut j = Jacobian::zeros(d);
        j.values.iter_mut().for_each(|x| *x = v);
        j
    }

    #[test]
    fn the_mean_is_the_mean() {
        let mut fit = CorpusFit::new(&[0, 3], 2, 5).unwrap();
        fit.observe(&[filled(2, 1.0), filled(2, 10.0)]).unwrap();
        fit.observe(&[filled(2, 3.0), filled(2, 20.0)]).unwrap();
        let lens = fit.finish().unwrap();
        assert_eq!(lens.n_prompts, 2);
        assert_eq!(lens.target_layer, 5);
        assert!(
            lens.get(0)
                .unwrap()
                .values
                .iter()
                .all(|v| (v - 2.0).abs() < 1e-6)
        );
        assert!(
            lens.get(3)
                .unwrap()
                .values
                .iter()
                .all(|v| (v - 15.0).abs() < 1e-6)
        );
    }

    /// The convergence signal: identical prompts must drive the relative shift
    /// to zero, and it should fall as more agreeing prompts arrive.
    #[test]
    fn relative_change_falls_as_the_estimate_settles() {
        let mut fit = CorpusFit::new(&[0], 4, 0).unwrap();
        let p = fit.observe(&[filled(4, 2.0)]).unwrap();
        assert!(
            p.mean_rel_change.is_nan(),
            "first prompt has no mean to move"
        );
        assert!((p.scaled_norm - frob(&filled(4, 2.0).values) / 2.0).abs() < 1e-5);

        let second = fit.observe(&[filled(4, 2.0)]).unwrap();
        assert!(
            second.mean_rel_change < 1e-6,
            "an identical prompt should not move the mean, got {}",
            second.mean_rel_change
        );

        // A different prompt moves it; a later identical one moves it less.
        let mut fit = CorpusFit::new(&[0], 4, 0).unwrap();
        fit.observe(&[filled(4, 1.0)]).unwrap();
        let early = fit.observe(&[filled(4, 3.0)]).unwrap().mean_rel_change;
        for _ in 0..8 {
            fit.observe(&[filled(4, 1.0)]).unwrap();
        }
        let late = fit.observe(&[filled(4, 3.0)]).unwrap().mean_rel_change;
        assert!(
            late < early,
            "shift should shrink with n: early {early}, late {late}"
        );
    }

    #[test]
    fn skipping_advances_without_counting() {
        let mut fit = CorpusFit::new(&[0], 2, 0).unwrap();
        fit.skip();
        fit.observe(&[filled(2, 1.0)]).unwrap();
        assert_eq!(fit.n_done(), 1, "a skipped prompt is not averaged in");
        assert_eq!(fit.next_idx(), 2, "but the corpus cursor still advanced");
    }

    #[test]
    fn a_checkpoint_resumes_where_it_stopped() {
        let path = std::env::temp_dir().join(format!("rlx_jlens_ckpt_{}.bin", std::process::id()));
        let mut fit = CorpusFit::new(&[1, 2], 3, 9).unwrap();
        fit.observe(&[filled(3, 1.0), filled(3, 2.0)]).unwrap();
        fit.skip();
        fit.save(&path).unwrap();

        let mut resumed = CorpusFit::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(resumed.n_done(), 1);
        assert_eq!(resumed.next_idx(), 2);
        assert_eq!(resumed.layers(), vec![1, 2]);

        // Continuing from the checkpoint must equal one uninterrupted run.
        resumed.observe(&[filled(3, 3.0), filled(3, 4.0)]).unwrap();
        let lens = resumed.finish().unwrap();
        assert!(
            lens.get(1)
                .unwrap()
                .values
                .iter()
                .all(|v| (v - 2.0).abs() < 1e-6)
        );
        assert!(
            lens.get(2)
                .unwrap()
                .values
                .iter()
                .all(|v| (v - 3.0).abs() < 1e-6)
        );
        assert_eq!(lens.n_prompts, 2);
    }

    #[test]
    fn finishing_an_empty_fit_is_an_error() {
        assert!(CorpusFit::new(&[0], 2, 0).unwrap().finish().is_err());
    }

    #[test]
    fn paragraphs_keeps_only_long_chunks() {
        let text = "short\n\nthis one is definitely long enough to keep around\n\ntiny";
        let got = paragraphs(text, 20);
        assert_eq!(
            got,
            vec!["this one is definitely long enough to keep around"]
        );
    }

    #[test]
    fn short_chunks_are_dropped_not_padded() {
        let chunks = vec!["aaaa", "bb"];
        let prompts = to_prompts(&chunks, 3, 10, |s| Ok((0..s.len() as u32).collect())).unwrap();
        // "aaaa" -> 4 tokens, truncated to 3; "bb" -> 2 tokens, too short.
        assert_eq!(prompts, vec![vec![0.0, 1.0, 2.0]]);
    }

    #[test]
    fn max_prompts_caps_the_corpus() {
        let chunks = vec!["aaaa", "bbbb", "cccc"];
        let prompts = to_prompts(&chunks, 2, 2, |s| Ok((0..s.len() as u32).collect())).unwrap();
        assert_eq!(prompts.len(), 2);
    }
}
