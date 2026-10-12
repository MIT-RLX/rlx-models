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

//! Scoring a checkpoint the way kev reports it, so the numbers are comparable.
//!
//! Definitions follow `kev/metrics.py` exactly, because they are not all the
//! obvious choice:
//!
//! * **Brier** is the multi-class *sum* `Σ(p − onehot)²`, not a mean, so a
//!   two-option question maxes out at 2.0.
//! * **ECE** uses 10 equal-width confidence bins and weights each by its
//!   share of the data; the last bin is closed on the right.
//! * **Confident error rate** counts `max p ≥ 0.9` *and* wrong — the number
//!   that matters for automating a decision, and the one calibration is meant
//!   to move without touching accuracy.
//!
//! Everything is per **question**, not per record, and averaged.

use anyhow::Result;

/// One scored question.
#[derive(Debug, Clone)]
pub struct Row {
    /// Predicted distribution over the question's options.
    pub p: Vec<f32>,
    pub label: usize,
    pub source: String,
}

impl Row {
    fn argmax(&self) -> usize {
        crate::api::argmax(&self.p)
    }
    fn conf(&self) -> f32 {
        self.p.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
    fn correct(&self) -> bool {
        self.argmax() == self.label
    }
    fn brier(&self) -> f64 {
        self.p
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let t = f64::from(u8::from(i == self.label));
                (*x as f64 - t).powi(2)
            })
            .sum()
    }
    fn nll(&self) -> f64 {
        const EPS: f64 = 1e-12;
        -(self.p.get(self.label).copied().unwrap_or(0.0) as f64)
            .max(EPS)
            .ln()
    }
}

/// Aggregate metrics over a set of questions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metrics {
    pub n: usize,
    pub accuracy: f64,
    pub nll: f64,
    pub brier: f64,
    pub ece: f64,
    pub mean_confidence: f64,
    pub confident_error_rate: f64,
    pub coverage_at_0_9: f64,
    /// Accuracy a uniform guesser gets on this exact set of questions:
    /// `mean(1 / n_options)`. Not a constant — `decision-v7` mixes 2-option
    /// yes/no with banking77's 77 intents, so the floor moves with the sample.
    pub chance_accuracy: f64,
    /// NLL a uniform guesser gets: `mean(ln n_options)`.
    ///
    /// Reported because a loss is only readable against it. A run whose loss
    /// settles on `ln(n_options)` has learned nothing, and at 2 options that is
    /// 0.693 — which reads like a good number if you are comparing against zero.
    /// Mistaking it for one cost a 1529-step reproduction here.
    pub chance_nll: f64,
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} acc={:.4} (chance {:.4}) brier={:.4} ece={:.4} \
             nll={:.4} (chance {:.4}) mean_conf={:.4} confident_err={:.4}{}",
            self.n,
            self.accuracy,
            self.chance_accuracy,
            self.brier,
            self.ece,
            self.nll,
            self.chance_nll,
            self.mean_confidence,
            self.confident_error_rate,
            if self.at_chance() {
                "  <== AT CHANCE"
            } else {
                ""
            }
        )
    }
}

impl Metrics {
    /// True when this is indistinguishable from guessing: accuracy no better
    /// than the uniform floor, or an NLL no better than `ln(n_options)`.
    ///
    /// Worth a flag rather than a mental calculation. Every failure mode this
    /// crate has hit — a broken backend gradient, a diverging learning rate —
    /// presents as "plausible loss, chance accuracy", and the two numbers that
    /// reveal it are not the ones a reader looks at first.
    pub fn at_chance(&self) -> bool {
        self.accuracy <= self.chance_accuracy || self.nll >= self.chance_nll
    }
}

/// Expected calibration error over `bins` equal-width confidence bins.
pub fn ece(conf: &[f32], correct: &[bool], bins: usize) -> f64 {
    assert_eq!(conf.len(), correct.len());
    if conf.is_empty() {
        return 0.0;
    }
    let n = conf.len() as f64;
    let mut total = 0.0;
    for b in 0..bins {
        let lo = b as f64 / bins as f64;
        let hi = (b + 1) as f64 / bins as f64;
        let last = b + 1 == bins;
        let idx: Vec<usize> = (0..conf.len())
            .filter(|&i| {
                let c = conf[i] as f64;
                c >= lo && (if last { c <= hi } else { c < hi })
            })
            .collect();
        if idx.is_empty() {
            continue;
        }
        let m = idx.len() as f64;
        let acc: f64 = idx
            .iter()
            .map(|&i| f64::from(u8::from(correct[i])))
            .sum::<f64>()
            / m;
        let cf: f64 = idx.iter().map(|&i| conf[i] as f64).sum::<f64>() / m;
        total += (m / n) * (acc - cf).abs();
    }
    total
}

/// Score a set of questions.
pub fn metrics(rows: &[Row]) -> Metrics {
    if rows.is_empty() {
        return Metrics::default();
    }
    let n = rows.len();
    let nf = n as f64;
    let conf: Vec<f32> = rows.iter().map(Row::conf).collect();
    let correct: Vec<bool> = rows.iter().map(Row::correct).collect();
    Metrics {
        n,
        accuracy: correct.iter().filter(|c| **c).count() as f64 / nf,
        nll: rows.iter().map(Row::nll).sum::<f64>() / nf,
        brier: rows.iter().map(Row::brier).sum::<f64>() / nf,
        ece: ece(&conf, &correct, 10),
        mean_confidence: conf.iter().map(|c| *c as f64).sum::<f64>() / nf,
        confident_error_rate: rows
            .iter()
            .zip(&conf)
            .filter(|(r, c)| **c >= 0.9 && !r.correct())
            .count() as f64
            / nf,
        coverage_at_0_9: conf.iter().filter(|c| **c >= 0.9).count() as f64 / nf,
        // Per question, since the option count varies across the suite.
        chance_accuracy: rows
            .iter()
            .map(|r| 1.0 / r.p.len().max(1) as f64)
            .sum::<f64>()
            / nf,
        chance_nll: rows
            .iter()
            .map(|r| (r.p.len().max(1) as f64).ln())
            .sum::<f64>()
            / nf,
    }
}

/// Binomial standard error of the accuracy — the thing that decides whether a
/// difference against a published number means anything.
pub fn accuracy_se(m: &Metrics) -> f64 {
    if m.n == 0 {
        return 0.0;
    }
    (m.accuracy * (1.0 - m.accuracy) / m.n as f64).sqrt()
}

/// Metrics broken out per source, plus the overall figure.
pub fn by_source(rows: &[Row]) -> Vec<(String, Metrics)> {
    let mut sources: Vec<String> = rows.iter().map(|r| r.source.clone()).collect();
    sources.sort();
    sources.dedup();
    let mut out: Vec<(String, Metrics)> = sources
        .into_iter()
        .map(|s| {
            let sub: Vec<Row> = rows.iter().filter(|r| r.source == s).cloned().collect();
            (s, metrics(&sub))
        })
        .collect();
    out.push(("ALL".into(), metrics(rows)));
    out
}

/// Evenly-spaced indices covering `len`, so a subsample spans every source.
///
/// Suite splits are grouped by source, so `records[..n]` is not a sample of
/// the split — it is a sample of whichever one or two sources come first. That
/// makes a truncated run look like a much narrower evaluation than it is.
pub fn strided_indices(len: usize, take: usize) -> Vec<usize> {
    if take == 0 || len == 0 {
        return Vec::new();
    }
    if take >= len {
        return (0..len).collect();
    }
    (0..take).map(|i| i * len / take).collect()
}

/// Rescale a distribution by a temperature, without re-running the model.
///
/// `p` came out of a softmax, so `ln p` is the logit vector up to an additive
/// constant — and that constant cancels in the softmax, so rescaling `ln p` by
/// `T` gives exactly `softmax(z / T)` for the original logits `z`. That is what
/// makes post-hoc calibration a pure host-side pass over scored rows.
pub fn rescale(p: &[f32], t: f32) -> Vec<f32> {
    let z: Vec<f64> = p
        .iter()
        .map(|x| (*x as f64).max(1e-30).ln() / t as f64)
        .collect();
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| (v / s) as f32).collect()
}

/// Rows rescaled by `t`, labels and sources preserved.
pub fn rescaled(rows: &[Row], t: f32) -> Vec<Row> {
    rows.iter()
        .map(|r| Row {
            p: rescale(&r.p, t),
            label: r.label,
            source: r.source.clone(),
        })
        .collect()
}

/// Fit the calibration temperature by minimising NLL on `rows` — the
/// temperature-scaling of Guo et al., and what kev's `temperature` field holds.
///
/// Optimised over the **inverse** temperature `b = 1/T`, where the objective is
/// convex (log-sum-exp is convex in `b`, the label term is linear), so a golden
/// section search on a bracketing interval is exact rather than a heuristic.
/// The interval is `T ∈ [0.05, 20]`, wide enough to contain any useful value and
/// to make a run that pins at an endpoint visibly suspicious.
///
/// This **cannot change accuracy**: a positive temperature is a monotone
/// transform of each logit vector, so the argmax is fixed. It moves NLL, Brier,
/// ECE and the confident-error rate, which is the whole point.
pub fn fit_temperature(rows: &[Row]) -> f32 {
    const T_LO: f64 = 0.05;
    const T_HI: f64 = 20.0;
    if rows.is_empty() {
        return 1.0;
    }
    let nll_at = |t: f64| -> f64 {
        let mut acc = 0.0;
        for r in rows {
            let z: Vec<f64> =
                r.p.iter()
                    .map(|x| (*x as f64).max(1e-30).ln() / t)
                    .collect();
            let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = m + z.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
            acc += lse - z[r.label];
        }
        acc / rows.len() as f64
    };
    // Golden section on b = 1/T, the convex parameterisation.
    let (mut lo, mut hi) = (1.0 / T_HI, 1.0 / T_LO);
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = hi - phi * (hi - lo);
    let mut d = lo + phi * (hi - lo);
    let (mut fc, mut fd) = (nll_at(1.0 / c), nll_at(1.0 / d));
    // 60 iterations shrinks the bracket by phi^60 — far below f32 resolution.
    for _ in 0..60 {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - phi * (hi - lo);
            fc = nll_at(1.0 / c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + phi * (hi - lo);
            fd = nll_at(1.0 / d);
        }
    }
    let b = 0.5 * (lo + hi);
    (1.0 / b) as f32
}

/// Score a checkpoint over labelled records, one forward per record.
///
/// `limit` subsamples with [`strided_indices`] rather than truncating.
#[cfg(feature = "tokenizer")]
pub fn score<T: crate::encode::TokenizerLike>(
    model: &mut crate::model::DecisionModel<T>,
    records: &[crate::recipe::LabelledRecord],
    limit: Option<usize>,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<Row>> {
    let take = limit.unwrap_or(records.len()).min(records.len());
    let idx = strided_indices(records.len(), take);

    // Score in SHAPE order, not suite order. The model compiles one graph per
    // (rows, seq) bucket, and a suite is ordered by source, so consecutive
    // records ask for unrelated shapes and it recompiles on almost every one —
    // which is why scoring 200 records took longer than the forward passes
    // warrant. Metrics are order-independent (`by_source` groups by name), so
    // this is free. A record whose shape cannot be resolved keeps its place;
    // `probs` will raise the real error below.
    let mut ordered: Vec<(usize, usize, usize)> = idx
        .iter()
        .map(|&i| {
            let hint = crate::recipe::materialize(&records[i])
                .ok()
                .and_then(|m| model.shape_hint(&m.record).ok())
                .unwrap_or((usize::MAX, usize::MAX));
            (hint.0, hint.1, i)
        })
        .collect();
    ordered.sort_unstable();
    let idx: Vec<usize> = ordered.into_iter().map(|(_, _, i)| i).collect();

    let mut rows = Vec::new();
    for (i, rec) in idx.iter().map(|&i| &records[i]).enumerate() {
        let m = crate::recipe::materialize(rec)?;
        let probs = model.probs(&m.record)?;
        for (qi, p) in probs.into_iter().enumerate() {
            rows.push(Row {
                p,
                label: m.record.questions[qi].label,
                source: rec.questions[qi].src.clone(),
            });
        }
        progress(i + 1, take);
    }
    Ok(rows)
}
