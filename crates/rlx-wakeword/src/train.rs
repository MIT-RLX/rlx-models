// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Multi-phrase training helpers: RLX CNN SGD → [`crate::WakewordBundle`].
//!
//! Prefer [`TrainBuilder`] for apps; CLI is `rlx-wakeword-train`.
//! After SGD, set [`TrainOpts::ternary`] / [`TrainBuilder::ternary`] for exact
//! `{−1,0,+1}` FC (or all) weights so bake can pack TQ2 and core can fuse add/sub.

use anyhow::{Context, Result, bail, ensure};
use rlx_wake::train::{
    CnnTrainConfig, LabeledClip, SgdConfig, load_pos_neg_dirs, synth_pos_neg_dataset, train_wake_cnn,
};
use rlx_wake::{TernaryOpts, TrainReport, WakeCnnConfig, WakeCnnWeights};
use std::path::{Path, PathBuf};

use crate::bundle::{WakewordBundle, host_to_core, save_bundle};
use crate::config::{DEFAULT_CONTEXT_MS, DEFAULT_HOP_SAMPLES, PhraseConfig, WakewordConfig};

#[derive(Debug, Clone)]
pub struct TrainOpts {
    pub epochs: usize,
    pub lr: f32,
    pub hop_samples: usize,
    pub context_ms: f32,
    pub threshold: f32,
    pub log_every: usize,
    /// After SGD, ternarize selected tensors to exact `{−1,0,+1}` for bake TQ2 / fused kernels.
    pub ternary: Option<TernaryOpts>,
}

impl Default for TrainOpts {
    fn default() -> Self {
        Self {
            epochs: 40,
            lr: 1e-2,
            hop_samples: DEFAULT_HOP_SAMPLES,
            context_ms: DEFAULT_CONTEXT_MS,
            threshold: 0.5,
            log_every: 5,
            ternary: None,
        }
    }
}

impl TrainOpts {
    pub fn epochs(mut self, n: usize) -> Self {
        self.epochs = n;
        self
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    pub fn hop_ms(mut self, hop_ms: u32) -> Result<Self> {
        self.hop_samples = validate_hop_ms(hop_ms)?;
        Ok(self)
    }

    pub fn context_ms(mut self, ms: f32) -> Self {
        self.context_ms = ms;
        self
    }

    pub fn with_ternary(mut self, opts: TernaryOpts) -> Self {
        self.ternary = Some(opts);
        self
    }
}

/// One phrase to train: `id` + positive/negative WAV dirs (or in-memory clips).
#[derive(Debug, Clone)]
pub struct PhraseTrainSpec {
    pub id: String,
    pub pos_dir: Option<PathBuf>,
    pub neg_dir: Option<PathBuf>,
    pub clips: Option<Vec<LabeledClip>>,
    pub threshold: Option<f32>,
}

impl PhraseTrainSpec {
    pub fn from_dirs(id: impl Into<String>, pos: impl Into<PathBuf>, neg: impl Into<PathBuf>) -> Self {
        Self {
            id: id.into(),
            pos_dir: Some(pos.into()),
            neg_dir: Some(neg.into()),
            clips: None,
            threshold: None,
        }
    }

    pub fn from_clips(id: impl Into<String>, clips: Vec<LabeledClip>) -> Self {
        Self {
            id: id.into(),
            pos_dir: None,
            neg_dir: None,
            clips: Some(clips),
            threshold: None,
        }
    }

    pub fn synth(id: impl Into<String>) -> Self {
        Self::synth_sized(id, 8, 8, 1.2)
    }

    pub fn synth_sized(
        id: impl Into<String>,
        n_pos: usize,
        n_neg: usize,
        seconds: f32,
    ) -> Self {
        Self {
            id: id.into(),
            pos_dir: None,
            neg_dir: None,
            clips: Some(synth_pos_neg_dataset(n_pos, n_neg, seconds)),
            threshold: None,
        }
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.threshold = Some(t);
        self
    }
}

/// Fluent multi-phrase trainer.
///
/// ```ignore
/// let bundle = TrainBuilder::new()
///     .epochs(20)
///     .synth_n(4)?
///     .out_dir("/tmp/wake")
///     .run()?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct TrainBuilder {
    opts: TrainOpts,
    specs: Vec<PhraseTrainSpec>,
    out_dir: Option<PathBuf>,
}

impl TrainBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn opts(mut self, opts: TrainOpts) -> Self {
        self.opts = opts;
        self
    }

    pub fn epochs(mut self, n: usize) -> Self {
        self.opts.epochs = n;
        self
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.opts.lr = lr;
        self
    }

    pub fn threshold(mut self, t: f32) -> Self {
        self.opts.threshold = t;
        self
    }

    pub fn hop_ms(mut self, hop_ms: u32) -> Result<Self> {
        self.opts.hop_samples = validate_hop_ms(hop_ms)?;
        Ok(self)
    }

    pub fn phrase(mut self, spec: PhraseTrainSpec) -> Self {
        self.specs.push(spec);
        self
    }

    pub fn phrase_dirs(
        mut self,
        id: impl Into<String>,
        pos: impl Into<PathBuf>,
        neg: impl Into<PathBuf>,
    ) -> Self {
        self.specs.push(PhraseTrainSpec::from_dirs(id, pos, neg));
        self
    }

    pub fn synth(mut self, id: impl Into<String>) -> Self {
        self.specs.push(PhraseTrainSpec::synth(id));
        self
    }

    pub fn synth_n(mut self, n: usize) -> Result<Self> {
        ensure!((1..=32).contains(&n), "synth-n must be 1..=32");
        for i in 0..n {
            self.specs.push(PhraseTrainSpec::synth(format!("word{i}")));
        }
        Ok(self)
    }

    pub fn phrases_dir(mut self, dir: impl AsRef<Path>) -> Result<Self> {
        self.specs.extend(specs_from_phrases_dir(dir.as_ref())?);
        Ok(self)
    }

    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    pub fn ternary(mut self, opts: TernaryOpts) -> Self {
        self.opts.ternary = Some(opts);
        self
    }

    pub fn run(self) -> Result<WakewordBundle> {
        ensure!(!self.specs.is_empty(), "add at least one phrase / synth_n / phrases_dir");
        train_phrases(&self.specs, &self.opts, self.out_dir.as_deref())
    }
}

/// Parse `id=pos_dir:neg_dir` or bare `id` (synth when `--synth`).
pub fn parse_phrase_arg(s: &str) -> Result<(String, Option<PathBuf>, Option<PathBuf>)> {
    let s = s.trim();
    ensure!(!s.is_empty(), "empty --phrase");
    if let Some((id, rest)) = s.split_once('=') {
        let (pos, neg) = rest
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("--phrase ID=POS:NEG (got {s})"))?;
        Ok((id.to_string(), Some(PathBuf::from(pos)), Some(PathBuf::from(neg))))
    } else {
        Ok((s.to_string(), None, None))
    }
}

/// Discover `phrases_dir/<id>/{positives,negatives}/` (or `pos`/`neg`).
pub fn specs_from_phrases_dir(dir: &Path) -> Result<Vec<PhraseTrainSpec>> {
    let mut out = Vec::new();
    for ent in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let ent = ent?;
        if !ent.file_type()?.is_dir() {
            continue;
        }
        let id = ent.file_name().to_string_lossy().into_owned();
        let base = ent.path();
        let pos = ["positives", "pos", "positive"]
            .iter()
            .map(|n| base.join(n))
            .find(|p| p.is_dir());
        let neg = ["negatives", "neg", "negative"]
            .iter()
            .map(|n| base.join(n))
            .find(|p| p.is_dir());
        let (Some(pos), Some(neg)) = (pos, neg) else {
            continue;
        };
        out.push(PhraseTrainSpec::from_dirs(id, pos, neg));
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    ensure!(!out.is_empty(), "no phrase dirs under {}", dir.display());
    Ok(out)
}

pub fn train_one_phrase(spec: &PhraseTrainSpec, opts: &TrainOpts) -> Result<(WakeCnnWeights, TrainReport)> {
    let clips = if let Some(c) = &spec.clips {
        c.clone()
    } else {
        let pos = spec
            .pos_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("phrase {} missing positives", spec.id))?;
        let neg = spec
            .neg_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("phrase {} missing negatives", spec.id))?;
        load_pos_neg_dirs(pos, neg)?
    };
    ensure!(!clips.is_empty(), "empty dataset for {}", spec.id);
    let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
    let cfg = CnnTrainConfig {
        keyword: spec.id.clone(),
        sgd: SgdConfig {
            lr: opts.lr,
            epochs: opts.epochs,
            log_every: opts.log_every,
            weight_decay: 1e-4,
        },
        ..CnnTrainConfig::default()
    };
    let report = train_wake_cnn(&mut w, &clips, &cfg);
    if let Some(topts) = opts.ternary {
        let stats = w.ternarize(topts);
        eprintln!(
            "  ternary tensors={} elems={} nz={} ({:.0}% sparse)",
            stats.tensors,
            stats.elems,
            stats.nonzero,
            100.0 * (1.0 - stats.nonzero as f32 / stats.elems.max(1) as f32)
        );
    }
    Ok((w, report))
}

/// Train every phrase and return a loadable bundle (also writes `out_dir` if set).
pub fn train_phrases(
    specs: &[PhraseTrainSpec],
    opts: &TrainOpts,
    out_dir: Option<&Path>,
) -> Result<WakewordBundle> {
    ensure!(!specs.is_empty(), "need at least one phrase");
    let mut trained: Vec<(String, f32, WakeCnnWeights)> = Vec::new();
    for spec in specs {
        eprintln!("[rlx-wakeword-train] phrase={} …", spec.id);
        let (w, report) = train_one_phrase(spec, opts)?;
        let thr = spec.threshold.unwrap_or(opts.threshold);
        eprintln!(
            "  loss {:.4}->{:.4}  acc={:.1}%",
            report.initial_loss,
            report.final_loss,
            report.train_acc * 100.0
        );
        trained.push((spec.id.clone(), thr, w));
    }
    if let Some(dir) = out_dir {
        save_bundle(dir, opts.hop_samples, opts.context_ms, &trained)?;
    }
    let weights: Vec<_> = trained
        .iter()
        .map(|(id, _, w)| (id.clone(), host_to_core(w)))
        .collect();
    let phrases: Vec<_> = trained
        .iter()
        .map(|(id, thr, _)| PhraseConfig::new(id, *thr))
        .collect();
    let config = WakewordConfig {
        hop_samples: opts.hop_samples,
        context_ms: opts.context_ms,
        phrases,
        vad_gate: false,
        speaker_id: false,
        ..WakewordConfig::default()
    };
    Ok(WakewordBundle { config, weights })
}

/// Quick multi-word synth: `word0` … `word{n-1}`.
pub fn train_synth_n(n: usize, opts: &TrainOpts, out_dir: Option<&Path>) -> Result<WakewordBundle> {
    ensure!((1..=32).contains(&n), "synth-n must be 1..=32");
    let specs: Vec<_> = (0..n)
        .map(|i| PhraseTrainSpec::synth(format!("word{i}")))
        .collect();
    train_phrases(&specs, opts, out_dir)
}

pub fn merge_phrase_into_dir(
    out_dir: &Path,
    id: &str,
    threshold: f32,
    weights: &WakeCnnWeights,
    hop_samples: usize,
    context_ms: f32,
) -> Result<()> {
    let mut rows = if out_dir.join("manifest.json").is_file() {
        let man: crate::bundle::BundleManifest =
            serde_json::from_str(&std::fs::read_to_string(out_dir.join("manifest.json"))?)?;
        let mut rows = Vec::new();
        for p in man.phrases {
            if p.id == id {
                continue;
            }
            let w = rlx_wake::WakeCnnWeights::load(&out_dir.join(&p.weights))?;
            rows.push((p.id, p.threshold, w));
        }
        rows
    } else {
        Vec::new()
    };
    rows.push((id.to_string(), threshold, weights.clone()));
    save_bundle(out_dir, hop_samples, context_ms, &rows)?;
    Ok(())
}

pub fn validate_hop_ms(hop_ms: u32) -> Result<usize> {
    match hop_ms {
        20 | 32 | 40 | 80 => Ok(crate::config::hop_ms_to_samples(hop_ms)),
        _ => bail!("hop-ms must be one of 20, 32, 40, 80"),
    }
}
