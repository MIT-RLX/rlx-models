// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Bundle load/save: `manifest.json` + `phrase_*.safetensors` (+ optional `.rlxw` pack).

use anyhow::{Context, Result, bail};
use rlx_wake::WakeCnnWeights as HostWeights;
use rlx_wakeword_core::{PackHeader, WakeCnnConfig, WakeCnnWeights};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{PhraseConfig, WakewordConfig, hop_ms_to_samples, samples_to_hop_ms};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    pub hop_ms: u32,
    pub context_ms: f32,
    /// [`WakewordConfig::min_hops`]; 1 when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_hops: Option<usize>,
    /// [`WakewordConfig::smooth_hops`]; 1 when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth_hops: Option<usize>,
    /// [`WakewordConfig::score_every_hops`]; 1 when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_every_hops: Option<usize>,
    /// A second network asked only when a phrase fires, whose score must agree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verifier: Option<BundleVerifier>,
    pub phrases: Vec<BundlePhrase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleVerifier {
    pub weights: String,
    /// The verifier score a phrase needs, unless the phrase says (`verify_threshold`).
    pub threshold: f32,
    /// Its window, if not the bundle's `context_ms`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_ms: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundlePhrase {
    pub id: String,
    pub threshold: f32,
    pub weights: String,
    /// Which output of `weights` scores this phrase (a multi-phrase network); 0 when absent.
    /// Phrases naming the same `weights` share one network, run once per hop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<usize>,
    /// This phrase's window if not the bundle's `context_ms`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_ms: Option<f32>,
    /// This phrase's confirmation hops if not the bundle's `min_hops`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_hops: Option<usize>,
    /// This phrase's score smoothing if not the bundle's `smooth_hops`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth_hops: Option<usize>,
    /// With a verifier: its score this phrase needs, and its output scoring the phrase
    /// (default: `output`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify_threshold: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify_output: Option<usize>,
}

pub struct WakewordBundle {
    pub config: WakewordConfig,
    pub weights: Vec<(String, WakeCnnWeights)>,
    /// Per phrase (same order as `weights`): its weights file and output index, which say
    /// which phrases share a network.
    pub sources: Vec<(String, usize)>,
    /// The verifier: weights, default threshold, window (ms).
    pub verifier: Option<(WakeCnnWeights, f32, f32)>,
}

impl WakewordBundle {
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let man_path = dir.join("manifest.json");
        let text = fs::read_to_string(&man_path)
            .with_context(|| format!("read {}", man_path.display()))?;
        let man: BundleManifest = serde_json::from_str(&text)?;
        let mut weights = Vec::new();
        let mut phrases = Vec::new();
        let mut sources = Vec::new();
        let mut loaded: std::collections::HashMap<String, WakeCnnWeights> =
            std::collections::HashMap::new();
        for p in &man.phrases {
            let w = match loaded.get(&p.weights) {
                Some(w) => w.clone(),
                None => {
                    let path = dir.join(&p.weights);
                    let host = HostWeights::load(&path)
                        .with_context(|| format!("load phrase weights {}", path.display()))?;
                    let w = host_to_core(&host);
                    loaded.insert(p.weights.clone(), w.clone());
                    w
                }
            };
            weights.push((p.id.clone(), w));
            sources.push((p.weights.clone(), p.output.unwrap_or(0)));
            let mut pc = PhraseConfig::new(&p.id, p.threshold);
            pc.context_ms = p.context_ms;
            pc.min_hops = p.min_hops;
            pc.smooth_hops = p.smooth_hops;
            pc.verify_threshold = p.verify_threshold;
            pc.verify_output = p.verify_output;
            phrases.push(pc);
        }
        let verifier = match &man.verifier {
            Some(v) => {
                let path = dir.join(&v.weights);
                let host = HostWeights::load(&path)
                    .with_context(|| format!("load verifier {}", path.display()))?;
                Some((
                    host_to_core(&host),
                    v.threshold,
                    v.context_ms.unwrap_or(man.context_ms),
                ))
            }
            None => None,
        };
        let config = WakewordConfig {
            hop_samples: hop_ms_to_samples(man.hop_ms),
            context_ms: man.context_ms,
            min_hops: man.min_hops.unwrap_or(1).max(1),
            smooth_hops: man.smooth_hops.unwrap_or(1).max(1),
            score_every_hops: man.score_every_hops.unwrap_or(1).max(1),
            phrases,
            ..WakewordConfig::default()
        };
        Ok(Self {
            config,
            weights,
            sources,
            verifier,
        })
    }

    /// Networks with the phrases they score, `sources` grouped by weights file.
    fn nets(&self) -> Vec<(WakeCnnWeights, Vec<(String, usize)>)> {
        let mut nets: Vec<(String, WakeCnnWeights, Vec<(String, usize)>)> = Vec::new();
        for ((id, w), (file, output)) in self.weights.iter().zip(&self.sources) {
            match nets.iter_mut().find(|n| &n.0 == file) {
                Some(n) => n.2.push((id.clone(), *output)),
                None => nets.push((file.clone(), w.clone(), vec![(id.clone(), *output)])),
            }
        }
        nets.into_iter().map(|(_, w, p)| (w, p)).collect()
    }

    pub fn into_session(self) -> Result<crate::session::WakewordSession> {
        self.open_session()
    }

    pub fn open_session(&self) -> Result<crate::session::WakewordSession> {
        let s = crate::session::WakewordSession::new_shared(self.config.clone(), self.nets())?;
        Ok(match &self.verifier {
            Some((w, thr, ms)) => s.with_verifier(w.clone(), *thr, *ms),
            None => s,
        })
    }
}

pub fn save_bundle(
    dir: &Path,
    hop_samples: usize,
    context_ms: f32,
    phrases: &[(String, f32, HostWeights)],
) -> Result<()> {
    fs::create_dir_all(dir)?;
    let mut man = BundleManifest {
        hop_ms: samples_to_hop_ms(hop_samples),
        context_ms,
        min_hops: None,
        smooth_hops: None,
        score_every_hops: None,
        verifier: None,
        phrases: Vec::new(),
    };
    for (id, thr, w) in phrases {
        let fname = format!("phrase_{id}.safetensors");
        let path = dir.join(&fname);
        w.save(&path)?;
        man.phrases.push(BundlePhrase {
            id: id.clone(),
            threshold: *thr,
            weights: fname,
            output: None,
            context_ms: None,
            min_hops: None,
            smooth_hops: None,
            verify_threshold: None,
            verify_output: None,
        });
    }
    let man_path = dir.join("manifest.json");
    fs::write(&man_path, serde_json::to_string_pretty(&man)? + "\n")?;
    Ok(())
}

/// Pack directory into a flat `.rlxw` (f32 payload = concatenated safetensors bytes).
pub fn pack_rlxw(dir: &Path, out: &Path) -> Result<()> {
    let bundle = WakewordBundle::load_dir(dir)?;
    let mut payload = Vec::new();
    for (id, _) in &bundle.weights {
        let path = dir.join(format!("phrase_{id}.safetensors"));
        let bytes = fs::read(&path)?;
        let len = bytes.len() as u32;
        payload.extend_from_slice(&len.to_le_bytes());
        let idb = id.as_bytes();
        payload.push(idb.len() as u8);
        payload.extend_from_slice(idb);
        payload.extend_from_slice(&bytes);
    }
    let header = PackHeader::new_f32(
        bundle.weights.len() as u32,
        bundle.config.hop_samples as u32,
        payload.len() as u32,
    );
    let mut out_bytes = vec![0u8; PackHeader::BYTES];
    header.write_to(&mut out_bytes);
    out_bytes.extend_from_slice(&payload);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, out_bytes)?;
    Ok(())
}

pub fn host_to_core(host: &HostWeights) -> WakeCnnWeights {
    WakeCnnWeights::from_parts(
        WakeCnnConfig {
            n_mels: host.cfg.n_mels,
            c1: host.cfg.c1,
            c2: host.cfg.c2,
            c3: host.cfg.c3,
            k: host.cfg.k,
            hidden: host.cfg.hidden,
            pool_chunks: host.cfg.pool_chunks,
            res_blocks: host.cfg.res_blocks,
            attn: host.cfg.attn,
            pcen: host.cfg.pcen,
            dual: host.cfg.dual,
            causal: host.cfg.causal,
        },
        host.conv1_w.clone(),
        host.conv1_b.clone(),
        host.conv2_w.clone(),
        host.conv2_b.clone(),
        host.conv3_w.clone(),
        host.conv3_b.clone(),
        host.fc1_w.clone(),
        host.fc1_b.clone(),
        host.fc2_w.clone(),
        host.fc2_b.clone(),
    )
    .with_extras(
        host.res_w.clone(),
        host.res_b.clone(),
        host.attn_w.clone(),
        host.attn_b.clone(),
    )
}

pub fn stub_bundle(phrase_id: &str, hop_ms: u32) -> WakewordBundle {
    stub_bundle_n(1, hop_ms, |i| {
        if i == 0 {
            phrase_id.to_string()
        } else {
            format!("{phrase_id}{i}")
        }
    })
}

/// Stub bundle with `n` lite CNN phrase heads (`word0`… or custom ids via `id_fn`).
pub fn stub_bundle_n(
    n: usize,
    hop_ms: u32,
    mut id_fn: impl FnMut(usize) -> String,
) -> WakewordBundle {
    let n = n.max(1);
    let mut phrases = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        let id = id_fn(i);
        phrases.push(PhraseConfig::new(&id, 0.5));
        weights.push((id, WakeCnnWeights::stub(WakeCnnConfig::lite())));
    }
    let config = WakewordConfig {
        hop_samples: hop_ms_to_samples(hop_ms),
        phrases,
        vad_gate: false,
        speaker_id: false,
        ..WakewordConfig::default()
    };
    let sources = weights
        .iter()
        .map(|(id, _)| (format!("phrase_{id}.safetensors"), 0))
        .collect();
    WakewordBundle {
        config,
        weights,
        sources,
        verifier: None,
    }
}

pub fn default_stub_path() -> PathBuf {
    PathBuf::from("crates/rlx-wakeword/weights")
}

pub fn validate_hop_ms(hop_ms: u32) -> Result<usize> {
    match hop_ms {
        20 | 32 | 40 | 80 => Ok(hop_ms_to_samples(hop_ms)),
        _ => bail!("hop-ms must be one of 20, 32, 40, 80 (got {hop_ms})"),
    }
}
