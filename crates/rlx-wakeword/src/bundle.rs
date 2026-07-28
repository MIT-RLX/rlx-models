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
    pub phrases: Vec<BundlePhrase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundlePhrase {
    pub id: String,
    pub threshold: f32,
    pub weights: String,
}

pub struct WakewordBundle {
    pub config: WakewordConfig,
    pub weights: Vec<(String, WakeCnnWeights)>,
}

impl WakewordBundle {
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let man_path = dir.join("manifest.json");
        let text = fs::read_to_string(&man_path)
            .with_context(|| format!("read {}", man_path.display()))?;
        let man: BundleManifest = serde_json::from_str(&text)?;
        let mut weights = Vec::new();
        let mut phrases = Vec::new();
        for p in &man.phrases {
            let path = dir.join(&p.weights);
            let host = HostWeights::load(&path)
                .with_context(|| format!("load phrase weights {}", path.display()))?;
            weights.push((p.id.clone(), host_to_core(&host)));
            phrases.push(PhraseConfig::new(&p.id, p.threshold));
        }
        let config = WakewordConfig {
            hop_samples: hop_ms_to_samples(man.hop_ms),
            context_ms: man.context_ms,
            phrases,
            ..WakewordConfig::default()
        };
        Ok(Self { config, weights })
    }

    pub fn into_session(self) -> Result<crate::session::WakewordSession> {
        crate::session::WakewordSession::new(self.config, self.weights)
    }

    pub fn open_session(&self) -> Result<crate::session::WakewordSession> {
        crate::session::WakewordSession::new(self.config.clone(), self.weights.clone())
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
    WakewordBundle { config, weights }
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
