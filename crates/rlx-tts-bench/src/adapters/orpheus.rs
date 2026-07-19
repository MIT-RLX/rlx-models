use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_orpheus::OrpheusTts;
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "orpheus",
        supports_clone: false,
        feature: "lm-tts",
        hints: WeightHints {
            default_dir: PathBuf::from("weights/tts/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf"),
            env_keys: vec!["ORPHEUS_GGUF_PATH", "ORPHEUS_PRETRAINED_GGUF"],
            marker_files: vec![],
        },
    }
}

fn resolve_gguf() -> Option<PathBuf> {
    for key in ["ORPHEUS_PRETRAINED_GGUF", "ORPHEUS_GGUF_PATH"] {
        if let Ok(p) = std::env::var(key) {
            let p = PathBuf::from(p);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    let candidates = [
        PathBuf::from("weights/tts/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf"),
        PathBuf::from("/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

fn resolve_snac() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ORPHEUS_SNAC_PATH") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let candidates = [
        PathBuf::from("weights/tts/snac_24khz/snac_24khz_decoder.safetensors"),
        PathBuf::from("weights/tts/snac/snac_24khz_decoder.safetensors"),
        PathBuf::from("/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let gguf = resolve_gguf().ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let snac = resolve_snac().ok_or_else(|| anyhow::anyhow!("missing ORPHEUS_SNAC_PATH"))?;
    let inner = OrpheusTts::load_on(&gguf, &snac, device).context("load orpheus")?;
    Ok(Box::new(OrpheusAdapter { inner }))
}

struct OrpheusAdapter {
    inner: OrpheusTts,
}

impl TtsAdapter for OrpheusAdapter {
    fn id(&self) -> &'static str {
        "orpheus"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let voice = std::env::var("ORPHEUS_VOICE").ok();
        let t0 = Instant::now();
        let result = self.inner.synthesize(req.text, voice.as_deref())?;
        Ok(SynthResult {
            pcm: result.samples,
            sample_rate: 24_000,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
