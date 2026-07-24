use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_runtime::Device;
use rlx_soprano::native_qwen3::SopranoQwen3;
use rlx_soprano::{DEFAULT_LOCAL_DIR, InferOpts, NativeSoprano, SAMPLE_RATE};

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "soprano",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_SOPRANO_DIR"],
            marker_files: vec![
                "soprano.rlxp",
                "soprano.rlx",
                "soprano.gguf",
                "tokenizer.json",
            ],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = NativeSoprano::open(&dir, device).context("load soprano")?;
    // Prefer the native (ort-free) rlx-qwen3 backbone when the stock
    // Qwen3ForCausalLM checkpoint (ekwek/Soprano-1.1-80M) is available — the
    // onnx-imported backbone diverges on CUDA (wrong output) and caps seq at 128.
    let bb = std::env::var("RLX_SOPRANO_BACKBONE_ST")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let p = dir.join("backbone.safetensors");
            p.is_file().then_some(p)
        });
    let native_bb = bb.and_then(|p| match SopranoQwen3::open(&p, device) {
        Ok(bb) => {
            eprintln!("[soprano] native rlx-qwen3 backbone: {}", p.display());
            Some(bb)
        }
        Err(e) => {
            eprintln!("[soprano] native backbone unavailable ({e}); using onnx backbone");
            None
        }
    });
    Ok(Box::new(SopranoAdapter { inner, native_bb }))
}

struct SopranoAdapter {
    inner: NativeSoprano,
    native_bb: Option<SopranoQwen3>,
}

impl TtsAdapter for SopranoAdapter {
    fn id(&self) -> &'static str {
        "soprano"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let t0 = Instant::now();
        let pcm = if let Some(bb) = &self.native_bb {
            // Native rlx-qwen3 backbone → 512-d latents → ONNX Vocos decoder.
            let ids = self.inner.encode_prompt(req.text)?;
            let max_new = std::env::var("SOPRANO_MAX_NEW")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(512);
            let (latents, _) = bb.generate_latents_greedy(&ids, max_new)?;
            anyhow::ensure!(!latents.is_empty(), "soprano: no latents produced");
            self.inner.decode_latents(&latents, true)?
        } else {
            let mut opts = InferOpts::default();
            opts.seed = req.seed;
            self.inner.synthesize(req.text, &opts)?
        };
        Ok(SynthResult {
            pcm,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
