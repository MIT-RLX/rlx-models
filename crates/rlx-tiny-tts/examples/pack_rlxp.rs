//! Pack TinyTTS/MeloTTS into a native outer `.rlxp` (no ONNX on Hub).
//!
//! Bakes each ONNX subgraph to `graphs/<name>.rlxp` (hot tensors + graph.json),
//! then embeds those plus `config.json` / `frontend/` into `tiny-tts.rlxp`.
//!
//! ```bash
//! cargo run -p rlx-tiny-tts --release --example pack_rlxp --features apple-silicon -- \
//!   weights/tts/tiny-tts-rlx weights/tts/tiny-tts-rlx/tiny-tts.rlxp
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rlx_assets::native_pack::{pack_native_from_onnx_dir, specs_from_root};

const COMPONENTS: &[&str] = &["text_encoder", "duration_predictor", "flow", "decoder"];

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "weights/tts/tiny-tts-rlx".into()),
    );
    let out = PathBuf::from(
        args.next()
            .unwrap_or_else(|| dir.join("tiny-tts.rlxp").display().to_string()),
    );
    if !dir.join("config.json").is_file() {
        bail!("missing {}/config.json", dir.display());
    }
    let specs = specs_from_root(&dir, COMPONENTS);
    for s in &specs {
        if !s.onnx_path.is_file() {
            bail!("missing pack source {}", s.onnx_path.display());
        }
    }
    pack_native_from_onnx_dir(&dir, &specs, &out, "tiny-tts")
        .with_context(|| format!("pack {} → {}", dir.display(), out.display()))?;
    let bytes = std::fs::metadata(&out)?.len();
    println!(
        "packed {} ({:.1} MiB) — nested graphs/*.rlxp, no ONNX",
        out.display(),
        bytes as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}
