//! Pack `onnx/wespeaker.onnx` → `graphs/wespeaker.rlxp` (native RLX weights).
//!
//! ```bash
//! cargo run -p rlx-wespeaker --release --example pack_rlxp --features pack -- \
//!   weights/wespeaker-voxceleb-resnet34-LM
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "weights/wespeaker-voxceleb-resnet34-LM".into()),
    );
    let onnx = dir.join("onnx/wespeaker.onnx");
    if !onnx.is_file() {
        bail!("missing {}", onnx.display());
    }
    let out = dir.join("graphs/wespeaker.rlxp");
    std::fs::create_dir_all(out.parent().unwrap())?;
    rlx_assets::native_pack::export_onnx_to_subgraph_rlxp(&onnx, &out, "wespeaker")
        .with_context(|| format!("pack {} → {}", onnx.display(), out.display()))?;
    println!(
        "wrote {} ({} bytes)",
        out.display(),
        std::fs::metadata(&out)?.len()
    );
    Ok(())
}
