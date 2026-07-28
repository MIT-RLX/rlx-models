//! Pack a TinyTTS/MeloTTS directory into a single `.rlxpack`.
//!
//! ```bash
//! cargo run -p rlx-tiny-tts --release --example pack_rlxpack -- \
//!   weights/tts/tiny-tts-rlx weights/tts/tiny-tts-rlx/tiny-tts.rlxpack
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rlx_tiny_tts::asset_source::pack;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "weights/tts/tiny-tts-rlx".into()),
    );
    let out = PathBuf::from(
        args.next()
            .unwrap_or_else(|| dir.join("tiny-tts.rlxpack").display().to_string()),
    );
    if !dir.join("config.json").is_file() {
        bail!("missing {}/config.json", dir.display());
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Skip misleading community/staging GGUF wraps and an existing pack.
    let skip = ["tiny-tts-rlx.f16.gguf", "tiny-tts.rlxpack"];
    let tmp = std::env::temp_dir().join(format!("rlx-tiny-pack-{}", std::process::id()));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)?;
    }
    std::fs::create_dir_all(&tmp)?;
    for ent in std::fs::read_dir(&dir)? {
        let ent = ent?;
        let name = ent.file_name();
        let name_s = name.to_string_lossy();
        if skip.iter().any(|s| *s == name_s) || name_s.ends_with(".gguf") {
            continue;
        }
        let dest = tmp.join(&name);
        if ent.file_type()?.is_dir() {
            copy_dir(&ent.path(), &dest)?;
        } else {
            std::fs::copy(ent.path(), &dest)?;
        }
    }
    pack::write_dir(&tmp, &out)
        .with_context(|| format!("pack {} → {}", tmp.display(), out.display()))?;
    let _ = std::fs::remove_dir_all(&tmp);
    let bytes = std::fs::metadata(&out)?.len();
    println!(
        "packed {} ({:.1} MiB)",
        out.display(),
        bytes as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for ent in std::fs::read_dir(src)? {
        let ent = ent?;
        let to = dst.join(ent.file_name());
        if ent.file_type()?.is_dir() {
            copy_dir(&ent.path(), &to)?;
        } else {
            std::fs::copy(ent.path(), &to)?;
        }
    }
    Ok(())
}
