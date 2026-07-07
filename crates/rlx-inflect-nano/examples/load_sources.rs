//! Validate the additive `AssetSource` loaders: load the same Inflect-Nano
//! bundle from a directory, a packed `.rlxpack` file, and in-memory bytes, and
//! confirm all three synthesize byte-identical audio. Requires `rlx-graph`.
//!
//! Run: cargo run -p rlx-inflect-nano --release --example load_sources -- weights/inflect-nano-rlx

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_core::asset_source::pack;
use rlx_core::AssetSource;
use rlx_inflect_nano::{InferOpts, InflectNano};

fn synth(m: &InflectNano) -> anyhow::Result<Vec<f32>> {
    let opts = InferOpts::default();
    Ok(m.synthesize("Hello from a versatile loader.", &opts)?.samples)
}

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "weights/inflect-nano-rlx".into()),
    );

    let reference = synth(&InflectNano::load(&dir)?)?;
    println!("dir      : {} samples", reference.len());

    let pack_path = std::env::temp_dir().join("inflect-nano-demo.rlxpack");
    pack::write_dir(&dir, &pack_path)?;
    let file = synth(&InflectNano::load(&pack_path)?)?;
    println!("pack file: {} samples", file.len());

    let bytes: std::sync::Arc<[u8]> = std::fs::read(&pack_path)?.into();
    let mem = synth(&InflectNano::load(AssetSource::pack_bytes(bytes)?)?)?;
    println!("pack mem : {} samples", mem.len());

    let mut map = HashMap::new();
    for name in AssetSource::dir(&dir).names()? {
        map.insert(name.clone(), std::fs::read(dir.join(&name))?);
    }
    let mmap = synth(&InflectNano::load(AssetSource::memory(map))?)?;
    println!("mem map  : {} samples", mmap.len());

    let ok = file == reference && mem == reference && mmap == reference;
    println!("\nall sources byte-identical: {}", if ok { "YES ✓" } else { "NO ✗" });
    let _ = std::fs::remove_file(&pack_path);
    anyhow::ensure!(ok, "source outputs diverged");
    Ok(())
}
