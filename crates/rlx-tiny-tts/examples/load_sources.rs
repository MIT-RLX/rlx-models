//! Demonstrate + validate the versatile `AssetSource` loaders: load the same
//! TinyTTS bundle from a directory, a packed `.rlxpack` file, and an in-memory
//! byte map, and confirm all three synthesize byte-identical audio.
//!
//! Run: cargo run -p rlx-tiny-tts --release --example load_sources -- weights/tiny-tts-rlx

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_tiny_tts::{AssetSource, InferOpts, TinyTts, asset_source::pack};

fn synth(model: &TinyTts) -> anyhow::Result<Vec<f32>> {
    let opts = InferOpts::from_config(model.config());
    Ok(model
        .synthesize("Hello from a versatile loader.", &opts)?
        .samples)
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/tiny-tts-rlx".into());
    let dir = PathBuf::from(dir);

    // 1) Directory (the classic path) — also the reference output.
    let ref_wav = synth(&TinyTts::load(&dir)?)?;
    println!("dir      : {} samples", ref_wav.len());

    // 2) Pack the whole bundle into one `.rlxpack` file, load from the file.
    let pack_path = std::env::temp_dir().join("tiny-tts-demo.rlxpack");
    pack::write_dir(&dir, &pack_path)?;
    let file_wav = synth(&TinyTts::load(&pack_path)?)?;
    println!(
        "pack file: {} samples ({} on disk)",
        file_wav.len(),
        pack_path.display()
    );

    // 3) Load the pack bytes into memory and load from RAM (no bundle on disk).
    let bytes: std::sync::Arc<[u8]> = std::fs::read(&pack_path)?.into();
    let mem_wav = synth(&TinyTts::load(AssetSource::pack_bytes(bytes)?)?)?;
    println!("pack mem : {} samples", mem_wav.len());

    // 4) Raw in-memory asset map (e.g. assets fetched over the network).
    let mut map = HashMap::new();
    for name in AssetSource::dir(&dir).names()? {
        map.insert(name.clone(), std::fs::read(dir.join(&name))?);
    }
    let map_wav = synth(&TinyTts::load(AssetSource::memory(map))?)?;
    println!("mem map  : {} samples", map_wav.len());

    let identical = file_wav == ref_wav && mem_wav == ref_wav && map_wav == ref_wav;
    println!(
        "\nall four sources byte-identical: {}",
        if identical { "YES ✓" } else { "NO ✗" }
    );
    let _ = std::fs::remove_file(&pack_path);
    anyhow::ensure!(identical, "source outputs diverged");
    Ok(())
}
