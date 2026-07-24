//! Dev decoder probe (not a product path).
//!
//! Expects dump tensors under `$RLX_TTS_DUMP` (default `/tmp/ttsdump`) and a
//! decoder graph from `$RLX_TINY_TTS_DECODER` or
//! `weights/tts/tiny-tts-rlx/graphs/decoder.rlxp` (legacy `onnx/decoder.onnx`).
//!
//! ```bash
//! cargo run -p rlx-tiny-tts --release --example dec_probe --features apple-silicon -- metal
//! ```
use rlx_runtime::{AotCache, CompileOptions, DType};
use rlx_tiny_tts::model::import_graph;
use std::path::PathBuf;

fn dump_dir() -> PathBuf {
    PathBuf::from(std::env::var("RLX_TTS_DUMP").unwrap_or_else(|_| "/tmp/ttsdump".into()))
}

fn rd(name: &str) -> Vec<f32> {
    std::fs::read(dump_dir().join(format!("{name}.f32")))
        .expect("read dump")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn decoder_path() -> PathBuf {
    if let Ok(p) = std::env::var("RLX_TINY_TTS_DECODER") {
        return PathBuf::from(p);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/tiny-tts-rlx");
    let nested = root.join("graphs/decoder.rlxp");
    if nested.is_file() {
        return nested;
    }
    root.join("onnx/decoder.onnx")
}

fn main() -> anyhow::Result<()> {
    let device = rlx_runtime::parse_device(&std::env::args().nth(1).unwrap_or("mlx".into()))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let z = rd("z");
    let g = rd("g");
    let length = z.len() / 80;
    eprintln!("[dec] length={length}");
    let path = decoder_path();
    anyhow::ensure!(path.is_file(), "missing decoder graph {}", path.display());
    let (mut hir, params, _r) = import_graph(&path, "decoder", length, true)?;
    if let Ok(tap) = std::env::var("RLX_TAP") {
        let mut found = None;
        for (i, n) in hir.nodes().iter().enumerate() {
            if n.name.as_deref() == Some(tap.as_str()) {
                found = Some(rlx_ir::hir::HirNodeId(i as u32));
            }
        }
        match found {
            Some(id) => {
                eprintln!("[tap] {tap} shape {:?}", hir.node(id).shape.dims());
                hir.outputs.push(id);
            }
            None => eprintln!("[tap] NOT FOUND {tap}"),
        }
    }
    let cache = AotCache::new(PathBuf::from("/tmp/dec_probe_aot"));
    let _ = std::fs::remove_dir_all("/tmp/dec_probe_aot");
    let mut c = cache
        .compile_hir_cached(
            &format!("dec_{device:?}_{length}"),
            device,
            hir,
            &CompileOptions::default(),
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    for (n, d) in &params {
        c.set_param(n, d);
    }
    c.finalize_params();
    let out = c.run_typed(&[
        ("z", &f32_bytes(&z), DType::F32),
        ("g", &f32_bytes(&g), DType::F32),
    ]);
    std::fs::create_dir_all("/tmp/rlx_dec")?;
    for (i, (b, dt)) in out.iter().enumerate() {
        if *dt != DType::F32 {
            continue;
        }
        let v: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        std::fs::write(format!("/tmp/rlx_dec/out{i}.f32"), f32_bytes(&v))?;
        eprintln!(
            "  out{i}: len={} absmean={:.4}",
            v.len(),
            v.iter().map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32
        );
    }
    Ok(())
}
