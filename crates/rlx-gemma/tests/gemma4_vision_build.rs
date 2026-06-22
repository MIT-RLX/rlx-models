// RLX — GPLv3. Vision-tower encoder build-smoke from the real E2B checkpoint.
use rlx_gemma::gemma4_vision::{VisionConfig, build_vision_encoder};
use rlx_gemma::qat_loader::GemmaQatLoader;
use std::path::PathBuf;

fn dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("RLX_GEMMA4_E2B_DIR") {
        let p = PathBuf::from(d);
        return p.join("config.json").is_file().then_some(p);
    }
    let home = std::env::var_os("HOME")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let s = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    s.join("config.json").is_file().then_some(s)
}

#[test]
fn gemma4_vision_encoder_builds_from_real_checkpoint() {
    let Some(d) = dir() else {
        eprintln!("[vision build] no ckpt — skip");
        return;
    };
    let cfg = VisionConfig::default();
    let mut loader = GemmaQatLoader::open(&d).expect("open loader");
    // small patch count for a quick build
    let (graph, params) =
        build_vision_encoder(&cfg, &mut loader, 1, 64).expect("build vision encoder");
    assert!(graph.nodes().len() > 100, "graph too small");
    assert!(!params.is_empty());
    // every encoder layer's quantized q_proj etc. must have resolved (8-bit dequant);
    // norms loaded; patch_embedder present.
    assert!(params.contains_key("model.vision_tower.patch_embedder.input_proj.weight"));
    assert!(
        params.contains_key("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight")
    );
    assert!(
        params.contains_key("model.vision_tower.encoder.layers.15.mlp.down_proj.linear.weight")
    );
    eprintln!(
        "[vision build] ok: {} nodes, {} f32 params",
        graph.nodes().len(),
        params.len()
    );
}
