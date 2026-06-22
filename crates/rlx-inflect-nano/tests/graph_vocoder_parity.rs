//! The rlx-ir graph vocoder (compiled via the RLX compiler → runs on every
//! backend) must match the host-eager vocoder. Validated on CPU here; the same
//! graph targets Metal/MLX/CUDA/etc. (compile-checked elsewhere).
#![cfg(feature = "rlx-graph")]

use std::path::PathBuf;

use rlx_inflect_nano::{InferOpts, InflectNano};
use serde_json::Value;

fn data_dir() -> Option<PathBuf> {
    let base = std::env::var("RLX_INFLECT_NANO_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/inflect-nano-rlx")
        });
    base.join("config.json").exists().then_some(base)
}

#[test]
fn graph_vocoder_matches_host() {
    let Some(dir) = data_dir() else {
        eprintln!("skip: bundle not found");
        return;
    };
    let model = InflectNano::load_from_dir(&dir).expect("load");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("fixtures/manifest.json")).unwrap())
            .unwrap();
    let opts = InferOpts::default();

    let mut worst = 0.0f32;
    for case in manifest["cases"].as_array().unwrap().iter().take(3) {
        let text = case["text"].as_str().unwrap();
        let (p, t, l) = model.text_to_ids(text).unwrap();
        let mel = model
            .mel_from_ids(&p, &t, &l, model.cfg.default_speaker(), &opts)
            .unwrap();
        let host = model.wav_from_mel(&mel).unwrap();

        let mut g = model
            .compile_vocoder_graph(mel.dim().1, rlx_runtime::Device::Cpu)
            .expect("compile graph");
        let gw = g.forward(&mel).expect("graph forward");

        assert_eq!(
            host.len(),
            gw.len(),
            "wav length mismatch {} vs {}",
            host.len(),
            gw.len()
        );
        let d = host
            .iter()
            .zip(&gw)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        worst = worst.max(d);
        eprintln!(
            "'{}': graph-vs-host wav maxdiff={d:.3e}",
            &text[..text.len().min(30)]
        );
    }
    eprintln!("GRAPH VOCODER WORST={worst:.3e}");
    assert!(worst < 1e-3, "graph vocoder parity too loose: {worst:.3e}");
}

/// Same graph, compiled + run on Metal (Apple GPU). Validates the multi-backend
/// path on real GPU hardware and reports timing vs the host-eager path.
#[cfg(feature = "metal")]
#[test]
fn graph_vocoder_on_metal() {
    let Some(dir) = data_dir() else { return };
    if !rlx_runtime::is_available(rlx_runtime::Device::Metal) {
        eprintln!("skip: Metal not available");
        return;
    }
    let model = InflectNano::load_from_dir(&dir).expect("load");
    let opts = InferOpts::default();
    let text = "The weather is nice today, and I feel very relaxed.";
    let (p, t, l) = model.text_to_ids(text).unwrap();
    let mel = model
        .mel_from_ids(&p, &t, &l, model.cfg.default_speaker(), &opts)
        .unwrap();
    let host = model.wav_from_mel(&mel).unwrap();

    let mut g = model
        .compile_vocoder_graph(mel.dim().1, rlx_runtime::Device::Metal)
        .expect("compile metal vocoder");
    let gw = g.forward(&mel).expect("metal forward"); // warm
    let t0 = std::time::Instant::now();
    let gw = g.forward(&mel).unwrap();
    let dt = t0.elapsed().as_secs_f32();
    let d = host
        .iter()
        .zip(&gw)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!(
        "Metal vocoder: {} samples in {dt:.4}s, maxdiff vs host = {d:.3e}",
        gw.len()
    );
    assert!(d < 2e-3, "metal vocoder parity too loose: {d:.3e}");
}
