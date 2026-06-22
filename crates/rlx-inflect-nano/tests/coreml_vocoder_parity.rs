//! ONNX Runtime vocoder (CoreML EP when available) must match the host vocoder.
#![cfg(feature = "onnx")]

use std::path::PathBuf;

use rlx_inflect_nano::{InferOpts, InflectNano};

fn data_dir() -> Option<PathBuf> {
    let base = std::env::var("RLX_INFLECT_NANO_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/inflect-nano-rlx")
        });
    (base.join("config.json").exists() && base.join("vocoder.onnx").exists()).then_some(base)
}

#[test]
fn onnx_coreml_vocoder_matches_host() {
    let Some(dir) = data_dir() else {
        eprintln!("skip: bundle / vocoder.onnx not found");
        return;
    };
    let model = InflectNano::load_from_dir(&dir).expect("load");
    let opts = InferOpts::default();
    let text = "The weather is nice today, and I feel very relaxed.";
    let (p, t, l) = model.text_to_ids(text).unwrap();
    let mel = model
        .mel_from_ids(&p, &t, &l, model.cfg.default_speaker(), &opts)
        .unwrap();

    let host = model.wav_from_mel(&mel).unwrap();
    let onnx = model.vocode_onnx(&mel, true).expect("onnx vocode");

    assert_eq!(
        host.len(),
        onnx.len(),
        "len {} vs {}",
        host.len(),
        onnx.len()
    );
    let d = host
        .iter()
        .zip(&onnx)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    // CoreML runs the static model on ANE/GPU in reduced precision (fp16), so the
    // bar is looser than the f32 paths — this larger-than-1e-4 diff is in fact the
    // signature that CoreML is genuinely executing (not silently falling back to
    // CPU, which matches at ~1e-4). The overlap-chunking itself is bit-exact
    // (see streaming_parity). ~5% on a tanh-bounded waveform is inaudible.
    eprintln!("onnx(coreml)-vs-host vocoder maxdiff = {d:.3e}");
    assert!(d < 1e-1, "onnx vocoder parity unexpectedly large: {d:.3e}");
}
