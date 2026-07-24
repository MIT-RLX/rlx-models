use rlx_openwakeword::onnx::OnnxWakeBundle;
use std::path::PathBuf;

#[test]
fn onnx_parity_soft_skip() {
    let dir = std::env::var("OPENWAKEWORD_ONNX_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/openwakeword/onnx"));
    let Some(bundle) = OnnxWakeBundle::try_load(&dir) else {
        eprintln!("skip: onnx models not found under {}", dir.display());
        return;
    };
    assert!(bundle.sessions_ok());
}
