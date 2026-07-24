use rlx_nanowakeword::onnx::OnnxNanoModel;
use std::path::PathBuf;

#[test]
fn onnx_parity_soft_skip() {
    let path = std::env::var("NANOWAKEWORD_ONNX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/nanowakeword/model.onnx"));
    let Some(m) = OnnxNanoModel::try_load(&path) else {
        eprintln!("skip: onnx model not found at {}", path.display());
        return;
    };
    assert!(m.ok());
}
