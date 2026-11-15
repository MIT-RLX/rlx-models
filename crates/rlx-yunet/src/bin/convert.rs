// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! `rlx-yunet-convert MODEL.onnx OUT.safetensors`: a YuNet ONNX export's weights, checked
//! against the architecture, for [`rlx_yunet::YuNet::load`].

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 3,
        "usage: rlx-yunet-convert MODEL.onnx OUT.safetensors"
    );
    let w = rlx_yunet::onnx::weights_from_onnx(&std::fs::read(&args[1])?)?;
    w.save(std::path::Path::new(&args[2]))?;
    let n: usize = w.tensors.values().map(Vec::len).sum();
    println!("{} tensors, {n} parameters -> {}", w.tensors.len(), args[2]);
    Ok(())
}
