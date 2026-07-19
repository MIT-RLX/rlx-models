// Parity-bisection harness: import a tone_color-style ONNX graph, compile it
// FRESH on CPU (no AOT cache), run it on the dumped inputs, and write every
// output (real + `RLX_ONNX_TAP` intermediates) to `native_tap_<i>.bin`.
//
// Usage:
//   RLX_ONNX_TAP=/enc_q/Mul_4_output_0 \
//   cargo run -p rlx-openvoice --example tap_run -- <tone_color.onnx> <dump_dir>
//
// The dump dir must contain audio.f32 / src_tone.f32 / dest_tone.f32 / tau.f32 /
// meta.txt (written by `RLX_OV_DUMP`).

use std::path::Path;

use anyhow::{Context, Result};
use rlx_runtime::{DType, Device, Session};
use rlx_tiny_tts::model::import_graph;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 3, "usage: tap_run <onnx> <dump_dir>");
    let onnx = Path::new(&args[1]);
    let dump = Path::new(&args[2]);

    let meta = std::fs::read_to_string(dump.join("meta.txt")).context("meta.txt")?;
    let t: usize = meta
        .split("t=")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .context("parse t")?;
    eprintln!("[tap_run] t={t} onnx={}", onnx.display());

    let device = match std::env::var("RLX_TAP_DEVICE").as_deref() {
        Ok("metal") => Device::Metal,
        Ok("mlx") => Device::Mlx,
        _ => Device::Cpu,
    };
    eprintln!("[tap_run] device={device:?}");
    let (hir, params, _report) = import_graph(onnx, "probe", t, true)?;
    let sess = Session::new(device);
    let mut g = sess
        .compile_hir(hir)
        .map_err(|e| anyhow::anyhow!("compile: {e:?}"))?;
    for (name, data) in &params {
        g.set_param(name, data);
    }
    g.finalize_params();

    let rd = |name: &str| -> Result<Vec<u8>> {
        std::fs::read(dump.join(name)).with_context(|| format!("read {name}"))
    };
    // tone_extract takes a single `input` [1, t, 513]; tone_color takes 5 inputs.
    let is_extract = onnx.to_string_lossy().contains("tone_extract");
    let outs = if is_extract {
        let inp = rd("te_input.f32")?;
        g.run_typed(&[("input", &inp, DType::F32)])
    } else {
        let audio = rd("audio.f32")?;
        let src = rd("src_tone.f32")?;
        let dst = rd("dest_tone.f32")?;
        let tau = rd("tau.f32")?;
        let alen = (t as i64).to_le_bytes().to_vec();
        g.run_typed(&[
            ("audio", &audio, DType::F32),
            ("audio_length", &alen, DType::I64),
            ("src_tone", &src, DType::F32),
            ("dest_tone", &dst, DType::F32),
            ("tau", &tau, DType::F32),
        ])
    };
    for (i, (bytes, dt)) in outs.iter().enumerate() {
        let p = dump.join(format!("native_tap_{i}.bin"));
        std::fs::write(&p, bytes)?;
        let n = bytes.len() / dt.size_bytes().max(1);
        eprintln!(
            "[tap_run] out[{i}] dtype={dt:?} elems={n} -> {}",
            p.display()
        );
    }
    Ok(())
}
