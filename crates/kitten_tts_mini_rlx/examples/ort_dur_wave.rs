//! Force ORT duration via production compile_slot_length + run_kitten_inference.
use kitten_tts_mini_rlx::bundle_compile::SeqCompileCache;
use rlx_ir::DType;
use rlx_runtime::Device;
use std::process::Command;

fn style_row() -> Vec<f32> {
    let out = Command::new("python3")
        .args([
            "-c",
            r#"
import numpy as np,sys
z=np.load('.cache/kittentts-mini-0.8/voices.npz')
sys.stdout.buffer.write(z['expr-voice-2-m'][6].astype(np.float32).tobytes())
"#,
        ])
        .current_dir("/Users/Shared/rlx-models")
        .output()
        .unwrap();
    out.stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn stats(wave: &[f32], trim: usize) {
    let w = &wave[..wave.len().min(trim)];
    let peak = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let mut zc = 0usize;
    for p in w.windows(2) {
        if (p[0] >= 0.0) != (p[1] >= 0.0) {
            zc += 1;
        }
    }
    eprintln!(
        "  trim={} peak={peak:.4} zc={zc} zc/s={:.1}",
        w.len(),
        zc as f32 / (w.len().max(1) as f32 / 24000.0)
    );
}

fn main() -> anyhow::Result<()> {
    let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 10, 0];
    let ort_dur: Vec<i64> = vec![3, 2, 2, 3, 4, 4, 13, 2, 1];
    let style = style_row();
    let token_len = ids.len();
    let compile_seq = kitten_tts_mini_rlx::compile_profile::compile_slot_length(token_len);
    let max_wave = 55_200usize;
    eprintln!("compile_seq={compile_seq} token_len={token_len}");
    let cache = SeqCompileCache::new(Device::Cpu, bundle, compile_seq.max(token_len), max_wave, 2);
    let graphs = cache.cached_graphs_for_seq(token_len)?;
    kitten_tts_mini_rlx::bundle_compile::shape_all_graphs_for_infer(
        &graphs,
        token_len,
        compile_seq,
    )?;
    let mut ids_padded = ids.clone();
    ids_padded.resize(compile_seq, 0);
    let ids_bytes: Vec<u8> = ids_padded.iter().flat_map(|v| v.to_le_bytes()).collect();
    let style_bytes: Vec<u8> = style.iter().flat_map(|v| v.to_le_bytes()).collect();
    let speed_bytes = 1.0f32.to_le_bytes().to_vec();
    let inputs = [
        ("input_ids", ids_bytes.as_slice(), DType::I64),
        ("style", style_bytes.as_slice(), DType::F32),
        ("speed", speed_bytes.as_slice(), DType::F32),
    ];
    let mut align = ort_dur.clone();
    align.resize(compile_seq, 0);
    let align_bytes: Vec<u8> = align.iter().flat_map(|d| d.to_le_bytes()).collect();

    eprintln!("A) free-run (no carry seed):");
    let outs =
        kitten_tts_mini_rlx::bundle_compile::run_kitten_inference(&graphs, &inputs, None, None);
    let wave: Vec<f32> = outs[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    stats(&wave, wave.len());

    eprintln!("B) ORT duration carry+align:");
    let outs = kitten_tts_mini_rlx::bundle_compile::run_kitten_inference(
        &graphs,
        &inputs,
        Some(&align_bytes),
        Some(&align_bytes),
    );
    let wave: Vec<f32> = outs[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    stats(&wave, 34 * 600);
    Ok(())
}
