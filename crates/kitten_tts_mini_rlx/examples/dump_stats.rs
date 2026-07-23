use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_probe_graph, import_from_bundle_cached, run_parity_inputs,
};
use rlx_runtime::Device;
use std::process::Command;

fn style() -> Vec<f32> {
    let o = Command::new("python3")
        .args([
            "-c",
            r#"
import numpy as np,sys
z=np.load('.cache/kittentts-mini-0.8/voices.npz')
sys.stdout.buffer.write(z['expr-voice-2-m'][6].astype('float32').tobytes())
"#,
        ])
        .current_dir("/Users/Shared/rlx-models")
        .output()
        .unwrap();
    o.stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let want = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/decoder/generator/m_source/l_sin_gen/Sin".into());
    let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let ids = vec![0i64, 50, 83, 156, 54, 57, 135, 10, 0];
    let seq = ids.len();
    let opts = GraphOptions {
        sequence_length: seq,
        max_waveform_samples: 55_200,
    };
    let import = import_from_bundle_cached(&bundle, &opts)?;
    let node = import
        .hir
        .nodes()
        .iter()
        .filter(|n| {
            n.name.as_deref() == Some(want.as_str())
                || n.name
                    .as_deref()
                    .is_some_and(|n| n.ends_with(want.as_str()))
        })
        .max_by_key(|n| {
            n.shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .product::<usize>()
        })
        .ok_or_else(|| anyhow::anyhow!("missing {want}"))?;
    eprintln!(
        "probe {} id={} shape={:?} op={:?}",
        want,
        node.id.0,
        node.shape.dims(),
        std::mem::discriminant(&node.op)
    );
    let mut g = compile_probe_graph(Device::Cpu, &bundle, &opts, &import, node.id, &want)?;
    let outs = run_parity_inputs(&mut g, seq, &ids, &style());
    let bytes = &outs[0].0;
    let v: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let peak = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let nz = v.iter().filter(|x| x.abs() > 1e-6).count();
    let n = v.len().min(4);
    eprintln!(
        "{want}: len={len} peak={peak:.4e} nonzero={nz}/{len} t0={t0:?}",
        len = v.len(),
        t0 = &v[..n]
    );
    Ok(())
}
