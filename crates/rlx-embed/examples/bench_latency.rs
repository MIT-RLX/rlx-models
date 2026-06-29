// Common-model embedding latency for RLX (all-MiniLM-L6-v2), matching the
// PyTorch/ONNX-Runtime harness: synthetic batch x seq inputs, p50 of forward.
// Run: MINILM_DIR=/tmp/minilm6 cargo run -p rlx-embed --release --example bench_latency
use rlx_embed::{Pooling, RlxEmbed};
use rlx_runtime::Device;
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("MINILM_DIR").unwrap_or_else(|_| "/tmp/minilm6".into());
    let devname = std::env::var("DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = match devname.as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        "ane" | "coreml" => Device::Ane,
        _ => Device::Cpu,
    };
    let mut model = RlxEmbed::from_dir_on(Path::new(&dir), Pooling::Mean, device)?;
    let label = format!("rlx-{devname}");
    let seq: usize = std::env::var("SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let (runs, warmup) = (50usize, 10usize);
    println!("framework,batch,p50_ms");
    for &b in &[1usize, 4, 8, 16, 32] {
        let n = b * seq;
        let ids: Vec<f32> = (0..n).map(|i| (i * 131 % 30000) as f32).collect();
        let mask = vec![1.0f32; n];
        let tt = vec![0.0f32; n];
        let pos: Vec<f32> = (0..b).flat_map(|_| (0..seq).map(|i| i as f32)).collect();
        let inputs = [
            ("input_ids", ids.as_slice()),
            ("attention_mask", mask.as_slice()),
            ("token_type_ids", tt.as_slice()),
            ("position_ids", pos.as_slice()),
        ];
        for _ in 0..warmup {
            let _ = model.forward(&inputs, b, seq)?;
        }
        let mut t: Vec<f64> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let s = Instant::now();
            let _ = model.forward(&inputs, b, seq)?;
            t.push(s.elapsed().as_secs_f64() * 1e3);
        }
        t.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let p50 = t[t.len() / 2];
        eprintln!("{label}  b={b:>3}  p50={p50:7.2} ms");
        println!("{label},{b},{p50:.2}");
    }
    Ok(())
}
