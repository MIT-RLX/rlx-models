// Loads the pretrained NomicBERT encoder on CPU and Metal with identical token
// ids and checks the per-token hidden states agree — the forward-parity
// milestone before warm-start fine-tuning for rlx-termclean.
//
//   cargo run -q -p rlx-embed --example nomic_metal_parity --features metal -- <model_dir>
use rlx_embed::RlxNomicModel;
use rlx_runtime::{Device, is_available};
use std::path::Path;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb + 1e-12)
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: nomic_metal_parity <model_dir>");
    let dir = Path::new(&dir);
    let config = dir.join("config.json");
    let weights = dir.join("model.safetensors");
    let weights = weights.to_str().unwrap();

    let (batch, seq) = (1usize, 16usize);
    // Fixed ids: [CLS] + a few real WordPiece ids + [SEP], then pad to seq.
    let real = [
        101.0f32, 2478.0, 1996.0, 3722.0, 7953.0, 2005.0, 4083.0, 102.0,
    ];
    let mut ids = real.to_vec();
    ids.resize(seq, 0.0);
    let mask: Vec<f32> = (0..seq)
        .map(|i| if i < real.len() { 1.0 } else { 0.0 })
        .collect();
    let tt = vec![0.0f32; seq];

    println!("loading NomicBERT (768d × 12L) on CPU…");
    let mut cpu = RlxNomicModel::load_sized_on(&config, weights, batch, seq, Device::Cpu)?;
    let h_cpu = cpu.forward(&ids, &mask, &tt);
    println!(
        "CPU forward: {} hidden values (expect {} = {batch}×{seq}×768)",
        h_cpu.len(),
        batch * seq * 768
    );

    if !is_available(Device::Metal) {
        println!(
            "Metal UNAVAILABLE — CPU-only. first 6 hidden = {:?}",
            &h_cpu[..6.min(h_cpu.len())]
        );
        return Ok(());
    }

    println!("loading on Metal…");
    let mut metal = RlxNomicModel::load_sized_on(&config, weights, batch, seq, Device::Metal)?;
    let h_metal = metal.forward(&ids, &mask, &tt);

    let n = h_cpu.len().min(h_metal.len());
    let mut max_abs = 0f32;
    let (mut ss_diff, mut ss_ref) = (0f64, 0f64);
    for i in 0..n {
        let d = (h_cpu[i] - h_metal[i]).abs();
        max_abs = max_abs.max(d);
        ss_diff += (d as f64).powi(2);
        ss_ref += (h_cpu[i] as f64).powi(2);
    }
    let rel_l2 = (ss_diff.sqrt() / (ss_ref.sqrt() + 1e-12)) as f32;
    let cos = cosine(&h_cpu[..n], &h_metal[..n]);

    println!("Metal forward: {} values", h_metal.len());
    println!("first 6 CPU  : {:?}", &h_cpu[..6]);
    println!("first 6 Metal: {:?}", &h_metal[..6]);
    println!(
        "\nCPU vs Metal hidden states: max_abs_diff={max_abs:.3e}  rel_l2={rel_l2:.3e}  cosine={cos:.6}"
    );
    if cos > 0.999 && rel_l2 < 0.02 {
        println!("PASS: pretrained NomicBERT encoder runs on Metal and matches CPU.");
    } else {
        println!("INVESTIGATE: CPU/Metal divergence beyond tolerance.");
    }
    Ok(())
}
