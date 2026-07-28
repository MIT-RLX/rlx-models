// RLX — GPLv3. Verify the RUST 2-bit affine dequant of a real gate_proj expert
// against mlx's reference (from mlx.dequantize).
//   RLX_DSV4_DIR=... cargo run --release -p rlx-models-core --example dsv4_dequant_check
use anyhow::{Context, Result};
use rlx_models_core::weight_loader::{MlxLoader, WeightLoader};

fn f32le(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<()> {
    let dir = std::env::var("RLX_DSV4_DIR").context("RLX_DSV4_DIR")?;
    let mut l = MlxLoader::open_lazy(&dir)?;
    let key = "model.layers.0.ffn.switch_mlp.gate_proj.weight";
    let p = l.take_packed_mlx(key)?.context("packed")?;
    let (bits, gs) = match p.scheme {
        rlx_ir::quant::QuantScheme::MlxAffine { bits, group_size } => (bits as u32, group_size),
        s => anyhow::bail!("not affine: {s:?}"),
    };
    let n_expert = 256usize;
    let out_dim = 2048usize; // gate_proj moe_intermediate (real per-expert out)
    let k = 4096usize;
    let n_groups = k / gs as usize;
    println!(
        "scheme bits={bits} gs={gs} n_expert={n_expert} out_dim={out_dim} n_groups={n_groups}"
    );
    println!(
        "w_q bytes {} scales bytes {} biases bytes {}",
        p.w_q.len(),
        p.scales.len(),
        p.biases.len()
    );

    let scales = f32le(&p.scales);
    let biases = if p.biases.is_empty() {
        vec![0f32; n_expert * out_dim * n_groups]
    } else {
        f32le(&p.biases)
    };
    let slab = p.w_q.len() / n_expert;
    let sb = out_dim * n_groups;
    let deq = rlx_mlx_io::dequant_affine_f32(
        &p.w_q[0..slab],
        &scales[0..sb],
        &biases[0..sb],
        bits,
        gs,
        out_dim,
        n_groups,
    )?;
    println!("rust dequant expert0 row0 [:8]: {:?}", &deq[..8]);
    println!(
        "mlx  reference          [:8]: [-0.0234375, 0.0, 0.0234375, 0.0234375, 0.0, 0.0, 0.0234375, 0.0234375]"
    );
    let mn = deq.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = deq.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("rust dequant range: {mn:.4} .. {mx:.4}  (mlx: -0.1875 .. 0.125)");
    Ok(())
}
