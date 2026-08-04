// Dump a representative set of REAL DeepSeek-V4-Flash weight tensors (dequantized
// to f32) as `.tensor` files (u32 rows, u32 cols, f32 LE data) so `../rlx`'s
// `opscope-layers` can inspect the real weight DATA — per-layer decomposition /
// quant / low-rank / 2:4-sparsity structure.
//
//   cargo run -q -p rlx-models-core --example dsv4_dump_weights --features metal -- <out_dir>
//   (then)  cargo run -q -p rlx-opscope --bin opscope-layers --release -- <out_dir> 0.5
use anyhow::Result;
use rlx_ir::quant::QuantScheme;
use rlx_mlx_io::{dequant_mxfp4_f32, dequant_mxfp8_f32};
use rlx_models_core::weight_loader::{MlxLoader, WeightLoader};

fn save_tensor(path: &str, rows: usize, cols: usize, data: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&(rows as u32).to_le_bytes())?;
    f.write_all(&(cols as u32).to_le_bytes())?;
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()
}

fn load_dequant(loader: &mut MlxLoader, key: &str) -> Option<(Vec<f32>, usize, usize, String)> {
    if let Ok(Some(p)) = loader.take_packed_mlx(key) {
        let (out, inn) = (p.out_shape[0], p.out_shape[1]);
        let (w, tag) = match p.scheme {
            QuantScheme::MlxMxfp4 { group_size } => (
                dequant_mxfp4_f32(
                    &p.w_q,
                    &p.scales,
                    group_size,
                    out,
                    inn / group_size as usize,
                )
                .ok()?,
                "mxfp4".to_string(),
            ),
            QuantScheme::MlxMxfp8 { group_size } => (
                dequant_mxfp8_f32(
                    &p.w_q,
                    &p.scales,
                    group_size,
                    out,
                    inn / group_size as usize,
                )
                .ok()?,
                "mxfp8".to_string(),
            ),
            other => {
                eprintln!("  {key}: unhandled packed scheme {other:?} — skip");
                return None;
            }
        };
        Some((w, out, inn, tag))
    } else if let Ok((w, shape)) = loader.take(key) {
        if shape.len() == 2 {
            Some((w, shape[0], shape[1], "dense".to_string()))
        } else {
            None
        }
    } else {
        None
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let out_dir = a
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/dsv4_weights".into());
    let ckpt = "/Volumes/FOUR/DeepSeek/DeepSeek-V4-Flash-0731-MXFP4-MLX";
    std::fs::create_dir_all(&out_dir)?;
    let mut loader = MlxLoader::open_lazy(ckpt)?;

    // Representative resident weights (layer 0) + one routed expert. Names as the
    // decode builder uses them; the loader maps to the ckpt's naming.
    // Raw ckpt keys (no `model.` prefix). attn projections = MXFP8, shared experts =
    // MXFP8, routed experts (experts.{e}.w1/w2/w3) = MXFP4, router/embed = dense.
    let keys = [
        ("l0.attn.wq_a(mxfp8)", "layers.0.attn.wq_a.weight"),
        ("l0.attn.wq_b(mxfp8)", "layers.0.attn.wq_b.weight"),
        ("l0.attn.wkv(mxfp8)", "layers.0.attn.wkv.weight"),
        ("l0.attn.wo_a(mxfp8)", "layers.0.attn.wo_a.weight"),
        ("l0.attn.wo_b(mxfp8)", "layers.0.attn.wo_b.weight"),
        ("l0.ffn.gate(router)", "layers.0.ffn.gate.weight"),
        (
            "l0.shared.w1(mxfp8)",
            "layers.0.ffn.shared_experts.w1.weight",
        ),
        (
            "l0.shared.w2(mxfp8)",
            "layers.0.ffn.shared_experts.w2.weight",
        ),
        ("l0.expert0.w1(mxfp4)", "layers.0.ffn.experts.0.w1.weight"),
        ("l0.expert0.w2(mxfp4)", "layers.0.ffn.experts.0.w2.weight"),
        ("l0.expert5.w1(mxfp4)", "layers.0.ffn.experts.5.w1.weight"),
        ("embed", "embed.weight"),
    ];
    let mut ok = 0;
    for (name, key) in keys {
        match load_dequant(&mut loader, key) {
            Some((w, rows, cols, tag)) => {
                let path = format!("{out_dir}/{name}.tensor");
                save_tensor(&path, rows, cols, &w)?;
                let (mn, mx) = w
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                eprintln!("  {name:<22} [{rows}×{cols}] {tag:<6} range [{mn:.4},{mx:.4}] → ok");
                ok += 1;
            }
            None => eprintln!("  {name:<22} MISSING ({key})"),
        }
    }
    eprintln!(
        "\nDumped {ok} tensors to {out_dir}. Now run:\n  cargo run -q -p rlx-opscope --bin opscope-layers --release -- {out_dir} 0.5"
    );
    Ok(())
}
