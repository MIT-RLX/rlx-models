//! Validate q8 weight loading against the REAL Descript-DAC-44kHz (the oracle).
//! TSAC's q8 model has the same architecture; if the q8-dequantized + weight-normed
//! weights match the real DAC, my loading is correct (and rlx-dac's verified pipeline
//! is a working codec for these weights).
//!
//! ```bash
//! RLX_TSAC_DIR=$PWD/.cache/tsac RLX_DAC_DIR=$PWD/.cache/dac44 \
//!   cargo run -p rlx-tsac --example oracle_cmp --release --features native-codec,oracle
//! ```
use anyhow::{Context, Result};
use ndarray::Array3;
use rlx_dac::ops::weight_norm;
use rlx_dac::weights::WeightStore;
use rlx_tsac::default_tsac_dir;
use rlx_tsac::rlx_decode::{q8_conv_std, q8_convt_std, q8_f32, q8_v_raw};

fn stats(name: &str, real: &[f32], mine: &[f32]) {
    let n = real.len().min(mine.len());
    let (mut dot, mut rr, mut mm, mut err, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let (a, b) = (real[i] as f64, mine[i] as f64);
        dot += a * b;
        rr += a * a;
        mm += b * b;
        err += (a - b) * (a - b);
        mx = mx.max((a - b).abs());
    }
    let corr = dot / (rr.sqrt() * mm.sqrt()).max(1e-12);
    let rel = (err / rr.max(1e-12)).sqrt();
    eprintln!(
        "{name:42} len={n:>8} corr={corr:.4} rel_rms_err={rel:.4} max_abs_err={mx:.4} (real_rms={:.4} mine_rms={:.4})",
        (rr / n as f64).sqrt(),
        (mm / n as f64).sqrt(),
    );
}

fn real_conv(store: &WeightStore, prefix: &str) -> Result<Array3<f32>> {
    let (g, gs) = store.get(&format!("{prefix}.weight_g"))?;
    let (v, vs) = store.get(&format!("{prefix}.weight_v"))?;
    let gv = ndarray::ArrayView3::from_shape((gs[0], gs[1], gs[2]), g)?;
    let vv = ndarray::ArrayView3::from_shape((vs[0], vs[1], vs[2]), v)?;
    Ok(weight_norm(gv, vv))
}

fn main() -> Result<()> {
    let tdir = default_tsac_dir();
    let ddir = std::env::var("RLX_DAC_DIR").unwrap_or_else(|_| ".cache/dac44".into());
    let ddir = std::path::PathBuf::from(ddir);
    std::fs::create_dir_all(&ddir).ok();
    let st_path = ddir.join("model.safetensors");
    if !st_path.is_file() {
        rlx_dac::download::fetch_dac(&ddir, "44khz").context("download real DAC-44kHz")?;
    }
    let store = WeightStore::open(&st_path)?;

    eprintln!("=== real DAC key sample ===");
    for k in store.keys().take(6) {
        eprintln!("  {k}");
    }

    // Permutation search on RAW weight_v: find which axis order maps q8 → real.
    {
        let (rv, rs) = store.get("decoder.model.0.weight_v")?; // real layout
        let (mv, md) = q8_v_raw(&tdir, "decoder.model.0")?; // q8 dims [d0,d1,d2]
        eprintln!("raw v: real shape={rs:?}  q8 dims={md:?}");
        // try all 6 permutations of q8 axes, reshape to real's flat order, corr.
        let perms = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let (_d0, d1, d2) = (md[0], md[1], md[2]);
        for p in perms {
            // permuted dims
            let pd = [md[p[0]], md[p[1]], md[p[2]]];
            if pd != [rs[0], rs[1], rs[2]] {
                continue; // only permutations matching real shape
            }
            // build permuted flat (real index order a,b,c over pd) from mv (q8 [d0,d1,d2])
            let mut perm_flat = vec![0f32; mv.len()];
            let mut idx = 0;
            for a in 0..pd[0] {
                for b in 0..pd[1] {
                    for c in 0..pd[2] {
                        let coord = [a, b, c];
                        // inverse map: q8 axis i gets coord[ position of i in p ]
                        let mut q = [0usize; 3];
                        for (pos, &ax) in p.iter().enumerate() {
                            q[ax] = coord[pos];
                        }
                        perm_flat[idx] = mv[(q[0] * d1 + q[1]) * d2 + q[2]];
                        idx += 1;
                    }
                }
            }
            let mut dot = 0f64;
            let mut rr = 0f64;
            let mut mm = 0f64;
            for i in 0..rv.len() {
                let (x, y) = (rv[i] as f64, perm_flat[i] as f64);
                dot += x * y;
                rr += x * x;
                mm += y * y;
            }
            let corr = dot / (rr.sqrt() * mm.sqrt()).max(1e-12);
            eprintln!("  perm {p:?} -> shape {pd:?}  corr={corr:.4}");
        }
    }

    // Conv1d layers: real (weight_norm) vs q8 (raw + weight_norm).
    for prefix in [
        "decoder.model.0",
        "decoder.model.6",
        "encoder.block.0",
        "encoder.block.6",
    ] {
        match (real_conv(&store, prefix), q8_conv_std(&tdir, prefix)) {
            (Ok(r), Ok(m)) => {
                eprintln!("{prefix}: real{:?} mine{:?}", r.dim(), m.dim());
                stats(prefix, r.as_slice().unwrap(), m.as_slice().unwrap());
            }
            (r, m) => eprintln!("{prefix}: real_ok={} mine_ok={}", r.is_ok(), m.is_ok()),
        }
    }

    // Conv-transpose: real weight is [Ci][Co][K]; rlx-dac weight_norm on it directly.
    for prefix in ["decoder.model.1.block.1", "decoder.model.4.block.1"] {
        match (real_conv(&store, prefix), q8_convt_std(&tdir, prefix)) {
            (Ok(r), Ok(m)) => {
                eprintln!("{prefix} (convT): real{:?} mine{:?}", r.dim(), m.dim());
                stats(prefix, r.as_slice().unwrap(), m.as_slice().unwrap());
            }
            (r, m) => eprintln!("{prefix}: real_ok={} mine_ok={}", r.is_ok(), m.is_ok()),
        }
    }

    // Codebook (raw f32 both sides).
    let cbk = "quantizer.quantizers.0.codebook.weight";
    if let (Ok((r, _)), Ok(m)) = (store.get(cbk), q8_f32(&tdir, cbk)) {
        stats(cbk, r, &m);
    }
    Ok(())
}
