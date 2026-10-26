//! Inputs and the expected velocity for the browser denoiser check
//! (`rlx-models-web/web/flux_spike.html`): one step of the exported web
//! GGUF on this machine, with fixed pseudo-random inputs.
//!
//! `cargo run -p rlx-flux1 --release --features metal --example web_reference -- GGUF OUT_DIR [DEVICE]`
use anyhow::Result;
use std::str::FromStr;

/// Deterministic N(0, sd) values (xorshift + Box-Muller).
fn normal(n: usize, sd: f32, mut s: u64) -> Vec<f32> {
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v) = (next(), next());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32 * sd
        })
        .collect()
}

fn write(path: &std::path::Path, v: &[f32]) -> Result<()> {
    std::fs::write(
        path,
        v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (gguf, out) = (&args[0], std::path::PathBuf::from(&args[1]));
    let device = rlx_runtime::Device::from_str(args.get(2).map_or("metal", String::as_str))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    std::fs::create_dir_all(&out)?;
    let file = rlx_gguf::GgufFile::from_bytes(std::fs::read(gguf)?)?;
    let cfg = rlx_flux1::web::config_from_meta(&file)?;
    let geo = rlx_flux1::DitGeometry {
        txt_len: 128,
        grid_h: 10,
        grid_w: 10,
    };
    let (img, txt, pooled) = (
        normal(geo.img_len() * cfg.in_channels, 1.0, 1),
        normal(geo.txt_len * cfg.context_dim, 0.2, 2),
        normal(cfg.vec_dim, 0.5, 3),
    );
    let (t, g) = (1.0f32, 3.5f32);
    let mut replay = rlx_flux1::web::Replay::new(file);
    let t0 = std::time::Instant::now();
    let mut dit = rlx_flux1::Flux1Dit::compile(&cfg, geo, &mut replay, device, 4, 8)?;
    eprintln!("compiled in {:.1}s", t0.elapsed().as_secs_f32());
    let t1 = std::time::Instant::now();
    let want = dit.forward(&img, &txt, &pooled, t, g)?;
    eprintln!("one step in {:.2}s", t1.elapsed().as_secs_f32());
    write(&out.join("img.bin"), &img)?;
    write(&out.join("txt.bin"), &txt)?;
    write(&out.join("pooled.bin"), &pooled)?;
    write(&out.join("want.bin"), &want)?;
    std::fs::write(
        out.join("meta.json"),
        format!(
            "{{\"t\": {t}, \"guidance\": {g}, \"txt_len\": {}, \"grid_h\": {}, \"grid_w\": {}}}",
            geo.txt_len, geo.grid_h, geo.grid_w
        ),
    )?;
    eprintln!("wrote {} (velocity {} values)", out.display(), want.len());
    Ok(())
}
