//! Time one denoiser stage of the web GGUF on a backend, built as a browser
//! worker builds it (header + the stage's byte ranges only).
//!
//! `cargo run -p rlx-flux1 --release --features gpu --example web_stage_profile -- GGUF INDEX COUNT [DEVICE]`
use anyhow::Result;
use std::io::{Read, Seek, SeekFrom};
use std::str::FromStr;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (path, index, count): (&str, usize, usize) = (&a[0], a[1].parse()?, a[2].parse()?);
    let device = rlx_runtime::Device::from_str(a.get(3).map_or("gpu", String::as_str))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut f = std::fs::File::open(path)?;
    let mut prefix = vec![0u8; 16 << 20];
    f.read_exact(&mut prefix)?;
    let header = rlx_gguf::GgufFile::header_from_bytes(&prefix)?;
    let cfg = rlx_flux1::web::config_from_meta(&header)?;
    let parts = rlx_flux1::web::stage_parts(&cfg, index, count)?;
    let keep = |n: &str| rlx_flux1::web::stage_keeps(&parts, n);
    let mut partial = rlx_gguf::GgufFile::header_from_bytes(&prefix)?.begin_partial(&keep)?;
    for r in header.tensor_ranges(keep, 1 << 20)? {
        let mut buf = vec![0u8; (r.end - r.start) as usize];
        f.seek(SeekFrom::Start(r.start))?;
        f.read_exact(&mut buf)?;
        partial.feed(r.start, &buf);
    }
    let geo = rlx_flux1::DitGeometry {
        txt_len: 128,
        grid_h: 10,
        grid_w: 10,
    };
    let mut replay = rlx_flux1::web::Replay::new(partial.finish()?);
    let t0 = std::time::Instant::now();
    let mut stage =
        rlx_flux1::web::Flux1DitStage::compile(&cfg, geo, &mut replay, device, parts.clone(), 1)?;
    eprintln!(
        "stage {index}/{count} {parts:?}: built in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    let d = cfg.hidden;
    let (x, txt) = if stage.is_first() {
        (
            vec![0.1f32; geo.img_len() * cfg.in_channels],
            vec![0.05f32; geo.txt_len * cfg.context_dim],
        )
    } else {
        (vec![0.1f32; geo.total() * d], Vec::new())
    };
    let vec = vec![0.01f32; d];
    for rep in 0..3 {
        let t = std::time::Instant::now();
        let out = stage.forward(&x, &txt, &vec)?;
        eprintln!(
            "run {rep}: {:.2}s ({} values)",
            t.elapsed().as_secs_f32(),
            out.len()
        );
    }
    Ok(())
}
