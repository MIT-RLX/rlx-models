//! Compare two/three WAVs by best-lag correlation (handles f32 & i16).
//! ```bash
//! cargo run -p rlx-tsac --example cmp_wavs --release --features native-codec -- a.wav b.wav [c.wav]
//! ```
use anyhow::Result;
use rlx_tsac::audio;

fn best_corr(a: &[f32], b: &[f32]) -> (i64, f32) {
    let (mut bl, mut bc) = (0i64, -2.0f32);
    for lag in -2000i64..=2000 {
        let (x, y): (&[f32], &[f32]) = if lag >= 0 {
            (&a[lag as usize..], b)
        } else {
            (a, &b[(-lag) as usize..])
        };
        let m = x.len().min(y.len());
        if m < 1000 {
            continue;
        }
        let c = audio::correlation(&x[..m], &y[..m]);
        if c > bc {
            bc = c;
            bl = lag;
        }
    }
    (bl, bc)
}

fn main() -> Result<()> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let pcms: Vec<(String, Vec<f32>)> = paths
        .iter()
        .map(|p| {
            Ok((
                p.clone(),
                audio::load_pcm_from_wav(std::path::Path::new(p))?,
            ))
        })
        .collect::<Result<_>>()?;
    for (p, v) in &pcms {
        eprintln!("{p}: {} samples", v.len());
    }
    for i in 0..pcms.len() {
        for j in (i + 1)..pcms.len() {
            let (lag, c) = best_corr(&pcms[i].1, &pcms[j].1);
            eprintln!(
                "{} vs {}: best-lag corr = {c:.4} (lag {lag})",
                pcms[i].0, pcms[j].0
            );
        }
    }
    Ok(())
}
