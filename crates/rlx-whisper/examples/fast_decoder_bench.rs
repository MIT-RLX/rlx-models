//! Time `FastDecoder::step` (5 beams) at a few cache lengths. usage: fast_decoder_bench MODEL.safetensors
use rlx_whisper::fast_decoder::{FastDecoder, SelfCache};
fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1).unwrap();
    let d = FastDecoder::load(std::path::Path::new(&p), 20)?;
    let ck: Vec<Vec<f32>> = (0..d.num_layers())
        .map(|_| {
            (0..1500 * d.d)
                .map(|i| ((i % 97) as f32 - 48.0) * 1e-3)
                .collect()
        })
        .collect();
    let cross = d.prepare_cross(&ck, &ck);
    let mut cache = SelfCache::new(d.num_layers(), 5);
    for _ in 0..3 {
        d.step(&[50258; 5], &mut cache, &cross);
    }
    for n in 0..40 {
        let t = std::time::Instant::now();
        d.step(&[440; 5], &mut cache, &cross);
        if n % 10 == 0 {
            eprintln!(
                "len {} step {:.1} ms",
                cache.len,
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    let t = std::time::Instant::now();
    cache.reorder(&[1, 0, 2, 2, 4]);
    eprintln!("reorder {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    Ok(())
}
