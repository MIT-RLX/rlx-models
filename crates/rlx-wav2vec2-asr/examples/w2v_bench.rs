//! Compile vs run time of `Wav2Vec2Ctc::emissions`. usage: w2v_bench PTH [metal]
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dev = if a.iter().any(|x| x == "metal") {
        rlx_runtime::Device::Metal
    } else {
        rlx_runtime::Device::Cpu
    };
    let mut m = rlx_wav2vec2_asr::model::Wav2Vec2Ctc::open(std::path::Path::new(&a[1]), dev)?;
    let pcm: Vec<f32> = (0..16000 * 20)
        .map(|i| ((i as f32) * 0.01).sin() * 0.1)
        .collect();
    for n in [pcm.len(), pcm.len(), pcm.len() - 4000] {
        let t = std::time::Instant::now();
        let (_, f) = m.emissions(&pcm[..n])?;
        eprintln!(
            "{n} samples -> {f} frames: {:.2}s",
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
