// Inspect maya1's generated SNAC codes to localize the babble (LM vs SNAC decode).
//   RLX_MAYA1_GGUF=... ORPHEUS_SNAC_PATH=... cargo run -p rlx-maya1 --example maya1_probe
use rlx_maya1::maya1_config;
use rlx_orpheus::{OrpheusTts, build_prompt_ids};
use rlx_runtime::Device;
use std::collections::HashSet;

const DESC: &str = "Realistic female voice in her 20s with a British accent. Normal pitch, warm timbre, conversational pacing.";
const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn main() -> anyhow::Result<()> {
    let gguf = std::env::var("RLX_MAYA1_GGUF")?;
    let snac = std::env::var("ORPHEUS_SNAC_PATH")?;
    let dev = match std::env::var("RLX_DEV").as_deref() {
        Ok("metal") => Device::Metal,
        Ok("mlx") => Device::Mlx,
        Ok("gpu") => Device::Gpu,
        _ => Device::Cpu,
    };
    eprintln!("[maya1] device={dev:?}");
    let mut tts = OrpheusTts::load_on(
        std::path::Path::new(&gguf),
        std::path::Path::new(&snac),
        dev,
    )?;
    tts.config = maya1_config();
    tts.config.max_new_tokens = std::env::var("MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1200);
    let body = format!("<description=\"{DESC}\"> {TEXT}");
    let prompt_ids = build_prompt_ids(tts.backbone.weights_path(), &body)?;
    eprintln!(
        "prompt_ids n={} first24={:?} last8={:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(24)],
        &prompt_ids[prompt_ids.len().saturating_sub(8)..]
    );
    let res = tts.synthesize_from_prompt_ids(&prompt_ids)?;
    let c = &res.codes;
    let uniq: HashSet<i32> = c.iter().copied().collect();
    eprintln!(
        "codes n={} unique={} min={:?} max={:?}",
        c.len(),
        uniq.len(),
        c.iter().min(),
        c.iter().max()
    );
    eprintln!("first 35 codes: {:?}", &c[..c.len().min(35)]);
    // per-slot code layout (7 tokens/frame): decode which slot each falls in
    eprintln!(
        "samples={} peak={:.4}",
        res.samples.len(),
        res.samples.iter().fold(0f32, |m, &x| m.max(x.abs()))
    );
    if let Ok(out) = std::env::var("WAV_OUT") {
        let mut w = hound::WavWriter::create(
            &out,
            hound::WavSpec {
                channels: 1,
                sample_rate: 24000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for &s in &res.samples {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        eprintln!("wrote {out}");
    }
    Ok(())
}
