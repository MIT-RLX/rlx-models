// Quick native multi-backend smoke test for supertonic (no ort).
use rlx_runtime::Device;
use rlx_supertonic::{InferOpts, Supertonic, Voice};
fn main() -> anyhow::Result<()> {
    let dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/supertonic-3");
    let dev = match std::env::var("RLX_DEV").as_deref() {
        Ok("metal") => Device::Metal,
        Ok("mlx") => Device::Mlx,
        Ok("gpu") => Device::Gpu,
        Ok("ane") | Ok("coreml") => Device::Ane,
        _ => Device::Cpu,
    };
    let tts = Supertonic::load_on(&dir, dev)?;
    let voice = Voice::load(&dir.join("voice_styles/F1.json"))?;
    let opts = InferOpts {
        total_step: 4,
        speed: 1.05,
        seed: 42,
    };
    let audio = tts.synthesize("Hello from Supertonic on RLX.", "en", &voice, &opts)?;
    let peak = rlx_supertonic::peak_amplitude(&audio);
    eprintln!(
        "[{dev:?}] samples={} peak={peak:.3} {}",
        audio.len(),
        if peak > 0.01 {
            "AUDIBLE ✅"
        } else {
            "SILENT ❌"
        }
    );
    Ok(())
}
