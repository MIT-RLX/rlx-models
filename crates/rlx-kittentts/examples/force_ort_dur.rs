use rlx_kittentts::{Device, KittenTTS, assets};
fn main() -> anyhow::Result<()> {
    let weights = assets::default_native_weights_dir().expect("weights");
    let voices = assets::default_model_dir()
        .ok()
        .and_then(|d| assets::ModelLayout::resolve(&d).ok())
        .map(|l| l.voices)
        .expect("voices");
    let tts = KittenTTS::load_native(
        &weights, &voices, Default::default(), Default::default(), Device::Cpu, 128, 48_000,
    )?;
    let ids = rlx_kittentts::tokenize::ipa_to_ids("həˈloʊ");
    eprintln!("ids={ids:?}");
    let ort_dur = [3i64, 2, 2, 3, 4, 4, 13, 2, 1];
    let style = tts.style_for_voice_index("Jasper", 6)?;
    // Access engine via generate path - need infer with duration
    // Use public API if any; else synthesize via generate and compare
    let audio = tts.generate_from_ipa("həˈloʊ", "Jasper", 1.0, 6)?;
    let peak = audio.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let mut zc=0; for w in audio.windows(2) { if (w[0]>=0.)!=(w[1]>=0.) {zc+=1;} }
    eprintln!("native free-run: n={} peak={peak:.4} zc={zc}", audio.len());
    Ok(())
}
