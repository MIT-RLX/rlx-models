#![cfg(all(feature = "native", feature = "metal", feature = "mlx"))]
mod support;
use rlx_kittentts::{Device, KittenTTS, assets, infer_opts};
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};
use support::{resample_linear, whisper_asr_dir};

fn synth(dev: Device, ipa: &str, weights: &std::path::Path, voices: &std::path::Path) -> Vec<f32> {
    let ids = rlx_kittentts::ipa_to_ids(ipa);
    let (seq, wave) = infer_opts::recommended_native_compile_opts(ids.len());
    let tts = KittenTTS::load_native(
        weights,
        voices,
        Default::default(),
        Default::default(),
        dev,
        seq,
        wave,
    )
    .expect("load");
    let voice = tts
        .voice_names()
        .iter()
        .find(|n| n.contains("expr-voice-2-m"))
        .cloned()
        .or_else(|| tts.voice_names().first().cloned())
        .unwrap_or_default();
    tts.generate_from_ipa(ipa, &voice, 1.0, 6).expect("gen")
}

#[test]
fn metal_then_mlx_whisper() {
    support::setup_native_smoke_env();
    unsafe {
        std::env::set_var("KITTEN_RLX_DEBUG_DURATION", "1");
    }
    let weights = assets::default_native_weights_dir().expect("weights");
    let voices = assets::default_model_dir()
        .ok()
        .and_then(|d| assets::ModelLayout::resolve(&d).ok())
        .map(|l| l.voices)
        .expect("voices");
    let ipa = "həˈloʊ";
    let metal = synth(Device::Metal, ipa, &weights, &voices);
    let mlx = synth(Device::Mlx, ipa, &weights, &voices);
    eprintln!(
        "metal peak={} n={} mlx peak={} n={}",
        metal.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
        metal.len(),
        mlx.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
        mlx.len()
    );
    let Some(dir) = whisper_asr_dir() else {
        return;
    };
    let mut w = WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .unwrap();
    for (name, pcm) in [("metal", &metal[..]), ("mlx", &mlx[..])] {
        let pcm16 = resample_linear(pcm, 24000, WHISPER_RATE as u32);
        let t = w.transcribe_greedy(&pcm16).unwrap_or_default();
        eprintln!("{name} => {:?}", t.trim());
    }
}
