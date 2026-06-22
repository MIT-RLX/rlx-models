// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Export native + ONNX WAVs for every phrase fixture (manual listening checks).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_kittentts::{
    Device, KittenTTS, assets, audio_util, infer_opts, ipa_style_index, peak_amplitude,
    phrase_fixtures::{PhraseCase, all_export_phrases},
};
use rlx_runtime::is_available;
use serde::Serialize;

fn parity_env() {
    unsafe {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let aot =
            std::env::temp_dir().join(format!("kitten_export_aot_{}_{nonce}", std::process::id()));
        std::env::set_var("KITTEN_RLX_AOT_CACHE", &aot);
        std::env::set_var("KITTENTTS_ORT_INTRA_THREADS", "1");
        std::env::set_var("KITTEN_RLX_INFER", "parity");
        std::env::set_var("KITTEN_RLX_PARITY", "1");
        std::env::set_var("KITTEN_RLX_FULL_GRAPH", "1");
        std::env::set_var("KITTEN_RLX_SINGLE_PASS", "1");
        std::env::remove_var("KITTEN_RLX_RNG_SEED");
        std::env::set_var("KITTENTTS_TAIL_TRIM", "0");
        std::env::remove_var("KITTEN_RLX_COMPILE_HEADROOM");
        std::env::remove_var("KITTEN_RLX_ORT_DURATION_CARRY");
        std::env::remove_var("KITTEN_RLX_NO_ORT_WAVEFORM_FALLBACK");
        std::env::remove_var("KITTEN_RLX_GRAPH_CACHE");
        std::env::remove_var("KITTEN_RLX_BUNDLE");
        std::env::remove_var("RLX_ONNX_BUNDLE");
        std::env::remove_var("KITTEN_RLX_WEIGHTS");
    }
}

fn export_devices() -> Vec<Device> {
    if let Ok(raw) = std::env::var("KITTEN_EXPORT_DEVICES") {
        return raw
            .split(',')
            .filter_map(|s| match s.trim().to_ascii_lowercase().as_str() {
                "cpu" => Some(Device::Cpu),
                "metal" if is_available(Device::Metal) => Some(Device::Metal),
                _ => None,
            })
            .collect();
    }
    vec![Device::Cpu]
}

#[derive(Serialize)]
struct PhraseReport {
    label: String,
    ipa: String,
    device: String,
    native_len: usize,
    native_peak: f32,
    onnx_len: usize,
    onnx_peak: f32,
    aligned_len: usize,
    ort_peak_diff: f32,
    aligned_peak: f32,
}

fn slug(label: &str) -> String {
    label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn best_ort_match(
    ort: &KittenTTS,
    phrase: &PhraseCase,
    voice: &str,
    native_audio: &[f32],
) -> (Vec<f32>, Vec<f32>, f32) {
    let style = ipa_style_index(phrase.ipa);
    let mut best_ort = Vec::new();
    let mut best_aligned = native_audio.to_vec();
    let mut best_peak = f32::MAX;
    let native_norm = audio_util::normalize_for_compare(native_audio);
    for _ in 0..phrase.ort_candidates.min(8) {
        let ort_audio = ort
            .generate_from_ipa(phrase.ipa, voice, 1.0, style)
            .expect("onnx");
        let ort_norm = audio_util::normalize_for_compare(&ort_audio);
        let lag = audio_util::effective_max_lag(phrase.max_lag, ort_norm.len(), native_norm.len());
        let aligned = audio_util::align_to_reference(&ort_audio, native_audio, lag);
        let (_, peak) = audio_util::max_abs_best_lag(&ort_norm, &native_norm, lag);
        if peak < best_peak {
            best_peak = peak;
            best_ort = ort_audio;
            best_aligned = aligned;
        }
        if peak <= 0.10 {
            break;
        }
    }
    (best_ort, best_aligned, best_peak)
}

fn load_native(
    layout: &assets::ModelLayout,
    weights: &Path,
    device: Device,
    token_len: usize,
) -> Result<KittenTTS> {
    let (seq_len, max_wave) = infer_opts::recommended_native_compile_opts(token_len);
    KittenTTS::load_native_with_ort(
        weights,
        &layout.voices,
        layout.config.speed_priors.clone(),
        layout.config.voice_aliases.clone(),
        device,
        seq_len,
        max_wave,
        Some(&layout.onnx),
    )
}

fn main() -> Result<()> {
    parity_env();
    let out_dir = std::env::var("KITTEN_PHRASE_OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/kitten_phrases"));
    fs::create_dir_all(&out_dir).with_context(|| format!("mkdir {}", out_dir.display()))?;

    let model_dir = assets::default_model_dir().context("run `just fetch-kittentts`")?;
    let layout = assets::ModelLayout::resolve(&model_dir)?;
    let weights = assets::default_native_weights_dir().context("native weights missing")?;
    let weights = weights.canonicalize().unwrap_or(weights);

    let ort = KittenTTS::load_on(
        &layout.onnx,
        &layout.voices,
        layout.config.speed_priors.clone(),
        layout.config.voice_aliases.clone(),
        Device::Cpu,
    )?;
    let default_voice = ort.voice_names().first().context("voice")?.clone();

    let mut reports = Vec::new();
    for device in export_devices() {
        if device == Device::Cpu {
            unsafe {
                std::env::set_var("KITTEN_RLX_PREFER_METAL", "0");
            }
        } else {
            unsafe {
                std::env::remove_var("KITTEN_RLX_PREFER_METAL");
            }
        }

        let mut native_cache: HashMap<usize, KittenTTS> = HashMap::new();
        for phrase in all_export_phrases() {
            let voice = phrase.voice.unwrap_or(default_voice.as_str());
            let token_len = rlx_kittentts::ipa_to_ids(phrase.ipa).len();
            let (seq_len, _) = infer_opts::recommended_native_compile_opts(token_len);
            eprintln!("export [{}] on {device:?} (seq={seq_len}) …", phrase.label);

            let native = match native_cache.entry(seq_len) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    match load_native(&layout, &weights, device, token_len) {
                        Ok(tts) => e.insert(tts),
                        Err(err) => {
                            eprintln!("  skip: native load failed: {err}");
                            continue;
                        }
                    }
                }
            };

            let style = ipa_style_index(phrase.ipa);
            let native_audio = match native.generate_from_ipa(phrase.ipa, voice, 1.0, style) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("  failed infer: {e}");
                    continue;
                }
            };

            let (ort_audio, aligned, ort_peak_diff) =
                best_ort_match(&ort, phrase, voice, &native_audio);
            let ort_peak = peak_amplitude(&ort_audio);
            let aligned_for_listen = audio_util::scale_to_peak(&aligned, ort_peak.max(0.25));

            let stem = format!("{}_{:?}", slug(phrase.label), device).to_lowercase();
            native.write_wav(&native_audio, &out_dir.join(format!("{stem}_native.wav")))?;
            native.write_wav(&ort_audio, &out_dir.join(format!("{stem}_onnx.wav")))?;
            native.write_wav(
                &aligned_for_listen,
                &out_dir.join(format!("{stem}_aligned.wav")),
            )?;

            eprintln!(
                "  native peak={:.4} onnx peak={:.4} diff={:.4}",
                peak_amplitude(&native_audio),
                ort_peak,
                ort_peak_diff
            );
            reports.push(PhraseReport {
                label: phrase.label.to_string(),
                ipa: phrase.ipa.to_string(),
                device: format!("{device:?}"),
                native_len: native_audio.len(),
                native_peak: peak_amplitude(&native_audio),
                onnx_len: ort_audio.len(),
                onnx_peak: ort_peak,
                aligned_len: aligned.len(),
                ort_peak_diff,
                aligned_peak: peak_amplitude(&aligned_for_listen),
            });
        }
    }

    let manifest_path = out_dir.join("manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&reports).context("manifest json")?,
    )?;
    eprintln!(
        "wrote {} WAV sets under {}",
        reports.len(),
        out_dir.display()
    );
    eprintln!("manifest: {}", manifest_path.display());
    Ok(())
}
