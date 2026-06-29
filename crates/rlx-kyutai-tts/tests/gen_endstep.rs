//! Print end_step and frame counts (env-gated).

use rlx_kyutai_tts::checkpoint::KyutaiTtsCheckpoint;
use rlx_kyutai_tts::config::KyutaiTtsConfig;
use rlx_kyutai_tts::download::{
    default_kyutai_tts_dir, default_voices_dir, ensure_voice_embedding, tokenizer_path,
};
use rlx_kyutai_tts::generate::GenerateConfig;
use rlx_kyutai_tts::model::{KyutaiTtsModel, load_voice_speaker_wavs};
use rlx_kyutai_tts::tokenizer::KyutaiTokenizer;
use rlx_runtime::Device;
use std::path::PathBuf;

#[test]
fn print_generation_trace() {
    let dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_kyutai_tts_dir());
    if !dir.join("dsm_tts_1e68beda@240.safetensors").is_file() {
        eprintln!("skip: missing weights");
        return;
    }
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let tok = KyutaiTokenizer::load(tokenizer_path(&dir)).expect("tok");
    let prompt = "Hello world, this is a test of the Kyutai text to speech system.";
    let mut m = KyutaiTtsModel::open(&dir, cfg.clone(), Device::Cpu).expect("model");
    let voice = ensure_voice_embedding(
        &default_voices_dir(),
        KyutaiTtsCheckpoint::V1_6bEnFr,
        "alba-mackenna/casual.wav",
    )
    .expect("voice");
    let spk = load_voice_speaker_wavs(&voice).expect("spk");
    let gen_cfg = GenerateConfig {
        max_steps: 200,
        n_q: 32,
        cfg_alpha: 2.0,
        text_temperature: 0.0,
        audio_temperature: 0.0,
        seed: 42,
    };
    m.reset_state();
    m.set_generation_conditions(gen_cfg.cfg_alpha, Some(&spk))
        .expect("cond");
    let mut state =
        rlx_kyutai_tts::generate::GenerateState::new(&cfg, &tok, prompt, gen_cfg).expect("st");
    while state.step_idx() < 200 {
        if let Some(e) = state.end_step() {
            if state.step_idx() >= e + cfg.audio_delay_frames() + 4 {
                break;
            }
        }
        state.step(&mut m).expect("step");
    }
    let frames = rlx_kyutai_tts::generate::GenerateState::trim_for_mimi(
        state.raw_lm_frames(),
        state.end_step(),
        cfg.audio_delay_frames(),
        32,
        cfg.audio_pad_token(),
    );
    let end = state.end_step();
    let raw = state.raw_lm_frames();
    eprintln!("raw lm_frames len={}", raw.len());
    for idx in [16usize, 17, 18, 19, 20] {
        if let Some(f) = raw.get(idx) {
            eprintln!("  lm_frames[{idx}] {:?}", &f[..8.min(f.len())]);
        }
    }
    eprintln!(
        "end_step={end:?} trimmed={} transcript={:?}",
        frames.len(),
        state.transcript(),
    );
    for (i, f) in frames.iter().enumerate().take(8) {
        eprintln!("  [{i}] {:?}", &f[..4.min(f.len())]);
    }
}

#[test]
fn print_long_prompt_entries() {
    let dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| default_kyutai_tts_dir());
    if !dir.join("dsm_tts_1e68beda@240.safetensors").is_file() {
        return;
    }
    let tok = KyutaiTokenizer::load(tokenizer_path(&dir)).expect("tok");
    let prompt = "Hello world, this is a test of the Kyutai text to speech system.";
    let entries = rlx_kyutai_tts::state_machine::script_to_entries(&tok, prompt).expect("e");
    eprintln!("rust entries {}", entries.len());
    for e in &entries {
        eprintln!("  {:?} tokens={:?} pad={}", e.text, e.tokens, e.padding);
    }
}
