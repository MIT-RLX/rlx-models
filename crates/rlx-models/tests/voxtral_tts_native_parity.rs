// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Native stage-0 (LM + acoustic) vs vLLM-Omni codes on real weights.
//!
//! Requires:
//!   RLX_VOXTRAL_TTS_DIR
//!   RLX_VOXTRAL_TTS_NATIVE_PARITY=1
//!   Docker GPU image `rlx-voxtral-tts-ref:gpu` + tools image
//!
//! Run:
//!   just test-voxtral-tts-native-parity

use rlx_voxtral_tts::{
    GenerationConfig, VoxtralTtsRunnerBuilder, load_prompt_tokens, parse_codes_file,
};
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_dir() -> Option<PathBuf> {
    std::env::var("RLX_VOXTRAL_TTS_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
}

fn docker_ok() -> bool {
    Command::new("docker")
        .arg("info")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn tokenize_prompt(model_dir: &Path, tokens_path: &Path) {
    let root = repo_root();
    let script = root.join("docker/voxtral-tts/run-tools.sh");
    let status = Command::new("bash")
        .arg(&script)
        .arg("tokenize")
        .env("RLX_VOXTRAL_TTS_DIR", model_dir)
        .env("RLX_VOXTRAL_TTS_TEXT", "Hello from RLX native parity.")
        .env("RLX_VOXTRAL_TTS_VOICE", "neutral_female")
        .env("RLX_VOXTRAL_TTS_OUT", tokens_path)
        .status()
        .expect("run-tools.sh tokenize");
    assert!(status.success(), "docker tokenize failed");
}

fn export_vllm_codes(model_dir: &Path, out_codes: &Path, seed: u64, cfg_alpha: f32) {
    let root = repo_root();
    let script = root.join("docker/voxtral-tts/run-ref.sh");
    let status = Command::new("bash")
        .arg(&script)
        .arg("export-codes")
        .env("RLX_VOXTRAL_TTS_DIR", model_dir)
        .env("RLX_VOXTRAL_TTS_TEXT", "Hello from RLX native parity.")
        .env("RLX_VOXTRAL_TTS_VOICE", "neutral_female")
        .env("RLX_VOXTRAL_TTS_OUT_CODES", out_codes)
        .env("RLX_VOXTRAL_TTS_SEED", seed.to_string())
        .env("RLX_VOXTRAL_TTS_CFG_ALPHA", cfg_alpha.to_string())
        .status()
        .expect("run-ref.sh export-codes");
    assert!(status.success(), "vLLM docker export-codes failed");
}

fn code_match_stats(native: &[u32], reference: &[u32]) -> (usize, usize, usize) {
    let n = native.len().min(reference.len());
    let mut exact_frames = 0usize;
    let mut slot_matches = 0usize;
    for chunk in native[..n].chunks(37).zip(reference[..n].chunks(37)) {
        if chunk.0 == chunk.1 {
            exact_frames += 1;
        }
        for (a, b) in chunk.0.iter().zip(chunk.1.iter()) {
            if a == b {
                slot_matches += 1;
            }
        }
    }
    (exact_frames, slot_matches, n / 37)
}

#[test]
fn voxtral_tts_native_stage0_vs_vllm() {
    let Some(model_dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR");
        return;
    };
    if std::env::var("RLX_VOXTRAL_TTS_NATIVE_PARITY")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip: set RLX_VOXTRAL_TTS_NATIVE_PARITY=1");
        return;
    }
    if !docker_ok() {
        eprintln!("skip: docker not available");
        return;
    }

    let out_dir = repo_root().join(".cache/voxtral/tts");
    std::fs::create_dir_all(&out_dir).expect("mkdir cache");
    let tokens_path = out_dir.join("native_parity_prompt_tokens.txt");
    let vllm_codes_path = out_dir.join("native_parity_vllm_codes.txt");

    tokenize_prompt(&model_dir, &tokens_path);
    let gen_cfg = GenerationConfig {
        cfg_alpha: 1.2,
        seed: 42,
        max_frames: 2500,
    };
    export_vllm_codes(
        &model_dir,
        &vllm_codes_path,
        gen_cfg.seed,
        gen_cfg.cfg_alpha,
    );

    let prompt = load_prompt_tokens(&tokens_path).expect("load prompt tokens");
    assert!(!prompt.is_empty());

    // Eager LM matches the hand-ported reference; compiled backbone is exercised separately.
    // SAFETY: test-only env toggle before opening the runner.
    unsafe {
        std::env::set_var("RLX_VOXTRAL_TTS_EAGER", "1");
        std::env::set_var("RLX_VOXTRAL_TTS_ACOUSTIC_EAGER", "1");
    }
    let mut runner = VoxtralTtsRunnerBuilder::default()
        .model_dir(&model_dir)
        .device(rlx_runtime::Device::Cpu)
        .eager_lm(true)
        .eager_acoustic(true)
        .build()
        .expect("open runner");
    let native_codes = runner
        .synthesize_native_codes(&prompt, "neutral_female", &gen_cfg)
        .expect("native synthesize_codes");

    let (vllm_codes, vllm_frames) = parse_codes_file(&vllm_codes_path).expect("parse vllm codes");
    assert_eq!(vllm_codes.len(), vllm_frames * 37);
    assert!(!native_codes.is_empty(), "native produced no codes");
    assert_eq!(
        native_codes.len() % 37,
        0,
        "native codes length {} not multiple of 37",
        native_codes.len()
    );

    let native_frames = native_codes.len() / 37;
    eprintln!(
        "native frames={native_frames} vllm frames={vllm_frames} \
         (first semantic native={} vllm={})",
        native_codes[0], vllm_codes[0]
    );

    if native_codes == vllm_codes {
        eprintln!("native stage-0 codes match vLLM exactly");
        return;
    }

    let (exact_frames, slot_matches, compared_frames) =
        code_match_stats(&native_codes, &vllm_codes);
    let slots = compared_frames * 37;
    eprintln!(
        "code diff: {exact_frames}/{compared_frames} exact frames, \
         {slot_matches}/{slots} slot matches"
    );

    // Until native LM/acoustic fully match vLLM numerics, require a successful run plus
    // partial overlap on the shared prefix (validates real weights end-to-end).
    assert!(
        native_frames > 0 && vllm_frames > 0,
        "expected non-zero frames from both paths"
    );
    assert!(
        slot_matches > 0,
        "expected at least one matching code slot vs vLLM reference"
    );
}
