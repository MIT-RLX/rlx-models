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

//! vLLM-Omni (Docker) → Rust codec decode parity for Voxtral-4B-TTS.
//!
//! Requires:
//!   RLX_VOXTRAL_TTS_DIR   model checkout with consolidated.safetensors
//!   RLX_VOXTRAL_TTS_PARITY=1
//!   Docker + GPU image `rlx-voxtral-tts-ref:gpu` (see docker/voxtral-tts/)
//!
//! Run:
//!   just test-voxtral-tts-parity
//!
//! Native stage-0 (LM + acoustic) on real weights: `just test-voxtral-tts-native-parity`

use rlx_voxtral_tts::{VoxtralTtsRunner, parse_codes_file};
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

fn export_vllm_codes(model_dir: &Path, out_codes: &Path) {
    let root = repo_root();
    let script = root.join("docker/voxtral-tts/run-ref.sh");
    assert!(script.is_file(), "missing {}", script.display());

    let status = Command::new("bash")
        .arg(&script)
        .arg("export-codes")
        .env("RLX_VOXTRAL_TTS_DIR", model_dir)
        .env("RLX_VOXTRAL_TTS_TEXT", "Hello from RLX parity.")
        .env("RLX_VOXTRAL_TTS_VOICE", "neutral_female")
        .env("RLX_VOXTRAL_TTS_OUT_CODES", out_codes)
        .status()
        .expect("run-ref.sh export-codes");
    assert!(
        status.success(),
        "vLLM docker export-codes failed (status {status})"
    );
}

#[test]
fn voxtral_tts_codec_decode_vllm_codes() {
    let Some(model_dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR");
        return;
    };
    if std::env::var("RLX_VOXTRAL_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_VOXTRAL_TTS_PARITY=1");
        return;
    }
    if !docker_ok() {
        eprintln!("skip: docker not available");
        return;
    }

    let out_dir = repo_root().join(".cache/voxtral/tts");
    std::fs::create_dir_all(&out_dir).expect("mkdir cache");
    let codes_path = out_dir.join("vllm_codes_parity.txt");

    export_vllm_codes(&model_dir, &codes_path);

    let runner = VoxtralTtsRunner::open(&model_dir).expect("open runner");
    let (codes, n_frames) = parse_codes_file(&codes_path).expect("parse codes");
    assert_eq!(codes.len(), n_frames * 37, "flat code length");
    assert!(n_frames > 0, "expected at least one frame");

    let pcm = runner
        .decode_codes_to_pcm(&codes, n_frames)
        .expect("rust codec decode");
    assert!(
        pcm.len() > runner.config().audio_config.codec_args.sampling_rate / 10,
        "PCM too short: {} samples",
        pcm.len()
    );
}
