// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Parity of the rlx-graph engine against upstream `pocket_tts`.
//!
//! Replays a dump from `scripts/pocket_tts_reference.py` — same text, voice and
//! per-frame sampler noise — and compares tokens, every frame's backbone output,
//! EOS logit and latent, the frame count, and the decoded audio.
//!
//! ```text
//! .venv-pocket/bin/python -I scripts/pocket_tts_reference.py \
//!     --lang-dir weights/tts/pocket-tts-kyutai/languages/english_2026-10 \
//!     --language english_2026-10 --voice alba \
//!     --out weights/tts/pocket-tts-kyutai/reference/english_2026-10-alba
//! RLX_POCKET_TTS_ROOT=weights/tts/pocket-tts-kyutai \
//! RLX_POCKET_TTS_REFERENCE=weights/tts/pocket-tts-kyutai/reference/english_2026-10-alba \
//!     cargo test -p rlx-pocket-tts --release --test reference_parity -- --nocapture
//! ```
//!
//! `RLX_POCKET_TTS_DEVICE` picks the device (default `cpu`). Skips without the env.

use std::path::PathBuf;

use anyhow::{Context, Result};
use rlx_pocket_tts::PocketTts;
use rlx_pocket_tts::weights::WeightFile;
use safetensors::tensor::Dtype;

fn env_path(key: &str) -> Option<PathBuf> {
    rlx_ir::env::var_result(key).ok().map(PathBuf::from)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

fn i64s(wf: &WeightFile, name: &str) -> Result<Vec<i64>> {
    let t = wf.tensor(name)?;
    anyhow::ensure!(t.dtype() == Dtype::I64, "{name}: dtype {:?}", t.dtype());
    Ok(t.data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect())
}

#[test]
fn matches_upstream_reference() -> Result<()> {
    let (Some(root), Some(refdir)) = (
        env_path("RLX_POCKET_TTS_ROOT"),
        env_path("RLX_POCKET_TTS_REFERENCE"),
    ) else {
        eprintln!("skip: set RLX_POCKET_TTS_ROOT and RLX_POCKET_TTS_REFERENCE");
        return Ok(());
    };
    let device = rlx_ir::env::var_result("RLX_POCKET_TTS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = rlx_runtime::parse_device(&device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refdir.join("reference.json"))?)?;
    let reference = WeightFile::open(refdir.join("reference.safetensors"))?;
    let language = meta["language"].as_str().context("language")?;
    let voice_name = meta["voice"].as_str().context("voice")?;

    let mut tts = PocketTts::open(&root, language, device)?;
    tts.options.eos_threshold = meta["eos_threshold"].as_f64().context("eos")? as f32;
    let voice = tts.load_voice(voice_name)?;

    let chunks = meta["chunks"].as_array().context("chunks")?;
    for (ci, chunk) in chunks.iter().enumerate() {
        let p = format!("chunk{ci}.");
        let text = chunk["text"].as_str().context("text")?;
        let want_tokens = i64s(&reference, &format!("{p}tokens"))?;
        let tokens: Vec<i64> = tts
            .tokenizer()
            .encode(text)?
            .into_iter()
            .map(i64::from)
            .collect();
        assert_eq!(tokens, want_tokens, "chunk {ci}: tokens");
        assert_eq!(voice.offset as u64, chunk["voice_offset"].as_u64().unwrap());

        let (noise, nshape) = reference.get_f32(&format!("{p}noise"))?;
        let ldim = nshape[1];
        let mut rows = noise.chunks(ldim).map(<[f32]>::to_vec);
        let fae = chunk["frames_after_eos"].as_u64().unwrap() as usize;
        let out = tts.generate_chunk(text, &voice, fae, &mut || {
            rows.next().expect("ran past the reference noise")
        })?;

        // Teacher forced: every frame gets the reference's own previous latent,
        // so per-frame error cannot compound. This is the strict check.
        let (want_lat_tf, _) = reference.get_f32(&format!("{p}latents"))?;
        let mut rows_tf = noise.chunks(ldim).map(<[f32]>::to_vec);
        let tf = tts.generate_chunk_forced(
            text,
            &voice,
            fae,
            &mut || rows_tf.next().expect("ran past the reference noise"),
            &want_lat_tf,
        )?;
        let (want_hidden, hshape) = reference.get_f32(&format!("{p}hidden"))?;
        let (want_eos_tf, _) = reference.get_f32(&format!("{p}eos_logit"))?;
        let hd_tf = hshape[1];
        let (mut tf_hidden, mut tf_eos, mut tf_lat) = (0f32, 0f32, 0f32);
        for f in 0..tf.eos_logits.len().min(want_eos_tf.len()) {
            tf_hidden = tf_hidden.max(max_abs(
                &tf.hidden[f],
                &want_hidden[f * hd_tf..(f + 1) * hd_tf],
            ));
            tf_eos = tf_eos.max((tf.eos_logits[f] - want_eos_tf[f]).abs());
        }
        for f in 0..(tf.latents.len() / ldim).min(want_lat_tf.len() / ldim) {
            tf_lat = tf_lat.max(max_abs(
                &tf.latents[f * ldim..(f + 1) * ldim],
                &want_lat_tf[f * ldim..(f + 1) * ldim],
            ));
        }
        eprintln!(
            "chunk {ci} teacher-forced: {} frames, hidden max|Δ| {tf_hidden:.2e}, eos max|Δ| {tf_eos:.2e}, \
             latent max|Δ| {tf_lat:.2e}",
            tf.eos_logits.len()
        );
        assert!(
            tf_hidden < 1e-3,
            "teacher-forced hidden mismatch {tf_hidden}"
        );
        assert!(tf_eos < 1e-3, "teacher-forced eos mismatch {tf_eos}");
        assert!(tf_lat < 1e-2, "teacher-forced latent mismatch {tf_lat}");

        let (want_eos, _) = reference.get_f32(&format!("{p}eos_logit"))?;
        let (want_lat, _) = reference.get_f32(&format!("{p}latents"))?;
        let (want_audio, _) = reference.get_f32(&format!("{p}audio_raw"))?;
        let hd = hshape[1];
        let frames = want_lat.len() / ldim;
        let mut worst_hidden = 0f32;
        let mut worst_eos = 0f32;
        let mut worst_lat = 0f32;
        for f in 0..out.eos_logits.len().min(want_eos.len()) {
            worst_hidden =
                worst_hidden.max(max_abs(&out.hidden[f], &want_hidden[f * hd..(f + 1) * hd]));
            worst_eos = worst_eos.max((out.eos_logits[f] - want_eos[f]).abs());
        }
        if rlx_ir::env::flag("RLX_POCKET_TTS_VERBOSE") {
            for f in 0..out.eos_logits.len().min(want_eos.len()) {
                let lat = if (f + 1) * ldim <= out.latents.len().min(want_lat.len()) {
                    max_abs(
                        &out.latents[f * ldim..(f + 1) * ldim],
                        &want_lat[f * ldim..(f + 1) * ldim],
                    )
                } else {
                    f32::NAN
                };
                eprintln!(
                    "  frame {f:3}: hidden {:.2e} eos {:+.3} vs {:+.3} latent {lat:.2e}",
                    max_abs(&out.hidden[f], &want_hidden[f * hd..(f + 1) * hd]),
                    out.eos_logits[f],
                    want_eos[f]
                );
            }
        }
        for f in 0..(out.latents.len() / ldim).min(frames) {
            worst_lat = worst_lat.max(max_abs(
                &out.latents[f * ldim..(f + 1) * ldim],
                &want_lat[f * ldim..(f + 1) * ldim],
            ));
        }
        // Decoder alone: the reference latents through the rlx Mimi graph.
        let dec = tts.decode_latents(&want_lat)?;
        let dec_cos = cosine(&dec, &want_audio);
        let dec_max = max_abs(&dec, &want_audio);
        let first_hidden = max_abs(&out.hidden[0], &want_hidden[..hd]);
        eprintln!(
            "chunk {ci}: frame-0 hidden max|Δ| {first_hidden:.2e}; decoder on reference latents: \
             cos {dec_cos:.7} max|Δ| {dec_max:.2e}"
        );
        assert!(dec_cos > 0.99999, "decoder cosine {dec_cos}");
        let n = out.audio_raw.len().min(want_audio.len());
        let audio_cos = cosine(&out.audio_raw[..n], &want_audio[..n]);
        let audio_max = max_abs(&out.audio_raw[..n], &want_audio[..n]);
        eprintln!(
            "chunk {ci} on {device:?}: frames {} (ref {frames}), hidden max|Δ| {worst_hidden:.2e}, \
             eos max|Δ| {worst_eos:.2e}, latent max|Δ| {worst_lat:.2e}, \
             audio cos {audio_cos:.6} max|Δ| {audio_max:.2e} ({} vs {} samples)",
            out.latents.len() / ldim,
            out.audio_raw.len(),
            want_audio.len()
        );
        // Free running, the sampler feeds back its own latents and amplifies
        // float noise (the one-step drifting head especially), so only the
        // outcome is checked: where it stopped, and that it is the same audio.
        assert_eq!(out.latents.len() / ldim, frames, "chunk {ci}: frame count");
        assert_eq!(out.audio_raw.len(), want_audio.len(), "audio length");
        assert!(audio_cos > 0.9, "audio cosine {audio_cos}");
    }
    Ok(())
}
