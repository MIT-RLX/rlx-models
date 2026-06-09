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

//! Fused codec-frame vs legacy eager path (talker hidden + codec groups).

use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::code_predictor::CodePredictorEngine;
use rlx_qwen3_tts::codec_frame_fused::CodecFrameFusedEngine;
use rlx_qwen3_tts::fused_e2e::{
    CodecFrameScratch, CodecFrameTimings, codec_frame_fused_step, codec_frame_step,
};
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use rlx_qwen3_tts::talker::engine::TalkerEngine;
use rlx_qwen3_tts::text_embed::TextEmbedder;
use rlx_runtime::Device;
use std::path::PathBuf;

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn fused_codec_frame_matches_legacy_one_frame() {
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY");
        return;
    }
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).expect("config");
    let store = Qwen3TtsWeightStore::open(&model_dir).expect("store");
    let tokenizer = load_text_tokenizer(&model_dir).expect("tok");
    let text_embedder = TextEmbedder::open(&store).expect("text");
    let prompt = build_custom_voice_prompt(
        &cfg,
        &store,
        &text_embedder,
        &tokenizer,
        "Hi.",
        "vivian",
        "english",
    )
    .expect("prompt");

    let talker_cfg = cfg.talker();
    let cp_cfg = cfg.code_predictor();
    let device = Device::Cpu;
    let hidden = talker_cfg.hidden_size;

    let mut talker_legacy = TalkerEngine::open(&store, talker_cfg, device).expect("talker legacy");
    let mut talker_fused = TalkerEngine::open(&store, talker_cfg, device).expect("talker fused");
    let mut cp = CodePredictorEngine::open(&store, cp_cfg, device).expect("cp");
    let mut fused = CodecFrameFusedEngine::open(&store, talker_cfg, cp_cfg, device).expect("fused");

    talker_legacy
        .prefill(prompt.embeds.view())
        .expect("prefill legacy");
    talker_fused
        .prefill(prompt.embeds.view())
        .expect("prefill fused");
    fused.warmup(&talker_fused, 8).expect("fused warmup");

    let mut scratch_legacy = CodecFrameScratch::new(hidden, talker_cfg.vocab_size);
    let mut scratch_fused = CodecFrameScratch::new(hidden, talker_cfg.vocab_size);
    scratch_legacy
        .hidden
        .copy_from_slice(talker_legacy.last_hidden_view().as_slice().unwrap());
    scratch_fused
        .hidden
        .copy_from_slice(talker_fused.last_hidden_view().as_slice().unwrap());

    let mut past_g0_legacy = Vec::new();
    let mut past_g0_fused = Vec::new();
    let mut timings_legacy = CodecFrameTimings::default();
    let mut timings_fused = CodecFrameTimings::default();

    let legacy_step = codec_frame_step(
        &mut talker_legacy,
        &mut cp,
        &mut scratch_legacy,
        talker_cfg,
        &prompt.tts_pad_embed,
        1.0,
        &mut past_g0_legacy,
        0,
        0,
        &mut timings_legacy,
    )
    .expect("legacy step");

    let fused_step = codec_frame_fused_step(
        &mut talker_fused,
        &mut fused,
        Some(&mut cp),
        &mut scratch_fused,
        talker_cfg,
        &prompt.tts_pad_embed,
        1.0,
        &mut past_g0_fused,
        0,
        0,
        &mut timings_fused,
    )
    .expect("fused step");

    let rlx_qwen3_tts::fused_e2e::CodecFrameStep::Frame(groups_legacy) = legacy_step else {
        panic!("legacy step did not emit a frame");
    };
    let rlx_qwen3_tts::fused_e2e::CodecFrameStep::Frame(groups_fused) = fused_step else {
        panic!("fused step did not emit a frame");
    };

    assert_eq!(groups_legacy, groups_fused, "codec groups mismatch");

    let diff = max_abs(
        scratch_legacy.hidden.as_slice(),
        scratch_fused.hidden.as_slice(),
    );
    eprintln!("[codec-frame fused] talk hidden max_abs={diff:.6}");
    assert!(
        diff < 0.05,
        "fused talker hidden diverged from legacy: max_abs={diff}"
    );
}

#[test]
fn fused_talk_decode_matches_eager_one_frame() {
    if std::env::var("RLX_QWEN3_TTS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip: RLX_QWEN3_TTS_PARITY");
        return;
    }
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).expect("config");
    let store = Qwen3TtsWeightStore::open(&model_dir).expect("store");
    let talker_cfg = cfg.talker();
    let cp_cfg = cfg.code_predictor();
    let device = Device::Cpu;
    let hidden = talker_cfg.hidden_size;

    let mut talker_a = TalkerEngine::open(&store, talker_cfg, device).expect("talker a");
    let mut talker_b = TalkerEngine::open(&store, talker_cfg, device).expect("talker b");
    let mut fused = CodecFrameFusedEngine::open(&store, talker_cfg, cp_cfg, device).expect("fused");

    let prefill = ndarray::Array2::<f32>::zeros((4, hidden));
    talker_a.prefill(prefill.view()).expect("prefill a");
    talker_b.prefill(prefill.view()).expect("prefill b");
    fused.warmup(&talker_a, 8).expect("fused warmup");

    let mut scratch = CodecFrameScratch::new(hidden, talker_cfg.vocab_size);
    scratch
        .hidden
        .copy_from_slice(talker_a.last_hidden_view().as_slice().unwrap());

    let groups = fused
        .predict_codec_groups(scratch.hidden.as_slice(), 1995, &[], &mut scratch.codec_emb)
        .expect("cp greedy");
    assert_eq!(groups.len(), 16);

    let mut eager_hidden = vec![0f32; hidden];
    talker_a
        .decode_hidden_into(
            ndarray::ArrayView1::from(scratch.codec_emb.as_slice()),
            &mut eager_hidden,
        )
        .expect("eager decode");

    let mut fused_hidden = vec![0f32; hidden];
    fused
        .run_talk_decode(
            &mut talker_b,
            scratch.codec_emb.as_slice(),
            &mut fused_hidden,
        )
        .expect("fused talk decode");

    let diff = max_abs(&eager_hidden, &fused_hidden);
    eprintln!("[codec-frame fused] talk-only max_abs={diff:.6}");
    assert!(
        diff < 0.05,
        "fused talker hidden diverged from eager: max_abs={diff}"
    );
}
