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

//! After golden frame-0 codec embed, next talker g0 must match HF (215).

use ndarray::ArrayView1;
use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::megakernel::Qwen3TtsMegakernel;
use rlx_qwen3_tts::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use rlx_qwen3_tts::talker::math::{linear_logits, sample_greedy_talker_codec};
use rlx_qwen3_tts::text_embed::TextEmbedder;
use std::path::PathBuf;

#[test]
fn talker_g0_after_golden_frame0_matches_hf() {
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

    let golden0 = [
        1995u32, 1642, 988, 1088, 246, 1543, 1579, 437, 1356, 86, 1042, 248, 1555, 781, 1772, 374,
    ];
    let expect_g0 = 215u32;

    let mut mk = Qwen3TtsMegakernel::open(
        &store,
        cfg.talker(),
        cfg.code_predictor(),
        rlx_runtime::Device::Cpu,
    )
    .expect("mk");
    mk.talker_prefill(prompt.embeds.view()).expect("prefill");

    let talker_cfg = cfg.talker();
    let eos = talker_cfg.codec_eos_token_id;
    let h_last = mk.talker_hidden_row();
    let logits0 = linear_logits(h_last, mk.codec_head()).expect("logits");
    let g0 = sample_greedy_talker_codec(&logits0, talker_cfg.vocab_size, eos);
    assert_eq!(g0, golden0[0]);

    // Build sum embed from golden frame-0 (isolates talker decode from CP)
    let snap = store
        .tensor_snapshot(&["talker.model.codec_embedding.weight"])
        .expect("snap");
    let (tc, sh) = snap.get("talker.model.codec_embedding.weight").unwrap();
    let codec = ndarray::Array2::from_shape_vec((sh[0], sh[1]), tc.clone()).unwrap();
    let hidden_sz = talker_cfg.hidden_size;
    let mut emb = vec![0f32; hidden_sz];
    for (gi, &tok) in golden0.iter().enumerate() {
        let row = if gi == 0 {
            codec.row(tok as usize).to_vec()
        } else {
            let key = format!(
                "talker.code_predictor.model.codec_embedding.{}.weight",
                gi - 1
            );
            let (data, shape) = store.tensor_snapshot(&[&key]).expect("e")[&key].clone();
            let table = ndarray::Array2::from_shape_vec((shape[0], shape[1]), data).unwrap();
            table.row(tok as usize).to_vec()
        };
        for (j, v) in row.iter().enumerate() {
            emb[j] += *v;
        }
    }
    for (j, v) in prompt.tts_pad_embed.iter().enumerate() {
        emb[j] += *v;
    }
    let mut hidden_row = vec![0f32; hidden_sz];
    mk.talker_decode_into(&emb, &mut hidden_row)
        .expect("decode");
    let logits1 =
        linear_logits(ArrayView1::from(hidden_row.as_slice()), mk.codec_head()).expect("logits1");
    let g1 = sample_greedy_talker_codec(&logits1, talker_cfg.vocab_size, eos);
    eprintln!("g1={g1} expect {expect_g0}");
    assert_eq!(g1, expect_g0, "talker g0 after golden frame-0 decode");
}
