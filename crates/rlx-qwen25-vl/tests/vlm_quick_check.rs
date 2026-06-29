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

//! Synthetic Qwen2.5-VL quick check: vision + mRoPE prefill + decode.

use rlx_qwen25_vl::{
    MEDIA_MARKER, MultimodalPrompt, Qwen25VlRunnerBuilder, synth,
    vision::{MmProjWeights, Qwen25VlVisionEncoder},
};
use rlx_runtime::Device;

fn fake_tokenizer(text: &str) -> anyhow::Result<Vec<u32>> {
    Ok(text.bytes().map(|b| (b as u32 % 31 + 1).max(1)).collect())
}

#[test]
fn qwen25_vlm_hidden_prefill_and_decode_quick_check() {
    let mmcfg = synth::tiny_mmproj_cfg();
    let mmweights = MmProjWeights::synthetic(&mmcfg);
    let lmcfg = synth::tiny_lm_cfg();
    let lmweights = synth::synth_lm_weight_map(&lmcfg);

    let mut runner = Qwen25VlRunnerBuilder::default()
        .lm_config(lmcfg.clone())
        .inline_lm_weights(lmweights.clone())
        .inline_mmproj(mmcfg.clone(), mmweights.clone())
        .device(Device::Cpu)
        .max_seq(64)
        .build()
        .expect("vlm runner");

    assert!(runner.has_vision());

    let img_w = 4;
    let img_h = 4;
    let rgb: Vec<u8> = (0..(img_w * img_h * 3)).map(|i| (i % 251) as u8).collect();
    let mut enc =
        Qwen25VlVisionEncoder::from_parts(mmcfg, mmweights, img_w, img_h).expect("vision encoder");
    let vision = enc.encode_rgb(&rgb, img_w, img_h).expect("encode");
    assert!(vision.embeddings.iter().all(|v| v.is_finite()));

    let prompt = format!("before{MEDIA_MARKER}after");
    let mm = MultimodalPrompt {
        prompt: &prompt,
        vision: &vision,
    };
    let embed = lmweights
        .get("model.embed_tokens.weight")
        .map(|(d, _)| d.as_slice())
        .expect("embed");
    let prefill = mm
        .assemble(fake_tokenizer, embed, lmcfg.lm.hidden_size, 0)
        .expect("assemble");
    assert_eq!(prefill.mrope_sections.len(), prefill.seq.len());

    let logits = runner
        .prefill_from_assembled(prefill)
        .expect("hidden prefill");
    assert_eq!(logits.len(), lmcfg.lm.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));

    let step = runner.decode_step(3).expect("decode step");
    assert_eq!(step.len(), lmcfg.lm.vocab_size);
    assert!(step.iter().all(|v| v.is_finite()));
}
