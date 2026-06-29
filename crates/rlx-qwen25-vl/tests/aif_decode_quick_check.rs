// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! AIF-Lite decode quick check: custom mask changes decode logits.

use rlx_qwen25_vl::{
    AifConfig, AifProbe, MEDIA_MARKER, MultimodalPrompt, Qwen25VlRunnerBuilder, synth,
    vision::{MmProjWeights, Qwen25VlVisionEncoder},
};
use rlx_runtime::Device;

fn fake_tokenizer(text: &str) -> anyhow::Result<Vec<u32>> {
    Ok(text.bytes().map(|b| (b as u32 % 31 + 1).max(1)).collect())
}

fn prefill_and_first_decode(
    runner: &mut rlx_qwen25_vl::Qwen25VlRunner,
    prefill: rlx_qwen25_vl::MultimodalPrefill,
    token: u32,
) -> Vec<f32> {
    runner.clear_aif_decode();
    runner.prefill_from_assembled(prefill).expect("prefill");
    runner.decode_step(token).expect("decode")
}

#[test]
fn aif_custom_mask_changes_decode_logits() {
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
        .expect("runner");

    let img = 8usize;
    let rgb: Vec<u8> = (0..(img * img * 3)).map(|i| (i % 251) as u8).collect();
    let mut enc = Qwen25VlVisionEncoder::from_parts(mmcfg.clone(), mmweights.clone(), img, img)
        .expect("vision encoder");
    let vision = enc.encode_rgb(&rgb, img, img).expect("encode");

    let prompt = format!("q{MEDIA_MARKER}a");
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

    assert!(prefill.n_vision_tokens > 0);
    let baseline = prefill_and_first_decode(&mut runner, prefill.clone(), 7);

    runner
        .prefill_from_assembled(prefill.clone())
        .expect("prefill");
    let span = runner.vision_key_span().expect("vision span");
    let n_vis = prefill.n_vision_tokens;
    let dynamics: Vec<Vec<f32>> = (0..n_vis).map(|_| vec![0.1, 0.2]).collect();
    let mut probe = AifProbe::build(dynamics);
    probe.mask_ratio = 0.5;
    let aif = AifConfig::from_probe(probe);
    runner.apply_aif_config(&aif).expect("aif");
    let blocked = aif.blocked_keys(span);
    assert!(!blocked.is_empty());
    let masked = runner.decode_step(7).expect("masked decode");

    assert_eq!(baseline.len(), masked.len());
    assert!(
        baseline != masked,
        "AIF mask should change decode logits (blocked {blocked:?})"
    );
}

#[test]
fn vision_key_span_matches_prefill_layout() {
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
        .expect("runner");

    let rgb: Vec<u8> = (0..48).map(|i| (i % 251) as u8).collect();
    let mut enc = Qwen25VlVisionEncoder::from_parts(mmcfg, mmweights, 4, 4).expect("vision");
    let vision = enc.encode_rgb(&rgb, 4, 4).expect("encode");
    let prompt = format!("x{MEDIA_MARKER}y");
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

    runner.prefill_from_assembled(prefill).expect("prefill");
    let span = runner.vision_key_span().expect("span");
    assert_eq!(span.len(), vision.n_tokens);
}
