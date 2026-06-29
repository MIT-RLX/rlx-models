// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Native AIF decode-step probe (Fig. 6) vs prefill_v2t on synthetic weights.

use rlx_qwen25_vl::{
    AifDynamicsMode, MEDIA_MARKER, MultimodalPrompt, Qwen25VlRunnerBuilder, synth,
    vision::{MmProjWeights, Qwen25VlVisionEncoder},
};
use rlx_runtime::Device;

fn fake_tokenizer(text: &str) -> anyhow::Result<Vec<u32>> {
    Ok(text.bytes().map(|b| (b as u32 % 31 + 1).max(1)).collect())
}

#[test]
fn decode_step_probe_produces_dynamics() {
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
        .aif_dynamics_mode(AifDynamicsMode::DecodeStep)
        .build()
        .expect("runner");

    let img = 8usize;
    let rgb: Vec<u8> = (0..(img * img * 3)).map(|i| (i % 251) as u8).collect();
    let mut enc = Qwen25VlVisionEncoder::from_parts(mmcfg, mmweights, img, img).expect("vision");
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

    runner.prefill_from_assembled(prefill).expect("prefill");
    let probe = runner.probe_aif_native().expect("decode-step probe");
    assert_eq!(probe.dynamics.len(), vision.n_tokens);
    assert_eq!(probe.dynamics[0].len(), lmcfg.lm.num_hidden_layers);
    assert!(probe.mu.iter().all(|v| v.is_finite()));
}

#[test]
fn decode_step_and_prefill_dynamics_differ_on_synthetic() {
    let mmcfg = synth::tiny_mmproj_cfg();
    let mmweights = MmProjWeights::synthetic(&mmcfg);
    let lmcfg = synth::tiny_lm_cfg();
    let lmweights = synth::synth_lm_weight_map(&lmcfg);

    let img = 8usize;
    let rgb: Vec<u8> = (0..(img * img * 3)).map(|i| (i % 251) as u8).collect();
    let prompt = format!("x{MEDIA_MARKER}y");
    let embed = lmweights
        .get("model.embed_tokens.weight")
        .map(|(d, _)| d.as_slice())
        .expect("embed");

    let mut runner_prefill = Qwen25VlRunnerBuilder::default()
        .lm_config(lmcfg.clone())
        .inline_lm_weights(lmweights.clone())
        .inline_mmproj(mmcfg.clone(), mmweights.clone())
        .device(Device::Cpu)
        .max_seq(64)
        .aif_dynamics_mode(AifDynamicsMode::PrefillV2t)
        .build()
        .expect("runner");
    let mut enc = Qwen25VlVisionEncoder::from_parts(mmcfg.clone(), mmweights.clone(), img, img)
        .expect("vision");
    let vision = enc.encode_rgb(&rgb, img, img).expect("encode");
    let mm = MultimodalPrompt {
        prompt: &prompt,
        vision: &vision,
    };
    let prefill = mm
        .assemble(fake_tokenizer, embed, lmcfg.lm.hidden_size, 0)
        .expect("assemble");
    runner_prefill
        .prefill_from_assembled_probe(prefill.clone())
        .expect("prefill");
    let prefill_probe = runner_prefill.probe_aif_native().expect("prefill probe");

    let mut runner_decode = Qwen25VlRunnerBuilder::default()
        .lm_config(lmcfg)
        .inline_lm_weights(lmweights)
        .inline_mmproj(mmcfg, mmweights)
        .device(Device::Cpu)
        .max_seq(64)
        .aif_dynamics_mode(AifDynamicsMode::DecodeStep)
        .build()
        .expect("runner");
    runner_decode
        .prefill_from_assembled(prefill)
        .expect("prefill");
    let decode_probe = runner_decode.probe_aif_native().expect("decode probe");

    assert_eq!(prefill_probe.dynamics.len(), decode_probe.dynamics.len());
    assert!(
        prefill_probe.dynamics != decode_probe.dynamics,
        "prefill_v2t and decode_step dynamics should differ on synthetic stack"
    );
}
