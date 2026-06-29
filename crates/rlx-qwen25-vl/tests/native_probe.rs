// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use rlx_qwen25_vl::{
    MEDIA_MARKER, MultimodalPrompt, Qwen25VlRunnerBuilder, synth,
    vision::{MmProjWeights, Qwen25VlVisionEncoder},
};
use rlx_runtime::Device;

fn fake_tokenizer(text: &str) -> anyhow::Result<Vec<u32>> {
    Ok(text.bytes().map(|b| (b as u32 % 31 + 1).max(1)).collect())
}

#[test]
fn native_probe_via_graph_qk_and_cpu_replay() {
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

    runner
        .prefill_from_assembled_probe(prefill.clone())
        .expect("probe prefill");
    let graph_probe = runner.probe_aif_native().expect("graph probe");

    runner
        .prefill_from_assembled(prefill)
        .expect("plain prefill");
    let cpu_probe = runner.probe_aif_native().expect("cpu probe");

    assert_eq!(graph_probe.dynamics.len(), cpu_probe.dynamics.len());
    for (g, c) in graph_probe.dynamics.iter().zip(cpu_probe.dynamics.iter()) {
        assert_eq!(g.len(), c.len());
        for (a, b) in g.iter().zip(c.iter()) {
            assert!(
                (a - b).abs() < 0.05,
                "graph vs cpu dynamics mismatch: graph={a} cpu={b}"
            );
        }
    }
}
