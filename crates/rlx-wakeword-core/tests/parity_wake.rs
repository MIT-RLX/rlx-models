//! Parity: core WakeCnn vs rlx-wake WakeCnn on identical stub weights.

use rlx_wake::{
    MelConfig as HostMelCfg, MelFrontend as HostMel, WakeCnn as HostCnn, WakeCnnConfig as HostCfg,
    WakeCnnWeights as HostW,
};
use rlx_wakeword_core::{MelConfig, MelFrontend, WakeCnn, WakeCnnConfig, WakeCnnWeights};

fn copy_weights(host: &HostW) -> WakeCnnWeights {
    WakeCnnWeights::from_parts(
        WakeCnnConfig {
            n_mels: host.cfg.n_mels,
            c1: host.cfg.c1,
            c2: host.cfg.c2,
            c3: host.cfg.c3,
            k: host.cfg.k,
            hidden: host.cfg.hidden,
        },
        host.conv1_w.clone(),
        host.conv1_b.clone(),
        host.conv2_w.clone(),
        host.conv2_b.clone(),
        host.conv3_w.clone(),
        host.conv3_b.clone(),
        host.fc1_w.clone(),
        host.fc1_b.clone(),
        host.fc2_w.clone(),
        host.fc2_b.clone(),
    )
}

#[test]
fn cnn_scores_match_host_within_tol() {
    let host_w = HostW::stub(HostCfg::lite());
    let core_w = copy_weights(&host_w);
    let mut host = HostCnn::new(host_w);
    let mut core = WakeCnn::new(core_w);

    let mut host_mel = HostMel::new(HostMelCfg::default());
    let mut core_mel = MelFrontend::new(MelConfig::default());
    let pcm: Vec<f32> = (0..16_000)
        .map(|i| ((i as f32) * 0.01).sin() * 0.1)
        .collect();

    let hf = host_mel.push(&pcm);
    let cf = core_mel.push(&pcm);
    assert_eq!(hf.len(), cf.len());
    for (a, b) in hf.iter().zip(cf.iter()) {
        assert!((a - b).abs() < 1e-5, "mel delta {}", (a - b).abs());
    }

    let hs = host.push_mel_frames(&hf);
    let cs = core.push_mel_frames(&cf);
    assert!(
        (hs - cs).abs() < 1e-4,
        "score host={hs} core={cs} delta={}",
        (hs - cs).abs()
    );
}

#[test]
fn hop_40ms_produces_frames() {
    let mut mel = MelFrontend::new(MelConfig::default());
    let frames = mel.push(&vec![0.01f32; 640]);
    assert!(!frames.is_empty());
    assert_eq!(frames.len() % mel.n_mels(), 0);
}
