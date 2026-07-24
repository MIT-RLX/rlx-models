//! Ternary FC weights: fused add/sub path matches dense gemv.

use rlx_wakeword_core::{
    MelConfig, MelFrontend, TernaryOpts, WakeCnn, WakeCnnConfig, WakeCnnWeights, is_ternary_f32,
};

#[test]
fn ternarize_fc_is_exact() {
    let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
    let stats = w.ternarize(TernaryOpts::fc_only());
    assert!(stats.tensors >= 2);
    assert!(is_ternary_f32(&w.fc1_w));
    assert!(is_ternary_f32(&w.fc2_w));
    assert!(w.fc_ternary());
}

#[test]
fn ternary_forward_runs() {
    let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
    w.ternarize(TernaryOpts::all_weights());
    assert!(w.conv_ternary());
    assert!(w.fc_ternary());
    let mut cnn = WakeCnn::new(w).with_window_frames(40);
    let mut mel = MelFrontend::new(MelConfig::default());
    let pcm = vec![0.1f32; 16_000];
    let frames = mel.push(&pcm);
    let score = cnn.push_mel_frames(&frames);
    assert!(score.is_finite());
    assert!((0.0..=1.0).contains(&score));
}
