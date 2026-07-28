use rlx_openwakeword::{
    OpenWakeWordEngine, OpenWakeWordWeights, WakeConfig, WakeEngine, score_wav,
};

#[test]
fn stub_scores_finite() {
    let w = OpenWakeWordWeights::stub("alexa");
    let mut eng = OpenWakeWordEngine::new(w, WakeConfig::default());
    let pcm = vec![0.01f32; 16_000 * 2];
    let steps = score_wav(&mut eng, &pcm).unwrap();
    assert!(!steps.is_empty());
    assert!(
        steps
            .iter()
            .all(|s| s.score.is_finite() && s.score >= 0.0 && s.score <= 1.0)
    );
    eng.reset();
    let again = eng.push_pcm(&vec![0.0; 1280]).unwrap();
    assert_eq!(again.len(), 1);
}
