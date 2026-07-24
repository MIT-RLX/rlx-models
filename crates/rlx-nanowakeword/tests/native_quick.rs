use rlx_nanowakeword::{NanoWakeWordEngine, NanoWakeWordWeights, WakeConfig, score_wav};

#[test]
fn stub_lite_and_full() {
    for lite in [true, false] {
        let w = NanoWakeWordWeights::stub(lite, "hey nano");
        let mut eng = NanoWakeWordEngine::new(w, WakeConfig::default());
        let steps = score_wav(&mut eng, &vec![0.02f32; 16_000]).unwrap();
        assert!(!steps.is_empty());
        assert!(steps.iter().all(|s| s.score.is_finite()));
    }
}
