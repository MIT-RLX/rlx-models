use rlx_voxrt::{VoxrtEngine, VoxrtWeights, WakeConfig, score_wav};

#[test]
fn stub_finite() {
    let mut eng =
        VoxrtEngine::new(VoxrtWeights::stub("hey assistant"), WakeConfig::default());
    let steps = score_wav(&mut eng, &vec![0.0f32; 16_000]).unwrap();
    assert!(steps.iter().all(|s| s.score.is_finite()));
}
