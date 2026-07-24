use rlx_porcupine::{
    PorcupineEngine, PorcupineWeights, WakeConfig, available_devices,
    bind_streaming_device, score_wav,
};
use rlx_wake::bench_device_label;

#[test]
fn backends_accept_stub() {
    for device in available_devices() {
        let (exec, label) = bind_streaming_device(device).unwrap();
        assert_eq!(exec, device);
        assert_eq!(label, bench_device_label(device));
        let mut eng = PorcupineEngine::new(
            PorcupineWeights::stub("porcupine"),
            WakeConfig::default(),
        )
        .with_device_label(label);
        assert!(
            !score_wav(&mut eng, &vec![0.0f32; 1280 * 2])
                .unwrap()
                .is_empty(),
            "device={label}"
        );
    }
}
