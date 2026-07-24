use rlx_openwakeword::{
    OpenWakeWordEngine, OpenWakeWordWeights, WakeConfig, available_devices, bind_streaming_device,
    score_wav,
};
use rlx_wake::bench_device_label;

#[test]
fn all_available_devices_accept_stub() {
    for device in available_devices() {
        let (exec, label) = bind_streaming_device(device).unwrap();
        assert_eq!(exec, device);
        assert_eq!(label, bench_device_label(device));
        let mut eng = OpenWakeWordEngine::new(
            OpenWakeWordWeights::stub("wake"),
            WakeConfig::default(),
        )
        .with_device_label(label);
        let steps = score_wav(&mut eng, &vec![0.0f32; 1280 * 4]).unwrap();
        assert!(!steps.is_empty(), "device={label}");
    }
}
