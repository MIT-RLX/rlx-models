use rlx_wake::{available_devices, bind_streaming_device};
use rlx_wakeword::bundle::stub_bundle;

#[test]
fn backends_bind_and_score() {
    let pcm = vec![0.0f32; 640 * 4];
    for device in available_devices() {
        let (exec, label) = bind_streaming_device(device).unwrap();
        assert_eq!(exec, device);
        let mut sess = stub_bundle("wake", 40)
            .open_session()
            .unwrap()
            .with_device_label(label);
        let events = sess.push(&pcm);
        assert!(!events.is_empty(), "device={label}");
    }
}
