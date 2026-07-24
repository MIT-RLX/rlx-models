//! Exact cross-backend parity for nanowakeword (no ONNX).

use rlx_nanowakeword::{
    NanoWakeWordEngine, NanoWakeWordWeights, WakeConfig, assert_100_percent_parity,
    run_backend_parity,
};
use rlx_wake::{bench_device_label, streaming_execution_device};

#[test]
fn nanowakeword_100_percent_backend_parity() {
    let mut pcm = vec![0.0f32; 16_000 * 2];
    for (i, s) in pcm.iter_mut().enumerate() {
        *s = ((i as f32) * 0.011).sin() * 0.05;
    }
    let rows = run_backend_parity(&pcm, |dev| {
        let _ = streaming_execution_device(dev);
        Ok(
            NanoWakeWordEngine::new(NanoWakeWordWeights::stub(true, "hey nano"), WakeConfig::default())
                .with_device_label(bench_device_label(dev)),
        )
    })
    .unwrap();
    assert_100_percent_parity(&rows).unwrap();
}
