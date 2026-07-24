//! CPU reference vs every available RLX backend — exact score match (no ONNX).

use rlx_openwakeword::{
    OpenWakeWordEngine, OpenWakeWordWeights, WakeConfig, assert_100_percent_parity,
    run_backend_parity,
};
use rlx_wake::{bench_device_label, streaming_execution_device};

#[test]
fn openwakeword_100_percent_backend_parity() {
    let mut pcm = vec![0.0f32; 16_000 * 2];
    for (i, s) in pcm.iter_mut().enumerate() {
        *s = ((i as f32) * 0.013).sin() * 0.06;
    }
    let rows = run_backend_parity(&pcm, |dev| {
        let _ = streaming_execution_device(dev);
        Ok(
            OpenWakeWordEngine::new(OpenWakeWordWeights::stub("wake"), WakeConfig::default())
                .with_device_label(bench_device_label(dev)),
        )
    })
    .expect("parity");
    for r in &rows {
        eprintln!(
            "oww {:>8}: {:.1}% exact={}",
            r.device,
            r.parity * 100.0,
            r.exact
        );
    }
    assert_100_percent_parity(&rows).unwrap();
}
