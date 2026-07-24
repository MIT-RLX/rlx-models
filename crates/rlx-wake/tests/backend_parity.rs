//! Exact cross-backend wake score parity (no ONNX).

use rlx_runtime::Device;
use rlx_wake::{
    MelConfig, MelFrontend, WakeCnn, WakeCnnConfig, WakeCnnWeights, WakeConfig, WakeEngine,
    WakeStep, assert_100_percent_parity, run_backend_parity, streaming_execution_device,
};

struct ProbeEngine {
    cfg: WakeConfig,
    mel: MelFrontend,
    cnn: WakeCnn,
    samples_seen: usize,
    last_fire_ms: f32,
    device: Device,
}

impl ProbeEngine {
    fn new(device: Device) -> Self {
        let _ = streaming_execution_device(device);
        let weights = WakeCnnWeights::stub(WakeCnnConfig::lite());
        Self {
            cfg: WakeConfig::default(),
            mel: MelFrontend::new(MelConfig::default()),
            cnn: WakeCnn::new(weights),
            samples_seen: 0,
            last_fire_ms: -1.0e9,
            device,
        }
    }
}

impl WakeEngine for ProbeEngine {
    fn push_pcm(&mut self, samples: &[f32]) -> anyhow::Result<Vec<WakeStep>> {
        let _ = self.device;
        let frames = self.mel.push(samples);
        let score = self.cnn.push_mel_frames(&frames);
        self.samples_seen += samples.len();
        let t_ms = self.samples_seen as f32 * 1000.0 / 16_000.0;
        let mut fired = score >= self.cfg.threshold;
        if fired && t_ms - self.last_fire_ms < self.cfg.cooldown_ms {
            fired = false;
        }
        if fired {
            self.last_fire_ms = t_ms;
        }
        Ok(vec![WakeStep { score, fired, t_ms }])
    }

    fn reset(&mut self) {
        self.mel.reset();
        self.cnn.reset();
        self.samples_seen = 0;
        self.last_fire_ms = -1.0e9;
    }

    fn config(&self) -> &WakeConfig {
        &self.cfg
    }
}

#[test]
fn wake_cnn_100_percent_backend_parity() {
    // Fixed synthetic PCM so trajectories are reproducible.
    let mut pcm = vec![0.0f32; 16_000 * 2];
    for (i, s) in pcm.iter_mut().enumerate() {
        *s = ((i as f32) * 0.017).sin() * 0.05;
    }
    let rows = run_backend_parity(&pcm, |dev| Ok(ProbeEngine::new(dev))).expect("parity run");
    for r in &rows {
        eprintln!(
            "device={} steps={} parity={:.1}% max_abs={:.3e} exact={}",
            r.device,
            r.steps,
            r.parity * 100.0,
            r.max_abs,
            r.exact
        );
    }
    assert_100_percent_parity(&rows).expect("expected 100% parity across RLX backends");
}
