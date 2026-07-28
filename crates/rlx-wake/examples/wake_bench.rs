//! Bench WakeCnn wake path across RLX backends + assert 100% score parity.
//!
//! ```sh
//! cargo run -p rlx-wake --example wake_bench --release --features all-backends
//! ```

use rlx_wake::{
    MelConfig, MelFrontend, WakeCnn, WakeCnnConfig, WakeCnnWeights, WakeConfig, WakeEngine,
    WakeStep, assert_100_percent_parity, available_devices, bench_device_label, bench_engine,
    print_bench_table, run_backend_parity, streaming_execution_device,
};

struct ProbeEngine {
    cfg: WakeConfig,
    mel: MelFrontend,
    cnn: WakeCnn,
    samples_seen: usize,
    last_fire_ms: f32,
}

impl ProbeEngine {
    fn new() -> Self {
        Self {
            cfg: WakeConfig::default(),
            mel: MelFrontend::new(MelConfig::default()),
            cnn: WakeCnn::new(WakeCnnWeights::stub(WakeCnnConfig::lite())),
            samples_seen: 0,
            last_fire_ms: -1.0e9,
        }
    }
}

impl WakeEngine for ProbeEngine {
    fn push_pcm(&mut self, samples: &[f32]) -> anyhow::Result<Vec<WakeStep>> {
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

fn synth_pcm(seconds: f32) -> Vec<f32> {
    let n = (seconds * 16_000.0) as usize;
    (0..n)
        .map(|i| ((i as f32) * 0.019).sin() * 0.04 + ((i as f32) * 0.003).cos() * 0.01)
        .collect()
}

fn main() -> anyhow::Result<()> {
    let pcm = synth_pcm(3.0);
    println!("=== wake WakeCnn backend parity (no ONNX) ===");
    let rows = run_backend_parity(&pcm, |_dev| Ok(ProbeEngine::new()))?;
    for r in &rows {
        println!(
            "  {:>8}: parity={:.1}%  max_abs={:.3e}  exact={}  steps={}",
            r.device,
            r.parity * 100.0,
            r.max_abs,
            r.exact,
            r.steps
        );
    }
    assert_100_percent_parity(&rows)?;
    println!("parity: 100% across {} RLX backend(s)", rows.len());

    println!("\n=== wake WakeCnn bench ===");
    let mut stats = Vec::new();
    for device in available_devices() {
        let _ = streaming_execution_device(device);
        let label = bench_device_label(device);
        let mut eng = ProbeEngine::new();
        stats.push(bench_engine("wake-cnn-lite", label, &mut eng, &pcm, 2, 8)?);
    }
    print_bench_table(&stats);
    Ok(())
}
