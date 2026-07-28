//! Measure wake-word **precision** (detection + float) and **latency** (p50/p99).
//!
//! ```sh
//! cargo run -p rlx-openwakeword --example engines_bench --release --features all-backends
//! ```

use rlx_nanowakeword::{NanoWakeWordEngine, NanoWakeWordWeights};
use rlx_openwakeword::{OpenWakeWordEngine, OpenWakeWordWeights};
use rlx_porcupine::{PorcupineEngine, PorcupineWeights};
use rlx_voxrt::{VoxrtEngine, VoxrtWeights};
use rlx_wake::{
    SAMPLE_RATE_16K, WakeConfig, WakeEngine, assert_100_percent_parity, available_devices,
    bench_device_label, bench_engine, best_f1_threshold, float_precision, peak_of,
    print_bench_table, print_detection_stats, run_backend_parity, score_wav,
    streaming_execution_device,
};

fn tone(seconds: f32, freq_hz: f32, amp: f32) -> Vec<f32> {
    let n = (seconds * SAMPLE_RATE_16K as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE_16K as f32;
            (t * freq_hz * std::f32::consts::TAU).sin() * amp
        })
        .collect()
}

fn silence(seconds: f32) -> Vec<f32> {
    vec![0.0f32; (seconds * SAMPLE_RATE_16K as f32) as usize]
}

fn noise(seconds: f32, amp: f32, seed: u64) -> Vec<f32> {
    let n = (seconds * SAMPLE_RATE_16K as f32) as usize;
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 33) as f32 / u32::MAX as f32;
            (u * 2.0 - 1.0) * amp
        })
        .collect()
}

/// Positive clips: stronger / multi-tone bursts. Negatives: silence + low noise.
fn eval_clips() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let positives = vec![
        tone(1.2, 220.0, 0.25),
        tone(1.2, 440.0, 0.30),
        {
            let mut a = tone(0.6, 330.0, 0.28);
            a.extend(tone(0.6, 550.0, 0.28));
            a
        },
        {
            let mut a = silence(0.3);
            a.extend(tone(0.8, 300.0, 0.35));
            a.extend(silence(0.3));
            a
        },
    ];
    let negatives = vec![
        silence(1.2),
        noise(1.2, 0.01, 1),
        noise(1.2, 0.02, 2),
        tone(1.2, 80.0, 0.02), // very quiet low tone
    ];
    (positives, negatives)
}

fn measure_detection<E: WakeEngine>(
    name: &str,
    device: &str,
    eng: &mut E,
    positives: &[Vec<f32>],
    negatives: &[Vec<f32>],
) -> anyhow::Result<()> {
    let mut pos_peaks = Vec::new();
    for clip in positives {
        pos_peaks.push(peak_of(eng, clip)?);
    }
    let mut neg_peaks = Vec::new();
    for clip in negatives {
        neg_peaks.push(peak_of(eng, clip)?);
    }
    let (thr, stats) = best_f1_threshold(&pos_peaks, &neg_peaks);
    let _ = thr;
    print_detection_stats(name, device, &stats);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let pcm = {
        let mut v = tone(2.0, 250.0, 0.08);
        v.extend(silence(1.0));
        v
    };
    let cfg = WakeConfig::default();
    let (positives, negatives) = eval_clips();

    println!("=== float precision vs CPU (bit-exact target) ===");
    for device in available_devices() {
        let _ = streaming_execution_device(device);
        let label = bench_device_label(device);
        let mut cpu = OpenWakeWordEngine::new(OpenWakeWordWeights::stub("wake"), cfg.clone());
        let mut other = OpenWakeWordEngine::new(OpenWakeWordWeights::stub("wake"), cfg.clone())
            .with_device_label(label);
        let ref_steps = score_wav(&mut cpu, &pcm)?;
        let cand = score_wav(&mut other, &pcm)?;
        let fp = float_precision(&ref_steps, &cand);
        println!(
            "  openwakeword {label:>8}: max_abs={:.3e} mean_abs={:.3e} matched={:.1}% n={}",
            fp.max_abs,
            fp.mean_abs,
            fp.matched_frac * 100.0,
            fp.n
        );
    }
    let rows = run_backend_parity(&pcm, |dev| {
        let _ = streaming_execution_device(dev);
        Ok(
            OpenWakeWordEngine::new(OpenWakeWordWeights::stub("wake"), cfg.clone())
                .with_device_label(bench_device_label(dev)),
        )
    })?;
    assert_100_percent_parity(&rows)?;
    println!("  backend score parity: 100%\n");

    println!("=== detection precision (stub weights; best-F1 threshold) ===");
    println!("  (positives=tones, negatives=silence/noise — measures separability of this build)");
    for device in available_devices().into_iter().take(1) {
        // CPU is enough for detection metrics (scores identical across backends).
        let label = bench_device_label(device);
        let _ = streaming_execution_device(device);
        {
            let mut e = OpenWakeWordEngine::new(OpenWakeWordWeights::stub("wake"), cfg.clone())
                .with_device_label(label);
            measure_detection("openwakeword", label, &mut e, &positives, &negatives)?;
        }
        {
            let mut e =
                NanoWakeWordEngine::new(NanoWakeWordWeights::stub(true, "hey nano"), cfg.clone())
                    .with_device_label(label);
            measure_detection("nanowakeword", label, &mut e, &positives, &negatives)?;
        }
        {
            let mut e = PorcupineEngine::new(PorcupineWeights::stub("porcupine"), cfg.clone())
                .with_device_label(label);
            measure_detection("porcupine", label, &mut e, &positives, &negatives)?;
        }
        {
            let mut e = VoxrtEngine::new(VoxrtWeights::stub("hey assistant"), cfg.clone())
                .with_device_label(label);
            measure_detection("voxrt", label, &mut e, &positives, &negatives)?;
        }
    }

    println!("\n=== latency (mean / p50 / p99 per 80ms chunk) ===");
    let mut stats = Vec::new();
    for device in available_devices() {
        let _ = streaming_execution_device(device);
        let label = bench_device_label(device);
        {
            let mut e = OpenWakeWordEngine::new(OpenWakeWordWeights::stub("wake"), cfg.clone())
                .with_device_label(label);
            stats.push(bench_engine("openwakeword", label, &mut e, &pcm, 2, 6)?);
        }
        {
            let mut e =
                NanoWakeWordEngine::new(NanoWakeWordWeights::stub(true, "hey nano"), cfg.clone())
                    .with_device_label(label);
            stats.push(bench_engine("nanowakeword", label, &mut e, &pcm, 2, 6)?);
        }
        {
            let mut e = PorcupineEngine::new(PorcupineWeights::stub("porcupine"), cfg.clone())
                .with_device_label(label);
            stats.push(bench_engine("porcupine", label, &mut e, &pcm, 2, 6)?);
        }
        {
            let mut e = VoxrtEngine::new(VoxrtWeights::stub("hey assistant"), cfg.clone())
                .with_device_label(label);
            stats.push(bench_engine("voxrt", label, &mut e, &pcm, 2, 6)?);
        }
    }
    print_bench_table(&stats);

    // Highlight CPU row summary
    println!("\n=== CPU summary ===");
    for r in stats.iter().filter(|r| r.device == "cpu") {
        println!(
            "  {}: wall={:.1}ms RTF={:.4} latency mean/p50/p99={:.0}/{:.0}/{:.0}µs peak={:.4}",
            r.engine,
            r.wall_s * 1e3,
            r.rtf,
            r.mean_chunk_us,
            r.p50_chunk_us,
            r.p99_chunk_us,
            r.peak_score
        );
    }
    Ok(())
}
