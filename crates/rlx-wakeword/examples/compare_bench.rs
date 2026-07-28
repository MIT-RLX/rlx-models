//! Compare rlx-wakeword vs other wake engines: precision, latency, size, memory.
//!
//! ```sh
//! cargo run -p rlx-wakeword --example compare_bench --release --features all-backends
//! ```

use rlx_nanowakeword::{NanoWakeWordEngine, NanoWakeWordWeights};
use rlx_openwakeword::{OpenWakeWordEngine, OpenWakeWordWeights};
use rlx_porcupine::{PorcupineEngine, PorcupineWeights};
use rlx_voxrt::{VoxrtEngine, VoxrtWeights};
use rlx_wake::{
    SAMPLE_RATE_16K, WakeConfig, WakeEngine, available_devices, bench_device_label, bench_engine,
    best_f1_threshold, peak_of, print_bench_table, print_detection_stats,
    streaming_execution_device,
};
use rlx_wakeword::bundle::stub_bundle;
use rlx_wakeword::session::{WakeEvent, WakewordSession};
use rlx_wakeword_core::{WakeCnnConfig, WakeCnnWeights};
use std::time::Instant;

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
        tone(1.2, 80.0, 0.02),
    ];
    (positives, negatives)
}

fn weight_bytes_cnn(w: &WakeCnnWeights) -> usize {
    (w.conv1_w.len()
        + w.conv1_b.len()
        + w.conv2_w.len()
        + w.conv2_b.len()
        + w.conv3_w.len()
        + w.conv3_b.len()
        + w.fc1_w.len()
        + w.fc1_b.len()
        + w.fc2_w.len()
        + w.fc2_b.len())
        * 4
}

fn oww_weight_bytes() -> usize {
    let w = OpenWakeWordWeights::stub("wake");
    let e = &w.embed;
    let p = &w.phrase;
    let embed = e.conv1_w.len()
        + e.conv1_b.len()
        + e.conv2_w.len()
        + e.conv2_b.len()
        + e.conv3_w.len()
        + e.conv3_b.len()
        + e.fc_w.len()
        + e.fc_b.len();
    let phrase = p.fc1_w.len() + p.fc1_b.len() + p.fc2_w.len() + p.fc2_b.len();
    (embed + phrase) * 4
}

/// Rough live working set: weights + mel ring (~1.2s × 32) + hop scratch + CNN temps.
fn estimate_ram_kb(weights_b: usize, hop: usize, context_frames: usize, n_mels: usize) -> f64 {
    let mel_ring = context_frames * n_mels * 4;
    let hop_buf = hop * 4;
    let scratch = 64 * 1024; // conv/FC temps upper bound for lite
    (weights_b + mel_ring + hop_buf + scratch) as f64 / 1024.0
}

fn peak_wakeword(sess: &mut WakewordSession, pcm: &[f32]) -> f32 {
    sess.reset();
    let mut peak = 0.0f32;
    for e in sess.push(pcm) {
        if let WakeEvent::Candidate { score, .. } = e {
            peak = peak.max(score);
        }
    }
    // Also consider non-fire scores aren't emitted — re-score via second path:
    // use max candidate or 0 if only Idle (stub scores stay low on silence).
    let _ = peak;
    // Force scoring without threshold by temporarily lowering threshold.
    sess.set_phrase_threshold("wake", 0.0);
    sess.reset();
    let mut peak = 0.0f32;
    for e in sess.push(pcm) {
        if let WakeEvent::Candidate { score, .. } = e {
            peak = peak.max(score);
        }
    }
    sess.set_phrase_threshold("wake", 0.5);
    peak
}

fn bench_wakeword(
    device: &str,
    sess: &mut WakewordSession,
    pcm: &[f32],
    warmup: usize,
    iters: usize,
) -> (f64, f64, f64, f64, f64) {
    let hop = sess.config().hop_samples.max(1);
    for _ in 0..warmup {
        sess.reset();
        let _ = sess.push(pcm);
    }
    let mut chunk_us = Vec::new();
    let t0 = Instant::now();
    for _ in 0..iters {
        sess.reset();
        let mut i = 0usize;
        while i < pcm.len() {
            let end = (i + hop).min(pcm.len());
            let mut chunk = pcm[i..end].to_vec();
            if chunk.len() < hop {
                chunk.resize(hop, 0.0);
            }
            let c0 = Instant::now();
            let _ = sess.push(&chunk);
            chunk_us.push(c0.elapsed().as_secs_f64() * 1e6);
            i += hop;
        }
    }
    let wall = t0.elapsed().as_secs_f64() / iters.max(1) as f64;
    let audio_s = pcm.len() as f64 / SAMPLE_RATE_16K as f64;
    let rtf = wall / audio_s.max(1e-12);
    chunk_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = chunk_us.iter().sum::<f64>() / chunk_us.len().max(1) as f64;
    let p50 = chunk_us[chunk_us.len() / 2];
    let p99 = chunk_us[((chunk_us.len() as f64 * 0.99) as usize).min(chunk_us.len() - 1)];
    let _ = device;
    (wall, rtf, mean, p50, p99)
}

fn measure_detection_engine<E: WakeEngine>(
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
    let (_thr, stats) = best_f1_threshold(&pos_peaks, &neg_peaks);
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

    let lite_w = WakeCnnWeights::stub(WakeCnnConfig::lite());
    let full_w = WakeCnnWeights::stub(WakeCnnConfig::full());
    let lite_b = weight_bytes_cnn(&lite_w);
    let full_b = weight_bytes_cnn(&full_w);
    let oww_b = oww_weight_bytes();

    println!("=== weight size (stub f32 tensors) ===");
    println!("{:<22} {:>10} {:>10}  notes", "engine", "params", "bytes");
    println!(
        "{:<22} {:>10} {:>10}  product default (1 phrase)",
        "wakeword-lite",
        lite_b / 4,
        lite_b
    );
    println!(
        "{:<22} {:>10} {:>10}  WakeCnn::full",
        "wakeword-full",
        full_b / 4,
        full_b
    );
    println!(
        "{:<22} {:>10} {:>10}  same lite CNN",
        "nanowakeword-lite",
        lite_b / 4,
        lite_b
    );
    println!(
        "{:<22} {:>10} {:>10}  same lite CNN",
        "porcupine",
        lite_b / 4,
        lite_b
    );
    println!(
        "{:<22} {:>10} {:>10}  same lite CNN",
        "voxrt",
        lite_b / 4,
        lite_b
    );
    println!(
        "{:<22} {:>10} {:>10}  embed + phrase head",
        "openwakeword",
        oww_b / 4,
        oww_b
    );

    println!("\n=== estimated live RAM (weights + mel ring + scratch) ===");
    println!("{:<22} {:>10}  hop", "engine", "RAM_KiB");
    println!(
        "{:<22} {:>10.1}  40 ms",
        "wakeword-lite",
        estimate_ram_kb(lite_b, 640, 40, 32)
    );
    println!(
        "{:<22} {:>10.1}  80 ms (OWW-style)",
        "nanowakeword-lite",
        estimate_ram_kb(lite_b, 1280, 40, 32)
    );
    println!(
        "{:<22} {:>10.1}  80 ms",
        "openwakeword",
        estimate_ram_kb(oww_b, 1280, 76, 32)
    );
    println!(
        "{:<22} {:>10.1}  40 ms",
        "wakeword-full",
        estimate_ram_kb(full_b, 640, 40, 32)
    );

    println!("\n=== detection precision (stub weights; best-F1 on synth tones) ===");
    println!("  (positives=tones, negatives=silence/noise — measures separability of this build)");
    let device = available_devices()[0];
    let _ = streaming_execution_device(device);
    let label = bench_device_label(device);
    {
        let mut e = OpenWakeWordEngine::new(OpenWakeWordWeights::stub("wake"), cfg.clone())
            .with_device_label(label);
        measure_detection_engine("openwakeword", label, &mut e, &positives, &negatives)?;
    }
    {
        let mut e =
            NanoWakeWordEngine::new(NanoWakeWordWeights::stub(true, "hey nano"), cfg.clone())
                .with_device_label(label);
        measure_detection_engine("nanowakeword", label, &mut e, &positives, &negatives)?;
    }
    {
        let mut e = PorcupineEngine::new(PorcupineWeights::stub("porcupine"), cfg.clone())
            .with_device_label(label);
        measure_detection_engine("porcupine", label, &mut e, &positives, &negatives)?;
    }
    {
        let mut e = VoxrtEngine::new(VoxrtWeights::stub("hey assistant"), cfg.clone())
            .with_device_label(label);
        measure_detection_engine("voxrt", label, &mut e, &positives, &negatives)?;
    }
    {
        let mut bundle = stub_bundle("wake", 40);
        bundle.config.vad_gate = false;
        let mut sess = bundle.open_session()?.with_device_label(label);
        let mut pos_peaks = Vec::new();
        for clip in &positives {
            pos_peaks.push(peak_wakeword(&mut sess, clip));
        }
        let mut neg_peaks = Vec::new();
        for clip in &negatives {
            neg_peaks.push(peak_wakeword(&mut sess, clip));
        }
        let (_thr, stats) = best_f1_threshold(&pos_peaks, &neg_peaks);
        print_detection_stats("wakeword@40ms", label, &stats);
    }

    println!("\n=== latency / speed (CPU; mean/p50/p99 per hop) ===");
    let mut stats = Vec::new();
    let cpu = available_devices()
        .into_iter()
        .find(|d| matches!(d, rlx_runtime::Device::Cpu))
        .unwrap_or(device);
    let _ = streaming_execution_device(cpu);
    let label = bench_device_label(cpu);
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
    print_bench_table(&stats);

    let mut bundle = stub_bundle("wake", 40);
    bundle.config.vad_gate = false;
    let mut sess = bundle.open_session()?.with_device_label(label);
    let (wall, rtf, mean, p50, p99) = bench_wakeword(label, &mut sess, &pcm, 2, 6);
    println!(
        "{:<18} {:<8} {:>7.3} {:>9.3} {:>8.4} {:>9.1} {:>9.1} {:>9.1}  (wakeword hop=40ms)",
        "wakeword",
        label,
        pcm.len() as f64 / SAMPLE_RATE_16K as f64,
        wall * 1e3,
        rtf,
        mean,
        p50,
        p99
    );

    // Also 20 ms hop
    let mut bundle20 = stub_bundle("wake", 20);
    bundle20.config.vad_gate = false;
    let mut sess20 = bundle20.open_session()?.with_device_label(label);
    let (wall, rtf, mean, p50, p99) = bench_wakeword(label, &mut sess20, &pcm, 2, 6);
    println!(
        "{:<18} {:<8} {:>7.3} {:>9.3} {:>8.4} {:>9.1} {:>9.1} {:>9.1}  (wakeword hop=20ms)",
        "wakeword-20ms",
        label,
        pcm.len() as f64 / SAMPLE_RATE_16K as f64,
        wall * 1e3,
        rtf,
        mean,
        p50,
        p99
    );

    println!("\n=== embedded packaging assessment ===");
    println!(
        "\
engine               MCU_flash   MCU_RAM     FPGA/ASIC_path              Verilog_ready
-------------------- ----------- ----------- --------------------------- ----------------
wakeword-lite        ~34 KiB     ~100 KiB    fixed f32→int8 graph        yes (core crate)
nanowakeword-lite    ~34 KiB     ~100 KiB    same CNN as wakeword        yes (via core)
porcupine / voxrt    ~34 KiB     ~100 KiB    same CNN                    yes (via core)
wakeword-full        ~443 KiB    ~520 KiB    still MCU-class if int8     yes
openwakeword         ~{oww_kib} KiB   larger     heavier embed CNN           harder (2-D conv)

Notes:
- rlx-wakeword-core is no_std + alloc, pure f32 ops (no BLAS) → CMSIS-NN / HLS / RTL export path.
- Pack magic RLXW for flat .rlxw blobs (dtype: f32 / reserved int8 / ternary trits).
- Always-on SoC budget: prefer wakeword-lite @ 20–40 ms hop; ASR confirm stays off-MCU.
- openWakeWord is the least MCU-friendly (larger embed + 2-D convs).
",
        oww_kib = oww_b.div_ceil(1024)
    );

    Ok(())
}
