//! Multi-phrase scale bench: N = 2..=10, f32 vs ternary.
//!
//! ```sh
//! cargo run -p rlx-wakeword --example multi_phrase_bench --release
//! ```

use rlx_wakeword::bundle::stub_bundle_n;
use rlx_wakeword_core::{
    MelConfig, SAMPLE_RATE_16K, TernaryOpts, WakeCnnConfig, WakeCnnWeights, pack_trits,
};
use std::time::Instant;

#[derive(Clone, Copy)]
enum WeightMode {
    F32,
    TernaryFc,
    TernaryAll,
}

impl WeightMode {
    fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::TernaryFc => "tern-fc",
            Self::TernaryAll => "tern-all",
        }
    }
}

fn tone(seconds: f32, freq_hz: f32, amp: f32) -> Vec<f32> {
    let n = (seconds * SAMPLE_RATE_16K as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE_16K as f32;
            (t * freq_hz * std::f32::consts::TAU).sin() * amp
        })
        .collect()
}

fn bias_bytes(w: &WakeCnnWeights) -> usize {
    (w.conv1_b.len() + w.conv2_b.len() + w.conv3_b.len() + w.fc1_b.len() + w.fc2_b.len()) * 4
}

fn weight_storage_bytes(mode: WeightMode) -> usize {
    let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
    match mode {
        WeightMode::F32 => {
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
        WeightMode::TernaryFc => {
            w.ternarize(TernaryOpts::fc_only());
            let packed = pack_trits(&w.fc1_w).len() + pack_trits(&w.fc2_w).len();
            let dense_rest = (w.conv1_w.len() + w.conv2_w.len() + w.conv3_w.len()) * 4;
            packed + dense_rest + bias_bytes(&w)
        }
        WeightMode::TernaryAll => {
            w.ternarize(TernaryOpts::all_weights());
            let packed = pack_trits(&w.conv1_w).len()
                + pack_trits(&w.conv2_w).len()
                + pack_trits(&w.conv3_w).len()
                + pack_trits(&w.fc1_w).len()
                + pack_trits(&w.fc2_w).len();
            packed + bias_bytes(&w)
        }
    }
}

fn estimate_ram_kb(weights_b: usize, hop: usize, context_frames: usize, n_mels: usize) -> f64 {
    let mel_ring = context_frames * n_mels * 4;
    let hop_buf = hop * 4;
    let pcm_ring = ((1.2f32 * SAMPLE_RATE_16K as f32) as usize) * 4;
    let scratch = 64 * 1024;
    (weights_b + mel_ring + hop_buf + pcm_ring + scratch) as f64 / 1024.0
}

struct Row {
    mode: WeightMode,
    n: usize,
    hops: usize,
    mean_hop_us: f64,
    wall_ms: f64,
    rtf: f64,
    weights_kib: f64,
    est_ram_kib: f64,
}

fn make_bundle(n: usize, hop_ms: u32, mode: WeightMode) -> rlx_wakeword::WakewordBundle {
    let mut bundle = stub_bundle_n(n, hop_ms, |i| format!("word{i}"));
    match mode {
        WeightMode::F32 => {}
        WeightMode::TernaryFc => {
            for (_, w) in &mut bundle.weights {
                w.ternarize(TernaryOpts::fc_only());
            }
        }
        WeightMode::TernaryAll => {
            for (_, w) in &mut bundle.weights {
                w.ternarize(TernaryOpts::all_weights());
            }
        }
    }
    bundle
}

fn bench_n(
    n: usize,
    pcm: &[f32],
    hop_ms: u32,
    mode: WeightMode,
    warmup: usize,
    rounds: usize,
) -> Row {
    let bundle = make_bundle(n, hop_ms, mode);
    let mut sess = bundle.into_session().expect("session");
    let hop = sess.config().hop_samples;
    let audio_s = pcm.len() as f64 / SAMPLE_RATE_16K as f64;
    let hops = pcm.len() / hop.max(1);

    for _ in 0..warmup {
        sess.reset();
        let _ = sess.push(pcm);
    }

    let mut total_us = 0.0f64;
    let wall = Instant::now();
    for _ in 0..rounds {
        sess.reset();
        let t0 = Instant::now();
        let _ = sess.push(pcm);
        total_us += t0.elapsed().as_secs_f64() * 1e6;
    }
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0 / rounds as f64;
    let mean_total_us = total_us / rounds as f64;
    let mean_hop_us = mean_total_us / hops.max(1) as f64;
    let rtf = (mean_total_us / 1e6) / audio_s;

    let w_b = weight_storage_bytes(mode) * n;
    let hop_len = MelConfig::default().hop_length;
    let ctx = sess.config().context_frames(hop_len);
    Row {
        mode,
        n,
        hops,
        mean_hop_us,
        wall_ms,
        rtf,
        weights_kib: w_b as f64 / 1024.0,
        est_ram_kib: estimate_ram_kb(w_b, hop, ctx, MelConfig::default().n_mels),
    }
}

fn print_table(title: &str, rows: &[Row]) {
    println!("{title}");
    println!(
        "{:>8}  {:>3}  {:>6}  {:>10}  {:>9}  {:>8}  {:>11}  {:>11}",
        "mode", "N", "hops", "mean_hop_us", "wall_ms", "RTF", "weights_KiB", "est_RAM_KiB"
    );
    println!("{}", "-".repeat(88));
    for row in rows {
        println!(
            "{:>8}  {:>3}  {:>6}  {:>10.1}  {:>9.2}  {:>8.4}  {:>11.1}  {:>11.1}",
            row.mode.label(),
            row.n,
            row.hops,
            row.mean_hop_us,
            row.wall_ms,
            row.rtf,
            row.weights_kib,
            row.est_ram_kib
        );
    }
    println!();
}

fn main() {
    let hop_ms: u32 = std::env::args()
        .skip(1)
        .find_map(|a| a.strip_prefix("--hop-ms=").map(|s| s.to_string()))
        .or_else(|| {
            let args: Vec<_> = std::env::args().collect();
            args.iter()
                .position(|a| a == "--hop-ms")
                .and_then(|i| args.get(i + 1).cloned())
        })
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let mut pcm = Vec::new();
    for (i, f) in [220.0, 330.0, 440.0, 550.0, 660.0].iter().enumerate() {
        pcm.extend(tone(1.2, *f, 0.25 + 0.02 * i as f32));
        pcm.extend(vec![0.0f32; (0.4 * SAMPLE_RATE_16K as f32) as usize]);
    }
    let audio_s = pcm.len() as f64 / SAMPLE_RATE_16K as f64;

    println!("rlx-wakeword multi-phrase scale (hop={hop_ms} ms, audio={audio_s:.2}s, CPU)\n");

    let modes = [
        WeightMode::F32,
        WeightMode::TernaryFc,
        WeightMode::TernaryAll,
    ];
    let mut all = Vec::new();
    for mode in modes {
        let mut rows = Vec::new();
        for n in 2..=10 {
            rows.push(bench_n(n, &pcm, hop_ms, mode, 2, 8));
        }
        print_table(&format!("── {} ──", mode.label()), &rows);
        all.extend(rows);
    }

    // Side-by-side at N=2 and N=10
    println!("── size @ N (packed storage) ──");
    println!(
        "{:>3}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "N", "f32_KiB", "tern-fc", "tern-all", "fc÷f32", "all÷f32"
    );
    println!("{}", "-".repeat(62));
    for n in 2..=10 {
        let f32b = weight_storage_bytes(WeightMode::F32) * n;
        let fcb = weight_storage_bytes(WeightMode::TernaryFc) * n;
        let allb = weight_storage_bytes(WeightMode::TernaryAll) * n;
        println!(
            "{:>3}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.2}  {:>10.2}",
            n,
            f32b as f64 / 1024.0,
            fcb as f64 / 1024.0,
            allb as f64 / 1024.0,
            fcb as f64 / f32b as f64,
            allb as f64 / f32b as f64
        );
    }

    println!("\n── latency @ N=2 / N=10 ──");
    for &n in &[2usize, 10] {
        print!("N={n}:");
        for mode in modes {
            if let Some(r) = all.iter().find(|r| {
                r.n == n
                    && matches!(
                        (r.mode, mode),
                        (WeightMode::F32, WeightMode::F32)
                            | (WeightMode::TernaryFc, WeightMode::TernaryFc)
                            | (WeightMode::TernaryAll, WeightMode::TernaryAll)
                    )
            }) {
                print!(
                    "  {}={:.0}µs/hop RTF={:.4}",
                    mode.label(),
                    r.mean_hop_us,
                    r.rtf
                );
            }
        }
        println!();
    }
}
