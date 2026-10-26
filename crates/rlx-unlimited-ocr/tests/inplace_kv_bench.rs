// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Does the in-place KV append actually make OCR decode faster?
//!
//! `#[ignore]`d — this is a measurement, not an assertion. Run it with:
//!
//! ```bash
//! cargo test --release -p rlx-unlimited-ocr --test inplace_kv_bench \
//!     -- --ignored --nocapture
//! ```
//!
//! The claim behind the migration is that `concat(past_kv, row)` recopies the
//! WHOLE cache every step — O(context) for one row of real work — while
//! `Op::KvAppend` writes just that row. Correctness is pinned elsewhere
//! (`inplace_kv_guard`, `chunk_decode`, `full_causal_decode`); this measures
//! the payoff, and it is worth measuring rather than assuming, because THIS
//! crate's host path feeds its cache in as a graph input every step. That feed
//! is itself O(context), so removing the concat removes one of two
//! context-sized costs per step, not the only one — unlike qwen3/qwen35 on
//! Metal, where the cache is GPU-resident and the concat is the only one left
//! (there it was 16%/30%/33% of decode GPU time at ctx 256/1024/2048).
//!
//! Arms alternate and report the best of several reps: a single pass is
//! worthless here, same-config runs have been seen to vary ~2x.
//!
//! MEASURED (CPU, M-series, hidden 512, 2 layers), and it is a SMALL win, not
//! the 1.3-1.4x a first single run suggested — that did not reproduce:
//!
//! ```text
//!  past_seq      in-place        concat   speedup
//!       256       1.08ms        1.17ms      1.08x
//!      1024       1.66ms        1.68ms      1.01x
//!      4096       6.17ms        6.50ms      1.05x
//! ```
//!
//! The reason is in the last column of the in-place arm, not the speedup: per
//! step it still grows ~6x from 256 to 4096 rows. An O(context) cost REMAINS
//! after the append, and on CPU it is the larger one — this crate hands its
//! whole padded cache to the graph as a `past_k_*` / `past_v_*` input every
//! step, so the feed copy is O(context) too. Removing the concat removes one of
//! two, so the ceiling here is low by construction.
//!
//! MEASURED (CUDA, RTX 3080 Ti, same config) on an IDLE box, four runs — not
//! even a small win, it is nothing, and the reproducibility is absolute:
//!
//! ```text
//!  past_seq      in-place        concat   speedup
//!       256     0.870-0.948ms  0.872-0.956ms   1.00-1.01x
//!      1024       2.856-2.861ms  2.889-2.907ms   1.01-1.02x
//!      4096      17.794ms (x3)  17.909-17.935ms   1.01x
//! ```
//!
//! The reason is NOT launch overhead, which is what a first pass on a loaded
//! box suggested: it is that BOTH arms here pay the host feed. This bench hands
//! `past_k_*`/`past_v_*` in as plain graph inputs every step, so a context-sized
//! upload dominates the step in both arms and the concat is a small fraction of
//! a big number. Cross-check with `resident_kv_flattens_decode` on the same
//! box: its host-feed arm at 2048 is 6.1ms, matching this bench's trend, and
//! removing the feed takes the step to 1.33ms. So on CUDA at 2048 the feed is
//! ~78% of a decode step while the in-graph concat is ~1%. Set
//! `RLX_OCR_BENCH_DEVICE` to pick the backend.
//!
//! WATCH THE BOX. The first CUDA pass here was taken while the machine ran
//! someone else's benchmark and read 38ms/step at 4096 — 2x inflated — and the
//! resident bench was inflated 8x. Contention is roughly additive, so it hits
//! the FASTER arm hardest and squeezes every ratio toward 1.0: the resident win
//! read 1.91x loaded and 4.6x idle. Print the load average next to any number
//! you intend to quote.
//!
//! So: the append is correct, worth ~1.05x on CPU and nothing on CUDA, and it
//! is the prerequisite for the change that actually pays (a RESIDENT cache the
//! host stops re-feeding — which is exactly why qwen3/qwen35 on Metal saw 30%+,
//! see `project_inplace_kv_plan`). Do not quote a big number for OCR off this
//! migration alone.

use rlx_core::flow_util::compile_built;
use rlx_runtime::Device;
use rlx_unlimited_ocr::config::UnlimitedOcrConfig;
use rlx_unlimited_ocr::expert_pack::PackedLmWeights;
use rlx_unlimited_ocr::lm_device::CompiledLm;
use rlx_unlimited_ocr::lm_graph::{build_unlimited_ocr_decode_built_ext, compute_rope_slice};
use rlx_unlimited_ocr::lm_precision::ResolvedLmPrecision;
use std::sync::Arc;
use std::time::{Duration, Instant};

mod tiny;
use tiny::{cfg, fill, synthetic_weights};

/// Wide enough that a context-sized copy is real work, small enough to compile
/// per arm in a test. Row width is `hidden_size` (kv heads == q heads here).
fn bench_cfg() -> UnlimitedOcrConfig {
    let base = cfg();
    UnlimitedOcrConfig {
        hidden_size: 512,
        num_attention_heads: 8,
        num_key_value_heads: 8,
        intermediate_size: 512,
        moe_intermediate_size: 128,
        max_position_embeddings: 16384,
        // `validate` requires the projector to land on `hidden_size`.
        projector: rlx_unlimited_ocr::config::ProjectorConfig {
            n_embed: 512,
            ..base.projector.clone()
        },
        ..base
    }
}

/// Which device to measure on.
///
/// Defaults to CPU so the numbers in the module docs stay reproducible with a
/// bare `cargo test`. Set `RLX_OCR_BENCH_DEVICE` to any name
/// `available_devices()` reports (`cpu`, `metal`, `cuda`, ...) to measure a
/// backend instead — that is how the CUDA row in `resident_kv_device`'s
/// verified set was obtained.
fn bench_device() -> (&'static str, Device) {
    // Memoised: `available_devices()` re-probes every backend on each call and
    // logs a line per unavailable one, which buries the table below.
    static PICK: std::sync::OnceLock<(&'static str, Device)> = std::sync::OnceLock::new();
    *PICK.get_or_init(pick_bench_device)
}

fn pick_bench_device() -> (&'static str, Device) {
    let want = rlx_ir::env::var("RLX_OCR_BENCH_DEVICE")
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "cpu".to_string());
    rlx_core::backend_matrix::available_devices()
        .into_iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(&want))
        .unwrap_or_else(|| {
            panic!(
                "RLX_OCR_BENCH_DEVICE={want} is not available; this build offers {:?}",
                rlx_core::backend_matrix::available_devices()
                    .iter()
                    .map(|(n, _)| *n)
                    .collect::<Vec<_>>()
            )
        })
}

/// Median-of-`reps` wall time per decode step at `past_seq` rows of history.
fn time_steps(cfg: &UnlimitedOcrConfig, past_seq: usize, steps: usize) -> Duration {
    let h = cfg.hidden_size;
    let kv = cfg.kv_hidden();
    let n_layers = cfg.num_hidden_layers;

    let mut wm = synthetic_weights(cfg);
    let built = build_unlimited_ocr_decode_built_ext(cfg, &mut wm, 1, past_seq, true)
        .expect("decode build");
    let mut compiled = compile_built(built, bench_device().1).expect("decode compile");

    let step = fill(h, 0.9);
    let (cos, sin) = compute_rope_slice(cfg, past_seq);
    // Every padded row masked out but the new token's: the shape the bucketed
    // host path actually feeds.
    let mut mask = vec![0f32; past_seq + 1];
    mask[past_seq] = 1.0;
    let past: Vec<(String, Vec<f32>)> = (0..n_layers)
        .flat_map(|i| {
            [
                (format!("past_k_{i}"), fill(past_seq * kv, 0.3)),
                (format!("past_v_{i}"), fill(past_seq * kv, 0.4)),
            ]
        })
        .collect();

    let mut pairs: Vec<(&str, &[f32])> = vec![
        ("inputs_embeds", step.as_slice()),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
        ("mask", mask.as_slice()),
    ];
    for (n, d) in &past {
        pairs.push((n.as_str(), d.as_slice()));
    }

    // Warm the arena and any lazily-built kernel before timing.
    let warm = compiled.run(&pairs);
    assert!(
        warm[0].iter().all(|v| v.is_finite()),
        "decode produced non-finite logits at past_seq={past_seq}"
    );

    let t0 = Instant::now();
    for _ in 0..steps {
        compiled.run(&pairs);
    }
    t0.elapsed() / steps as u32
}

#[test]
#[ignore = "measurement, not an assertion — run with --ignored --nocapture"]
fn in_place_kv_append_beats_concat_as_context_grows() {
    let cfg = bench_cfg();
    cfg.validate().expect("config");
    const STEPS: usize = 24;
    const REPS: usize = 5;

    println!(
        "\nOCR decode, {}, hidden={} kv_row={} layers={}, {STEPS} steps/rep, best of {REPS}",
        bench_device().0,
        cfg.hidden_size,
        cfg.kv_hidden(),
        cfg.num_hidden_layers
    );
    println!(
        "{:>9}  {:>12}  {:>12}  {:>8}",
        "past_seq", "in-place", "concat", "speedup"
    );

    let mut first_inplace = Duration::ZERO;
    let mut last_inplace = Duration::ZERO;
    for past_seq in [256usize, 1024, 4096] {
        let mut best_inplace = Duration::MAX;
        let mut best_concat = Duration::MAX;
        for _ in 0..REPS {
            // Arms alternate so a thermal or scheduler drift hits both.
            rlx_ir::env::unset(rlx_flow::blocks::NO_INPLACE_KV);
            best_inplace = best_inplace.min(time_steps(&cfg, past_seq, STEPS));
            rlx_ir::env::set(rlx_flow::blocks::NO_INPLACE_KV, "1");
            best_concat = best_concat.min(time_steps(&cfg, past_seq, STEPS));
        }
        rlx_ir::env::unset(rlx_flow::blocks::NO_INPLACE_KV);
        let speedup = best_concat.as_secs_f64() / best_inplace.as_secs_f64();
        println!(
            "{past_seq:>9}  {:>12.3?}  {:>12.3?}  {speedup:>7.2}x",
            best_inplace, best_concat
        );
        if first_inplace.is_zero() {
            first_inplace = best_inplace;
        }
        last_inplace = best_inplace;
    }
    // The headline number. If the in-place arm still scales with context then
    // the append was not the binding cost, and the next lever is the host feed,
    // not the graph.
    println!(
        "\nin-place arm grew {:.1}x from the shortest to the longest context — \
         an O(context) cost remains (this crate re-feeds its whole cache every \
         step); a resident cache is the next lever, not the append",
        last_inplace.as_secs_f64() / first_inplace.as_secs_f64()
    );
}

/// The follow-up the first measurement pointed at: does keeping the cache
/// RESIDENT remove the O(context) cost the append left behind?
///
/// MEASURED (CPU, hidden 512, 2 layers): yes, and it is the real lever —
/// 1.61x @256, 1.47x @1024, 1.46x @2048 per decode step, vs ~1.1x for the
/// append alone. Both arms still scale with context because decode attention
/// genuinely reads the whole cache; what went away is the per-step copy of it.
///
/// MEASURED (CUDA, RTX 3080 Ti, IDLE box, four runs within 0.5%): far bigger,
/// as the mechanism predicts —
///
/// ```text
///  context      resident     host-feed   speedup
///      256     0.498ms       1.502ms      3.01x
///     1024     0.856ms       3.396ms      3.96x
///     2048     1.326ms       6.099ms      4.60x
/// ```
///
/// The feed removed is a per-step upload of 2 x layers x past_seq x kv_row
/// floats (16.8 MB at this config) over PCIe out of pageable host memory, not a
/// memcpy — so it is O(context) and the win GROWS with context, unlike on CPU.
/// At 2048 it is ~78% of the step. The sibling append bench is flat (1.01x) on
/// the same box for the complementary reason: there BOTH arms pay that feed, so
/// the in-graph concat is ~1% of a much larger number.
///
/// A loaded box reads this completely wrong. The first pass gave 1.91x @2048
/// and 0.79x @256 while someone else's benchmark ran at load ~4; contention is
/// roughly additive, so it hits the faster arm hardest and squeezes every ratio
/// toward 1.0. Print the load average beside anything you plan to quote.
///
/// Driven through `CompiledLm::decode_step`, which is what the runner uses, so
/// this measures the real loop rather than a bare graph. `RLX_DISABLE_GPU_KV`
/// picks the arm, `RLX_OCR_BENCH_DEVICE` the backend.
#[test]
#[ignore = "measurement, not an assertion — run with --ignored --nocapture"]
fn resident_kv_flattens_decode() {
    let cfg = bench_cfg();
    cfg.validate().expect("config");
    const STEPS: usize = 16;
    const REPS: usize = 3;

    let mut wm = synthetic_weights(&cfg);
    let pack = Arc::new(
        PackedLmWeights::from_weight_map(&mut wm, cfg.clone(), ResolvedLmPrecision::F32)
            .expect("pack"),
    );

    println!(
        "\nOCR decode via CompiledLm, {}, hidden={} layers={}, {STEPS} steps/rep, best of {REPS}",
        bench_device().0,
        cfg.hidden_size,
        cfg.num_hidden_layers
    );
    println!(
        "{:>8}  {:>12}  {:>12}  {:>8}",
        "context", "resident", "host-feed", "speedup"
    );

    let mut first = (Duration::ZERO, Duration::ZERO);
    let mut last = (Duration::ZERO, Duration::ZERO);
    for ctx in [256usize, 1024, 2048] {
        let mut best = [Duration::MAX; 2];
        for _ in 0..REPS {
            for (arm, on) in [(0usize, true), (1usize, false)] {
                if on {
                    rlx_ir::env::unset("RLX_DISABLE_GPU_KV");
                } else {
                    rlx_ir::env::set("RLX_DISABLE_GPU_KV", "1");
                }
                best[arm] = best[arm].min(time_lm_steps(&cfg, &pack, ctx, STEPS, on));
            }
        }
        rlx_ir::env::unset("RLX_DISABLE_GPU_KV");
        let speedup = best[1].as_secs_f64() / best[0].as_secs_f64();
        println!(
            "{ctx:>8}  {:>12.3?}  {:>12.3?}  {speedup:>7.2}x",
            best[0], best[1]
        );
        if first.0.is_zero() {
            first = (best[0], best[1]);
        }
        last = (best[0], best[1]);
    }
    // Read this carefully: BOTH arms still grow with context, and that is
    // correct. Decode attention reads `past_seq + 1` keys, so it is O(context)
    // by definition — real work, not overhead. What the resident cache removes
    // is the OVERHEAD O(context): the feed copy. Hence a constant-factor win at
    // every length rather than a flattening. Flattening the curve itself needs
    // fewer keys per step (selective/HNSW KV), not a cheaper way to hand over
    // the same ones.
    println!(
        "\ngrowth from shortest to longest context: resident {:.1}x, host-feed {:.1}x \
         — both grow because decode attention is genuinely O(context); the win here \
         is the constant factor (the feed copy), not the slope",
        last.0.as_secs_f64() / first.0.as_secs_f64(),
        last.1.as_secs_f64() / first.1.as_secs_f64()
    );
}

/// Prefill `ctx` tokens, then median per-step time over `steps` decode steps.
fn time_lm_steps(
    cfg: &UnlimitedOcrConfig,
    pack: &Arc<PackedLmWeights>,
    ctx: usize,
    steps: usize,
    expect_resident: bool,
) -> Duration {
    let h = cfg.hidden_size;
    let mut lm = CompiledLm::new(bench_device().1, Arc::clone(pack));
    let embeds = fill(ctx * h, 0.5);
    let (_, mut kv) = lm.prefill(&embeds, ctx).expect("prefill");
    let step = fill(h, 0.9);

    // One untimed step: compiles the decode graph and binds the handles.
    lm.decode_step(&step, ctx, &mut kv).expect("warm step");
    assert_eq!(
        lm.resident_decode_graphs() > 0,
        expect_resident,
        "resident path did not match the requested arm — the numbers would be \
         the same path twice"
    );

    let t0 = Instant::now();
    for i in 0..steps {
        lm.decode_step(&step, ctx + 1 + i, &mut kv).expect("step");
    }
    t0.elapsed() / steps as u32
}
