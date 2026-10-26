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

//! Keeping the K/V cache resident must not change a single logit.
//!
//! Full-causal decode normally hands its whole padded cache back in as a
//! `past_k_*` / `past_v_*` input every step, which is O(context) per token on
//! top of the O(1) in-place append. The resident path binds that buffer once,
//! feeds nothing, and folds the new row in on device. Faster, and invisible —
//! so the test is a straight A/B of the same steps with
//! `RLX_DISABLE_GPU_KV` flipped.
//!
//! Two failure modes this is really here for:
//!
//! * a STALE handle. Resident K/V re-seeded only when the bucket changes makes
//!   a second generate() on the same runner answer the PREVIOUS prompt — that
//!   shipped once in rlx-qwen35's Metal path. So one case runs two independent
//!   prompts through one `CompiledLm` and demands the second match a fresh one.
//! * a rollback. Speculative decode drops `valid` without a decode step, which
//!   leaves the handles holding more history than the cache admits to.

use rlx_core::backend_matrix::{Failures, available_devices, max_rel_diff};
use rlx_runtime::Device;
use rlx_unlimited_ocr::config::UnlimitedOcrConfig;
use rlx_unlimited_ocr::expert_pack::PackedLmWeights;
use rlx_unlimited_ocr::lm_device::{CompiledLm, resident_kv_available};
use rlx_unlimited_ocr::lm_precision::ResolvedLmPrecision;
use std::sync::Arc;

mod tiny;
use tiny::{cfg, fill, synthetic_weights};

const PROMPT: usize = 6;
const STEPS: usize = 5;

fn pack_for(cfg: &UnlimitedOcrConfig) -> Arc<PackedLmWeights> {
    let mut wm = synthetic_weights(cfg);
    Arc::new(
        PackedLmWeights::from_weight_map(&mut wm, cfg.clone(), ResolvedLmPrecision::F32)
            .expect("pack"),
    )
}

/// `STEPS` decode steps after a `PROMPT`-token prefill, with the resident path
/// on or off. Returns the per-step logits.
fn run_steps(
    cfg: &UnlimitedOcrConfig,
    pack: &Arc<PackedLmWeights>,
    device: Device,
    resident: bool,
    seed: f32,
) -> Option<Vec<Vec<f32>>> {
    set_resident(resident);
    let h = cfg.hidden_size;
    let embeds = fill(PROMPT * h, seed);
    let steps_in = fill(STEPS * h, seed + 0.4);
    let mut lm = CompiledLm::new(device, Arc::clone(pack));
    let (_, mut kv) = lm.prefill(&embeds, PROMPT).expect("prefill");
    let out = (0..STEPS)
        .map(|i| {
            lm.decode_step(&steps_in[i * h..(i + 1) * h], PROMPT + i, &mut kv)
                .expect("decode step")
        })
        .collect();
    // Without this the A/B is vacuous: both arms could be the host path.
    let bound = lm.resident_decode_graphs();
    set_resident(true);
    if !resident {
        assert_eq!(
            bound, 0,
            "RLX_DISABLE_GPU_KV did not disable the resident path"
        );
        return Some(out);
    }
    // `None`, not a panic: the caller records it against this device and moves
    // on to the next one. Panicking here made the FIRST bad backend mask every
    // backend behind it — on a box with both wgpu and vulkan, wgpu's refusal
    // meant vulkan was never exercised at all, which is the opposite of what a
    // backend matrix is for.
    (bound > 0).then_some(out)
}

/// `RLX_DISABLE_GPU_KV` is process-global and these tests flip it, so they
/// cannot run concurrently with each other — the default harness would.
static ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_env_lock<T>(f: impl FnOnce() -> T) -> T {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    // Also BEFORE, not just after: a panicking test leaves the flag set, and
    // `skip_unverified` reads it at the top of the loop.
    set_resident(true);
    let out = f();
    set_resident(true);
    out
}

/// A backend outside `resident_kv_device`'s verified set is skipped, loudly,
/// rather than failed: the default gate refusing it is the intended behaviour,
/// and `RLX_UNLIMITED_OCR_RESIDENT_KV_ANY=1` is how you promote one (run these
/// tests with it, then add the device to the gate). Inside the set, a resident
/// path that fails to engage is still a hard failure — see `run_steps`.
fn skip_unverified(name: &str, device: Device) -> bool {
    if resident_kv_available(device) {
        return true;
    }
    eprintln!(
        "{name}: outside resident_kv_device's verified set — skipping. \
         Re-run with RLX_UNLIMITED_OCR_RESIDENT_KV_ANY=1 to measure it."
    );
    false
}

fn set_resident(on: bool) {
    if on {
        rlx_ir::env::unset("RLX_DISABLE_GPU_KV");
    } else {
        rlx_ir::env::set("RLX_DISABLE_GPU_KV", "1");
    }
}

/// Unwrap a resident arm, or record "never engaged" against this device and
/// move on. Returns `None` when the caller should `continue`.
fn engaged(fails: &mut Failures, name: &str, got: Option<Vec<Vec<f32>>>) -> Option<Vec<Vec<f32>>> {
    if got.is_none() {
        fails.push(
            name,
            "resident path never engaged — the comparison would have been the \
             host feed against itself. Either the backend refused to bind (it \
             is outside the verified set and was forced on with \
             RLX_UNLIMITED_OCR_RESIDENT_KV_ANY=1), or a device in the set \
             regressed.",
        );
    }
    got
}

fn compare(fails: &mut Failures, name: &str, what: &str, a: &[Vec<f32>], b: &[Vec<f32>]) {
    assert_eq!(a.len(), b.len(), "step count");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if !x.iter().all(|v| v.is_finite()) {
            fails.push(name, format!("{what}: step {i} produced non-finite logits"));
            return;
        }
        let d = max_rel_diff(x, y);
        if d > 1e-5 {
            fails.push(
                name,
                format!("{what}: step {i} differs by {d:.3e} (resident vs host feed)"),
            );
            return;
        }
    }
}

#[test]
fn resident_kv_matches_the_host_feed() {
    with_env_lock(|| {
        let cfg: UnlimitedOcrConfig = cfg();
        let mut fails = Failures::default();
        for (name, device) in available_devices() {
            if !skip_unverified(name, device) {
                continue;
            }
            let pack = pack_for(&cfg);
            let host = run_steps(&cfg, &pack, device, false, 0.5).expect("host arm");
            let Some(resident) =
                engaged(&mut fails, name, run_steps(&cfg, &pack, device, true, 0.5))
            else {
                continue;
            };
            compare(&mut fails, name, "steps", &host, &resident);
        }
        fails.assert_empty("resident K/V parity");
    });
}

#[test]
fn a_second_prompt_on_the_same_lm_does_not_answer_the_first() {
    with_env_lock(|| {
        let cfg: UnlimitedOcrConfig = cfg();
        let h = cfg.hidden_size;
        let mut fails = Failures::default();
        for (name, device) in available_devices() {
            if !skip_unverified(name, device) {
                continue;
            }
            let pack = pack_for(&cfg);
            // Reference: prompt B alone, on its own CompiledLm.
            let Some(fresh) = engaged(&mut fails, name, run_steps(&cfg, &pack, device, true, 0.8))
            else {
                continue;
            };

            // Same prompt B, but on an LM that already ran prompt A — so the
            // decode graph for B's bucket is already compiled AND already has
            // handles bound from A's cache.
            set_resident(true);
            let mut lm = CompiledLm::new(device, Arc::clone(&pack));
            let a_embeds = fill(PROMPT * h, 0.5);
            let (_, mut kv_a) = lm.prefill(&a_embeds, PROMPT).expect("prefill A");
            let a_steps = fill(STEPS * h, 0.9);
            for i in 0..STEPS {
                lm.decode_step(&a_steps[i * h..(i + 1) * h], PROMPT + i, &mut kv_a)
                    .expect("step A");
            }
            let b_embeds = fill(PROMPT * h, 0.8);
            let (_, mut kv_b) = lm.prefill(&b_embeds, PROMPT).expect("prefill B");
            let b_steps = fill(STEPS * h, 1.2);
            let reused: Vec<Vec<f32>> = (0..STEPS)
                .map(|i| {
                    lm.decode_step(&b_steps[i * h..(i + 1) * h], PROMPT + i, &mut kv_b)
                        .expect("step B")
                })
                .collect();
            compare(&mut fails, name, "second prompt", &fresh, &reused);
        }
        fails.assert_empty("resident K/V across prompts");
    });
}

#[test]
fn a_rollback_rebinds_the_resident_cache() {
    with_env_lock(|| {
        let cfg: UnlimitedOcrConfig = cfg();
        let h = cfg.hidden_size;
        let mut fails = Failures::default();
        for (name, device) in available_devices() {
            if !skip_unverified(name, device) {
                continue;
            }
            let pack = pack_for(&cfg);
            // Reference: two steps, no rollback.
            let Some(two) = engaged(&mut fails, name, run_steps(&cfg, &pack, device, true, 0.5))
            else {
                continue;
            };

            // Same two steps, but after speculatively taking three and dropping
            // back to the prefill — `valid` moves without a decode step, so the
            // handles hold rows the cache no longer admits to.
            set_resident(true);
            let mut lm = CompiledLm::new(device, Arc::clone(&pack));
            let embeds = fill(PROMPT * h, 0.5);
            let (_, mut kv) = lm.prefill(&embeds, PROMPT).expect("prefill");
            let throwaway = fill(3 * h, 2.0);
            for i in 0..3 {
                lm.decode_step(&throwaway[i * h..(i + 1) * h], PROMPT + i, &mut kv)
                    .expect("speculative step");
            }
            kv.rollback_to(PROMPT);
            let steps_in = fill(STEPS * h, 0.9);
            let after: Vec<Vec<f32>> = (0..STEPS)
                .map(|i| {
                    lm.decode_step(&steps_in[i * h..(i + 1) * h], PROMPT + i, &mut kv)
                        .expect("step after rollback")
                })
                .collect();
            compare(&mut fails, name, "after rollback", &two, &after);
        }
        fails.assert_empty("resident K/V after rollback");
    });
}
