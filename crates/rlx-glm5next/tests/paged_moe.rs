// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! A **paged** MoE layer must compute what the resident one computes.
//!
//! Paging replaces the model's `n_routed`-expert banks with a per-token gather
//! of just the fired experts (`rlx_distributed::ExpertPager`), and replaces the
//! router-derived expert index with a constant slot index. That is a real change
//! of formulation, not a cache in front of the same maths, so it needs to be
//! shown equivalent rather than assumed to be.
//!
//! The two ways it goes wrong are both silent:
//!
//!   * **Slot order.** Slot `r * top_k + k` must hold the expert row `r` fired
//!     at position `k`. Gather in any other order and every token is multiplied
//!     by another token's gate — plausible output, wrong model.
//!   * **Gather order vs probability order.** The routing weights stay in the
//!     graph and are applied by position, so the host gather has to agree with
//!     the router's ordering, not merely with its *set* of ids.
//!
//! Both are checked here by building the same layer twice and comparing.

mod common;

use common::{EXPERTS, HIDDEN, MOE_INTER, TOPK, dev, fill};
use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_distributed::{BankLocation, ExpertPager};
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_glm5next::moe::{MoeDims, emit_glm5next_moe};
use rlx_glm5next::{BANKS, PagedMoeLayer};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;
use std::sync::Arc;

const SEQ: usize = 3;
const PREFIX: &str = "blk.0";

fn dims(paged: bool) -> MoeDims {
    MoeDims {
        hidden: HIDDEN,
        moe_inter: MOE_INTER,
        n_routed: EXPERTS,
        top_k: TOPK,
        n_group: 1,
        topk_group: 1,
        routed_scaling: 2.5,
        swiglu_limit: Some(10.0),
        seq: SEQ,
        paged,
    }
}

/// Every tensor one MoE layer needs, with `n_routed`-wide banks.
fn layer_tensors() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 41u64;
    let mut put = |t: &mut HashMap<_, _>, k: String, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        seed += 13;
        t.insert(k, (fill(n, seed), shape));
    };
    put(
        &mut t,
        format!("{PREFIX}.ffn_gate_inp.weight"),
        vec![EXPERTS, HIDDEN],
    );
    t.insert(
        format!("{PREFIX}.exp_probs_b.bias"),
        (vec![0.0; EXPERTS], vec![EXPERTS]),
    );
    // Banks in GGUF orientation: [experts, out, in].
    put(
        &mut t,
        format!("{PREFIX}.ffn_gate_exps.weight"),
        vec![EXPERTS, MOE_INTER, HIDDEN],
    );
    put(
        &mut t,
        format!("{PREFIX}.ffn_up_exps.weight"),
        vec![EXPERTS, MOE_INTER, HIDDEN],
    );
    put(
        &mut t,
        format!("{PREFIX}.ffn_down_exps.weight"),
        vec![EXPERTS, HIDDEN, MOE_INTER],
    );
    // Shared expert.
    put(
        &mut t,
        format!("{PREFIX}.ffn_gate_shexp.weight"),
        vec![MOE_INTER, HIDDEN],
    );
    put(
        &mut t,
        format!("{PREFIX}.ffn_up_shexp.weight"),
        vec![MOE_INTER, HIDDEN],
    );
    put(
        &mut t,
        format!("{PREFIX}.ffn_down_shexp.weight"),
        vec![HIDDEN, MOE_INTER],
    );
    t
}

/// Run one MoE layer over `x`, returning `[SEQ * HIDDEN]`.
fn run_layer(tensors: HashMap<String, (Vec<f32>, Vec<usize>)>, d: MoeDims, x: &[f32]) -> Vec<f32> {
    let mut wm = WeightMap::from_tensors(tensors);
    let shape = Shape::new(&[1, SEQ, HIDDEN], DType::F32);
    let flow = ModelFlow::new("paged_moe_probe")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", shape.clone())
        .plugin_named("moe", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let out = emit_glm5next_moe(emit, PREFIX, x, d)?;
            Ok(Some(emit.wrap(out, shape.clone())))
        })
        .output("y");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build moe layer");
    let mut compiled = compile_built(built, dev()).expect("compile");
    compiled
        .run(&[("x", x)])
        .into_iter()
        .next()
        .expect("moe output")
}

/// Which experts the router fires, in the order it fires them.
///
/// Recomputed here from the same weights rather than read out of the graph:
/// a host that pages has to be able to derive the routing itself, and if this
/// disagrees with the graph's router the equivalence test below fails — which
/// is the point.
fn routing(tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>, x: &[f32]) -> Vec<Vec<usize>> {
    let (gate_w, gshape) = &tensors[&format!("{PREFIX}.ffn_gate_inp.weight")];
    assert_eq!(gshape, &vec![EXPERTS, HIDDEN]);
    let (bias, _) = &tensors[&format!("{PREFIX}.exp_probs_b.bias")];

    (0..SEQ)
        .map(|r| {
            let row = &x[r * HIDDEN..(r + 1) * HIDDEN];
            let mut scored: Vec<(usize, f32)> = (0..EXPERTS)
                .map(|e| {
                    let logit: f32 = (0..HIDDEN).map(|i| gate_w[e * HIDDEN + i] * row[i]).sum();
                    let sig = 1.0 / (1.0 + (-logit).exp());
                    (e, sig + bias[e])
                })
                .collect();
            // One group, so group-limited top-k degenerates to plain top-k.
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            scored.into_iter().take(TOPK).map(|(e, _)| e).collect()
        })
        .collect()
}

/// Rebuild the three banks as `SEQ * TOPK` gathered slots — what an
/// [`rlx_distributed::ExpertPager`] hands over, done here in f32 so the
/// comparison isolates the slot formulation from quantization.
fn gathered(
    tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    fired: &[Vec<usize>],
) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut out = tensors.clone();
    let slots = SEQ * TOPK;
    for (bank, rows_out, cols_in) in [
        ("ffn_gate_exps", MOE_INTER, HIDDEN),
        ("ffn_up_exps", MOE_INTER, HIDDEN),
        ("ffn_down_exps", HIDDEN, MOE_INTER),
    ] {
        let key = format!("{PREFIX}.{bank}.weight");
        let (data, shape) = &tensors[&key];
        assert_eq!(shape, &vec![EXPERTS, rows_out, cols_in]);
        let per = rows_out * cols_in;
        let mut g = Vec::with_capacity(slots * per);
        for (r, ids) in fired.iter().enumerate() {
            assert_eq!(ids.len(), TOPK, "row {r} fired {} experts", ids.len());
            for &id in ids {
                g.extend_from_slice(&data[id * per..(id + 1) * per]);
            }
        }
        out.insert(key, (g, vec![slots, rows_out, cols_in]));
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The paged layer equals the resident layer.
#[test]
fn a_paged_moe_layer_matches_the_resident_one() {
    let tensors = layer_tensors();
    let x = fill(SEQ * HIDDEN, 909);

    let want = run_layer(tensors.clone(), dims(false), &x);
    assert!(
        want.iter().all(|v| v.is_finite()),
        "reference is not finite"
    );

    let fired = routing(&tensors, &x);
    let got = run_layer(gathered(&tensors, &fired), dims(true), &x);

    let d = max_abs_diff(&want, &got);
    assert!(
        d <= 1e-5,
        "paged MoE differs from resident by {d:.3e}; routing was {fired:?}"
    );
}

/// Slot order is load-bearing. Reversing each row's fired experts keeps the
/// same SET of experts and the same bank bytes, and pairs every one with the
/// wrong gate — so a test that only checked which experts were gathered would
/// pass while the model was wrong.
#[test]
fn the_slot_order_must_match_the_routing_order() {
    let tensors = layer_tensors();
    let x = fill(SEQ * HIDDEN, 909);
    let want = run_layer(tensors.clone(), dims(false), &x);

    let mut fired = routing(&tensors, &x);
    // Only meaningful if the router actually fired distinct experts.
    assert!(
        fired.iter().any(|ids| ids[0] != ids[TOPK - 1]),
        "every row fired the same expert twice; the permutation below is a no-op"
    );
    for ids in &mut fired {
        ids.reverse();
    }
    let got = run_layer(gathered(&tensors, &fired), dims(true), &x);
    assert!(
        max_abs_diff(&want, &got) > 1e-5,
        "permuting the gathered slots changed nothing — the slot index is not \
         actually selecting per-position experts, so a real gather could be \
         mis-ordered and never be caught"
    );
}

/// Gathering the same expert for every slot must also change the answer —
/// otherwise the banks are not being read through the slot index at all.
#[test]
fn the_gathered_banks_are_actually_used() {
    let tensors = layer_tensors();
    let x = fill(SEQ * HIDDEN, 909);
    let want = run_layer(tensors.clone(), dims(false), &x);

    let fired: Vec<Vec<usize>> = (0..SEQ).map(|_| vec![0usize; TOPK]).collect();
    let got = run_layer(gathered(&tensors, &fired), dims(true), &x);
    assert!(
        max_abs_diff(&want, &got) > 1e-5,
        "replacing every fired expert with expert 0 left the output unchanged"
    );
}

/// What paging is for, and its exact condition.
///
/// A paged bank holds `seq * top_k` slots against the resident bank's
/// `n_routed`, so it only *saves* while `seq * top_k < n_routed`. That is the
/// decode regime — one token, `top_k` experts — and it is where the planner
/// applies it: GLM-5.3-Flash decode reads 8 of 288 experts, 0.05 GB against
/// 1.88 GB per layer. A long prefill fires most of the bank and should stay
/// resident, which is worth stating rather than discovering.
#[test]
fn a_paged_layer_holds_seq_times_top_k_experts() {
    let tensors = layer_tensors();
    let x = fill(SEQ * HIDDEN, 909);
    let fired = routing(&tensors, &x);
    let paged = gathered(&tensors, &fired);

    let per_expert = MOE_INTER * HIDDEN;
    let resident = tensors[&format!("{PREFIX}.ffn_gate_exps.weight")].0.len();
    let held = paged[&format!("{PREFIX}.ffn_gate_exps.weight")].0.len();
    assert_eq!(resident, EXPERTS * per_expert);
    assert_eq!(
        held,
        SEQ * TOPK * per_expert,
        "a paged bank must be exactly the fired slots"
    );

    // The decode case, which is the one that pays.
    let decode_slots = TOPK;
    assert!(
        decode_slots < EXPERTS,
        "top_k {TOPK} of {EXPERTS} experts: paging saves \
         {:.0}% of the bank per decode step",
        100.0 * (1.0 - decode_slots as f64 / EXPERTS as f64)
    );
}

// ─────────────── end to end: disk → pager → graph ───────────────

/// Write the routed banks to a file and register them with a pager, the way a
/// checkpoint on local storage would be.
fn bank_file(
    name: &str,
    tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> (std::path::PathBuf, ExpertPager) {
    use std::io::Write;
    // Per-test path: these run in parallel and each removes its own file.
    let dir = std::env::temp_dir().join(format!("rlx-glm5next-paged-{name}"));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("banks.bin");
    let mut f = std::fs::File::create(&path).unwrap();

    let mut pager = ExpertPager::new(4 << 20);
    let mut offset = 0u64;
    for bank in BANKS {
        let (data, shape) = &tensors[&format!("{PREFIX}.{bank}.weight")];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        f.write_all(&bytes).unwrap();
        let per = bytes.len() / shape[0];
        pager
            .register(
                0,
                bank,
                BankLocation {
                    path: path.clone(),
                    offset,
                    num_experts: shape[0],
                    bytes_per_expert: per,
                },
            )
            .unwrap();
        offset += bytes.len() as u64;
    }
    f.flush().unwrap();
    (path, pager)
}

/// The whole loop: route on the host, page the fired experts off disk, run the
/// slot-indexed graph — and get what the resident layer computes.
///
/// This is the claim the cluster planner has been making all along whenever it
/// reports a stage as `paged`. Each piece was tested on its own; what this adds
/// is that they compose, and in particular that the host router agrees with the
/// graph's router. A single differing pick would pair a token with another
/// expert's weights and still produce finite, plausible output.
#[test]
fn paging_from_disk_reproduces_the_resident_layer() {
    let tensors = layer_tensors();
    let x = fill(SEQ * HIDDEN, 909);
    let want = run_layer(tensors.clone(), dims(false), &x);

    let (path, pager) = bank_file("e2e", &tensors);
    let mut layer = PagedMoeLayer::new(
        PREFIX,
        0,
        dims(true),
        dev(),
        tensors.clone(),
        Arc::new(pager),
    )
    .expect("build paged layer");

    let got = layer.forward(&x).expect("paged forward");
    let d = max_abs_diff(&want, &got);
    assert!(
        d <= 1e-5,
        "paging off disk differs from the resident layer by {d:.3e}"
    );

    // It really went to disk — and read each expert once, not once per slot.
    // Several rows commonly fire the same expert, and the cache collapses those
    // within a single gather as well as across tokens; the planner's IO term is
    // an upper bound that this comes in under.
    let s = layer.stats();
    assert!(s.bytes_read > 0, "nothing was read from storage");
    let fired = layer.route(&x).unwrap();
    let distinct: std::collections::BTreeSet<usize> = fired.iter().flatten().copied().collect();
    let slots = SEQ * TOPK;
    assert_eq!(
        s.misses,
        (distinct.len() * BANKS.len()) as u64,
        "expected one disk read per DISTINCT expert per bank; fired {fired:?}"
    );
    assert!(
        s.misses <= (slots * BANKS.len()) as u64,
        "more reads than fired slots"
    );
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// The host router must agree with the graph's, expert for expert and in order.
///
/// Checked against the same routing the equivalence test derives independently,
/// so a divergence in either shows up.
#[test]
fn the_host_router_agrees_with_the_graphs() {
    let tensors = layer_tensors();
    let x = fill(SEQ * HIDDEN, 909);
    let (path, pager) = bank_file("router", &tensors);
    let mut layer = PagedMoeLayer::new(
        PREFIX,
        0,
        dims(true),
        dev(),
        tensors.clone(),
        Arc::new(pager),
    )
    .expect("build");

    assert_eq!(
        layer.route(&x).expect("route"),
        routing(&tensors, &x),
        "the router picks different experts than the reference routing"
    );
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// A second token over the same input must be served from the cache rather than
/// re-read — the reuse the planner's `io_mbps` term is discounted by.
#[test]
fn re_firing_the_same_experts_stops_touching_disk() {
    let tensors = layer_tensors();
    let x = fill(SEQ * HIDDEN, 909);
    let (path, pager) = bank_file("reuse", &tensors);
    let mut layer =
        PagedMoeLayer::new(PREFIX, 0, dims(true), dev(), tensors, Arc::new(pager)).expect("build");

    layer.forward(&x).unwrap();
    let after_first = layer.stats().bytes_read;
    layer.forward(&x).unwrap();
    assert_eq!(
        layer.stats().bytes_read,
        after_first,
        "the second pass re-read experts it already held"
    );
    assert!(
        layer.stats().hit_rate() > 0.4,
        "{}",
        layer.stats().hit_rate()
    );
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// Experts must land directly in the graph's arena, not via an assembled bank.
///
/// The fallback path is correct, so if the backend refuses sub-range writes
/// every test above still passes while every active byte is copied twice —
/// ~2 GB per token of redundant memcpy at GLM-5.3-Flash scale. The only way to
/// tell the two apart is to count.
#[test]
fn experts_are_written_straight_into_the_arena() {
    let tensors = layer_tensors();
    let x = fill(SEQ * HIDDEN, 909);
    let (path, pager) = bank_file("partial", &tensors);
    let mut layer =
        PagedMoeLayer::new(PREFIX, 0, dims(true), dev(), tensors, Arc::new(pager)).expect("build");

    layer.forward(&x).unwrap();
    assert_eq!(
        layer.whole_bank_uploads(),
        0,
        "every bank was uploaded whole — the per-slot write is not being taken, \
         so each active byte is copied twice"
    );
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

// ─────────────────────────── precision ───────────────────────────

/// Paged and resident must agree to f32 rounding across many inputs.
///
/// Not bit-exact, and the reason is worth stating: the segmented GEMM groups
/// tokens by expert, and a `seq * top_k`-slot bank groups them differently from
/// an `n_routed`-expert one, so the two sum the same products in different
/// orders. Measured at 1 ULP on this fixture.
///
/// The tolerance is set far below the failure it has to catch. Routing is a
/// top-k comparison; if the paged path ever selected a different expert the
/// output would move by a large fraction of itself, not by an ULP. So 1e-6
/// relative separates "same decisions, different summation order" from "wrong
/// expert" with four orders of magnitude to spare — where the original 1e-5
/// single-input check had less room and only one draw.
#[test]
fn paged_and_resident_agree_to_rounding_across_inputs() {
    let tensors = layer_tensors();
    for seed in 0..24u64 {
        let x = fill(SEQ * HIDDEN, 4000 + seed);
        let want = run_layer(tensors.clone(), dims(false), &x);
        let fired = routing(&tensors, &x);
        let got = run_layer(gathered(&tensors, &fired), dims(true), &x);

        let scale = want.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        let rel = max_abs_diff(&want, &got) / scale;
        assert!(
            rel < 1e-6,
            "seed {seed}: paged differs from resident by {rel:.3e} relative — \
             an ULP is ~1e-7, so this is a different expert, not a different \
             summation order"
        );
    }
}

/// Top-k selection can turn on summation order alone.
///
/// This is why the fired experts come from a compiled graph rather than a host
/// recomputation ([`PagedMoeLayer::route`]): a second implementation of the
/// routing is correct only while it happens to accumulate the way the graph's
/// lowering does, and a BLAS threshold or a device change moves that silently.
///
/// Constructed rather than sampled, so the outcome does not depend on how many
/// random draws it took. Two experts whose logits are EQUAL in exact
/// arithmetic; summing their products front-to-back and back-to-front — both
/// valid — selects different experts.
#[test]
fn top_k_selection_can_turn_on_summation_order_alone() {
    // 1 + big - big. `big` is past f32's integer range, so `big + 1 == big` and
    // the 1 survives only if the cancellation happens first.
    let big = 1.0e8f32;
    let row_a = [1.0f32, big, -big];
    let row_b = [1.0f32, 0.0, 0.0];
    let x = [1.0f32; 3];

    let fwd = |r: &[f32; 3]| r.iter().zip(&x).fold(0.0f32, |acc, (w, v)| acc + w * v);
    let rev = |r: &[f32; 3]| {
        r.iter()
            .zip(&x)
            .rev()
            .fold(0.0f32, |acc, (w, v)| acc + w * v)
    };
    // The gate's rule: highest score wins, ties to the lower expert id.
    let pick = |a: f32, b: f32| -> usize { if a >= b { 0 } else { 1 } };

    // Exact arithmetic: both are 1.0, so the tie-break should select expert 0.
    let exact_a = 1.0f64 + big as f64 - big as f64;
    let exact_b = 1.0f64;
    assert_eq!(exact_a, exact_b, "the two experts must tie exactly");

    let (fa, fb) = (fwd(&row_a), fwd(&row_b));
    let (ra, rb) = (rev(&row_a), rev(&row_b));
    assert_ne!(
        pick(fa, fb),
        pick(ra, rb),
        "front-to-back ({fa}, {fb}) and back-to-front ({ra}, {rb}) select the \
         same expert, so this case no longer demonstrates the hazard and needs \
         rebuilding rather than deleting"
    );
    // And one of them contradicts exact arithmetic, which is the point.
    assert_eq!(
        pick(ra, rb),
        0,
        "back-to-front agrees with exact arithmetic"
    );
    assert_eq!(pick(fa, fb), 1, "front-to-back does not");
}
