// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
#![cfg(feature = "vision")]
//
//! The native vision tower against an independent implementation of the same
//! architecture.
//!
//! `rlx-qwen35` carries the Qwen3-VL ViT too, and its captions are known-good
//! on real Qwen3.6-27B weights. That makes it an **oracle**: feeding both towers
//! byte-identical synthetic weights and comparing element for element is what
//! distinguishes a correct port from a plausible one.
//!
//! `rlx-qwen35` is a **dev-dependency only**; nothing in `src/` touches it.
//!
//! ## What this does and does not resolve
//!
//! Verified by sabotage — each of these breaks the comparison:
//!
//! * dropping the second patch convolution (0.75 relative, 375x the tolerance);
//! * leaving the position table in raster order instead of applying the merge
//!   permutation (0.076, 38x).
//!
//! Two things it is blind to, and where they *are* covered:
//!
//! * **The cos/sin tables themselves.** Both towers are fed the same
//!   [`vision_rope_tables`] output, so only the *application* of the rotation
//!   is compared here. Swapping the row and column axes inside the table
//!   generator passes this test; it is caught by `vit`'s own
//!   `the_axial_rope_duplicates_its_first_half`, which checks the layout
//!   against the closed form.
//! * **Which GELU is which.** The exact erf and tanh approximations agree to
//!   well under this tolerance at any sane activation magnitude, so swapping
//!   the merger's `Gelu` for `GeluApprox` also passes. That choice rests on
//!   reading the reference (`nn.GELU()` default vs `gelu_pytorch_tanh`), not on
//!   this test.
//!
//! The two implementations also differ in how weights reach the device — this
//! crate goes through a `WeightSource` (so a quantized mmproj runs packed),
//! `rlx-qwen35` registers transposed `Vec<f32>` params and stores matmul
//! weights as F16 by default. `RLX_QWEN35_VISION_F16=0` puts the oracle on F32
//! so the comparison is about the *math*, not the storage.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_qwen4exp::synth::fill;
use rlx_qwen4exp::vit::{
    IMAGE_INPUT, VISION_ROPE_COS_INPUT, VISION_ROPE_SIN_INPUT, VisionConfig,
    build_qwen4exp_vision_flow, spatial_merge_gather_idx, vision_position_hw, vision_rope_tables,
    vision_tensor_shapes,
};
use rlx_qwen35::{MmProjConfig, MmProjWeights, VisionBlockWeights};
use rlx_runtime::Device;
use std::collections::HashMap;

/// The published tower's shape, shrunk. Everything that matters structurally is
/// preserved: dual patch kernels, `merge = 2`, a head wide enough for the axial
/// RoPE's four-way split, and a learned position table on a square grid that
/// does **not** match the test's patch grid (so the bilinear resize runs).
const NUM_POSITION_EMBEDDINGS: usize = 36; // 6x6 source grid
const IMG_W: usize = 16;
const IMG_H: usize = 24;

fn cfg() -> VisionConfig {
    VisionConfig {
        patch_size: 4,
        n_embd: 16,
        n_head: 2,
        n_layer: 3,
        image_size: 32,
        n_ff: 24,
        n_merge: 2,
        eps: 1e-6,
        projector_type: "qwen3vl_merger".into(),
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        llm_hidden_size: 20,
        image_min_pixels: 64,
        image_max_pixels: 4096,
        rope_theta: 10_000.0,
        deepstack_layers: vec![],
    }
}

/// The oracle's config, field for field.
fn oracle_cfg(c: &VisionConfig) -> MmProjConfig {
    MmProjConfig {
        patch_size: c.patch_size,
        n_embd: c.n_embd,
        n_head: c.n_head,
        n_layer: c.n_layer,
        image_size: c.image_size,
        image_min_pixels: c.image_min_pixels,
        image_max_pixels: c.image_max_pixels,
        n_merge: c.n_merge,
        eps: c.eps as f64,
        projector_type: c.projector_type.clone(),
        image_mean: c.image_mean,
        image_std: c.image_std,
        spatial_merge_size: c.n_merge,
        llm_hidden_size: c.llm_hidden_size,
        n_ff: c.n_ff,
        deepstack_layers: vec![],
    }
}

/// Synthesize the tower's tensors in the orientation the GGUF loader reports,
/// which is what both implementations read.
fn tensors(c: &VisionConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut out = HashMap::new();
    let mut seed = 31u64;
    for (name, shape) in vision_tensor_shapes(c, NUM_POSITION_EMBEDDINGS) {
        let n: usize = shape.iter().product();
        seed = seed.wrapping_add(17);
        // LayerNorm gains centre on 1 so the activations keep a sane scale.
        let data = if name.ends_with("ln1.weight")
            || name.ends_with("ln2.weight")
            || name.ends_with("post_ln.weight")
        {
            fill(n, seed).iter().map(|x| 1.0 + x).collect()
        } else {
            fill(n, seed)
        };
        out.insert(name, (data, shape));
    }
    out
}

fn get<'a>(t: &'a HashMap<String, (Vec<f32>, Vec<usize>)>, k: &str) -> &'a [f32] {
    &t.get(k).unwrap_or_else(|| panic!("{k} missing")).0
}

/// Repack the same tensors into the oracle's weight bundle.
///
/// The oracle's fields hold the tensors exactly as its GGUF loader hands them
/// over — the same orientation as here — so this is a copy, not a conversion.
/// Two mappings are not one-to-one and are the reason this helper exists:
/// `ffn_up` is the oracle's *gate* slot (its non-gated path reads fc1 from
/// there), and `v.post_ln` goes into `mm_norm` with both `has_*_ln` false,
/// because that LayerNorm is the merger's, not a post-block one.
fn oracle_weights(c: &VisionConfig, t: &HashMap<String, (Vec<f32>, Vec<usize>)>) -> MmProjWeights {
    let n = c.n_embd;
    let blocks = (0..c.n_layer)
        .map(|il| {
            let p = format!("v.blk.{il}");
            VisionBlockWeights {
                ln1_w: get(t, &format!("{p}.ln1.weight")).to_vec(),
                ln1_b: get(t, &format!("{p}.ln1.bias")).to_vec(),
                ln2_w: get(t, &format!("{p}.ln2.weight")).to_vec(),
                ln2_b: get(t, &format!("{p}.ln2.bias")).to_vec(),
                qkv_w: get(t, &format!("{p}.attn_qkv.weight")).to_vec(),
                qkv_b: get(t, &format!("{p}.attn_qkv.bias")).to_vec(),
                attn_out_w: get(t, &format!("{p}.attn_out.weight")).to_vec(),
                attn_out_b: get(t, &format!("{p}.attn_out.bias")).to_vec(),
                // `ffn_up` IS fc1 for the non-gated MLP.
                ffn_gate_w: get(t, &format!("{p}.ffn_up.weight")).to_vec(),
                ffn_gate_b: get(t, &format!("{p}.ffn_up.bias")).to_vec(),
                ffn_up_w: Vec::new(),
                ffn_up_b: Vec::new(),
                ffn_down_w: get(t, &format!("{p}.ffn_down.weight")).to_vec(),
                ffn_down_b: get(t, &format!("{p}.ffn_down.bias")).to_vec(),
                ffn_gated: false,
                deepstack: None,
            }
        })
        .collect();
    MmProjWeights {
        patch_embd_0: get(t, "v.patch_embd.weight").to_vec(),
        patch_embd_1: get(t, "v.patch_embd.weight.1").to_vec(),
        patch_bias: get(t, "v.patch_embd.bias").to_vec(),
        position_embd: get(t, "v.position_embd.weight").to_vec(),
        pre_ln_w: vec![1.0; n],
        pre_ln_b: vec![0.0; n],
        post_ln_w: vec![1.0; n],
        post_ln_b: vec![0.0; n],
        has_pre_ln: false,
        has_post_ln: false,
        // The single `v.post_ln` is the MERGER norm, applied once.
        mm_norm_w: get(t, "v.post_ln.weight").to_vec(),
        mm_norm_b: get(t, "v.post_ln.bias").to_vec(),
        mm_0_w: get(t, "mm.0.weight").to_vec(),
        mm_0_b: get(t, "mm.0.bias").to_vec(),
        mm_1_w: get(t, "mm.2.weight").to_vec(),
        mm_1_b: get(t, "mm.2.bias").to_vec(),
        blocks,
    }
}

fn dev() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

/// A normalized NCHW image whose values vary across every axis, so a
/// transposed or mis-permuted read cannot coincide.
fn image() -> Vec<f32> {
    (0..3 * IMG_H * IMG_W)
        .map(|i| ((i as f32) * 0.019).sin() * 0.8)
        .collect()
}

#[test]
fn the_native_tower_matches_the_qwen35_implementation() {
    // Put the oracle on F32 weights: this is a test of the math, not of F16
    // storage. (It defaults to F16 to halve resident vision weights.)
    rlx_ir::env::set("RLX_QWEN35_VISION_F16", "0");

    let c = cfg();
    let oc = oracle_cfg(&c);
    let t = tensors(&c);
    let (wp, hp) = c.grid(IMG_W, IMG_H);
    let n_pos = wp * hp;
    let n_out = c.n_out_tokens(IMG_W, IMG_H);
    assert_eq!((wp, hp), (4, 6), "the grid must not be square");
    assert_ne!(
        n_pos, NUM_POSITION_EMBEDDINGS,
        "the position table must need resizing, or the bilinear path is untested"
    );

    let img = image();
    let position_hw = vision_position_hw(&c, IMG_W, IMG_H);
    let (cos, sin) = vision_rope_tables(&c, &position_hw);
    assert_eq!(cos.len(), n_pos * c.head_dim());

    // ── this crate ──
    let mine = {
        let mut w = WeightMap::from_tensors(t.clone());
        let built =
            build_qwen4exp_vision_flow(&c, &mut w, IMG_W, IMG_H).expect("build the native tower");
        let leftover: Vec<String> = w.keys().map(str::to_string).collect();
        assert!(
            leftover.is_empty(),
            "the native tower left mmproj tensors unread: {leftover:?}"
        );
        compile_built(built, dev())
            .expect("compile the native tower")
            .run(&[
                (IMAGE_INPUT, img.as_slice()),
                (VISION_ROPE_COS_INPUT, cos.as_slice()),
                (VISION_ROPE_SIN_INPUT, sin.as_slice()),
            ])
            .pop()
            .expect("native output")
    };

    // ── the oracle ──
    let theirs = {
        let ow = oracle_weights(&c, &t);
        let built = rlx_qwen35::build_qwen35_vision_built(&oc, &ow, IMG_W, IMG_H)
            .expect("build the oracle tower");
        compile_built(built, dev())
            .expect("compile the oracle tower")
            .run(&[
                ("image", img.as_slice()),
                ("vision_rope_cos", cos.as_slice()),
                ("vision_rope_sin", sin.as_slice()),
            ])
            .pop()
            .expect("oracle output")
    };

    assert_eq!(mine.len(), n_out * c.llm_hidden_size);
    assert_eq!(theirs.len(), mine.len(), "both towers must agree on shape");
    assert!(mine.iter().all(|v| v.is_finite()));

    let sd = {
        let m = theirs.iter().sum::<f32>() / theirs.len() as f32;
        (theirs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / theirs.len() as f32).sqrt()
    };
    assert!(
        sd > 1e-4,
        "the oracle output is degenerate (sd {sd}); nothing would be compared"
    );

    let max_abs = mine
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs / sd < 2e-3,
        "the native tower and the rlx-qwen35 oracle differ by {max_abs} absolute \
         ({:.3e} relative to a {sd} output sd)",
        max_abs / sd
    );
}

/// A non-square, non-uniform image is the case where a permutation mistake
/// shows up; a square grid can hide a row/column swap.
#[test]
fn the_merge_permutation_and_the_position_table_agree() {
    let c = cfg();
    let (wp, hp) = c.grid(IMG_W, IMG_H);
    let idx = spatial_merge_gather_idx(hp, wp, c.n_merge);
    assert_eq!(idx.len(), wp * hp);

    // Every patch appears exactly once.
    let mut seen = vec![false; wp * hp];
    for &i in &idx {
        let i = i as usize;
        assert!(!seen[i], "patch {i} appears twice in the permutation");
        seen[i] = true;
    }
    assert!(seen.iter().all(|s| *s));

    // The RoPE coordinates must be generated in the permuted order, so token
    // `k` has the coordinates of patch `idx[k]`.
    let hw = vision_position_hw(&c, IMG_W, IMG_H);
    for (k, &i) in idx.iter().enumerate() {
        let i = i as usize;
        assert_eq!(hw[k], ((i / wp) as i32, (i % wp) as i32), "token {k}");
    }

    // And each group of merge² consecutive tokens must be one 2x2 block — this
    // is what makes the merger's reshape a reshape.
    for blk in idx.chunks(c.merge_sq()) {
        let rows: Vec<usize> = blk.iter().map(|&i| i as usize / wp).collect();
        let cols: Vec<usize> = blk.iter().map(|&i| i as usize % wp).collect();
        let (r0, c0) = (*rows.iter().min().unwrap(), *cols.iter().min().unwrap());
        for (r, cc) in rows.iter().zip(&cols) {
            assert!(
                r - r0 < c.n_merge && cc - c0 < c.n_merge,
                "block is not contiguous"
            );
        }
    }
}
