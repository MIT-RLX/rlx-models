// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Experts paged off disk against the same layer with all of them resident.
//!
//! This is the claim: reading the 10 experts a token fires computes what
//! holding all 512 computes. Every piece is independently plausible and wrong
//! in a way that still produces finite, well-scaled output — a slot written in
//! the wrong order pairs a token with another token's gate; a router that
//! disagrees with the graph's by one pick weights a token by another expert's
//! matrix. So the test is the whole loop against the resident layer, not the
//! pieces.

use std::collections::HashMap;
use std::sync::Arc;

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_distributed::{BankLocation, ExpertPager, register_gguf_expert_banks};
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_gguf::GgmlType;
use rlx_ir::{DType, Shape};
use rlx_qwen4exp::moe::{MoeDims, emit_moe};
use rlx_qwen4exp::paged_moe::{BANKS, PagedMoeLayer};
use rlx_runtime::Device;

const PREFIX: &str = "blk.0";

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

/// Small enough to run anywhere, wide enough that a wrong expert shows.
fn dims(seq: usize) -> MoeDims {
    MoeDims {
        hidden: 16,
        expert_ffn: 12,
        shared_ffn: 20,
        num_experts: 8,
        top_k: 3,
        seq,
        paged: false,
    }
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.6
        })
        .collect()
}

/// Every tensor the layer reads, banks included, in GGUF orientation.
fn tensors(d: MoeDims) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let (h, ff, n, sh) = (d.hidden, d.expert_ffn, d.num_experts, d.shared_ffn);
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    // `[out, in]` as the GGUF loader reports a 2-D projection.
    t.insert(
        format!("{PREFIX}.ffn_gate_inp.weight"),
        (fill(n * h, 1), vec![n, h]),
    );
    t.insert(
        format!("{PREFIX}.ffn_gate_inp_shexp.weight"),
        (fill(h, 2), vec![h]),
    );
    t.insert(
        format!("{PREFIX}.ffn_gate_shexp.weight"),
        (fill(sh * h, 3), vec![sh, h]),
    );
    t.insert(
        format!("{PREFIX}.ffn_up_shexp.weight"),
        (fill(sh * h, 4), vec![sh, h]),
    );
    t.insert(
        format!("{PREFIX}.ffn_down_shexp.weight"),
        (fill(h * sh, 5), vec![h, sh]),
    );
    // `[experts, out, in]`, which is what the GGUF's trailing expert axis
    // becomes once the loader reverses the dims.
    t.insert(
        format!("{PREFIX}.ffn_gate_exps.weight"),
        (fill(n * ff * h, 6), vec![n, ff, h]),
    );
    t.insert(
        format!("{PREFIX}.ffn_up_exps.weight"),
        (fill(n * ff * h, 7), vec![n, ff, h]),
    );
    t.insert(
        format!("{PREFIX}.ffn_down_exps.weight"),
        (fill(n * h * ff, 8), vec![n, h, ff]),
    );
    t
}

/// The layer with all `num_experts` banks resident — the oracle.
fn resident(d: MoeDims, t: &HashMap<String, (Vec<f32>, Vec<usize>)>, x: &[f32]) -> Vec<f32> {
    let mut wm = WeightMap::from_tensors(t.clone());
    let shape = Shape::new(&[1, d.seq, d.hidden], DType::F32);
    let s = shape.clone();
    let built = ModelFlow::new("qwen4exp_resident_moe")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", shape)
        .plugin_named("moe", move |emit, _prev| {
            let inp = emit.flow_input("x")?.hir_id();
            let out = emit_moe(emit, PREFIX, inp, d)?;
            Ok(Some(emit.wrap(out, s.clone())))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build resident moe");
    compile_built(built, dev())
        .expect("compile resident moe")
        .run(&[("x", x)])
        .pop()
        .expect("resident output")
}

/// Write the banks to a file and register them, as local storage would hold them.
fn bank_file(
    tag: &str,
    t: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> (std::path::PathBuf, ExpertPager) {
    use std::io::Write;
    let dir = std::env::temp_dir().join(format!("rlx-qwen4exp-paged-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path = dir.join("banks.bin");
    let mut f = std::fs::File::create(&path).expect("create banks");

    let mut pager = ExpertPager::new(4 << 20);
    let mut offset = 0u64;
    for bank in BANKS {
        let (data, shape) = &t[&format!("{PREFIX}.{bank}.weight")];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        f.write_all(&bytes).expect("write bank");
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
            .expect("register bank");
        offset += bytes.len() as u64;
    }
    f.flush().expect("flush");
    (path, pager)
}

/// The whole loop, against the resident layer.
#[test]
fn paging_the_fired_experts_reproduces_the_resident_layer() {
    for seq in [1usize, 3] {
        let d = dims(seq);
        let t = tensors(d);
        let x = fill(seq * d.hidden, 100 + seq as u64);

        let want = resident(d, &t, &x);
        assert_eq!(want.len(), seq * d.hidden);
        assert!(want.iter().all(|v| v.is_finite()));

        let (path, pager) = bank_file(&format!("resident-{seq}"), &t);
        let mut layer = PagedMoeLayer::new(PREFIX, 0, d, dev(), t.clone(), Arc::new(pager))
            .expect("build paged layer");
        let got = layer.forward(&x).expect("paged forward");

        assert_eq!(got.len(), want.len());
        let max_abs = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = want.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
        assert!(
            max_abs / scale < 1e-5,
            "seq {seq}: paged and resident layers differ by {max_abs} \
             ({:.3e} relative to a {scale} scale)",
            max_abs / scale
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}

/// The router's ids are in range and shaped as the slot contract needs.
#[test]
fn routing_returns_one_row_of_top_k_ids_per_position() {
    let d = dims(3);
    let t = tensors(d);
    let x = fill(3 * d.hidden, 7);
    let (path, pager) = bank_file("route", &t);
    let mut layer =
        PagedMoeLayer::new(PREFIX, 0, d, dev(), t.clone(), Arc::new(pager)).expect("build");
    let fired = layer.route(&x).expect("route");
    assert_eq!(fired.len(), 3);
    for row in &fired {
        assert_eq!(row.len(), d.top_k);
        assert!(row.iter().all(|&e| e < d.num_experts), "{row:?}");
        let mut sorted = row.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), row.len(), "an expert fired twice: {row:?}");
    }
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// What the layer reads per token against what holding the banks would cost.
#[test]
fn the_paged_layer_reads_top_k_of_the_banks_not_all_of_them() {
    let d = dims(1);
    let t = tensors(d);
    let (path, pager) = bank_file("cost", &t);
    let layer = PagedMoeLayer::new(PREFIX, 0, d, dev(), t.clone(), Arc::new(pager)).expect("build");
    let per_expert = 2 * d.expert_ffn * d.hidden + d.hidden * d.expert_ffn;
    assert_eq!(layer.bytes_per_token(), d.top_k * per_expert * 4);
    assert_eq!(layer.resident_bytes(), d.num_experts * per_expert * 4);
    // The ratio is the whole point: top_k of num_experts.
    assert_eq!(
        layer.resident_bytes() / layer.bytes_per_token(),
        d.num_experts / d.top_k
    );
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// A bank whose experts are a different size than the graph's slots is caught
/// at build time, not mid-forward.
#[test]
fn a_bank_with_the_wrong_expert_size_is_refused() {
    let d = dims(1);
    let t = tensors(d);
    let dir = std::env::temp_dir().join(format!("rlx-qwen4exp-paged-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path = dir.join("banks.bin");
    std::fs::write(&path, vec![0u8; 1 << 16]).expect("write");

    let mut pager = ExpertPager::new(1 << 20);
    for bank in BANKS {
        pager
            .register(
                0,
                bank,
                BankLocation {
                    path: path.clone(),
                    offset: 0,
                    num_experts: d.num_experts,
                    // Deliberately wrong: half the expert.
                    bytes_per_expert: d.expert_ffn * d.hidden * 2,
                },
            )
            .expect("register");
    }
    let err = PagedMoeLayer::new(PREFIX, 0, d, dev(), t, Arc::new(pager))
        .err()
        .expect("a mis-sized bank must be refused");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("bytes_per_expert") || msg.contains("byte experts"),
        "{msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `forward_timed` agrees with `forward`, and the fast upload path is the one
/// actually taken.
///
/// Both accessors exist to diagnose a paging stall, which means they are
/// exactly the kind of thing that rots untested. The upload count is the one
/// that matters: a backend that cannot write a sub-range of a param makes every
/// token assemble whole banks instead, copying the active bytes twice, and the
/// only visible symptom is that it is slower.
#[test]
fn the_timed_entry_point_agrees_and_the_uploads_are_per_slot() {
    let d = dims(2);
    let t = tensors(d);
    let x = fill(2 * d.hidden, 31);
    let (path, pager) = bank_file("timed", &t);
    let mut layer =
        PagedMoeLayer::new(PREFIX, 0, d, dev(), t.clone(), Arc::new(pager)).expect("build");

    let plain = layer.forward(&x).expect("forward");
    let (timed, phases) = layer.forward_timed(&x).expect("forward_timed");
    assert_eq!(plain, timed, "the two entry points must agree");
    assert!(
        phases.route + phases.page + phases.compute > std::time::Duration::ZERO,
        "no phase was timed"
    );
    assert_eq!(
        layer.whole_bank_uploads(),
        0,
        "experts should go straight into their arena slots; a non-zero count \
         means this backend fell back to assembling whole banks"
    );
    assert!(layer.stats().bytes_read > 0, "nothing was read from disk");
    assert_eq!(layer.layer(), 0);
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

// ──────────────── quantized banks: the only form the model ships in ────────────────

/// Write the banks into a GGUF at `dtype`, as a real checkpoint holds them.
fn quantized_gguf(
    tag: &str,
    t: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    dtype: GgmlType,
) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("rlx-qwen4exp-qbank-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let path = dir.join("banks.gguf");
    let mut w = rlx_gguf::GgufWriter::new();
    w.set_arch("qwen4exp");
    for bank in BANKS {
        let (data, shape) = &t[&format!("{PREFIX}.{bank}.weight")];
        // Check the data is block-aligned for the type rather than assuming a
        // block size: the geometry comes from the one function that knows it.
        let blk = rlx_qwen4exp::ple::block_elements(dtype).expect("block geometry");
        assert_eq!(
            data.len() % blk,
            0,
            "{bank} is not {blk}-aligned for {dtype:?}"
        );
        let bytes = match dtype {
            GgmlType::Q8_0 => rlx_gguf::quantize::quantize_q8_0(data).expect("quantize"),
            other => panic!("fixture does not write {other:?}"),
        };
        // GGML order: the expert axis is outermost in torch terms, so it is the
        // LAST dimension here — which is what makes an expert contiguous.
        let ggml_shape: Vec<usize> = shape.iter().rev().copied().collect();
        w.add_tensor_bytes(format!("{PREFIX}.{bank}.weight"), ggml_shape, dtype, bytes)
            .expect("add bank");
    }
    w.write_to_path(&path).expect("write gguf");
    path
}

/// Paging a quantized checkpoint computes what the dequantized layer computes.
///
/// The published banks are IQ1_S / IQ2_XXS / IQ4_NL, never f32, so this is the
/// path that matters. Q8_0 stands in for them here because the fixture can be
/// written with the crate's own quantizer; the code under test reads the dtype
/// out of the header and is not told which it is.
#[test]
fn paging_quantized_banks_matches_the_dequantized_layer() {
    let d = dims(2);
    let t = tensors(d);
    let x = fill(2 * d.hidden, 55);
    let path = quantized_gguf("q8", &t, GgmlType::Q8_0);

    // The oracle is the layer over what the file actually holds after
    // quantization — otherwise this would compare two different models.
    let g = rlx_gguf::GgufFile::from_path(&path).expect("open gguf");
    let mut dequantized = t.clone();
    for bank in BANKS {
        let name = format!("{PREFIX}.{bank}.weight");
        let (data, _) = g.dequant_f32(&name).expect("dequant bank");
        let shape = t[&name].1.clone();
        dequantized.insert(name, (data, shape));
    }
    let want = resident(d, &dequantized, &x);

    let mut pager = ExpertPager::new(4 << 20);
    let n = register_gguf_expert_banks(
        &mut pager,
        &path,
        &[
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ],
    )
    .expect("register banks");
    assert_eq!(n, 3, "all three banks should register");

    let mut layer =
        PagedMoeLayer::from_gguf(PREFIX, 0, d, dev(), t.clone(), Arc::new(pager), &path)
            .expect("build quantized paged layer");
    let got = layer.forward(&x).expect("paged forward");

    let max_abs = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
    assert!(
        max_abs / scale < 1e-5,
        "paged-quantized and dequantized-resident differ by {max_abs} \
         ({:.3e} relative to a {scale} scale)",
        max_abs / scale
    );

    // The disk read is the quantized size, not the f32 size — the whole point.
    let per_f32 = (2 * d.expert_ffn * d.hidden + d.hidden * d.expert_ffn) * 4;
    assert!(
        layer.stats().bytes_read < (d.top_k * per_f32) as u64,
        "read {} bytes, which is not less than the f32 cost of {}",
        layer.stats().bytes_read,
        d.top_k * per_f32
    );
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// Quantized banks driven through the f32 constructor are refused.
///
/// The pager rejects a registration that runs past the file, so the way to
/// reach this guard is a registration that is perfectly consistent with the
/// *file* and inconsistent with the *graph*: real Q8_0 banks, registered
/// correctly, handed to `new` — which declares f32 slots.
#[test]
fn quantized_banks_through_the_f32_constructor_are_refused() {
    let d = dims(1);
    let t = tensors(d);
    let path = quantized_gguf("mismatch", &t, GgmlType::Q8_0);
    let mut pager = ExpertPager::new(1 << 20);
    register_gguf_expert_banks(
        &mut pager,
        &path,
        &[
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ],
    )
    .expect("register banks");

    let err = PagedMoeLayer::new(PREFIX, 0, d, dev(), t, Arc::new(pager))
        .err()
        .expect("Q8_0 experts cannot fill f32 slots");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("f32") && msg.contains("wrong shape"),
        "the refusal should name both sides: {msg}"
    );
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

/// The expert count comes from the bank's shape, and disagreement is an error.
///
/// This is the number a sliced checkpoint lies about in its metadata, so the
/// derivation has to be the tensors'. The GGML dim order is the trap: the
/// expert axis is **last**, and reading the first would give 2560 experts of a
/// 640-wide projection.
#[test]
fn the_expert_count_is_read_from_the_banks_not_the_metadata() {
    let d = dims(1);
    let t = tensors(d);
    let path = quantized_gguf("count", &t, GgmlType::Q8_0);
    assert_eq!(
        rlx_qwen4exp::moe::gguf_expert_count(&path).expect("read"),
        Some(d.num_experts),
        "the count should be the expert axis, not a projection dimension"
    );

    // A file with no routed banks has no count to report, rather than zero.
    let dir = path.parent().unwrap();
    let empty = dir.join("empty.gguf");
    let mut w = rlx_gguf::GgufWriter::new();
    w.set_arch("qwen4exp");
    w.add_tensor_bytes(
        "blk.0.ffn_gate_inp.weight",
        vec![4, 4],
        GgmlType::F32,
        vec![0u8; 64],
    )
    .expect("add");
    w.write_to_path(&empty).expect("write");
    assert_eq!(
        rlx_qwen4exp::moe::gguf_expert_count(&empty).expect("read"),
        None
    );

    // Banks that disagree are refused: "the expert count" would not be a number.
    let mixed = dir.join("mixed.gguf");
    let mut w = rlx_gguf::GgufWriter::new();
    w.set_arch("qwen4exp");
    for (blk, n) in [(0usize, 8usize), (1, 4)] {
        w.add_tensor_bytes(
            format!("blk.{blk}.ffn_gate_exps.weight"),
            vec![32, 1, n],
            GgmlType::F32,
            vec![0u8; 32 * n * 4],
        )
        .expect("add");
    }
    w.write_to_path(&mixed).expect("write");
    let err = rlx_qwen4exp::moe::gguf_expert_count(&mixed)
        .expect_err("disagreeing banks must be refused");
    assert!(format!("{err:#}").contains("experts but"), "{err:#}");
    let _ = std::fs::remove_dir_all(dir);
}
