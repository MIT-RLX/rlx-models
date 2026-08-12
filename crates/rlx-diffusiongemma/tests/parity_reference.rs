// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Numerical parity against the PyTorch reference forward.
//!
//! Generate the fixture, then point the test at it:
//!
//! ```sh
//! python3 scripts/diffusiongemma_reference.py .fixtures/diffusiongemma-parity
//! RLX_DG_PARITY_DIR=.fixtures/diffusiongemma-parity \
//!     cargo test -p rlx-diffusiongemma --test parity_reference
//! ```
//!
//! Without `RLX_DG_PARITY_DIR` the test skips — the fixture needs torch and is
//! not committed.
//!
//! This is what actually pins the arithmetic. The smoke test only proves the
//! graphs are finite; this proves they compute DiffusionGemma: the per-layer
//! head geometry, V aliased to the pre-`k_norm` K on full-attention layers, the
//! `scaling = 1.0` convention, proportional RoPE with its NoPE tail, the
//! two-branch FFN with an unnormalized router input, top-k expert dispatch, the
//! windowed encoder mask, the bidirectional denoiser over the tapped cache, the
//! self-conditioning fold-in, and the soft-cap-then-temperature logit path.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_diffusiongemma::flow::{
    CANVAS_INPUT, EncoderCacheLens, SC_SIGNAL_INPUT, TEMPERATURE_INPUT, build_decoder_flow,
    build_encoder_flow, enc_k_name, enc_v_name,
};
use rlx_diffusiongemma::preprocess::resize_bicubic_u8;
use rlx_diffusiongemma::prompt::{ChatMessage, ChatOptions, format_chat};
use rlx_diffusiongemma::vision::{
    ENCODER_TAP, PATCH_EMBED_TAP, PIXELS_INPUT, POOL_INPUT, POOLED_TAP, POS_X_INPUT, POS_Y_INPUT,
    ROPE_COS_INPUT, ROPE_SIN_INPUT, SOFT_TOKENS_OUTPUT, VALID_INPUT,
};
use rlx_diffusiongemma::{
    DiffusionGemmaConfig, build_vision_flow, grid_positions, merge_multimodal_embeds,
    prepare_checkpoint, vision_pool_matrix, vision_rope_tables,
};
use rlx_runtime::Device;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fixture_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("RLX_DG_PARITY_DIR").ok()?);
    assert!(
        dir.join("model.safetensors").is_file(),
        "RLX_DG_PARITY_DIR={dir:?} has no model.safetensors — \
         run scripts/diffusiongemma_reference.py first"
    );
    Some(dir)
}

fn read_bin(dir: &Path, name: &str) -> Vec<f32> {
    let raw = std::fs::read(dir.join(format!("{name}.bin")))
        .unwrap_or_else(|e| panic!("reading {name}.bin: {e}"));
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Compare against the reference with both a shape and a direction check.
fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let cos = cosine(got, want);
    let mad = max_abs_diff(got, want);
    assert!(
        cos > 0.99999,
        "{label}: cosine {cos:.8} (max |Δ| {mad:.3e}) — the graph disagrees with the reference"
    );
    assert!(
        mad <= tol,
        "{label}: max |Δ| {mad:.3e} exceeds {tol:.3e} (cosine {cos:.8})"
    );
}

struct Fixture {
    dir: PathBuf,
    cfg: DiffusionGemmaConfig,
    wm: WeightMap,
    prompt_ids: Vec<f32>,
    canvas_ids: Vec<f32>,
    prompt_len: usize,
    canvas: usize,
    temperature: f32,
}

fn load() -> Option<Fixture> {
    let dir = fixture_dir()?;
    let cfg = DiffusionGemmaConfig::from_file(dir.join("config.json")).expect("config");
    let mut wm = WeightMap::from_safetensors_dir(&dir).expect("weights");
    prepare_checkpoint(&cfg, &mut wm).expect("prepare");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<f32> {
        meta[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    Some(Fixture {
        prompt_ids: ids("prompt_ids"),
        canvas_ids: ids("canvas_ids"),
        prompt_len: meta["prompt_len"].as_u64().unwrap() as usize,
        canvas: meta["canvas"].as_u64().unwrap() as usize,
        temperature: meta["temperature"].as_f64().unwrap() as f32,
        dir,
        cfg,
        wm,
    })
}

/// Run the encoder, returning `(hidden, [(k, v)])`.
fn run_encoder(f: &Fixture) -> (Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>) {
    let t = &f.cfg.text_config;
    let built = build_encoder_flow(&f.cfg, &f.wm, f.prompt_len).expect("build encoder");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile encoder");

    let (cos_s, sin_s) = t.rope_tables(0, 0, f.prompt_len);
    let (cos_f, sin_f) = t.rope_tables(3, 0, f.prompt_len);
    let outs = compiled.run(&[
        ("input_ids", f.prompt_ids.as_slice()),
        ("rope_cos_sliding", cos_s.as_slice()),
        ("rope_sin_sliding", sin_s.as_slice()),
        ("rope_cos_full", cos_f.as_slice()),
        ("rope_sin_full", sin_f.as_slice()),
    ]);
    let by: HashMap<&str, &Vec<f32>> = names.iter().map(|s| s.as_str()).zip(outs.iter()).collect();
    let kv = (0..t.num_hidden_layers)
        .map(|l| {
            (
                by[enc_k_name(l).as_str()].clone(),
                by[enc_v_name(l).as_str()].clone(),
            )
        })
        .collect();
    (by["hidden"].clone(), kv)
}

/// Run one denoiser step against the encoder taps.
fn run_decoder(
    f: &Fixture,
    kv: &[(Vec<f32>, Vec<f32>)],
    sc_signal: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let t = &f.cfg.text_config;
    let cache = EncoderCacheLens::for_prompt(t, f.prompt_len);
    let built = build_decoder_flow(&f.cfg, &f.wm, f.canvas, cache).expect("build decoder");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile decoder");

    let (cos_s, sin_s) = t.rope_tables(0, f.prompt_len, f.canvas);
    let (cos_f, sin_f) = t.rope_tables(3, f.prompt_len, f.canvas);
    let temp = vec![f.temperature];

    let sliced: Vec<(Vec<f32>, Vec<f32>)> = (0..t.num_hidden_layers)
        .map(|l| {
            let kv_dim = t.layer_kv_heads(l) * t.layer_head_dim(l);
            let start = (f.prompt_len - cache.for_layer(t, l)) * kv_dim;
            (kv[l].0[start..].to_vec(), kv[l].1[start..].to_vec())
        })
        .collect();
    let kn: Vec<String> = (0..t.num_hidden_layers).map(enc_k_name).collect();
    let vn: Vec<String> = (0..t.num_hidden_layers).map(enc_v_name).collect();

    let mut inputs: Vec<(&str, &[f32])> = vec![
        (CANVAS_INPUT, f.canvas_ids.as_slice()),
        (SC_SIGNAL_INPUT, sc_signal),
        (TEMPERATURE_INPUT, temp.as_slice()),
        ("rope_cos_sliding", cos_s.as_slice()),
        ("rope_sin_sliding", sin_s.as_slice()),
        ("rope_cos_full", cos_f.as_slice()),
        ("rope_sin_full", sin_f.as_slice()),
    ];
    for l in 0..t.num_hidden_layers {
        inputs.push((kn[l].as_str(), sliced[l].0.as_slice()));
        inputs.push((vn[l].as_str(), sliced[l].1.as_slice()));
    }
    let outs = compiled.run(&inputs);
    let by: HashMap<&str, &Vec<f32>> = names.iter().map(|s| s.as_str()).zip(outs.iter()).collect();
    (by["logits"].clone(), by["soft_embeds"].clone())
}

/// The vision tower + projector: patch embed with looked-up 2-D position
/// embeddings, 2-D RoPE split across each head, bidirectional layers, k² average
/// pooling, `sqrt(hidden)` scaling, standardization, and the scale-free-RMS
/// projector into LM width.
#[test]
fn vision_tower_matches_pytorch_reference() {
    let Some(f) = load() else {
        eprintln!("skipping: set RLX_DG_PARITY_DIR (see the module docs)");
        return;
    };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(f.dir.join("meta.json")).unwrap()).unwrap();
    let v = &meta["vision"];
    let patches = v["patches"].as_u64().unwrap() as usize;
    let soft_len = v["soft_tokens"].as_u64().unwrap() as usize;
    let grid = v["grid"].as_u64().unwrap() as usize;
    let vcfg = f
        .cfg
        .vision_config
        .as_ref()
        .expect("fixture has a vision config");

    let positions = grid_positions(grid, grid);
    assert_eq!(positions.len(), patches);
    // Cross-check the host helpers against the positions the reference used.
    let xs: Vec<u32> = v["positions_x"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as u32)
        .collect();
    let ys: Vec<u32> = v["positions_y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(
        positions,
        xs.iter()
            .copied()
            .zip(ys.iter().copied())
            .collect::<Vec<_>>()
    );

    let pool = vision_pool_matrix(&positions, vcfg.pooling_kernel_size, soft_len);
    assert_close("vision_pool", &pool, &read_bin(&f.dir, "vision_pool"), 1e-6);

    let (cos, sin) = vision_rope_tables(vcfg, &positions);
    let pixels = read_bin(&f.dir, "vision_pixels");
    let pos_x: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
    let pos_y: Vec<f32> = ys.iter().map(|&y| y as f32).collect();
    let valid = vec![1f32; patches];

    let built = build_vision_flow(&f.cfg, &f.wm, patches, soft_len).expect("build vision");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile vision");
    let outs = compiled.run(&[
        (PIXELS_INPUT, pixels.as_slice()),
        (POS_X_INPUT, pos_x.as_slice()),
        (POS_Y_INPUT, pos_y.as_slice()),
        (ROPE_COS_INPUT, cos.as_slice()),
        (ROPE_SIN_INPUT, sin.as_slice()),
        (VALID_INPUT, valid.as_slice()),
        (POOL_INPUT, pool.as_slice()),
    ]);
    let by: HashMap<&str, &Vec<f32>> = names.iter().map(|s| s.as_str()).zip(outs.iter()).collect();

    // Compare the RoPE tables and each stage in order, so a mismatch names the
    // stage that introduced it rather than only the final output.
    assert_close(
        "vision_rope_cos",
        &cos,
        &read_bin(&f.dir, "vision_rope_cos"),
        1e-6,
    );
    assert_close(
        "vision_rope_sin",
        &sin,
        &read_bin(&f.dir, "vision_rope_sin"),
        1e-6,
    );
    assert_close(
        "vision_patch_embed",
        by[PATCH_EMBED_TAP],
        &read_bin(&f.dir, "vision_patch_embed"),
        2e-5,
    );
    assert_close(
        "vision_encoder_out",
        by[ENCODER_TAP],
        &read_bin(&f.dir, "vision_encoder_out"),
        2e-4,
    );
    assert_close(
        "vision_pooled",
        by[POOLED_TAP],
        &read_bin(&f.dir, "vision_pooled"),
        2e-4,
    );
    assert_close(
        "soft_tokens",
        by[SOFT_TOKENS_OUTPUT],
        &read_bin(&f.dir, "soft_tokens"),
        2e-4,
    );
}

/// Soft tokens replace token embeddings at `image_token_id` positions, and must
/// *not* pick up the `sqrt(hidden)` embedding scale that text rows get.
#[test]
fn multimodal_embed_merge_places_soft_tokens_unscaled() {
    let Some(f) = load() else {
        eprintln!("skipping: set RLX_DG_PARITY_DIR (see the module docs)");
        return;
    };
    let t = &f.cfg.text_config;
    let hidden = t.hidden_size;
    let img = f.cfg.image_token_id;
    // Two image slots surrounded by text.
    let ids = vec![3u32, img, img, 5];
    let soft: Vec<f32> = (0..2 * hidden).map(|i| i as f32 * 0.01).collect();
    let merged = merge_multimodal_embeds(&f.cfg, &f.wm, &ids, &soft).expect("merge");
    assert_eq!(merged.len(), ids.len() * hidden);

    // Image rows are the soft tokens verbatim.
    assert_eq!(&merged[hidden..3 * hidden], &soft[..]);
    // Text rows are the embedding table scaled by sqrt(hidden).
    let (table, _) = f.wm.get(rlx_diffusiongemma::EMBED_KEY).unwrap();
    let scale = t.embed_scale();
    for &(i, id) in [(0usize, 3u32), (3, 5)].iter() {
        for c in 0..hidden {
            let want = table[id as usize * hidden + c] * scale;
            assert!((merged[i * hidden + c] - want).abs() < 1e-5);
        }
    }

    // A slot-count mismatch is an error, not a silent truncation.
    assert!(merge_multimodal_embeds(&f.cfg, &f.wm, &ids, &soft[..hidden]).is_err());
}

#[test]
fn encoder_matches_pytorch_reference() {
    let Some(f) = load() else {
        eprintln!("skipping: set RLX_DG_PARITY_DIR (see the module docs)");
        return;
    };
    let t = &f.cfg.text_config;
    let (hidden, kv) = run_encoder(&f);

    // Per-layer K/V taps first: they localize a mismatch to the layer that
    // introduced it instead of only reporting the final hidden state.
    for l in 0..t.num_hidden_layers {
        assert_close(
            &format!("enc_k[{l}]"),
            &kv[l].0,
            &read_bin(&f.dir, &format!("enc_k_{l}")),
            2e-4,
        );
        assert_close(
            &format!("enc_v[{l}]"),
            &kv[l].1,
            &read_bin(&f.dir, &format!("enc_v_{l}")),
            2e-4,
        );
    }
    assert_close(
        "encoder_hidden",
        &hidden,
        &read_bin(&f.dir, "encoder_hidden"),
        2e-4,
    );
}

#[test]
fn decoder_matches_pytorch_reference() {
    let Some(f) = load() else {
        eprintln!("skipping: set RLX_DG_PARITY_DIR (see the module docs)");
        return;
    };
    let (_, kv) = run_encoder(&f);
    let hidden = f.cfg.text_config.hidden_size;

    // First denoising step: zero self-conditioning signal.
    let zeros = vec![0f32; f.canvas * hidden];
    let (logits, soft) = run_decoder(&f, &kv, &zeros);
    assert_close("logits", &logits, &read_bin(&f.dir, "logits"), 2e-3);
    assert_close("soft_embeds", &soft, &read_bin(&f.dir, "soft_embeds"), 2e-4);

    // Second step: feed the reference's soft embeddings back in, which is what
    // every step after the first sees. Uses the *reference* signal rather than
    // ours so a drift here is attributable to this step alone.
    let sc = read_bin(&f.dir, "soft_embeds");
    let (logits2, soft2) = run_decoder(&f, &kv, &sc);
    assert_close("logits_sc", &logits2, &read_bin(&f.dir, "logits_sc"), 2e-3);
    assert_close(
        "soft_embeds_sc",
        &soft2,
        &read_bin(&f.dir, "soft_embeds_sc"),
        2e-4,
    );
}

/// `resize_bicubic_u8` against Pillow itself — the resampler HF's image
/// processor calls under the hood. Covers downscale (where the antialiasing
/// support widening matters), upscale, and a non-square change.
#[test]
fn resize_matches_pillow_bicubic() {
    let Some(f) = load() else {
        eprintln!("skipping: set RLX_DG_PARITY_DIR (see the module docs)");
        return;
    };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(f.dir.join("meta.json")).unwrap()).unwrap();
    let cases = meta["resize_cases"].as_array().expect("resize_cases");

    for (i, c) in cases.iter().enumerate() {
        let g = |k: &str| c[k].as_u64().unwrap() as usize;
        let (sh, sw, dh, dw) = (g("src_h"), g("src_w"), g("dst_h"), g("dst_w"));
        let src = std::fs::read(f.dir.join(format!("resize_src_{i}.bin"))).unwrap();
        let want = std::fs::read(f.dir.join(format!("resize_dst_{i}.bin"))).unwrap();
        assert_eq!(src.len(), sh * sw * 3);
        assert_eq!(want.len(), dh * dw * 3);

        let got = resize_bicubic_u8(&src, sh, sw, 3, dh, dw).expect("resize");
        assert_eq!(got.len(), want.len(), "case {i}: length");

        // Pillow accumulates in fixed point; this reproduces the same filter in
        // f64. Allow a 1-LSB rounding difference, but require it to be rare.
        let mut off_by_one = 0usize;
        for (j, (a, b)) in got.iter().zip(&want).enumerate() {
            let d = (*a as i32 - *b as i32).abs();
            assert!(d <= 1, "case {i} byte {j}: {a} vs {b} (differs by {d})");
            if d == 1 {
                off_by_one += 1;
            }
        }
        // Measured: worst case is 10/13824 bytes (0.07%).
        let frac = off_by_one as f64 / want.len() as f64;
        assert!(
            frac < 0.005,
            "case {i}: {off_by_one}/{} bytes differ by 1 ({:.2}%) — \
             more rounding drift than expected",
            want.len(),
            frac * 100.0
        );
    }
}

/// `format_chat` against the model's own `chat_template.jinja`, rendered by
/// Jinja and then expanded by the processor's image rule. This is what pins the
/// turn markers, the system/thinking block placement, and the per-image
/// `boi + slots + eoi` expansion to the real template rather than to my reading
/// of it.
#[test]
fn chat_prompts_match_the_shipped_template() {
    let Some(f) = load() else {
        eprintln!("skipping: set RLX_DG_PARITY_DIR (see the module docs)");
        return;
    };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(f.dir.join("meta.json")).unwrap()).unwrap();
    let cases = meta["chat_cases"].as_array().expect("chat_cases");
    assert!(
        !cases.is_empty(),
        "fixture has no chat cases — is jinja2 installed?"
    );

    for c in cases {
        let name = c["name"].as_str().unwrap();
        let want = c["expected"].as_str().unwrap();
        let soft: Vec<usize> = c["soft_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let (msgs, thinking) = match name {
            "plain" => (vec![ChatMessage::user("  Why is the sky blue?  ")], false),
            "system" => (
                vec![
                    ChatMessage::system("You are terse."),
                    ChatMessage::user("Hi"),
                ],
                false,
            ),
            "thinking" => (vec![ChatMessage::user("Hi")], true),
            "multi" => (
                vec![
                    ChatMessage::user("2+2?"),
                    ChatMessage::model("4"),
                    ChatMessage::user("and 3+3?"),
                ],
                false,
            ),
            "image" => (
                vec![ChatMessage::user_with_images(1, "What is this?")],
                false,
            ),
            "two_images" => (
                vec![ChatMessage::user_with_images(2, "Compare these.")],
                false,
            ),
            other => panic!("fixture has an unhandled chat case: {other}"),
        };
        let opts = ChatOptions {
            add_generation_prompt: true,
            enable_thinking: thinking,
        };
        let got = format_chat(&msgs, opts, &soft).expect("format_chat");
        assert_eq!(got, want, "chat case `{name}` diverges from the template");
    }
}
