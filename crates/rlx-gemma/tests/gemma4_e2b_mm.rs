// RLX — GPLv3. E2B multimodal glue: verify the embed-lazy `inputs_embeds`
//! path reproduces the `input_ids` path (text-only), and that media soft tokens
//! splice into the fused embeddings at placeholder positions.
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::gemma4_e2b_mm::{
    build_e2b_multimodal_prefill, build_e2b_multimodal_prefill_ext,
    e2b_fused_inputs_embeds_prescale, e2b_media_attn_bias,
};
use rlx_gemma::multimodal::GemmaMultimodalConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn dir() -> Option<PathBuf> {
    let h = std::env::var_os("HOME")?;
    let b = Path::new(&h).join(
        ".cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let s = std::fs::read_dir(&b).ok()?.flatten().next()?.path();
    s.join("config.json").is_file().then_some(s)
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    d / (na * nb + 1e-12)
}
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(
            (0, f32::MIN),
            |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
        )
        .0
}

/// The embed-lazy `inputs_embeds` path must reproduce the `input_ids` path
/// for a text-only prompt (same logits) — the foundation for multimodal splice.
#[test]
fn e2b_inputs_embeds_matches_input_ids() {
    let Some(d) = dir() else {
        eprintln!("[mm] no ckpt — skip");
        return;
    };
    let cfg = GemmaConfig::from_file(&d.join("config.json")).unwrap();
    let vocab = cfg.vocab_size;
    let prompt: Vec<u32> = vec![818, 5279, 529, 7001, 563];
    let seq = 16usize;
    let mut ids = vec![0u32; seq];
    ids[..prompt.len()].copy_from_slice(&prompt);

    let loader = GemmaQatLoader::open(&d).unwrap();
    let ple = loader.compute_per_layer_inputs(&cfg, &ids).unwrap();

    // Path A: input_ids.
    let mut bld_a = GemmaQatLoader::open(&d).unwrap();
    let mut packed = HashMap::new();
    let (ga, pa) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut bld_a,
        1,
        seq,
        true,
        false,
        false,
        &mut packed,
        None,
        None,
    )
    .unwrap();
    let mut ca = compile_graph_gemma_prefill_with_params(Device::Cpu, ga, pa).unwrap();
    let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let la = ca.run(&[
        ("input_ids", idf.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
    ])[0]
        .clone();

    // Path B: fused inputs_embeds (no media → just raw text rows, builder ×√h).
    let mm_cfg = GemmaMultimodalConfig::from_file(&d.join("config.json")).unwrap();
    let emb =
        e2b_fused_inputs_embeds_prescale(&loader, &cfg, &mm_cfg, &ids, &[], &[], &[]).unwrap();
    let mut bld_b = GemmaQatLoader::open(&d).unwrap();
    let (gb, pb) = build_e2b_multimodal_prefill(&cfg, &mut bld_b, 1, seq).unwrap();
    let mut cb = compile_graph_gemma_prefill_with_params(Device::Cpu, gb, pb).unwrap();
    let lb = cb.run(&[
        ("input_embeddings", emb.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
    ])[0]
        .clone();

    let last = prompt.len() - 1;
    let ra = &la[last * vocab..(last + 1) * vocab];
    let rb = &lb[last * vocab..(last + 1) * vocab];
    let cv = cos(ra, rb);
    eprintln!(
        "[mm] inputs_embeds vs input_ids: last-token cos = {cv:.6}, argmax A={} B={}",
        argmax(ra),
        argmax(rb)
    );
    assert_eq!(argmax(ra), argmax(rb), "argmax differs between paths");
    assert!(
        cv > 0.9999,
        "inputs_embeds path diverges from input_ids: cos {cv}"
    );
}

/// With no media tokens the attn bias is pure causal, so the `MaskKind::Bias`
/// path (with score_scale + softcap) must reproduce the `input_ids` path —
/// verifying the bidirectional-mask machinery is correct for E2B.
#[test]
fn e2b_media_bias_matches_causal() {
    let Some(d) = dir() else {
        eprintln!("[mm bias] no ckpt — skip");
        return;
    };
    let cfg = GemmaConfig::from_file(&d.join("config.json")).unwrap();
    let vocab = cfg.vocab_size;
    let prompt: Vec<u32> = vec![818, 5279, 529, 7001, 563];
    let seq = 16usize;
    let mut ids = vec![0u32; seq];
    ids[..prompt.len()].copy_from_slice(&prompt);
    let loader = GemmaQatLoader::open(&d).unwrap();
    let ple = loader.compute_per_layer_inputs(&cfg, &ids).unwrap();
    let mm_cfg = GemmaMultimodalConfig::from_file(&d.join("config.json")).unwrap();

    // input_ids reference.
    let mut bld_a = GemmaQatLoader::open(&d).unwrap();
    let mut packed = HashMap::new();
    let (ga, pa) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut bld_a,
        1,
        seq,
        true,
        false,
        false,
        &mut packed,
        None,
        None,
    )
    .unwrap();
    let mut ca = compile_graph_gemma_prefill_with_params(Device::Cpu, ga, pa).unwrap();
    let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let la = ca.run(&[
        ("input_ids", idf.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
    ])[0]
        .clone();

    // bias path (no media → pure causal bias).
    let emb =
        e2b_fused_inputs_embeds_prescale(&loader, &cfg, &mm_cfg, &ids, &[], &[], &[]).unwrap();
    let bias = e2b_media_attn_bias(&ids, &cfg, &mm_cfg, 1);
    let mut bld_b = GemmaQatLoader::open(&d).unwrap();
    let (gb, pb) = build_e2b_multimodal_prefill_ext(&cfg, &mut bld_b, 1, seq, true).unwrap();
    let mut cb = compile_graph_gemma_prefill_with_params(Device::Cpu, gb, pb).unwrap();
    let lb = cb.run(&[
        ("input_embeddings", emb.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
        ("attn_bias", bias.as_slice()),
    ])[0]
        .clone();

    let last = prompt.len() - 1;
    let ra = &la[last * vocab..(last + 1) * vocab];
    let rb = &lb[last * vocab..(last + 1) * vocab];
    let cv = cos(ra, rb);
    eprintln!(
        "[mm bias] causal-bias vs input_ids: last-token cos = {cv:.6}, argmax A={} B={}",
        argmax(ra),
        argmax(rb)
    );
    assert_eq!(argmax(ra), argmax(rb), "argmax differs (bias path)");
    assert!(
        cv > 0.9999,
        "MaskKind::Bias path diverges from causal: cos {cv}"
    );
}

/// Media soft tokens land at the placeholder positions in the fused embeds,
/// scaled by 1/√hidden (the builder restores them with ×√hidden).
#[test]
fn e2b_fusion_places_media_tokens() {
    let Some(d) = dir() else {
        eprintln!("[mm fuse] no ckpt — skip");
        return;
    };
    let cfg = GemmaConfig::from_file(&d.join("config.json")).unwrap();
    let h = cfg.hidden_size;
    let mm_cfg = GemmaMultimodalConfig::from_file(&d.join("config.json")).unwrap();
    let Some(img_tok) = mm_cfg.image_token_id else {
        eprintln!("[mm fuse] no image_token_id — skip");
        return;
    };
    // sequence: text, 2 image placeholders, text
    let ids = vec![818u32, img_tok, img_tok, 563];
    // distinctive image soft tokens
    let mut image = vec![0f32; 2 * h];
    for d_ in 0..h {
        image[d_] = 1.0;
        image[h + d_] = 2.0;
    }
    let loader = GemmaQatLoader::open(&d).unwrap();
    let emb =
        e2b_fused_inputs_embeds_prescale(&loader, &cfg, &mm_cfg, &ids, &image, &[], &[]).unwrap();
    let inv = 1.0f32 / (h as f32).sqrt();
    // positions 1,2 should hold image rows / √h
    let r1 = &emb[h..2 * h];
    let r2 = &emb[2 * h..3 * h];
    assert!(
        (r1[0] - inv).abs() < 1e-6 && (r1[h - 1] - inv).abs() < 1e-6,
        "img tok0 not placed"
    );
    assert!((r2[0] - 2.0 * inv).abs() < 1e-6, "img tok1 not placed");
    // text rows (0,3) must differ from the image fill
    assert!((emb[0] - inv).abs() > 1e-4, "text row 0 overwritten?");
    eprintln!("[mm fuse] media tokens placed at positions 1,2 (×1/√h); text rows intact");
}
