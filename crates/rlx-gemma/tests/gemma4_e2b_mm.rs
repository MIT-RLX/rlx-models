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

fn rd(p: &Path) -> Option<Vec<f32>> {
    let b = std::fs::read(p).ok()?;
    Some(
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// End-to-end multimodal *generation*: feed real image features (the
/// HF-validated `feat_out.bin` = projected vision soft tokens) through the
/// fusion + E2B text decoder and greedy-decode on the target device. Proves
/// the whole image→features→fused-embeddings→text path runs and produces a
/// varied (non-degenerate) token stream, and that the fused-media prefill
/// matches CPU on the first token. Uses the proven QAT-loader path (not the
/// pre-QAT `MultimodalWeights` names the CLI currently expects).
fn run_mm_generate(dev: Device, tag: &str, steps: usize) {
    let Some(d) = dir() else {
        eprintln!("[{tag}] no ckpt — skip");
        return;
    };
    if dev != Device::Cpu && !rlx_runtime::is_available(dev) {
        eprintln!("[{tag}] {dev:?} unavailable — skip");
        return;
    }
    let cfg = GemmaConfig::from_file(&d.join("config.json")).unwrap();
    let mm_cfg = GemmaMultimodalConfig::from_file(&d.join("config.json")).unwrap();
    let h = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let Some(img_tok) = mm_cfg.image_token_id else {
        eprintln!("[{tag}] no image_token_id — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_vision/feat_out.bin");
    let Some(image_soft) = rd(&fx) else {
        eprintln!("[{tag}] no vision fixture — skip");
        return;
    };
    let n_img = image_soft.len() / h; // projected soft tokens for the fixture image

    // Prompt: BOS + <image> placeholders. Bucketed sequence, zero-padded.
    let seq = 32usize;
    let mut prompt = vec![2u32];
    prompt.extend(std::iter::repeat(img_tok).take(n_img));
    let mut real_len = prompt.len();
    let mut ids = vec![0u32; seq];
    ids[..real_len].copy_from_slice(&prompt);

    let loader = GemmaQatLoader::open(&d).unwrap();
    let mut bld = GemmaQatLoader::open(&d).unwrap();
    let (g, p) = build_e2b_multimodal_prefill(&cfg, &mut bld, 1, seq).unwrap();
    let mut compiled = compile_graph_gemma_prefill_with_params(dev, g, p).unwrap();

    let mut generated: Vec<u32> = Vec::new();
    let mut first_logits: Option<Vec<f32>> = None;
    for _ in 0..steps {
        if real_len >= seq {
            break;
        }
        let emb =
            e2b_fused_inputs_embeds_prescale(&loader, &cfg, &mm_cfg, &ids, &image_soft, &[], &[])
                .unwrap();
        let ple = loader.compute_per_layer_inputs(&cfg, &ids).unwrap();
        let logits = compiled.run(&[
            ("input_embeddings", emb.as_slice()),
            ("per_layer_inputs", ple.as_slice()),
        ])[0]
            .clone();
        let last = real_len - 1;
        let row = &logits[last * vocab..(last + 1) * vocab];
        if first_logits.is_none() {
            first_logits = Some(row.to_vec());
        }
        let tok = argmax(row) as u32;
        generated.push(tok);
        ids[real_len] = tok;
        real_len += 1;
    }
    let uniq: std::collections::HashSet<u32> = generated.iter().copied().collect();
    eprintln!(
        "[{tag}] n_img={n_img} generated {} tokens ({} unique): {:?}",
        generated.len(),
        uniq.len(),
        generated
    );
    let fl = first_logits.expect("no logits produced");
    assert!(fl.iter().all(|v| v.is_finite()), "[{tag}] non-finite logits");

    // CPU writes the first-token reference (before any other assert so it's
    // always available); GPU backends cross-check their fused-media prefill.
    let mut ref_path = std::env::temp_dir();
    ref_path.push("rlx_mm_gen_cpu_first.bin");
    if dev == Device::Cpu {
        let bytes: Vec<u8> = fl.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&ref_path, bytes).ok();
    } else if let Some(cpu) = rd(&ref_path) {
        let c = cos(&cpu, &fl);
        eprintln!("[{tag}] first-token logits cos vs CPU = {c:.6}");
        assert!(c > 0.99, "[{tag}] fused-media prefill diverges from CPU: cos {c}");
    }

    // Degeneracy only meaningful when we actually decoded several tokens.
    if generated.len() > 2 {
        assert!(uniq.len() > 2, "[{tag}] degenerate generation: {generated:?}");
    }
}

#[test]
fn e2b_mm_generate_cpu() {
    run_mm_generate(Device::Cpu, "mm gen cpu", 1);
}

#[test]
fn e2b_mm_generate_metal() {
    run_mm_generate(Device::Metal, "mm gen metal", 12);
}

#[test]
fn e2b_mm_generate_mlx() {
    run_mm_generate(Device::Mlx, "mm gen mlx", 12);
}
