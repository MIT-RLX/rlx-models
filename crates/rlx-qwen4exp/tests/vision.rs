// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
#![cfg(feature = "vision")]
//
//! The multimodal seam.
//!
//! The ViT itself is `rlx-qwen35`'s and is tested there; what is new here is the
//! *seam*, and all three of its parts fail silently:
//!
//! * the mmproj contract — a projector or a projection width this model does
//!   not have still splices cleanly into a `[seq, hidden]` buffer;
//! * the PLE substitution — hashing the raw `<|image_pad|>` id instead of
//!   `image_token_id` shifts every n-gram across an image;
//! * the M-RoPE advance — an image consumes `max(nx, ny)` positions, not its
//!   token count, and getting it wrong de-synchronizes everything after it.
//!
//! The mmproj metadata here is the real `unsloth/Qwen3.8-Flash-Next-GGUF/
//! mmproj-F16.gguf` header, read out of the published file.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_gguf::{GgufFile, MetaValue};
use rlx_qwen4exp::vision::{
    ImageSpan, PROJECTOR_TYPE, assemble_multimodal, check_mmproj, ple_vision_token,
};
use rlx_qwen4exp::vit::{VisionConfig, vision_tensor_names, vision_tensor_shapes};
use rlx_qwen4exp::{
    FlowOptions, OutputKind, PleHash, Qwen4ExpConfig, ROPE_COS_INPUT, ROPE_SIN_INPUT,
    build_qwen4exp_text_flow_with_options, ngram_embeddings, ngram_rows, ple_embed_input, synth,
};
use rlx_runtime::Device;

/// The published `mmproj-F16.gguf` metadata, verbatim.
fn published_mmproj() -> GgufFile {
    let mut g = GgufFile::empty();
    g.metadata.insert(
        "general.architecture".into(),
        MetaValue::String("clip".into()),
    );
    g.metadata
        .insert("general.type".into(), MetaValue::String("mmproj".into()));
    g.metadata.insert(
        "clip.projector_type".into(),
        MetaValue::String(PROJECTOR_TYPE.into()),
    );
    g.metadata
        .insert("clip.has_vision_encoder".into(), MetaValue::Bool(true));
    g.metadata
        .insert("clip.use_gelu".into(), MetaValue::Bool(true));
    for (k, v) in [
        ("clip.vision.projection_dim", 2560u32),
        ("clip.vision.image_size", 768),
        ("clip.vision.patch_size", 16),
        ("clip.vision.embedding_length", 1152),
        ("clip.vision.feed_forward_length", 4304),
        ("clip.vision.block_count", 27),
        ("clip.vision.attention.head_count", 16),
        ("clip.vision.spatial_merge_size", 2),
    ] {
        g.metadata.insert(k.into(), MetaValue::U32(v));
    }
    g.metadata.insert(
        "clip.vision.attention.layer_norm_epsilon".into(),
        MetaValue::F32(1e-6),
    );
    let half = || MetaValue::Array(vec![MetaValue::F32(0.5); 3]);
    g.metadata.insert("clip.vision.image_mean".into(), half());
    g.metadata.insert("clip.vision.image_std".into(), half());
    // All 27 entries are `false` in the published file — no deepstack layers.
    g.metadata.insert(
        "clip.vision.is_deepstack_layers".into(),
        MetaValue::Array(vec![MetaValue::Bool(false); 27]),
    );
    g
}

#[test]
fn the_published_mmproj_is_this_models_vision_tower() {
    let cfg = Qwen4ExpConfig::qwen38_flash_next();
    let mm = VisionConfig::from_gguf(&published_mmproj()).expect("parse mmproj metadata");
    check_mmproj(&cfg, &mm).expect("the published mmproj must be accepted");

    // The seam's load-bearing agreement: the projector maps to the text width.
    assert_eq!(mm.llm_hidden_size, cfg.hidden_size);
    assert_eq!(mm.projector_type, PROJECTOR_TYPE);
    assert_eq!(mm.n_layer, 27);
    assert_eq!(mm.n_embd, 1152);
    assert_eq!(mm.n_merge, 2);
    assert!(mm.deepstack_layers.is_empty());
    // Derived, not read: the merger's input width and the RoPE head.
    assert_eq!(mm.merger_in(), 4608, "n_embd x merge^2");
    assert_eq!(mm.head_dim(), 72);
    assert_eq!(mm.align_size(), 32, "patch 16 x merge 2");

    // The tower's tensor contract against the published file's own list.
    let names = vision_tensor_names(&mm);
    assert_eq!(names.len(), 10 + 27 * 12, "10 globals + 12 per block");
    let mut from_shapes: Vec<String> = vision_tensor_shapes(&mm, 2304)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    from_shapes.sort();
    assert_eq!(from_shapes, names);
}

/// `image_token_id` is what the PLE hash sees at a vision position — and it is
/// a *different* array from the one the embeddings came from.
#[test]
fn the_ple_hash_sees_the_image_token_not_the_placeholder() {
    let cfg = Qwen4ExpConfig::qwen38_flash_next();
    assert_eq!(ple_vision_token(&cfg), 248_056);
    assert_ne!(ple_vision_token(&cfg), cfg.ple_eos_token_id);

    // With the key absent, llama.cpp falls back to EOS; so do we.
    let mut no_key = cfg.clone();
    no_key.ple_image_token_id = 0;
    assert_eq!(ple_vision_token(&no_key), no_key.ple_eos_token_id);
}

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

/// A multimodal prefill runs end to end: host-assembled embeddings in, logits
/// out, with the PLE rows hashed from the substituted ids and the positions
/// advanced by the grid side.
#[test]
fn a_multimodal_prefill_runs() {
    let cfg = synth::tiny_config(2048);
    let hidden = cfg.hidden_size;
    // `a <pad × 4> b b` — a 2×2 image between text.
    let ids: Vec<u32> = vec![5, 0, 0, 0, 0, 7, 9];
    let seq = ids.len();

    let mut tensors = synth::synth_tensors(&cfg, 23);
    tensors.remove("per_layer_token_embd.weight");
    // The embedding table is read on the host under `host_embed`, so take it
    // out of the graph's weights and keep a copy here.
    let (embed_table, _) = tensors
        .remove("token_embd.weight")
        .expect("token_embd in the fixture");
    let mut weights = WeightMap::from_tensors(tensors);

    let embeds: Vec<f32> = (0..4 * hidden)
        .map(|i| (i as f32 * 0.01).cos() * 0.2)
        .collect();
    let batch = assemble_multimodal(
        &cfg,
        &ids,
        &embed_table,
        &[ImageSpan {
            start: 1,
            nx: 2,
            ny: 2,
            embeds: &embeds,
        }],
    )
    .expect("assemble");

    assert_eq!(batch.seq, seq);
    assert_eq!(batch.inputs_embeds.len(), seq * hidden);
    // The vision rows really are the encoder's, not the placeholder's.
    assert_eq!(&batch.inputs_embeds[hidden..5 * hidden], embeds.as_slice());
    // 4 tokens, max(2, 2) = 2 positions: the text after resumes at 3.
    assert_eq!(batch.section_positions[5], [3, 3, 3, 0]);
    assert_eq!(batch.section_positions[6], [4, 4, 4, 0]);

    // PLE rows come from the SUBSTITUTED ids.
    let hash = PleHash::derive(&cfg, 0);
    assert_eq!(
        ngram_rows(&hash, &batch.ple_token_ids),
        ngram_rows(&hash, &batch.ple_token_ids),
    );
    assert_ne!(
        ngram_rows(&hash, &batch.ple_token_ids),
        ngram_rows(&hash, &ids),
        "substituting image_token_id must change the hashed rows"
    );
    let ple = ngram_embeddings(&hash, &batch.ple_token_ids, &[], cfg.ple_head_dim, |r| {
        Ok((0..cfg.ple_head_dim)
            .map(|j| (((r as usize + j) % 71) as f32 - 35.0) * 0.004)
            .collect())
    })
    .expect("ple rows");

    let mut opts = FlowOptions::prefill(seq);
    opts.output = OutputKind::Logits;
    opts.host_embed = true;
    opts.runtime_mrope = true;
    let built =
        build_qwen4exp_text_flow_with_options(&cfg, &mut WeightMapSource(&mut weights), opts)
            .expect("build multimodal flow");

    let (cos, sin) = rlx_qwen4exp::rope::mrope_tables(&cfg, seq, Some(&batch.section_positions));
    let id_f: Vec<f32> = ids.iter().map(|t| *t as f32).collect();
    let logits = compile_built(built, dev())
        .expect("compile multimodal flow")
        .run(&[
            ("input_ids", id_f.as_slice()),
            ("inputs_embeds", batch.inputs_embeds.as_slice()),
            (ROPE_COS_INPUT, cos.as_slice()),
            (ROPE_SIN_INPUT, sin.as_slice()),
            (&ple_embed_input(0), ple.as_slice()),
        ])
        .pop()
        .expect("logits");

    assert_eq!(logits.len(), seq * cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// `host_embed` must actually remove the embedding table from the graph —
/// otherwise a 1.2 GB param is still resident and the splice is ignored.
#[test]
fn host_embed_takes_the_table_off_the_graph() {
    let cfg = synth::tiny_config(2048);
    let seq = 3;
    let mut tensors = synth::synth_tensors(&cfg, 3);
    tensors.remove("per_layer_token_embd.weight");
    tensors.remove("token_embd.weight");
    let mut weights = WeightMap::from_tensors(tensors);

    let mut opts = FlowOptions::prefill(seq);
    opts.output = OutputKind::Hidden;
    opts.host_embed = true;
    // Builds despite `token_embd` being absent: the graph never asks for it.
    build_qwen4exp_text_flow_with_options(&cfg, &mut WeightMapSource(&mut weights), opts)
        .expect("host_embed must not need token_embd");
}

/// The whole multimodal path: an image through the **native tower**, its rows
/// spliced into the embedding stream, and the LM run over the result.
///
/// `a_multimodal_prefill_runs` above feeds synthetic vision rows, so it checks
/// the seam in isolation. This one checks that the tower's *actual* output —
/// its token count, its grid, and its width — is what the seam and the LM
/// expect. A post-merge grid reported pre-merge, or a projector width off by
/// anything, breaks here and nowhere else.
#[test]
fn an_image_goes_through_the_native_tower_into_the_lm() {
    use rlx_qwen4exp::vision::{VisionEncoder, assemble_multimodal};
    use rlx_qwen4exp::vit::{VisionConfig, vision_tensor_shapes};

    let cfg = synth::tiny_config(2048);
    let hidden = cfg.hidden_size;

    // A tower that projects to the text model's width — `check_mmproj`'s whole
    // job is to refuse one that does not.
    let vcfg = VisionConfig {
        patch_size: 4,
        n_embd: 16,
        n_head: 2,
        n_layer: 2,
        image_size: 32,
        n_ff: 24,
        n_merge: 2,
        eps: 1e-6,
        projector_type: rlx_qwen4exp::vision::PROJECTOR_TYPE.into(),
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        llm_hidden_size: hidden,
        // Bounds that make `smart_resize` land on a small, fixed grid.
        image_min_pixels: 256,
        image_max_pixels: 1024,
        rope_theta: 10_000.0,
        deepstack_layers: vec![],
    };

    let mut vt: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)> =
        std::collections::HashMap::new();
    let mut seed = 5u64;
    for (name, shape) in vision_tensor_shapes(&vcfg, 36) {
        let n: usize = shape.iter().product();
        seed = seed.wrapping_add(13);
        let data: Vec<f32> = (0..n)
            .map(|i| (((i as u64 + seed) % 61) as f32 - 30.0) * 0.01)
            .collect();
        let data = if name.ends_with("norm.weight")
            || name.ends_with("ln1.weight")
            || name.ends_with("ln2.weight")
        {
            data.iter().map(|x| 1.0 + x).collect()
        } else {
            data
        };
        vt.insert(name, (data, shape));
    }

    let mut encoder = VisionEncoder::from_tensors(vcfg.clone(), vt, dev());
    let (iw, ih) = (24usize, 24usize);
    let rgb: Vec<u8> = (0..iw * ih * 3).map(|i| (i % 251) as u8).collect();
    let out = encoder.encode_rgb(&rgb, iw, ih).expect("encode");

    assert_eq!(out.n_tokens, out.grid_x * out.grid_y);
    assert_eq!(out.embeddings.len(), out.n_tokens * hidden);
    assert!(out.embeddings.iter().all(|v| v.is_finite()));
    assert_eq!(encoder.compiled_sizes(), 1);
    // A second image of the same size must reuse the compiled graph.
    let again = encoder.encode_rgb(&rgb, iw, ih).expect("encode again");
    assert_eq!(encoder.compiled_sizes(), 1, "the graph cache did not hit");
    assert_eq!(again.embeddings, out.embeddings, "same image, same rows");

    // `a <pad × n> b`: the placeholder count comes from the tower.
    let mut ids: Vec<u32> = vec![5];
    ids.extend(std::iter::repeat_n(0u32, out.n_tokens));
    ids.push(7);
    let seq = ids.len();

    let mut tensors = synth::synth_tensors(&cfg, 29);
    tensors.remove("per_layer_token_embd.weight");
    let (embed_table, _) = tensors
        .remove("token_embd.weight")
        .expect("token_embd in the fixture");
    let mut weights = WeightMap::from_tensors(tensors);

    let batch = assemble_multimodal(
        &cfg,
        &ids,
        &embed_table,
        &[ImageSpan::from_encode(1, &out).expect("span")],
    )
    .expect("assemble");

    // The tower's rows landed where the placeholders were.
    assert_eq!(
        &batch.inputs_embeds[hidden..(1 + out.n_tokens) * hidden],
        out.embeddings.as_slice()
    );
    // The text after the image resumed at max(grid), not at the token count.
    let after = batch.section_positions[1 + out.n_tokens][0];
    assert_eq!(after, 1 + out.grid_x.max(out.grid_y));

    let hash = PleHash::derive(&cfg, 0);
    let ple = ngram_embeddings(&hash, &batch.ple_token_ids, &[], cfg.ple_head_dim, |r| {
        Ok((0..cfg.ple_head_dim)
            .map(|j| (((r as usize + j) % 67) as f32 - 33.0) * 0.004)
            .collect())
    })
    .expect("ple rows");

    let mut opts = FlowOptions::prefill(seq);
    opts.output = OutputKind::Logits;
    opts.host_embed = true;
    opts.runtime_mrope = true;
    let built =
        build_qwen4exp_text_flow_with_options(&cfg, &mut WeightMapSource(&mut weights), opts)
            .expect("build multimodal flow");
    let (cos, sin) = rlx_qwen4exp::rope::mrope_tables(&cfg, seq, Some(&batch.section_positions));
    let id_f: Vec<f32> = ids.iter().map(|t| *t as f32).collect();
    let logits = compile_built(built, dev())
        .expect("compile")
        .run(&[
            ("input_ids", id_f.as_slice()),
            ("inputs_embeds", batch.inputs_embeds.as_slice()),
            (ROPE_COS_INPUT, cos.as_slice()),
            (ROPE_SIN_INPUT, sin.as_slice()),
            (&ple_embed_input(0), ple.as_slice()),
        ])
        .pop()
        .expect("logits");

    assert_eq!(logits.len(), seq * cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// The native tower on **real published mmproj weights**.
///
/// ```sh
/// python3 scripts/qwen4exp_subset.py --mmproj 3 mmproj-sub.gguf   # ~172 MB
/// RLX_QWEN4EXP_MMPROJ=mmproj-sub.gguf \
///   cargo test -p rlx-qwen4exp --features vision --test vision
/// ```
///
/// A 3-of-27-block slice with `clip.vision.block_count` rewritten, so the real
/// patch embedding, the real 2304-row position table, the real merger and three
/// real encoder blocks all run — at 172 MB instead of 904. Without the env var
/// this passes trivially.
///
/// Trained ViT weights are what make this worth running: the position table has
/// real structure, so the bilinear resize and the merge permutation produce a
/// distribution that a transposed read would not, and the patch kernels are not
/// symmetric, so summing the wrong pair of them shows up in the scale.
#[test]
fn the_native_tower_runs_on_real_mmproj_weights() {
    use rlx_qwen4exp::vision::VisionEncoder;
    use rlx_qwen4exp::vit::VisionConfig;

    let Some(path) = std::env::var("RLX_QWEN4EXP_MMPROJ")
        .ok()
        .filter(|s| !s.is_empty())
    else {
        return;
    };
    let text = Qwen4ExpConfig::qwen38_flash_next();
    let vcfg = VisionConfig::from_gguf_path(&path).expect("parse the real mmproj");

    // The slice rewrote the block count; everything else is the published
    // tower's shape and must match the metadata fixture above.
    let published = VisionConfig::from_gguf(&published_mmproj()).expect("fixture");
    assert_eq!(vcfg.n_embd, published.n_embd);
    assert_eq!(vcfg.n_head, published.n_head);
    assert_eq!(vcfg.patch_size, published.patch_size);
    assert_eq!(vcfg.n_merge, published.n_merge);
    assert_eq!(vcfg.n_ff, published.n_ff);
    assert_eq!(vcfg.llm_hidden_size, published.llm_hidden_size);
    assert_eq!(vcfg.llm_hidden_size, text.hidden_size, "projects to the LM");
    check_mmproj(&text, &vcfg).expect("the real mmproj must be accepted");

    let mut encoder = VisionEncoder::open(&text, &path, dev()).expect("load the real tower");
    // A small image so the ViT stays cheap; `smart_resize` still aligns it to
    // a merge block and the min-pixel floor still applies.
    let (iw, ih) = (96usize, 64usize);
    let rgb: Vec<u8> = (0..iw * ih * 3)
        .map(|i| ((i * 7 + i / 97) % 256) as u8)
        .collect();
    let out = encoder
        .encode_rgb(&rgb, iw, ih)
        .expect("encode a real image");

    assert_eq!(out.n_tokens, out.grid_x * out.grid_y);
    assert_eq!(out.embeddings.len(), out.n_tokens * text.hidden_size);
    assert!(
        out.embeddings.iter().all(|v| v.is_finite()),
        "the real tower produced non-finite rows"
    );

    // Bound the scale rather than only checking finiteness: a transposed
    // projection or a mis-summed patch kernel still yields finite numbers.
    let n = out.embeddings.len() as f32;
    let mean = out.embeddings.iter().sum::<f32>() / n;
    let sd = (out
        .embeddings
        .iter()
        .map(|x| (x - mean) * (x - mean))
        .sum::<f32>()
        / n)
        .sqrt();
    assert!(
        sd > 1e-3 && sd < 100.0,
        "the real tower's output sd is {sd} (mean {mean}) — outside anything a \
         projected patch embedding should carry"
    );

    // Distinct images must give distinct rows: a tower that collapsed to its
    // bias would pass every check above.
    let flat: Vec<u8> = vec![128u8; iw * ih * 3];
    let other = encoder
        .encode_rgb(&flat, iw, ih)
        .expect("encode a flat image");
    assert_eq!(other.n_tokens, out.n_tokens);
    let delta = out
        .embeddings
        .iter()
        .zip(&other.embeddings)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        delta > 1e-3,
        "the tower ignored its input (max delta {delta})"
    );
}
