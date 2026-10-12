// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `Qwen4ExpConfig` against the two shapes Qwen3.8-Flash-Next actually ships
//! in.
//!
//! The HF fixture is `Qwen/Qwen3.8-Flash-Next`'s real `config.json`. The GGUF
//! case is built from the metadata keys read out of
//! `unsloth/Qwen3.8-Flash-Next-GGUF`'s first shard, which carries all 67 KV
//! pairs and zero tensors.
//!
//! Both readers must land on the same 48-layer model, which is the point: the
//! converter renames nearly every field, turns `layer_types` into an integer
//! `attention.compress_ratios` array, re-bases `ple_layer_ids` from 1 to 0, and
//! folds `partial_rotary_factor` into a `rope.dimension_count`.

use rlx_gguf::{GgufFile, MetaValue};
use rlx_qwen4exp::Qwen4ExpConfig;
use rlx_qwen4exp::config::LayerKind;

const HF_CONFIG: &str = include_str!("fixtures/qwen38_flash_next_config.json");

/// `qwen4exp.ple.head_vocab_sizes` — 16 distinct primes just above 20 M.
const HEAD_VOCAB_SIZES: [u64; 16] = [
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
    20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
];
/// `qwen4exp.ple.head_offsets` — the running sum of the above.
const HEAD_OFFSETS: [u64; 16] = [
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
    180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
];

/// The published `unsloth/Qwen3.8-Flash-Next-GGUF` metadata, verbatim.
fn published_gguf() -> GgufFile {
    let mut g = GgufFile::empty();
    g.metadata.insert(
        "general.architecture".into(),
        MetaValue::String("qwen4exp".into()),
    );
    g.metadata.insert(
        "general.name".into(),
        MetaValue::String("Qwen3.8 Flash Next".into()),
    );

    for (k, v) in [
        ("qwen4exp.block_count", 48u32),
        ("qwen4exp.context_length", 262_144),
        ("qwen4exp.embedding_length", 2560),
        ("qwen4exp.attention.head_count", 24),
        ("qwen4exp.attention.head_count_kv", 2),
        ("qwen4exp.attention.key_length", 256),
        ("qwen4exp.attention.value_length", 256),
        ("qwen4exp.expert_count", 512),
        ("qwen4exp.expert_used_count", 10),
        ("qwen4exp.expert_feed_forward_length", 640),
        ("qwen4exp.expert_shared_feed_forward_length", 640),
        ("qwen4exp.ssm.conv_kernel", 4),
        ("qwen4exp.ssm.state_size", 128),
        ("qwen4exp.ssm.group_count", 16),
        ("qwen4exp.ssm.time_step_rank", 48),
        ("qwen4exp.ssm.inner_size", 6144),
        ("qwen4exp.full_attention_interval", 4),
        ("qwen4exp.rope.dimension_count", 64),
        ("qwen4exp.hyper_connection.count", 4),
        ("qwen4exp.hyper_connection.low_rank", 320),
        ("qwen4exp.attention.indexer.head_count", 4),
        ("qwen4exp.attention.indexer.key_length", 128),
        ("qwen4exp.attention.indexer.top_k", 2048),
        ("qwen4exp.ple.ngram_size", 3),
        ("qwen4exp.ple.heads_per_ngram", 8),
        ("qwen4exp.ple.conv_kernel", 4),
        ("qwen4exp.ple.eos_token_id", 248_044),
        ("qwen4exp.ple.image_token_id", 248_056),
        ("qwen4exp.embedding_length_per_layer_input", 160),
    ] {
        g.metadata.insert(k.into(), MetaValue::U32(v));
    }
    g.metadata.insert(
        "qwen4exp.attention.layer_norm_rms_epsilon".into(),
        MetaValue::F32(1e-6),
    );
    g.metadata
        .insert("qwen4exp.rope.freq_base".into(), MetaValue::F32(1e7));

    let arr_u32 = |v: Vec<u32>| MetaValue::Array(v.into_iter().map(MetaValue::U32).collect());
    let arr_u64 = |v: Vec<u64>| MetaValue::Array(v.into_iter().map(MetaValue::U64).collect());

    g.metadata.insert(
        "qwen4exp.rope.dimension_sections".into(),
        arr_u32(vec![11, 11, 10, 0]),
    );
    // The layer schedule, as an integer array: 0 is a linear-attention layer.
    g.metadata.insert(
        "qwen4exp.attention.compress_ratios".into(),
        arr_u32(
            (0..48)
                .map(|i| if (i + 1) % 4 == 0 { 4 } else { 0 })
                .collect(),
        ),
    );
    // ZERO-based here; the HF config writes `[2]`, one-based.
    g.metadata
        .insert("qwen4exp.ple.layers".into(), arr_u32(vec![1]));
    g.metadata.insert(
        "qwen4exp.ple.layer_multipliers".into(),
        arr_u64(vec![
            23_703_573_157_769,
            20_109_073_645_365,
            8_052_911_324_071,
        ]),
    );
    g.metadata.insert(
        "qwen4exp.ple.head_offsets".into(),
        arr_u64(HEAD_OFFSETS.to_vec()),
    );
    g.metadata.insert(
        "qwen4exp.ple.head_vocab_sizes".into(),
        arr_u64(HEAD_VOCAB_SIZES.to_vec()),
    );

    // The published file has NO `qwen4exp.vocab_size` key, so the reader falls
    // back to the tokenizer's token count. Only the length is read, so the
    // entries can be anything.
    g.metadata.insert(
        "tokenizer.ggml.tokens".into(),
        MetaValue::Array(vec![MetaValue::U32(0); 248_320]),
    );
    g
}

#[test]
fn gguf_metadata_parses_the_published_model() {
    let cfg = Qwen4ExpConfig::from_gguf(&published_gguf()).expect("parse GGUF metadata");

    assert_eq!(cfg.vocab_size, 248_320, "read off the tokenizer token list");
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.rope_dim_count, 64, "partial rotary: 64 of 256");
    // The converter already folded the reference's `1 +` into every gamma.
    assert!(!cfg.rms_norm_offset);

    // 36 linear + 12 QSA, derived from `compress_ratios` alone.
    assert_eq!(
        cfg.layer_types
            .iter()
            .filter(|k| **k == LayerKind::Linear)
            .count(),
        36
    );
    assert_eq!(cfg.layer_kind(0), LayerKind::Linear);
    assert_eq!(cfg.layer_kind(3), LayerKind::Indexed);
    assert_eq!(cfg.indexer_compress_ratio, 4, "derived from the array");

    assert_eq!(cfg.ple_layers, vec![1]);
    assert_eq!(cfg.ple_head_dim, 160);
    assert_eq!(cfg.ple_embed_dim(), 2560, "16 heads × 160");
    assert_eq!(cfg.gdn_conv_dim(), 10_240);
    assert_eq!(cfg.hc_dim(), 10_240);

    // `validate` checks the derived hash against these, so reaching here at all
    // means the derivation reproduced the checkpoint's own constants.
    let c = cfg.ple_constants.as_ref().expect("hash constants");
    assert_eq!(c.head_vocab_sizes, HEAD_VOCAB_SIZES);
    assert_eq!(c.head_offsets, HEAD_OFFSETS);
}

/// The two readers must describe the same model.
///
/// This is the test that catches a converter field being misread: the HF config
/// and the GGUF metadata share almost no key names, so agreeing on all 40-odd
/// fields is not something two independent mistakes would do.
#[test]
fn the_gguf_and_hf_readers_agree() {
    let gguf = Qwen4ExpConfig::from_gguf(&published_gguf()).expect("gguf");
    let mut hf = Qwen4ExpConfig::from_hf_json(HF_CONFIG).expect("hf");

    // The three fields that legitimately differ by format:
    //  * HF gammas are zero-centred, a GGUF's are pre-offset;
    //  * HF keeps the hash constants as checkpoint buffers, not config;
    //  * the MTP block is declared in the HF config but shipped as its own
    //    GGUF, so the trunk file does not count it.
    assert!(hf.rms_norm_offset && !gguf.rms_norm_offset);
    hf.rms_norm_offset = false;
    hf.ple_constants = gguf.ple_constants.clone();
    assert_eq!(hf.num_nextn_predict_layers, 1);
    assert_eq!(gguf.num_nextn_predict_layers, 0);
    hf.num_nextn_predict_layers = 0;

    assert_eq!(hf, gguf);
}

/// A GGUF whose hash constants disagree with the derivation must be rejected,
/// not silently trusted: the derived multipliers are what the row addressing
/// uses, so a mismatch means the embeddings come from the wrong rows of a
/// 320-million-row table and nothing else goes wrong.
#[test]
fn a_tampered_hash_constant_is_rejected() {
    let mut g = published_gguf();
    g.metadata.insert(
        "qwen4exp.ple.layer_multipliers".into(),
        MetaValue::Array(
            vec![23_703_573_157_769u64, 20_109_073_645_365, 8_052_911_324_072]
                .into_iter()
                .map(MetaValue::U64)
                .collect(),
        ),
    );
    let err = Qwen4ExpConfig::from_gguf(&g).unwrap_err().to_string();
    assert!(err.contains("PLE multipliers"), "{err}");
}

/// A 45-bit multiplier must survive the metadata read. `as_u32` would truncate
/// it to something that still looks like a plausible odd constant.
#[test]
fn hash_multipliers_are_read_at_full_width() {
    let cfg = Qwen4ExpConfig::from_gguf(&published_gguf()).expect("gguf");
    let m = &cfg.ple_constants.as_ref().unwrap().layer_multipliers;
    assert_eq!(m[0], 23_703_573_157_769);
    assert!(m[0] > u32::MAX as u64, "the multiplier does not fit a u32");
}

/// QSA layers must share one block size; a mixed array is a converter bug and
/// the loader should say so rather than pick one.
#[test]
fn mixed_compress_ratios_are_rejected() {
    let mut g = published_gguf();
    g.metadata.insert(
        "qwen4exp.attention.compress_ratios".into(),
        MetaValue::Array(
            (0..48)
                .map(|i| {
                    MetaValue::U32(if (i + 1) % 4 == 0 {
                        4 + (i as u32 / 16)
                    } else {
                        0
                    })
                })
                .collect(),
        ),
    );
    let err = Qwen4ExpConfig::from_gguf(&g).unwrap_err().to_string();
    assert!(err.contains("share one compress ratio"), "{err}");
}

/// Without the array, the schedule has to fall back to
/// `full_attention_interval` — and land on the same 36/12 split.
#[test]
fn the_schedule_falls_back_to_the_interval() {
    let mut g = published_gguf();
    g.metadata.remove("qwen4exp.attention.compress_ratios");
    let cfg = Qwen4ExpConfig::from_gguf(&g).expect("interval fallback");
    assert_eq!(
        cfg.layer_types,
        Qwen4ExpConfig::qwen38_flash_next().layer_types
    );
    assert_eq!(cfg.indexer_compress_ratio, 4, "from the indexer key group");
}
