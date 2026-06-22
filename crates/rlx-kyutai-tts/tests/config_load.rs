//! Round-trip the static [`KyutaiTtsConfig`] preset and verify it agrees with
//! the values published in `kyutai/tts-1.6b-en_fr/config.json`.

use rlx_kyutai_tts::{
    ConditionerKind, DELAYS_DEFAULT, DEPFORMER_WEIGHTS_PER_STEP_SCHEDULE, KyutaiTtsConfig,
    PositionalEmbedding,
};

#[test]
fn v1_6b_en_fr_matches_published_config() {
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();

    // Backbone.
    assert_eq!(cfg.dim, 2048);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.num_layers, 16);
    assert_eq!(cfg.context, 500);
    assert_eq!(cfg.max_period, 10_000);
    assert!((cfg.hidden_scale - 4.125).abs() < 1e-6);
    assert!(cfg.causal);
    assert_eq!(cfg.gating, "silu");
    assert_eq!(cfg.norm, "rms_norm_f32");
    assert_eq!(cfg.positional_embedding, PositionalEmbedding::Rope);

    // Codebooks / vocab.
    assert_eq!(cfg.card, 2048);
    assert_eq!(cfg.n_q, 32);
    assert_eq!(cfg.dep_q, 32);
    assert_eq!(cfg.text_card, 8000);
    assert_eq!(cfg.text_card_out, None);
    assert_eq!(cfg.existing_text_padding_id, 3);

    // Delays: 33 entries, first two zero, rest 2.
    assert_eq!(cfg.delays.len(), 33);
    assert_eq!(cfg.delays[0], 0);
    assert_eq!(cfg.delays[1], 0);
    for d in &cfg.delays[2..] {
        assert_eq!(*d, 2);
    }
    assert_eq!(cfg.delays.as_slice(), &DELAYS_DEFAULT[..]);

    // DepFormer.
    assert_eq!(cfg.depformer.dim, 1024);
    assert_eq!(cfg.depformer.num_heads, 16);
    assert_eq!(cfg.depformer.num_layers, 4);
    assert_eq!(cfg.depformer.dim_feedforward, 3072);
    assert!(cfg.depformer.multi_linear);
    assert_eq!(
        cfg.depformer.positional_embedding,
        PositionalEmbedding::None
    );
    assert!(cfg.depformer.weights_per_step);
    assert_eq!(cfg.depformer.low_rank_embeddings, 128);
    assert_eq!(cfg.depformer.weights_per_step_schedule.len(), 32);
    assert_eq!(
        cfg.depformer.weights_per_step_schedule.as_slice(),
        &DEPFORMER_WEIGHTS_PER_STEP_SCHEDULE[..]
    );

    // Multistream / cross-attention.
    assert!(cfg.demux_second_stream);
    assert!(cfg.cross_attention);

    // Conditioners.
    assert!(cfg.conditioners.contains_key("speaker_wavs"));
    assert!(cfg.conditioners.contains_key("cfg"));
    assert!(cfg.conditioners.contains_key("control"));
    match cfg.conditioners.get("speaker_wavs").unwrap() {
        ConditionerKind::Tensor { tensor } => assert_eq!(tensor.dim, 512),
        _ => panic!("speaker_wavs must be a tensor conditioner"),
    }
    match cfg.conditioners.get("cfg").unwrap() {
        ConditionerKind::Lut { lut } => {
            assert_eq!(lut.n_bins, 7);
            assert_eq!(lut.dim, 16);
            assert_eq!(lut.possible_values.len(), 7);
        }
        _ => panic!("cfg must be a LUT conditioner"),
    }
    match cfg.conditioners.get("control").unwrap() {
        ConditionerKind::Lut { lut } => {
            assert_eq!(lut.n_bins, 1);
            assert_eq!(lut.dim, 2048);
        }
        _ => panic!("control must be a LUT conditioner"),
    }

    // Fuser.
    assert!(cfg.fuser.cross_attention_pos_emb);
    assert!((cfg.fuser.cross_attention_pos_emb_scale - 1.0).abs() < 1e-6);
    assert_eq!(cfg.fuser.sum, vec!["control", "cfg"]);
    assert!(cfg.fuser.prepend.is_empty());
    assert_eq!(cfg.fuser.cross, vec!["speaker_wavs"]);

    // TTS config.
    assert!((cfg.tts_config.audio_delay - 1.28).abs() < 1e-6);
    assert_eq!(cfg.tts_config.second_stream_ahead, 2);
    assert_eq!(cfg.audio_delay_frames(), 16); // 1.28 s × 12.5 Hz
    assert_eq!(cfg.audio_pad_token(), 2048);
}

#[test]
fn backbone_view_derives_dim_feedforward() {
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let backbone = cfg.backbone();
    assert_eq!(backbone.d_model, cfg.dim);
    assert_eq!(backbone.num_heads, cfg.num_heads);
    assert_eq!(backbone.num_layers, cfg.num_layers);
    assert_eq!(backbone.context, cfg.context);
    // dim_ff ≈ dim × hidden_scale = 2048 × 4.125 = 8448.
    assert_eq!(backbone.dim_feedforward, 8448);
}

#[test]
fn config_roundtrips_through_serde_json() {
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: KyutaiTtsConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.dim, cfg.dim);
    assert_eq!(
        back.depformer.weights_per_step_schedule,
        cfg.depformer.weights_per_step_schedule
    );
    assert_eq!(back.delays, cfg.delays);
    assert_eq!(back.conditioners.len(), cfg.conditioners.len());
}

#[test]
fn depformer_schedule_maps_32_codebooks_to_11_heads() {
    use std::collections::HashSet;
    let unique: HashSet<usize> = DEPFORMER_WEIGHTS_PER_STEP_SCHEDULE
        .iter()
        .copied()
        .collect();
    assert_eq!(
        unique.len(),
        11,
        "33-entry schedule collapses to 11 distinct heads"
    );
    assert!(unique.contains(&0));
    assert!(unique.contains(&10));
}
