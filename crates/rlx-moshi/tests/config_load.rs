use rlx_moshi::{GenerateConfig, LmConfig, MoshiVariant};

#[test]
fn presets_match_kyutai_v0_1() {
    let lm = LmConfig::v0_1();
    assert_eq!(lm.transformer.d_model, 4096);
    assert_eq!(lm.transformer.num_layers, 32);
    assert_eq!(lm.audio_vocab_size, 2049);
    assert_eq!(lm.text_in_vocab_size, 32_001);

    let gcfg = GenerateConfig::v0_1_one_way();
    assert_eq!(gcfg.generated_audio_codebooks, 8);
    assert_eq!(gcfg.input_audio_codebooks, 0);
    assert_eq!(gcfg.acoustic_delay, 2);
    assert_eq!(gcfg.text_start_token, 32_000);
}

#[test]
fn variant_configs() {
    let one = MoshiVariant::MoshikoOneWay.lm_config();
    assert_eq!(one.audio_codebooks, 8);
    let full = MoshiVariant::Moshiko.lm_config();
    assert_eq!(full.audio_codebooks, 16);
    let moshika = MoshiVariant::Moshika.lm_config();
    assert_eq!(moshika.audio_codebooks, 16);
    let moshika_one = MoshiVariant::MoshikaOneWay.lm_config();
    assert_eq!(moshika_one.audio_codebooks, 8);
}
