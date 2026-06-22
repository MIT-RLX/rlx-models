use rlx_moshi::{MoshiCheckpoint, MoshiVariant, MoshiVoice};

#[test]
fn checkpoint_hf_repo_routing() {
    assert_eq!(
        MoshiCheckpoint::Q4MlxSafetensors.hf_repo(MoshiVoice::Moshiko),
        "kyutai/moshiko-mlx-q4"
    );
    assert_eq!(
        MoshiCheckpoint::Q4MlxSafetensors.hf_repo(MoshiVoice::Moshika),
        "kyutai/moshika-mlx-q4"
    );
    assert_eq!(
        MoshiCheckpoint::MlxBf16Safetensors.hf_repo(MoshiVoice::Moshiko),
        "kyutai/moshiko-mlx-bf16"
    );
    assert_eq!(
        MoshiCheckpoint::MlxBf16Safetensors.hf_repo(MoshiVoice::Moshika),
        "kyutai/moshika-mlx-bf16"
    );
    assert_eq!(
        MoshiVariant::Moshika.hf_repo(MoshiCheckpoint::Q8Gguf),
        "kyutai/moshika-candle-q8"
    );
}

#[test]
fn checkpoint_cache_dirs_by_voice() {
    assert_eq!(
        MoshiCheckpoint::Q8Gguf
            .default_cache_dir(MoshiVoice::Moshiko)
            .to_str()
            .unwrap(),
        ".cache/moshiko-q8"
    );
    assert_eq!(
        MoshiCheckpoint::Q8Gguf
            .default_cache_dir(MoshiVoice::Moshika)
            .to_str()
            .unwrap(),
        ".cache/moshika-q8"
    );
    assert_eq!(
        MoshiCheckpoint::MlxBf16Safetensors
            .default_cache_dir(MoshiVoice::Moshiko)
            .to_str()
            .unwrap(),
        ".cache/moshiko-mlx-bf16"
    );
}

#[test]
fn checkpoint_parse_mlx_bf16() {
    assert_eq!(
        MoshiCheckpoint::parse("mlx-bf16"),
        Some(MoshiCheckpoint::MlxBf16Safetensors)
    );
    assert!(MoshiCheckpoint::MlxBf16Safetensors.is_mlx());
    assert!(!MoshiCheckpoint::MlxBf16Safetensors.is_mlx_quantized());
}

#[test]
fn variant_voice_and_mode() {
    assert_eq!(MoshiVariant::MoshikaOneWay.voice(), MoshiVoice::Moshika);
    assert!(MoshiVariant::Moshika.is_duplex());
    assert!(MoshiVariant::MoshikaOneWay.is_one_way());
    assert_eq!(
        MoshiVariant::parse("moshika-one-way"),
        Some(MoshiVariant::MoshikaOneWay)
    );
}
