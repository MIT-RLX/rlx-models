use serde::{Deserialize, Serialize};

/// Positional embedding style for a transformer stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionalEmbedding {
    Rope,
    Sin,
    None,
}

/// Single transformer stack config (temporal or depth).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dim_feedforward: usize,
    pub causal: bool,
    pub norm_first: bool,
    pub context: usize,
    pub max_period: usize,
    pub positional_embedding: PositionalEmbedding,
    pub kv_repeat: usize,
}

impl TransformerConfig {
    /// SwiGLU hidden width when `dim_feedforward == 4 * d_model` (Kyutai Moshi v0.1).
    pub fn swiglu_hidden(&self) -> usize {
        if self.dim_feedforward == 4 * self.d_model {
            11 * self.d_model / 4
        } else {
            2 * self.dim_feedforward / 3
        }
    }
}

/// Depth decoder config — one mini-transformer per generated codebook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepFormerConfig {
    pub transformer: TransformerConfig,
    pub num_slices: usize,
}

/// Full LM config (Helium temporal + optional DepFormer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmConfig {
    pub transformer: TransformerConfig,
    pub depformer: Option<DepFormerConfig>,
    pub text_in_vocab_size: usize,
    pub text_out_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_codebooks: usize,
}

impl LmConfig {
    fn depformer_cfg(num_slices: usize) -> DepFormerConfig {
        DepFormerConfig {
            num_slices,
            transformer: TransformerConfig {
                d_model: 1024,
                num_heads: 16,
                num_layers: 6,
                dim_feedforward: 1024 * 4,
                causal: true,
                norm_first: true,
                context: num_slices,
                max_period: 10_000,
                positional_embedding: PositionalEmbedding::None,
                kv_repeat: 1,
            },
        }
    }

    /// Moshiko / Moshika 7B (8 generated + 8 user codebooks).
    pub fn v0_1() -> Self {
        Self {
            transformer: TransformerConfig {
                d_model: 4096,
                num_heads: 32,
                num_layers: 32,
                dim_feedforward: 4096 * 4,
                causal: true,
                norm_first: true,
                context: 3000,
                max_period: 10_000,
                positional_embedding: PositionalEmbedding::Rope,
                kv_repeat: 1,
            },
            depformer: Some(Self::depformer_cfg(8)),
            audio_vocab_size: 2049,
            text_in_vocab_size: 32_001,
            text_out_vocab_size: 32_000,
            audio_codebooks: 8,
        }
    }

    /// Full-duplex with 16 audio embedding tables (8 Moshi + 8 user).
    pub fn v0_1_streaming(num_slices: usize) -> Self {
        let mut s = Self::v0_1();
        s.audio_codebooks = 16;
        if let Some(dep) = s.depformer.as_mut() {
            dep.num_slices = num_slices;
            dep.transformer.context = num_slices;
        }
        s
    }
}

/// Runtime generation hyper-parameters (acoustic delay, codebook counts).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateConfig {
    pub generated_audio_codebooks: usize,
    pub input_audio_codebooks: usize,
    pub audio_vocab_size: usize,
    pub acoustic_delay: usize,
    pub text_pad_token: u32,
    pub text_eop_token: u32,
    pub text_start_token: u32,
}

impl GenerateConfig {
    pub fn v0_1() -> Self {
        Self {
            generated_audio_codebooks: 8,
            input_audio_codebooks: 8,
            audio_vocab_size: 2049,
            acoustic_delay: 2,
            text_eop_token: 0,
            text_pad_token: 3,
            text_start_token: 32_000,
        }
    }

    /// TTS / one-way: no user audio codebooks.
    pub fn v0_1_one_way() -> Self {
        Self {
            generated_audio_codebooks: 8,
            input_audio_codebooks: 0,
            ..Self::v0_1()
        }
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    pub fn total_audio_codebooks(&self) -> usize {
        self.generated_audio_codebooks + self.input_audio_codebooks
    }
}

/// Kyutai Moshi voice family — routes HuggingFace repos for a given quant preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoshiVoice {
    /// Male voice (`kyutai/moshiko-*` repos).
    Moshiko,
    /// Female voice (`kyutai/moshika-*` repos).
    Moshika,
}

impl MoshiVoice {
    /// Parse CLI / env voice names (`moshiko`, `moshika`, `male`, `female`).
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "moshiko" | "male" => Some(Self::Moshiko),
            "moshika" | "female" => Some(Self::Moshika),
            _ => None,
        }
    }

    /// Read `RLX_MOSHI_VOICE`, defaulting to Moshiko.
    pub fn from_env_or_default() -> Self {
        std::env::var("RLX_MOSHI_VOICE")
            .ok()
            .and_then(|s| Self::parse(&s))
            .unwrap_or(Self::Moshiko)
    }
}

/// Runtime mode: voice family × one-way vs full-duplex codebook layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoshiVariant {
    /// Moshiko full-duplex (8+8 codebooks).
    Moshiko,
    /// Moshiko one-way TTS-style (8 generated codebooks only).
    MoshikoOneWay,
    /// Moshika full-duplex.
    Moshika,
    /// Moshika one-way TTS-style.
    MoshikaOneWay,
}

impl MoshiVariant {
    /// Voice family for HuggingFace checkpoint routing.
    pub fn voice(self) -> MoshiVoice {
        match self {
            Self::Moshiko | Self::MoshikoOneWay => MoshiVoice::Moshiko,
            Self::Moshika | Self::MoshikaOneWay => MoshiVoice::Moshika,
        }
    }

    /// True for TTS-style presets with no user audio codebooks.
    pub fn is_one_way(self) -> bool {
        matches!(self, Self::MoshikoOneWay | Self::MoshikaOneWay)
    }

    /// True for full-duplex presets (user + Moshi audio streams).
    pub fn is_duplex(self) -> bool {
        matches!(self, Self::Moshiko | Self::Moshika)
    }

    /// Parse CLI variant names (`moshiko`, `moshika-one-way`, `duplex`, …).
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "moshiko-one-way" | "moshiko-oneway" | "oneway" => Some(Self::MoshikoOneWay),
            "moshiko" | "duplex" => Some(Self::Moshiko),
            "moshika-one-way" | "moshika-oneway" => Some(Self::MoshikaOneWay),
            "moshika" => Some(Self::Moshika),
            _ => None,
        }
    }

    /// HuggingFace repo for this variant and checkpoint preset.
    pub fn hf_repo(self, checkpoint: super::checkpoint::MoshiCheckpoint) -> &'static str {
        checkpoint.hf_repo(self.voice())
    }

    /// Static LM architecture config (layer counts, codebook tables).
    pub fn lm_config(self) -> LmConfig {
        match self {
            Self::Moshiko | Self::Moshika => LmConfig::v0_1_streaming(8),
            Self::MoshikoOneWay | Self::MoshikaOneWay => LmConfig::v0_1(),
        }
    }

    /// Per-generation hyper-parameters (acoustic delay, pad tokens).
    pub fn generate_config(self) -> GenerateConfig {
        match self {
            Self::Moshiko | Self::Moshika => GenerateConfig::v0_1(),
            Self::MoshikoOneWay | Self::MoshikaOneWay => GenerateConfig::v0_1_one_way(),
        }
    }
}
