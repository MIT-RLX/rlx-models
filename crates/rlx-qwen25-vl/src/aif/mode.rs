// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

/// AIF token-dynamics source (matches `RLX_AIF_DYNAMICS` in HF reference scripts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AifDynamicsMode {
    /// Eq. 2 — visual queries attend to text keys during prefill.
    #[default]
    PrefillV2t,
    /// Fig. 6 — one decode step: text query attends to visual keys.
    DecodeStep,
}

impl AifDynamicsMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "prefill_v2t" | "prefill" | "v2t" => Some(Self::PrefillV2t),
            "decode_step" | "decode" => Some(Self::DecodeStep),
            _ => None,
        }
    }

    pub fn from_env() -> Self {
        std::env::var("RLX_AIF_DYNAMICS")
            .ok()
            .and_then(|s| Self::parse(&s))
            .unwrap_or_default()
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::PrefillV2t => "prefill_v2t",
            Self::DecodeStep => "decode_step",
        }
    }
}
