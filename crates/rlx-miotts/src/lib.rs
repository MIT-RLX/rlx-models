//! MioTTS-0.6B — Qwen3 LM + MioCodec (25 Hz / 24 kHz) on RLX.
//!
//! Pipeline: chat prompt → Qwen3 AR (`<|s_N|>` speech tokens) → MioCodec decode
//! (ONNX body → mag/phase + host ISTFT) → 24 kHz PCM. Voice via preset global
//! embeddings (`en_female`, …). Codec runs on any RLX backend; LM is eager CPU.

pub mod codec;
pub mod lm;
pub mod session;
pub mod tokens;

pub use codec::{MioCodec, SAMPLE_RATE, load_preset_embedding};
pub use lm::{MioLm, MioLmConfig};
pub use session::{
    GenerateOpts, MioSession, SynthesisResult, default_codec_dir, default_model_dir,
};
pub use tokens::{SPEECH_BASE, SPEECH_LEN, fit_speech_len, parse_speech_codes};
