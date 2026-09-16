//! ChatML prompt builders — must match FireRedAudio training character-for-character.

use anyhow::{Result, bail};

/// Default ASR user question (upstream `DEFAULT_ASR_PROMPT`).
pub const DEFAULT_ASR_PROMPT: &str = "Transcribe speech to text.";

/// Generic assistant system prompt (TTS / edit / voice design).
pub const GENERIC_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

/// Understanding / ASR system prompt.
pub const UNDERSTAND_SYSTEM_PROMPT: &str =
    "You are an audio understanding expert. Please answer user questions based on the audio.";

/// Closed think block used when chain-of-thought is off.
pub const THINK_CLOSED: &str = "<think>\n\n</think>\n\n";

/// Open think block when `enable_thinking` leaves reasoning unfinished.
pub const THINK_OPEN: &str = "<think>\n";

/// Marker that ends a CoT span in model output.
pub const THINK_END: &str = "</think>";

/// Which FireRedAudio CLI / API task is being run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FireRedTask {
    Asr,
    Understand,
    Tts,
    Edit,
    VoiceDesign,
}

impl FireRedTask {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::Understand => "understand",
            Self::Tts => "tts",
            Self::Edit => "edit",
            Self::VoiceDesign => "voice_design",
        }
    }

    pub fn is_understanding(self) -> bool {
        matches!(self, Self::Asr | Self::Understand)
    }

    pub fn is_generation(self) -> bool {
        matches!(self, Self::Tts | Self::Edit | Self::VoiceDesign)
    }

    /// Only `understand` is trained with a non-empty think block.
    pub fn supports_thinking(self) -> bool {
        matches!(self, Self::Understand)
    }
}

/// Semantic (rewrite content) vs acoustic (pitch / speed / volume) edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditType {
    Semantic,
    Acoustic,
}

impl EditType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Acoustic => "acoustic",
        }
    }
}

/// Assemble the shared ChatML skeleton used by every task.
pub fn chatml(
    system: &str,
    user: &str,
    assistant_prefix: &str,
    enable_thinking: bool,
) -> Result<String> {
    if enable_thinking && !assistant_prefix.is_empty() {
        bail!("enable_thinking cannot be combined with assistant_prefix");
    }
    let think = if enable_thinking {
        THINK_OPEN
    } else {
        THINK_CLOSED
    };
    Ok(format!(
        "<|im_start|>system\n{system}<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n{think}{assistant_prefix}"
    ))
}

/// ASR and audio understanding: numbered `Audio N:` segments then the question.
pub fn build_understand_prompt(
    prompt: &str,
    num_audios: usize,
    audio_sp_token: &str,
    enable_thinking: bool,
) -> Result<String> {
    if num_audios == 0 {
        bail!("understand / asr require at least one audio");
    }
    let mut audio_segs = String::new();
    for i in 0..num_audios {
        audio_segs.push_str(&format!(
            "Audio {}: <|sosp|>{audio_sp_token}<|eosp|>\n",
            i + 1
        ));
    }
    chatml(
        UNDERSTAND_SYSTEM_PROMPT,
        &format!("{audio_segs}{prompt}"),
        "",
        enable_thinking,
    )
}

/// Convenience: ASR with the default transcription prompt and closed think block.
pub fn build_asr_prompt(num_audios: usize, audio_sp_token: &str) -> Result<String> {
    build_understand_prompt(DEFAULT_ASR_PROMPT, num_audios, audio_sp_token, false)
}

/// ICL voice cloning. `language == "en"` inserts a space between prompt and target text.
pub fn build_tts_prompt(
    prompt_text: &str,
    target_text: &str,
    language: &str,
    audio_sp_token: &str,
) -> Result<String> {
    let sep = if language == "en" { " " } else { "" };
    chatml(
        GENERIC_SYSTEM_PROMPT,
        &format!("Convert text to speech.\n{prompt_text}{sep}{target_text}"),
        &format!("<|sosp|>{audio_sp_token}"),
        false,
    )
}

/// Speech editing. Semantic prepends the identify-content cue; acoustic uses the
/// instruction alone (must be a trained pitch/speed/volume template).
pub fn build_edit_prompt(
    instruction: &str,
    edit_type: EditType,
    audio_sp_token: &str,
) -> Result<String> {
    let user_text = match edit_type {
        EditType::Semantic => format!("Identify the content of the audio. {instruction}"),
        EditType::Acoustic => instruction.to_string(),
    };
    chatml(
        GENERIC_SYSTEM_PROMPT,
        &format!("Audio 1: <|sosp|>{audio_sp_token}<|eosp|>\n{user_text}"),
        "",
        false,
    )
}

/// Voice design from a timbre description. The Chinese bridge sentence is fixed
/// training text and must stay even when `instruction` / `text` are English.
pub fn build_voice_design_prompt(instruction: &str, text: &str) -> Result<String> {
    chatml(
        GENERIC_SYSTEM_PROMPT,
        &format!("{instruction}\n\n根据上述音色描述，合成以下文本对应的音频：\n{text}"),
        "",
        false,
    )
}

/// Split `"{reasoning}</think>…{answer}"` into `(Some(reasoning), answer)`.
/// Returns `(None, text)` when `</think>` is absent.
pub fn split_thinking(text: &str) -> (Option<&str>, &str) {
    match text.split_once(THINK_END) {
        Some((reasoning, answer)) => (Some(reasoning.trim()), answer.trim()),
        None => (None, text),
    }
}
