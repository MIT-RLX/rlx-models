// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Prompt assembly — mirrors VibeASR.cpp `utils/prompt_builder.h`.
//
// Special tokens are inserted by id (they may not round-trip through the GGUF
// BPE vocab's `parse_special`); only the plain-text segments go through the
// tokenizer. Layout (Qwen2.5 chat template):
//
//   <|im_start|> system\n{SYSTEM_PROMPT} <|im_end|> \n
//   <|im_start|> user\n <|speech_start|> <|speech_pad|>*N <|speech_end|>
//                \n{suffix} <|im_end|> \n
//
// N = ceil(n_samples / compress_ratio). No generation prompt is appended —
// the model emits `<|im_start|>assistant\n` itself.

use crate::config::{
    COMPRESS_RATIO, SYSTEM_PROMPT, TOK_IM_END, TOK_IM_START, TOK_SPEECH_END, TOK_SPEECH_PAD,
    TOK_SPEECH_START,
};

/// The assembled prompt and where the speech-embedding rows go.
#[derive(Debug, Clone)]
pub struct PromptTokens {
    /// Full token id sequence (speech_pad rows are placeholders).
    pub tokens: Vec<i64>,
    /// Index of the first `<|speech_pad|>` token.
    pub speech_pad_start: usize,
    /// Number of `<|speech_pad|>` tokens (== VAE frame count).
    pub speech_pad_count: usize,
}

/// Build the transcription suffix string (matches `build_prompt`).
fn user_suffix(duration_sec: f32, context_info: Option<&str>, json_format: bool) -> String {
    let instr = if json_format {
        "please transcribe it with these keys: Start, End, Speaker, Content"
    } else {
        "please transcribe it."
    };
    match context_info {
        Some(ctx) if !ctx.is_empty() => {
            let tail = if json_format {
                "Please transcribe it with these keys: Start, End, Speaker, Content"
            } else {
                "Please transcribe it."
            };
            format!("\nThis is a {duration_sec:.2} seconds audio, with extra info: {ctx}\n\n{tail}")
        }
        _ => format!("\nThis is a {duration_sec:.2} seconds audio, {instr}"),
    }
}

/// Assemble the prompt. `tokenize` must encode plain text WITHOUT adding or
/// parsing special tokens (equivalent to `llama_tokenize(add_special=false,
/// parse_special=false)`).
pub fn build_prompt(
    tokenize: impl Fn(&str) -> Vec<i64>,
    n_samples: usize,
    compress_ratio: usize,
    duration_sec: f32,
    context_info: Option<&str>,
    json_format: bool,
) -> PromptTokens {
    let vae_tok_len = n_samples.div_ceil(compress_ratio);

    let system_content = tokenize(&format!("system\n{SYSTEM_PROMPT}"));
    let newline = tokenize("\n");
    let user_prefix = tokenize("user\n");
    let suffix = tokenize(&user_suffix(duration_sec, context_info, json_format));

    let mut tokens: Vec<i64> = Vec::new();
    // system: <|im_start|> system\n{SYSTEM_PROMPT} <|im_end|> \n
    tokens.push(TOK_IM_START);
    tokens.extend_from_slice(&system_content);
    tokens.push(TOK_IM_END);
    tokens.extend_from_slice(&newline);
    // user: <|im_start|> user\n <|speech_start|> pad*N <|speech_end|> \n{suffix} <|im_end|> \n
    tokens.push(TOK_IM_START);
    tokens.extend_from_slice(&user_prefix);
    tokens.push(TOK_SPEECH_START);

    let speech_pad_start = tokens.len();
    tokens.extend(std::iter::repeat_n(TOK_SPEECH_PAD, vae_tok_len));

    tokens.push(TOK_SPEECH_END);
    tokens.extend_from_slice(&suffix);
    tokens.push(TOK_IM_END);
    tokens.extend_from_slice(&newline);

    PromptTokens {
        tokens,
        speech_pad_start,
        speech_pad_count: vae_tok_len,
    }
}

/// Convenience: default compression ratio (3200).
pub fn build_prompt_default(
    tokenize: impl Fn(&str) -> Vec<i64>,
    n_samples: usize,
    duration_sec: f32,
    context_info: Option<&str>,
    json_format: bool,
) -> PromptTokens {
    build_prompt(
        tokenize,
        n_samples,
        COMPRESS_RATIO,
        duration_sec,
        context_info,
        json_format,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fake tokenizer: 1 token per whitespace-split word (ids arbitrary but
    // distinct from the special ids so structure is checkable).
    fn fake(s: &str) -> Vec<i64> {
        s.split_whitespace().map(|_| 1000i64).collect()
    }

    #[test]
    fn speech_pad_region_placed() {
        let p = build_prompt(fake, 6400, 3200, 1.5, None, false);
        assert_eq!(p.speech_pad_count, 2);
        for i in 0..p.speech_pad_count {
            assert_eq!(p.tokens[p.speech_pad_start + i], TOK_SPEECH_PAD);
        }
        // speech_start immediately precedes the pad region.
        assert_eq!(p.tokens[p.speech_pad_start - 1], TOK_SPEECH_START);
        // speech_end immediately follows.
        assert_eq!(
            p.tokens[p.speech_pad_start + p.speech_pad_count],
            TOK_SPEECH_END
        );
        assert_eq!(p.tokens[0], TOK_IM_START);
    }

    #[test]
    fn frames_ceil() {
        let p = build_prompt(fake, 3201, 3200, 1.0, None, false);
        assert_eq!(p.speech_pad_count, 2);
    }
}
