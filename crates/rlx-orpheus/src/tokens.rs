// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Orpheus prompt + speech-code token helpers.

use anyhow::Context;
use once_cell::sync::Lazy;
use regex::Regex;

/// First GGUF token id for `<custom_token_0>` on Orpheus / Llama-3 vocab.
pub const CUSTOM_TOKEN_BASE: u32 = 128_256;

/// Offset applied when mapping SNAC frame codes to GGUF token ids (see Orpheus
/// `turn_token_into_id`: `id = CUSTOM_TOKEN_BASE + code + 10 + slot * 4096`).
pub const SNAC_TOKEN_OFFSET: u32 = CUSTOM_TOKEN_BASE + 10;

/// Generation stop token (matches Orpheus `stop_token_ids=[49158]`).
pub const STOP_TOKEN_ID: u32 = 49_158;

/// Built-in English voices (finetune-prod / unsloth GGUF).
pub const VOICES: &[&str] = &["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"];

static RE_CUSTOM: Lazy<Regex> = Lazy::new(|| Regex::new(r"<custom_token_(\d+)>").unwrap());

/// Map a generated `<custom_token_N>` string to a SNAC code index.
pub fn custom_token_str_to_code(token: &str, stream_index: usize) -> Option<i32> {
    let caps = RE_CUSTOM.captures(token.trim())?;
    let n: i32 = caps[1].parse().ok()?;
    let code = n - 10 - ((stream_index % 7) as i32 * 4096);
    if (0..4096).contains(&code) {
        Some(code)
    } else {
        None
    }
}

/// Map a GGUF token id to a SNAC code index when it is a custom speech token.
pub fn custom_token_id_to_code(token_id: u32, stream_index: usize) -> Option<i32> {
    if !is_custom_token_id(token_id) {
        return None;
    }
    let n = (token_id - CUSTOM_TOKEN_BASE) as i32;
    let code = n - 10 - ((stream_index % 7) as i32 * 4096);
    if (0..4096).contains(&code) {
        Some(code)
    } else {
        None
    }
}

pub fn is_custom_token_id(token_id: u32) -> bool {
    token_id >= CUSTOM_TOKEN_BASE
}

/// Inclusive-exclusive token id range for SNAC slot `stream_index % 7`.
pub fn snac_slot_token_range(stream_index: usize) -> (u32, u32) {
    let slot = stream_index % 7;
    let lo = SNAC_TOKEN_OFFSET + slot as u32 * 4096;
    (lo, lo + 4096)
}

/// True when `token_id` is a custom speech token for the current stream slot.
pub fn is_snac_slot_token(token_id: u32, stream_index: usize) -> bool {
    if !is_custom_token_id(token_id) {
        return false;
    }
    let (lo, hi) = snac_slot_token_range(stream_index);
    token_id >= lo && token_id < hi
}

/// Process one LM token into a SNAC code (if any), advancing the stream index
/// per upstream `orpheus_tts/decoder.py` (`count` increments only when code > 0).
pub fn accept_orpheus_stream_token(tok: u32, stream_index: &mut usize) -> Option<i32> {
    if !is_snac_slot_token(tok, *stream_index) {
        return None;
    }
    let code = custom_token_id_to_code(tok, *stream_index)?;
    if code > 0 {
        *stream_index += 1;
        Some(code)
    } else {
        None
    }
}

/// Restrict LM logits to the SNAC codebook slice for `stream_index % 7` plus stop.
pub fn mask_logits_for_snac_slot(logits: &mut [f32], stream_index: usize) {
    let (lo, hi) = snac_slot_token_range(stream_index);
    for (i, v) in logits.iter_mut().enumerate() {
        let id = i as u32;
        if id != STOP_TOKEN_ID && (id < lo || id >= hi) {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Whether to hard-mask LM logits to the active SNAC slot during sampling.
pub fn use_snac_logit_mask() -> bool {
    match std::env::var("ORPHEUS_MASK_LOGITS").ok().as_deref() {
        Some("0") | Some("false") | Some("FALSE") => false,
        Some("1") | Some("true") | Some("TRUE") => true,
        _ => false,
    }
}

/// Opt-in via `ORPHEUS_MASK_LOGITS=1`; upstream Orpheus/vLLM does NOT mask logits.
///
/// Hard-masking every step to the active SNAC slot forces the LM out of its
/// trained joint distribution and yields non-speech audio on accurate backends
/// (CPU/Metal/CUDA/ROCm). It defaults on only for wgpu/Vulkan **native GPU
/// decode**, whose kernels otherwise occasionally emit out-of-slot ids.
pub fn use_snac_logit_mask_for(lm_device: rlx_runtime::Device, lm_decode_on_cpu: bool) -> bool {
    if use_snac_logit_mask() {
        return true;
    }
    if matches!(
        std::env::var("ORPHEUS_MASK_LOGITS").ok().as_deref(),
        Some("0") | Some("false") | Some("FALSE")
    ) {
        return false;
    }
    std::env::var("ORPHEUS_MASK_LOGITS")
        .ok()
        .as_deref()
        .is_none()
        && matches!(
            lm_device,
            rlx_runtime::Device::Gpu | rlx_runtime::Device::Vulkan
        )
        && !lm_decode_on_cpu
}

/// Llama-3 `<|begin_of_text|>` — included in HF `tokenizer()` output after
/// [`PROMPT_START_ID`] (see canopyai Orpheus `_format_prompt`).
pub const BOS_TOKEN_ID: u32 = 128_000;

/// Prompt control tokens appended after the user text (Orpheus medium-3b format).
pub const PROMPT_START_ID: u32 = 128_259; // `<custom_token_3>`
pub const PROMPT_END_IDS: [u32; 4] = [128_009, 128_260, 128_261, 128_257];

/// Build `{voice}: {text}` and return token ids using GGUF embedded vocab.
#[cfg(feature = "llama")]
pub fn build_prompt(
    weights: &std::path::Path,
    text: &str,
    voice: Option<&str>,
) -> anyhow::Result<Vec<u32>> {
    let body = match voice {
        Some(v) => format!("{v}: {text}"),
        None => text.to_string(),
    };
    build_prompt_ids(weights, &body)
}

/// Tokenize `body`, prepend start token and append end control tokens.
#[cfg(feature = "llama")]
pub fn build_prompt_ids(weights: &std::path::Path, body: &str) -> anyhow::Result<Vec<u32>> {
    use anyhow::Context;
    use rlx_qwen35::encode_prompt_from_gguf;

    let mut ids = encode_prompt_from_gguf(weights, body)
        .with_context(|| format!("tokenize prompt for {}", weights.display()))?;
    let mut out = Vec::with_capacity(ids.len() + 2 + PROMPT_END_IDS.len());
    out.push(PROMPT_START_ID);
    // Match HF `AutoTokenizer` + Orpheus `_format_prompt`: BOS after start token.
    if ids.first().copied() != Some(BOS_TOKEN_ID) {
        out.push(BOS_TOKEN_ID);
    }
    out.append(&mut ids);
    out.extend_from_slice(&PROMPT_END_IDS);
    Ok(out)
}

/// Map one interleaved Orpheus SNAC frame (7 codebook indices, each 0..4095)
/// to the seven GGUF token ids the LM expects.
pub fn frame_codes_to_token_ids(frame: &[i32]) -> Option<[u32; 7]> {
    if frame.len() != 7 {
        return None;
    }
    let valid = |c: i32| (0..4096).contains(&c);
    if !frame.iter().all(|&c| valid(c)) {
        return None;
    }
    let mut out = [0u32; 7];
    for (slot, &code) in frame.iter().enumerate() {
        out[slot] = SNAC_TOKEN_OFFSET + code as u32 + (slot as u32 * 4096);
    }
    Some(out)
}

/// Map a flat interleaved code stream (multiple of 7) to GGUF audio token ids.
pub fn orpheus_frame_codes_to_token_ids(codes: &[i32]) -> Option<Vec<u32>> {
    if codes.len() < 7 || !codes.len().is_multiple_of(7) {
        return None;
    }
    let mut out = Vec::with_capacity(codes.len());
    for chunk in codes.chunks_exact(7) {
        out.extend_from_slice(&frame_codes_to_token_ids(chunk)?);
    }
    Some(out)
}

/// Build a zero-shot clone prompt: `{ref_text} + ref audio tokens + {target}`.
///
/// Uses the **pretrained** Orpheus layout (`Text1 [AudioTokens] Text2`). The
/// finetune-prod checkpoint (`orpheus-3b-0.1-ft`) is trained on named voices
/// (`tara: …`) and may ignore reference audio — prefer
/// `canopylabs/orpheus-3b-0.1-pretrained` (GGUF) for cloning.
#[cfg(feature = "llama")]
pub fn build_voice_clone_prompt_ids(
    weights: &std::path::Path,
    ref_text: &str,
    ref_audio_token_ids: &[u32],
    target_text: &str,
) -> anyhow::Result<Vec<u32>> {
    if ref_audio_token_ids.is_empty() {
        anyhow::bail!("voice clone reference must include at least one audio token");
    }

    use rlx_qwen35::encode_prompt_from_gguf;

    let mut ref_ids = encode_prompt_from_gguf(weights, ref_text)
        .with_context(|| format!("tokenize reference text for {}", weights.display()))?;
    ref_ids.extend_from_slice(ref_audio_token_ids);

    let mut target_ids = encode_prompt_from_gguf(weights, target_text)
        .with_context(|| format!("tokenize target text for {}", weights.display()))?;
    ref_ids.append(&mut target_ids);

    let mut out = Vec::with_capacity(1 + ref_ids.len() + PROMPT_END_IDS.len());
    out.push(PROMPT_START_ID);
    if ref_ids.first().copied() != Some(BOS_TOKEN_ID) {
        out.push(BOS_TOKEN_ID);
    }
    out.append(&mut ref_ids);
    out.extend_from_slice(&PROMPT_END_IDS);
    Ok(out)
}

/// JSON bundle written by [`scripts/orpheus_encode_reference.py`].
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct VoiceCloneReference {
    pub transcript: String,
    pub token_ids: Vec<u32>,
    #[serde(default)]
    pub frame_codes: Vec<i32>,
}

impl VoiceCloneReference {
    pub fn load_json(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read voice clone reference {}", path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("parse voice clone reference {}", path.display()))
    }
}

/// Pack interleaved Orpheus frame tokens (7 per frame) into SNAC codebooks.
///
/// Layout matches `orpheus_tts/decoder.py::convert_to_audio`.
pub fn pack_orpheus_codes(frame_tokens: &[i32]) -> Option<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    if frame_tokens.len() < 7 {
        return None;
    }
    let num_frames = frame_tokens.len() / 7;
    let frame = &frame_tokens[..num_frames * 7];

    let mut codes_0 = Vec::with_capacity(num_frames);
    let mut codes_1 = Vec::with_capacity(num_frames * 2);
    let mut codes_2 = Vec::with_capacity(num_frames * 4);

    for j in 0..num_frames {
        let i = 7 * j;
        codes_0.push(frame[i]);
        codes_1.push(frame[i + 1]);
        codes_1.push(frame[i + 4]);
        codes_2.push(frame[i + 2]);
        codes_2.push(frame[i + 3]);
        codes_2.push(frame[i + 5]);
        codes_2.push(frame[i + 6]);
    }

    let valid = |c: i32| (0..4096).contains(&c);
    if !codes_0.iter().all(|&c| valid(c))
        || !codes_1.iter().all(|&c| valid(c))
        || !codes_2.iter().all(|&c| valid(c))
    {
        return None;
    }

    Some((codes_0, codes_1, codes_2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_index_advances_only_on_positive_code() {
        let mut ix = 0usize;
        // Slot 0, code 0 → index stays 0 (matches Python `count`).
        let tok0 = SNAC_TOKEN_OFFSET; // custom_token_10 → code 0
        assert_eq!(accept_orpheus_stream_token(tok0, &mut ix), None);
        assert_eq!(ix, 0);
        // Slot 0, code 1 → accepted, index → 1.
        let tok1 = SNAC_TOKEN_OFFSET + 1;
        assert_eq!(accept_orpheus_stream_token(tok1, &mut ix), Some(1));
        assert_eq!(ix, 1);
    }

    #[test]
    fn snac_logit_mask_off_by_default_on_accurate_backends() {
        let _guard = EnvGuard::unset("ORPHEUS_MASK_LOGITS");
        // Accurate backends run unmasked (upstream parity) — masking them yields
        // non-speech audio.
        assert!(!use_snac_logit_mask_for(rlx_runtime::Device::Cpu, true));
        assert!(!use_snac_logit_mask_for(rlx_runtime::Device::Metal, true));
        assert!(!use_snac_logit_mask_for(rlx_runtime::Device::Metal, false));
        assert!(!use_snac_logit_mask_for(rlx_runtime::Device::Cuda, false));
        assert!(!use_snac_logit_mask_for(rlx_runtime::Device::Rocm, false));
        // wgpu/Vulkan native GPU decode (decode not delegated to CPU) masks.
        assert!(use_snac_logit_mask_for(rlx_runtime::Device::Gpu, false));
        assert!(use_snac_logit_mask_for(rlx_runtime::Device::Vulkan, false));
        assert!(!use_snac_logit_mask_for(rlx_runtime::Device::Gpu, true));
    }

    /// Restore env after tests that set ORPHEUS_* vars.
    struct EnvGuard {
        prev: Option<String>,
    }

    impl EnvGuard {
        fn unset(_key: &'static str) -> Self {
            let prev = std::env::var("ORPHEUS_MASK_LOGITS").ok();
            unsafe { std::env::remove_var("ORPHEUS_MASK_LOGITS") };
            Self { prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { std::env::set_var("ORPHEUS_MASK_LOGITS", v) },
                None => unsafe { std::env::remove_var("ORPHEUS_MASK_LOGITS") },
            }
        }
    }

    #[test]
    fn frame_codes_roundtrip_token_ids() {
        let frame = [1, 20, 30, 40, 50, 60, 70];
        let ids = frame_codes_to_token_ids(&frame).unwrap();
        assert_eq!(ids.len(), 7);
        for (slot, (&code, &id)) in frame.iter().zip(ids.iter()).enumerate() {
            assert_eq!(custom_token_id_to_code(id, slot), Some(code), "slot {slot}");
        }
    }

    #[test]
    fn custom_token_mapping_matches_python() {
        assert_eq!(custom_token_str_to_code("<custom_token_10>", 0), Some(0));
        assert_eq!(custom_token_str_to_code("<custom_token_11>", 0), Some(1));
        assert_eq!(custom_token_str_to_code("<custom_token_4107>", 1), Some(1));
    }

    #[test]
    fn pack_orpheus_codes_layout() {
        let frame = [10, 20, 30, 40, 50, 60, 70];
        let (c0, c1, c2) = pack_orpheus_codes(&frame).unwrap();
        assert_eq!(c0, vec![10]);
        assert_eq!(c1, vec![20, 50]);
        assert_eq!(c2, vec![30, 40, 60, 70]);
    }

    /// Env-gated: `ORPHEUS_GGUF_PATH=/path/to/orpheus.gguf`
    #[test]
    fn prompt_ids_for_tara_hi() {
        let Some(path) = std::env::var("ORPHEUS_GGUF_PATH")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.is_file())
            .or_else(|| {
                let p = std::path::PathBuf::from(
                    "/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf",
                );
                p.is_file().then_some(p)
            })
        else {
            eprintln!("skip: set ORPHEUS_GGUF_PATH or run `just fetch-orpheus`");
            return;
        };

        let ids = build_prompt(&path, "Hi.", Some("tara")).expect("build prompt");
        assert_eq!(
            ids,
            vec![
                128_259, 128_000, 83, 5169, 25, 21694, 13, 128_009, 128_260, 128_261, 128_257
            ],
            "Orpheus medium-3b prompt layout for `tara: Hi.`"
        );
    }
}
