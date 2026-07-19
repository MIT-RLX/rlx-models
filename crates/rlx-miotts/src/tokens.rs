//! MioTTS speech-token layout (`<|s_N|>`) + chat prompt helpers.

/// `<|s_0|>` LM vocab id (`added_tokens.json`).
pub const SPEECH_BASE: u32 = 151_669;
/// MioCodec content codebook size.
pub const SPEECH_CODEBOOK: u32 = 12_800;
/// `<|im_end|>` / generation eos.
pub const EOS: u32 = 151_645;
/// Fixed ONNX decode length (pad/truncate).
pub const SPEECH_LEN: usize = 100;

/// Map a content code `0..12800` to its LM vocab id.
#[inline]
pub fn speech_id(code: u32) -> u32 {
    SPEECH_BASE + code
}

/// Extract content codes from generated LM ids (stop at EOS).
pub fn parse_speech_codes(generated: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for &id in generated {
        if id == EOS {
            break;
        }
        if (SPEECH_BASE..SPEECH_BASE + SPEECH_CODEBOOK).contains(&id) {
            out.push(id - SPEECH_BASE);
        }
    }
    out
}

/// Parse `<|s_N|>` markers from decoded LM text (HF chat path).
pub fn parse_speech_tokens_text(text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 5 < bytes.len() {
        if &bytes[i..i + 4] == b"<|s_" {
            let start = i + 4;
            let mut j = start;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > start && j + 1 < bytes.len() && bytes[j] == b'|' && bytes[j + 1] == b'>' {
                if let Some(n) = std::str::from_utf8(&bytes[start..j])
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    out.push(n);
                }
                i = j + 2;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Pad with 0 / truncate to [`SPEECH_LEN`] for the fixed ONNX body.
pub fn fit_speech_len(codes: &[u32]) -> Vec<u32> {
    let mut v = codes.to_vec();
    v.resize(SPEECH_LEN, 0);
    v.truncate(SPEECH_LEN);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_text_and_ids() {
        let t = "<|s_5051|><|s_11221|><|im_end|>";
        assert_eq!(parse_speech_tokens_text(t), vec![5051, 11221]);
        let ids = [speech_id(5051), speech_id(11221), EOS, speech_id(1)];
        assert_eq!(parse_speech_codes(&ids), vec![5051, 11221]);
    }

    #[test]
    fn fit_pads() {
        assert_eq!(fit_speech_len(&[1, 2]).len(), SPEECH_LEN);
        assert_eq!(
            fit_speech_len(&(0..200).collect::<Vec<_>>()).len(),
            SPEECH_LEN
        );
    }
}
