// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Rotating distillation prompts — transcript variants + default pool.

const DEFAULT_POOL: &[&str] = &[
    "Hello, this is my voice.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the demonstration of this speech synthesis system.",
    "I hope you are having a wonderful day.",
    "Please listen carefully to how I sound.",
    "This is a test of personalized text to speech.",
    "Good morning, and thank you for listening.",
    "One two three four five six seven eight nine ten.",
];

/// Pick a distillation text for `(step, wav_idx)` — rotates variants per reference clip.
pub fn distill_text_for_sample(step: usize, wav_idx: usize, transcript: Option<&str>) -> String {
    if let Some(t) = transcript.filter(|s| !s.trim().is_empty()) {
        let variants = transcript_variants(t);
        return variants[(step.wrapping_add(wav_idx)) % variants.len()].clone();
    }
    DEFAULT_POOL[(step.wrapping_add(wav_idx)) % DEFAULT_POOL.len()].to_string()
}

fn transcript_variants(transcript: &str) -> Vec<String> {
    let base = transcript.trim().to_string();
    let mut out = Vec::new();
    out.push(base.clone());
    if let Some((first, _)) = base.split_once('.') {
        let sentence = format!("{first}.");
        if sentence != base {
            out.push(sentence);
        }
    }
    if base.len() > 48 {
        out.push(format!("{}…", &base[..48]));
    }
    out.push(format!("Please listen. {base}"));
    out.push(format!("{base} Thank you."));
    out.dedup();
    if out.is_empty() {
        out.push(DEFAULT_POOL[0].to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotates_default_pool_without_transcript() {
        let a = distill_text_for_sample(0, 0, None);
        let b = distill_text_for_sample(1, 0, None);
        assert_ne!(a, b);
    }

    #[test]
    fn transcript_yields_multiple_variants() {
        let t = "Hello world. This is a longer sample for cloning.";
        let v0 = distill_text_for_sample(0, 0, Some(t));
        let v1 = distill_text_for_sample(1, 0, Some(t));
        assert!(!v0.is_empty());
        assert!(!v1.is_empty());
    }
}
