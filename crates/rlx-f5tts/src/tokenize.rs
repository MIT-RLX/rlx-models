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

//! F5-TTS text tokenizer. F5 concatenates `ref_text + gen_text` and maps each
//! character to a `vocab.txt` id (`convert_char_to_pinyin`). For English this is
//! plain character mapping (unknown → 0); Chinese requires pinyin conversion
//! (`jieba`/`pypinyin`), which is not yet ported here.

use crate::config::Vocab;

/// Match official `preprocess_ref_audio_text`: ref must end with `". "` or `"。"`.
pub fn normalize_ref_text(ref_text: &str) -> String {
    let t = ref_text.trim_end();
    if t.ends_with(". ") || t.ends_with('。') {
        t.to_string()
    } else if t.ends_with('.') {
        format!("{t} ")
    } else {
        format!("{t}. ")
    }
}

/// Encode `ref_text + gen_text` to token ids (English char-level).
/// `ref_text` is normalized first (trailing `". "`).
pub fn encode(ref_text: &str, gen_text: &str, vocab: &Vocab) -> Vec<i32> {
    let ref_text = normalize_ref_text(ref_text);
    let combined = format!("{ref_text}{gen_text}");
    combined
        .chars()
        .map(|c| vocab.id_of(&c.to_string()))
        .collect()
}

/// Byte length + Chinese-punctuation weighting used by F5's duration estimate.
pub fn text_len(text: &str) -> usize {
    const ZH_PUNC: [char; 7] = ['。', '，', '、', '；', '：', '？', '！'];
    text.len() + 3 * text.chars().filter(|c| ZH_PUNC.contains(c)).count()
}
