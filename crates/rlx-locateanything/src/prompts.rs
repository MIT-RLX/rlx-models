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

//! Task prompt templates from the LocateAnything HF card.

/// Object detection / document layout — categories joined with `</c>`.
pub fn detect(categories: &[&str]) -> String {
    let cats = categories.join("</c>");
    format!("Locate all the instances that matches the following description: {cats}.")
}

/// Phrase grounding — single instance.
pub fn ground_single(phrase: &str) -> String {
    format!("Locate a single instance that matches the following description: {phrase}.")
}

/// Phrase grounding — multiple instances.
pub fn ground_multi(phrase: &str) -> String {
    format!("Locate all the instances that match the following description: {phrase}.")
}

/// Text grounding.
pub fn ground_text(phrase: &str) -> String {
    format!("Please locate the text referred as {phrase}.")
}

/// Scene text detection.
pub fn detect_text() -> String {
    "Detect all the text in box format.".into()
}

/// GUI grounding (box).
pub fn ground_gui_box(phrase: &str) -> String {
    format!("Locate the region that matches the following description: {phrase}.")
}

/// GUI grounding (point) / pointing.
pub fn point(phrase: &str) -> String {
    format!("Point to: {phrase}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_joins_categories() {
        let p = detect(&["person", "car"]);
        assert!(p.contains("person</c>car"));
    }
}
