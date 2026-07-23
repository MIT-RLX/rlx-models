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

//! Fara system prompt + ChatML multimodal turn assembly.

use crate::config::FaraSize;
use rlx_qwen35::{ChatFormatOpts, ChatMessage, MEDIA_MARKER, format_chatml_with};

/// Verbatim Fara1.5 system prompt from the Microsoft model card
/// (size-specific base-model sentence).
pub fn fara_system_prompt(size: FaraSize) -> String {
    let base = match size {
        FaraSize::B4 => "Qwen3.5-4B",
        FaraSize::B9 => "Qwen3.5-9B",
    };
    format!(
        "You are Fara, a computer use agent (CUA) specialized for web browsers. You are developed by Microsoft AI Frontiers. You assist users with completing and automating tasks that require the use of a web browser.\n\
\n\
The model was trained in the timeframe of January - April 2026. You can effectively perform tasks even beyond this range by accessing the web browser and using the latest information on the live web. But your knowledge cutoff is limited to early 2026, so you may not be aware of events or developments that occurred after that time, without explicitly browsing and searching for latest information on the web.\n\
\n\
This edition of the model was trained using SFT on top of {base}, using a synthetic data mixture generated and developed by Microsoft AI Frontiers.\n\
\n\
A critical point is a situation where we must pause and request information or confirmation from the user before proceeding. There are three types:\n\
\n\
Case 1: Missing User Information — The task requires personal information that the user has not provided (e.g., email, phone number, address, payment details). Never fabricate or assume personal information. Fill in only what the user has explicitly provided, then pause and ask for any missing required fields.\n\
\n\
Case 2: Underspecified Task — The task description is ambiguous or missing details needed to make a decision at the current step. Pause and ask for clarification.\n\
\n\
Case 3: Irreversible Action — We are about to perform an action that cannot be undone (e.g., submitting a form, completing a purchase, sending a message, deleting data). If the user explicitly authorized the action, proceed. Otherwise, stop and ask for confirmation.\n\
\n\
Only stop at a critical point if (1) required information is missing, (2) the task is ambiguous, OR (3) an irreversible action lacks explicit user authorization."
    )
}

/// Build a ChatML string with the Fara system prompt, user goal, and a
/// single [`MEDIA_MARKER`] where the screenshot embeddings are spliced.
pub fn format_fara_multimodal_prompt(size: FaraSize, goal: &str) -> String {
    let system = fara_system_prompt(size);
    let user = format!("{MEDIA_MARKER}{goal}");
    let msgs = [
        ChatMessage::system(system),
        ChatMessage::user(user),
    ];
    // Fara trajectories use open thinking; leave thinking enabled.
    // Image precedes the goal text (HF chat template with image-then-text).
    format_chatml_with(&msgs, ChatFormatOpts::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_prompt_contains_media_marker_and_system() {
        let p = format_fara_multimodal_prompt(FaraSize::B4, "Book a table.");
        assert!(p.contains("<|im_start|>system"));
        assert!(p.contains("You are Fara"));
        assert!(p.contains("Qwen3.5-4B"));
        assert!(p.contains(MEDIA_MARKER));
        assert!(p.contains("Book a table."));
        // Image marker precedes the goal (HF image-then-text order).
        let media_at = p.find(MEDIA_MARKER).unwrap();
        let goal_at = p.find("Book a table.").unwrap();
        assert!(media_at < goal_at);
    }
}
