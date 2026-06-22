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

//! SRT / VTT / TSV export for [`crate::transcript::WhisperTranscript`].

use crate::transcript::WhisperTranscript;
use anyhow::Result;

fn fmt_srt_time(sec: f32) -> String {
    let ms = (sec * 1000.0).round() as u64;
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1000;
    let f = ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{f:03}")
}

fn fmt_vtt_time(sec: f32) -> String {
    let ms = (sec * 1000.0).round() as u64;
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1000;
    let f = ms % 1000;
    format!("{h:02}:{m:02}:{s:02}.{f:03}")
}

pub fn to_srt(t: &WhisperTranscript) -> String {
    let mut out = String::new();
    for (i, seg) in t.segments.iter().enumerate() {
        if seg.text.trim().is_empty() {
            continue;
        }
        let label = seg
            .speaker
            .as_ref()
            .map(|s| format!("[{s}] "))
            .unwrap_or_default();
        out.push_str(&(i + 1).to_string());
        out.push('\n');
        out.push_str(&fmt_srt_time(seg.start));
        out.push_str(" --> ");
        out.push_str(&fmt_srt_time(seg.end));
        out.push('\n');
        out.push_str(&label);
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}

pub fn to_vtt(t: &WhisperTranscript) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in &t.segments {
        if seg.text.trim().is_empty() {
            continue;
        }
        out.push_str(&fmt_vtt_time(seg.start));
        out.push_str(" --> ");
        out.push_str(&fmt_vtt_time(seg.end));
        out.push('\n');
        if let Some(spk) = &seg.speaker {
            out.push_str(&format!("<v {spk}>"));
        }
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}

pub fn to_tsv(t: &WhisperTranscript) -> String {
    let mut out = String::from("start\tend\ttext\tspeaker\n");
    for seg in &t.segments {
        let spk = seg.speaker.as_deref().unwrap_or("");
        out.push_str(&format!(
            "{:.3}\t{:.3}\t{}\t{spk}\n",
            seg.start,
            seg.end,
            seg.text.replace('\t', " ").trim()
        ));
    }
    out
}

pub fn to_json_pretty(t: &WhisperTranscript) -> Result<String> {
    Ok(serde_json::to_string_pretty(t)?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    Srt,
    Vtt,
    Tsv,
    Json,
}

impl SubtitleFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "srt" => Some(Self::Srt),
            "vtt" | "webvtt" => Some(Self::Vtt),
            "tsv" => Some(Self::Tsv),
            "json" => Some(Self::Json),
            _ => None,
        }
    }

    pub fn render(self, t: &WhisperTranscript) -> Result<String> {
        match self {
            Self::Srt => Ok(to_srt(t)),
            Self::Vtt => Ok(to_vtt(t)),
            Self::Tsv => Ok(to_tsv(t)),
            Self::Json => to_json_pretty(t),
        }
    }
}
