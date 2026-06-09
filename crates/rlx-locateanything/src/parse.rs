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

//! Parse structured grounding output from model text.

use std::collections::HashSet;

/// Pixel-coordinate bounding box parsed from model output.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Pixel-coordinate point parsed from model output.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedPoint {
    pub x: f32,
    pub y: f32,
}

/// Coordinates in model output are integers in `[0, 1000]` (normalized).
const COORD_SCALE: f32 = 1000.0;

/// Parse `<box><x1><y1><x2><y2></box>` spans into pixel boxes.
pub fn parse_boxes(answer: &str, image_width: u32, image_height: u32) -> Vec<ParsedBox> {
    let w = image_width as f32;
    let h = image_height as f32;
    let mut out = Vec::new();
    let mut rest = answer;
    while let Some(start) = rest.find("<box><") {
        let after = &rest[start + 6..];
        let Some(end) = after.find("></box>") else {
            break;
        };
        let inner = &after[..end];
        let nums: Vec<u32> = inner
            .split("><")
            .filter_map(|s| s.trim_matches(|c| c == '<' || c == '>').parse().ok())
            .collect();
        if nums.len() == 4 {
            let (x1, y1, x2, y2) = (nums[0], nums[1], nums[2], nums[3]);
            out.push(ParsedBox {
                x1: x1 as f32 / COORD_SCALE * w,
                y1: y1 as f32 / COORD_SCALE * h,
                x2: x2 as f32 / COORD_SCALE * w,
                y2: y2 as f32 / COORD_SCALE * h,
            });
        }
        rest = &after[end + 7..];
    }
    out
}

/// Parse two-coordinate `<box><x><y></box>` spans into pixel points.
pub fn parse_points(answer: &str, image_width: u32, image_height: u32) -> Vec<ParsedPoint> {
    let w = image_width as f32;
    let h = image_height as f32;
    let mut out = Vec::new();
    let mut rest = answer;
    while let Some(start) = rest.find("<box><") {
        let after = &rest[start + 6..];
        let Some(end) = after.find("></box>") else {
            break;
        };
        let inner = &after[..end];
        let nums: Vec<u32> = inner
            .split("><")
            .filter_map(|s| s.trim_matches(|c| c == '<' || c == '>').parse().ok())
            .collect();
        if nums.len() == 2 {
            out.push(ParsedPoint {
                x: nums[0] as f32 / COORD_SCALE * w,
                y: nums[1] as f32 / COORD_SCALE * h,
            });
        }
        rest = &after[end + 7..];
    }
    out
}

/// Labels from `<ref>…</ref>` spans (HF / processor decode).
pub fn parse_refs(answer: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = answer;
    while let Some(start) = rest.find("<ref>") {
        let after = &rest[start + 5..];
        let Some(end) = after.find("</ref>") else {
            break;
        };
        out.push(after[..end].to_string());
        rest = &after[end + 6..];
    }
    out
}

/// Collect normalized `[0, 1000]` integers from angle-bracket spans.
fn bracket_coord_nums(s: &str) -> Vec<u32> {
    let mut nums = Vec::new();
    let mut rest = s;
    while let Some(start) = rest.find('<') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('>') else {
            break;
        };
        if let Ok(n) = after[..end].trim().parse::<u32>() {
            if n <= 1000 {
                nums.push(n);
            }
        }
        rest = &after[end + 1..];
    }
    nums
}

/// Parse `<box>…</box>` spans with any number of bracket coords (groups of 4 → boxes).
pub fn parse_ref_boxes(answer: &str, image_width: u32, image_height: u32) -> Vec<ParsedBox> {
    let w = image_width as f32;
    let h = image_height as f32;
    let mut out = Vec::new();
    let mut rest = answer;
    while let Some(start) = rest.find("<box>") {
        let after = &rest[start + 5..];
        let end_tag = after.find("</box>").or_else(|| after.find("></box>"));
        let Some(end) = end_tag else {
            break;
        };
        let inner = &after[..end];
        let nums = bracket_coord_nums(inner);
        for chunk in nums.chunks(4).filter(|c| c.len() == 4) {
            let (x1, y1, x2, y2) = (chunk[0], chunk[1], chunk[2], chunk[3]);
            out.push(ParsedBox {
                x1: x1 as f32 / COORD_SCALE * w,
                y1: y1 as f32 / COORD_SCALE * h,
                x2: x2 as f32 / COORD_SCALE * w,
                y2: y2 as f32 / COORD_SCALE * h,
            });
        }
        rest = &after[end..];
    }
    out
}

/// Full grounding parse (text + geometry).
#[derive(Debug, Clone, Default)]
pub struct GroundingParse {
    pub text: String,
    pub raw: String,
    pub refs: Vec<String>,
    pub boxes: Vec<ParsedBox>,
    pub points: Vec<ParsedPoint>,
    pub prompt_len: usize,
    pub new_tokens: usize,
}

/// Parse model answer into refs, boxes, and points (merges plain and ref-style boxes).
pub fn parse_grounding(answer: &str, image_width: u32, image_height: u32) -> GroundingParse {
    let refs = parse_refs(answer);
    let mut boxes: Vec<ParsedBox> = parse_boxes(answer, image_width, image_height);
    boxes.extend(parse_ref_boxes(answer, image_width, image_height));
    dedupe_boxes(&mut boxes);
    let points = parse_points(answer, image_width, image_height);
    GroundingParse {
        text: answer.to_string(),
        raw: String::new(),
        refs,
        boxes,
        points,
        ..Default::default()
    }
}

fn dedupe_boxes(boxes: &mut Vec<ParsedBox>) {
    let mut seen = HashSet::new();
    boxes.retain(|b| {
        let key = (
            (b.x1 * 10.0).round() as i32,
            (b.y1 * 10.0).round() as i32,
            (b.x2 * 10.0).round() as i32,
            (b.y2 * 10.0).round() as i32,
        );
        seen.insert(key)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_box_and_point() {
        let answer = "<box><100><200><300><400></box> <box><500><600></box>";
        let boxes = parse_boxes(answer, 1000, 800);
        assert_eq!(boxes.len(), 1);
        assert!((boxes[0].x1 - 100.0).abs() < 1e-3);
        let points = parse_points(answer, 1000, 800);
        assert_eq!(points.len(), 1);
        assert!((points[0].x - 500.0).abs() < 1e-3);
    }

    #[test]
    fn parse_ref_and_ref_boxes() {
        let answer = "<ref>bus</ref><box><100><200><300><400></box>";
        assert_eq!(parse_refs(answer), vec!["bus"]);
        let g = parse_grounding(answer, 1000, 800);
        assert_eq!(g.refs, vec!["bus"]);
        assert_eq!(g.boxes.len(), 1);
    }
}
