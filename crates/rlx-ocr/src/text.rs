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

//! Recognized text items with character-level bounding boxes.

use rten_imageproc::{BoundingRect, Rect, RotatedRect};

/// A single recognized character with its axis-aligned bounding box.
#[derive(Clone, Debug, PartialEq)]
pub struct TextChar {
    pub char: char,
    pub rect: Rect,
}

/// A word composed of one or more characters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextWord {
    pub chars: Vec<TextChar>,
}

impl TextWord {
    pub fn text(&self) -> String {
        self.chars.iter().map(|c| c.char).collect()
    }
}

/// A line of text composed of words.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextLine {
    pub words: Vec<TextWord>,
}

impl TextLine {
    pub fn new(chars: Vec<TextChar>) -> Self {
        Self {
            words: vec![TextWord { chars }],
        }
    }

    pub fn text(&self) -> String {
        self.words
            .iter()
            .map(TextWord::text)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Any recognized text item (line, word, or character).
#[derive(Clone, Debug, PartialEq)]
pub enum TextItem {
    Line(TextLine),
    Word(TextWord),
    Char(TextChar),
}

impl TextItem {
    pub fn as_line(&self) -> Option<&TextLine> {
        match self {
            Self::Line(l) => Some(l),
            _ => None,
        }
    }
}

impl std::fmt::Display for TextLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text())
    }
}

/// Convenience: bounding box union of word rects in a line.
pub fn line_bounding_rect(words: &[RotatedRect]) -> Option<Rect> {
    words.iter().fold(None, |br: Option<Rect>, r| match br {
        Some(br) => Some(br.union(r.bounding_rect().integral_bounding_rect())),
        None => Some(r.bounding_rect().integral_bounding_rect()),
    })
}
