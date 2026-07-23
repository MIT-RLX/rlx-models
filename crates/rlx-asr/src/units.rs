// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! SentencePiece `units.txt` loader.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct Units {
    pub id_to_piece: Vec<String>,
    pub piece_to_id: std::collections::HashMap<String, u32>,
}

impl Units {
    pub fn from_pieces(pieces: Vec<String>) -> Self {
        let mut piece_to_id = std::collections::HashMap::new();
        for (i, p) in pieces.iter().enumerate() {
            piece_to_id.insert(p.clone(), i as u32);
        }
        Self {
            id_to_piece: pieces,
            piece_to_id,
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut id_to_piece = Vec::new();
        for line in BufReader::new(f).lines() {
            let line = line?;
            let piece = line.split_whitespace().next().unwrap_or("").to_string();
            id_to_piece.push(piece);
        }
        Ok(Self::from_pieces(id_to_piece))
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            if let Some(p) = self.id_to_piece.get(id as usize) {
                if let Some(rest) = p.strip_prefix('\u{2581}') {
                    // SentencePiece word boundary
                    if !s.is_empty() {
                        s.push(' ');
                    }
                    s.push_str(rest);
                } else {
                    s.push_str(p);
                }
            }
        }
        s
    }

    pub fn seg_id(&self) -> Option<u32> {
        self.piece_to_id
            .get("▁<segE>")
            .or_else(|| self.piece_to_id.get("<segE>"))
            .copied()
    }
}
