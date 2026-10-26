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

//! Stage-by-stage parity against `scripts/birefnet_reference.py` dumps.
//!
//! `cargo run -p rlx-birefnet --release --example parity -- MODEL_DIR REF_DIR [device]`

use rlx_birefnet::{Weights, graph};
use rlx_runtime::Device;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

fn f32s(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (model, refd) = (Path::new(&a[0]), Path::new(&a[1]));
    let device = Device::from_str(a.get(2).map(String::as_str).unwrap_or("cpu"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let meta: std::collections::HashMap<String, serde_like::V> =
        serde_like::parse(&std::fs::read_to_string(refd.join("meta.json"))?);
    let side = meta["size"].num() as usize;
    let taps = graph::TAPS;
    let w = Arc::new(Weights::load(model)?);
    let built = graph::build(w, side, taps)?;
    let mut g = rlx_core::flow_util::compile_built(built, device)?;
    let x = f32s(&refd.join("input.bin"));
    let t0 = std::time::Instant::now();
    let out = g.run(&[("image", &x)]).remove(0);
    eprintln!("forward {:.2}s", t0.elapsed().as_secs_f32());
    let mut off = 0;
    for t in taps {
        let want = f32s(&refd.join(format!("{t}.bin")));
        let got = &out[off..off + want.len()];
        off += want.len();
        let (mut d2, mut w2, mut dm) = (0f64, 0f64, 0f64);
        for (&g, &w) in got.iter().zip(&want) {
            let d = (g - w) as f64;
            d2 += d * d;
            w2 += (w as f64).powi(2);
            dm = dm.max(d.abs());
        }
        println!(
            "{t:<7} rel {:.3e}  max|Δ| {:.3e}  ({} values)",
            (d2 / w2).sqrt(),
            dm,
            want.len()
        );
    }
    Ok(())
}

/// Tiny JSON number reader for `meta.json` (avoids a serde dependency here).
mod serde_like {
    use std::collections::HashMap;
    pub struct V(f64);
    impl V {
        pub fn num(&self) -> f64 {
            self.0
        }
    }
    pub fn parse(s: &str) -> HashMap<String, V> {
        let mut m = HashMap::new();
        for part in s.split(',') {
            if let Some((k, v)) = part.split_once(':') {
                let k = k
                    .trim()
                    .trim_matches(|c| c == '{' || c == '"' || c == '\n' || c == ' ')
                    .to_string();
                if let Ok(n) = v.trim().trim_end_matches('}').trim().parse::<f64>() {
                    m.insert(k, V(n));
                }
            }
        }
        m
    }
}
