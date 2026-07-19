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

//! Microbench: one host sparse-structure DiT forward (real 1.3B weights).

use rlx_trellis2::config::{DitConfig, DitKind};
use rlx_trellis2::dit_host::dit_forward;
use rlx_trellis2::rope::grid_coords;
use std::env;
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let ckpt = env::args()
        .nth(1)
        .expect("usage: dit_host_bench <ss_flow.safetensors> [json]");
    let json = env::args().nth(2).unwrap_or_else(|| {
        Path::new(&ckpt)
            .with_extension("json")
            .display()
            .to_string()
    });
    let cfg = DitConfig::from_file(&json)?;
    assert_eq!(cfg.kind, DitKind::SparseStructureFlow);
    let wm = rlx_core::load_weight_map(&ckpt, &[])?;
    let res = cfg.args.resolution;
    let n_pos = res * res * res;
    let in_ch = cfg.args.in_channels;
    let n_cond = 16usize;
    let tokens = vec![0.01f32; n_pos * in_ch];
    let coords = grid_coords(res);
    let cond = vec![0.0f32; n_cond * cfg.args.cond_channels];
    eprintln!(
        "dit_host_bench: n_pos={n_pos} C={} blocks={} heads={}",
        cfg.args.model_channels,
        cfg.args.num_blocks,
        cfg.num_heads()
    );
    let t0 = Instant::now();
    let out = dit_forward(
        &cfg, &wm, &tokens, &coords, n_pos, &cond, n_cond, 500.0, None,
    )?;
    eprintln!(
        "dit_host_bench: forward {:.2}s  out_sum={:.4}",
        t0.elapsed().as_secs_f64(),
        out.iter().sum::<f32>()
    );
    Ok(())
}
