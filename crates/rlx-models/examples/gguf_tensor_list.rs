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

//! List every tensor + shape + dtype in a GGUF file. Used during
//! qwen35 weight-loader development to enumerate the tensor inventory
//! of a hybrid-arch file. Usage:
//!
//! ```text
//! cargo run --release -p rlx-models --example gguf_tensor_list -- <file.gguf>
//! ```

use rlx_gguf::GgufFile;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: gguf_tensor_list <path>"))?;
    let raw = GgufFile::from_path(&path)?;
    let mut keys: Vec<&String> = raw.tensors.keys().collect();
    keys.sort();
    for k in keys {
        let t = &raw.tensors[k];
        println!("{:60} {:?} {:?}", k, t.shape, t.dtype);
    }
    Ok(())
}
