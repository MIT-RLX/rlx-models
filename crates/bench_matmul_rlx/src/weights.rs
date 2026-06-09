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

use std::collections::HashMap;
use std::path::Path;

/// Load decomposed weights from `dir` (`model.safetensors`).
pub fn load_weights(dir: &Path) -> anyhow::Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let path = dir.join("model.safetensors");
    let bytes = std::fs::read(&path)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut out = HashMap::new();
    for name in st.names() {
        let view = st.tensor(name)?;
        let shape: Vec<usize> = view.shape().iter().copied().collect();
        let mut data = Vec::with_capacity(view.data().len() / 4);
        for chunk in view.data().chunks_exact(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out.insert(name.to_string(), (data, shape));
    }
    Ok(out)
}

/// ONNX initializer names in this model (2 tensors).
pub const PARAM_NAMES: &[&str] = &["\"b\"", "\"w\""];
