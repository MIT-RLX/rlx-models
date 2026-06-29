//! Load Kyutai MLX Q4/Q8/bf16 safetensors → dequantized f32 weight map (native, candle-free).

use crate::checkpoint::MoshiCheckpoint;
use crate::mlx_dequant::{bf16_to_f32, dequantize_affine};
use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct MlxSafetensorsFile {
    data: Vec<u8>,
}

impl MlxSafetensorsFile {
    /// Read a Kyutai MLX safetensors file into memory.
    pub fn open(path: &Path) -> Result<Self> {
        let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Ok(Self { data })
    }

    /// List tensor names in the file (includes `.scales` / `.biases` sidecars).
    pub fn tensor_names(&self) -> Result<Vec<String>> {
        let st = SafeTensors::deserialize(&self.data)?;
        Ok(st.names().iter().map(|s| (*s).clone()).collect())
    }
}

/// Map MLX tensor base name → Kyutai Candle / rlx-moshi key.
pub fn mlx_to_candle_key(mlx_key: &str) -> Option<String> {
    if mlx_key.ends_with(".scales") || mlx_key.ends_with(".biases") {
        return None;
    }
    let mut key = mlx_key.to_string();
    key = key.replace("audio_embs.", "emb.");
    key = key.replace("depformer.slices.", "depformer.");
    key = key.replace(".norm1.weight", ".norm1.alpha");
    key = key.replace(".norm2.weight", ".norm2.alpha");
    key = key.replace(".self_attn.in_proj.weight", ".self_attn.in_proj_weight");
    key = key.replace("out_norm.weight", "out_norm.alpha");
    Some(key)
}

/// Load Kyutai MLX Q4/Q8/bf16 safetensors → dequantized f32 weight map, **natively**
/// (no candle). Quantized weights use the native [`dequantize_affine`]; this is the
/// candle-free counterpart of the old candle-`Tensor` round-trip and drives both the
/// eager `LmModel` and `RlxLm` MLX paths.
pub fn load_eager_weight_map(
    path: &Path,
    checkpoint: MoshiCheckpoint,
    _cfg: &crate::config::LmConfig,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let (bits, group_size) = checkpoint.mlx_quant_params();
    let file = MlxSafetensorsFile::open(path)?;
    let st = SafeTensors::deserialize(&file.data)?;
    let mut out: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    for name in st.names() {
        if name.ends_with(".scales") || name.ends_with(".biases") {
            continue;
        }
        let Some(key) = mlx_to_candle_key(name) else {
            continue;
        };
        let tensor = st.tensor(name)?;
        let shape: Vec<usize> = tensor.shape().to_vec();
        let (data, mut out_shape): (Vec<f32>, Vec<usize>) = if checkpoint.is_mlx_quantized()
            && name.ends_with(".weight")
            && tensor.dtype() == safetensors::Dtype::U32
        {
            let base = &name[..name.len() - ".weight".len()];
            let scales = st
                .tensor(&format!("{base}.scales"))
                .with_context(|| format!("missing {base}.scales"))?;
            let biases = st
                .tensor(&format!("{base}.biases"))
                .with_context(|| format!("missing {base}.biases"))?;
            let packed: Vec<u32> = tensor
                .data()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let scales_bf16: Vec<u16> = scales
                .data()
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let biases_bf16: Vec<u16> = biases
                .data()
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let rows = shape[0];
            let packed_cols = shape[1];
            let out_cols = packed_cols * (32 / bits as usize);
            let flat = dequantize_affine(
                &packed,
                packed_cols,
                &scales_bf16,
                &biases_bf16,
                rows,
                out_cols,
                group_size,
                bits,
            )?;
            (flat, vec![rows, out_cols])
        } else if tensor.dtype() == safetensors::Dtype::BF16 {
            let flat: Vec<f32> = tensor
                .data()
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            (flat, shape)
        } else if tensor.dtype() == safetensors::Dtype::F32 {
            let flat: Vec<f32> = tensor
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (flat, shape)
        } else {
            bail!("unsupported mlx dtype {:?} for {name}", tensor.dtype());
        };
        // MLX stores RMSNorm `alpha` as 1-D `[d]`; consumers expect `[1, 1, d]`.
        if key.ends_with(".alpha") && out_shape.len() == 1 {
            out_shape = vec![1, 1, out_shape[0]];
        }
        out.insert(key, (data, out_shape));
    }
    Ok(out)
}
