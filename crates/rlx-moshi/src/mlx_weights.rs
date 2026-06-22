//! Load Kyutai MLX Q4/Q8/bf16 safetensors → Candle tensor map for `moshi::lm`.

use crate::checkpoint::MoshiCheckpoint;
#[cfg(feature = "gpu-lm")]
use crate::mlx_dequant::{bf16_to_f32, dequantize_affine};
use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[cfg(feature = "gpu-lm")]
use candle::{DType, Device, Tensor};

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

#[cfg(feature = "gpu-lm")]
pub fn load_candle_tensors(
    path: &Path,
    checkpoint: MoshiCheckpoint,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let (bits, group_size) = checkpoint.mlx_quant_params();
    let file = MlxSafetensorsFile::open(path)?;
    let st = SafeTensors::deserialize(&file.data)?;
    let mut out: HashMap<String, Tensor> = HashMap::new();

    for name in st.names() {
        if name.ends_with(".scales") || name.ends_with(".biases") {
            continue;
        }
        let Some(candle_key) = mlx_to_candle_key(name) else {
            continue;
        };
        let tensor = st.tensor(name)?;
        let shape: Vec<usize> = tensor.shape().to_vec();
        let t = if checkpoint.is_mlx_quantized()
            && name.ends_with(".weight")
            && tensor.dtype() == safetensors::Dtype::U32
        {
            let base = &name[..name.len() - ".weight".len()];
            let scales_name = format!("{base}.scales");
            let biases_name = format!("{base}.biases");
            let scales = st
                .tensor(&scales_name)
                .with_context(|| format!("missing {scales_name}"))?;
            let biases = st
                .tensor(&biases_name)
                .with_context(|| format!("missing {biases_name}"))?;
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
            Tensor::from_vec(flat, (rows, out_cols), device)?.to_dtype(dtype)?
        } else if tensor.dtype() == safetensors::Dtype::BF16 {
            let flat: Vec<f32> = tensor
                .data()
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            Tensor::from_vec(flat, shape.as_slice(), device)?.to_dtype(dtype)?
        } else if tensor.dtype() == safetensors::Dtype::F32 {
            let flat: Vec<f32> = tensor
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_vec(flat, shape.as_slice(), device)?.to_dtype(dtype)?
        } else {
            bail!("unsupported mlx dtype {:?} for {name}", tensor.dtype());
        };
        // MLX checkpoints store RMSNorm `alpha` as 1-D `[d]`, but the moshi model
        // expects a broadcastable `[1, 1, d]`. Reshape to match.
        let t = if candle_key.ends_with(".alpha") && t.dims().len() == 1 {
            let d = t.dims()[0];
            t.reshape((1, 1, d))?
        } else {
            t
        };
        out.insert(candle_key, t);
    }
    Ok(out)
}

#[cfg(not(feature = "gpu-lm"))]
pub fn load_eager_weight_map(
    _path: &Path,
    _checkpoint: MoshiCheckpoint,
    _cfg: &crate::config::LmConfig,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    bail!("mlx checkpoints on CPU require `gpu-lm` for dequant or use --device metal")
}

#[cfg(feature = "gpu-lm")]
pub fn load_eager_weight_map(
    path: &Path,
    checkpoint: MoshiCheckpoint,
    _cfg: &crate::config::LmConfig,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    use candle::Device;
    let dev = Device::Cpu;
    let tensors = load_candle_tensors(path, checkpoint, &dev, candle::DType::F32)?;
    let mut map = HashMap::new();
    for (k, t) in tensors {
        let shape: Vec<usize> = t.dims().to_vec();
        let data = if shape.len() == 2 {
            t.to_vec2::<f32>()?.into_iter().flatten().collect()
        } else {
            t.flatten_all()?.to_vec1()?
        };
        map.insert(k, (data, shape));
    }
    Ok(map)
}

/// Build a Kyutai `moshi::lm::LmModel` from MLX-layout safetensors on a Candle device.
#[cfg(feature = "gpu-lm")]
pub fn load_lm_model_mlx<P: AsRef<Path>>(
    cfg: moshi::lm::Config,
    path: P,
    checkpoint: MoshiCheckpoint,
    dtype: DType,
    dev: &Device,
) -> Result<moshi::lm::LmModel> {
    use moshi::nn::MaybeQuantizedVarBuilder;
    let tensors = load_candle_tensors(path.as_ref(), checkpoint, dev, dtype)?;
    let vb =
        MaybeQuantizedVarBuilder::Real(candle_nn::VarBuilder::from_tensors(tensors, dtype, dev));
    Ok(moshi::lm::LmModel::new(&cfg, vb)?)
}
