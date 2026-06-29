//! Kyutai `moshi::mimi` on Candle Metal/CUDA.

use crate::codes::MimiCodes;
use crate::config::MimiConfig;
use anyhow::{Context, Result};
use rlx_runtime::Device;
use std::path::Path;

pub struct GpuMimiCodec {
    inner: moshi::mimi::Mimi,
    cfg: MimiConfig,
    candle_device: candle::Device,
}

impl GpuMimiCodec {
    pub fn open_weights(
        weights: &Path,
        cfg: MimiConfig,
        num_codebooks: usize,
        device: Device,
    ) -> Result<Self> {
        let candle_device = rlx_device_to_candle(device)?;
        let nq = num_codebooks.min(cfg.num_quantizers);
        let inner = moshi::mimi::load(
            weights.to_str().context("mimi weights path utf-8")?,
            Some(nq),
            &candle_device,
        )
        .with_context(|| format!("load mimi on {device:?} from {}", weights.display()))?;
        Ok(Self {
            inner,
            cfg,
            candle_device,
        })
    }

    pub fn config(&self) -> &MimiConfig {
        &self.cfg
    }

    pub fn candle_device(&self) -> &candle::Device {
        &self.candle_device
    }

    pub fn encode_pcm(&mut self, pcm: &[f32], num_quantizers: Option<usize>) -> Result<MimiCodes> {
        let nq = num_quantizers.unwrap_or(self.cfg.num_quantizers);
        let pcm_t = candle::Tensor::from_slice(pcm, (1, 1, pcm.len()), &self.candle_device)?
            .to_dtype(candle::DType::F32)?;
        self.inner.reset_state();
        let codes = self
            .inner
            .encode(&pcm_t)
            .context("mimi gpu encode")?
            .to_dtype(candle::DType::U32)?;
        let codes3 = codes.to_vec3::<u32>().context("codes to vec")?;
        let num_frames = codes3[0][0].len();
        let mut frames = Vec::with_capacity(num_frames);
        for fi in 0..num_frames {
            let mut frame = Vec::with_capacity(nq);
            for qi in 0..nq.min(codes3[0].len()) {
                frame.push(codes3[0][qi][fi]);
            }
            frames.push(frame);
        }
        Ok(MimiCodes {
            frames,
            num_quantizers: nq,
        })
    }

    pub fn decode_codes(&mut self, codes: &MimiCodes) -> Result<Vec<f32>> {
        let codes_t = codes_tensor(codes, &self.candle_device)?;
        self.inner.reset_state();
        let pcm = self
            .inner
            .decode(&codes_t)
            .context("mimi gpu decode")?
            .to_dtype(candle::DType::F32)?;
        let pcm3 = pcm.to_vec3::<f32>()?;
        Ok(pcm3[0][0].clone())
    }
}

fn codes_tensor(codes: &MimiCodes, device: &candle::Device) -> Result<candle::Tensor> {
    let nq = codes.num_quantizers;
    let nf = codes.num_frames();
    let mut flat = vec![0u32; nq * nf];
    for (fi, frame) in codes.frames.iter().enumerate() {
        for (qi, &tok) in frame.iter().enumerate().take(nq) {
            flat[qi * nf + fi] = tok;
        }
    }
    Ok(candle::Tensor::from_slice(&flat, (1, nq, nf), device)?)
}

pub fn rlx_device_to_candle(device: Device) -> Result<candle::Device> {
    Ok(match device {
        Device::Cpu => candle::Device::Cpu,
        Device::Metal => candle::Device::new_metal(0).context("Metal device")?,
        Device::Cuda => candle::Device::new_cuda(0).context("CUDA device")?,
        Device::Mlx => candle::Device::new_metal(0).context("MLX→Metal mimi codec")?,
        other => anyhow::bail!("mimi parity-mimi codec unsupported device {other:?}"),
    })
}
