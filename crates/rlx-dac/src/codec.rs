use crate::audio::{load_wav_mono, write_wav_mono};
use crate::build::{build_decoder, build_encoder};
use crate::codes::DacCodes;
use crate::config::DacConfig;
use crate::graph::CodecGraph;
use crate::layers::{Decoder, Encoder};
use crate::quantize::{ResidualVectorQuantizer, build_quantizer};
use crate::weights::WeightStore;
use anyhow::{Result, ensure};
use ndarray::{Array2, ArrayView2};
use rlx_runtime::Device;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const SAMPLE_RATE_24KHZ: u32 = 24_000;
pub const SAMPLE_RATE_16KHZ: u32 = 16_000;
pub const SAMPLE_RATE_44KHZ: u32 = 44_100;

#[derive(Debug, Clone)]
pub struct RoundtripStats {
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub num_frames: usize,
    pub pcm_samples: usize,
}

pub struct DacCodec {
    model_dir: PathBuf,
    cfg: DacConfig,
    device: Device,
    encoder: Encoder,
    quantizer: ResidualVectorQuantizer,
    decoder: Decoder,
    // Compiled encoder/decoder graphs, cached by input length. Empty (and
    // unused) when `device == Cpu`, which runs the exact ndarray path.
    enc_graphs: RefCell<HashMap<usize, CodecGraph>>,
    dec_graphs: RefCell<HashMap<(usize, usize), CodecGraph>>,
}

impl DacCodec {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_on(model_dir, Device::Cpu)
    }

    pub fn open_on(model_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let cfg = if model_dir.join("config.json").is_file() {
            DacConfig::load(&model_dir.join("config.json"))?
        } else {
            DacConfig::config_24khz()
        };
        let weights = model_dir.join("model.safetensors");
        ensure!(
            weights.is_file(),
            "missing {} — run fetch or set RLX_DAC_DIR",
            weights.display()
        );
        let store = WeightStore::open(&weights)?;
        Ok(Self {
            model_dir,
            cfg: cfg.clone(),
            device,
            encoder: build_encoder(&cfg, &store)?,
            quantizer: build_quantizer(&cfg, &store)?,
            decoder: build_decoder(&cfg, &store)?,
            enc_graphs: RefCell::new(HashMap::new()),
            dec_graphs: RefCell::new(HashMap::new()),
        })
    }

    /// Run the encoder on the selected backend. CPU uses the exact ndarray
    /// reference; every other backend runs the compiled rlx graph.
    fn encode_z(&self, input: ArrayView2<f32>) -> Result<Array2<f32>> {
        if self.device == Device::Cpu && std::env::var_os("RLX_DAC_CPU_GRAPH").is_none() {
            return Ok(self.encoder.forward(input));
        }
        let in_len = input.shape()[1];
        let mut cache = self.enc_graphs.borrow_mut();
        let graph = match cache.get_mut(&in_len) {
            Some(g) => g,
            None => {
                let g = CodecGraph::encoder(self.device, &self.encoder, in_len)?;
                cache.entry(in_len).or_insert(g)
            }
        };
        let flat: Vec<f32> = input.iter().copied().collect();
        graph.run(&flat)
    }

    /// Run the decoder on the selected backend (see [`Self::encode_z`]).
    fn decode_z(&self, z_q: ArrayView2<f32>) -> Result<Array2<f32>> {
        if self.device == Device::Cpu && std::env::var_os("RLX_DAC_CPU_GRAPH").is_none() {
            return Ok(self.decoder.forward(z_q));
        }
        let (in_c, in_t) = (z_q.shape()[0], z_q.shape()[1]);
        let mut cache = self.dec_graphs.borrow_mut();
        let graph = match cache.get_mut(&(in_c, in_t)) {
            Some(g) => g,
            None => {
                let g = CodecGraph::decoder(self.device, &self.decoder, in_c, in_t)?;
                cache.entry((in_c, in_t)).or_insert(g)
            }
        };
        let flat: Vec<f32> = z_q.iter().copied().collect();
        graph.run(&flat)
    }

    pub fn config(&self) -> &DacConfig {
        &self.cfg
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate
    }

    pub fn hop_length(&self) -> usize {
        self.cfg.hop_length
    }

    fn preprocess(&self, pcm: &[f32]) -> Vec<f32> {
        let len = pcm.len();
        let target = len.div_ceil(self.cfg.hop_length) * self.cfg.hop_length;
        let mut out = pcm.to_vec();
        out.resize(target, 0.0);
        out
    }

    /// Encode mono PCM at the model sample rate.
    pub fn encode_pcm(&self, pcm: &[f32], num_quantizers: Option<usize>) -> Result<DacCodes> {
        ensure!(!pcm.is_empty(), "empty PCM");
        let padded = self.preprocess(pcm);
        let mut input = Array2::<f32>::zeros((1, padded.len()));
        for (i, &s) in padded.iter().enumerate() {
            input[[0, i]] = s;
        }
        let z = self.encode_z(input.view())?;
        let nq = num_quantizers.unwrap_or(self.cfg.n_codebooks);
        let (_z_q, code_rows) = self.quantizer.encode(z.view(), nq);
        let t = code_rows.first().map(|c| c.len()).unwrap_or(0);
        let mut frames = Vec::with_capacity(t);
        for ti in 0..t {
            let mut row = Vec::with_capacity(nq);
            for qi in 0..nq {
                row.push(code_rows[qi][ti]);
            }
            frames.push(row);
        }
        Ok(DacCodes {
            frames,
            num_quantizers: nq,
        })
    }

    /// Decode codec frames → mono PCM at the model sample rate.
    pub fn decode_codes(&self, codes: &DacCodes) -> Result<Vec<f32>> {
        ensure!(!codes.frames.is_empty(), "empty codec frames");
        let z_q = self.quantizer.from_codes(&codes.to_quantizer_rows());
        let wav = self.decode_z(z_q.view())?;
        ensure!(wav.dim().0 >= 1, "decoder produced no channels");
        Ok(wav.row(0).to_vec())
    }

    /// Full forward pass: encode then decode, trimmed to the original PCM length.
    pub fn forward_pcm(
        &self,
        pcm: &[f32],
        num_quantizers: Option<usize>,
    ) -> Result<(DacCodes, Vec<f32>)> {
        let orig_len = pcm.len();
        let codes = self.encode_pcm(pcm, num_quantizers)?;
        let mut recon = self.decode_codes(&codes)?;
        recon.truncate(orig_len.min(recon.len()));
        Ok((codes, recon))
    }

    pub fn encode_wav(
        &self,
        wav: impl AsRef<Path>,
        num_quantizers: Option<usize>,
    ) -> Result<DacCodes> {
        let pcm = load_wav_mono(wav.as_ref(), self.cfg.sample_rate)?;
        self.encode_pcm(&pcm, num_quantizers)
    }

    pub fn decode_wav(
        &self,
        codes: &DacCodes,
        out: impl AsRef<Path>,
        trim_to_samples: Option<usize>,
    ) -> Result<()> {
        let mut pcm = self.decode_codes(codes)?;
        if let Some(n) = trim_to_samples {
            pcm.truncate(n.min(pcm.len()));
        }
        write_wav_mono(out.as_ref(), &pcm, self.cfg.sample_rate)
    }

    pub fn roundtrip_pcm(
        &self,
        pcm: &[f32],
        num_quantizers: Option<usize>,
    ) -> Result<(DacCodes, Vec<f32>, RoundtripStats)> {
        let t0 = Instant::now();
        let codes = self.encode_pcm(pcm, num_quantizers)?;
        let num_frames = codes.num_frames();
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        let mut recon = self.decode_codes(&codes)?;
        let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;
        recon.truncate(pcm.len().min(recon.len()));
        Ok((
            codes,
            recon,
            RoundtripStats {
                encode_ms,
                decode_ms,
                num_frames,
                pcm_samples: pcm.len(),
            },
        ))
    }

    pub fn roundtrip_wav(
        &self,
        in_wav: impl AsRef<Path>,
        out_wav: impl AsRef<Path>,
        num_quantizers: Option<usize>,
    ) -> Result<DacCodes> {
        let pcm = load_wav_mono(in_wav.as_ref(), self.cfg.sample_rate)?;
        let (codes, recon, _) = self.roundtrip_pcm(&pcm, num_quantizers)?;
        write_wav_mono(out_wav.as_ref(), &recon, self.cfg.sample_rate)?;
        Ok(codes)
    }
}

/// Unified [`rlx_core::AudioCodec`] view (see rlx-mimi for the rationale).
impl rlx_core::AudioCodec for DacCodec {
    fn info(&self) -> rlx_core::CodecInfo {
        let hop = self.cfg.hop_length.max(1);
        rlx_core::CodecInfo {
            sample_rate: self.cfg.sample_rate,
            frame_rate: self.cfg.sample_rate as f32 / hop as f32,
            hop_length: self.cfg.hop_length,
            channels: 1,
            max_quantizers: self.cfg.n_codebooks,
            codebook_size: self.cfg.codebook_size,
        }
    }

    fn device(&self) -> Device {
        self.device
    }

    fn encode_pcm(&self, pcm: &[f32], num_quantizers: Option<usize>) -> Result<rlx_core::RvqCodes> {
        let codes = DacCodec::encode_pcm(self, pcm, num_quantizers)?;
        Ok(rlx_core::RvqCodes::new(codes.frames, codes.num_quantizers))
    }

    fn decode_codes(&self, codes: &rlx_core::RvqCodes) -> Result<Vec<f32>> {
        let dc = DacCodes {
            frames: codes.frames.clone(),
            num_quantizers: codes.num_quantizers,
        };
        DacCodec::decode_codes(self, &dc)
    }
}
