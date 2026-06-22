use crate::audio::{load_wav_mono, write_wav_mono};
use crate::codes::MimiCodes;
use crate::config::MimiConfig;
use crate::graph::{CodecGraph, DecodeWeights, EncodeWeights};
use crate::layout::{ct_to_tc, tc_to_ct};
use crate::rvq::SplitRvq;
use crate::rvq::build_split_rvq;
use crate::seanet::{
    FrameRateDownsample, FrameRateUpsample, SeanetDecoder, SeanetEncoder, build_decoder,
    build_encoder,
};
use crate::transformer::{MimiTransformer, build_transformer};
use anyhow::{Context, Result, ensure};
use ndarray::Array2;
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_runtime::{Device, is_available};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const SAMPLE_RATE: u32 = 24_000;
pub const FRAME_RATE: f32 = 12.5;

struct EagerCodec {
    cfg: MimiConfig,
    encoder: SeanetEncoder,
    encoder_transformer: MimiTransformer,
    downsample: FrameRateDownsample,
    quantizer: SplitRvq,
    upsample: FrameRateUpsample,
    decoder_transformer: MimiTransformer,
    decoder: SeanetDecoder,
}

pub struct MimiCodec {
    model_dir: PathBuf,
    cfg: MimiConfig,
    device: Device,
    eager: EagerCodec,
    // Compiled encode/decode graphs, cached by input length. Unused when
    // `device == Cpu` (which runs the exact ndarray path).
    enc_graphs: RefCell<HashMap<usize, CodecGraph>>,
    dec_graphs: RefCell<HashMap<usize, CodecGraph>>,
}

#[derive(Debug, Clone)]
pub struct RoundtripStats {
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub num_frames: usize,
    pub pcm_samples: usize,
}

impl MimiCodec {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_on(model_dir, Device::Cpu)
    }

    pub fn open_on(model_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::open_on_with_moshi(model_dir, None, device, None)
    }

    pub fn open_on_with_moshi(
        model_dir: impl AsRef<Path>,
        _moshi_dir: Option<&Path>,
        device: Device,
        _mimi_codebooks: Option<usize>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let cfg = MimiConfig::load(&model_dir)?;
        // Non-CPU devices run the rlx-runtime graph; fall back to CPU eager if
        // the requested backend isn't compiled in / available on this host.
        let actual_device = if device == Device::Cpu || is_available(device) {
            device
        } else {
            eprintln!("mimi: {device:?} not available — using CPU");
            Device::Cpu
        };
        let eager = EagerCodec::open(&model_dir, &cfg)?;
        Ok(Self {
            model_dir,
            cfg,
            device: actual_device,
            eager,
            enc_graphs: RefCell::new(HashMap::new()),
            dec_graphs: RefCell::new(HashMap::new()),
        })
    }

    fn encode_weights(&self) -> EncodeWeights<'_> {
        EncodeWeights {
            encoder: &self.eager.encoder,
            transformer: &self.eager.encoder_transformer,
            downsample: &self.eager.downsample,
            audio_channels: self.cfg.audio_channels,
            hidden_size: self.cfg.hidden_size,
        }
    }

    fn decode_weights(&self) -> DecodeWeights<'_> {
        DecodeWeights {
            upsample: &self.eager.upsample,
            transformer: &self.eager.decoder_transformer,
            decoder: &self.eager.decoder,
            hidden_size: self.cfg.hidden_size,
        }
    }

    /// Run SEANet-encoder + transformer + downsample on the selected backend,
    /// returning the pre-quantization latent `[hidden, t_ds]`.
    fn run_encode(&self, pcm: &[f32]) -> Result<Array2<f32>> {
        let in_len = pcm.len();
        let mut cache = self.enc_graphs.borrow_mut();
        let graph = match cache.get_mut(&in_len) {
            Some(g) => g,
            None => {
                let g = CodecGraph::encoder(self.device, &self.encode_weights(), in_len)?;
                cache.entry(in_len).or_insert(g)
            }
        };
        graph.run(pcm)
    }

    /// Run upsample + transformer + SEANet-decoder on the selected backend,
    /// returning the waveform `[1, out_len]`.
    fn run_decode(&self, emb: &Array2<f32>) -> Result<Array2<f32>> {
        let in_t = emb.shape()[1];
        let mut cache = self.dec_graphs.borrow_mut();
        let graph = match cache.get_mut(&in_t) {
            Some(g) => g,
            None => {
                let g = CodecGraph::decoder(self.device, &self.decode_weights(), in_t)?;
                cache.entry(in_t).or_insert(g)
            }
        };
        let flat: Vec<f32> = emb.iter().copied().collect();
        graph.run(&flat)
    }

    pub fn config(&self) -> &MimiConfig {
        &self.cfg
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Encode mono PCM @ [`SAMPLE_RATE`].
    pub fn encode_pcm(&self, pcm: &[f32], num_quantizers: Option<usize>) -> Result<MimiCodes> {
        ensure!(!pcm.is_empty(), "empty PCM");
        if self.device == Device::Cpu {
            return self.eager.encode_pcm(pcm, num_quantizers);
        }
        let ds = self.run_encode(pcm)?;
        let nq = num_quantizers.unwrap_or(self.cfg.num_quantizers);
        let frames = self.eager.quantizer.encode_frames(&ds, Some(nq));
        Ok(MimiCodes {
            frames,
            num_quantizers: nq,
        })
    }

    /// Decode codec frames → mono PCM @ [`SAMPLE_RATE`].
    pub fn decode_codes(&self, codes: &MimiCodes) -> Result<Vec<f32>> {
        ensure!(!codes.frames.is_empty(), "empty codec frames");
        if self.device == Device::Cpu {
            return self.eager.decode_codes(codes);
        }
        let emb = self.eager.quantizer.decode_frames(&codes.frames);
        let wav = self.run_decode(&emb)?;
        ensure!(wav.dim().0 >= 1, "decoder produced no channels");
        Ok(wav.row(0).to_vec())
    }

    pub fn encode_wav(
        &self,
        wav: impl AsRef<Path>,
        num_quantizers: Option<usize>,
    ) -> Result<MimiCodes> {
        let pcm = load_wav_mono(wav.as_ref(), SAMPLE_RATE)?;
        self.encode_pcm(&pcm, num_quantizers)
    }

    pub fn decode_wav(
        &self,
        codes: &MimiCodes,
        out: impl AsRef<Path>,
        trim_to_samples: Option<usize>,
    ) -> Result<()> {
        let mut pcm = self.decode_codes(codes)?;
        if let Some(n) = trim_to_samples {
            pcm.truncate(n.min(pcm.len()));
        }
        write_wav_mono(out.as_ref(), &pcm, SAMPLE_RATE)
    }

    pub fn roundtrip_pcm(
        &self,
        pcm: &[f32],
        num_quantizers: Option<usize>,
    ) -> Result<(MimiCodes, Vec<f32>, RoundtripStats)> {
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
    ) -> Result<MimiCodes> {
        let pcm = load_wav_mono(in_wav.as_ref(), SAMPLE_RATE)?;
        let (codes, recon, _) = self.roundtrip_pcm(&pcm, num_quantizers)?;
        write_wav_mono(out_wav.as_ref(), &recon, SAMPLE_RATE)?;
        Ok(codes)
    }
}

impl EagerCodec {
    fn open(model_dir: &Path, cfg: &MimiConfig) -> Result<Self> {
        let map = load_weight_map(model_dir)?;
        let mut aux: HashMap<String, _> = map
            .iter()
            .filter(|(k, _)| k.starts_with("downsample.") || k.starts_with("upsample."))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Ok(Self {
            cfg: cfg.clone(),
            encoder: build_encoder(cfg, subset_prefix(&map, "encoder."))?,
            encoder_transformer: build_transformer(
                cfg,
                "encoder_transformer",
                subset_prefix(&map, "encoder_transformer."),
            )?,
            downsample: FrameRateDownsample::from_weights(cfg, &mut aux)?,
            quantizer: build_split_rvq(cfg, subset_prefix(&map, "quantizer."))?,
            upsample: FrameRateUpsample::from_weights(cfg, &mut aux)?,
            decoder_transformer: build_transformer(
                cfg,
                "decoder_transformer",
                subset_prefix(&map, "decoder_transformer."),
            )?,
            decoder: build_decoder(cfg, subset_prefix(&map, "decoder."))?,
        })
    }

    fn encode_pcm(&self, pcm: &[f32], num_quantizers: Option<usize>) -> Result<MimiCodes> {
        let mut input = Array2::<f32>::zeros((self.cfg.audio_channels, pcm.len()));
        for (i, &s) in pcm.iter().enumerate() {
            input[[0, i]] = s;
        }
        let conv = self.encoder.forward(input.view());
        let pre_tf = ct_to_tc(conv.view());
        let post_tf = self.encoder_transformer.forward(pre_tf.view());
        let pre_ds = tc_to_ct(post_tf.view());
        let ds = self.downsample.forward(pre_ds.view());
        let nq = num_quantizers.unwrap_or(self.cfg.num_quantizers);
        let frames = self.quantizer.encode_frames(&ds, Some(nq));
        Ok(MimiCodes {
            frames,
            num_quantizers: nq,
        })
    }

    fn decode_codes(&self, codes: &MimiCodes) -> Result<Vec<f32>> {
        let emb = self.quantizer.decode_frames(&codes.frames);
        let up = self.upsample.forward(emb.view());
        let pre_tf = ct_to_tc(up.view());
        let post_tf = self.decoder_transformer.forward(pre_tf.view());
        let pre_dec = tc_to_ct(post_tf.view());
        let wav = self.decoder.forward(pre_dec.view());
        ensure!(wav.dim().0 >= 1, "decoder produced no channels");
        Ok(wav.row(0).to_vec())
    }
}

/// Raw safetensors map: tensor name -> `(data, shape)`.
type RawTensorMap = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn load_weight_map(model_dir: &Path) -> Result<RawTensorMap> {
    let ckpt = SafetensorsCheckpoint::open(model_dir)?;
    let keys: std::collections::HashSet<String> = ckpt.keys().map(str::to_string).collect();
    let mut wm = ckpt.load_selected(&keys)?;
    let mut map = HashMap::with_capacity(keys.len());
    for key in keys {
        let (data, shape) = wm
            .take(&key)
            .with_context(|| format!("tensor {key} missing after load"))?;
        map.insert(key, (data, shape));
    }
    Ok(map)
}

fn subset_prefix(map: &RawTensorMap, prefix: &str) -> RawTensorMap {
    map.iter()
        .filter(|(k, _)| k.starts_with(prefix))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// Unified [`rlx_core::AudioCodec`] view so TTS/ASR consumers can use Mimi
/// interchangeably with other codecs (bitrate control + resampling for free).
impl rlx_core::AudioCodec for MimiCodec {
    fn info(&self) -> rlx_core::CodecInfo {
        rlx_core::CodecInfo {
            sample_rate: self.cfg.sampling_rate,
            frame_rate: self.cfg.frame_rate,
            hop_length: self.cfg.samples_per_codec_frame(),
            channels: self.cfg.audio_channels,
            max_quantizers: self.cfg.num_quantizers,
            codebook_size: self.cfg.codebook_size,
        }
    }

    fn device(&self) -> Device {
        self.device
    }

    fn encode_pcm(&self, pcm: &[f32], num_quantizers: Option<usize>) -> Result<rlx_core::RvqCodes> {
        // Inherent `MimiCodec::encode_pcm` (inherent methods take resolution
        // priority over the trait method of the same name).
        let codes = MimiCodec::encode_pcm(self, pcm, num_quantizers)?;
        Ok(rlx_core::RvqCodes::new(codes.frames, codes.num_quantizers))
    }

    fn decode_codes(&self, codes: &rlx_core::RvqCodes) -> Result<Vec<f32>> {
        let mc = MimiCodes {
            frames: codes.frames.clone(),
            num_quantizers: codes.num_quantizers,
        };
        MimiCodec::decode_codes(self, &mc)
    }
}
