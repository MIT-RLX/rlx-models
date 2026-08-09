//! TSAC decode on RLX backends.
//!
//! TSAC's decoder is the Descript-DAC 44.1 kHz decoder (latent 1024, decoder dim
//! 1536, upsample rates [8,8,4,2] → 512× hop). We reuse `rlx-dac`'s HIR graph
//! (verified bit-exact on cpu/metal/mlx/wgpu) and feed it the *exact* weights the
//! C reference uses: the vendored `dequant_weights` is exposed via FFI so the q8
//! group-scale + weight-norm math is identical. The RVQ codebook → latent step is
//! a tiny gather + matmul and stays host-side (as in rlx-dac's quantizer).
//!
//! The serial pieces (TXC container parse, range coder) remain in C — they are
//! bit I/O, not a backend workload.

use crate::native::ffi;
use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3};
use rlx_dac::layers::{
    Decoder, DecoderBlock, Encoder, EncoderBlock, ResidualUnit, WnConv1d, WnConvTranspose1d,
};
use rlx_runtime::Device;
use std::ffi::CString;

/// Residual-unit conv1 dilations within each DAC decoder block.
const RES_DILATIONS: [usize; 3] = [1, 3, 9];
/// RVQ latent dimensionality.
const RVQ_DIM: usize = 1024;

/// C decoder batching (must match vendored `cpu_decoder.c`).
const BATCH_FRAMES: usize = 16;
const CONTEXT_PAD: usize = 10;

/// A loaded, device-bound TSAC/DAC decoder ready to turn RVQ codes into PCM.
pub struct RlxDecoder {
    device: Device,
    decoder: Decoder,
    /// Per-codebook in_proj `[8, 1024]` (row-major `[Co=8][Ci=1024]`).
    in_proj: Vec<Array2<f32>>,
    /// Per-codebook out_proj `[1024, 8]` (row-major `[Co=1024][Ci=8]`).
    out_proj: Vec<Array2<f32>>,
    /// Output channel count of the DAC head (1 mono / 2 stereo model).
    out_channels: usize,
    /// Total temporal upscale (product of upsample strides; 512 for TSAC).
    total_upscale: usize,
    /// Faithful tsac-ng mode: scrambled snake + K/2 conv-T crop + 16-frame
    /// batching (bit-exact with the vendored C decoder). `false` = standard DAC
    /// (per-channel snake, stride/2 crop, whole-sequence) — the "correct" path.
    faithful: bool,
    /// GPU decoder graphs, cached by context-window length (frames).
    graphs: std::cell::RefCell<std::collections::HashMap<usize, rlx_dac::graph::CodecGraph>>,
}

impl RlxDecoder {
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Open a decoder from an install dir, pulling weights from a freshly loaded
    /// native model. The native context is dropped before returning — all weights
    /// are copied out — so the result owns everything it needs.
    pub fn open(install_dir: impl AsRef<std::path::Path>, device: Device) -> Result<Self> {
        Self::open_mode(install_dir, device, true)
    }

    /// Like [`open`](Self::open) but selects the decode convention. `faithful=true`
    /// matches the vendored C decoder bit-exactly; `false` is standard DAC.
    pub fn open_mode(
        install_dir: impl AsRef<std::path::Path>,
        device: Device,
        faithful: bool,
    ) -> Result<Self> {
        let opts = crate::TsacOptions {
            device: Device::Cpu,
            backend: crate::TsacBackendKind::Native,
            ..Default::default()
        };
        let native = crate::native::NativeCodec::open(install_dir, &opts)?;
        Self::from_ctx_mode(native.ctx_raw(), device, faithful)
    }

    /// Build from a live native context (faithful tsac-ng mode).
    pub fn from_ctx(ctx: *mut ffi::TSACContext, device: Device) -> Result<Self> {
        Self::from_ctx_mode(ctx, device, true)
    }

    /// Build from a live native context (its model must be loaded). Pulls all
    /// dequantized weights once; the resulting `RlxDecoder` is self-contained.
    pub fn from_ctx_mode(
        ctx: *mut ffi::TSACContext,
        device: Device,
        faithful: bool,
    ) -> Result<Self> {
        if ctx.is_null() {
            bail!("null tsac context");
        }

        // Stem: decoder.model.0 conv1d (1024 -> 1536, K=7, same padding).
        let stem = load_conv1d(ctx, "decoder.model.0", 1, 3, 1, true)?
            .context("missing decoder.model.0")?;

        // Four upsample blocks: decoder.model.{1..=4}.
        let mut blocks = Vec::with_capacity(4);
        let mut total_upscale = 1usize;
        for i in 1..=4 {
            let (blk, stride) = load_decoder_block(ctx, &format!("decoder.model.{i}"), faithful)?;
            total_upscale *= stride;
            blocks.push(blk);
        }

        // Final snake + head conv (96 -> out_channels, K=7).
        let snake_alpha = load_alpha(ctx, "decoder.model.5.alpha")?;
        let head = load_conv1d(ctx, "decoder.model.6", 1, 3, 1, true)?
            .context("missing decoder.model.6")?;
        let out_channels = head.weight.dim().0;

        // Faithful mode uses tsac-ng's non-standard snake α indexing.
        let decoder = Decoder::from_parts(stem, blocks, snake_alpha, head, faithful);

        // RVQ projections for up to 12 codebooks (stop at the first absent one).
        let mut in_proj = Vec::new();
        let mut out_proj = Vec::new();
        for cb in 0..12 {
            let ip = load_conv_raw(ctx, &format!("quantizer.quantizers.{cb}.in_proj"));
            let op = load_conv_raw(ctx, &format!("quantizer.quantizers.{cb}.out_proj"));
            match (ip, op) {
                (Some((ipw, ico, ici, ik)), Some((opw, oco, oci, ok))) => {
                    debug_assert_eq!((ik, ok), (1, 1), "RVQ projections must be K=1");
                    in_proj.push(Array2::from_shape_vec((ico, ici), ipw)?);
                    out_proj.push(Array2::from_shape_vec((oco, oci), opw)?);
                }
                _ => break,
            }
        }
        if in_proj.is_empty() {
            bail!("no RVQ quantizer projections found in model");
        }

        Ok(Self {
            device,
            decoder,
            in_proj,
            out_proj,
            out_channels,
            total_upscale,
            faithful,
            graphs: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    /// Decode a full continuous latent `[1024, T]` directly to PCM `[Co, T*up]`
    /// (no RVQ, no batching). Used to measure the autoencoder (enc→dec) on its own.
    pub fn decode_latent_direct(&self, z: &Array2<f32>) -> Result<Array2<f32>> {
        self.decode_latent(z, z.dim().1)
    }

    pub fn in_proj(&self) -> &[Array2<f32>] {
        &self.in_proj
    }
    pub fn out_proj(&self) -> &[Array2<f32>] {
        &self.out_proj
    }

    /// RVQ for frames `[start, start+len)` (codes row-major `[frame][cb]`) →
    /// latent `[1024, len]`. Mirrors `decode_batch`'s RVQ loop exactly.
    fn rvq_latent_range(
        &self,
        codes: &[i32],
        n_cb: usize,
        start: usize,
        len: usize,
    ) -> Array2<f32> {
        let mut latent = Array2::<f32>::zeros((RVQ_DIM, len));
        let n_cb = n_cb.min(self.in_proj.len());
        for cb in 0..n_cb {
            let ip = &self.in_proj[cb]; // [8, 1024]
            let op = &self.out_proj[cb]; // [1024, 8]
            let (n_o, ci) = ip.dim();
            let (d_dim, _) = op.dim();
            for lf in 0..len {
                let mut raw = codes[(start + lf) * n_cb + cb] as usize;
                if raw >= ci {
                    raw = ci - 1;
                }
                let mut ip_vec = [0f32; 16];
                for o in 0..n_o {
                    ip_vec[o] = ip[[o, raw]];
                }
                for d in 0..d_dim {
                    let mut sum = 0f32;
                    for o in 0..n_o {
                        sum += ip_vec[o] * op[[d, o]];
                    }
                    latent[[d, lf]] += sum;
                }
            }
        }
        latent
    }

    /// Decode RVQ codes to PCM `[out_channels, n_frames * total_upscale]`.
    ///
    /// Decodes in the SAME 16-frame batches with 10-frame context as the C
    /// reference. This matters for bit-parity: TSAC's snake α index is a function
    /// of each layer's time length `T`, and C runs the decoder per context window
    /// — so the batching must match for the snake (and thus the output) to align.
    pub fn decode_codes(&self, codes: &[i32], n_frames: usize, n_cb: usize) -> Result<Array2<f32>> {
        let up = self.total_upscale;
        let total_samples = n_frames * up;
        let mut out = Array2::<f32>::zeros((self.out_channels, total_samples));
        if n_frames == 0 {
            return Ok(out);
        }

        // Standard mode: whole-sequence decode (no per-batch context needed — the
        // snake is genuinely per-channel, independent of T).
        if !self.faithful {
            let latent = self.rvq_latent_range(codes, n_cb, 0, n_frames);
            return self.decode_latent(&latent, n_frames);
        }

        let mut batch_start = 0;
        while batch_start < n_frames {
            let batch_end = (batch_start + BATCH_FRAMES).min(n_frames);
            let batch_frames = batch_end - batch_start;
            let ctx_start = batch_start.saturating_sub(CONTEXT_PAD);
            let ctx_end = (batch_end + CONTEXT_PAD).min(n_frames);
            let ctx_frames = ctx_end - ctx_start;

            // RVQ for this context window → latent [1024, ctx_frames].
            let latent = self.rvq_latent_range(codes, n_cb, ctx_start, ctx_frames);
            let pcm = self.decode_latent(&latent, ctx_frames)?; // [Co, ctx_frames*up]

            let discard = (batch_start - ctx_start) * up;
            let out_start = batch_start * up;
            let count = (batch_frames * up).min(pcm.dim().1.saturating_sub(discard));
            for c in 0..self.out_channels {
                for s in 0..count {
                    if out_start + s < total_samples {
                        out[[c, out_start + s]] = pcm[[c, discard + s]];
                    }
                }
            }
            batch_start = batch_end;
        }
        Ok(out)
    }

    /// Decode one latent window `[1024, t]` → PCM `[Co, t*up]` on the device.
    fn decode_latent(&self, latent: &Array2<f32>, t: usize) -> Result<Array2<f32>> {
        if self.device == Device::Cpu {
            return Ok(self.decoder.forward(latent.view()));
        }
        let flat: Vec<f32> = latent.iter().copied().collect();
        let mut cache = self.graphs.borrow_mut();
        let graph = match cache.get_mut(&t) {
            Some(g) => g,
            None => {
                let g =
                    rlx_dac::graph::CodecGraph::decoder(self.device, &self.decoder, RVQ_DIM, t)?;
                cache.entry(t).or_insert(g)
            }
        };
        graph.run(&flat)
    }

    /// Full file decode: read TXC codes (C), RVQ + DAC on the device, write WAV.
    pub fn decode_file(&self, in_tsac: &std::path::Path, out_wav: &std::path::Path) -> Result<()> {
        let (codes, n_frames, n_cb) = read_codes(in_tsac)?;
        let pcm = self.decode_codes(&codes, n_frames, n_cb)?; // [Co][T]
        let (co, t) = pcm.dim();
        let mut interleaved = Vec::with_capacity(co * t);
        for s in 0..t {
            for c in 0..co {
                interleaved.push(pcm[[c, s]]);
            }
        }
        crate::audio::write_wav_f32(out_wav, &interleaved, crate::SAMPLE_RATE, co as u16)?;
        Ok(())
    }
}

/// DAC encoder on RLX backends: PCM → continuous latent `[1024, nf]`. The conv
/// stack runs as rlx-dac's encoder graph (all backends); RVQ quantization (codes)
/// is layered on top via `RlxCodec`. Standard DAC conventions (the Bellard q8
/// weights are a standard DAC, not tsac-ng's quirky decode).
pub struct RlxEncoder {
    device: Device,
    encoder: Encoder,
    in_channels: usize,
    hop: usize,
    graphs: std::cell::RefCell<std::collections::HashMap<usize, rlx_dac::graph::CodecGraph>>,
}

impl RlxEncoder {
    pub fn open(install_dir: impl AsRef<std::path::Path>, device: Device) -> Result<Self> {
        let opts = crate::TsacOptions {
            device: Device::Cpu,
            backend: crate::TsacBackendKind::Native,
            ..Default::default()
        };
        let native = crate::native::NativeCodec::open(install_dir, &opts)?;
        Self::from_ctx(native.ctx_raw(), device)
    }

    pub fn from_ctx(ctx: *mut ffi::TSACContext, device: Device) -> Result<Self> {
        if ctx.is_null() {
            bail!("null tsac context");
        }
        let (encoder, in_channels, hop) = load_encoder(ctx)?;
        Ok(Self {
            device,
            encoder,
            in_channels,
            hop,
            graphs: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Encode interleaved PCM (`[n_samples * channels]`) to a continuous latent
    /// `[1024, nf]` where `nf = ceil(n_samples / hop)`.
    pub fn encode_latent(&self, pcm: &[f32], channels: usize) -> Result<Array2<f32>> {
        let channels = channels.max(1);
        let n_samples = pcm.len() / channels;
        let nf = n_samples.div_ceil(self.hop).max(1);
        let in_len = nf * self.hop;
        // [in_channels, in_len] row-major; mono → duplicate into both stem inputs.
        let mut input = vec![0f32; self.in_channels * in_len];
        for t in 0..n_samples {
            for c in 0..self.in_channels {
                let src = if c < channels { c } else { 0 };
                input[c * in_len + t] = pcm[t * channels + src];
            }
        }
        if self.device == Device::Cpu {
            let xa = Array2::from_shape_vec((self.in_channels, in_len), input)?;
            return Ok(self.encoder.forward(xa.view()));
        }
        let mut cache = self.graphs.borrow_mut();
        let g = match cache.get_mut(&in_len) {
            Some(g) => g,
            None => {
                let g = rlx_dac::graph::CodecGraph::encoder(self.device, &self.encoder, in_len)?;
                cache.entry(in_len).or_insert(g)
            }
        };
        g.run(&input)
    }
}

/// Debug: open a native ctx for an install dir and run a closure with it.
fn with_ctx<T>(
    install_dir: impl AsRef<std::path::Path>,
    f: impl FnOnce(*mut ffi::TSACContext) -> T,
) -> Result<T> {
    let opts = crate::TsacOptions {
        device: Device::Cpu,
        backend: crate::TsacBackendKind::Native,
        ..Default::default()
    };
    let native = crate::native::NativeCodec::open(install_dir, &opts)?;
    Ok(f(native.ctx_raw()))
}

/// Debug: q8 conv1d weight via standard weight-norm (raw v + g·v/‖v‖), `[Co][Ci][K]`.
pub fn q8_conv_std(install_dir: impl AsRef<std::path::Path>, prefix: &str) -> Result<Array3<f32>> {
    with_ctx(install_dir, |ctx| std_conv1d_weight(ctx, prefix))?
        .with_context(|| format!("q8 conv {prefix}"))
}

/// Debug: q8 conv-transpose weight via standard weight-norm, `[Ci][Co][K]`.
pub fn q8_convt_std(install_dir: impl AsRef<std::path::Path>, prefix: &str) -> Result<Array3<f32>> {
    with_ctx(install_dir, |ctx| std_convt_weight(ctx, prefix))?
        .with_context(|| format!("q8 convT {prefix}"))
}

/// Debug: raw q8-dequantized `{prefix}.weight_v` + its on-disk dims `[d0,d1,d2]`.
pub fn q8_v_raw(
    install_dir: impl AsRef<std::path::Path>,
    prefix: &str,
) -> Result<(Vec<f32>, [usize; 3])> {
    with_ctx(install_dir, |ctx| load_v_raw(ctx, prefix))?
        .map(|(v, d0, d1, d2)| (v, [d0, d1, d2]))
        .with_context(|| format!("q8 raw v {prefix}"))
}

/// Debug: copy a plain f32 tensor (codebook/alpha/bias) by exact name.
pub fn q8_f32(install_dir: impl AsRef<std::path::Path>, name: &str) -> Result<Vec<f32>> {
    with_ctx(install_dir, |ctx| load_f32(ctx, name))?.with_context(|| format!("q8 tensor {name}"))
}

/// Read RVQ code indices from a `.tsac`/`.txc` file via the C container parser.
pub fn read_codes(path: &std::path::Path) -> Result<(Vec<i32>, usize, usize)> {
    let c = CString::new(path.to_string_lossy().into_owned()).context("path utf-8")?;
    let mut n_frames: i32 = 0;
    let mut n_cb: i32 = 0;
    let ptr = unsafe { ffi::tsac_rlx_read_codes(c.as_ptr(), &mut n_frames, &mut n_cb) };
    if ptr.is_null() {
        bail!("failed to read TXC codes from {}", path.display());
    }
    let len = (n_frames as usize) * (n_cb as usize);
    let codes = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
    unsafe { ffi::tsac_free_buffer(ptr as *mut std::os::raw::c_void) };
    Ok((codes, n_frames as usize, n_cb as usize))
}

// ----------------------------------------------------------------------------
// FFI weight pulls (all data copied out + freed immediately).
// ----------------------------------------------------------------------------

/// Dequantize a conv layer, returning `(weights[Co*Ci*K], Co, Ci, K, is_ct)`.
fn load_conv_dequant(
    ctx: *mut ffi::TSACContext,
    prefix: &str,
) -> Option<(Vec<f32>, usize, usize, usize, bool)> {
    let c = CString::new(prefix).ok()?;
    let (mut co, mut ci, mut k, mut is_ct) = (0i32, 0i32, 0i32, 0i32);
    let ptr =
        unsafe { ffi::tsac_rlx_conv_weight(ctx, c.as_ptr(), &mut co, &mut ci, &mut k, &mut is_ct) };
    if ptr.is_null() {
        return None;
    }
    let n = (co as usize) * (ci as usize) * (k as usize);
    let data = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();
    unsafe { ffi::tsac_free_buffer(ptr as *mut std::os::raw::c_void) };
    Some((data, co as usize, ci as usize, k as usize, is_ct != 0))
}

/// Like [`load_conv_dequant`] but drops the `is_ct` flag (used for RVQ K=1 projs).
fn load_conv_raw(
    ctx: *mut ffi::TSACContext,
    prefix: &str,
) -> Option<(Vec<f32>, usize, usize, usize)> {
    load_conv_dequant(ctx, prefix).map(|(w, co, ci, k, _)| (w, co, ci, k))
}

/// True when the standard-DAC weight-norm experiment is enabled (`RLX_STD=1`).
fn std_weightnorm() -> bool {
    std::env::var("RLX_STD").is_ok()
}

/// Raw q8-dequantized `weight_v` in on-disk `[d0][d1][d2]` layout.
fn load_v_raw(ctx: *mut ffi::TSACContext, prefix: &str) -> Option<(Vec<f32>, usize, usize, usize)> {
    let c = CString::new(prefix).ok()?;
    let (mut d0, mut d1, mut d2) = (0i32, 0i32, 0i32);
    let ptr = unsafe { ffi::tsac_rlx_weight_v_raw(ctx, c.as_ptr(), &mut d0, &mut d1, &mut d2) };
    if ptr.is_null() {
        return None;
    }
    let n = (d0 * d1 * d2) as usize;
    let data = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();
    unsafe { ffi::tsac_free_buffer(ptr as *mut std::os::raw::c_void) };
    Some((data, d0 as usize, d1 as usize, d2 as usize))
}

/// Standard-DAC conv1d weight: raw `v` is `[Ci][K][Co]`; apply weight-norm
/// `g·v/‖v‖` per output channel and return `[Co][Ci][K]`.
fn std_conv1d_weight(ctx: *mut ffi::TSACContext, prefix: &str) -> Option<Array3<f32>> {
    let (v, ci, k, co) = load_v_raw(ctx, prefix)?; // [Ci][K][Co]
    let g = load_f32(ctx, &format!("{prefix}.weight_g")); // [Co]
    let mut w = vec![0f32; co * ci * k];
    for c in 0..co {
        let mut nrm = 0f64;
        for i in 0..ci {
            for j in 0..k {
                let val = v[(i * k + j) * co + c] as f64;
                nrm += val * val;
            }
        }
        let nrm = nrm.sqrt().max(1e-12);
        let gc = g.as_ref().map(|g| g[c]).unwrap_or(nrm as f32) as f64;
        let scale = gc / nrm;
        for i in 0..ci {
            for j in 0..k {
                w[(c * ci + i) * k + j] = (v[(i * k + j) * co + c] as f64 * scale) as f32;
            }
        }
    }
    Array3::from_shape_vec((co, ci, k), w).ok()
}

/// Standard-DAC conv-transpose weight: raw `v` is `[Co][K][Ci]`; PyTorch
/// `ConvTranspose1d` weight is `[Ci][Co][K]` with weight-norm per the axis whose
/// length matches `weight_g`. Returns `[Ci][Co][K]` for rlx-dac.
fn std_convt_weight(ctx: *mut ffi::TSACContext, prefix: &str) -> Option<Array3<f32>> {
    let (v, co, k, ci) = load_v_raw(ctx, prefix)?; // [Co][K][Ci]
    let g = load_f32(ctx, &format!("{prefix}.weight_g"));
    let per_in = g.as_ref().map(|g| g.len() == ci).unwrap_or(true);
    // norms over the complementary axes.
    let mut norm = vec![0f64; if per_in { ci } else { co }];
    for oc in 0..co {
        for j in 0..k {
            for ic in 0..ci {
                let val = v[(oc * k + j) * ci + ic] as f64;
                norm[if per_in { ic } else { oc }] += val * val;
            }
        }
    }
    for x in norm.iter_mut() {
        *x = x.sqrt().max(1e-12);
    }
    let mut w = vec![0f32; ci * co * k]; // [Ci][Co][K]
    for ic in 0..ci {
        for oc in 0..co {
            for j in 0..k {
                let idx = if per_in { ic } else { oc };
                let gc = g.as_ref().map(|g| g[idx]).unwrap_or(norm[idx] as f32) as f64;
                let scale = gc / norm[idx];
                w[(ic * co + oc) * k + j] = (v[(oc * k + j) * ci + ic] as f64 * scale) as f32;
            }
        }
    }
    Array3::from_shape_vec((ci, co, k), w).ok()
}

/// Copy a plain f32 tensor (alpha/bias) by exact name.
fn load_f32(ctx: *mut ffi::TSACContext, name: &str) -> Option<Vec<f32>> {
    let c = CString::new(name).ok()?;
    let mut n: i32 = 0;
    let ptr = unsafe { ffi::tsac_rlx_f32(ctx, c.as_ptr(), &mut n) };
    if ptr.is_null() {
        return None;
    }
    let data = unsafe { std::slice::from_raw_parts(ptr, n as usize) }.to_vec();
    unsafe { ffi::tsac_free_buffer(ptr as *mut std::os::raw::c_void) };
    Some(data)
}

fn load_alpha(ctx: *mut ffi::TSACContext, name: &str) -> Result<Array1<f32>> {
    let data = load_f32(ctx, name).with_context(|| format!("missing {name}"))?;
    Ok(Array1::from_vec(data))
}

/// Load a conv1d as an `rlx-dac` `WnConv1d` (final weights, no re-normalization).
fn load_conv1d(
    ctx: *mut ffi::TSACContext,
    prefix: &str,
    stride: usize,
    pad: usize,
    dilation: usize,
    same_length: bool,
) -> Result<Option<WnConv1d>> {
    let weight = if std_weightnorm() {
        match std_conv1d_weight(ctx, prefix) {
            Some(w) => w,
            None => return Ok(None),
        }
    } else {
        let Some((w, co, ci, k, _is_ct)) = load_conv_dequant(ctx, prefix) else {
            return Ok(None);
        };
        Array3::from_shape_vec((co, ci, k), w)
            .with_context(|| format!("{prefix} conv weight shape [{co},{ci},{k}]"))?
    };
    let bias = load_f32(ctx, &format!("{prefix}.bias")).map(Array1::from_vec);
    Ok(Some(WnConv1d {
        weight,
        bias,
        stride,
        pad,
        dilation,
        same_length,
    }))
}

/// Load a conv-transpose as an `rlx-dac` `WnConvTranspose1d`.
///
/// The C `dequant_weights` returns conv-transpose weights in `[Co][K][Ci]` flat
/// order (`buf[oc*K*Ci + j*Ci + ic]`), which — matching the C `convt1d_s`
/// scatter `out[oc][ii*s - pad + j] += in[ic][ii] * buf[oc][j][ic]` — is exactly
/// PyTorch's `ConvTranspose1d` weight `W[ic][oc][j]`. rlx-dac expects that same
/// `[c_in, c_out, k]` layout, so we reindex `buf → W[ic][oc][j]`.
fn load_conv_transpose1d(
    ctx: *mut ffi::TSACContext,
    prefix: &str,
    faithful: bool,
) -> Result<WnConvTranspose1d> {
    let (weight, k) = if std_weightnorm() {
        let w = std_convt_weight(ctx, prefix).with_context(|| format!("missing {prefix}"))?;
        let k = w.dim().2;
        (w, k)
    } else {
        let (buf, co, ci, k, is_ct) =
            load_conv_dequant(ctx, prefix).with_context(|| format!("missing {prefix}"))?;
        debug_assert!(is_ct, "{prefix} expected conv-transpose");
        let mut w = vec![0f32; ci * co * k];
        for oc in 0..co {
            for j in 0..k {
                for ic in 0..ci {
                    // src: [Co][K][Ci]; dst: [Ci][Co][K]
                    w[(ic * co + oc) * k + j] = buf[(oc * k + j) * ci + ic];
                }
            }
        }
        let w = Array3::from_shape_vec((ci, co, k), w)
            .with_context(|| format!("{prefix} convT weight shape [{ci},{co},{k}]"))?;
        (w, k)
    };
    let bias = load_f32(ctx, &format!("{prefix}.bias")).map(Array1::from_vec);
    let stride = k / 2;
    let pad = stride.div_ceil(2);
    // Faithful: C's `convt1d_s` centers the kernel at `P = K/2 = stride` (a
    // stride/2 right-shift of the crop). Standard DAC uses 0.
    let out_offset = if faithful { pad } else { 0 };
    Ok(WnConvTranspose1d {
        weight,
        bias,
        stride,
        pad,
        out_offset,
    })
}

/// Load a DAC residual unit at `{rp}` = `…block.N.block`: snake → dilated conv(K=7)
/// → snake → conv(K=1) → skip. Shared by encoder and decoder blocks.
fn load_residual_unit(ctx: *mut ffi::TSACContext, rp: &str, dil: usize) -> Result<ResidualUnit> {
    let pad = ((7 - 1) * dil) / 2;
    let snake1 = load_alpha(ctx, &format!("{rp}.0.alpha"))?;
    let conv1 = load_conv1d(ctx, &format!("{rp}.1"), 1, pad, dil, true)?
        .with_context(|| format!("missing {rp}.1"))?;
    let snake2 = load_alpha(ctx, &format!("{rp}.2.alpha"))?;
    let conv2 = load_conv1d(ctx, &format!("{rp}.3"), 1, 0, 1, true)?
        .with_context(|| format!("missing {rp}.3"))?;
    Ok(ResidualUnit::from_parts(snake1, conv1, snake2, conv2))
}

fn units3(v: Vec<ResidualUnit>, what: &str) -> Result<[ResidualUnit; 3]> {
    v.try_into()
        .map_err(|_| anyhow::anyhow!("{what}: expected 3 residual units"))
}

fn load_decoder_block(
    ctx: *mut ffi::TSACContext,
    prefix: &str,
    faithful: bool,
) -> Result<(DecoderBlock, usize)> {
    let snake_alpha = load_alpha(ctx, &format!("{prefix}.block.0.alpha"))?;
    let upsample = load_conv_transpose1d(ctx, &format!("{prefix}.block.1"), faithful)?;
    let stride = upsample.stride;
    let mut units: Vec<ResidualUnit> = Vec::with_capacity(3);
    for (u, &dil) in RES_DILATIONS.iter().enumerate() {
        units.push(load_residual_unit(
            ctx,
            &format!("{prefix}.block.{}.block", u + 2),
            dil,
        )?);
    }
    Ok((
        DecoderBlock::from_parts(snake_alpha, upsample, units3(units, prefix)?),
        stride,
    ))
}

/// DAC encoder block: 3 residual units (dil 1,3,9) → snake → strided downsample.
fn load_encoder_block(ctx: *mut ffi::TSACContext, prefix: &str) -> Result<(EncoderBlock, usize)> {
    let mut units: Vec<ResidualUnit> = Vec::with_capacity(3);
    for (u, &dil) in RES_DILATIONS.iter().enumerate() {
        units.push(load_residual_unit(
            ctx,
            &format!("{prefix}.block.{u}.block"),
            dil,
        )?);
    }
    let snake = load_alpha(ctx, &format!("{prefix}.block.3.alpha"))?;
    // Downsample conv (block.4): stride = K/2, standard DAC pad = ceil(stride/2).
    let dprefix = format!("{prefix}.block.4");
    let weight = if std_weightnorm() {
        std_conv1d_weight(ctx, &dprefix).with_context(|| format!("missing {dprefix}"))?
    } else {
        let (w, co, ci, k, _) =
            load_conv_dequant(ctx, &dprefix).with_context(|| format!("missing {dprefix}"))?;
        Array3::from_shape_vec((co, ci, k), w)?
    };
    let stride = weight.dim().2 / 2;
    let pad = stride.div_ceil(2);
    let bias = load_f32(ctx, &format!("{dprefix}.bias")).map(Array1::from_vec);
    let down = WnConv1d {
        weight,
        bias,
        stride,
        pad,
        dilation: 1,
        same_length: false,
    };
    Ok((
        EncoderBlock::from_parts(units3(units, prefix)?, snake, down),
        stride,
    ))
}

/// Load the full DAC encoder. Returns `(encoder, in_channels, total_downscale)`.
fn load_encoder(ctx: *mut ffi::TSACContext) -> Result<(Encoder, usize, usize)> {
    let stem =
        load_conv1d(ctx, "encoder.block.0", 1, 3, 1, true)?.context("missing encoder.block.0")?;
    let in_channels = stem.weight.dim().1;
    let mut blocks = Vec::with_capacity(4);
    let mut down = 1usize;
    for i in 1..=4 {
        let (b, s) = load_encoder_block(ctx, &format!("encoder.block.{i}"))?;
        down *= s;
        blocks.push(b);
    }
    let snake = load_alpha(ctx, "encoder.block.5.alpha")?;
    // Head conv (block.6): K=3, same-length pad=1.
    let head =
        load_conv1d(ctx, "encoder.block.6", 1, 1, 1, true)?.context("missing encoder.block.6")?;
    Ok((
        Encoder::from_parts(stem, blocks, snake, head),
        in_channels,
        down,
    ))
}
