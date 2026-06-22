//! SNAC decoder on native RLX backends (`Device::Ane` / CoreML, or Metal).
//!
//! Quantizer (`from_codes`) stays on the host eager path; the conv decoder stack
//! is compiled per latent length via `rlx-ir` + `rlx-runtime`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Result, anyhow, ensure};
use ndarray::{Array2, ArrayView3};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::Activation;
use rlx_ir::{DType, HirGraphExt, HirModule, Op, Shape};
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, Device, is_available};

use super::eager::NoiseState;
use super::eager::{
    DecoderBlockWeights, ResidualUnitWeights, SnacDecoder, SnacDecoderInner, audio_len_for_latent,
    decoder_block_times, from_codes_inner,
};

const GRAPH_VERSION: &str = "snac_decoder_v1";

const LATENT_BUCKETS: &[usize] = &[16, 32, 48, 64, 96, 128, 192, 256, 384, 512];

fn bucket_latent(t: usize) -> usize {
    LATENT_BUCKETS
        .iter()
        .copied()
        .find(|&b| b >= t)
        .unwrap_or(512)
}

fn aot_root() -> PathBuf {
    std::env::var("ORPHEUS_SNAC_AOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("rlx-orpheus-snac-aot"))
}

struct GraphBuilder<'a> {
    inner: &'a SnacDecoderInner,
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
}

impl<'a> GraphBuilder<'a> {
    fn new(inner: &'a SnacDecoderInner) -> Self {
        Self {
            inner,
            hir: HirModule::new("orpheus_snac_decoder"),
            params: HashMap::new(),
        }
    }

    fn m(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }

    fn param(&mut self, name: &str, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let id = HirMut::new(&mut self.hir).param(name, Shape::new(shape, DType::F32));
        self.params.insert(name.to_string(), data);
        id
    }

    fn conv_w3(&mut self, name: &str, w: ArrayView3<f32>) -> HirNodeId {
        let (c0, c1, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
        self.param(name, w.iter().copied().collect(), &[c0, c1, k, 1])
    }

    fn bias1d(&mut self, name: &str, b: &[f32]) -> HirNodeId {
        self.param(name, b.to_vec(), &[1, b.len(), 1, 1])
    }

    fn alpha1d(&mut self, name: &str, alpha: &[f32]) -> HirNodeId {
        self.param(name, alpha.to_vec(), &[1, alpha.len(), 1, 1])
    }

    fn snake(&mut self, x: HirNodeId, alpha: HirNodeId, tag: &str) -> HirNodeId {
        let eps = self.param(&format!("{tag}.snake_eps"), vec![1e-9f32], &[1, 1, 1, 1]);
        let mut m = self.m();
        let ax = m.mul(x, alpha);
        let sh = m.shape(ax).clone();
        let s = m.activation(Activation::Sin, ax, sh);
        let s2 = m.mul(s, s);
        let a_safe = m.add(alpha, eps);
        let q = m.div(s2, a_safe);
        m.add(x, q)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1d(
        &mut self,
        x: HirNodeId,
        w: HirNodeId,
        c_out: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dil: usize,
        groups: usize,
        t_in: usize,
    ) -> (HirNodeId, usize) {
        let t_out = (t_in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        let out = Shape::new(&[1, c_out, t_out, 1], DType::F32);
        let id = self.m().add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![stride, 1],
                padding: vec![pad, 0],
                dilation: vec![dil, 1],
                groups,
            },
            vec![x, w],
            out,
        );
        (id, t_out)
    }

    fn conv_transpose1d(
        &mut self,
        x: HirNodeId,
        w: ArrayView3<f32>,
        name: &str,
        c_in: usize,
        c_out: usize,
        k: usize,
        stride: usize,
        pad: usize,
        t_in: usize,
    ) -> (HirNodeId, usize) {
        let mut wrev = vec![0f32; c_out * c_in * k];
        for ic in 0..c_in {
            for oc in 0..c_out {
                for kk in 0..k {
                    wrev[oc * c_in * k + ic * k + kk] = w[[ic, oc, k - 1 - kk]];
                }
            }
        }
        let wid = self.param(name, wrev, &[c_out, c_in, k, 1]);
        let l_up = (t_in - 1) * stride + 1;
        let up = if stride > 1 {
            let zeros = self.param(
                &format!("{name}.__zeros"),
                vec![0f32; c_in * t_in * (stride - 1)],
                &[1, c_in, t_in, stride - 1],
            );
            let mut m = self.m();
            let cat = m.concat_(vec![x, zeros], 3);
            let flat = m.reshape_(cat, vec![1, c_in as i64, (t_in * stride) as i64, 1]);
            m.narrow_(flat, 2, 0, l_up)
        } else {
            x
        };
        let pad2 = k - 1 - pad;
        self.conv1d(up, wid, c_out, k, 1, pad2, 1, 1, l_up)
    }

    fn noise_block(
        &mut self,
        x: HirNodeId,
        w: HirNodeId,
        noise: HirNodeId,
        c: usize,
        t: usize,
    ) -> HirNodeId {
        let (h, _) = self.conv1d(x, w, c, 1, 1, 0, 1, 1, t);
        let h = self.m().mul(h, noise);
        self.m().add(x, h)
    }

    fn residual_unit(
        &mut self,
        x: HirNodeId,
        ru: &ResidualUnitWeights,
        prefix: &str,
        groups: usize,
        t: usize,
    ) -> HirNodeId {
        let a1 = self.alpha1d(
            &format!("{prefix}.snake1"),
            ru.snake1_alpha.as_slice().unwrap(),
        );
        let y = self.snake(x, a1, &format!("{prefix}.ru.snake1"));
        let w1 = self.conv_w3(&format!("{prefix}.conv1"), ru.conv1_w.view());
        let (y, _) = self.conv1d(
            y,
            w1,
            ru.conv1_w.shape()[0],
            7,
            1,
            ru.conv1_pad,
            ru.conv1_dilation,
            groups,
            t,
        );
        let y = self.add_bias(
            y,
            &format!("{prefix}.conv1.bias"),
            ru.conv1_b.as_slice().unwrap(),
        );
        let a2 = self.alpha1d(
            &format!("{prefix}.snake2"),
            ru.snake2_alpha.as_slice().unwrap(),
        );
        let y = self.snake(y, a2, &format!("{prefix}.ru.snake2"));
        let w2 = self.conv_w3(&format!("{prefix}.conv2"), ru.conv2_w.view());
        let (y, _) = self.conv1d(y, w2, ru.conv2_w.shape()[0], 1, 1, 0, 1, 1, t);
        let y = self.add_bias(
            y,
            &format!("{prefix}.conv2.bias"),
            ru.conv2_b.as_slice().unwrap(),
        );
        self.m().add(x, y)
    }

    fn add_bias(&mut self, x: HirNodeId, name: &str, b: &[f32]) -> HirNodeId {
        let bias = self.bias1d(name, b);
        self.m().add(x, bias)
    }

    fn decoder_block(
        &mut self,
        x: HirNodeId,
        block: &DecoderBlockWeights,
        prefix: &str,
        noise_in: HirNodeId,
        depthwise: bool,
        t_in: usize,
    ) -> (HirNodeId, usize) {
        let a = self.alpha1d(
            &format!("{prefix}.snake"),
            block.snake_alpha.as_slice().unwrap(),
        );
        let h = self.snake(x, a, &format!("{prefix}.block.snake"));
        let k = 2 * block.stride;
        let padding = block.stride.div_ceil(2);
        let (mut h, t) = self.conv_transpose1d(
            h,
            block.upsample_w.view(),
            &format!("{prefix}.upsample"),
            block.upsample_w.shape()[0],
            block.upsample_w.shape()[1],
            k,
            block.stride,
            padding,
            t_in,
        );
        h = self.add_bias(
            h,
            &format!("{prefix}.upsample.bias"),
            block.upsample_b.as_slice().unwrap(),
        );
        let nw = self.conv_w3(&format!("{prefix}.noise"), block.noise_w.view());
        let c = block.noise_w.shape()[0];
        if self.inner.config.noise {
            h = self.noise_block(h, nw, noise_in, c, t);
        }
        let ru_groups = if depthwise { c } else { 1 };
        for (i, ru) in block.residual_units.iter().enumerate() {
            h = self.residual_unit(h, ru, &format!("{prefix}.ru{i}"), ru_groups, t);
        }
        (h, t)
    }

    fn build(mut self, t_latent: usize) -> (HirModule, HashMap<String, Vec<f32>>) {
        let latent = self.inner.config.latent_dim();
        let decoder_dim = self.inner.config.decoder_dim;
        let depthwise = self.inner.config.depthwise;

        let z_q = HirMut::new(&mut self.hir)
            .input("z_q", Shape::new(&[1, latent, t_latent, 1], DType::F32));
        let mut noise_inputs = Vec::new();
        for i in 0..self.inner.blocks.len() {
            let t_noise = decoder_block_times(&self.inner.config.decoder_rates, t_latent)[i];
            let n = HirMut::new(&mut self.hir).input(
                format!("noise_{i}"),
                Shape::new(&[1, 1, t_noise, 1], DType::F32),
            );
            noise_inputs.push(n);
        }

        let mut x = z_q;
        let mut t = t_latent;
        if depthwise {
            let w = self.conv_w3("init_dw", self.inner.init_dw_w.view());
            let (x0, t0) = self.conv1d(x, w, latent, 7, 1, 3, 1, latent, t);
            x = self.add_bias(x0, "init_dw.bias", self.inner.init_dw_b.as_slice().unwrap());
            t = t0;
        } else {
            let w = self.conv_w3("init_dw", self.inner.init_dw_w.view());
            let (x0, t0) = self.conv1d(x, w, latent, 7, 1, 3, 1, 1, t);
            x = self.add_bias(x0, "init_dw.bias", self.inner.init_dw_b.as_slice().unwrap());
            t = t0;
        }
        let pw = self.conv_w3("init_pw", self.inner.init_pw_w.view());
        let (x0, t0) = self.conv1d(x, pw, decoder_dim, 1, 1, 0, 1, 1, t);
        x = self.add_bias(x0, "init_pw.bias", self.inner.init_pw_b.as_slice().unwrap());
        t = t0;

        for (i, block) in self.inner.blocks.iter().enumerate() {
            let (x0, t0) = self.decoder_block(
                x,
                block,
                &format!("block{i}"),
                noise_inputs[i],
                depthwise,
                t,
            );
            x = x0;
            t = t0;
        }

        let fa = self.alpha1d(
            "final.snake",
            self.inner.final_snake_alpha.as_slice().unwrap(),
        );
        x = self.snake(x, fa, "final.block.snake");
        let fw = self.conv_w3("final.conv", self.inner.final_conv_w.view());
        let (x_conv, _) = self.conv1d(x, fw, 1, 7, 1, 3, 1, 1, t);
        x = self.add_bias(
            x_conv,
            "final.conv.bias",
            self.inner.final_conv_b.as_slice().unwrap(),
        );
        let out = self.m().tanh(x);
        HirMut::new(&mut self.hir).set_outputs(vec![out]);
        (self.hir, self.params)
    }
}

struct CompiledSlot {
    compiled: CompiledGraph,
}

/// SNAC decode via native RLX ([`Device::Ane`] CoreML by default).
///
/// RVQ quantizer runs on the host; the conv decoder stack is compiled per latent
/// length bucket and cached under [`aot_root`] (`ORPHEUS_SNAC_AOT`).
pub struct RlxSnacDecoder {
    eager: SnacDecoder,
    device: Device,
    cache: Mutex<HashMap<usize, CompiledSlot>>,
}

impl RlxSnacDecoder {
    pub fn load(weights: &Path, device: Device) -> Result<Self> {
        ensure!(
            is_available(device),
            "SNAC RLX backend {device:?} is not available in this build \
             (rebuild with --features coreml,metal)"
        );
        Ok(Self {
            eager: SnacDecoder::from_file(weights)?,
            device,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn decode_codes(
        &self,
        codes_0: &[i32],
        codes_1: &[i32],
        codes_2: &[i32],
    ) -> Result<Vec<f32>> {
        if codes_0.is_empty() {
            return Ok(Vec::new());
        }
        let z_q = from_codes_inner(self.eager.inner(), codes_0, codes_1, codes_2)?;
        self.decode_latent(&z_q)
    }

    pub fn decode_orpheus_frames(&self, frame_tokens: &[i32]) -> Result<Vec<f32>> {
        let (c0, c1, c2) = crate::tokens::pack_orpheus_codes(frame_tokens)
            .ok_or_else(|| anyhow!("invalid Orpheus frame token layout"))?;
        self.decode_codes(&c0, &c1, &c2)
    }

    fn decode_latent(&self, z_q: &Array2<f32>) -> Result<Vec<f32>> {
        let t_actual = z_q.shape()[1];
        let t_bucket = bucket_latent(t_actual);
        let audio_len = audio_len_for_latent(&self.eager.inner().config.decoder_rates, t_actual);
        let z_flat = pad_latent(z_q, t_bucket);
        let noise = self.noise_inputs(t_bucket)?;
        let mut cache = self.cache.lock().expect("snac compile cache");
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(t_bucket) {
            let compiled = self.compile_graph(t_bucket)?;
            e.insert(CompiledSlot { compiled });
        }
        let slot = cache.get_mut(&t_bucket).expect("compiled slot");
        let noise_names: Vec<String> = (0..noise.len()).map(|i| format!("noise_{i}")).collect();
        let mut feed: Vec<(&str, &[f32])> = Vec::with_capacity(1 + noise.len());
        feed.push(("z_q", z_flat.as_slice()));
        for (i, plane) in noise.iter().enumerate() {
            feed.push((&noise_names[i], plane.as_slice()));
        }
        let outs = slot.compiled.run(&feed);
        let wav = outs.into_iter().next().unwrap_or_default();
        Ok(wav.into_iter().take(audio_len).collect())
    }

    fn compile_graph(&self, t_latent: usize) -> Result<CompiledGraph> {
        let inner = self.eager.inner();
        let builder = GraphBuilder::new(inner);
        let (hir, params) = builder.build(t_latent);
        let key = format!("{GRAPH_VERSION}_{:?}_t{t_latent}", self.device);
        let aot = AotCache::new(aot_root());
        let mut compiled = aot
            .compile_hir_cached(&key, self.device, hir, &CompileOptions::default())
            .map_err(|e| anyhow!("compile SNAC graph t={t_latent}: {e}"))?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        compiled.finalize_params();
        Ok(compiled)
    }

    fn noise_inputs(&self, t_latent: usize) -> Result<Vec<Vec<f32>>> {
        let times = decoder_block_times(&self.eager.inner().config.decoder_rates, t_latent);
        let mut state = NoiseState::seeded(42);
        Ok(times
            .iter()
            .map(|&t| state.next_plane(t).to_vec())
            .collect())
    }
}

fn pad_latent(z_q: &Array2<f32>, t_bucket: usize) -> Vec<f32> {
    let (c, t) = (z_q.shape()[0], z_q.shape()[1]);
    let mut flat = vec![0f32; c * t_bucket];
    for ci in 0..c {
        for ti in 0..t.min(t_bucket) {
            flat[ci * t_bucket + ti] = z_q[[ci, ti]];
        }
    }
    flat
}
