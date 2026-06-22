//! rlx-ir graph path: builds the Snake HiFi-GAN vocoder as an HIR graph compiled
//! per device, so it runs on every RLX backend (CPU/Metal/MLX/wgpu validated;
//! CUDA/ROCm wired). Mirrors the host-eager vocoder op-for-op (validated against it).
//!
//! Backend-portability choices:
//! - 1-D convs are NCHW convs with time in the H axis (`[N,C,L,1]`, kernel `[K,1]`).
//! - Transposed convs are rewritten as zero-insertion + a regular conv, so no
//!   backend needs a native `ConvTranspose` kernel (several lack one).
//! - Snake α = `exp(log_alpha).clamp(1e-4,100)` is folded to a constant tensor.

#![cfg(feature = "rlx-graph")]

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use ndarray::Array2;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::Activation;
use rlx_ir::{DType, HirGraphExt, HirModule, Op, Shape};
use rlx_runtime::{AotCache, CompileOptions, Device};

use crate::config::VocoderConfig;
use crate::weights::Weights;

pub fn device_from_str(s: &str) -> Device {
    match s.to_ascii_lowercase().as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        _ => Device::Cpu,
    }
}

fn get_padding(k: usize, d: usize) -> usize {
    (k * d - d) / 2
}

/// Builds the vocoder HIR + the param tensors (already shaped 4-D / folded).
struct VocoderBuilder<'a> {
    w: &'a Weights,
    cfg: &'a VocoderConfig,
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
}

impl<'a> VocoderBuilder<'a> {
    fn m(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }

    /// Bind a weight tensor under `name` with shape `shape` (data taken verbatim).
    fn param(&mut self, name: &str, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let id = HirMut::new(&mut self.hir).param(name, Shape::new(shape, DType::F32));
        self.params.insert(name.to_string(), data);
        id
    }

    /// Bind a conv weight `[c_out, c_in, k]`, reshaped to NCHW 4-D `[c_out, c_in, k, 1]`.
    fn conv_w(&mut self, key: &str) -> HirNodeId {
        let (data, sh) = self.w.get(key).expect(key).clone();
        self.param(key, data, &[sh[0], sh[1], sh[2], 1])
    }

    fn bias(&mut self, key: &str, c_out: usize) -> HirNodeId {
        let data = self.w.data(key).expect(key).to_vec();
        self.param(key, data, &[1, c_out, 1, 1])
    }

    /// Snake α constant = exp(log_alpha).clamp(1e-4,100), shape [1,C,1,1].
    fn alpha(&mut self, key: &str, c: usize) -> HirNodeId {
        let data: Vec<f32> = self
            .w
            .data(key)
            .expect(key)
            .iter()
            .map(|v| v.exp().clamp(1e-4, 100.0))
            .collect();
        self.param(&format!("{key}.alpha"), data, &[1, c, 1, 1])
    }

    fn add_bias(&mut self, x: HirNodeId, b: HirNodeId) -> HirNodeId {
        self.m().add(x, b)
    }

    fn snake(&mut self, x: HirNodeId, alpha: HirNodeId) -> HirNodeId {
        let mut m = self.m();
        let ax = m.mul(x, alpha);
        let sh = m.shape(ax).clone();
        let s = m.activation(Activation::Sin, ax, sh);
        let s2 = m.mul(s, s);
        let q = m.div(s2, alpha);
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

    /// Transposed conv expressed as zero-insertion + a regular Conv, so it runs
    /// on every backend (several backends lack a native ConvTranspose kernel).
    /// Exact identity: ConvTranspose1d(x,W,stride,pad) == Conv1d(upsample0(x,stride),
    /// W', pad'=k-1-pad) with W'[oc,ic,kk] = W[ic,oc,K-1-kk].
    fn conv_transpose1d(
        &mut self,
        x: HirNodeId,
        ups_key: &str,
        c_in: usize,
        c_out: usize,
        k: usize,
        stride: usize,
        pad: usize,
        t_in: usize,
    ) -> (HirNodeId, usize) {
        // reversed/transposed weight [c_out, c_in, k] from PyTorch [c_in, c_out, k]
        let raw = self.w.data(&format!("{ups_key}.weight")).expect(ups_key);
        let mut wrev = vec![0f32; c_out * c_in * k];
        for ic in 0..c_in {
            for oc in 0..c_out {
                for kk in 0..k {
                    wrev[oc * c_in * k + ic * k + kk] = raw[ic * c_out * k + oc * k + (k - 1 - kk)];
                }
            }
        }
        let wid = self.param(
            &format!("{ups_key}.weight.conv"),
            wrev,
            &[c_out, c_in, k, 1],
        );

        // zero-insert (O(T) memory): [1,c,T,1] ++ zeros[1,c,T,s-1] along W → [1,c,T,s]
        // → reshape [1,c,T*s,1] (sample at t*s, zeros after) → narrow to (T-1)*s+1.
        let l_up = (t_in - 1) * stride + 1;
        let up = if stride > 1 {
            let zeros = self.param(
                &format!("{ups_key}.__zeros"),
                vec![0f32; c_in * t_in * (stride - 1)],
                &[1, c_in, t_in, stride - 1],
            );
            let mut m = self.m();
            let cat = m.concat_(vec![x, zeros], 3); // [1,c,T,s]
            let flat = m.reshape_(cat, vec![1, c_in as i64, (t_in * stride) as i64, 1]);
            m.narrow_(flat, 2, 0, l_up)
        } else {
            x
        };
        // regular conv with pad' = k-1-pad
        let pad2 = k - 1 - pad;
        self.conv1d(up, wid, c_out, k, 1, pad2, 1, 1, l_up)
    }

    fn resblock(
        &mut self,
        mut x: HirNodeId,
        idx: usize,
        ch: usize,
        k: usize,
        dils: &[usize],
        t: usize,
    ) -> HirNodeId {
        for (j, &dl) in dils.iter().enumerate() {
            let a1 = self.alpha(&format!("resblocks.{idx}.acts1.{j}.log_alpha"), ch);
            let y = self.snake(x, a1);
            let w1 = self.conv_w(&format!("resblocks.{idx}.convs1.{j}.weight"));
            let (y, _) = self.conv1d(y, w1, ch, k, 1, get_padding(k, dl), dl, 1, t);
            let b1 = self.bias(&format!("resblocks.{idx}.convs1.{j}.bias"), ch);
            let y = self.add_bias(y, b1);
            let a2 = self.alpha(&format!("resblocks.{idx}.acts2.{j}.log_alpha"), ch);
            let y = self.snake(y, a2);
            let w2 = self.conv_w(&format!("resblocks.{idx}.convs2.{j}.weight"));
            let (y, _) = self.conv1d(y, w2, ch, k, 1, get_padding(k, 1), 1, 1, t);
            let b2 = self.bias(&format!("resblocks.{idx}.convs2.{j}.bias"), ch);
            let y = self.add_bias(y, b2);
            x = self.m().add(x, y);
        }
        x
    }

    fn build(mut self, t_frames: usize) -> (HirModule, HashMap<String, Vec<f32>>) {
        let init = self.cfg.upsample_initial_channel;
        let num_up = self.cfg.upsample_rates.len();
        let num_k = self.cfg.resblock_kernel_sizes.len();

        let mel = HirMut::new(&mut self.hir).input(
            "mel",
            Shape::new(&[1, self.cfg.num_mels, t_frames, 1], DType::F32),
        );

        // conv_pre
        let wpre = self.conv_w("conv_pre.weight");
        let (mut x, mut t) = self.conv1d(mel, wpre, init, 7, 1, 3, 1, 1, t_frames);
        let bpre = self.bias("conv_pre.bias", init);
        x = self.add_bias(x, bpre);

        for i in 0..num_up {
            let in_ch = init >> i;
            let out_ch = init >> (i + 1);
            let k = self.cfg.upsample_kernel_sizes[i];
            let rate = self.cfg.upsample_rates[i];
            let au = self.alpha(&format!("up_acts.{i}.log_alpha"), in_ch);
            x = self.snake(x, au);
            let (xt, t_new) = self.conv_transpose1d(
                x,
                &format!("ups.{i}"),
                in_ch,
                out_ch,
                k,
                rate,
                (k - rate) / 2,
                t,
            );
            let bup = self.bias(&format!("ups.{i}.bias"), out_ch);
            x = self.add_bias(xt, bup);
            t = t_new;

            // sum of resblocks / num_kernels
            let mut acc: Option<HirNodeId> = None;
            for j in 0..num_k {
                let idx = i * num_k + j;
                let kk = self.cfg.resblock_kernel_sizes[j];
                let dils = self.cfg.resblock_dilation_sizes[j].clone();
                let r = self.resblock(x, idx, out_ch, kk, &dils, t);
                acc = Some(match acc {
                    None => r,
                    Some(a) => self.m().add(a, r),
                });
            }
            let summed = acc.unwrap();
            // divide by num_kernels (constant)
            let inv = self.param(
                &format!("__inv_k_{i}"),
                vec![1.0 / num_k as f32],
                &[1, 1, 1, 1],
            );
            x = self.m().mul(summed, inv);
        }

        let final_ch = init >> num_up;
        let ap = self.alpha("post_act.log_alpha", final_ch);
        x = self.snake(x, ap);
        let wpost = self.conv_w("conv_post.weight");
        let (xp, _) = self.conv1d(x, wpost, 1, 7, 1, 3, 1, 1, t);
        let bpost = self.bias("conv_post.bias", 1);
        let xp = self.add_bias(xp, bpost);
        let out = self.m().tanh(xp);
        HirMut::new(&mut self.hir).set_outputs(vec![out]);
        (self.hir, self.params)
    }
}

/// Compiled vocoder graph for a fixed frame length on a chosen device.
pub struct VocoderGraph {
    compiled: rlx_runtime::CompiledGraph,
    t_frames: usize,
    device: Device,
}

fn aot_root() -> PathBuf {
    std::env::var("RLX_INFLECT_NANO_AOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("rlx-inflect-nano-aot"))
}

impl VocoderGraph {
    pub fn compile(
        w: &Weights,
        cfg: &VocoderConfig,
        t_frames: usize,
        device: Device,
    ) -> Result<Self> {
        if !rlx_runtime::is_available(device) {
            return Err(anyhow!("device {device:?} not available in this build"));
        }
        let builder = VocoderBuilder {
            w,
            cfg,
            hir: HirModule::new("inflect_nano_vocoder"),
            params: HashMap::new(),
        };
        let (hir, params) = builder.build(t_frames);
        // `v3` = concat/reshape zero-insert transposed-conv graph (O(T) memory);
        // bump on structural change so the AOT cache never serves stale LIR.
        let key = format!("inflect_nano_vocoder_v3_{device:?}_t{t_frames}");
        let cache = AotCache::new(aot_root());
        let mut compiled = cache
            .compile_hir_cached(&key, device, hir, &CompileOptions::default())
            .map_err(|e| anyhow!("compile vocoder graph: {e}"))?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        compiled.finalize_params();
        Ok(Self {
            compiled,
            t_frames,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// `mel: [80, T]` (channel-major) → waveform `[T*hop]`.
    pub fn forward(&mut self, mel: &Array2<f32>) -> Result<Vec<f32>> {
        let (c, t) = mel.dim();
        if t != self.t_frames {
            return Err(anyhow!("mel frames {t} != compiled {}", self.t_frames));
        }
        // mel is already [C, T] row-major == NCHW [1,C,T,1] flat.
        let flat: Vec<f32> = mel.iter().copied().collect();
        debug_assert_eq!(flat.len(), c * t);
        let outs = self.compiled.run(&[("mel", &flat)]);
        Ok(outs.into_iter().next().unwrap_or_default())
    }
}
