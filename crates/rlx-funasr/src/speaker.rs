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

//! **CAM++** speaker embedding (192-d).
//!
//! A 2-D `FCM` front-end, a context-aware-masking densely-connected TDNN trunk,
//! statistics pooling, and a final dense layer produce a fixed 192-d speaker
//! vector. The full graph runs on the selected RLX device; per-utterance
//! feature mean-normalization runs on the host.
//!
//! Validated against the real `funasr/campplus` checkpoint: the embedding has
//! cosine similarity ~0.987 with the reference (`dump-keys` confirmed the 937
//! tensors). The CAM context is `global_time_mean + segment_avg_pool`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::built_from_hir;
use rlx_core::weight_map::WeightMap;
use rlx_ir::Shape;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirNodeId};
use rlx_runtime::Device;

use crate::cache::GraphCache;
use crate::config::CamPlusConfig;
use crate::frontend::{FrontendConfig, WavFrontend};
use crate::sanm::Graph;
use crate::weights::RefSource;

/// A loaded CAM++ speaker model.
pub struct CamPlus {
    cfg: CamPlusConfig,
    weights: WeightMap,
    frontend: WavFrontend,
    device: Device,
    cache: GraphCache,
}

impl CamPlus {
    /// Open a CAM++ model directory.
    pub fn open(dir: &Path, device: Device) -> Result<Self> {
        let cfg = CamPlusConfig::default();
        let weights = crate::weights::load_dir(dir)?;
        let fe = FrontendConfig {
            n_mels: cfg.feat_dim,
            lfr_m: 1,
            lfr_n: 1,
            ..FrontendConfig::default()
        };
        let frontend = WavFrontend::new(fe, None);
        Ok(Self {
            cfg,
            weights,
            frontend,
            device,
            cache: GraphCache::new(4),
        })
    }

    /// Construct from an in-memory config + weights (used by tests).
    pub fn from_parts(cfg: CamPlusConfig, weights: WeightMap, device: Device) -> Self {
        let fe = FrontendConfig {
            n_mels: cfg.feat_dim,
            lfr_m: 1,
            lfr_n: 1,
            ..FrontendConfig::default()
        };
        let frontend = WavFrontend::new(fe, None);
        Self {
            cfg,
            weights,
            frontend,
            device,
            cache: GraphCache::new(4),
        }
    }

    /// The model configuration.
    pub fn config(&self) -> &CamPlusConfig {
        &self.cfg
    }

    /// Run the network over fbank features `[t, feat_dim]`; returns the 192-d
    /// embedding.
    pub fn run_embedding(&self, feats: &[f32], t: usize) -> Result<Vec<f32>> {
        let fd = self.cfg.feat_dim;
        ensure!(feats.len() == t * fd, "feature length mismatch");
        let cfg = &self.cfg;
        let weights = &self.weights;
        let build = || -> anyhow::Result<rlx_flow::BuiltModel> {
            let mut params = HashMap::new();
            let mut hir = HirModule::new("campplus").with_fusion_policy(FusionPolicy::Direct);
            {
                let mut src = RefSource(weights);
                let mut g = Graph::new(&mut hir, &mut params, &mut src);
                let x = g.input("feats", &[1, t, fd]);
                let emb = build_campplus(&mut g, x, cfg, t)?;
                g.set_output(emb);
            }
            built_from_hir(hir, params)
        };
        self.cache
            .run(t as u64, self.device, build, &[("feats", feats)])?
            .into_iter()
            .next()
            .context("campplus produced no output")
    }

    /// Compute the speaker embedding from mono 16 kHz PCM.
    pub fn embedding(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let mut fb = self.frontend.fbank(pcm);
        ensure!(fb.n_frames > 0, "audio too short for CAM++");
        mean_normalize(&mut fb.data, fb.n_frames, fb.feat_dim);
        self.run_embedding(&fb.data, fb.n_frames)
    }
}

/// Per-feature mean subtraction over time (CAM++ input normalization).
fn mean_normalize(data: &mut [f32], t: usize, d: usize) {
    for c in 0..d {
        let mut m = 0.0f32;
        for ti in 0..t {
            m += data[ti * d + c];
        }
        m /= t as f32;
        for ti in 0..t {
            data[ti * d + c] -= m;
        }
    }
}

/// Build the full CAM++ graph → `[1, 192]`.
fn build_campplus(g: &mut Graph, x: HirNodeId, cfg: &CamPlusConfig, t: usize) -> Result<HirNodeId> {
    let eps = cfg.bn_eps;
    // [1,T,F] -> [1,F,T] -> [1,1,F,T]
    let xt = g.g().transpose_(x, vec![0, 2, 1]);
    let mut h2 = g
        .g()
        .reshape_(xt, vec![1, 1, cfg.feat_dim as i64, t as i64]);
    let (mut hf, mut wt) = (cfg.feat_dim, t);

    // FCM: conv1+bn1+relu
    let (n, ho, wo) = conv2d(g, h2, "head.conv1.weight", 1, 32, 3, 3, 1, 1, 1, 1, hf, wt)?;
    h2 = g.batchnorm2d(n, "head.bn1", 32, eps)?;
    h2 = g.g().relu(h2);
    hf = ho;
    wt = wo;
    // layer1 (2 res blocks, first stride (2,1) on freq)
    let (n, hf1, wt1) = res_layer(g, h2, "head.layer1", 32, 32, hf, wt, eps)?;
    h2 = n;
    hf = hf1;
    wt = wt1;
    // layer2
    let (n, hf2, wt2) = res_layer(g, h2, "head.layer2", 32, 32, hf, wt, eps)?;
    h2 = n;
    hf = hf2;
    wt = wt2;
    // conv2 stride (2,1) + bn2 + relu
    let (n, ho, wo) = conv2d(g, h2, "head.conv2.weight", 32, 32, 3, 3, 2, 1, 1, 1, hf, wt)?;
    h2 = g.batchnorm2d(n, "head.bn2", 32, eps)?;
    h2 = g.g().relu(h2);
    hf = ho;
    wt = wo;

    // reshape [1,32,F',T'] -> [1, 32*F', T']
    let channels = 32 * hf;
    let mut h = g.g().reshape_(h2, vec![1, channels as i64, wt as i64]);
    let mut t1 = wt;

    // xvector.tdnn: Conv1d(channels->128, k5, stride2, pad2) + bn + relu
    let pad = 2;
    let h_t = g.conv1d(
        h,
        "xvector.tdnn.linear.weight",
        None,
        channels,
        128,
        5,
        2,
        pad,
        1,
        t1,
    )?;
    t1 = (t1 + 2 * pad - 4 - 1) / 2 + 1;
    h = g.batchnorm1d(h_t, "xvector.tdnn.nonlinear.batchnorm", 128, eps, true)?;
    h = g.g().relu(h);
    let mut ch = 128usize;

    // dense blocks + transitions
    for (bi, &(num_layers, k, dil)) in cfg.blocks.iter().enumerate() {
        let bprefix = format!("xvector.block{}", bi + 1);
        let (n, out_ch) = dense_block(
            g,
            h,
            &bprefix,
            num_layers,
            ch,
            cfg.growth_rate,
            cfg.bn_size * cfg.growth_rate,
            k,
            dil,
            t1,
            eps,
        )?;
        h = n;
        ch = out_ch;
        // transit: nonlinear(bn+relu) -> Conv1d(ch -> ch/2, 1, bias=False)
        let tprefix = format!("xvector.transit{}", bi + 1);
        h = g.batchnorm1d(h, &format!("{tprefix}.nonlinear.batchnorm"), ch, eps, true)?;
        h = g.g().relu(h);
        let out = ch / 2;
        h = g.conv1d(
            h,
            &format!("{tprefix}.linear.weight"),
            None,
            ch,
            out,
            1,
            1,
            0,
            1,
            t1,
        )?;
        ch = out;
    }

    // out_nonlinear: bn + relu
    h = g.batchnorm1d(h, "xvector.out_nonlinear.batchnorm", ch, eps, true)?;
    h = g.g().relu(h);

    // statistics pooling -> [1, 2*ch]
    let pooled = stats_pool(g, h, ch, t1);

    // dense: Conv1d 1x1 (2*ch -> 192) + bn(affine=false)
    let two = 2 * ch;
    let pooled3 = g.g().reshape_(pooled, vec![1, two as i64, 1]);
    let has_bias = g.weights.has("xvector.dense.linear.bias");
    let bk = if has_bias {
        Some("xvector.dense.linear.bias")
    } else {
        None
    };
    let dense = g.conv1d(
        pooled3,
        "xvector.dense.linear.weight",
        bk,
        two,
        cfg.embedding_size,
        1,
        1,
        0,
        1,
        1,
    )?;
    let dense = g.batchnorm1d(
        dense,
        "xvector.dense.nonlinear.batchnorm",
        cfg.embedding_size,
        eps,
        false,
    )?;
    Ok(g.g().reshape_(dense, vec![1, cfg.embedding_size as i64]))
}

/// 2-D conv producing `[1, out, H', W']`; returns the node and out dims.
#[allow(clippy::too_many_arguments)]
fn conv2d(
    g: &mut Graph,
    x: HirNodeId,
    w_key: &str,
    in_c: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    h: usize,
    w: usize,
) -> Result<(HirNodeId, usize, usize)> {
    let wnode = g.synth_weight(w_key, &[out_c, in_c, kh, kw])?;
    let ho = (h + 2 * ph - kh) / sh + 1;
    let wo = (w + 2 * pw - kw) / sw + 1;
    let f = g.f;
    let node = g.g().conv2d(
        x,
        wnode,
        [kh, kw],
        [sh, sw],
        [ph, pw],
        1,
        Shape::new(&[1, out_c, ho, wo], f),
    );
    Ok((node, ho, wo))
}

/// Two `BasicResBlock`s (the first downsamples freq by `(2,1)`).
fn res_layer(
    g: &mut Graph,
    x: HirNodeId,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    eps: f32,
) -> Result<(HirNodeId, usize, usize)> {
    let (n, h1, w1) = res_block(g, x, &format!("{prefix}.0"), in_c, out_c, 2, h, w, eps)?;
    let (n, h2, w2) = res_block(g, n, &format!("{prefix}.1"), out_c, out_c, 1, h1, w1, eps)?;
    Ok((n, h2, w2))
}

#[allow(clippy::too_many_arguments)]
fn res_block(
    g: &mut Graph,
    x: HirNodeId,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    stride: usize,
    h: usize,
    w: usize,
    eps: f32,
) -> Result<(HirNodeId, usize, usize)> {
    let (n, h1, w1) = conv2d(
        g,
        x,
        &format!("{prefix}.conv1.weight"),
        in_c,
        out_c,
        3,
        3,
        stride,
        1,
        1,
        1,
        h,
        w,
    )?;
    let n = g.batchnorm2d(n, &format!("{prefix}.bn1"), out_c, eps)?;
    let n = g.g().relu(n);
    let (n, h2, w2) = conv2d(
        g,
        n,
        &format!("{prefix}.conv2.weight"),
        out_c,
        out_c,
        3,
        3,
        1,
        1,
        1,
        1,
        h1,
        w1,
    )?;
    let conv = g.batchnorm2d(n, &format!("{prefix}.bn2"), out_c, eps)?;
    let shortcut = if stride != 1 || in_c != out_c {
        let (s, _, _) = conv2d(
            g,
            x,
            &format!("{prefix}.shortcut.0.weight"),
            in_c,
            out_c,
            1,
            1,
            stride,
            1,
            0,
            0,
            h,
            w,
        )?;
        g.batchnorm2d(s, &format!("{prefix}.shortcut.1"), out_c, eps)?
    } else {
        x
    };
    let sum = g.g().add(conv, shortcut);
    Ok((g.g().relu(sum), h2, w2))
}

/// `CAMDenseTDNNBlock`: dense-net growth over `num_layers` CAM layers.
#[allow(clippy::too_many_arguments)]
fn dense_block(
    g: &mut Graph,
    x: HirNodeId,
    prefix: &str,
    num_layers: usize,
    in_c: usize,
    growth: usize,
    bn_c: usize,
    k: usize,
    dil: usize,
    t: usize,
    eps: f32,
) -> Result<(HirNodeId, usize)> {
    let mut h = x;
    let mut ch = in_c;
    for i in 0..num_layers {
        let lprefix = format!("{prefix}.tdnnd{}", i + 1);
        let y = dense_layer(g, h, &lprefix, ch, growth, bn_c, k, dil, t, eps)?;
        h = g.g().concat_(vec![h, y], 1); // grow channels
        ch += growth;
    }
    Ok((h, ch))
}

/// `CAMDenseTDNNLayer`: bn-relu → 1×1 bottleneck → bn-relu → CAM layer.
#[allow(clippy::too_many_arguments)]
fn dense_layer(
    g: &mut Graph,
    x: HirNodeId,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    bn_c: usize,
    k: usize,
    dil: usize,
    t: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let h = g.batchnorm1d(
        x,
        &format!("{prefix}.nonlinear1.batchnorm"),
        in_c,
        eps,
        true,
    )?;
    let h = g.g().relu(h);
    let h = g.conv1d(
        h,
        &format!("{prefix}.linear1.weight"),
        None,
        in_c,
        bn_c,
        1,
        1,
        0,
        1,
        t,
    )?;
    let h = g.batchnorm1d(
        h,
        &format!("{prefix}.nonlinear2.batchnorm"),
        bn_c,
        eps,
        true,
    )?;
    let h = g.g().relu(h);
    cam_layer(g, h, &format!("{prefix}.cam_layer"), bn_c, out_c, k, dil, t)
}

/// `CAMLayer`: a local conv modulated by a per-frame context-aware sigmoid
/// gate. The context is `global_time_mean + segment_avg_pool(seg_len=100)`,
/// both broadcast to `[1, C, T]`, exactly as in `campplus/components.py`.
#[allow(clippy::too_many_arguments)]
fn cam_layer(
    g: &mut Graph,
    x: HirNodeId,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    dil: usize,
    t: usize,
) -> Result<HirNodeId> {
    let pad = (k - 1) / 2 * dil;
    let y = g.conv1d(
        x,
        &format!("{prefix}.linear_local.weight"),
        None,
        in_c,
        out_c,
        k,
        1,
        pad,
        dil,
        t,
    )?;
    // context = global mean (broadcast over T) + segment average pool
    let gmean = g.g().mean(x, vec![2], true); // [1,in,1]
    let seg = seg_pool(g, x, in_c, t, 100); // [1,in,T]
    let ctx = g.g().add(seg, gmean); // broadcast [1,in,1] over T
    let red = (in_c / 2).max(1);
    let m = g.conv1d(
        ctx,
        &format!("{prefix}.linear1.weight"),
        Some(&format!("{prefix}.linear1.bias")),
        in_c,
        red,
        1,
        1,
        0,
        1,
        t,
    )?;
    let m = g.g().relu(m);
    let m = g.conv1d(
        m,
        &format!("{prefix}.linear2.weight"),
        Some(&format!("{prefix}.linear2.bias")),
        red,
        out_c,
        1,
        1,
        0,
        1,
        t,
    )?;
    let m = g.sigmoid(m); // [1,out,T]
    Ok(g.g().mul(y, m))
}

/// Segment average pooling (`seg_len`-frame non-overlapping windows, the last
/// partial window averaged over its valid frames), broadcast back to `[1,C,T]`.
/// Implemented as a right-multiply by a host-built `[T,T]` averaging matrix.
fn seg_pool(g: &mut Graph, x: HirNodeId, _c: usize, t: usize, seg_len: usize) -> HirNodeId {
    let mut a = vec![0f32; t * t];
    let mut s = 0;
    while s < t {
        let end = (s + seg_len).min(t);
        let cnt = (end - s) as f32;
        for ss in s..end {
            for jj in s..end {
                a[ss * t + jj] = 1.0 / cnt;
            }
        }
        s = end;
    }
    let key = g.fresh("segA");
    let an = g.synth(&key, a, &[t, t]);
    g.g().mm(x, an) // [1,C,T] @ [T,T] -> [1,C,T]
}

/// Statistics pooling: concat(mean, std) over time → `[1, 2C]`. `std` uses the
/// unbiased (N-1) estimator to match `torch.std` in `StatsPool`.
fn stats_pool(g: &mut Graph, x: HirNodeId, c: usize, t: usize) -> HirNodeId {
    let mean = g.g().mean(x, vec![2], true); // [1,c,1]
    let x2 = g.g().mul(x, x);
    let m2 = g.g().mean(x2, vec![2], true); // [1,c,1]
    let mm = g.g().mul(mean, mean);
    let var = g.g().sub(m2, mm);
    let var = g.g().relu(var); // guard fp round-off making variance slightly negative
    // biased → unbiased: var * N/(N-1)
    if t > 1 {
        let corr = g.scalar(t as f32 / (t as f32 - 1.0));
        let var = g.g().mul(var, corr);
        let std = g.g().sqrt(var);
        let cat = g.g().concat_(vec![mean, std], 1);
        return g.g().reshape_(cat, vec![1, (2 * c) as i64]);
    }
    let std = g.g().sqrt(var);
    let cat = g.g().concat_(vec![mean, std], 1); // [1,2c,1]
    g.g().reshape_(cat, vec![1, (2 * c) as i64])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_norm_zeroes_mean() {
        let mut d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // t=3,d=2
        mean_normalize(&mut d, 3, 2);
        // column 0: 1,3,5 mean 3 -> -2,0,2 ; column1: 2,4,6 mean4 -> -2,0,2
        assert!((d[0] + 2.0).abs() < 1e-6);
        assert!((d[2]).abs() < 1e-6);
    }
}
