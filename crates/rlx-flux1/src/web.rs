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

//! Weights for a browser: every tensor exactly as a graph builder receives it,
//! in one GGUF.
//!
//! Preparing FLUX weights — merging LoRAs, permuting q/k rows, quantizing
//! 12B parameters — is neither cheap nor something to re-implement in a page.
//! [`Recorder`] wraps a real source while the native builder runs and keeps
//! what each call returned; [`write_gguf`] stores that, and [`Replay`] serves
//! it back to the same builder from a [`rlx_gguf::GgufFile`] — whole, or just
//! one pipeline stage's tensors fetched by HTTP `Range`. No weight logic is
//! duplicated, so a replayed graph is the native graph.
//!
//! Tensor names carry how they were asked for: `f/KEY` (`take`, f32),
//! `ft/KEY` (transposed), `h/KEY` / `ht/KEY` (`take_half`, raw F16/BF16),
//! `q/KEY` (`take_packed`), `m/KEY` (`load_merged`).

use anyhow::{Context, Result, anyhow, bail};
use rlx_flow::{GgufPackedLinear, WeightSource};
use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue};
use rlx_ir::DType;
use std::collections::{BTreeMap, HashSet};

use crate::config::Flux1Config;
use crate::weights::{DitWeights, LORA_A};

/// One recorded tensor: GGUF name, row-major shape, dtype, bytes.
pub struct Recorded {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgmlType,
    pub bytes: Vec<u8>,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn half_type(dtype: Option<DType>) -> GgmlType {
    match dtype {
        Some(DType::F16) => GgmlType::F16,
        _ => GgmlType::BF16,
    }
}

/// Passes every call through to `inner` and keeps a copy of the result.
pub struct Recorder<'a, W: ?Sized> {
    pub inner: &'a mut W,
    /// By name: a builder asks for some tensors more than once.
    pub tensors: BTreeMap<String, Recorded>,
}

impl<'a, W: WeightSource + ?Sized> Recorder<'a, W> {
    pub fn new(inner: &'a mut W) -> Self {
        Self {
            inner,
            tensors: BTreeMap::new(),
        }
    }

    pub fn keep(&mut self, name: String, shape: Vec<usize>, dtype: GgmlType, bytes: Vec<u8>) {
        self.tensors.entry(name.clone()).or_insert(Recorded {
            name,
            shape,
            dtype,
            bytes,
        });
    }
}

impl<W: WeightSource + ?Sized> WeightSource for Recorder<'_, W> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.inner.take(key, transpose)?;
        let name = format!("{}/{key}", if transpose { "ft" } else { "f" });
        self.keep(name, shape.clone(), GgmlType::F32, f32_bytes(&data));
        Ok((data, shape))
    }

    fn has(&self, key: &str) -> bool {
        self.inner.has(key)
    }

    fn stored_dtype(&self, key: &str) -> Option<DType> {
        self.inner.stored_dtype(key)
    }

    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        let Some(p) = self.inner.take_packed(key)? else {
            return Ok(None);
        };
        let dtype = ggml_for(p.scheme)?;
        self.keep(
            format!("q/{key}"),
            vec![p.out_dim, p.in_dim],
            dtype,
            p.w_q.clone(),
        );
        if !p.bias.is_empty() {
            bail!("web export: packed `{key}` carries a bias");
        }
        Ok(Some(p))
    }

    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let Some((bytes, shape)) = self.inner.take_half(key, transpose)? else {
            return Ok(None);
        };
        let dtype = half_type(self.inner.stored_dtype(key));
        let name = format!("{}/{key}", if transpose { "ht" } else { "h" });
        self.keep(name, shape.clone(), dtype, bytes.clone());
        Ok(Some((bytes, shape)))
    }
}

impl<W: DitWeights + ?Sized> DitWeights for Recorder<'_, W> {
    fn lora_targets(&self) -> HashSet<String> {
        // The replay derives these from the recorded `…@lora_a` tensors.
        self.inner.lora_targets()
    }

    fn load_merged(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.inner.load_merged(key)?;
        self.keep(
            format!("m/{key}"),
            shape.clone(),
            GgmlType::F32,
            f32_bytes(&data),
        );
        Ok((data, shape))
    }
}

fn ggml_for(scheme: rlx_ir::quant::QuantScheme) -> Result<GgmlType> {
    use rlx_ir::quant::QuantScheme as Q;
    Ok(match scheme {
        Q::GgufQ8_0 => GgmlType::Q8_0,
        Q::GgufQ6K => GgmlType::Q6K,
        Q::GgufQ5K => GgmlType::Q5K,
        Q::GgufQ4K => GgmlType::Q4K,
        other => bail!("web export: no GGUF type for {other:?}"),
    })
}

/// [`Flux1Config`] as GGUF metadata (`flux1.*`).
pub fn config_meta(cfg: &Flux1Config) -> Vec<(String, MetaValue)> {
    let u = |v: usize| MetaValue::U32(v as u32);
    vec![
        ("flux1.hidden".into(), u(cfg.hidden)),
        ("flux1.num_heads".into(), u(cfg.num_heads)),
        ("flux1.head_dim".into(), u(cfg.head_dim)),
        ("flux1.mlp_hidden".into(), u(cfg.mlp_hidden)),
        ("flux1.depth".into(), u(cfg.depth)),
        ("flux1.depth_single".into(), u(cfg.depth_single)),
        ("flux1.in_channels".into(), u(cfg.in_channels)),
        ("flux1.context_dim".into(), u(cfg.context_dim)),
        ("flux1.vec_dim".into(), u(cfg.vec_dim)),
        (
            "flux1.axes_dim".into(),
            MetaValue::Array(cfg.axes_dim.iter().map(|&a| u(a)).collect()),
        ),
        ("flux1.theta".into(), MetaValue::F64(cfg.theta)),
        (
            "flux1.guidance_embed".into(),
            MetaValue::Bool(cfg.guidance_embed),
        ),
    ]
}

/// [`Flux1Config`] back from [`config_meta`].
pub fn config_from_meta(file: &GgufFile) -> Result<Flux1Config> {
    let m = &file.metadata;
    let u = |k: &str| -> Result<usize> {
        m.get(k)
            .and_then(MetaValue::as_u32)
            .map(|v| v as usize)
            .with_context(|| format!("GGUF metadata `{k}` missing"))
    };
    let axes = match m.get("flux1.axes_dim") {
        Some(MetaValue::Array(a)) if a.len() == 3 => {
            let v: Vec<usize> = a
                .iter()
                .map(|x| x.as_u32().map(|v| v as usize))
                .collect::<Option<_>>()
                .context("flux1.axes_dim")?;
            [v[0], v[1], v[2]]
        }
        _ => bail!("GGUF metadata `flux1.axes_dim` missing"),
    };
    Ok(Flux1Config {
        hidden: u("flux1.hidden")?,
        num_heads: u("flux1.num_heads")?,
        head_dim: u("flux1.head_dim")?,
        mlp_hidden: u("flux1.mlp_hidden")?,
        depth: u("flux1.depth")?,
        depth_single: u("flux1.depth_single")?,
        in_channels: u("flux1.in_channels")?,
        context_dim: u("flux1.context_dim")?,
        vec_dim: u("flux1.vec_dim")?,
        axes_dim: axes,
        theta: match m.get("flux1.theta") {
            Some(MetaValue::F64(t)) => *t,
            _ => bail!("GGUF metadata `flux1.theta` missing"),
        },
        guidance_embed: matches!(m.get("flux1.guidance_embed"), Some(MetaValue::Bool(true))),
    })
}

/// Write recorded tensors (+ metadata) as a GGUF at `path`. Shapes are stored
/// in GGML order (innermost first).
pub fn write_gguf(
    path: &std::path::Path,
    tensors: impl IntoIterator<Item = Recorded>,
    meta: Vec<(String, MetaValue)>,
) -> Result<()> {
    let mut w = GgufWriter::new();
    w.set_arch("rlx-web");
    for (k, v) in meta {
        w.set_meta(k, v);
    }
    for t in tensors {
        let shape: Vec<usize> = t.shape.iter().rev().copied().collect();
        w.add_tensor_bytes(t.name, shape, t.dtype, t.bytes)?;
    }
    w.write_to_path(path)
}

/// Serves what a [`Recorder`] kept, from a GGUF [`write_gguf`] wrote.
pub struct Replay {
    pub file: GgufFile,
}

impl Replay {
    pub fn new(file: GgufFile) -> Self {
        Self { file }
    }

    fn raw(&self, name: &str) -> Option<(&[u8], Vec<usize>, GgmlType)> {
        let t = self.file.get(name)?;
        let bytes = self.file.tensor_bytes(t).ok()?;
        Some((bytes, t.shape.iter().rev().copied().collect(), t.dtype))
    }

    fn f32s(&self, name: &str, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (bytes, shape, dtype) = self
            .raw(name)
            .ok_or_else(|| anyhow!("`{key}` was not exported ({name})"))?;
        if dtype != GgmlType::F32 {
            bail!("{name}: expected F32, found {dtype:?}");
        }
        let data = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok((data, shape))
    }
}

impl WeightSource for Replay {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let name = format!("{}/{key}", if transpose { "ft" } else { "f" });
        self.f32s(&name, key)
    }

    fn has(&self, key: &str) -> bool {
        ["f", "ft", "h", "ht", "q", "m"]
            .iter()
            .any(|p| self.file.get(&format!("{p}/{key}")).is_some())
    }

    fn stored_dtype(&self, key: &str) -> Option<DType> {
        // Only half tensors were stored at their own width; asking for any
        // other key's dtype must not send the builder down `take_half`.
        let t = self
            .file
            .get(&format!("h/{key}"))
            .or_else(|| self.file.get(&format!("ht/{key}")))?;
        Some(match t.dtype {
            GgmlType::F16 => DType::F16,
            GgmlType::BF16 => DType::BF16,
            _ => return None,
        })
    }

    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        let Some((bytes, shape, dtype)) = self.raw(&format!("q/{key}")) else {
            return Ok(None);
        };
        let scheme = rlx_gguf::quant_scheme(dtype)
            .ok_or_else(|| anyhow!("q/{key}: {dtype:?} is not a packed type"))?;
        Ok(Some(GgufPackedLinear {
            w_q: bytes.to_vec(),
            scheme,
            in_dim: shape[1],
            out_dim: shape[0],
            bias: Vec::new(),
        }))
    }

    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let name = format!("{}/{key}", if transpose { "ht" } else { "h" });
        Ok(self
            .raw(&name)
            .map(|(bytes, shape, _)| (bytes.to_vec(), shape)))
    }
}

impl DitWeights for Replay {
    fn lora_targets(&self) -> HashSet<String> {
        // Weights whose LoRA stayed a graph branch were recorded with their
        // pairs (`…weight@lora_a`).
        self.file
            .tensors
            .keys()
            .filter_map(|name| {
                name.split_once('/')
                    .and_then(|(_, k)| k.strip_suffix(LORA_A))
                    .map(str::to_string)
            })
            .collect()
    }

    fn load_merged(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.f32s(&format!("m/{key}"), key)
    }
}

/// Record the whole denoiser — LoRAs merged, projections packed as `quant` —
/// and write it to `out` as one GGUF (metadata: [`config_meta`]). Graphs are
/// only built, never compiled; peak memory is the recorded weights plus one
/// chunk's graph.
pub fn export_denoiser(
    transformer: &std::path::Path,
    loras: &[(std::path::PathBuf, f32)],
    quant: crate::weights::WeightQuant,
    out: &std::path::Path,
) -> Result<usize> {
    use crate::transformer::{DitGeometry, build_chunk, plan_chunks};
    let file =
        crate::weights::TensorFile::open(transformer)?.with_common_prefix("model.diffusion_model.");
    let cfg = Flux1Config::from_shapes(|k| file.shape(k))?;
    let modules: Vec<String> = file
        .keys()
        .into_iter()
        .filter_map(|k| k.strip_suffix(".weight").map(str::to_string))
        .collect();
    let mut lora = crate::lora::LoraSet::default();
    for (path, scale) in loras {
        lora.add_file(path, *scale, &cfg, &modules)?;
    }
    let mut src = crate::weights::DitSource {
        file: &file,
        lora: &lora,
        cfg: &cfg,
        quant: Some(quant),
    };
    let mut rec = Recorder::new(&mut src);
    crate::embed::VecEmbedder::load(&mut rec, &cfg)?;
    // Any geometry: it shapes the graphs, not the weights they read.
    let geo = DitGeometry {
        txt_len: 16,
        grid_h: 2,
        grid_w: 2,
    };
    for parts in plan_chunks(&cfg, 1, 1) {
        drop(build_chunk(&cfg, geo, &parts, &mut rec)?);
    }
    let n = rec.tensors.len();
    write_gguf(
        out,
        std::mem::take(&mut rec.tensors).into_values(),
        config_meta(&cfg),
    )?;
    Ok(n)
}

/// Stage `index` of `count` of the denoiser: a contiguous run of
/// [`crate::transformer::DitPart`]s with about `1/count` of the weights (a
/// double block weighs ~2.4 single blocks). Stage 0 starts with the embedding,
/// the last ends with the final layer.
pub fn stage_parts(
    cfg: &Flux1Config,
    index: usize,
    count: usize,
) -> Result<Vec<crate::transformer::DitPart>> {
    use crate::transformer::DitPart;
    if count == 0 || index >= count {
        bail!("stage {index} of {count}");
    }
    let mut all = vec![DitPart::Embed];
    all.extend((0..cfg.depth).map(DitPart::Double));
    all.extend((0..cfg.depth_single).map(DitPart::Single));
    all.push(DitPart::Final);
    let weight = |p: &DitPart| match p {
        DitPart::Double(_) => 2.4,
        DitPart::Single(_) => 1.0,
        _ => 0.05,
    };
    let total: f64 = all.iter().map(weight).sum();
    // Cut where the running weight crosses each `k/count` mark.
    let mut stage = 0;
    let mut acc = 0.0;
    let mut out = Vec::new();
    for p in &all {
        let mid = acc + weight(p) / 2.0;
        while stage + 1 < count && mid > total * (stage + 1) as f64 / count as f64 {
            stage += 1;
        }
        acc += weight(p);
        if stage == index {
            out.push(*p);
        }
    }
    if out.is_empty() {
        bail!("denoiser stage {index} of {count} holds no blocks: use fewer stages");
    }
    Ok(out)
}

/// Whether the recorded tensor `name` (`f/KEY`, `q/KEY`, …) belongs to a stage
/// running `parts` (the first stage also holds the host conditioning MLPs).
pub fn stage_keeps(parts: &[crate::transformer::DitPart], name: &str) -> bool {
    use crate::transformer::DitPart;
    let key = name.split_once('/').map_or(name, |(_, k)| k);
    if parts.first() == Some(&DitPart::Embed)
        && ["time_in.", "guidance_in.", "vector_in."]
            .iter()
            .any(|p| key.starts_with(p))
    {
        return true;
    }
    parts.iter().any(|p| match p {
        DitPart::Embed => key.starts_with("img_in.") || key.starts_with("txt_in."),
        DitPart::Double(i) => key.starts_with(&format!("double_blocks.{i}.")),
        DitPart::Single(i) => key.starts_with(&format!("single_blocks.{i}.")),
        DitPart::Final => key.starts_with("final_layer."),
    })
}

/// One stage of a denoiser split across processes: its parts compiled one
/// graph each (one block's weights in host memory at a time). Stage 0 takes
/// the packed latents, T5 states and the conditioning vector it computes
/// ([`Self::vec`]); every other stage the previous one's hidden rows.
pub struct Flux1DitStage {
    pub cfg: Flux1Config,
    pub geo: crate::transformer::DitGeometry,
    pub parts: Vec<crate::transformer::DitPart>,
    vec_embed: Option<crate::embed::VecEmbedder>,
    chunks: Vec<rlx_runtime::CompiledGraph>,
}

impl Flux1DitStage {
    /// `group` parts per graph (≥ 1): more means fewer runs (and, in a
    /// browser, fewer readbacks and submissions) per step, at the cost of
    /// that many parts' weights in host memory while each graph is built.
    pub fn compile(
        cfg: &Flux1Config,
        geo: crate::transformer::DitGeometry,
        weights: &mut dyn DitWeights,
        device: rlx_runtime::Device,
        parts: Vec<crate::transformer::DitPart>,
        group: usize,
    ) -> Result<Self> {
        use crate::transformer::{DitPart, build_chunk};
        let vec_embed = if parts.first() == Some(&DitPart::Embed) {
            Some(crate::embed::VecEmbedder::load(weights, cfg)?)
        } else {
            None
        };
        let mut chunks = Vec::with_capacity(parts.len());
        for g in parts.chunks(group.max(1)) {
            let built = build_chunk(cfg, geo, g, weights)?;
            chunks.push(rlx_core::flow_util::compile_built(built, device)?);
        }
        Ok(Self {
            cfg: cfg.clone(),
            geo,
            parts,
            vec_embed,
            chunks,
        })
    }

    /// Whether this stage starts the denoiser (takes latents + T5 states).
    pub fn is_first(&self) -> bool {
        self.vec_embed.is_some()
    }

    /// The conditioning vector `[hidden]` (first stage only).
    pub fn vec(&self, t: f32, guidance: f32, pooled: &[f32]) -> Result<Vec<f32>> {
        Ok(self
            .vec_embed
            .as_ref()
            .context("only the first denoiser stage holds the conditioning MLPs")?
            .forward(t, guidance, pooled))
    }

    #[cfg(target_arch = "wasm32")]
    fn feeds<'a>(&self, first: &'a [f32], txt: &'a [f32]) -> Vec<(&'static str, &'a [f32])> {
        if self.is_first() {
            vec![("img", first), ("txt", txt)]
        } else {
            vec![("x", first)]
        }
    }

    /// Run the stage: `input` is the packed latents (first stage, with `txt`)
    /// or the previous stage's hidden rows (`txt` ignored).
    pub fn forward(&mut self, input: &[f32], txt: &[f32], vec: &[f32]) -> Result<Vec<f32>> {
        let mut x: Vec<f32> = Vec::new();
        for (n, chunk) in self.chunks.iter_mut().enumerate() {
            let out = if n == 0 {
                let mut feeds = if self.vec_embed.is_some() {
                    vec![("img", input), ("txt", txt)]
                } else {
                    vec![("x", input)]
                };
                feeds.push(("vec", vec));
                chunk.run(&feeds)
            } else {
                chunk.run(&[("x", &x), ("vec", vec)])
            };
            x = out.into_iter().next().context("chunk produced no output")?;
        }
        Ok(x)
    }

    /// [`Self::forward`] where a GPU readback cannot block (the browser).
    /// `on_part(i)` runs after part `i` (timing).
    #[cfg(target_arch = "wasm32")]
    pub async fn forward_async(
        &mut self,
        input: &[f32],
        txt: &[f32],
        vec: &[f32],
        on_part: &mut dyn FnMut(usize),
    ) -> Result<Vec<f32>> {
        let first_feeds = self.feeds(input, txt);
        let mut x: Vec<f32> = Vec::new();
        for (n, chunk) in self.chunks.iter_mut().enumerate() {
            let out = if n == 0 {
                let mut feeds = first_feeds.clone();
                feeds.push(("vec", vec));
                chunk.run_async(&feeds).await
            } else {
                chunk.run_async(&[("x", x.as_slice()), ("vec", vec)]).await
            }
            .map_err(|e| anyhow!("denoiser stage: {e}"))?;
            x = out.into_iter().next().context("chunk produced no output")?;
            on_part(n);
        }
        Ok(x)
    }
}

/// The T5 encoder's token embedding key in `file`.
fn t5_embed_key(file: &crate::weights::TensorFile) -> Result<String> {
    ["shared.weight", "encoder.embed_tokens.weight"]
        .into_iter()
        .find(|k| file.contains(k))
        .map(str::to_string)
        .context("T5 checkpoint has no token embedding")
}

/// The relative-attention bias table every T5 layer shares.
const T5_REL_BIAS: &str = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight";

/// Record the T5 encoder (projections packed as `quant`) and its host-side
/// tensors — the token embedding at its stored half width, the relative
/// bias table — into one GGUF; `config.json` rides along as `t5.config_json`.
pub fn export_t5(
    dir: &std::path::Path,
    quant: crate::weights::WeightQuant,
    out: &std::path::Path,
) -> Result<usize> {
    let config_json = std::fs::read_to_string(dir.join("config.json"))
        .with_context(|| format!("{}/config.json", dir.display()))?;
    let cfg: crate::config::T5Config = serde_json::from_str(&config_json)?;
    let file = crate::weights::TensorFile::open(dir)?;
    let embed_key = t5_embed_key(&file)?;
    let mut src = crate::weights::PlainSource {
        file: &file,
        prefix: "",
        quant: Some(quant),
    };
    let mut rec = Recorder::new(&mut src);
    for i in 0..cfg.num_layers {
        let last = i + 1 == cfg.num_layers;
        drop(crate::t5::build_t5_chunk(
            &cfg,
            16,
            i..i + 1,
            last,
            &mut rec,
        )?);
    }
    let (table, shape) = file.load(T5_REL_BIAS)?;
    rec.keep(
        format!("f/{T5_REL_BIAS}"),
        shape,
        GgmlType::F32,
        f32_bytes(&table),
    );
    let (bytes, shape) = file
        .load_half_bytes(&embed_key)?
        .context("T5 embedding is not stored at half width")?;
    let dtype = half_type(file.stored_dtype(&embed_key));
    rec.keep(format!("h/{embed_key}"), shape, dtype, bytes);
    let n = rec.tensors.len();
    write_gguf(
        out,
        std::mem::take(&mut rec.tensors).into_values(),
        vec![
            ("t5.config_json".into(), MetaValue::String(config_json)),
            ("t5.embed_key".into(), MetaValue::String(embed_key)),
        ],
    )?;
    Ok(n)
}

/// [`crate::config::T5Config`] and the embedding key from [`export_t5`]'s metadata.
pub fn t5_config_from_meta(file: &GgufFile) -> Result<(crate::config::T5Config, String)> {
    let s = |k: &str| {
        file.metadata
            .get(k)
            .and_then(MetaValue::as_str)
            .map(str::to_string)
            .with_context(|| format!("GGUF metadata `{k}` missing"))
    };
    Ok((
        serde_json::from_str(&s("t5.config_json")?)?,
        s("t5.embed_key")?,
    ))
}

/// Encoder layers `[start, end)` of T5 stage `index` of `count` (an even split).
pub fn t5_stage_layers(
    cfg: &crate::config::T5Config,
    index: usize,
    count: usize,
) -> Result<std::ops::Range<usize>> {
    if count == 0 || index >= count || count > cfg.num_layers {
        bail!("T5 stage {index} of {count} ({} layers)", cfg.num_layers);
    }
    let at = |i: usize| i * cfg.num_layers / count;
    Ok(at(index)..at(index + 1))
}

/// Whether recorded tensor `name` belongs to the T5 stage running `layers`.
pub fn t5_stage_keeps(
    cfg: &crate::config::T5Config,
    embed_key: &str,
    layers: &std::ops::Range<usize>,
    name: &str,
) -> bool {
    let key = name.split_once('/').map_or(name, |(_, k)| k);
    key == T5_REL_BIAS
        || (layers.start == 0 && key == embed_key)
        || (layers.end == cfg.num_layers && key.starts_with("encoder.final_layer_norm."))
        || layers
            .clone()
            .any(|i| key.starts_with(&format!("encoder.block.{i}.")))
}

/// One stage of a T5 encoder split across processes: its layers compiled one
/// graph each. The first stage turns token ids into embedding rows; every
/// stage hands its hidden rows `[len, d_model]` to the next.
pub struct T5WebStage {
    pub cfg: crate::config::T5Config,
    pub len: usize,
    bias: Vec<f32>,
    /// The token embedding (first stage): raw half bytes, its dtype, width.
    embed: Option<(Vec<u8>, DType, usize)>,
    chunks: Vec<rlx_runtime::CompiledGraph>,
}

impl T5WebStage {
    pub fn compile(
        file: GgufFile,
        len: usize,
        index: usize,
        count: usize,
        device: rlx_runtime::Device,
    ) -> Result<Self> {
        let (cfg, embed_key) = t5_config_from_meta(&file)?;
        let layers = t5_stage_layers(&cfg, index, count)?;
        let mut replay = Replay::new(file);
        let (table, _) = replay.take(T5_REL_BIAS, false)?;
        let bias = crate::t5::position_bias(&cfg, &table, len);
        let embed = if layers.start == 0 {
            let dtype = replay
                .stored_dtype(&embed_key)
                .context("T5 embedding missing")?;
            let (bytes, shape) = replay
                .take_half(&embed_key, false)?
                .context("T5 embedding missing")?;
            Some((bytes, dtype, shape[1]))
        } else {
            None
        };
        let mut chunks = Vec::with_capacity(layers.len());
        for i in layers.clone() {
            let last = i + 1 == cfg.num_layers;
            let built = crate::t5::build_t5_chunk(&cfg, len, i..i + 1, last, &mut replay)?;
            chunks.push(rlx_core::flow_util::compile_built(built, device)?);
        }
        Ok(Self {
            cfg,
            len,
            bias,
            embed,
            chunks,
        })
    }

    pub fn is_first(&self) -> bool {
        self.embed.is_some()
    }

    /// Embedding rows `[len, d_model]` for `ids` (first stage only).
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let (bytes, dtype, width) = self
            .embed
            .as_ref()
            .context("only the first T5 stage holds the embedding")?;
        ensure_len(ids.len(), self.len)?;
        let mut out = Vec::with_capacity(ids.len() * width);
        for &id in ids {
            let row = &bytes[id as usize * width * 2..(id as usize + 1) * width * 2];
            out.extend(row.as_chunks::<2>().0.iter().map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                match dtype {
                    DType::F16 => half::f16::from_bits(bits).to_f32(),
                    _ => half::bf16::from_bits(bits).to_f32(),
                }
            }));
        }
        Ok(out)
    }

    /// Run the stage's layers on hidden rows `x` (`[len, d_model]`).
    pub fn forward(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        let mut x = x.to_vec();
        for chunk in &mut self.chunks {
            x = chunk
                .run(&[("x", &x), ("bias", &self.bias)])
                .into_iter()
                .next()
                .context("T5 chunk produced no output")?;
        }
        Ok(x)
    }

    /// [`Self::forward`] where a GPU readback cannot block (the browser).
    #[cfg(target_arch = "wasm32")]
    pub async fn forward_async(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        let mut x = x.to_vec();
        for chunk in &mut self.chunks {
            x = chunk
                .run_async(&[("x", x.as_slice()), ("bias", self.bias.as_slice())])
                .await
                .map_err(|e| anyhow!("T5 stage: {e}"))?
                .into_iter()
                .next()
                .context("T5 chunk produced no output")?;
        }
        Ok(x)
    }
}

fn ensure_len(got: usize, want: usize) -> Result<()> {
    if got != want {
        bail!("T5 expects {want} ids, got {got}");
    }
    Ok(())
}

/// Record CLIP (as stored) and the VAE decoder's tensors into one small GGUF:
/// CLIP's graph weights, its token / position embedding tables (host-side),
/// its `config.json` (`clip.config_json`), and every VAE tensor as `v/KEY`.
pub fn export_aux(
    clip_dir: &std::path::Path,
    vae: &std::path::Path,
    out: &std::path::Path,
) -> Result<usize> {
    use crate::clip::PREFIX;
    let config_json = std::fs::read_to_string(clip_dir.join("config.json"))
        .with_context(|| format!("{}/config.json", clip_dir.display()))?;
    let cfg = crate::config::ClipTextConfig::from_json(&config_json)?;
    let file = crate::weights::TensorFile::open(clip_dir)?;
    let mut src = crate::weights::PlainSource {
        file: &file,
        prefix: PREFIX,
        quant: None,
    };
    let mut rec = Recorder::new(&mut src);
    drop(crate::clip::build_clip_text(&cfg, &mut rec)?);
    for key in [
        "embeddings.token_embedding.weight",
        "embeddings.position_embedding.weight",
    ] {
        let (data, shape) = file.load(&format!("{PREFIX}{key}"))?;
        rec.keep(
            format!("f/clip.{key}"),
            shape,
            GgmlType::F32,
            f32_bytes(&data),
        );
    }
    let (_, mut wm) = crate::vae::flux1_vae_weight_map(vae)?;
    for (key, (data, shape)) in wm.take_all() {
        rec.keep(format!("v/{key}"), shape, GgmlType::F32, f32_bytes(&data));
    }
    let n = rec.tensors.len();
    write_gguf(
        out,
        std::mem::take(&mut rec.tensors).into_values(),
        vec![("clip.config_json".into(), MetaValue::String(config_json))],
    )?;
    Ok(n)
}

/// CLIP's pooled embedding and the VAE decoder, from [`export_aux`]'s GGUF —
/// the small models of the browser pipeline, in one worker.
pub struct AuxStage {
    pub clip_cfg: crate::config::ClipTextConfig,
    clip: rlx_runtime::CompiledGraph,
    tok_table: Vec<f32>,
    pos_table: Vec<f32>,
    vae: crate::vae::Flux1VaeDecoder,
    pub latent_h: usize,
    pub latent_w: usize,
}

impl AuxStage {
    /// VAE compiled for `latent_h × latent_w` latents (160 px: 20 × 20).
    pub fn compile(
        file: GgufFile,
        latent_h: usize,
        latent_w: usize,
        device: rlx_runtime::Device,
    ) -> Result<Self> {
        let config_json = file
            .metadata
            .get("clip.config_json")
            .and_then(MetaValue::as_str)
            .context("GGUF metadata `clip.config_json` missing")?
            .to_string();
        let clip_cfg = crate::config::ClipTextConfig::from_json(&config_json)?;
        let mut replay = Replay::new(file);
        let (tok_table, _) = replay.take("clip.embeddings.token_embedding.weight", false)?;
        let (pos_table, _) = replay.take("clip.embeddings.position_embedding.weight", false)?;
        let built = crate::clip::build_clip_text(&clip_cfg, &mut replay)?;
        let clip = rlx_core::flow_util::compile_built(built, device)?;
        let mut tensors = std::collections::HashMap::new();
        let names: Vec<String> = replay
            .file
            .tensors
            .keys()
            .filter(|n| n.starts_with("v/"))
            .cloned()
            .collect();
        for name in names {
            let key = name["v/".len()..].to_string();
            let t = replay.f32s(&name, &key)?;
            tensors.insert(key, t);
        }
        let vae_cfg = crate::vae::flux1_vae_config();
        let weights = rlx_flux2::vae::extract_flux2_vae_weights(
            rlx_core::weight_map::WeightMap::from_tensors(tensors),
            &vae_cfg,
        )?;
        let vae =
            crate::vae::Flux1VaeDecoder::compile(&vae_cfg, &weights, latent_h, latent_w, device)?;
        Ok(Self {
            clip_cfg,
            clip,
            tok_table,
            pos_table,
            vae,
            latent_h,
            latent_w,
        })
    }

    /// CLIP's input rows (token + position embedding) for `ids`.
    fn clip_hidden(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let (ctx, w) = (
            self.clip_cfg.max_position_embeddings,
            self.clip_cfg.hidden_size,
        );
        if ids.len() != ctx {
            bail!("CLIP expects {ctx} ids, got {}", ids.len());
        }
        let mut hidden = Vec::with_capacity(ctx * w);
        for (p, &id) in ids.iter().enumerate() {
            let row = &self.tok_table[id as usize * w..(id as usize + 1) * w];
            let pos = &self.pos_table[p * w..(p + 1) * w];
            hidden.extend(row.iter().zip(pos).map(|(a, b)| a + b));
        }
        Ok(hidden)
    }

    fn pool(&self, ids: &[u32], seq: &[f32]) -> Vec<f32> {
        let w = self.clip_cfg.hidden_size;
        let eos = ids
            .iter()
            .position(|&t| t == self.clip_cfg.eos_token_id)
            .unwrap_or(ids.len() - 1);
        seq[eos * w..(eos + 1) * w].to_vec()
    }

    /// Pooled CLIP embedding for exactly `max_position_embeddings` ids.
    pub fn pooled(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.clip_hidden(ids)?;
        let out = self.clip.run(&[("hidden", &hidden)]);
        let seq = out.into_iter().next().context("CLIP produced no output")?;
        Ok(self.pool(ids, &seq))
    }

    /// Unpacked latents `[16, latent_h, latent_w]` → RGB8, row-major.
    pub fn decode_rgb8(&mut self, latents: &[f32]) -> Result<Vec<u8>> {
        let rgb = self.vae.decode(latents)?;
        Ok(crate::vae::rgb_to_u8(
            &rgb,
            self.latent_h * 8,
            self.latent_w * 8,
        ))
    }

    /// [`Self::pooled`] in a browser.
    #[cfg(target_arch = "wasm32")]
    pub async fn pooled_async(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.clip_hidden(ids)?;
        let seq = self
            .clip
            .run_async(&[("hidden", hidden.as_slice())])
            .await
            .map_err(|e| anyhow!("CLIP: {e}"))?
            .into_iter()
            .next()
            .context("CLIP produced no output")?;
        Ok(self.pool(ids, &seq))
    }

    /// [`Self::decode_rgb8`] in a browser.
    #[cfg(target_arch = "wasm32")]
    pub async fn decode_rgb8_async(&mut self, latents: &[f32]) -> Result<Vec<u8>> {
        let rgb = self.vae.decode_async(latents).await?;
        Ok(crate::vae::rgb_to_u8(
            &rgb,
            self.latent_h * 8,
            self.latent_w * 8,
        ))
    }
}
