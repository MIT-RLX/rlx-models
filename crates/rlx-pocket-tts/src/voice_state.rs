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

//! Voice conditioning in the formats the released voices ship in.
//!
//! - **State** (`languages/<lang>/embeddings/*.safetensors`, `export-voice`
//!   output): the FlowLM KV cache after the voice prompt, per layer
//!   `transformer.layers.{l}.self_attn/cache` `[2, 1, T, H_kv, D]` plus
//!   `…/offset` (positions used; older exports store `…/current_end`, whose
//!   *length* is the offset). Ready to generate from.
//! - **Prompt** (the January mirror's `audio_prompt [1, T, d_model]`): the
//!   conditioning sequence itself, which still has to be run through the
//!   backbone ([`crate::PocketTts::voice_from_prompt`]).

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use safetensors::tensor::Dtype;

use crate::weights::WeightFile;

/// The FlowLM KV cache after a voice prompt: per layer, `offset` rows of
/// `kv_dim` keys and values (post-RoPE, as upstream stores them).
#[derive(Debug, Clone)]
pub struct VoiceState {
    pub offset: usize,
    pub kv_dim: usize,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl VoiceState {
    /// Empty state (no voice): generation starts from position 0.
    pub fn empty(num_layers: usize, kv_dim: usize) -> Self {
        Self {
            offset: 0,
            kv_dim,
            k: vec![Vec::new(); num_layers],
            v: vec![Vec::new(); num_layers],
        }
    }

    pub fn num_layers(&self) -> usize {
        self.k.len()
    }
}

/// A voice file, in whichever format it came.
#[derive(Debug, Clone)]
pub enum VoiceFile {
    State(VoiceState),
    /// `[T, d_model]` conditioning rows.
    Prompt {
        rows: usize,
        dim: usize,
        data: Vec<f32>,
    },
}

impl VoiceFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let wf =
            WeightFile::open(path).with_context(|| format!("open voice {}", path.display()))?;
        let names = wf.names()?;
        if names.iter().any(|n| n.ends_with("self_attn/cache")) {
            return read_state(&wf, &names).map(Self::State);
        }
        for key in ["audio_prompt", "conditioning"] {
            if let Ok((data, shape)) = wf.get_f32(key) {
                let (rows, dim) = match shape.as_slice() {
                    [1, t, d] | [t, d] => (*t, *d),
                    s => bail!("{key}: unexpected shape {s:?}"),
                };
                return Ok(Self::Prompt { rows, dim, data });
            }
        }
        bail!(
            "{}: neither a voice state (…self_attn/cache) nor an audio_prompt",
            path.display()
        )
    }
}

fn read_state(wf: &WeightFile, names: &[String]) -> Result<VoiceState> {
    let mut layers = Vec::new();
    for n in names {
        if let Some(rest) = n.strip_prefix("transformer.layers.")
            && let Some((l, tail)) = rest.split_once('.')
            && tail == "self_attn/cache"
        {
            layers.push(l.parse::<usize>().context("layer index")?);
        }
    }
    layers.sort_unstable();
    if layers.iter().enumerate().any(|(i, &l)| i != l) {
        bail!("voice state layers are not 0..n: {layers:?}");
    }
    let mut state: Option<VoiceState> = None;
    for &l in &layers {
        let pre = format!("transformer.layers.{l}.self_attn");
        let (cache, shape) = wf.get_f32(&format!("{pre}/cache"))?;
        let [two, batch, t, h, d] = shape[..] else {
            bail!("{pre}/cache: shape {shape:?}, expected [2, 1, T, H, D]");
        };
        if two != 2 || batch != 1 {
            bail!("{pre}/cache: shape {shape:?}, expected [2, 1, T, H, D]");
        }
        let offset = read_offset(wf, &pre)?.unwrap_or(t);
        if offset > t {
            bail!("{pre}: offset {offset} exceeds the cache length {t}");
        }
        let kv_dim = h * d;
        let st = state.get_or_insert_with(|| VoiceState::empty(layers.len(), kv_dim));
        if st.kv_dim != kv_dim || (l > 0 && st.offset != offset) {
            bail!("{pre}: layers disagree on shape/offset");
        }
        st.offset = offset;
        let plane = t * kv_dim;
        st.k[l] = cache[..offset * kv_dim].to_vec();
        st.v[l] = cache[plane..plane + offset * kv_dim].to_vec();
        if st.k[l].iter().chain(&st.v[l]).any(|x| !x.is_finite()) {
            bail!("{pre}: non-finite values inside the used part of the cache");
        }
    }
    state.ok_or_else(|| anyhow!("voice state has no layers"))
}

/// `…/offset` (an i64 scalar) or the legacy `…/current_end` (its length).
fn read_offset(wf: &WeightFile, pre: &str) -> Result<Option<usize>> {
    if let Ok(t) = wf.tensor(&format!("{pre}/offset")) {
        let raw = t.data();
        let v = match t.dtype() {
            Dtype::I64 => i64::from_le_bytes(raw[..8].try_into()?),
            Dtype::I32 => i32::from_le_bytes(raw[..4].try_into()?) as i64,
            other => bail!("{pre}/offset: dtype {other:?}"),
        };
        return Ok(Some(usize::try_from(v).context("negative offset")?));
    }
    if let Ok(t) = wf.tensor(&format!("{pre}/current_end")) {
        return Ok(Some(*t.shape().first().unwrap_or(&0)));
    }
    Ok(None)
}

/// Write `state` in upstream's `export-voice` format (loadable by
/// `pocket_tts` and by [`VoiceFile::open`]). `head_dim` splits `kv_dim` into
/// heads for the `[2, 1, T, H, D]` cache layout.
pub fn save_state(state: &VoiceState, head_dim: usize, path: impl AsRef<Path>) -> Result<()> {
    use safetensors::tensor::TensorView;
    if head_dim == 0 || !state.kv_dim.is_multiple_of(head_dim) {
        bail!(
            "kv_dim {} is not a multiple of head_dim {head_dim}",
            state.kv_dim
        );
    }
    let heads = state.kv_dim / head_dim;
    let t = state.offset;
    let mut caches = Vec::with_capacity(state.num_layers());
    for l in 0..state.num_layers() {
        let mut bytes = Vec::with_capacity(2 * t * state.kv_dim * 4);
        for x in state.k[l].iter().chain(&state.v[l]) {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        caches.push(bytes);
    }
    let offset = (t as i64).to_le_bytes();
    let pad = 0i64.to_le_bytes();
    let mut views = Vec::new();
    for (l, bytes) in caches.iter().enumerate() {
        let pre = format!("transformer.layers.{l}.self_attn");
        views.push((
            format!("{pre}/cache"),
            TensorView::new(Dtype::F32, vec![2, 1, t, heads, head_dim], bytes)?,
        ));
        views.push((
            format!("{pre}/offset"),
            TensorView::new(Dtype::I64, vec![1], &offset)?,
        ));
        views.push((
            format!("{pre}/pad"),
            TensorView::new(Dtype::I64, vec![1], &pad)?,
        ));
    }
    safetensors::serialize_to_file(views, None, path.as_ref())
        .with_context(|| format!("write {}", path.as_ref().display()))?;
    Ok(())
}
