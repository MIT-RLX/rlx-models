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

//! [`Lfm3Audio`]: the loaded model, its compiled graphs and the generation
//! loop of `Lfm3AudioForConditionalGeneration.generate`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_core::weight_storage::HostTable;
use rlx_runtime::{CompiledGraph, Device};

use crate::config::{CODEC_FRAME_STACK, CODEC_N_FFT, Lfm3AudioConfig};
use crate::diag::{Timer, trace};
use crate::graph::{self, Gb};
use crate::istft::istft;
use crate::sampling::{Rng, Sampling, sample};
use crate::trunk::{self, Mixer, StepState};
use crate::{mel, stages};

/// Mel frames per encoder bucket (8 encoder tokens).
const ENCODER_BUCKET: usize = 64;
/// Code frames per codec bucket.
const CODEC_BUCKET: usize = 16;
/// Positions per LM prefill chunk.
const PREFILL_CHUNK: usize = 32;
/// LM cache slots are allocated in multiples of this.
const KV_ROUND: usize = 256;
/// Additive mask value of an invisible key.
const MASKED: f32 = -1e9;

/// What the model writes next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Text until `<|im_end|>` / `<|text_end|>`.
    Text,
    /// Speech frames until an end-of-audio frame, then text.
    Audio,
    /// Blocks of text and speech (`<|mixed_start|>`).
    Interleaved,
}

/// How frames' codebooks are revealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevealOrder {
    /// A fresh random permutation per frame, as the reference does.
    Random,
    /// Codebook index order — deterministic, for comparisons.
    Index,
}

/// Generation settings.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub text: Sampling,
    /// Tokens of text history the text repetition penalty looks at.
    pub text_repetition_window: usize,
    pub speech: Sampling,
    pub speech_repetition_window: usize,
    /// Forbid `<|mixed_start|>` and `<|audio_start|>`: a text-only reply.
    pub text_only: bool,
    pub reveal: RevealOrder,
    pub seed: u64,
}

impl GenerateOptions {
    /// Greedy text, the card's speech sampling.
    pub fn new(cfg: &Lfm3AudioConfig) -> Self {
        let sp = cfg.speech_sampling;
        Self {
            max_new_tokens: 512,
            text: Sampling::GREEDY,
            text_repetition_window: 32,
            speech: Sampling {
                temperature: sp.temperature,
                top_k: sp.top_k,
                repetition_penalty: sp.repetition_penalty,
            },
            speech_repetition_window: sp.repetition_window,
            text_only: false,
            reveal: RevealOrder::Random,
            seed: 0,
        }
    }
}

/// What [`Lfm3Audio::generate`] wrote.
#[derive(Debug, Clone, Default)]
pub struct Generated {
    /// Text tokens, with one `<|audio_out|>` per speech frame in place.
    pub tokens: Vec<u32>,
    /// Speech frames (`num_codebooks` codes each), end-of-audio frames excluded.
    pub frames: Vec<Vec<u32>>,
}

impl Generated {
    /// The text tokens alone.
    pub fn text_tokens(&self, audio_out: u32) -> Vec<u32> {
        self.tokens
            .iter()
            .copied()
            .filter(|&t| t != audio_out)
            .collect()
    }
}

/// One prompt element.
#[derive(Debug, Clone)]
pub enum Segment {
    Tokens(Vec<u32>),
    /// Speech-encoder embeddings `[n, lm_hidden]` (one `<|audio|>` each).
    Audio(Vec<f32>),
    /// Speech frames the model wrote earlier (one `<|audio_out|>` each).
    Frames(Vec<Vec<u32>>),
}

struct LmGraph {
    compiled: CompiledGraph,
    state: StepState,
}

struct LmGraphs {
    max_kv: usize,
    one: LmGraph,
    /// `PREFILL_CHUNK`-row graph for prompts; absent in low-memory mode.
    chunk: Option<LmGraph>,
}

/// The LM's position and caches.
pub struct LmState {
    pos: usize,
    max_kv: usize,
    caches: HashMap<String, Vec<f32>>,
}

impl LmState {
    pub fn position(&self) -> usize {
        self.pos
    }
}

/// LFM3-Audio-2B: the LM GGUF plus its mmproj.
pub struct Lfm3Audio {
    lm_path: PathBuf,
    mmproj_path: PathBuf,
    device: Device,
    storage: crate::WeightStorage,
    pub cfg: Lfm3AudioConfig,
    embed: HostTable,
    frame_embed: HostTable,
    /// Per-codebook code embeddings `[codebooks, codebook_size, dim]` (f16).
    depth_table: HostTable,
    mask_embed: Vec<f32>,
    fsq: Vec<f32>,
    encoders: HashMap<usize, CompiledGraph>,
    codecs: HashMap<usize, CompiledGraph>,
    depth: Option<CompiledGraph>,
    lm: Option<LmGraphs>,
    /// Score the tied LM head on the host instead of in the step graph.
    host_head: bool,
    /// One LM graph (prompts fed a position at a time) and stage graphs
    /// dropped after use.
    low_memory: bool,
}

/// `dir/*.gguf` → (LM, mmproj), preferring `prefer` in the LM's name.
///
/// `prefer_mmproj` picks the mmproj quant (default `F16`; `Q8_0` / `Q4_K_M`
/// keep the speech stacks packed — less memory, see the crate README).
pub fn resolve(
    path: &Path,
    prefer: Option<&str>,
    prefer_mmproj: Option<&str>,
) -> Result<(PathBuf, PathBuf)> {
    let mm_pref = Some(prefer_mmproj.unwrap_or("F16"));
    if path.is_file() {
        let dir = path.parent().unwrap_or(Path::new("."));
        let mmproj = find(dir, |n| n.starts_with("mmproj"), mm_pref)?;
        return Ok((path.to_path_buf(), mmproj));
    }
    let lm = find(path, |n| !n.starts_with("mmproj"), prefer)?;
    let mmproj = find(path, |n| n.starts_with("mmproj"), mm_pref)?;
    Ok((lm, mmproj))
}

fn find(dir: &Path, want: impl Fn(&str) -> bool, prefer: Option<&str>) -> Result<PathBuf> {
    let mut hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| anyhow!("read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|x| x == "gguf")
                && p.file_name().and_then(|n| n.to_str()).is_some_and(&want)
        })
        .collect();
    hits.sort();
    if let Some(pref) = prefer
        && let Some(p) = hits.iter().find(|p| p.to_string_lossy().contains(pref))
    {
        return Ok(p.clone());
    }
    hits.into_iter()
        .next()
        .ok_or_else(|| anyhow!("no matching .gguf in {}", dir.display()))
}

impl Lfm3Audio {
    /// Load from the LM GGUF and the mmproj GGUF.
    pub fn open(lm_path: &Path, mmproj_path: &Path, device: Device) -> Result<Self> {
        // CoreML runs fp32 graphs on CPU+GPU by default, and its GPU unit
        // aborts the process inside Apple's MPS compiler on the speech encoder
        // (`XPC_ERROR_CONNECTION_INTERRUPTED`). CPU + Neural Engine runs every
        // graph here at parity; an explicit `RLX_COREML_UNITS` still wins.
        if device == Device::Ane && rlx_ir::env::var("RLX_COREML_UNITS").is_none() {
            rlx_ir::env::set("RLX_COREML_UNITS", "ane");
        }
        trace("open: start");
        let mut lm = graph::open(lm_path)?;
        let mut mm = graph::open(mmproj_path)?;
        trace("open: GGUF headers");
        let cfg = Lfm3AudioConfig::from_gguf(
            lm.gguf_file().ok_or_else(|| anyhow!("not a GGUF loader"))?,
            mm.gguf_file().ok_or_else(|| anyhow!("not a GGUF loader"))?,
        )?;
        trace("open: config");
        let embed = host_table(&mut lm, "token_embd.weight", cfg.lm.hidden)?;
        trace("open: token table");
        let frame_embed = host_table(&mut mm, "a.gen.code.out_embd.weight", cfg.lm.hidden)?;
        let depth_table = host_table(&mut mm, "a.gen.code.embd.weight", cfg.depth.hidden)?;
        let (mask_embed, _) = mm.take("a.gen.code.mask_embd")?;
        let (fsq, _) = mm.take("a.gen.wav.quant.first.codebook.weight")?;
        trace("open: speech tables");
        Ok(Self {
            lm_path: lm_path.to_path_buf(),
            mmproj_path: mmproj_path.to_path_buf(),
            device,
            storage: graph::default_storage(),
            cfg,
            embed,
            frame_embed,
            depth_table,
            mask_embed,
            fsq,
            encoders: HashMap::new(),
            codecs: HashMap::new(),
            depth: None,
            lm: None,
            host_head: host_lm_head(device),
            low_memory: cfg!(target_os = "ios"),
        })
    }

    /// Load from a directory holding both files (`prefer` picks the LM
    /// quant, `prefer_mmproj` the mmproj's; see [`resolve`]).
    pub fn open_dir(
        dir: &Path,
        prefer: Option<&str>,
        prefer_mmproj: Option<&str>,
        device: Device,
    ) -> Result<Self> {
        let (lm, mm) = resolve(dir, prefer, prefer_mmproj)?;
        Self::open(&lm, &mm, device)
    }

    pub fn lm_path(&self) -> &Path {
        &self.lm_path
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Trade speed for memory: keep one LM graph (no 32-row prefill graph, so
    /// prompts are fed a position at a time) and drop the speech-encoder and
    /// codec graphs once used. On by default on iOS. Drops the compiled graphs.
    pub fn set_low_memory(&mut self, on: bool) {
        self.low_memory = on;
        self.encoders.clear();
        self.codecs.clear();
        self.lm = None;
    }

    /// Store dense weights as `storage` in graphs built from now on (packed
    /// quants are unaffected). Drops the compiled graphs.
    pub fn set_weight_storage(&mut self, storage: crate::WeightStorage) {
        self.storage = storage;
        self.encoders.clear();
        self.codecs.clear();
        self.depth = None;
        self.lm = None;
    }

    /// Build a graph over `weights` with `build` and compile it.
    fn compile_stage<T>(
        &self,
        weights: &Path,
        name: &str,
        build: impl FnOnce(&mut Gb) -> Result<T>,
    ) -> Result<(CompiledGraph, T)> {
        let mut w = graph::open(weights)?;
        let mut gb = Gb::new(name, &mut w, self.storage).dense_only(graph::dense_only(self.device));
        let made = build(&mut gb)?;
        Ok((graph::compile(gb, self.device, weights)?, made))
    }

    // ------------------------------------------------------------- inputs

    /// Input embeddings of text tokens `[ids.len(), hidden]`.
    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let h = self.cfg.lm.hidden;
        let mut out = vec![0f32; ids.len() * h];
        for (row, &t) in out.chunks_exact_mut(h).zip(ids) {
            if t as usize >= self.cfg.vocab_size {
                bail!("token {t} past the vocabulary");
            }
            self.embed.read_into(t as usize * h, row);
        }
        Ok(out)
    }

    /// LM input embedding of one speech frame: the sum of its codebook rows.
    pub fn embed_frame(&self, codes: &[u32]) -> Vec<f32> {
        let (h, cs) = (self.cfg.lm.hidden, self.cfg.codebook_size);
        let mut out = vec![0f32; h];
        let mut row = vec![0f32; h];
        for (q, &c) in codes.iter().enumerate() {
            self.frame_embed
                .read_into((q * cs + c as usize) * h, &mut row);
            for (o, r) in out.iter_mut().zip(&row) {
                *o += r;
            }
        }
        out
    }

    /// Speech-encoder embeddings `[ceil(frames/8), lm_hidden]` of 16 kHz audio.
    pub fn encode_audio(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let mels = self.cfg.num_mel_bins;
        let feats = mel::log_mel(samples, mels);
        self.encode_mel(&feats)
    }

    /// [`encode_audio`](Self::encode_audio) from log-mel features
    /// `[n_mels, frames]` (mel-major).
    pub fn encode_mel(&mut self, feats: &[f32]) -> Result<Vec<f32>> {
        let mels = self.cfg.num_mel_bins;
        let frames = feats.len() / mels;
        if frames == 0 {
            return Ok(Vec::new());
        }
        let bucket = frames.div_ceil(ENCODER_BUCKET) * ENCODER_BUCKET;
        if !self.encoders.contains_key(&bucket) {
            let _t = Timer::new(|| format!("compile encoder frames={bucket}"));
            let (compiled, ()) = self.compile_stage(&self.mmproj_path, "lfm3a_encoder", |gb| {
                stages::encoder(gb, &self.cfg, bucket)
            })?;
            self.encoders.insert(bucket, compiled);
        }
        let mut mel = vec![0f32; mels * bucket];
        for m in 0..mels {
            mel[m * bucket..m * bucket + frames]
                .copy_from_slice(&feats[m * frames..(m + 1) * frames]);
        }
        let (t1, t2) = (bucket.div_ceil(2), bucket.div_ceil(4));
        let (l1, l2) = (frames.div_ceil(2), frames.div_ceil(4));
        let mask1: Vec<f32> = (0..t1).map(|i| (i < l1) as u8 as f32).collect();
        let mask2: Vec<f32> = (0..t2).map(|i| (i < l2) as u8 as f32).collect();
        let g = self.encoders.get_mut(&bucket).expect("compiled above");
        let mut out = g.run(&[("mel", &mel), ("mask1", &mask1), ("mask2", &mask2)]);
        if self.low_memory {
            self.encoders.clear();
        }
        let mut emb = out.swap_remove(0);
        emb.truncate(frames.div_ceil(8) * self.cfg.lm.hidden);
        Ok(emb)
    }

    // ------------------------------------------------------------- LM

    fn lm_graphs(&mut self, need: usize) -> Result<&mut LmGraphs> {
        let fits = self.lm.as_ref().is_some_and(|g| g.max_kv >= need);
        if !fits {
            let max_kv = need.div_ceil(KV_ROUND) * KV_ROUND;
            let build = |rows: usize| -> Result<LmGraph> {
                let _t = Timer::new(|| format!("compile LM step rows={rows} max_kv={max_kv}"));
                let (compiled, state) =
                    self.compile_stage(&self.lm_path, "lfm3a_lm_step", |gb| {
                        stages::lm_step(gb, &self.cfg, rows, max_kv, self.host_head)
                    })?;
                Ok(LmGraph { compiled, state })
            };
            self.lm = Some(LmGraphs {
                max_kv,
                one: build(1)?,
                chunk: if self.low_memory {
                    None
                } else {
                    Some(build(PREFILL_CHUNK)?)
                },
            });
        }
        Ok(self.lm.as_mut().expect("built above"))
    }

    /// A fresh LM state with room for `capacity` positions.
    pub fn lm_state(&mut self, capacity: usize) -> Result<LmState> {
        let g = self.lm_graphs(capacity)?;
        let caches = g
            .one
            .state
            .names
            .iter()
            .zip(&g.one.state.sizes)
            .map(|(n, &s)| (n.clone(), vec![0f32; s]))
            .collect();
        Ok(LmState {
            pos: 0,
            max_kv: g.max_kv,
            caches,
        })
    }

    /// Feed `embeds` (`[n, hidden]`) at the state's position. Returns the
    /// last position's final-normed hidden state and logits.
    pub fn lm_feed(&mut self, st: &mut LmState, embeds: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let h = self.cfg.lm.hidden;
        let n = embeds.len() / h;
        if n == 0 {
            bail!("lm_feed: nothing to feed");
        }
        if st.pos + n > st.max_kv {
            bail!("LM cache full: {} + {n} > {}", st.pos, st.max_kv);
        }
        let _t = Timer::new(|| format!("LM feed {n} rows at {}", st.pos));
        let mut out = None;
        let mut i = 0;
        while i < n {
            let chunked = self.lm.as_ref().is_some_and(|g| g.chunk.is_some());
            let rows = if chunked && n - i >= PREFILL_CHUNK {
                PREFILL_CHUNK
            } else {
                1
            };
            out = Some(self.lm_run(st, &embeds[i * h..(i + rows) * h], rows)?);
            i += rows;
        }
        Ok(out.expect("n > 0"))
    }

    fn lm_run(&mut self, st: &mut LmState, x: &[f32], rows: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let spec = self.cfg.lm.clone();
        let (pos, max_kv) = (st.pos, st.max_kv);
        let half = spec.head_dim / 2;
        let mut feeds: Vec<(String, Vec<f32>)> = Vec::new();
        for (r, rope) in spec.ropes.iter().enumerate() {
            let (cos, sin) = trunk::rope_tables(*rope, spec.head_dim, pos..pos + rows);
            debug_assert_eq!(cos.len(), rows * half);
            feeds.push((format!("rope{r}_cos"), cos));
            feeds.push((format!("rope{r}_sin"), sin));
        }
        let mut write = vec![0f32; max_kv * rows];
        let mut keep = vec![1f32; max_kv];
        for j in 0..rows {
            write[(pos + j) * rows + j] = 1.0;
            keep[pos + j] = 0.0;
        }
        feeds.push(("kv_write".into(), write));
        feeds.push(("kv_keep".into(), keep));
        for w in spec.windows() {
            let visible = |q: usize, k: usize| k <= q && w.is_none_or(|w| k + w > q);
            let mask = if rows == 1 {
                (0..max_kv).map(|k| visible(pos, k) as u8 as f32).collect()
            } else {
                let mut m = vec![MASKED; rows * max_kv];
                for j in 0..rows {
                    for k in 0..max_kv {
                        if visible(pos + j, k) {
                            m[j * max_kv + k] = 0.0;
                        }
                    }
                }
                m
            };
            feeds.push((trunk::mask_name(w), mask));
        }
        let graphs = self
            .lm
            .as_mut()
            .ok_or_else(|| anyhow!("LM graphs not built"))?;
        let g = if rows == 1 {
            &mut graphs.one
        } else {
            graphs
                .chunk
                .as_mut()
                .ok_or_else(|| anyhow!("LM prefill graph not built"))?
        };
        let mut inputs: Vec<(&str, &[f32])> = vec![("x", x)];
        inputs.extend(feeds.iter().map(|(n, v)| (n.as_str(), v.as_slice())));
        for name in &g.state.names {
            inputs.push((name.as_str(), st.caches[name].as_slice()));
        }
        let t_run = Timer::new(|| format!("  LM run rows={rows}"));
        let mut outs = g.compiled.run(&inputs).into_iter();
        drop(t_run);
        let hidden = outs
            .next()
            .ok_or_else(|| anyhow!("LM step: no hidden output"))?;
        let logits = if self.host_head {
            None
        } else {
            Some(
                outs.next()
                    .ok_or_else(|| anyhow!("LM step: no logits output"))?,
            )
        };
        for (name, v) in g.state.names.iter().zip(outs) {
            st.caches.insert(name.clone(), v);
        }
        st.pos += rows;
        let logits = match logits {
            Some(l) => l,
            None => self.host_logits(&hidden),
        };
        Ok((hidden, logits))
    }

    /// The tied LM head on the host: `embed · hidden` for every token, rows
    /// decoded from the (packed) table across threads.
    fn host_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let (vocab, h) = (self.cfg.vocab_size, self.cfg.lm.hidden);
        let mut out = vec![0f32; vocab];
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunk = vocab.div_ceil(threads);
        std::thread::scope(|s| {
            for (c, part) in out.chunks_mut(chunk).enumerate() {
                let table = &self.embed;
                s.spawn(move || {
                    for (j, v) in part.iter_mut().enumerate() {
                        *v = table.dot((c * chunk + j) * h, hidden) as f32;
                    }
                });
            }
        });
        out
    }

    /// The prompt's input embeddings.
    pub fn prompt_embeds(&self, segments: &[Segment]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        for s in segments {
            match s {
                Segment::Tokens(ids) => out.extend(self.embed_tokens(ids)?),
                Segment::Audio(e) => out.extend_from_slice(e),
                Segment::Frames(fs) => {
                    for f in fs {
                        out.extend(self.embed_frame(f));
                    }
                }
            }
        }
        Ok(out)
    }

    /// The prompt's last token, which picks the generation mode.
    fn last_token(&self, segments: &[Segment]) -> Option<u32> {
        let t = &self.cfg.tokens;
        match segments.last()? {
            Segment::Tokens(ids) => ids.last().copied(),
            Segment::Audio(_) => Some(t.audio),
            Segment::Frames(_) => Some(t.audio_out),
        }
    }

    // ------------------------------------------------------------- speech out

    /// Depth-decoder logits `[codebooks, codebook_size]` for one hidden state,
    /// with the codebooks in `revealed` showing their code in `codes`.
    pub fn depth_logits(
        &mut self,
        hidden: &[f32],
        codes: &[u32],
        revealed: &[bool],
    ) -> Result<Vec<f32>> {
        if self.depth.is_none() {
            let _t = Timer::new(|| "compile depth decoder".into());
            let (compiled, ()) = self.compile_stage(&self.mmproj_path, "lfm3a_depth", |gb| {
                stages::depth(gb, &self.cfg)
            })?;
            self.depth = Some(compiled);
        }
        let (nq, cs, d) = (
            self.cfg.num_codebooks,
            self.cfg.codebook_size,
            self.cfg.depth.hidden,
        );
        let mut slots = vec![0f32; nq * d];
        for q in 0..nq {
            let row = &mut slots[q * d..(q + 1) * d];
            if revealed[q] {
                let at = (q * cs + codes[q] as usize) * d;
                self.depth_table.read_into(at, row);
            } else {
                row.copy_from_slice(&self.mask_embed);
            }
        }
        let g = self.depth.as_mut().expect("compiled above");
        let mut out = g.run(&[("hidden", hidden), ("slots", &slots)]);
        Ok(out.swap_remove(0))
    }

    /// Sample one speech frame (`sample_audio_frame`): codebooks revealed over
    /// the MaskGIT passes, each pass scoring the still-hidden ones.
    pub fn sample_frame(
        &mut self,
        hidden: &[f32],
        s: Sampling,
        history: &[Vec<u32>],
        order: RevealOrder,
        rng: &mut Rng,
    ) -> Result<Vec<u32>> {
        let (nq, cs) = (self.cfg.num_codebooks, self.cfg.codebook_size);
        let order = match order {
            RevealOrder::Random => rng.permutation(nq),
            RevealOrder::Index => (0..nq).collect(),
        };
        let mut codes = vec![0u32; nq];
        let mut revealed = vec![false; nq];
        let mut at = 0;
        for &n in &self.cfg.maskgit_schedule.clone() {
            let logits = self.depth_logits(hidden, &codes, &revealed)?;
            let pick = &order[at..at + n];
            for &q in pick {
                let mut row = logits[q * cs..(q + 1) * cs].to_vec();
                codes[q] = sample(&mut row, s, history.iter().map(|f| f[q]), &[], rng);
            }
            for &q in pick {
                revealed[q] = true;
            }
            at += n;
        }
        Ok(codes)
    }

    // ------------------------------------------------------------- generation

    /// Generate after `segments` (`generate` of the reference, one row).
    ///
    /// The mode follows the prompt's last token: `<|audio_start|>` or a
    /// speech frame → speech, `<|mixed_start|>` → interleaved, else text.
    /// `on_text` sees each text token as it is written, `on_frame` each
    /// speech frame.
    pub fn generate(
        &mut self,
        segments: &[Segment],
        opts: &GenerateOptions,
        mut on_text: impl FnMut(u32),
        mut on_frame: impl FnMut(&[u32]),
    ) -> Result<Generated> {
        let t = self.cfg.tokens.clone();
        let eos_code = self.cfg.audio_eos_code();
        let (n_text, n_audio) = (6, 12);
        let prompt = self.prompt_embeds(segments)?;
        let prompt_len = prompt.len() / self.cfg.lm.hidden;
        let mut st = self.lm_state(prompt_len + opts.max_new_tokens + 1)?;
        let (mut hidden, mut logits) = self.lm_feed(&mut st, &prompt)?;

        let last = self.last_token(segments);
        let mut mode = match last {
            Some(x) if x == t.audio_start || x == t.audio_out => Mode::Audio,
            Some(x) if x == t.mixed_start => Mode::Interleaved,
            _ => Mode::Text,
        };
        if let Some(Segment::Frames(fs)) = segments.last()
            && fs.last().is_some_and(|f| f.contains(&eos_code))
        {
            mode = Mode::Text;
        }
        let mut suppress = vec![t.audio, t.audio_out];
        if opts.text_only {
            suppress.extend([t.mixed_start, t.audio_start]);
        }

        let mut rng = Rng::new(opts.seed);
        let mut out = Generated::default();
        let mut text_history: Vec<u32> = Vec::new();
        let mut frame_history: Vec<Vec<u32>> = Vec::new();
        let (mut slots_left, mut speaking, mut text_done) = (n_text, false, false);
        for step in 0..opts.max_new_tokens {
            let interleaving = mode == Mode::Interleaved;
            let mut finished = false;
            let next = if mode == Mode::Audio || (interleaving && speaking) {
                let lo = frame_history
                    .len()
                    .saturating_sub(opts.speech_repetition_window);
                let mut codes = self.sample_frame(
                    &hidden,
                    opts.speech,
                    &frame_history[lo..],
                    opts.reveal,
                    &mut rng,
                )?;
                if codes.contains(&eos_code) {
                    codes.fill(eos_code);
                    mode = Mode::Text;
                } else {
                    on_frame(&codes);
                    frame_history.push(codes.clone());
                    out.frames.push(codes.clone());
                    out.tokens.push(t.audio_out);
                }
                self.embed_frame(&codes)
            } else {
                let lo = text_history
                    .len()
                    .saturating_sub(opts.text_repetition_window);
                let tok = sample(
                    &mut logits,
                    opts.text,
                    text_history[lo..].iter().copied(),
                    &suppress,
                    &mut rng,
                );
                text_history.push(tok);
                out.tokens.push(tok);
                on_text(tok);
                if tok == t.text_end {
                    if interleaving {
                        text_done = true;
                        speaking = true;
                    } else {
                        finished = true;
                    }
                } else if tok == t.im_end {
                    finished = true;
                } else if interleaving {
                    // Mode markers only open a block; inside one they are text.
                } else if tok == t.audio_start {
                    mode = Mode::Audio;
                } else if tok == t.mixed_start {
                    mode = Mode::Interleaved;
                    (slots_left, speaking, text_done) = (n_text, false, false);
                }
                self.embed_tokens(&[tok])?
            };
            if interleaving && !text_done {
                slots_left -= 1;
                if slots_left == 0 {
                    speaking = !speaking;
                    slots_left = if speaking { n_audio } else { n_text };
                }
            }
            if finished || step + 1 == opts.max_new_tokens {
                break;
            }
            (hidden, logits) = self.lm_feed(&mut st, &next)?;
        }
        Ok(out)
    }

    // ------------------------------------------------------------- codec

    /// FSQ latents `[frames, codebooks * fsq_dim]` of code frames.
    pub fn fsq_latents(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        let d = self.cfg.fsq_dim;
        let entries = self.fsq.len() / d;
        let mut out = Vec::with_capacity(frames.len() * frames.first().map_or(0, Vec::len) * d);
        for f in frames {
            for &c in f {
                let c = c as usize;
                if c >= entries {
                    bail!("code {c} is not a codec code (end-of-audio frame?)");
                }
                out.extend_from_slice(&self.fsq[c * d..(c + 1) * d]);
            }
        }
        Ok(out)
    }

    /// Decode speech frames to 24 kHz audio (1920 samples per frame).
    pub fn decode_frames(&mut self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        let n = frames.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let bucket = n.div_ceil(CODEC_BUCKET) * CODEC_BUCKET;
        if !self.codecs.contains_key(&bucket) {
            let _t = Timer::new(|| format!("compile codec frames={bucket}"));
            let (compiled, ()) = self.compile_stage(&self.mmproj_path, "lfm3a_codec", |gb| {
                stages::codec(gb, &self.cfg, bucket)
            })?;
            self.codecs.insert(bucket, compiled);
        }
        let lat_dim = self.cfg.num_codebooks * self.cfg.fsq_dim;
        let mut lat = self.fsq_latents(frames)?;
        lat.resize(bucket * lat_dim, 0.0);
        let g = self.codecs.get_mut(&bucket).expect("compiled above");
        let spec = g.run(&[("latents", &lat)]).swap_remove(0);
        if self.low_memory {
            self.codecs.clear();
        }
        let bins = CODEC_N_FFT / 2 + 1;
        Ok(istft(&spec[..n * CODEC_FRAME_STACK * 2 * bins], bins))
    }
}

/// Whether `device` scores the LM head on the host, the step graph stopping
/// at the hidden state. The tied head is 128000 × 2048: CoreML refuses to
/// bake a weight past 64M elements, and WebGL cannot hold its 278 MB (Q8_0)
/// in a texture under the 16384² limit.
fn host_lm_head(device: Device) -> bool {
    matches!(device, Device::Ane | Device::OpenGl)
}

/// A `[rows, row_elems]` lookup table on the host at its stored width: a
/// quant stays packed (rows decode on demand), f16 / bf16 is read as is —
/// neither passes through an f32 copy. A quant whose rows are not whole
/// blocks is widened to f32.
fn host_table(w: &mut GgufLoader, key: &str, row_elems: usize) -> Result<HostTable> {
    use rlx_gguf::GgmlType;
    let dtype = w
        .gguf_file()
        .and_then(|f| f.get(key))
        .map(|t| t.dtype)
        .ok_or_else(|| anyhow!("{key} missing"))?;
    if dtype != GgmlType::F32 {
        let mut bytes = Vec::new();
        graph::read_tensor_bytes(w, key, &mut bytes)?;
        fn halves(b: &[u8]) -> impl Iterator<Item = u16> + '_ {
            b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c))
        }
        match dtype {
            GgmlType::F16 => {
                return Ok(HostTable::F16(
                    halves(&bytes).map(half::f16::from_bits).collect(),
                ));
            }
            GgmlType::BF16 => {
                return Ok(HostTable::Bf16(
                    halves(&bytes).map(half::bf16::from_bits).collect(),
                ));
            }
            _ => {
                if let Some(t) = HostTable::packed(bytes, dtype, row_elems) {
                    return Ok(t);
                }
            }
        }
    }
    let (data, _) = w.take(key)?;
    Ok(HostTable::F32(data))
}

/// `Mixer` kinds of the LM, for diagnostics.
pub fn layer_summary(cfg: &Lfm3AudioConfig) -> String {
    cfg.lm
        .layers
        .iter()
        .map(|m| match m {
            Mixer::Conv => "c".to_string(),
            Mixer::Attn { window: None, .. } => "A".to_string(),
            Mixer::Attn {
                window: Some(_), ..
            } => "s".to_string(),
        })
        .collect()
}
