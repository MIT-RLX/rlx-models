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

//! The d1 engine: state + questions (+ pictures) in, calibrated typed
//! answers out, one forward pass per question and nothing decoded.
//!
//! The backbone is LFM2.5-VL-3B: the text model runs through
//! [`rlx_lfm::Lfm2GgufRunner`] (embeddings in, final hidden out), the
//! vision half through [`crate::vision::VisionEncoder`]. Each question is
//! read at the last position of `state ++ question`; d1's reference packs the
//! questions of one state into a tree that reads the state once, which is
//! the same function as one causal pass per question — what this does.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_core::ordered_json::Json;
use rlx_core::weight_storage::WeightStorage;
use rlx_lfm::Lfm2GgufRunner;
use rlx_runtime::Device;

use crate::prompt::{self, PromptStyle, Question, TextTokenizer};
use crate::vision::{self, IMAGE_TOKEN, ProcessedImage, RgbImage, VisionEncoder};

/// `tokenizer.json` through the `tokenizers` crate, no special-token wrapping.
pub struct HfTokenizer(tokenizers::Tokenizer);

impl HfTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        tokenizers::Tokenizer::from_file(path)
            .map(Self)
            .map_err(|e| anyhow!("loading {}: {e}", path.display()))
    }
}

impl TextTokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .0
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }
}

/// One request's per-question probabilities and the tokens it read.
#[derive(Debug, Clone)]
pub struct Decision {
    /// One distribution per question, over its options (`yes`, `no` for a
    /// noul), in the questions' order.
    pub probabilities: Vec<Vec<f64>>,
    /// What d1 reports as `input_tokens`: the whole row for a single
    /// question, otherwise the state's tokens once plus every question's.
    pub input_tokens: usize,
}

/// Which checkpoint files [`D1::open_with`] uses.
#[derive(Debug, Clone, Default)]
pub struct OpenOptions {
    /// Pick the text GGUF whose name contains this (`Q4_K_M`, `Q8_0`, …) in a
    /// directory that holds several. Also selects GGUF over a
    /// `model.safetensors` in the same directory.
    pub prefer: Option<String>,
    /// The vision `mmproj` GGUF; by default the `mmproj-*.gguf` next to the
    /// text model (same quantization if there is one, else F16, else any).
    pub mmproj: Option<PathBuf>,
    /// How dense weight matrices are stored (text and vision): `Auto` (the
    /// default) keeps a bf16 / f16 checkpoint at its own width — lossless, half
    /// the memory of f32; `F32` expands; `F16` / `Bf16` convert. GGUF
    /// quantized projections stay packed under every setting.
    pub weight_storage: WeightStorage,
}

/// Where the vision half comes from.
#[derive(Debug, Clone)]
enum VisionSource {
    /// The HF checkpoint directory itself.
    Hf,
    /// An llama.cpp mmproj GGUF.
    Mmproj(PathBuf),
    /// Text-only GGUF: no mmproj found.
    Missing,
}

/// The text GGUF to use in `dir`, if `dir` holds GGUFs rather than (or as
/// well as, with `prefer`) a safetensors checkpoint.
fn pick_text_gguf(dir: &Path, prefer: Option<&str>) -> Result<Option<PathBuf>> {
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("listing {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "gguf")
                && !file_name(p).to_ascii_lowercase().starts_with("mmproj")
        })
        .collect();
    ggufs.sort();
    let has_safetensors = dir.join("model.safetensors").is_file()
        || dir.join("model.safetensors.index.json").is_file();
    if ggufs.is_empty() || (has_safetensors && prefer.is_none()) {
        return Ok(None);
    }
    if let Some(tag) = prefer {
        let tag = tag.to_ascii_lowercase();
        return ggufs
            .into_iter()
            .find(|p| file_name(p).to_ascii_lowercase().contains(&tag))
            .map(Some)
            .ok_or_else(|| anyhow!("no text GGUF matching `{tag}` in {}", dir.display()));
    }
    if ggufs.len() == 1 {
        return Ok(ggufs.pop());
    }
    let names: Vec<String> = ggufs.iter().map(|p| file_name(p)).collect();
    bail!(
        "{} holds several text GGUFs ({}); choose one with --prefer",
        dir.display(),
        names.join(", ")
    )
}

/// The `mmproj-*.gguf` beside `text`: the same quantization tag when there
/// is one, else F16, else BF16, else the first.
fn pick_mmproj(text: &Path) -> Option<PathBuf> {
    let dir = text.parent()?;
    let mut cands: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "gguf")
                && file_name(p).to_ascii_lowercase().starts_with("mmproj")
        })
        .collect();
    cands.sort();
    let stem = file_name(text).to_ascii_uppercase();
    let tag = stem
        .trim_end_matches(".GGUF")
        .rsplit('-')
        .next()
        .unwrap_or_default()
        .to_string();
    let find = |t: &str| {
        cands
            .iter()
            .find(|p| file_name(p).to_ascii_uppercase().contains(t))
            .cloned()
    };
    find(&format!("-{tag}."))
        .or_else(|| find("-F16."))
        .or_else(|| find("-BF16."))
        .or_else(|| cands.first().cloned())
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// A loaded d1 checkpoint.
pub struct D1 {
    /// Where `tokenizer.json`, `config.json` and `processor_config.json` are.
    dir: PathBuf,
    device: Device,
    text: Lfm2GgufRunner,
    tokenizer: HfTokenizer,
    style: PromptStyle,
    image_token_id: u32,
    vision_source: VisionSource,
    vision: Option<VisionEncoder>,
    weight_storage: WeightStorage,
}

impl D1 {
    /// Open a d1 (or any LFM2-VL) checkpoint, see [`Self::open_with`].
    pub fn open(path: impl Into<PathBuf>, device: Device) -> Result<Self> {
        Self::open_with(path, device, &OpenOptions::default())
    }

    /// Open a d1 checkpoint in either form `LiquidAI` ships:
    ///
    /// * the HF directory (`LiquidAI/d1-3B`): `config.json`,
    ///   `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
    ///   `processor_config.json`;
    /// * the quantized GGUFs (`LiquidAI/d1-3B-GGUF`): a text GGUF
    ///   (`d1-3B-Q4_K_M.gguf`, `-Q8_0`, `-F16`, `-BF16`) — the path itself,
    ///   or one picked in a directory — with an `mmproj-*.gguf` for pictures.
    ///   The GGUF repo has no tokenizer files; `tokenizer.json` (and
    ///   `tokenizer_config.json`, `processor_config.json`) from the HF repo go
    ///   next to it (`just fetch-d1-gguf` does this).
    ///
    /// GGUF projections stay packed (`DequantMatMul` on the quantized
    /// blocks); the vision tower loads on the first picture.
    pub fn open_with(path: impl Into<PathBuf>, device: Device, opts: &OpenOptions) -> Result<Self> {
        let path = path.into();
        let (dir, text_gguf) = if path.is_file() {
            ensure!(
                path.extension().is_some_and(|e| e == "gguf"),
                "{} is neither a checkpoint directory nor a .gguf",
                path.display()
            );
            let dir = path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            (dir, Some(path.clone()))
        } else {
            let pick = pick_text_gguf(&path, opts.prefer.as_deref())?;
            (path.clone(), pick)
        };
        let (mut text, vision_source) = match &text_gguf {
            Some(g) => {
                let text = Lfm2GgufRunner::open(g, device)
                    .with_context(|| format!("loading the text model from {}", g.display()))?;
                let mmproj = opts.mmproj.clone().or_else(|| pick_mmproj(g));
                let vs = mmproj.map_or(VisionSource::Missing, VisionSource::Mmproj);
                (text, vs)
            }
            None => {
                let text = Lfm2GgufRunner::open(&dir, device)
                    .with_context(|| format!("loading the text model from {}", dir.display()))?;
                (text, VisionSource::Hf)
            }
        };
        text.set_weight_storage(opts.weight_storage);
        let tok_path = dir.join("tokenizer.json");
        ensure!(
            tok_path.is_file(),
            "{} has no tokenizer.json: the GGUF release ships none — copy it from \
             LiquidAI/d1-3B (`just fetch-d1-gguf`)",
            dir.display()
        );
        let tokenizer = HfTokenizer::from_file(&tok_path)?;
        let read_json = |name: &str| -> Result<Option<serde_json::Value>> {
            let p = dir.join(name);
            if !p.is_file() {
                return Ok(None);
            }
            let raw = std::fs::read_to_string(&p).with_context(|| format!("reading {name}"))?;
            Ok(Some(
                serde_json::from_str(&raw).with_context(|| format!("parsing {name}"))?,
            ))
        };
        let config = read_json("config.json")?;
        ensure!(
            config.is_some() || text_gguf.is_some(),
            "{} has no config.json",
            dir.display()
        );
        let bos = match read_json("tokenizer_config.json")? {
            Some(tc) => tc
                .get("bos_token")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string(),
            // d1's BOS; the template writes it as text.
            None => "<|startoftext|>".to_string(),
        };
        let image_token_id = match config
            .as_ref()
            .and_then(|c| c.get("image_token_id"))
            .and_then(serde_json::Value::as_u64)
        {
            Some(id) => id as u32,
            None => {
                let ids = tokenizer.encode(IMAGE_TOKEN)?;
                ensure!(ids.len() == 1, "`{IMAGE_TOKEN}` is not a single token");
                ids[0]
            }
        };
        Ok(Self {
            dir,
            device,
            text,
            tokenizer,
            style: PromptStyle {
                bos,
                ..PromptStyle::default()
            },
            image_token_id,
            vision_source,
            vision: None,
            weight_storage: opts.weight_storage,
        })
    }

    pub fn tokenizer(&self) -> &HfTokenizer {
        &self.tokenizer
    }

    pub fn style(&self) -> &PromptStyle {
        &self.style
    }

    /// Change how prompts are rendered (state style, system turn, options).
    /// The defaults are the ones d1 was trained with.
    pub fn style_mut(&mut self) -> &mut PromptStyle {
        &mut self.style
    }

    fn vision(&mut self) -> Result<&mut VisionEncoder> {
        if self.vision.is_none() {
            let enc = match &self.vision_source {
                VisionSource::Hf => {
                    VisionEncoder::load(&self.dir, self.device, self.weight_storage)
                }
                VisionSource::Mmproj(p) => {
                    VisionEncoder::load_mmproj(p, Some(&self.dir), self.device, self.weight_storage)
                }
                VisionSource::Missing => bail!(
                    "pictures need the vision tower: no mmproj-*.gguf next to the text \
                     GGUF in {} (or pass one explicitly)",
                    self.dir.display()
                ),
            };
            self.vision = Some(enc.context("loading the vision tower")?);
        }
        Ok(self.vision.as_mut().expect("just loaded"))
    }

    /// The vision processor's settings (read from the checkpoint).
    pub fn processor_config(&self) -> Result<vision::ProcessorConfig> {
        vision::ProcessorConfig::from_dir(&self.dir)
    }

    /// Each question's distribution over its options, one pass per question.
    pub fn decide(
        &mut self,
        state: &Json,
        questions: &[Question],
        images: &[RgbImage],
    ) -> Result<Decision> {
        ensure!(!questions.is_empty(), "no questions");
        let groups: Vec<Vec<Vec<u32>>> = questions
            .iter()
            .map(|q| prompt::readout_ids(&self.tokenizer, q))
            .collect::<Result<_>>()?;
        let suffixes: Vec<String> = questions
            .iter()
            .map(|q| prompt::suffix_text(&self.tokenizer, q, &self.style))
            .collect::<Result<_>>()?;

        if images.is_empty() {
            let prefix = prompt::prefix_text(state, &self.style, "");
            if questions.len() == 1 {
                // The whole prompt tokenized as one string.
                let ids = self.tokenizer.encode(&(prefix + &suffixes[0]))?;
                let probs = self.read_row(&self.text.embed_ids(&ids)?, &groups[0])?;
                return Ok(Decision {
                    probabilities: vec![probs],
                    input_tokens: ids.len(),
                });
            }
            // The state tokenized once, each question on its own: d1's tree.
            let trunk = self.tokenizer.encode(&prefix)?;
            return self.branches(
                &self.text.embed_ids(&trunk)?,
                trunk.len(),
                &suffixes,
                &groups,
            );
        }

        let processor = self.processor_config()?;
        let pics: Vec<ProcessedImage> = images
            .iter()
            .map(|im| {
                vision::process_image(
                    &vision::cap_pixels(im, vision::VISION_MAX_PIXELS),
                    &processor,
                )
            })
            .collect::<Result<_>>()?;
        // What the chat template writes for each picture at the head of the turn.
        let prefix = prompt::prefix_text(state, &self.style, &IMAGE_TOKEN.repeat(pics.len()));
        let mut features = Vec::new();
        for p in &pics {
            features.extend(self.vision()?.encode(p)?);
        }
        if questions.len() == 1 {
            let ids = self.expand_and_encode(&(prefix + &suffixes[0]), &pics, &processor)?;
            let embeds = self.embed_with_images(&ids, &features)?;
            let probs = self.read_row(&embeds, &groups[0])?;
            return Ok(Decision {
                probabilities: vec![probs],
                input_tokens: ids.len(),
            });
        }
        let trunk = self.expand_and_encode(&prefix, &pics, &processor)?;
        let embeds = self.embed_with_images(&trunk, &features)?;
        self.branches(&embeds, trunk.len(), &suffixes, &groups)
    }

    /// Questions continuing a shared trunk (given as embeddings), each read
    /// at its own last token.
    fn branches(
        &self,
        trunk: &[f32],
        trunk_tokens: usize,
        suffixes: &[String],
        groups: &[Vec<Vec<u32>>],
    ) -> Result<Decision> {
        let mut probabilities = Vec::with_capacity(suffixes.len());
        let mut input_tokens = trunk_tokens;
        for (suffix, g) in suffixes.iter().zip(groups) {
            let ids = self.tokenizer.encode(suffix)?;
            input_tokens += ids.len();
            let mut row = trunk.to_vec();
            row.extend(self.text.embed_ids(&ids)?);
            probabilities.push(self.read_row(&row, g)?);
        }
        Ok(Decision {
            probabilities,
            input_tokens,
        })
    }

    /// The token ids of a rendered prompt, with each `<image>` expanded to
    /// its picture's token layout as the processor does it.
    pub fn prompt_ids(&self, text: &str, images: &[RgbImage]) -> Result<Vec<u32>> {
        if images.is_empty() {
            return self.tokenizer.encode(text);
        }
        let processor = self.processor_config()?;
        let pics: Vec<ProcessedImage> = images
            .iter()
            .map(|im| {
                vision::process_image(
                    &vision::cap_pixels(im, vision::VISION_MAX_PIXELS),
                    &processor,
                )
            })
            .collect::<Result<_>>()?;
        self.expand_and_encode(text, &pics, &processor)
    }

    /// `<image>` expanded to each picture's token layout, then tokenized as
    /// one string (`Lfm2VlProcessor.__call__`).
    fn expand_and_encode(
        &self,
        text: &str,
        pics: &[ProcessedImage],
        cfg: &vision::ProcessorConfig,
    ) -> Result<Vec<u32>> {
        let parts: Vec<&str> = text.split(IMAGE_TOKEN).collect();
        ensure!(
            parts.len() == pics.len() + 1,
            "the prompt holds {} `{IMAGE_TOKEN}` for {} pictures",
            parts.len() - 1,
            pics.len()
        );
        let mut expanded = String::from(parts[0]);
        for (p, rest) in pics.iter().zip(&parts[1..]) {
            expanded.push_str(&p.placeholder(cfg));
            expanded.push_str(rest);
        }
        self.tokenizer.encode(&expanded)
    }

    /// Token embeddings with the image features scattered over the image
    /// tokens, in order.
    fn embed_with_images(&self, ids: &[u32], features: &[f32]) -> Result<Vec<f32>> {
        let h = self.text.config().hidden_size;
        let mut embeds = self.text.embed_ids(ids)?;
        let slots: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == self.image_token_id).then_some(i))
            .collect();
        ensure!(
            slots.len() * h == features.len(),
            "{} image tokens for {} image features",
            slots.len(),
            features.len() / h
        );
        for (k, &i) in slots.iter().enumerate() {
            embeds[i * h..(i + 1) * h].copy_from_slice(&features[k * h..(k + 1) * h]);
        }
        Ok(embeds)
    }

    /// One causal pass over `embeds`, scored at the last position against the
    /// options' token rows of the tied LM head.
    fn read_row(&self, embeds: &[f32], groups: &[Vec<u32>]) -> Result<Vec<f64>> {
        let h = self.text.config().hidden_size;
        let hidden = self.text.hidden_from_embeds(embeds)?;
        let last = &hidden[hidden.len() - h..];
        let table = self.text.embed_table()?;
        Ok(prompt::readout(groups, |id| {
            table.dot(id as usize * h, last)
        }))
    }

    /// d1's `system_one`: named questions (a JSON object of Decision Index
    /// questions, in order) over one state, and its pictures if any.
    ///
    /// Returns `{"answers": {name: answer}, "usage": {"input_tokens": n,
    /// "output_tokens": 0}}` with the same fields as the reference.
    pub fn system_one(
        &mut self,
        state: &Json,
        questions: &Json,
        images: &[RgbImage],
    ) -> Result<Json> {
        let named = questions
            .as_object()
            .ok_or_else(|| anyhow!("questions must be an object of name -> question"))?;
        let qs: Vec<Question> = named
            .iter()
            .map(|(n, q)| Question::from_json(q).with_context(|| format!("question `{n}`")))
            .collect::<Result<_>>()?;
        let d = self.decide(state, &qs, images)?;
        let answers = named
            .iter()
            .zip(&qs)
            .zip(&d.probabilities)
            .map(|(((n, _), q), p)| (n.clone(), crate::answer(q, p)))
            .collect();
        Ok(Json::Object(vec![
            ("answers".into(), Json::Object(answers)),
            (
                "usage".into(),
                Json::Object(vec![
                    ("input_tokens".into(), Json::UInt(d.input_tokens as u64)),
                    ("output_tokens".into(), Json::UInt(0)),
                ]),
            ),
        ]))
    }
}
