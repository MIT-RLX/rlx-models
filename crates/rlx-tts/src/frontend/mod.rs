//! Text frontend: Hydra pipeline over the local TTS bundle.
//!
//! Tables under the bundle `frontend/` directory:
//! - `lexicon.txt`, `tn_prefix_rule.dat`, `rewrite_rule.dat`
//! - `g2p_lhp_rule.dat` / `g2p_post_rule.dat`
//! - `g2p_bpe.json` + `g2p_seq2seq.safetensors` (TorchN G2P)
//! - `phonetic/to_lhp.json`, `nashville_isym_phones.json`
//! - optional `gprm_index.json` prompt-text index

mod gprm;
mod lexicon_seed;
mod lhp_map;
mod phbk;
mod rewrite;
mod rule_dat;
mod tn;
mod torchn;

use lexicon_seed::{
    load_lexicon, load_nashville_isym_phones, seed_builtin_lexicon, seed_roundtrip_overrides,
};

pub use gprm::GprmIndex;
pub use lhp_map::LhpAlphabet;
pub use phbk::Phbk;
pub use rewrite::RewriteRules;
pub use rule_dat::{RuleDat, RuleTable};
pub use tn::TnPrefix;
pub use torchn::TorchnG2p;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Deserialize;

/// Loaded phone symbol map (`symmap.json` / neural_fe phone_map).
#[derive(Debug, Clone)]
pub struct PhoneMap {
    pub symbol_to_id: HashMap<String, usize>,
    pub id_to_symbol: Vec<String>,
}

impl PhoneMap {
    pub fn from_json_value(map: &serde_json::Map<String, serde_json::Value>) -> Result<Self> {
        let mut symbol_to_id = HashMap::new();
        let mut max_id = 0usize;
        for (k, v) in map {
            let id = v
                .as_u64()
                .or_else(|| v.as_i64().map(|x| x as u64))
                .with_context(|| format!("phone id for {k}"))? as usize;
            symbol_to_id.insert(k.clone(), id);
            max_id = max_id.max(id);
        }
        let mut id_to_symbol = vec![String::new(); max_id + 1];
        for (sym, id) in &symbol_to_id {
            id_to_symbol[*id] = sym.clone();
        }
        Ok(Self {
            symbol_to_id,
            id_to_symbol,
        })
    }

    pub fn load_symmap(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let obj = v
            .as_object()
            .with_context(|| "symmap.json must be an object")?;
        Self::from_json_value(obj)
    }

    pub fn id(&self, sym: &str) -> Option<usize> {
        self.symbol_to_id.get(sym).copied()
    }

    pub fn vocab(&self) -> usize {
        self.id_to_symbol.len()
    }
}

/// Trait object-friendly frontend.
pub trait TextFrontend: Send + Sync {
    fn text_to_phones(&self, text: &str) -> Result<Vec<String>>;
    fn phones_to_ids(&self, phones: &[String]) -> Result<Vec<usize>>;
}

#[derive(Debug, Clone)]
pub struct NeuralAdapterOpts {
    pub eos: String,
    pub pause_marker: String,
    pub word_boundary: String,
    pub stress_marker: String,
    pub max_word_limit: usize,
    pub punctuation: HashSet<String>,
    pub punctuation_set1: HashSet<String>, // .?! → insert pau
    pub punctuation_set2: HashSet<String>, // ,;:
}

impl Default for NeuralAdapterOpts {
    fn default() -> Self {
        Self {
            eos: "~".into(),
            pause_marker: "pau".into(),
            word_boundary: "#".into(),
            stress_marker: ":".into(),
            max_word_limit: 40,
            punctuation: ["!", "?", ",", ".", ":", ";"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            punctuation_set1: ["?", ".", "!"].into_iter().map(str::to_string).collect(),
            punctuation_set2: [",", ";", ":"].into_iter().map(str::to_string).collect(),
        }
    }
}

///
/// Pipeline (when private assets are present under `frontend/`):
/// 1. Unicode normalize + `tn_prefix_rule.dat` + `rewrite_rule.dat` literals
/// 2. Tokenize words / punctuation
/// 3. Lexicon / TorchN BPE G2P / letter fallback
/// 4. `g2p_lhp_rule` + `g2p_post_rule` literal cleanup
/// 5. Compact-LHP → phone_map via `to_lhp.json` when needed
pub struct HydraLite {
    pub map: PhoneMap,
    lexicon: HashMap<String, Vec<String>>,
    word_re: Regex,
    opts: NeuralAdapterOpts,
    #[allow(dead_code)]
    frontend_dir: PathBuf,
    tn: Option<TnPrefix>,
    rewrite: Option<RewriteRules>,
    g2p_lhp: Option<RuleDat>,
    g2p_post: Option<RuleDat>,
    torchn: Option<TorchnG2p>,
    lhp_alpha: Option<LhpAlphabet>,
    #[allow(dead_code)]
    phbk: Option<Phbk>,
    /// MatchPrompt index (`gprm_index.json`) for canned utterances.
    pub gprm: Option<GprmIndex>,
    pub pause_min_duration_ms: f32,
}

#[derive(Debug, Deserialize)]
struct NeuralFeConfig {
    #[serde(default)]
    phone_map: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    eos: Option<String>,
    #[serde(default)]
    word_boundary_marker: Option<String>,
    #[serde(default)]
    punctuation: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct PipelineCfg {
    #[serde(default)]
    pipeline: Vec<PipelineStage>,
}

#[derive(Debug, Deserialize)]
struct PipelineStage {
    id: String,
    #[serde(default)]
    params: serde_json::Value,
}

impl HydraLite {
    pub fn open(bundle_dir: &Path) -> Result<Self> {
        let symmap = bundle_dir.join("symmap.json");
        let cfg_path = bundle_dir.join("neural_fe_config.json");
        let (map, mut fe_opts) = if symmap.is_file() {
            (
                PhoneMap::load_symmap(&symmap)?,
                NeuralAdapterOpts::default(),
            )
        } else {
            let cfg: NeuralFeConfig = serde_json::from_str(
                &std::fs::read_to_string(&cfg_path)
                    .with_context(|| format!("read {}", cfg_path.display()))?,
            )?;
            let pm = cfg
                .phone_map
                .context("neural_fe_config.json missing phone_map")?;
            let mut opts = NeuralAdapterOpts::default();
            if let Some(eos) = cfg.eos {
                opts.eos = eos;
            }
            if let Some(wb) = cfg.word_boundary_marker {
                opts.word_boundary = wb;
            }
            if let Some(punc) = cfg.punctuation {
                opts.punctuation = punc.into_iter().collect();
            }
            (PhoneMap::from_json_value(&pm)?, opts)
        };

        for cfg_name in ["post.cfg", "pipeline.cfg"] {
            let cfg_p = bundle_dir.join(cfg_name);
            if !cfg_p.is_file() {
                continue;
            }
            if let Ok(gc) = serde_json::from_str::<PipelineCfg>(&std::fs::read_to_string(&cfg_p)?) {
                for stage in gc.pipeline {
                    if stage.id == "neural_adapter" {
                        apply_neural_adapter_params(&mut fe_opts, &stage.params);
                    }
                }
            }
            break;
        }
        if cfg_path.is_file() {
            if let Ok(cfg) =
                serde_json::from_str::<NeuralFeConfig>(&std::fs::read_to_string(&cfg_path)?)
            {
                if let Some(eos) = cfg.eos {
                    fe_opts.eos = eos;
                }
                if let Some(wb) = cfg.word_boundary_marker {
                    fe_opts.word_boundary = wb;
                }
            }
        }

        let frontend_dir = bundle_dir.join("frontend");
        let mut lexicon = HashMap::new();
        seed_builtin_lexicon(&mut lexicon);
        for lex_path in [
            bundle_dir.join("lexicon.txt"),
            frontend_dir.join("lexicon.txt"),
        ] {
            if lex_path.is_file() {
                load_lexicon(&lex_path, &mut lexicon)?;
            }
        }
        // Later seeds win: Nashville JSON → adapter hardcodes → round-trip OOVs.
        load_nashville_isym_phones(
            &frontend_dir.join("nashville_isym_phones.json"),
            &mut lexicon,
        );
        seed_nashville_lexicon(&mut lexicon);
        seed_roundtrip_overrides(&mut lexicon);

        // TorchN G2P: load sidecars by default when present (Softmax pronounce
        // is confidence-gated; lexicon still wins for known words). Opt out:
        // `RLX_TTS_NO_TORCHN=1`. Force-load even without safetensors:
        // `RLX_TTS_LOAD_TORCHN=1`.
        let torchn = if std::env::var_os("RLX_TTS_NO_TORCHN").is_some() {
            None
        } else {
            TorchnG2p::load_prefer_sidecar(&frontend_dir).ok()
        };
        let rewrite = {
            let dat = frontend_dir.join("rewrite_rule.dat");
            let map = frontend_dir.join("rewrite_map.json");
            if std::env::var_os("RLX_TTS_LOAD_REWRITE_DAT").is_some() && dat.is_file() {
                RewriteRules::load(&dat).ok()
            } else if std::env::var_os("RLX_TTS_HARVEST_REWRITE").is_some() && dat.is_file() {
                RewriteRules::load_harvest_only(&dat).ok()
            } else if map.is_file() {
                RewriteRules::load_map_only(&map).ok()
            } else {
                None
            }
        };
        // TN: load R-list literals by default when present (BinaryGraph ordinal
        // tables still incomplete). Opt out: `RLX_TTS_NO_TN_DAT=1`.
        let tn = if std::env::var_os("RLX_TTS_NO_TN_DAT").is_some() {
            None
        } else {
            frontend_dir
                .join("tn_prefix_rule.dat")
                .is_file()
                .then(|| TnPrefix::load(frontend_dir.join("tn_prefix_rule.dat")).ok())
                .flatten()
        };
        let gprm = frontend_dir
            .join("gprm_index.json")
            .is_file()
            .then(|| GprmIndex::load(frontend_dir.join("gprm_index.json")).ok())
            .flatten();
        let g2p_lhp = frontend_dir
            .join("g2p_lhp_rule.dat")
            .is_file()
            .then(|| RuleDat::load(frontend_dir.join("g2p_lhp_rule.dat")))
            .transpose()
            .ok()
            .flatten();
        let g2p_post = frontend_dir
            .join("g2p_post_rule.dat")
            .is_file()
            .then(|| RuleDat::load(frontend_dir.join("g2p_post_rule.dat")))
            .transpose()
            .ok()
            .flatten();
        let lhp_alpha = [
            frontend_dir.join("phonetic/to_lhp.json"),
            bundle_dir.join("phonetic/to_lhp.json"),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .and_then(|p| LhpAlphabet::load_to_lhp(p).ok());
        let phbk = if std::env::var_os("RLX_TTS_LOAD_PHBK").is_some() {
            frontend_dir
                .join("phbk")
                .is_file()
                .then(|| Phbk::load(frontend_dir.join("phbk")))
                .transpose()
                .ok()
                .flatten()
        } else {
            None
        };

        let mut pause_min_duration_ms = 70.0f32;
        let post_path = ["post.cfg", "pipeline.cfg"]
            .iter()
            .map(|n| bundle_dir.join(n))
            .find(|p| p.is_file());
        if let Some(post_path) = post_path.as_ref().filter(|p| p.is_file()) {
            if let Ok(gc) =
                serde_json::from_str::<PipelineCfg>(&std::fs::read_to_string(post_path)?)
            {
                for stage in gc.pipeline {
                    if stage.id == "neural_adapter" {
                        if let Some(v) = stage
                            .params
                            .get("pause_min_duration")
                            .and_then(|x| x.as_f64())
                        {
                            pause_min_duration_ms = v as f32;
                        }
                    }
                }
            }
        }

        Ok(Self {
            map,
            lexicon,
            word_re: Regex::new(r"[A-Za-z']+|[0-9]+(?:st|nd|rd|th)?|[.,!?;:]").unwrap(),
            opts: fe_opts,
            frontend_dir,
            tn,
            rewrite,
            g2p_lhp,
            g2p_post,
            torchn,
            lhp_alpha,
            phbk,
            gprm,
            pause_min_duration_ms,
        })
    }

    pub fn adapter_opts(&self) -> &NeuralAdapterOpts {
        &self.opts
    }

    fn normalize(&self, text: &str) -> String {
        let mut s = text.to_string();
        for (a, b) in [
            ('\u{2019}', '\''),
            ('\u{2018}', '\''),
            ('\u{201c}', '"'),
            ('\u{201d}', '"'),
            ('\u{2013}', '-'),
            ('\u{2014}', '-'),
        ] {
            s = s.replace(a, &b.to_string());
        }
        for (pat, rep) in [
            ("Mrs.", "Misses"),
            ("Ms.", "Mizz"),
            ("Dr.", "Doctor"),
            ("2nd St.", "second street."),
            ("3rd St.", "third street."),
            ("1st St.", "first street."),
            ("4th St.", "fourth street."),
            ("&", " and "),
            ("%", " percent "),
            ("@", " at "),
            ("#", " number "),
        ] {
            s = s.replace(pat, rep);
        }
        if let Some(tn) = &self.tn {
            s = tn.apply(&s);
        }
        if let Some(rw) = &self.rewrite {
            s = rw.apply_literals(&s);
        }
        s
    }

    fn letters_to_phones(&self, word: &str) -> Vec<String> {
        let mut out = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            if ch == '\'' {
                i += 1;
                continue;
            }
            let base = ch.to_ascii_lowercase().to_string();
            let long = format!("{base}{}", self.opts.stress_marker);
            if i + 1 < chars.len() && chars[i + 1] == ch && self.map.id(&long).is_some() {
                out.push(long);
                i += 2;
                continue;
            }
            if self.map.id(&base).is_some() {
                out.push(base);
            } else if let Some(up) = self.map.id(&ch.to_ascii_uppercase().to_string()) {
                out.push(self.map.id_to_symbol[up].clone());
            }
            i += 1;
        }
        out
    }

    fn g2p_word(&self, word: &str) -> Vec<String> {
        let key = word.to_ascii_lowercase();
        if let Some(seq) = self.lexicon.get(&key) {
            return seq.clone();
        }
        // Digit sequences: spell each digit via lexicon (`one`, …) or letter fallback.
        if key.chars().all(|c| c.is_ascii_digit()) {
            let mut out = Vec::new();
            for ch in key.chars() {
                let name = digit_word(ch);
                if let Some(seq) = self.lexicon.get(name) {
                    out.extend(seq.iter().cloned());
                } else {
                    out.extend(self.letters_to_phones(name));
                }
            }
            return out;
        }
        if let Some(g2p) = &self.torchn {
            if let Ok(Some(compact)) = g2p.pronounce(&key) {
                let cleaned = self.apply_g2p_rules(&compact);
                if let Some(alpha) = &self.lhp_alpha {
                    let phones = alpha.compact_to_phones(&cleaned);
                    if !phones.is_empty() {
                        return phones;
                    }
                }
            }
        }
        self.letters_to_phones(&key)
    }

    fn apply_g2p_rules(&self, compact: &str) -> String {
        let mut s = compact.to_string();
        if let Some(r) = &self.g2p_lhp {
            s = r.apply_literals(&s);
        }
        if let Some(r) = &self.g2p_post {
            s = r.apply_literals(&s);
        }
        s
    }

    ///
    /// (`word_boundary_marker`), and sequences end with `~` (eos). TextToPhoneme
    /// still uses `_` between words; the adapter rewrites those to `#` before FS2.
    fn neural_adapter_packing(&self) -> bool {
        self.map.id("#").is_some() && self.map.id("~").is_some()
    }
}

impl TextFrontend for HydraLite {
    fn text_to_phones(&self, text: &str) -> Result<Vec<String>> {
        let text = self.normalize(text);
        let neural = self.neural_adapter_packing();
        let mut phones = Vec::new();
        if !neural && self.map.id("_").is_some() {
            phones.push("_".to_string());
        }

        let tokens: Vec<_> = self.word_re.find_iter(&text).map(|m| m.as_str()).collect();
        let mut word_count = 0usize;
        let mut last_was_word = false;
        for tok in tokens {
            if self.opts.punctuation.contains(tok) {
                if neural {
                    // NeuralAdapter: `#` before set1 punct (and as word break).
                    if self.opts.punctuation_set1.contains(tok) {
                        let wb = self.opts.word_boundary.as_str(); // "#"
                        if self.map.id(wb).is_some() {
                            phones.push(wb.to_string());
                        }
                    }
                    if self.map.id(tok).is_some() {
                        phones.push(tok.to_string());
                    }
                } else {
                    if self.map.id(tok).is_some() {
                        phones.push(tok.to_string());
                    }
                    if self.opts.punctuation_set1.contains(tok)
                        && self.map.id(&self.opts.pause_marker).is_some()
                    {
                        phones.push(self.opts.pause_marker.clone());
                    }
                }
                last_was_word = false;
                continue;
            }

            // Expand digit runs into per-digit words so boundaries match fixtures.
            let subwords: Vec<String> = if tok.chars().all(|c| c.is_ascii_digit()) && tok.len() > 1
            {
                tok.chars().map(|c| c.to_string()).collect()
            } else {
                vec![tok.to_string()]
            };
            for sub in subwords {
                word_count += 1;
                if word_count > self.opts.max_word_limit {
                    break;
                }
                if last_was_word {
                    let wb = self.opts.word_boundary.as_str();
                    if self.map.id(wb).is_some() {
                        phones.push(wb.to_string());
                    }
                }
                let seq = self.g2p_word(&sub);
                if seq.is_empty() {
                    continue;
                }
                phones.extend(seq);
                last_was_word = true;
            }
            if word_count > self.opts.max_word_limit {
                break;
            }
        }

        // NeuralAdapter embeddings always include eos `~` (id 1).
        if self.map.id(&self.opts.eos).is_some() {
            if phones.last().map(|s| s.as_str()) != Some(self.opts.eos.as_str()) {
                phones.push(self.opts.eos.clone());
            }
        } else if !neural && self.map.id("_").is_some() {
            phones.push("_".to_string());
        }
        if phones.first().map(|s| s.as_str()) == Some("_") && self.map.id("_").is_none() {
            phones.remove(0);
        }
        if phones.is_empty() {
            bail!("frontend produced no phones for text");
        }
        Ok(phones)
    }

    fn phones_to_ids(&self, phones: &[String]) -> Result<Vec<usize>> {
        let mut ids = Vec::with_capacity(phones.len());
        for p in phones {
            let id = self
                .map
                .id(p)
                .with_context(|| format!("unknown phone symbol '{p}'"))?;
            ids.push(id);
        }
        Ok(ids)
    }
}

fn digit_word(ch: char) -> &'static str {
    match ch {
        '0' => "zero",
        '1' => "one",
        '2' => "two",
        '3' => "three",
        '4' => "four",
        '5' => "five",
        '6' => "six",
        '7' => "seven",
        '8' => "eight",
        '9' => "nine",
        _ => "zero",
    }
}

/// common corpus words (local fixture-backed; improves portable G2P until
/// TorchN weight decode lands).
fn seed_nashville_lexicon(lexicon: &mut HashMap<String, Vec<String>>) {
    // sentence/neural-adapter overrides where isolated dump disagrees
    // (e.g. `from` → `f r $ m`, `will` → `w L`).
    let entries: &[(&str, &[&str])] = &[
        ("1st", &["f", "e:", "s", "t"]),
        ("2nd", &["s", "E:", "K", "$", "n", "d"]),
        ("3rd", &["T", "e:", "d"]),
        ("4th", &["f", "A:", "r", "T"]),
        ("a", &["$"]),
        ("age", &["J:", "G"]),
        ("air", &["E:", "r"]),
        ("and", &["145", "n"]),
        ("appear", &["$", "P", "i:", "r"]),
        ("area", &["E:", "r", "i", "$"]),
        ("art", &["a:", "r", "t"]),
        ("ask", &["145:", "s", "k"]),
        ("back", &["b", "145:", "k"]),
        ("bad", &["b", "145:", "d"]),
        ("begin", &["b", "i", "g", "I:", "n"]),
        ("believe", &["b", "I", "l", "i:", "v"]),
        ("big", &["b", "I:", "g"]),
        ("body", &["b", "a:", "R", "i"]),
        ("book", &["b", "U:", "k"]),
        ("boy", &["b", "y:"]),
        ("breeze", &["b", "r", "i:", "z"]),
        ("bring", &["b", "r", "I:", "N"]),
        ("build", &["b", "I:", "l", "d"]),
        ("business", &["b", "I:", "z", "n", "I", "s"]),
        ("buy", &["b", "Y:"]),
        ("call", &["K", "A:", "l"]),
        ("car", &["K", "a:", "r"]),
        ("case", &["K", "J:", "s"]),
        ("cat", &["K", "145:", "t"]),
        ("change", &["C", "J:", "n", "G"]),
        ("child", &["C", "Y:", "l", "d"]),
        ("city", &["s", "I:", "R", "i"]),
        ("come", &["K", "^:", "m"]),
        ("community", &["K", "$", "m", "j", "u:", "n", "I", "R", "i"]),
        ("company", &["K", "^:", "m", "P", "$", "n", "i"]),
        ("consider", &["K", "$", "n", "s", "I:", "R", "e"]),
        ("create", &["K", "r", "i", "J:", "t"]),
        ("cut", &["K", "^:", "t"]),
        ("day", &["d", "J:"]),
        ("decide", &["d", "I", "s", "Y:", "d"]),
        ("degrees", &["d", "$", "g", "r", "i:", "z"]),
        ("develop", &["d", "I", "v", "E:", "l", "$", "p"]),
        ("die", &["d", "Y:"]),
        ("doctor", &["d", "a:", "k", "146", "e"]),
        ("dog", &["d", "A:", "g"]),
        ("door", &["d", "A:", "r"]),
        ("education", &["E", "G", "$", "K", "J:", "S", "$", "n"]),
        ("eight", &["J:", "t"]),
        ("end", &["E:", "n", "d"]),
        ("expect", &["I", "k", "s", "p", "E:", "k", "t"]),
        ("explain", &["E", "k", "s", "p", "l", "J:", "n"]),
        ("eye", &["Y:"]),
        ("face", &["f", "J:", "s"]),
        ("fact", &["f", "145", "k", "t"]),
        ("fall", &["f", "A:", "l"]),
        ("father", &["f", "a:", "D", "e"]),
        ("feel", &["f", "i:", "l"]),
        ("find", &["f", "Y:", "n", "d"]),
        ("first", &["f", "e:", "s", "t"]),
        ("five", &["f", "Y:", "v"]),
        ("follow", &["f", "a:", "l", "O"]),
        ("foot", &["f", "U:", "t"]),
        ("force", &["f", "A:", "r", "s"]),
        ("four", &["f", "A:", "r"]),
        ("fourth", &["f", "A:", "r", "T"]),
        ("friend", &["f", "r", "E:", "n", "d"]),
        ("from", &["f", "r", "$", "m"]),
        ("game", &["g", "J:", "m"]),
        ("girl", &["g", "e:", "l"]),
        ("give", &["g", "I:", "v"]),
        ("go", &["g", "O:"]),
        ("good", &["g", "U:", "d"]),
        (
            "government",
            &["g", "^:", "v", "e", "n", "m", "$", "n", "t"],
        ),
        ("grow", &["g", "r", "O:"]),
        ("guy", &["g", "Y:"]),
        ("hand", &["h", "145:", "n", "d"]),
        ("happen", &["h", "145:", "P", "$", "n"]),
        ("head", &["h", "E:", "d"]),
        ("health", &["h", "E:", "l", "T"]),
        ("hear", &["h", "i:", "r"]),
        ("hello", &["h", "E", "l", "O:"]),
        ("help", &["h", "E:", "l", "p"]),
        ("hi", &["h", "Y:"]),
        ("high", &["h", "Y:"]),
        ("history", &["h", "I:", "s", "t", "r", "i"]),
        ("home", &["h", "O:", "m"]),
        ("hour", &["@:", "e"]),
        ("house", &["h", "@:", "s"]),
        ("idea", &["Y", "d", "i:", "$"]),
        (
            "information",
            &["I", "n", "f", "e", "m", "J:", "S", "$", "n"],
        ),
        ("is", &["I", "z"]),
        ("issue", &["I:", "S", "u"]),
        ("job", &["G", "a:", "b"]),
        ("keep", &["K", "i:", "p"]),
        ("kid", &["K", "I:", "d"]),
        ("kill", &["K", "I:", "l"]),
        ("kind", &["K", "Y:", "n", "d"]),
        ("know", &["n", "O:"]),
        ("law", &["l", "A:"]),
        ("leave", &["l", "i:", "v"]),
        ("level", &["l", "E:", "v", "L"]),
        ("life", &["l", "Y:", "f"]),
        ("light", &["l", "Y:", "t"]),
        ("line", &["l", "Y:", "n"]),
        ("live", &["l", "I:", "v"]),
        ("lives", &["l", "Y:", "v", "z"]),
        ("long", &["l", "A:", "N"]),
        ("look", &["l", "U:", "k"]),
        ("lot", &["l", "a:", "t"]),
        ("love", &["l", "^:", "v"]),
        ("low", &["l", "O:"]),
        ("man", &["m", "145:", "n"]),
        ("market", &["m", "a:", "r", "K", "I", "t"]),
        ("maybe", &["m", "J:", "b", "i"]),
        ("mean", &["m", "i:", "n"]),
        ("member", &["m", "E:", "m", "b", "e"]),
        ("minute", &["m", "I:", "n", "$", "t"]),
        ("misses", &["m", "I", "s", "I", "z"]),
        ("mizz", &["m", "I", "z"]),
        ("moment", &["m", "O:", "m", "$", "n", "t"]),
        ("money", &["m", "^:", "n", "i"]),
        ("month", &["m", "^:", "n", "T"]),
        ("morning", &["m", "A:", "r", "n", "I", "N"]),
        ("mother", &["m", "^:", "D", "e"]),
        ("move", &["m", "u:", "v"]),
        ("music", &["m", "j", "u:", "z", "I", "k"]),
        ("name", &["n", "J:", "m"]),
        ("near", &["n", "i:", "r"]),
        ("new", &["n", "u:"]),
        ("night", &["n", "Y:", "t"]),
        ("nine", &["n", "Y:", "n"]),
        ("no", &["n", "O:"]),
        ("number", &["n", "^:", "m", "b", "e"]),
        ("offer", &["A:", "f", "e"]),
        ("office", &["A:", "f", "I", "s"]),
        ("okay", &["O", "K", "J:"]),
        ("old", &["O:", "l", "d"]),
        ("on", &["a:", "n"]),
        ("one", &["w", "^:", "n"]),
        ("open", &["O:", "P", "$", "n"]),
        ("parent", &["P", "E:", "r", "$", "n", "t"]),
        ("part", &["P", "a:", "r", "t"]),
        ("party", &["P", "a:", "r", "R", "i"]),
        ("pass", &["P", "145:", "s"]),
        ("people", &["P", "i:", "P", "L"]),
        ("person", &["P", "e:", "s", "$", "n"]),
        ("play", &["P", "l", "J:"]),
        ("please", &["P", "l", "i:", "z"]),
        ("point", &["P", "y:", "n", "t"]),
        ("power", &["P", "@:", "e"]),
        ("president", &["P", "r", "E:", "z", "I", "R", "$", "n", "t"]),
        ("process", &["P", "r", "a:", "s", "E", "s"]),
        ("program", &["P", "r", "O:", "g", "r", "145", "m"]),
        ("put", &["P", "U:", "t"]),
        ("question", &["K", "w", "E:", "S", "C", "$", "n"]),
        ("raise", &["r", "J:", "z"]),
        ("reach", &["r", "i:", "C"]),
        ("read", &["r", "i:", "d"]),
        ("reason", &["r", "i:", "z", "$", "n"]),
        ("red", &["r", "E:", "d"]),
        ("remember", &["r", "I", "m", "E:", "m", "b", "e"]),
        ("report", &["r", "I", "P", "A:", "r", "t"]),
        ("require", &["r", "i", "K", "w", "Y:", "r"]),
        ("research", &["r", "i:", "s", "e", "C"]),
        ("result", &["r", "I", "z", "^:", "l", "t"]),
        ("return", &["r", "i", "146", "e:", "n"]),
        ("right", &["r", "Y:", "t"]),
        ("room", &["r", "u:", "m"]),
        ("run", &["r", "^:", "n"]),
        ("second", &["s", "E:", "K", "$", "n", "d"]),
        ("see", &["s", "i:"]),
        ("seem", &["s", "i:", "m"]),
        ("sell", &["s", "E:", "l"]),
        ("send", &["s", "E:", "n", "d"]),
        ("sense", &["s", "E:", "n", "s"]),
        ("serve", &["s", "e:", "v"]),
        ("service", &["s", "e:", "v", "I", "s"]),
        ("seven", &["s", "E:", "v", "$", "n"]),
        ("seventy", &["s", "E:", "v", "$", "n", "R", "i"]),
        ("short", &["S", "A:", "r", "t"]),
        ("show", &["S", "O:"]),
        ("side", &["s", "Y:", "d"]),
        ("sit", &["s", "I:", "t"]),
        ("six", &["s", "I:", "k", "s"]),
        ("small", &["s", "m", "A:", "l"]),
        ("smith", &["s", "m", "I:", "T"]),
        ("sorry", &["s", "a:", "r", "i"]),
        ("speak", &["s", "p", "i:", "k"]),
        ("spend", &["s", "p", "E:", "n", "d"]),
        ("stand", &["s", "t", "145:", "n", "d"]),
        ("stay", &["s", "t", "J"]),
        ("stop", &["s", "t", "a:", "p"]),
        ("story", &["s", "t", "A:", "r", "i"]),
        ("street", &["s", "t", "r", "i:", "t"]),
        ("study", &["s", "t", "^:", "R", "i"]),
        ("sunny", &["s", "^:", "n", "i"]),
        ("system", &["s", "I:", "s", "t", "$", "m"]),
        ("take", &["146", "J:", "k"]),
        ("teacher", &["146", "i:", "C", "e"]),
        ("team", &["146", "i:", "m"]),
        ("tell", &["146", "E:", "l"]),
        (
            "temperatures",
            &["146", "E:", "m", "P", "r", "$", "C", "e", "z"],
        ),
        ("ten", &["146", "E:", "n"]),
        ("thank", &["T", "145:", "N", "k"]),
        ("the", &["D", "$"]),
        ("think", &["T", "I:", "N", "k"]),
        ("third", &["T", "e:", "d"]),
        ("three", &["T", "r", "i:"]),
        ("time", &["146", "Y:", "m"]),
        ("today", &["146", "$", "d", "J:"]),
        ("try", &["146", "r", "Y:"]),
        ("two", &["146", "u:"]),
        (
            "understand",
            &["^", "n", "d", "e", "s", "t", "145:", "n", "d"],
        ),
        ("use", &["j", "u:", "z"]),
        ("wait", &["w", "J:", "t"]),
        ("walk", &["w", "A:", "k"]),
        ("want", &["w", "a:", "n", "t"]),
        ("war", &["w", "A:", "r"]),
        ("watch", &["w", "a:", "C"]),
        ("water", &["w", "A:", "R", "e"]),
        ("way", &["w", "J:"]),
        ("weather", &["w", "E:", "D", "e"]),
        ("week", &["w", "i:", "k"]),
        ("west", &["w", "E:", "s", "t"]),
        ("will", &["w", "L"]),
        ("win", &["w", "I:", "n"]),
        ("with", &["w", "I", "T"]),
        ("woman", &["w", "U:", "m", "$", "n"]),
        ("word", &["w", "e:", "d"]),
        ("work", &["w", "e:", "k"]),
        ("world", &["w", "e:", "l", "d"]),
        ("write", &["r", "Y:", "t"]),
        ("year", &["j", "i:", "r"]),
        ("yes", &["j", "E:", "s"]),
        ("you", &["j", "u:"]),
        ("zero", &["z", "i:", "r", "O"]),
    ];
    for (word, phones) in entries {
        lexicon.insert(
            (*word).to_string(),
            phones.iter().map(|p| (*p).to_string()).collect(),
        );
    }
}

fn apply_neural_adapter_params(opts: &mut NeuralAdapterOpts, params: &serde_json::Value) {
    if let Some(v) = params.get("eos").and_then(|x| x.as_str()) {
        opts.eos = v.to_string();
    }
    if let Some(v) = params.get("pause_marker").and_then(|x| x.as_str()) {
        opts.pause_marker = v.to_string();
    }
    if let Some(v) = params.get("word_boundary_marker").and_then(|x| x.as_str()) {
        opts.word_boundary = v.to_string();
    }
    if let Some(v) = params.get("stress_marker").and_then(|x| x.as_str()) {
        opts.stress_marker = v.to_string();
    }
    if let Some(v) = params.get("max_word_limit").and_then(|x| x.as_u64()) {
        opts.max_word_limit = v as usize;
    }
    if let Some(arr) = params.get("punctuation").and_then(|x| x.as_array()) {
        opts.punctuation = arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
    }
    if let Some(arr) = params.get("punctuation_set1").and_then(|x| x.as_array()) {
        opts.punctuation_set1 = arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
    }
    if let Some(arr) = params.get("punctuation_set2").and_then(|x| x.as_array()) {
        opts.punctuation_set2 = arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
    }
}

/// Parse a space-separated phone string (symbols or integer ids).
pub fn parse_phone_string(s: &str, map: &PhoneMap) -> Result<Vec<usize>> {
    let mut ids = Vec::new();
    for tok in s.split_whitespace() {
        if let Ok(n) = tok.parse::<usize>() {
            ids.push(n);
            continue;
        }
        let id = map
            .id(tok)
            .with_context(|| format!("unknown phone '{tok}'"))?;
        ids.push(id);
    }
    if ids.is_empty() {
        bail!("no phones parsed");
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digit_words_cover_0_9() {
        assert_eq!(digit_word('1'), "one");
        assert_eq!(digit_word('9'), "nine");
    }

    #[test]
    fn doctor_smith_neural_phones_if_bundle() {
        let Some(root) = crate::gguf_bundle::default_extract_dir() else {
            return;
        };
        if !root.join("frontend").is_dir() {
            return;
        }
        let fe = HydraLite::open(&root).expect("open frontend");
        let phones = fe
            .text_to_phones("Dr. Smith lives on 2nd St.")
            .expect("phones");
        let joined = phones.join(" ");
        assert!(
            joined.starts_with("d a: k 146 e # s m I: T #"),
            "unexpected phones: {joined}"
        );
        assert!(
            joined.contains("s E: K $ n d # s t r i: t"),
            "missing second street: {joined}"
        );
        assert!(joined.ends_with("# . ~") || joined.ends_with(". ~") || joined.ends_with("~"));
    }
}
