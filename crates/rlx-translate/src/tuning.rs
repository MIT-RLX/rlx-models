//! Every knob that changes *what the search does*, in one place.
//!
//! These settings arrived one at a time and ended up in three different
//! shapes: four setters on [`crate::decode::Nmt`], three constants written
//! into `translate_nbest`, and a handful of `env::var` reads inside the bench
//! subcommand. That made the interesting ones — beam width, the length budget,
//! how many finished hypotheses to wait for — reachable only by editing the
//! source, which is exactly backwards: those are the ones worth sweeping.
//!
//! So: one struct, one env layer, one place the CLI parses. A field is `Option`
//! when [`crate::pdec::PDecParams`] already carries the OS's own value and
//! `None` means "use it"; a plain value is one this port chose, and the doc
//! comment says why.
//!
//! ```no_run
//! # use rlx_translate::tuning::Tuning;
//! let mut t = Tuning::from_env()?;   // RLX_TRANSLATE_BEAM=16, ...
//! t.beam = Some(16);                 // or set it directly
//! # Ok::<_, anyhow::Error>(())
//! ```

use anyhow::{Context, Result};

use crate::pdec::PDecParams;

/// Search settings, layered over what the shipped config asks for.
#[derive(Debug, Clone, PartialEq)]
pub struct Tuning {
    /// Refuse to emit a target n-gram this hypothesis already contains.
    ///
    /// **3.** the config names no such constraint, so this was 0 at first; measured
    /// on non-lexicon sentences it moves output *closer* to the OS (chrF
    /// 0.693 -> 0.720), because the shipped decoder audibly loops. 0 for strict
    /// fidelity.
    pub no_repeat_ngram: usize,
    /// Refuse to repeat a run of this many characters. 0 (the framework's behaviour).
    ///
    /// Catches stutters that span token boundaries — `ornithornithaque` repeats
    /// letters without repeating any token n-gram.
    pub no_repeat_char_ngram: usize,
    /// Exponent in `score / len^alpha`. `None` keeps the config's `norm-costs`
    /// boolean, whose endpoints are 1.0 and 0.0.
    pub length_penalty: Option<f64>,
    /// Overrides `norm-costs` itself. Ignored when `length_penalty` is set.
    pub norm_costs: Option<bool>,
    /// Score the whole vocabulary rather than the shortlist. Diagnostic, and
    /// much slower: it answers whether a bad output was a *search* failure or a
    /// *reachability* one.
    pub ignore_shortlist: bool,
    /// Put the second target tag *after* the source text rather than before it.
    ///
    /// A direction's `target-token` is often two tags joined by `> <`: the
    /// language, `<tar-tr_TR>`, and a corpus or domain tag, `<en_US-tr_TR-
    /// optimal>`. This port put both in front of the sentence, which is right
    /// for the French-family bundle and for `en-hi`, and **wrong for `en-tr`**:
    /// with the domain tag leading, `en_US-tr_TR` emits word-for-word output in
    /// English word order and scores chrF 0.526, the worst of 43 directions.
    ///
    /// Greedy, one sentence, all three bundles:
    ///
    /// | prefix | tr | fr | hi |
    /// | --- | --- | --- | --- |
    /// | `src tar opt` + text | word salad | correct | correct |
    /// | `src tar` + text (no opt) | **exact** | correct | garbage |
    /// | `src tar` + text + `opt` | good | correct | correct |
    ///
    /// So the tag cannot simply be dropped — `hi` needs it — and trailing it
    /// looked like the answer: with `domain_tag_last`, beam search renders that
    /// Turkish sentence exactly as the OS does.
    ///
    /// **Measured over 120 sentences in 8 directions, it is not.** Overall chrF
    /// 0.757 -> 0.753 and 39 exact matches -> 34, and `en_US-tr_TR` itself gets
    /// *worse*, 0.526 -> 0.455. The effect is real but direction-dependent:
    ///
    /// | direction | tag first | tag last |
    /// | --- | --- | --- |
    /// | `en_US-fr_FR` | **0.966** | 0.896 |
    /// | `en_US-es_ES` | **0.969** | 0.953 |
    /// | `en_US-tr_TR` | **0.526** | 0.455 |
    /// | `en_US-zh_TW` | 0.533 | **0.607** |
    /// | `en_US-ja_JP` | 0.862 | **0.884** |
    ///
    /// So it stays off. Kept because the single-sentence evidence for it was as
    /// convincing as evidence gets and still did not survive the corpus, which
    /// is worth being able to re-run rather than re-derive. Choosing it
    /// per-direction would be fitting the reference set, not reading the
    /// format.
    pub domain_tag_last: bool,
    /// Reuse the decoder state a prefix leaves behind instead of replaying it.
    ///
    /// True, and worth ~2x. Off restores the original quadratic replay, which
    /// is the only way to prove the cache is not silently serving a stale
    /// state — a defect that would change translations rather than crash.
    /// `tests/real_decode.rs` compares the two.
    pub incremental: bool,

    /// Beam width, overriding the config's.
    pub beam: Option<usize>,
    /// Relative-score pruning factor (`rs-beam`), overriding the config's.
    pub rs_beam: Option<f64>,
    /// Apply `rs-beam` as a pruning cutoff. **On.**
    ///
    /// The config sets `rs-beam: 0.66` on every direction and this port did not
    /// use it: `PruningPolicy` was written with the note that guessing the OS's
    /// semantics "would corrupt output in a way that is hard to attribute", and
    /// to wire it up once there was a reference decode to diff against. There
    /// is one now, and the reading `cutoff = best - factor * |best|` measures as
    /// a wash on quality and a clear win on time:
    ///
    /// | | exact / 645 | chrF | time |
    /// | --- | --- | --- | --- |
    /// | off | 507 | 0.959 | 376 s |
    /// | on | 506 | **0.960** | **212 s** |
    ///
    /// One exact match in 645 either way, so this is not evidence that the
    /// semantics are *right* — only that they are not corrupting. It is on
    /// because the config asks for it and it is faster; `RS_BEAM_PRUNE=0` is
    /// the way back.
    pub rs_beam_prune: bool,
    /// Absolute output length cap.
    pub max_len: Option<usize>,
    /// Source tokens to keep. **`None` — no limit — is the default.**
    ///
    /// This port capped the source at 64, the length the embedding graph was
    /// traced at, **silently and after appending the terminator**, so a longer
    /// source lost both its tail and its `<s>`. FLORES sentences are Wikipedia
    /// prose and routinely exceed 64 pieces; a 71-piece one came out as
    ///
    /// > Cependant, cependant, ces plans ont été rendus rendus, quand plus de
    /// > plus de plans, plus d'en plus de l'armée rouge est entrée et créée...
    ///
    /// and uncapped, as
    ///
    /// > Cependant, ces plans sont devenus obsolètes presque du jour au
    /// > lendemain, lorsque plus de 800 000 soldats de l'Armée rouge de
    /// > l'Union soviétique sont entrés et ont créé les fronts biélorusse et
    /// > ukrainien...
    ///
    /// The graphs are length-polymorphic — 501 source pieces run without
    /// complaint — so 64 was never a model limit, only a traced shape. Set this
    /// to restore a cap.
    pub max_source_tokens: Option<usize>,
    /// Lower bound on the length budget regardless of source length.
    pub max_len_floor: Option<usize>,
    /// Length budget as a multiple of the source token count.
    pub max_len_relative: Option<f64>,
    /// How many finished hypotheses to collect before stopping.
    ///
    /// This is the single most expensive setting and the least obvious. The
    /// search treats `nbest` as both "results to return" and "completions to
    /// wait for", so asking for extra results to deduplicate also made it run
    /// to the full 80-step budget — 10.9 s of a 11.3 s translation. Left
    /// `None`, [`Tuning::apply`] derives it from the request via
    /// `nbest_multiple`.
    pub stop_after: Option<usize>,

    /// Completions to request per result asked for, before deduplication. 2.
    ///
    /// Several token paths decode to the same string, so a list that repeats
    /// one answer tells the caller nothing. Higher costs time; see
    /// `stop_after`.
    pub nbest_multiple: usize,
    /// Beam width per result asked for, when more than one is asked for. 4.
    ///
    /// Width buys *diversity* without the stopping cost, which is why the
    /// over-generation lives here rather than in `nbest_multiple`.
    ///
    /// A **1-best request ignores this and uses the config's own beam**, which
    /// is 3 on every shipped direction. Widening it to 8 was measured over 120
    /// sentences in 8 directions and changed nothing at all — 72 exact matches
    /// and chrF 0.898 either way — while costing 2.2x the time (198 s against
    /// 88 s). the OS searches at 3; there is nothing above it to find.
    pub beam_multiple: usize,
    /// Minimum beam width when several variants are asked for. 8.
    pub beam_floor: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            no_repeat_ngram: 3,
            no_repeat_char_ngram: 0,
            length_penalty: None,
            norm_costs: None,
            ignore_shortlist: false,
            incremental: true,
            domain_tag_last: false,
            beam: None,
            rs_beam: None,
            rs_beam_prune: true,
            max_len: None,
            max_source_tokens: None,
            max_len_floor: None,
            max_len_relative: None,
            stop_after: None,
            nbest_multiple: 2,
            beam_multiple: 4,
            beam_floor: 8,
        }
    }
}

/// Environment variable prefix for every setting.
pub const ENV_PREFIX: &str = "RLX_TRANSLATE_";

/// `(suffix, help)` for each setting, in the order [`Tuning::describe`] prints.
pub const KEYS: &[(&str, &str)] = &[
    ("NO_REPEAT", "target n-gram not to repeat (0 off)"),
    ("NO_REPEAT_CHARS", "character run not to repeat (0 off)"),
    ("LENGTH_PENALTY", "exponent in score/len^alpha"),
    ("NORM_COSTS", "length-normalise costs (0/1)"),
    ("IGNORE_SHORTLIST", "score the whole vocabulary (0/1)"),
    ("INCREMENTAL", "reuse decoder state across steps (0/1)"),
    (
        "DOMAIN_TAG_LAST",
        "trail the domain tag after the text (0/1)",
    ),
    ("BEAM", "beam width"),
    ("RS_BEAM", "relative-score pruning factor"),
    ("RS_BEAM_PRUNE", "apply rs-beam as a cutoff (0/1)"),
    ("MAX_LEN", "absolute output length cap"),
    ("MAX_SOURCE_TOKENS", "source tokens kept (graph default 64)"),
    ("MAX_LEN_FLOOR", "length budget floor"),
    ("MAX_LEN_RELATIVE", "length budget / source tokens"),
    ("STOP_AFTER", "completions to collect before stopping"),
    ("NBEST_MULTIPLE", "completions requested per result"),
    ("BEAM_MULTIPLE", "beam width per result"),
    ("BEAM_FLOOR", "minimum beam width"),
];

/// Parses `name`, reporting the variable that was wrong rather than ignoring
/// it — a typo in a sweep script should not read as "this lever does nothing".
fn var<T: std::str::FromStr>(name: &str) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    let full = format!("{ENV_PREFIX}{name}");
    match std::env::var(&full) {
        Err(_) => Ok(None),
        Ok(v) if v.trim().is_empty() => Ok(None),
        Ok(v) => v
            .trim()
            .parse::<T>()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("{full}={v:?}: {e}")),
    }
}

fn flag(name: &str) -> Result<Option<bool>> {
    Ok(var::<String>(name)?.map(|v| !matches!(v.as_str(), "0" | "false" | "no" | "off")))
}

impl Tuning {
    /// Defaults with every `RLX_TRANSLATE_*` variable applied over the top.
    pub fn from_env() -> Result<Self> {
        let mut t = Tuning::default();
        t.override_from_env()
            .context("reading RLX_TRANSLATE_* settings")?;
        Ok(t)
    }

    /// Applies the environment to an existing set of settings.
    pub fn override_from_env(&mut self) -> Result<()> {
        if let Some(v) = var("NO_REPEAT")? {
            self.no_repeat_ngram = v;
        }
        if let Some(v) = var("NO_REPEAT_CHARS")? {
            self.no_repeat_char_ngram = v;
        }
        if let Some(v) = var("LENGTH_PENALTY")? {
            self.length_penalty = Some(v);
        }
        if let Some(v) = flag("NORM_COSTS")? {
            self.norm_costs = Some(v);
        }
        // `NO_SHORTLIST` was this lever's first name and is in sweep scripts.
        if let Some(v) = flag("NO_SHORTLIST")?.or(flag("IGNORE_SHORTLIST")?) {
            self.ignore_shortlist = v;
        }
        if let Some(v) = flag("INCREMENTAL")? {
            self.incremental = v;
        }
        if let Some(v) = flag("DOMAIN_TAG_LAST")? {
            self.domain_tag_last = v;
        }
        if let Some(v) = var("BEAM")? {
            self.beam = Some(v);
        }
        if let Some(v) = var("RS_BEAM")? {
            self.rs_beam = Some(v);
        }
        if let Some(v) = flag("RS_BEAM_PRUNE")? {
            self.rs_beam_prune = v;
        }
        if let Some(v) = var("MAX_LEN")? {
            self.max_len = Some(v);
        }
        if let Some(v) = var("MAX_SOURCE_TOKENS")? {
            self.max_source_tokens = Some(v);
        }
        if let Some(v) = var("MAX_LEN_FLOOR")? {
            self.max_len_floor = Some(v);
        }
        if let Some(v) = var("MAX_LEN_RELATIVE")? {
            self.max_len_relative = Some(v);
        }
        if let Some(v) = var("STOP_AFTER")? {
            self.stop_after = Some(v);
        }
        if let Some(v) = var("NBEST_MULTIPLE")? {
            self.nbest_multiple = v;
        }
        if let Some(v) = var("BEAM_MULTIPLE")? {
            self.beam_multiple = v;
        }
        if let Some(v) = var("BEAM_FLOOR")? {
            self.beam_floor = v;
        }
        Ok(())
    }

    /// Applies one `key=value` pair, accepting either the bare suffix
    /// (`beam=16`) or the full variable name.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let k = key
            .trim()
            .trim_start_matches("--")
            .replace('-', "_")
            .to_uppercase();
        let k = k.strip_prefix(ENV_PREFIX).unwrap_or(&k).to_string();
        anyhow::ensure!(
            KEYS.iter().any(|(n, _)| *n == k),
            "unknown setting {key:?}; known: {}",
            KEYS.iter()
                .map(|(n, _)| n.to_lowercase())
                .collect::<Vec<_>>()
                .join(" ")
        );
        // Reuse the env parsers so a CLI flag and a variable cannot disagree
        // about what a value means.
        let full = format!("{ENV_PREFIX}{k}");
        let prior = std::env::var(&full).ok();
        // SAFETY-adjacent: single-threaded CLI argument parsing, restored below.
        unsafe { std::env::set_var(&full, value) };
        let mut fresh = self.clone();
        let r = fresh.override_from_env();
        match prior {
            Some(p) => unsafe { std::env::set_var(&full, p) },
            None => unsafe { std::env::remove_var(&full) },
        }
        r?;
        *self = fresh;
        Ok(())
    }

    /// Writes these settings into `params` for a request of `want` results.
    ///
    /// Returns the effective `(beam, stop_after)` so a caller can report what
    /// the search actually ran with.
    pub fn apply(&self, params: &mut PDecParams, want: usize) -> (usize, usize) {
        let want = want.max(1);
        params.nbest = self
            .stop_after
            .unwrap_or_else(|| want.saturating_mul(self.nbest_multiple.max(1)).max(2));
        params.beam = self.beam.unwrap_or(if want == 1 {
            // The config's own width. Wider finds nothing the OS's search misses.
            params.beam
        } else {
            params
                .beam
                .max(want.saturating_mul(self.beam_multiple))
                .max(self.beam_floor)
        });
        if let Some(v) = self.rs_beam {
            params.rs_beam = v;
        }
        if let Some(v) = self.norm_costs {
            params.norm_costs = v;
        }
        if let Some(v) = self.max_len {
            params.max_seq_length = v;
        }
        if let Some(v) = self.max_len_floor {
            params.max_seq_length_floor = v;
        }
        if let Some(v) = self.max_len_relative {
            params.max_seq_length_relative = v;
        }
        (params.beam, params.nbest)
    }

    /// One `key = value` line per setting, for `rlx-translate tune`.
    pub fn describe(&self) -> Vec<(String, String, &'static str)> {
        // `None` means two different things depending on the field: fall back
        // to the OS's block, or derive it from the request. Say which.
        fn opt<T: std::fmt::Display>(v: &Option<T>) -> String {
            v.as_ref()
                .map_or_else(|| "config".to_string(), T::to_string)
        }
        fn auto<T: std::fmt::Display>(v: &Option<T>) -> String {
            v.as_ref().map_or_else(|| "auto".to_string(), T::to_string)
        }
        let vals = [
            self.no_repeat_ngram.to_string(),
            self.no_repeat_char_ngram.to_string(),
            opt(&self.length_penalty),
            opt(&self.norm_costs),
            self.ignore_shortlist.to_string(),
            self.incremental.to_string(),
            self.domain_tag_last.to_string(),
            auto(&self.beam),
            opt(&self.rs_beam),
            self.rs_beam_prune.to_string(),
            opt(&self.max_len),
            opt(&self.max_source_tokens),
            opt(&self.max_len_floor),
            opt(&self.max_len_relative),
            auto(&self.stop_after),
            self.nbest_multiple.to_string(),
            self.beam_multiple.to_string(),
            self.beam_floor.to_string(),
        ];
        KEYS.iter()
            .zip(vals)
            .map(|((k, h), v)| (k.to_lowercase(), v, *h))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_one_best_request_searches_exactly_as_the_config_asks() {
        let mut p = PDecParams {
            beam: 3,
            nbest: 1,
            ..PDecParams::default()
        };
        let (beam, stop) = Tuning::default().apply(&mut p, 1);
        // Neither widened: waiting for eight completions was the 10.9 s bug,
        // and a beam above the config's 3 was measured to find nothing.
        assert_eq!(stop, 2);
        assert_eq!(beam, 3);
    }

    #[test]
    fn asking_for_more_variants_widens_both() {
        let mut p = PDecParams::default();
        let (beam, stop) = Tuning::default().apply(&mut p, 3);
        assert_eq!(stop, 6);
        // Distinct variants need somewhere to come from, so width returns here.
        assert_eq!(beam, 12);
    }

    #[test]
    fn explicit_settings_beat_the_derived_ones() {
        let t = Tuning {
            beam: Some(5),
            stop_after: Some(1),
            max_len_floor: Some(24),
            ..Tuning::default()
        };
        let mut p = PDecParams {
            max_seq_length: 200,
            max_seq_length_floor: 80,
            max_seq_length_relative: 2.0,
            ..PDecParams::default()
        };
        let (beam, stop) = t.apply(&mut p, 4);
        assert_eq!((beam, stop), (5, 1));
        assert_eq!(p.length_budget(3), 24);
    }

    #[test]
    fn key_value_accepts_the_forms_a_person_types() {
        let mut t = Tuning::default();
        t.set("beam", "16").unwrap();
        t.set("--no-repeat", "0").unwrap();
        t.set("RLX_TRANSLATE_LENGTH_PENALTY", "0.6").unwrap();
        t.set("norm_costs", "0").unwrap();
        assert_eq!(t.beam, Some(16));
        assert_eq!(t.no_repeat_ngram, 0);
        assert_eq!(t.length_penalty, Some(0.6));
        assert_eq!(t.norm_costs, Some(false));
    }

    #[test]
    fn a_bad_value_is_an_error_not_a_shrug() {
        let mut t = Tuning::default();
        let e = t.set("beam", "wide").unwrap_err().to_string();
        assert!(e.contains("BEAM"), "{e}");
        assert!(t.set("bean", "8").is_err());
        assert_eq!(t, Tuning::default());
    }

    #[test]
    fn describe_covers_every_key() {
        assert_eq!(Tuning::default().describe().len(), KEYS.len());
    }
}
