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

//! `rlx-translate` — inspect the on-device translation assets macOS installed.

use anyhow::{Context, Result, bail};
use rlx_translate::assets::{Assets, load_config};
use rlx_translate::casemap::sentence_case;
use rlx_translate::espresso;
use rlx_translate::normalizer::Normalizer;
use rlx_translate::phrasebook::Phrasebook;
use rlx_translate::pipeline::{Support, TranslationPlan};
use rlx_translate::postproc::normalize_punctuation_for;
use rlx_translate::quasar::BlockKind;
use rlx_translate::quasar::{LangPair, TASK_MT_APP};
use rlx_translate::tuning::Tuning;
use std::path::Path;

const USAGE: &str = "\
rlx-translate — on-device system translation, natively on RLX

USAGE:
    rlx-translate status                 what is installed on this machine
    rlx-translate pairs                  language pairs with a config present
    rlx-translate directions             every ordered <src>-<tgt> that has an
                                         mt_app graph (both directions)
    rlx-translate plan <src-tgt> [task]  resolved pipeline for a pair
    rlx-translate config <src-tgt>       dump a pair's blocks
    rlx-translate normalize <file> [text]  run a normalizer sed script
    rlx-translate translate <src-tgt> <text>   translate (phrasebook stage only)
    rlx-translate bench [src-tgt]        score every installed pair on the OS's
                                         own test vectors, with timings
    rlx-translate parity <ref-dir>       score the native pipeline against a
                                         reference dump of the OS's live output
    rlx-translate export <out-dir> [lang]  export the NMT to safetensors + JSON
                                         (f32; `lang` picks the per-language
                                         decoder/handover/input graphs)
    rlx-translate convert <out-dir> [dtype]  convert EVERY installed bundle and
                                         every config to safetensors + GGUF +
                                         .rlxp (dtype: f16 default, f32, q8_0)
    rlx-translate nbest <src-tgt> <text> [n]  the n best translations, with
                                         scores (default 3)
    rlx-translate probe <file>           dump a model manifest
    rlx-translate tune                   every search setting and its value

Any subcommand accepts search settings as trailing key=value arguments, or the
same names as RLX_TRANSLATE_* variables:

    rlx-translate nbest en_US-fr_FR \"the woman of my dreams\" 3 beam=16
    RLX_TRANSLATE_NO_REPEAT=0 rlx-translate bench en_US-fr_FR

Run `rlx-translate tune` for the list. Set RLX_TRANSLATE_ASSETS to a
colon-separated list of extra asset roots.
";

/// Splits `key=value` search settings out of the positional arguments.
///
/// Only *recognised* keys are consumed, so text containing an `=` still
/// translates. Each is validated through [`Tuning::set`] — a typo is an error
/// here rather than a lever that silently does nothing — and then exported,
/// because `Nmt::load` reads the environment and every subcommand loads its own
/// model.
fn split_settings(raw: &[String]) -> Result<(Vec<String>, Tuning)> {
    let mut tuning = Tuning::from_env()?;
    let mut positional = Vec::with_capacity(raw.len());
    for a in raw {
        let Some((k, v)) = a.split_once('=') else {
            positional.push(a.clone());
            continue;
        };
        let norm = k.trim_start_matches("--").replace('-', "_").to_uppercase();
        if !rlx_translate::tuning::KEYS.iter().any(|(n, _)| *n == norm) {
            // Something shaped like a setting but not one is a typo, and
            // ignoring it would mean a sweep silently measures the default.
            // Text to translate is what else can hold an `=`, and that has
            // spaces; a bare `word=word` is treated as the typo it probably is.
            anyhow::ensure!(
                a.contains(char::is_whitespace)
                    || !k
                        .trim_start_matches("--")
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "unknown setting {k:?}; run `rlx-translate tune` for the list"
            );
            positional.push(a.clone());
            continue;
        }
        tuning.set(k, v)?;
        unsafe { std::env::set_var(format!("{}{norm}", rlx_translate::tuning::ENV_PREFIX), v) };
    }
    Ok((positional, tuning))
}

fn main() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let (args, tuning) = split_settings(&raw)?;
    let Some(cmd) = args.first().map(String::as_str) else {
        print!("{USAGE}");
        return Ok(());
    };
    match cmd {
        "status" => status(),
        "pairs" => pairs(),
        "directions" => directions(),
        "plan" => plan(args.get(1), args.get(2).map(String::as_str)),
        "config" => dump_config(args.get(1)),
        "normalize" => normalize(args.get(1), args.get(2)),
        "translate" => translate(args.get(1), args.get(2)),
        "bench" => bench(args.get(1)),
        "parity" => parity(args.get(1), args.get(2).map(String::as_str)),
        "export" => export(args.get(1), args.get(2).map(String::as_str)),
        "convert" => convert(args.get(1), args.get(2).map(String::as_str)),
        "nbest" => nbest(
            args.get(1),
            args.get(2).map(String::as_str),
            args.get(3).and_then(|v| v.parse().ok()),
        ),
        "probe" => probe(args.get(1)),
        "tune" => {
            println!(
                "search settings (set as key=value, or {}KEY)\n",
                rlx_translate::tuning::ENV_PREFIX
            );
            for (k, v, help) in tuning.describe() {
                println!("  {k:<18} {v:<10} {help}");
            }
            println!(
                "\n  \"config\" takes the value from the OS's own block for the pair;\
                 \n  \"auto\" derives it from how many results are asked for."
            );
            // These do not change the answer, only how it is produced or
            // reported, so they are not part of `Tuning` — but someone asking
            // "what can I set?" wants them in the same list.
            println!("\n  execution and diagnostics (environment only)\n");
            let assets = std::env::var(format!("{}ASSETS", rlx_translate::tuning::ENV_PREFIX))
                .unwrap_or_else(|_| "system".to_string());
            for (k, v, help) in [
                (
                    "profile",
                    rlx_translate::profile::enabled().to_string(),
                    "per-op timing table after each run",
                ),
                (
                    "gemm_lanes",
                    rlx_translate::exec::gemm_lanes().to_string(),
                    "threads per int8 GEMM (1 measured fastest here)",
                ),
                ("assets", assets, "colon-separated extra asset roots"),
                (
                    "trace",
                    std::env::var(format!("{}TRACE", rlx_translate::tuning::ENV_PREFIX))
                        .is_ok()
                        .to_string(),
                    "print the stage graph's trace",
                ),
            ] {
                println!("  {k:<18} {v:<10} {help}");
            }
            Ok(())
        }
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => {
            bail!("unknown command {other:?}\n\n{USAGE}");
        }
    }
}

fn status() -> Result<()> {
    let assets = Assets::discover();
    println!("asset roots: {}", assets.roots.len());
    println!(
        "named assets: {}",
        assets
            .asset_dirs
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    );
    let pairs = assets.pairs();
    println!(
        "configs: {} files covering {} pairs",
        assets.configs.len(),
        pairs.len()
    );
    if pairs.is_empty() {
        println!("\nNo translation configs found.");
        return Ok(());
    }

    // Which pairs can actually be run end to end on this machine?
    let mut installed = Vec::new();
    let mut version = String::new();
    for pair in &pairs {
        let Ok((cf, config)) = assets.best_config(pair) else {
            continue;
        };
        if version.is_empty() {
            version = config.model_info.version.clone();
        }
        if let Ok(a) = assets.availability(&config, TASK_MT_APP, pair)
            && a.is_complete()
            && !a.present.is_empty()
        {
            installed.push((pair.clone(), cf.variant.clone(), a.present.len()));
        }
    }
    println!("model version: {version}");
    println!("\npairs with every file installed: {}", installed.len());
    for (pair, variant, files) in installed.iter().take(30) {
        println!("  {pair}  variant={variant}  files={files}");
    }
    if installed.len() > 30 {
        println!("  \u{2026} and {} more", installed.len() - 30);
    }
    if installed.is_empty() {
        println!(
            "\n  Install a language with tools/install-language.m, or via\n  \
             System Settings \u{2192} General \u{2192} Language & Region \u{2192} Translation Languages."
        );
    }
    Ok(())
}

fn pairs() -> Result<()> {
    let assets = Assets::discover();
    for p in assets.pairs() {
        let variants: Vec<_> = assets
            .configs_for(&p)
            .iter()
            .map(|c| c.variant.clone())
            .collect();
        println!("{p}  variants={variants:?}");
    }
    Ok(())
}

fn parse_pair(arg: Option<&String>) -> Result<LangPair> {
    let Some(s) = arg else {
        bail!("expected a language pair like en_US-fr_FR\n\n{USAGE}");
    };
    LangPair::parse(s)
}

fn plan(pair_arg: Option<&String>, task: Option<&str>) -> Result<()> {
    let pair = parse_pair(pair_arg)?;
    let task = task.unwrap_or(TASK_MT_APP);
    let assets = Assets::discover();
    let config = load_config(&assets, &pair)?;
    let plan = TranslationPlan::build(&config, task, &pair)?.with_asset_status(&assets);

    println!(
        "{} · task {} · {} stages",
        plan.pair,
        plan.task,
        plan.stages.len()
    );
    if let Some(p) = &plan.pdec {
        println!(
            "\ndecode: beam={} rs-beam={} nbest={} norm-costs={} lm={:?}@{} budget(30 tok)={}",
            p.beam,
            p.rs_beam,
            p.nbest,
            p.norm_costs,
            p.lm_mode,
            p.lm_weight,
            p.length_budget(30)
        );
        println!("source tokens: {:?}", p.source_tokens());
        println!("target tokens: {:?}", p.target_tokens());
        if p.shortlist.enabled {
            println!(
                "shortlist: cond-n={} freq-n={} table={} suppress={:?}",
                p.shortlist.cond_n,
                p.shortlist.freq_n,
                p.shortlist.lang_pair,
                p.shortlist.suppress_tokens
            );
        }
    }

    println!("\nstages (dependency order):");
    for s in &plan.stages {
        let mark = match s.support {
            Support::Ready => "ok  ",
            Support::HasAsset => "have",
            Support::NeedsAsset => "want",
            Support::Pending => "todo",
        };
        println!(
            "  [{mark}] {:<20} {:<26} <- {:?}",
            s.name,
            s.kind.as_str(),
            s.inputs
        );
        for f in &s.files {
            println!("           {f}");
        }
    }
    let missing = plan.missing_assets();
    println!(
        "\nstages with all files installed: {} · missing files: {}",
        plan.with_assets().len(),
        missing.len()
    );
    println!("(no block executor is implemented yet; `have` means the data is present)");
    Ok(())
}

fn dump_config(pair_arg: Option<&String>) -> Result<()> {
    let pair = parse_pair(pair_arg)?;
    let assets = Assets::discover();
    let (cfg_file, config) = assets.best_config(&pair)?;
    println!(
        "{}  (variant {})",
        cfg_file.path.display(),
        cfg_file.variant
    );
    println!(
        "version {}.{} · {} · tasks {:?}",
        config.version_major,
        config.version_minor,
        config.model_info.version,
        config.model_info.tasks
    );
    let decoder = config.mt_app()?;
    for b in decoder.blocks_for(&pair) {
        println!("{}", b.report());
    }
    Ok(())
}

fn normalize(file: Option<&String>, text: Option<&String>) -> Result<()> {
    let Some(file) = file else {
        bail!("expected a normalizer file\n\n{USAGE}");
    };
    let n = Normalizer::load(file)?;
    println!("{} substitution(s)", n.len());
    if let Some(t) = text {
        println!("{}", n.normalize(t));
    }
    Ok(())
}

fn probe(file: Option<&String>) -> Result<()> {
    let Some(file) = file else {
        bail!("expected a pyespresso.mdl.bin path\n\n{USAGE}");
    };
    let m = espresso::Manifest::load(file)?;
    println!("engine: {}", m.str("BEspressoEngine").unwrap_or("?"));
    println!("decoder layers: {}", m.decoder_layers());
    println!("target languages: {:?}", m.languages());

    println!("\nshared graphs:");
    for k in ["EncoderGraph", "EmbeddingGraph", "ReadoutGraph"] {
        if let Some(v) = m.str(k) {
            println!("  {k:<16} {v}");
        }
    }
    println!("\nper-language graphs:");
    for (k, langs) in &m.lang_graphs {
        println!("  {k}: {:?}", langs.keys().collect::<Vec<_>>());
    }

    println!("\ntensor wiring:");
    for k in [
        "SourceInputStr",
        "TargetInputStr",
        "EncoderValuesStr",
        "InputNetValuesStr",
        "ScoresStr",
        "AlignmentLayerStr",
        "ReadoutInputStr",
        "ReadoutOutputStr",
    ] {
        if let Some(v) = m.str(k) {
            println!("  {k:<20} {v}");
        }
    }
    for k in ["HandoverStrings", "StateStrings"] {
        let parts = m.csv(k);
        if !parts.is_empty() {
            println!("  {k}:");
            for p in parts {
                println!("      {p}");
            }
        }
    }

    println!("\nflags:");
    let mut flags: Vec<String> = m
        .values
        .iter()
        .filter_map(|(k, v)| v.as_bool().map(|b| format!("{k}={b}")))
        .collect();
    flags.sort();
    println!("  {}", flags.join(" "));
    let ints: Vec<String> = m
        .values
        .iter()
        .filter_map(|(k, v)| v.as_int().map(|i| format!("{k}={i}")))
        .collect();
    if !ints.is_empty() {
        println!("ints:\n  {}", ints.join(" "));
    }
    if let Some(off) = m.symbol_table_offset {
        println!("\nsymbol table starts at byte {off} (framing not decoded)");
    }
    Ok(())
}

/// Phrasebook files a pair's `PhraseBookBlock`s reference, resolved to paths.
/// The locale of the pair's `CaseMapBlock`, if it has one.
///
/// Note this is the *source* language in the shipped configs (`locale: "en"`
/// on the en->fr block), even though the casing is applied to the target.
/// Reproduced faithfully rather than "corrected".
fn casemap_locale(config: &rlx_translate::quasar::QuasarConfig, pair: &LangPair) -> Option<String> {
    let d = config.mt_app().ok()?;
    d.blocks_for(pair)
        .into_iter()
        .find(|b| b.kind == BlockKind::CaseMap)
        .and_then(|b| b.str("locale").map(str::to_string))
}

/// The `MT/` directory backing a pair, for the shipped `.pat` scripts and the
/// feature test vectors.
fn mt_dir(
    assets: &Assets,
    config: &rlx_translate::quasar::QuasarConfig,
    pair: &LangPair,
) -> Option<std::path::PathBuf> {
    let decoder = config.mt_app().ok()?;
    let block = decoder.translator(pair).ok()?;
    let model = block.str("model-file")?;
    assets.resolve(model)?.parent().map(Path::to_path_buf)
}

fn load_normalizers(mt: Option<&std::path::PathBuf>) -> (Option<Normalizer>, Option<Normalizer>) {
    let Some(mt) = mt else {
        return (None, None);
    };
    (
        Normalizer::load(mt.join("normalizer.pat")).ok(),
        Normalizer::load(mt.join("tokenizer.pat")).ok(),
    )
}

fn translate(pair_arg: Option<&String>, text: Option<&String>) -> Result<()> {
    let pair = parse_pair(pair_arg)?;
    let Some(text) = text else {
        bail!("expected text to translate\n\n{USAGE}");
    };
    let assets = Assets::discover();
    let (_, config) = assets.best_config(&pair)?;

    let mt = mt_dir(&assets, &config, &pair);
    let (norm, _tok) = load_normalizers(mt.as_ref());
    // The framework returns input with nothing to translate exactly as given —
    // `"   "` comes back as `"   "`. Normalising first would trim it away, so
    // the check has to come before the normalizer, not inside the graph.
    if !rlx_translate::execute::has_translatable_content(text) {
        println!("\ntranslation: {text}");
        return Ok(());
    }
    let normalized = norm
        .as_ref()
        .map_or_else(|| text.clone(), |n| n.normalize(text));

    // Run the config's own stage graph rather than a hand-wired order: the
    // fan-in and the phrasebook/NMT precedence are the OS's, read from the
    // shipped JSON.
    let plan = rlx_translate::pipeline::TranslationPlan::build(&config, "mt_app", &pair)?;

    let paths = assets.phrasebook_files(&config, &pair);
    let pb = if paths.is_empty() {
        None
    } else {
        Some(Phrasebook::load_all(&paths)?)
    };
    let qe = mt
        .as_ref()
        .and_then(|m| {
            rlx_translate::quality::QualityEstimator::load(
                &m.join("qualityEstimator"),
                pair.source.split('_').next().unwrap_or("en"),
                pair.target.split('_').next().unwrap_or("en"),
            )
            .ok()
        })
        .filter(|q| !q.is_empty());

    println!("source     : {text:?}");
    println!("normalized : {normalized:?}");
    println!(
        "graph      : {} stages{}",
        plan.stages.len(),
        pb.as_ref()
            .map(|p| format!(", phrasebook {} sources", p.len()))
            .unwrap_or_default()
    );

    // The SentencePiece stages need the vocabulary to do real work; without it
    // they pass through. Loading it here also means the NMT closure does not
    // have to be the only thing that owns it.
    let graph_vocab = assets
        .model_home(&pair)
        .and_then(|h| rlx_translate::spm::Vocab::load(h.join("spm.model")).ok());

    let lookup = |s: &str| pb.as_ref().and_then(|p| p.translate(s).map(str::to_string));
    // One model per translator stage, cached by its asset. A pivot pair walks
    // two of them — `ar_AE-de_DE` is ar->en on `MT-bi-en-ar` then en->de on the
    // seven-language bundle — so the model cannot be hoisted out of the graph.
    let models: std::cell::RefCell<std::collections::BTreeMap<String, LoadedNmt>> =
        std::cell::RefCell::new(std::collections::BTreeMap::new());
    let run_nmt = |p: &rlx_translate::pdec::PDecParams, s: &str| -> Result<String> {
        let key = format!("{}|{}", p.model_file, p.target_locale);
        if !models.borrow().contains_key(&key) {
            let m = load_nmt(&assets, &pair, p)?;
            models.borrow_mut().insert(key.clone(), m);
        }
        let b = models.borrow();
        let m = b.get(&key).expect("just inserted");
        let mut best = m.nmt.translate_nbest(&m.vocab, p, s, 1)?;
        if best.is_empty() {
            bail!("no hypothesis finished");
        }
        Ok(best.remove(0).text)
    };

    let ctx = rlx_translate::execute::Context {
        phrasebook: Some(&lookup),
        nmt: Some(&run_nmt),
        case_locale: casemap_locale(&config, &pair),
        target: pair.target.clone(),
        source: Some(pair.source.clone()),
        quality: qe.as_ref(),
        vocab: graph_vocab.as_ref(),
    };
    let out = rlx_translate::execute::run(&plan, &ctx, &normalized)?;

    match &out.text {
        Some(t) => {
            if let Some(from) = &out.source_stage {
                println!("produced by: {from}");
            }
            println!("\ntranslation: {t}");
        }
        None => println!("\nno stage produced a translation"),
    }
    if std::env::var("RLX_TRANSLATE_TRACE").is_ok() {
        for (name, v) in &out.trace {
            if let Some(t) = v {
                println!("  trace {name:<38} {t:?}");
            }
        }
    }
    for f in &out.flags {
        println!("quality    : {f:?}");
    }
    if !out.lost_spans.is_empty() {
        println!(
            "warning    : {:?} did not survive translation",
            out.lost_spans
        );
    }
    Ok(())
}

/// A model plus what it needs, loaded on demand per translator stage.
struct LoadedNmt {
    nmt: rlx_translate::decode::Nmt,
    vocab: rlx_translate::spm::Vocab,
}

/// Loads the bundle a translator block names.
fn load_nmt(
    assets: &Assets,
    pair: &LangPair,
    params: &rlx_translate::pdec::PDecParams,
) -> Result<LoadedNmt> {
    let home = assets
        .model_home_for(pair, &params.model_file)
        .ok_or_else(|| anyhow::anyhow!("{} is not installed for {pair}", params.model_file))?;
    let vocab = rlx_translate::spm::Vocab::load(home.join("spm.model"))?;
    let tgt: String = params
        .target_locale
        .chars()
        .take(2)
        .collect::<String>()
        .to_lowercase();
    let src_lang: String = pair
        .source
        .chars()
        .take(2)
        .collect::<String>()
        .to_lowercase();
    let nmt = rlx_translate::decode::Nmt::load_for_pair(
        &home,
        &[home.as_path()],
        &src_lang,
        &tgt,
        &params.shortlist.lang_pair,
    )?;
    Ok(LoadedNmt { nmt, vocab })
}

/// the OS's own test vectors for a pair, from `MT/featureTestDicts/<sl>-<tl>.dict`.
/// the OS's own live output for a direction, from a reference dump.
///
/// `featureTestDicts` is a *specification*, not a transcript: measured over the
/// 139 of its entries this machine can check, the OS's own shipped framework
/// scores 51.1% exact against it. So "how close are we to the gold" has a
/// ceiling well under 100% and is the wrong question for a port. "Do we
/// reproduce what the OS actually produces" is the right one, and this is the
/// oracle for it.
fn os_output(pair: &LangPair) -> Option<std::collections::HashMap<String, String>> {
    let dir = std::env::var("RLX_TRANSLATE_REFERENCE").ok()?;
    let text = std::fs::read_to_string(
        Path::new(&dir).join(format!("{}-{}.tsv", pair.source, pair.target)),
    )
    .ok()?;
    let mut m = std::collections::HashMap::new();
    for line in text.lines() {
        if let Some((s, t)) = line.split_once('\t') {
            m.entry(s.to_string()).or_insert_with(|| t.to_string());
        }
    }
    Some(m)
}

/// Gold pairs, grouped by source: one source may list **several acceptable
/// translations**.
///
/// `featureTestDicts` is a feature *test* set, and some of its entries probe
/// ambiguity deliberately — `the plant is open` lists both
/// `La fábrica está abierta` and `La planta está abierta`, and both
/// `工厂开着` and `植物开着`. Reading those as independent rows makes the same
/// source get translated twice and scores one of the two answers wrong no
/// matter which is produced, so a correct translator cannot exceed 50% on
/// them. A hit is a match against *any* listed reference.
fn feature_tests(mt: &Path, pair: &LangPair) -> Option<Vec<(String, Vec<String>)>> {
    let sl = pair.source.split('_').next()?;
    let tl = pair.target.split('_').next()?;
    let path = mt.join("featureTestDicts").join(format!("{sl}-{tl}.dict"));
    let text = std::fs::read_to_string(path).ok()?;
    let mut order: Vec<String> = Vec::new();
    let mut refs: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for line in text.lines() {
        let mut p = line.splitn(3, rlx_translate::phrasebook::SEP);
        if let (Some(s), Some(t)) = (p.next(), p.next())
            && !s.is_empty()
        {
            let e = refs.entry(s.to_string()).or_default();
            if e.is_empty() {
                order.push(s.to_string());
            }
            e.push(t.to_string());
        }
    }
    Some(
        order
            .into_iter()
            .map(|s| {
                let v = refs.remove(&s).unwrap_or_default();
                (s, v)
            })
            .collect(),
    )
}

fn bench(only: Option<&String>) -> Result<()> {
    let assets = Assets::discover();
    let wanted = only.map(|s| LangPair::parse(s)).transpose()?;

    println!(
        "{:<14} {:>7} {:>8} {:>10} {:>5} {:>6} {:>6} {:>9} {:>9}",
        "pair", "sources", "load_ms", "lookup/s", "gold", "found", "exact", "accuracy", "nmt chrF"
    );
    println!("{}", "-".repeat(86));

    let mut any = false;
    let (mut t_gold, mut t_found, mut t_exact, mut t_pairs) = (0usize, 0usize, 0usize, 0usize);
    let (mut t_nmt_scored, mut t_nmt_exact) = (0usize, 0usize);
    let mut t_nmt_chrf = 0.0f64;
    let (mut t_ap_n, mut t_ap_exact) = (0usize, 0usize);
    let mut t_ap_chrf = 0.0f64;
    let (mut t_cos, mut t_cos_n) = (0.0f64, 0usize);
    for pair in assets.pairs() {
        if let Some(w) = &wanted
            && &pair != w
        {
            continue;
        }
        let Ok((_, config)) = assets.best_config(&pair) else {
            continue;
        };
        let paths = assets.phrasebook_files(&config, &pair);
        if paths.is_empty() {
            continue;
        }

        let t0 = std::time::Instant::now();
        let Ok(pb) = Phrasebook::load_all(&paths) else {
            continue;
        };
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if pb.is_empty() {
            continue;
        }
        any = true;

        let mt = mt_dir(&assets, &config, &pair);
        let (norm, _) = load_normalizers(mt.as_ref());
        let gold = mt
            .as_ref()
            .and_then(|m| feature_tests(m, &pair))
            .unwrap_or_default();

        // Throughput on a fixed probe set drawn from the gold pairs, falling
        // back to a synthetic miss so every pair reports a number.
        let probes: Vec<String> = if gold.is_empty() {
            vec!["a phrase that is certainly absent".to_string()]
        } else {
            gold.iter().map(|(s, _)| s.clone()).collect()
        };
        let t1 = std::time::Instant::now();
        let mut hits = 0usize;
        const REPS: usize = 200;
        for _ in 0..REPS {
            for p in &probes {
                if pb.translate(p).is_some() {
                    hits += 1;
                }
            }
        }
        let per_sec = (REPS * probes.len()) as f64 / t1.elapsed().as_secs_f64();
        let _ = hits;

        let (mut exact, mut scored) = (0usize, 0usize);
        let mut misses: Vec<(String, Vec<String>)> = Vec::new();
        let os_ref = os_output(&pair);
        // (source, what we produced) for every gold source we translated.
        let mut ours: Vec<(String, String)> = Vec::new();
        for (src, want) in &gold {
            let q = norm
                .as_ref()
                .map_or_else(|| src.clone(), |n| n.normalize(src));
            if let Some(got) = pb.translate(&q) {
                scored += 1;
                ours.push((src.clone(), got.to_string()));
                if want.iter().any(|w| got == *w) {
                    exact += 1;
                } else if std::env::var("RLX_BENCH_SHOW").is_ok() {
                    println!("    PB   {pair} {src:?}\n      got  {got:?}\n      want {want:?}");
                }
            } else {
                misses.push((q, want.clone()));
            }
        }

        // What the phrasebook does not cover is exactly what the NMT is for.
        // Only single-hop pairs have a model of their own; the rest pivot
        // through English, which is not implemented.
        let (mut nmt_scored, mut nmt_exact) = (0usize, 0usize);
        let mut nmt_chrf = 0.0f64;
        // (our answer, its acceptable references) for the semantic score.
        let mut nmt_pairs: Vec<(String, Vec<String>)> = Vec::new();
        if !misses.is_empty()
            && let Some(home) = assets.model_home(&pair)
            && let Ok(params) = config
                .mt_app()
                .and_then(|d| d.translator(&pair))
                .and_then(rlx_translate::pdec::PDecParams::from_block)
            && let Ok(vocab) = rlx_translate::spm::Vocab::load(home.join("spm.model"))
        {
            let tgt: String = pair
                .target
                .chars()
                .take(2)
                .collect::<String>()
                .to_lowercase();
            // Search settings come from `Tuning`, which `Nmt::load` seeds from
            // the environment — `rlx-translate bench beam=16 no_repeat=0` and
            // the matching variables reach the decoder the same way.
            let src_lang: String = pair
                .source
                .chars()
                .take(2)
                .collect::<String>()
                .to_lowercase();
            if let Ok(nmt) = rlx_translate::decode::Nmt::load_for_pair(
                &home,
                &[home.as_path()],
                &src_lang,
                &tgt,
                &params.shortlist.lang_pair,
            ) {
                for (src, want) in &misses {
                    // Beam search, as the config asks for (`beam: 3`,
                    // `norm-costs`). It is not a cosmetic difference: for
                    // `i love the summer` -> ja, greedy gives 夏が大好きです and
                    // beam gives 夏が好きです, which is the reference answer.
                    let Ok(mut v) = nmt.translate_nbest(&vocab, &params, src, 1) else {
                        continue;
                    };
                    let Some(best) = v.drain(..).next() else {
                        continue;
                    };
                    let got = best.text;
                    ours.push((src.clone(), got.clone()));
                    nmt_pairs.push((got.clone(), want.clone()));
                    nmt_scored += 1;
                    nmt_chrf += want
                        .iter()
                        .map(|w| rlx_translate::score::chrf(&got, w))
                        .fold(0.0f64, f64::max);
                    if want
                        .iter()
                        .any(|w| got.trim().eq_ignore_ascii_case(w.trim()))
                    {
                        nmt_exact += 1;
                    } else if std::env::var("RLX_BENCH_SHOW").is_ok() {
                        println!(
                            "    NMT  {pair} {src:?}\n      got  {got:?}\n      want {want:?}"
                        );
                    }
                }
            }
        }
        // Semantic score: how close our answer is to a reference *in meaning*,
        // which character overlap cannot see. Centred on this pair's own
        // sentences, because raw cosine on post-norm encoder states is squashed
        // near 1.0 by a shared direction.
        let mut nmt_cos = 0.0f64;
        let mut cos_n = 0usize;
        if !nmt_pairs.is_empty()
            && let Some(home) = assets.model_home(&pair)
            && let Ok(vocab) = rlx_translate::spm::Vocab::load(home.join("spm.model"))
        {
            let tgt: String = pair
                .target
                .chars()
                .take(2)
                .collect::<String>()
                .to_lowercase();
            // Embeddings only — `sentence_embedding` never touches the
            // shortlist, so the default table is fine here.
            if let Ok(nmt) = rlx_translate::decode::Nmt::load(&home, &[home.as_path()], &tgt) {
                let mut texts: Vec<String> = Vec::new();
                for (got, refs) in &nmt_pairs {
                    texts.push(got.clone());
                    texts.extend(refs.iter().cloned());
                }
                let vecs: Vec<Vec<f32>> = texts
                    .iter()
                    .filter_map(|t| nmt.sentence_embedding(&vocab, &pair.target, t).ok())
                    .collect();
                if vecs.len() == texts.len() {
                    let mean = rlx_translate::score::center(&vecs);
                    let mut i = 0usize;
                    for (_, refs) in &nmt_pairs {
                        let g = &vecs[i];
                        let best = (1..=refs.len())
                            .map(|k| rlx_translate::score::cosine_centered(g, &vecs[i + k], &mean))
                            .fold(f64::MIN, f64::max);
                        if best > f64::MIN {
                            nmt_cos += best;
                            cos_n += 1;
                        }
                        i += 1 + refs.len();
                    }
                }
            }
        }
        t_cos += nmt_cos;
        t_cos_n += cos_n;

        t_nmt_scored += nmt_scored;
        t_nmt_exact += nmt_exact;
        t_nmt_chrf += nmt_chrf;

        let (mut ap_n, mut ap_exact, mut ap_chrf) = (0usize, 0usize, 0.0f64);
        if let Some(ref a) = os_ref {
            for (src, got) in &ours {
                let Some(theirs) = a.get(src) else { continue };
                ap_n += 1;
                if got.trim().eq_ignore_ascii_case(theirs.trim()) {
                    ap_exact += 1;
                }
                ap_chrf += rlx_translate::score::chrf(got, theirs);
                if got.trim() != theirs.trim() && std::env::var("RLX_BENCH_SHOW").is_ok() {
                    println!(
                        "    VS-OS {pair} {src:?}\n      ours  {got:?}\n      os    {theirs:?}"
                    );
                }
            }
        }
        t_ap_n += ap_n;
        t_ap_exact += ap_exact;
        t_ap_chrf += ap_chrf;
        let acc = if scored == 0 {
            "n/a".to_string()
        } else {
            format!("{:.1}%", 100.0 * exact as f64 / scored as f64)
        };
        t_gold += gold.len();
        t_found += scored;
        t_exact += exact;
        t_pairs += 1;
        println!(
            "{:<14} {:>7} {:>8.1} {:>10.0} {:>5} {:>6} {:>6} {:>9} {:>9}",
            pair.to_string(),
            pb.len(),
            load_ms,
            per_sec,
            gold.len(),
            scored,
            exact,
            acc,
            if nmt_scored == 0 {
                "-".to_string()
            } else {
                format!("{:.2}", nmt_chrf / nmt_scored as f64)
            }
        );
    }
    if !any {
        println!("(no pair has phrasebooks installed)");
        return Ok(());
    }
    println!("{}", "-".repeat(86));
    println!("{t_pairs} pairs · gold {t_gold} · covered by phrasebook {t_found} · exact {t_exact}");
    if t_found > 0 {
        println!(
            "accuracy on covered gold pairs: {:.1}%",
            100.0 * t_exact as f64 / t_found as f64
        );
    }
    if t_nmt_scored > 0 {
        println!(
            "NMT on the {t_nmt_scored} gold pairs the phrasebook misses: chrF {:.3}, \
             {t_nmt_exact} exact ({:.1}%)",
            t_nmt_chrf / t_nmt_scored as f64,
            100.0 * t_nmt_exact as f64 / t_nmt_scored as f64
        );
        if t_cos_n > 0 {
            println!(
                "  semantic agreement on those (centred encoder cosine): {:.3} over {t_cos_n}",
                t_cos / t_cos_n as f64
            );
        }
        println!(
            "  (these are dictionary entries the phrasebook exists to cover, so exact\n\
              match is a harsh reading of them; chrF is the comparable figure.)"
        );
        let covered = t_found + t_nmt_scored;
        println!(
            "end to end: {} of {t_gold} gold pairs translated, {} exact ({:.1}%)",
            covered,
            t_exact + t_nmt_exact,
            100.0 * (t_exact + t_nmt_exact) as f64 / covered.max(1) as f64
        );
    }
    if t_ap_n > 0 {
        println!(
            "agreement with the OS's live output on {t_ap_n} of those sources: \
             {t_ap_exact} identical ({:.1}%), chrF {:.3}",
            100.0 * t_ap_exact as f64 / t_ap_n as f64,
            t_ap_chrf / t_ap_n as f64
        );
    } else {
        println!(
            "(set RLX_TRANSLATE_REFERENCE to a dump of the OS's live output to also\n\
             score agreement with the OS, which is the target a port can reach.)"
        );
    }
    println!(
        "\nNote: gold sets come from `featureTestDicts`, which most pairs do not ship.\n\
         The `nmt` column is blank for pairs the OS routes through English, which\n\
         have no single-hop model; `tests/all_pairs.rs` scores the NMT on its own.\n\
         featureTestDicts is a spec, not a transcript: the OS's own framework scores\n\
         51.1% exact against it, so the gold columns have a ceiling well under 100%."
    );
    Ok(())
}

/// How a native output differs from the reference.
#[derive(Default)]
struct Diff {
    exact: usize,
    case_only: usize,
    space_only: usize,
    substantive: usize,
    no_entry: usize,
}

impl Diff {
    fn scored(&self) -> usize {
        self.exact + self.case_only + self.space_only + self.substantive
    }
}

/// Fold the cosmetic axes so each can be attributed separately.
fn fold(s: &str, case: bool, space: bool) -> String {
    let mut out = s.to_string();
    if space {
        out = out
            .chars()
            .map(|c| match c {
                '\u{00A0}' | '\u{202F}' => ' ',
                '\u{2019}' => '\'',
                other => other,
            })
            .collect();
    }
    if case {
        out = out.to_lowercase();
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Scores the native pipeline against `<ref-dir>/<src>-<tgt>.tsv`, each line
/// `source\ttarget` as produced by the real Translation framework.
fn parity(dir: Option<&String>, show: Option<&str>) -> Result<()> {
    let Some(dir) = dir else {
        bail!("expected a reference directory\n\n{USAGE}");
    };
    // Optional second argument: print up to N mismatching cases.
    let show: usize = show.and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut shown = 0usize;
    let assets = Assets::discover();
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {dir}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "tsv"))
        .collect();
    entries.sort();

    println!(
        "{:<14} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "pair", "lines", "hit", "exact", "case", "space", "exact%"
    );
    println!("{}", "-".repeat(64));

    let mut tot = Diff::default();
    let mut pairs = 0usize;
    for path in &entries {
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(pair) = LangPair::parse(stem) else {
            continue;
        };
        let Ok((_, config)) = assets.best_config(&pair) else {
            continue;
        };
        let paths = assets.phrasebook_files(&config, &pair);
        if paths.is_empty() {
            continue;
        }
        let Ok(pb) = Phrasebook::load_all(&paths) else {
            continue;
        };
        let mt = mt_dir(&assets, &config, &pair);
        let (norm, _) = load_normalizers(mt.as_ref());
        let case_locale = casemap_locale(&config, &pair);
        let text = std::fs::read_to_string(path)?;

        let mut d = Diff::default();
        let mut lines = 0usize;
        for line in text.lines() {
            let Some((src, want)) = line.split_once('\t') else {
                continue;
            };
            lines += 1;
            let q = norm
                .as_ref()
                .map_or_else(|| src.to_string(), |n| n.normalize(src));
            let Some(hit) = pb.translate(&q) else {
                d.no_entry += 1;
                continue;
            };
            // CaseMapBlock is the last stage of the shipped graph.
            let cased = match &case_locale {
                Some(loc) => sentence_case(hit, loc),
                None => hit.to_string(),
            };
            let got = &normalize_punctuation_for(&cased, &pair.target);
            if got == want {
                d.exact += 1;
            } else if fold(got, true, false) == fold(want, true, false) {
                d.case_only += 1;
                if shown < show {
                    shown += 1;
                    println!(
                        "  CASE  {pair} src={src:?}\n       got  {got:?}\n       live {want:?}"
                    );
                }
            } else if fold(got, true, true) == fold(want, true, true) {
                d.space_only += 1;
                if shown < show {
                    shown += 1;
                    println!(
                        "  SPACE {pair} src={src:?}\n       got  {got:?}\n       live {want:?}"
                    );
                }
            } else {
                d.substantive += 1;
                if shown < show {
                    shown += 1;
                    println!(
                        "  DIFF {pair} src={src:?}\n       got  {got:?}\n       live {want:?}"
                    );
                }
            }
        }
        if d.scored() == 0 {
            continue;
        }
        pairs += 1;
        println!(
            "{:<14} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7.1}%",
            pair.to_string(),
            lines,
            d.scored(),
            d.exact,
            d.case_only,
            d.space_only,
            100.0 * d.exact as f64 / d.scored() as f64
        );
        tot.exact += d.exact;
        tot.case_only += d.case_only;
        tot.space_only += d.space_only;
        tot.substantive += d.substantive;
        tot.no_entry += d.no_entry;
    }

    println!("{}", "-".repeat(64));
    let n = tot.scored();
    println!(
        "{pairs} pairs · phrasebook hit {n} · no entry {} (needs the NMT)",
        tot.no_entry
    );
    if n > 0 {
        println!(
            "exact {} ({:.1}%) · case-only {} ({:.1}%) · space/apostrophe {} ({:.1}%) · substantive {} ({:.1}%)",
            tot.exact,
            100.0 * tot.exact as f64 / n as f64,
            tot.case_only,
            100.0 * tot.case_only as f64 / n as f64,
            tot.space_only,
            100.0 * tot.space_only as f64 / n as f64,
            tot.substantive,
            100.0 * tot.substantive as f64 / n as f64
        );
    }
    Ok(())
}

/// Every ordered direction that has an `mt_app` graph.
///
/// A config file is named for one direction but defines both, so the number of
/// runnable directions is about twice the number of config-bearing pairs.
fn directions() -> Result<()> {
    let assets = Assets::discover();
    let mut seen = std::collections::BTreeSet::new();
    for pair in assets.pairs() {
        let Ok((_, config)) = assets.best_config(&pair) else {
            continue;
        };
        let Ok(decoder) = config.mt_app() else {
            continue;
        };
        for p in decoder.graphs.keys() {
            seen.insert(p.to_string());
        }
    }
    for p in &seen {
        println!("{p}");
    }
    eprintln!("{} ordered directions", seen.len());
    Ok(())
}

/// Exports every graph of the installed bundle to `<out>/`.
fn export(out: Option<&String>, lang: Option<&str>) -> Result<()> {
    let Some(out) = out else {
        bail!("expected an output directory\n\n{USAGE}");
    };
    let out = Path::new(out);
    let assets = Assets::discover();
    let dirs: Vec<std::path::PathBuf> = assets
        .roots
        .iter()
        .map(|r| r.join("MT"))
        .filter(|p| p.is_dir())
        .collect();
    // Several bundles are installed, each covering a different language group.
    // Pick the manifest that actually names the requested language, otherwise
    // its per-language decoder/handover/input graphs are silently missing.
    // Keep each manifest with the directory it came from: the shared graphs
    // (embedding/encoder/readout) have the SAME file name in every bundle, so
    // resolving them by name across all directories silently mixes a 48k-vocab
    // bundle's embedding with another bundle's decoder.
    let manifests: Vec<(std::path::PathBuf, espresso::Manifest)> = dirs
        .iter()
        .filter(|d| d.join("pyespresso.mdl.bin").exists())
        .filter_map(|d| {
            espresso::Manifest::load(d.join("pyespresso.mdl.bin"))
                .ok()
                .map(|m| (d.clone(), m))
        })
        .collect();
    if manifests.is_empty() {
        bail!("no pyespresso.mdl.bin installed");
    }
    let (home, m, lang) = match lang {
        Some(want) => {
            let (d, m) = manifests
                .iter()
                .find(|(_, m)| m.languages().contains(&want))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no installed bundle covers {want:?}; available: {:?}",
                        manifests
                            .iter()
                            .flat_map(|(_, m)| m.languages())
                            .collect::<Vec<_>>()
                    )
                })?;
            (d, m, want.to_string())
        }
        None => {
            let (d, m) = &manifests[0];
            let l = m
                .languages()
                .first()
                .map(|s| s.to_string())
                .ok_or_else(|| anyhow::anyhow!("manifest names no target languages"))?;
            (d, m, l)
        }
    };
    let wanted = m.graph_files_for(&lang);
    println!(
        "exporting {} graphs for {lang} -> {}",
        wanted.len(),
        out.display()
    );

    let mut graphs = Vec::new();
    let mut total = 0u64;
    for net in &wanted {
        // Prefer the manifest's own directory; only per-language graphs (which
        // ship in a separate `partial-<lang>` asset) may come from elsewhere.
        let Some(dir) = std::iter::once(home)
            .chain(dirs.iter())
            .find(|d| d.join(net).exists())
        else {
            println!("  skip {net} (not installed)");
            continue;
        };
        let g = rlx_translate::net::Graph::load(dir, net)?;
        let e = rlx_translate::export::export_graph(&g, out)?;
        let path = out.join(format!("{}.safetensors", g.name));
        let sz = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        total += sz;
        println!(
            "  {:<28} {:>4} ops  {:>8.1} MB",
            g.name,
            e.ops.len(),
            sz as f64 / 1e6
        );
        graphs.push(e);
    }

    // One manifest tying the graphs together, mirroring pyespresso.mdl.bin.
    let spec = serde_json::json!({
        "engine": m.str("BEspressoEngine"),
        "language": lang,
        "decoder_layers": m.decoder_layers(),
        "source_input": m.str("SourceInputStr"),
        "target_input": m.str("TargetInputStr"),
        "encoder_values": m.str("EncoderValuesStr"),
        "input_net_values": m.str("InputNetValuesStr"),
        "scores": m.str("ScoresStr"),
        "readout_input": m.str("ReadoutInputStr"),
        "readout_output": m.str("ReadoutOutputStr"),
        "handover": m.csv("HandoverStrings"),
        "state": m.csv("StateStrings"),
        "state_width": m.int("StateWidth"),
        "graphs": graphs,
    });
    let sp = out.join("model.json");
    std::fs::write(&sp, serde_json::to_vec_pretty(&spec)?)?;
    println!(
        "\nwrote {} + model.json ({:.1} MB of f32 weights)",
        wanted.len(),
        total as f64 / 1e6
    );
    println!("load with rlx_core::weight_map::WeightMap::from_file(<name>.safetensors)");
    Ok(())
}

/// Converts every installed bundle, and every shipped config, to all three
/// formats.
///
/// Bundles are keyed by the directory holding their own `pyespresso.mdl.bin`:
/// the shared graphs have identical file names across bundles, so they must be
/// read from their own manifest's directory or a decoder ends up paired with
/// another vocabulary's embedding.
fn convert(out: Option<&String>, dtype: Option<&str>) -> Result<()> {
    let Some(out) = out else {
        bail!("expected an output directory\n\n{USAGE}");
    };
    let out = Path::new(out);
    let dtype = match dtype.unwrap_or("f16") {
        "f16" => rlx_gguf::GgmlType::F16,
        "f32" => rlx_gguf::GgmlType::F32,
        "q8_0" | "q8" => rlx_gguf::GgmlType::Q8_0,
        other => bail!("unknown dtype {other:?}; expected f16, f32 or q8_0"),
    };
    let assets = Assets::discover();
    let dirs: Vec<std::path::PathBuf> = assets
        .roots
        .iter()
        .map(|r| r.join("MT"))
        .filter(|p| p.is_dir())
        .collect();

    // One entry per distinct bundle, deduplicated by the languages it covers:
    // the same bundle is symlinked under every pair directory that uses it.
    let mut seen: std::collections::BTreeMap<String, std::path::PathBuf> =
        std::collections::BTreeMap::new();
    for d in &dirs {
        if !d.join("pyespresso.mdl.bin").exists() {
            continue;
        }
        let Ok(m) = espresso::Manifest::load(d.join("pyespresso.mdl.bin")) else {
            continue;
        };
        let langs = m.languages();
        if langs.is_empty() {
            continue;
        }
        let key = format!("mt-{}", langs.join("-"));
        // Prefer the copy that has the most graphs beside it, so per-language
        // decoders resolve locally rather than through the whole root list.
        let score = |p: &std::path::Path| {
            std::fs::read_dir(p)
                .map(|it| it.filter_map(|e| e.ok()).count())
                .unwrap_or(0)
        };
        match seen.get(&key) {
            Some(prev) if score(prev) >= score(d) => {}
            _ => {
                seen.insert(key, d.clone());
            }
        }
    }
    if seen.is_empty() {
        bail!("no installed bundle has a pyespresso.mdl.bin");
    }
    println!(
        "converting {} bundles as {dtype:?} -> {}",
        seen.len(),
        out.display()
    );

    let mut reports = Vec::new();
    for (name, home) in &seen {
        print!("  {name} ... ");
        use std::io::Write;
        std::io::stdout().flush().ok();
        let r = rlx_translate::convert::convert_bundle(home, &dirs, out, name, dtype)?;
        println!(
            "{} graphs, {} tensors | safetensors {:.1} MB, gguf {:.1} MB, rlxp {:.1} MB{}",
            r.graphs.len(),
            r.tensors,
            r.safetensors_bytes as f64 / 1e6,
            r.gguf_bytes as f64 / 1e6,
            r.rlxp_bytes as f64 / 1e6,
            if r.missing.is_empty() {
                String::new()
            } else {
                format!(" ({} graphs not installed)", r.missing.len())
            }
        );
        reports.push(r);
    }

    // Every shipped configuration, as its own pack: these are what name the
    // pipeline stages, so weights without them are not runnable.
    let cfg_dir = out.join("configs");
    std::fs::create_dir_all(&cfg_dir)?;
    let mut n_cfg = 0usize;
    for c in &assets.configs {
        let Some(file) = c.path.file_name() else {
            continue;
        };
        if std::fs::copy(&c.path, cfg_dir.join(file)).is_ok() {
            n_cfg += 1;
        }
    }
    let cfg_pack = out.join("quasar-configs.rlxp");
    rlx_assets::pack::write_dir_rlxp(&cfg_dir, &cfg_pack)?;
    println!(
        "  configs ... {n_cfg} files | rlxp {:.1} MB",
        std::fs::metadata(&cfg_pack).map(|m| m.len()).unwrap_or(0) as f64 / 1e6
    );

    // The phrasebooks are lexicon data rather than tensors, so there is nothing
    // to re-encode — but they decide the answer for every sentence they cover,
    // and weights without them reproduce the OS only where it misses. One pack
    // per source language, named as the asset names them.
    let mut books: std::collections::BTreeMap<String, std::path::PathBuf> =
        std::collections::BTreeMap::new();
    for (name, dir) in &assets.asset_dirs {
        // Key on the *logical* asset name. Keying on the directory name put all
        // eighteen phrasebooks under `AssetData` and packed one.
        let pb = dir.join("PB");
        if pb.is_dir() {
            books.insert(name.clone(), pb);
        }
    }
    let mut n_pb = 0usize;
    let mut pb_bytes = 0u64;
    for (name, dir) in &books {
        // `PB-en` and friends: the asset directory name is the logical one.
        let pack = out.join(format!("{name}.rlxp"));
        if rlx_assets::pack::write_dir_rlxp(dir, &pack).is_ok() {
            pb_bytes += std::fs::metadata(&pack).map(|m| m.len()).unwrap_or(0);
            n_pb += 1;
        }
    }
    if n_pb > 0 {
        println!(
            "  phrasebooks ... {n_pb} packs | rlxp {:.1} MB",
            pb_bytes as f64 / 1e6
        );
    }

    let index = serde_json::json!({
        "format": "quasar-nmt",
        "gguf_dtype": format!("{dtype:?}"),
        "bundles": reports,
        "configs": n_cfg,
        "phrasebooks": n_pb,
    });
    std::fs::write(
        out.join("index.json"),
        serde_json::to_string_pretty(&index)?,
    )?;
    println!("wrote {}", out.join("index.json").display());
    Ok(())
}

/// Prints the `n` best translations of one sentence, with their scores.
///
/// This model frequently has more than one correct answer — for *i love the
/// summer* both `J'adore l'été` and `J'aime l'été` are right — so the
/// runners-up carry real information, and beam search computes them anyway.
fn nbest(pair: Option<&String>, text: Option<&str>, n: Option<usize>) -> Result<()> {
    let (Some(pair), Some(text)) = (pair, text) else {
        bail!("expected <src-tgt> and text\n\n{USAGE}");
    };
    let pair = LangPair::parse(pair)?;
    let n = n.unwrap_or(3).clamp(1, 16);
    let assets = Assets::discover();
    let home = assets.model_home(&pair).ok_or_else(|| {
        anyhow::anyhow!(
            "{pair} has no single-hop model installed; the OS routes it through English \
             and the two-block pivot is not implemented"
        )
    })?;
    let (_, config) = assets.best_config(&pair)?;
    let params = rlx_translate::pdec::PDecParams::from_block(config.mt_app()?.translator(&pair)?)?;
    let vocab = rlx_translate::spm::Vocab::load(home.join("spm.model"))?;
    let tgt: String = pair
        .target
        .chars()
        .take(2)
        .collect::<String>()
        .to_lowercase();
    // Every search setting — beam, length budget, repetition, `norm-costs` —
    // arrives through `Tuning`, seeded from the environment by `Nmt::load` and
    // from `key=value` arguments by `split_settings`. `rlx-translate tune`
    // prints what is in effect.
    let src_lang: String = pair
        .source
        .chars()
        .take(2)
        .collect::<String>()
        .to_lowercase();
    let nmt = rlx_translate::decode::Nmt::load_for_pair(
        &home,
        &[home.as_path()],
        &src_lang,
        &tgt,
        &params.shortlist.lang_pair,
    )?;
    let variants = nmt.translate_nbest(&vocab, &params, text, n)?;
    if variants.is_empty() {
        println!("(no hypothesis finished; try a longer length budget)");
        return Ok(());
    }
    println!("{pair}  {text:?}");
    for (i, v) in variants.iter().enumerate() {
        println!(
            "  {}. {:<48} score {:>9.3}  normalized {:>8.3}",
            i + 1,
            v.text,
            v.score,
            v.normalized_score
        );
    }
    Ok(())
}
