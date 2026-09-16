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

//! Locating the assets macOS installs for on-device translation.
//!
//! This crate ships and redistributes no model data; everything is read from
//! what the operating system has downloaded.
//!
//! # Two separate downloads
//!
//! The **configs** arrive with the `com.apple.MobileAsset.UAF.Translation.Assets`
//! catalog and land in
//!
//! ```text
//!   /System/Library/AssetsV2/com_apple_MobileAsset_UAF_Translation_Assets/
//!       purpose_auto/<sha1>.asset/AssetData/mt_app.<src>-<tgt>.<variant>.json
//! ```
//!
//! The **weights** do not. They belong to separately named assets
//! (`MT-bi-en-es-de-it-fr-pt-nl-0` for the western-European bundle, `PB-<lang>`
//! for phrasebooks) that only appear once a language pair has actually been
//! installed from System Settings → General → Language & Region →
//! *Translation Languages*. Every file a block references is stored under its
//! asset name as the first path component, e.g.
//! `MT-bi-en-es-de-it-fr-pt-nl-0/MT/spm.model`, which is what
//! [`Assets::resolve`] looks for.
//!
//! Set `RLX_TRANSLATE_ASSETS` to a colon-separated list of directories to
//! search additional roots first — useful for pointing at a copy of the asset
//! tree rather than the live system one.

use crate::quasar::{LangPair, QuasarConfig};
use anyhow::{Context, Result, anyhow};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Where macOS keeps the translation config asset.
pub const SYSTEM_CONFIG_ASSET_ROOT: &str =
    "/System/Library/AssetsV2/com_apple_MobileAsset_UAF_Translation_Assets/purpose_auto";

/// Pre-installed EMT packages that ship in the OS image for a few languages.
pub const SYSTEM_LINGUISTIC_DATA: &str = "/System/Library/LinguisticData";

/// Environment override: colon-separated extra roots, searched first.
pub const ENV_ASSET_PATH: &str = "RLX_TRANSLATE_ASSETS";

/// A config file found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigFile {
    /// Task the config belongs to, from the file name (`mt_app`).
    pub task: String,
    pub pair: LangPair,
    /// Variant suffix — macOS ships `0` and `20` per pair.
    pub variant: String,
    pub path: PathBuf,
}

/// The translation assets visible on this machine.
#[derive(Debug, Clone, Default)]
pub struct Assets {
    /// Directories that may contain model files.
    pub roots: Vec<PathBuf>,
    /// Discovered per-pair configs, keyed by `(pair, task, variant)`.
    pub configs: Vec<ConfigFile>,
    /// Normalised asset specifier → the directory holding that asset's files.
    ///
    /// A block names its files `<asset-name>/<path>`, but on disk each asset is
    /// an opaque `<sha1>.asset/AssetData/` whose contents start at `<path>`;
    /// the logical name lives in the asset's `Info.plist` as `AssetSpecifier`.
    /// Both sides go through [`normalize_asset_name`] before matching.
    pub asset_dirs: BTreeMap<String, PathBuf>,
}

impl Assets {
    /// Scans the default locations plus anything in `RLX_TRANSLATE_ASSETS`.
    pub fn discover() -> Self {
        let mut roots = Vec::new();
        if let Ok(extra) = std::env::var(ENV_ASSET_PATH) {
            for p in extra.split(':').filter(|s| !s.is_empty()) {
                roots.push(PathBuf::from(p));
            }
        }
        for dir in asset_data_dirs(Path::new(SYSTEM_CONFIG_ASSET_ROOT)) {
            roots.push(dir);
        }
        for dir in emt_package_dirs(Path::new(SYSTEM_LINGUISTIC_DATA)) {
            roots.push(dir);
        }
        Self::from_roots(roots)
    }

    /// Builds from explicit roots, scanning each for config files.
    pub fn from_roots(roots: Vec<PathBuf>) -> Self {
        let mut configs = Vec::new();
        for root in &roots {
            let Ok(entries) = std::fs::read_dir(root) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && let Some(cfg) = parse_config_name(name, &path)
                {
                    configs.push(cfg);
                }
            }
        }
        configs.sort_by(|a, b| (&a.pair, &a.task, &a.variant).cmp(&(&b.pair, &b.task, &b.variant)));
        let asset_dirs = roots
            .iter()
            .filter_map(|r| asset_specifier_for(r).map(|s| (s, r.clone())))
            .collect();
        Self {
            roots,
            configs,
            asset_dirs,
        }
    }

    /// Language pairs with at least one config present.
    pub fn pairs(&self) -> BTreeSet<LangPair> {
        self.configs.iter().map(|c| c.pair.clone()).collect()
    }

    /// Configs covering a pair, in variant order.
    ///
    /// A config *file* is named for one direction but its `mt-decoders` define
    /// **both**, so `fr_FR-en_US` is served by `mt_app.en_US-fr_FR.*.json`.
    /// Matching on the file name alone silently loses every reverse direction —
    /// half the runnable pairs — so the reversed name is a fallback.
    pub fn configs_for(&self, pair: &LangPair) -> Vec<&ConfigFile> {
        let direct: Vec<&ConfigFile> = self.configs.iter().filter(|c| &c.pair == pair).collect();
        if !direct.is_empty() {
            return direct;
        }
        let rev = pair.reversed();
        self.configs.iter().filter(|c| c.pair == rev).collect()
    }

    /// The lowest-numbered variant for `pair`.
    ///
    /// Prefer [`Assets::best_config`] for anything that will actually read
    /// model files: the variant number selects a *different asset* (variant
    /// `20` uses `MT-bi-…-20`), and only the installed one resolves.
    pub fn lowest_variant_config(&self, pair: &LangPair) -> Result<&ConfigFile> {
        self.configs_for(pair)
            .into_iter()
            .min_by_key(|c| {
                (
                    c.variant.parse::<i64>().unwrap_or(i64::MAX),
                    c.variant.clone(),
                )
            })
            .ok_or_else(|| {
                anyhow!(
                    "no translation config for {pair} in any of {:?}",
                    self.roots
                )
            })
    }

    /// The config variant whose files are actually installed.
    ///
    /// A pair ships several variants and each points at a *different* model
    /// asset — variant `20` at `MT-bi-…-20`, variant `0` at `MT-bi-…-0`. Only
    /// one is normally downloaded, so picking by variant number alone yields a
    /// config whose every file is missing. This loads each candidate and keeps
    /// the one with the fewest unresolved files, breaking ties on the lower
    /// variant number.
    pub fn best_config(&self, pair: &LangPair) -> Result<(ConfigFile, QuasarConfig)> {
        let candidates = self.configs_for(pair);
        if candidates.is_empty() {
            return Err(anyhow!(
                "no translation config for {pair} in any of {:?}",
                self.roots
            ));
        }
        let mut best: Option<(usize, i64, ConfigFile, QuasarConfig)> = None;
        let mut last_err = None;
        for cf in candidates {
            let cfg = match QuasarConfig::load(&cf.path) {
                Ok(c) => c,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };
            let missing = self
                .availability(&cfg, &cf.task, pair)
                .map(|a| a.missing.len())
                .unwrap_or(usize::MAX);
            let variant = cf.variant.parse::<i64>().unwrap_or(i64::MAX);
            let better = best
                .as_ref()
                .is_none_or(|(m, v, _, _)| (missing, variant) < (*m, *v));
            if better {
                best = Some((missing, variant, cf.clone(), cfg));
            }
        }
        match best {
            Some((_, _, cf, cfg)) => Ok((cf, cfg)),
            None => {
                Err(last_err
                    .unwrap_or_else(|| anyhow!("no readable translation config for {pair}")))
            }
        }
    }

    /// Resolves an asset-relative path such as
    /// `MT-bi-en-es-de-it-fr-pt-nl-20/MT/spm.model`.
    ///
    /// Tried in order: the asset's own directory (matched on its `Info.plist`
    /// specifier), then the path verbatim under each root.
    ///
    /// Deliberately no "drop the asset name and search" fallback. Variants of
    /// the same model use identical inner paths — `MT-bi-…-0/MT/spm.model` and
    /// `MT-bi-…-20/MT/spm.model` — so such a fallback silently resolves the
    /// uninstalled variant to the installed variant's weights. A root supplied
    /// through `RLX_TRANSLATE_ASSETS` should therefore either carry the asset's
    /// `Info.plist` or lay files out under `<asset-name>/`.
    /// Phrasebook files this pair's graph consults, in graph order.
    ///
    /// A pivot pair has several `PhraseBookBlock`s — one per hop plus a
    /// pivot-level one — and they name their files in `pb-file-list`.
    pub fn phrasebook_files(&self, config: &QuasarConfig, pair: &LangPair) -> Vec<PathBuf> {
        let Ok(decoder) = config.mt_app() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for b in decoder.blocks_for(pair) {
            if b.kind != crate::quasar::BlockKind::PhraseBook {
                continue;
            }
            for rel in b.csv("pb-file-list") {
                if let Some(p) = self.resolve(&rel)
                    && !out.contains(&p)
                {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Directory of the bundle a translator block names, for this pair.
    ///
    /// A pivot pair has both bundles side by side under its `AssetsV3`
    /// directory — `ar_AE-de_DE` carries `MT-bi-en-ar` *and* the seven-language
    /// one — so the block's `model-file` is what picks between them.
    ///
    /// The config routinely names a variant that is not installed (blocks say
    /// `MT-bi-en-ar-0`, the machine has `-20`), so an exact match is tried
    /// first and then the trailing variant number is ignored. Exact first
    /// matters: variants are genuinely different models, and silently taking
    /// another one is the failure mode [`Assets::resolve`] exists to avoid.
    pub fn model_home_for(&self, pair: &LangPair, model_file: &str) -> Option<PathBuf> {
        let want = normalize_asset_name(model_file.split('/').next()?);
        let base = PathBuf::from(std::env::var("HOME").ok()?).join("Library/Translation/AssetsV3");
        let dir = [
            base.join(format!("{}-{}", pair.source, pair.target)),
            base.join(format!("{}-{}", pair.target, pair.source)),
        ]
        .into_iter()
        .find(|d| d.is_dir())?;
        let entries: Vec<PathBuf> = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        let name_of = |p: &std::path::Path| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(normalize_asset_name)
                .unwrap_or_default()
        };
        let devariant = |n: &str| {
            n.rsplit_once('-')
                .filter(|(_, v)| !v.is_empty() && v.chars().all(|c| c.is_ascii_digit()))
                .map_or_else(|| n.to_string(), |(stem, _)| stem.to_string())
        };
        entries
            .iter()
            .find(|p| name_of(p) == want)
            .or_else(|| {
                let bare = devariant(&want);
                entries.iter().find(|p| devariant(&name_of(p)) == bare)
            })
            .map(|p| p.join("MT"))
    }

    /// Directory holding a coherent model for `pair`, if it is a single hop.
    ///
    /// The model is split across assets: `MT-bi-...-partial-<lang>` ships only
    /// that language's decoder/input graphs, while the shared encoder,
    /// embedding, `spm.model` and manifest live in another. `AssetsV3` is where
    /// the OS symlinks them into one per-pair view, and a *direct* pair has
    /// exactly one bundle there. Resolving the config's `model-file` instead
    /// lands on the config asset's own `MT/`, which holds no decoders at all.
    ///
    /// Returns `None` for a pair the OS routes through English, which has no
    /// single bundle of its own.
    pub fn model_home(&self, pair: &LangPair) -> Option<PathBuf> {
        let base = PathBuf::from(std::env::var("HOME").ok()?).join("Library/Translation/AssetsV3");
        let dir = [
            base.join(format!("{}-{}", pair.source, pair.target)),
            base.join(format!("{}-{}", pair.target, pair.source)),
        ]
        .into_iter()
        .find(|d| d.is_dir())?;
        let bundles: Vec<PathBuf> = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("MT-"))
            })
            .collect();
        // More than one bundle means the pair is served by a pivot.
        let [only] = bundles.as_slice() else {
            return None;
        };
        let home = only.join("MT");
        let tgt: String = pair
            .target
            .chars()
            .take(2)
            .collect::<String>()
            .to_lowercase();
        home.join(format!("decoder_{tgt}.espresso.net"))
            .exists()
            .then_some(home)
    }

    pub fn resolve(&self, relative: &str) -> Option<PathBuf> {
        if let Some((asset, rest)) = relative.split_once('/')
            && let Some(dir) = self.asset_dirs.get(&normalize_asset_name(asset))
        {
            let candidate = dir.join(rest);
            if candidate.exists() {
                return Some(candidate);
            }
        }
        for root in &self.roots {
            let candidate = root.join(relative);
            if candidate.exists() {
                return Some(candidate);
            }
        }
        None
    }

    /// Splits the files `pair` needs under `task` into present and missing.
    ///
    /// This is the check that answers "can we actually translate yet?" — with
    /// only the config asset installed, every entry comes back missing.
    pub fn availability(
        &self,
        config: &QuasarConfig,
        task: &str,
        pair: &LangPair,
    ) -> Result<Availability> {
        let required = config.required_files(task, pair)?;
        let mut present = BTreeMap::new();
        let mut missing = BTreeSet::new();
        for rel in required {
            match self.resolve(&rel) {
                Some(p) => {
                    present.insert(rel, p);
                }
                None => {
                    missing.insert(rel);
                }
            }
        }
        Ok(Availability {
            pair: pair.clone(),
            task: task.to_string(),
            present,
            missing,
        })
    }
}

/// Result of [`Assets::availability`].
#[derive(Debug, Clone)]
pub struct Availability {
    pub pair: LangPair,
    pub task: String,
    /// Asset-relative path → resolved absolute path.
    pub present: BTreeMap<String, PathBuf>,
    /// Asset-relative paths with no file behind them.
    pub missing: BTreeSet<String>,
}

impl Availability {
    /// True when every referenced file is on disk.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }

    /// Distinct asset names among the missing files — what still has to be
    /// downloaded, rather than the individual files.
    pub fn missing_assets(&self) -> BTreeSet<String> {
        self.missing
            .iter()
            .filter_map(|p| p.split('/').next())
            .map(str::to_string)
            .collect()
    }
}

/// `mt_app.en_US-fr_FR.0.json` → a [`ConfigFile`].
fn parse_config_name(name: &str, path: &Path) -> Option<ConfigFile> {
    let stem = name.strip_suffix(".json")?;
    let mut parts = stem.split('.');
    let task = parts.next()?.to_string();
    let pair = LangPair::parse(parts.next()?).ok()?;
    let variant = parts.next().unwrap_or("0").to_string();
    if parts.next().is_some() {
        return None;
    }
    Some(ConfigFile {
        task,
        pair,
        variant,
        path: path.to_path_buf(),
    })
}

/// Reduces an asset name or an `AssetSpecifier` to a comparable form.
///
/// The two spellings differ in punctuation and in one filler word: the config
/// says `MT-bi-en-es-de-it-fr-pt-nl-partial-fr-20`, the installed asset says
/// `com.apple.sequoia.asset.mt.bi-en-es-de-it-fr-pt-nl.fr.20`. Lower-casing,
/// dropping the vendor prefix, folding `.`/`-` to a single separator and
/// removing the `partial` filler makes them equal.
pub fn normalize_asset_name(name: &str) -> String {
    let name = name.to_ascii_lowercase();
    let name = name
        .strip_prefix("com.apple.sequoia.asset.")
        .unwrap_or(&name);
    name.split(['.', '-'])
        .filter(|p| !p.is_empty() && *p != "partial")
        .collect::<Vec<_>>()
        .join("-")
}

/// Reads an asset directory's logical name from the `Info.plist` beside it.
///
/// The plist is binary, so rather than take a plist dependency this scans for
/// the `com.apple.sequoia.asset.` string, which is stored literally in both the
/// binary and XML encodings.
fn asset_specifier_for(asset_data_dir: &Path) -> Option<String> {
    const PREFIX: &str = "com.apple.sequoia.asset.";
    let info = asset_data_dir.parent()?.join("Info.plist");
    let bytes = std::fs::read(info).ok()?;
    let start = bytes
        .windows(PREFIX.len())
        .position(|w| w == PREFIX.as_bytes())?;
    let tail = &bytes[start..];
    let end = tail
        .iter()
        .position(|b| !(b.is_ascii_alphanumeric() || *b == b'.' || *b == b'-'))
        .unwrap_or(tail.len());
    let s = std::str::from_utf8(&tail[..end]).ok()?;
    Some(normalize_asset_name(s))
}

/// `<root>/<sha1>.asset/AssetData` directories.
fn asset_data_dirs(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path().join("AssetData"))
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out
}

/// `<root>/RequiredAssets_<lang>.bundle/AssetData/EMT_package` directories.
fn emt_package_dirs(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path().join("AssetData").join("EMT_package"))
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out
}

/// Loads the config for `pair` whose model files are actually installed.
pub fn load_config(assets: &Assets, pair: &LangPair) -> Result<QuasarConfig> {
    assets
        .best_config(pair)
        .map(|(_, cfg)| cfg)
        .with_context(|| format!("loading config for {pair}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_file_names_parse() {
        let c = parse_config_name("mt_app.en_US-fr_FR.0.json", Path::new("/x")).expect("parses");
        assert_eq!(c.task, "mt_app");
        assert_eq!(c.pair.to_string(), "en_US-fr_FR");
        assert_eq!(c.variant, "0");

        let c = parse_config_name("mt_app.zh_TW-de_DE.20.json", Path::new("/x")).expect("parses");
        assert_eq!(c.variant, "20");
    }

    #[test]
    fn non_config_names_are_ignored() {
        assert!(parse_config_name("assets.json", Path::new("/x")).is_none());
        assert!(parse_config_name("detector.json", Path::new("/x")).is_none());
        assert!(parse_config_name("SPG.nnet", Path::new("/x")).is_none());
        assert!(parse_config_name("mt_app.en_US-fr_FR.0.extra.json", Path::new("/x")).is_none());
    }

    #[test]
    fn lowest_variant_config_picks_numerically_not_lexically() {
        let mk = |variant: &str| ConfigFile {
            task: "mt_app".into(),
            pair: LangPair::parse("en_US-fr_FR").expect("pair"),
            variant: variant.into(),
            path: PathBuf::from(format!("/x/{variant}")),
        };
        let assets = Assets {
            // "20" sorts before "3" as a string; the numeric key must win.
            configs: vec![mk("20"), mk("3")],
            ..Assets::default()
        };
        let pair = LangPair::parse("en_US-fr_FR").expect("pair");
        assert_eq!(
            assets.lowest_variant_config(&pair).expect("found").variant,
            "3"
        );
    }

    #[test]
    fn missing_pair_is_an_error_naming_the_pair() {
        let assets = Assets::default();
        let pair = LangPair::parse("en_US-fr_FR").expect("pair");
        let err = assets
            .lowest_variant_config(&pair)
            .expect_err("nothing installed");
        assert!(err.to_string().contains("en_US-fr_FR"), "{err}");
    }

    #[test]
    fn missing_assets_are_grouped_by_asset_name() {
        let a = Availability {
            pair: LangPair::parse("en_US-fr_FR").expect("pair"),
            task: "mt_app".into(),
            present: BTreeMap::new(),
            missing: [
                "MT-bi-en-es-de-it-fr-pt-nl-0/MT/spm.model".to_string(),
                "MT-bi-en-es-de-it-fr-pt-nl-0/MT/pyespresso.mdl.bin".to_string(),
                "PB-en/PB/en_US-fr_FR.mt_app.dict".to_string(),
            ]
            .into_iter()
            .collect(),
        };
        assert!(!a.is_complete());
        let names = a.missing_assets();
        assert_eq!(names.len(), 2);
        assert!(names.contains("MT-bi-en-es-de-it-fr-pt-nl-0"));
        assert!(names.contains("PB-en"));
    }

    #[test]
    fn asset_names_and_specifiers_normalise_to_the_same_form() {
        assert_eq!(
            normalize_asset_name("com.apple.sequoia.asset.mt.bi-en-es-de-it-fr-pt-nl.20"),
            normalize_asset_name("MT-bi-en-es-de-it-fr-pt-nl-20")
        );
        // The config inserts a `partial` the installed specifier omits.
        assert_eq!(
            normalize_asset_name("com.apple.sequoia.asset.mt.bi-en-es-de-it-fr-pt-nl.fr.20"),
            normalize_asset_name("MT-bi-en-es-de-it-fr-pt-nl-partial-fr-20")
        );
        assert_eq!(
            normalize_asset_name("com.apple.sequoia.asset.pb.fr"),
            normalize_asset_name("PB-fr")
        );
        // Distinct assets must stay distinct.
        assert_ne!(
            normalize_asset_name("MT-bi-en-es-de-it-fr-pt-nl-0"),
            normalize_asset_name("MT-bi-en-es-de-it-fr-pt-nl-20")
        );
        assert_ne!(normalize_asset_name("PB-fr"), normalize_asset_name("PB-en"));
    }

    #[test]
    fn resolve_does_not_confuse_model_variants() {
        // A root holding variant 20's files must NOT satisfy a variant 0 path:
        // the inner paths are identical, so a name-insensitive search would
        // silently hand back the wrong weights.
        let tmp = std::env::temp_dir().join(format!("rlx-tr-var-{}", std::process::id()));
        let mt = tmp.join("root/MT");
        std::fs::create_dir_all(&mt).expect("mkdir");
        std::fs::write(mt.join("spm.model"), b"x").expect("write");
        let assets = Assets {
            roots: vec![tmp.join("root")],
            asset_dirs: [(
                normalize_asset_name("MT-bi-en-es-de-it-fr-pt-nl-20"),
                tmp.join("root"),
            )]
            .into_iter()
            .collect(),
            ..Assets::default()
        };
        assert!(
            assets
                .resolve("MT-bi-en-es-de-it-fr-pt-nl-20/MT/spm.model")
                .is_some()
        );
        assert!(
            assets
                .resolve("MT-bi-en-es-de-it-fr-pt-nl-0/MT/spm.model")
                .is_none(),
            "variant 0 must not resolve to variant 20's file"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_prefers_earlier_roots() {
        let tmp = std::env::temp_dir().join(format!("rlx-translate-{}", std::process::id()));
        let a = tmp.join("a/ASSET/MT");
        let b = tmp.join("b/ASSET/MT");
        std::fs::create_dir_all(&a).expect("mkdir a");
        std::fs::create_dir_all(&b).expect("mkdir b");
        std::fs::write(a.join("spm.model"), b"A").expect("write a");
        std::fs::write(b.join("spm.model"), b"B").expect("write b");

        let assets = Assets {
            roots: vec![tmp.join("a"), tmp.join("b")],
            ..Assets::default()
        };
        let hit = assets.resolve("ASSET/MT/spm.model").expect("resolves");
        assert_eq!(std::fs::read(&hit).expect("read"), b"A");
        assert!(assets.resolve("ASSET/MT/nope").is_none());

        std::fs::remove_dir_all(&tmp).ok();
    }
}
