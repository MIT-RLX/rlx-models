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

//! Materialize HuggingFace ClinicalBERT layouts for RLX:
//! `vocab.txt` → `tokenizer.json`, `pytorch_model.bin` → `model.safetensors`.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "tokenizer")]
use tokenizers::Tokenizer;
#[cfg(feature = "tokenizer")]
use tokenizers::models::wordpiece::WordPiece;
#[cfg(feature = "tokenizer")]
use tokenizers::normalizers::BertNormalizer;
#[cfg(feature = "tokenizer")]
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
#[cfg(feature = "tokenizer")]
use tokenizers::processors::template::TemplateProcessing;

/// Ensure `tokenizer.json` and `model.safetensors` exist under `dir`.
pub fn prepare_clinicalbert_dir(dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        bail!("prepare_clinicalbert_dir: not a directory: {dir:?}");
    }
    #[cfg(feature = "tokenizer")]
    ensure_tokenizer_json(dir)?;
    #[cfg(not(feature = "tokenizer"))]
    if !dir.join("tokenizer.json").is_file() {
        bail!(
            "missing {} — rebuild rlx-clinicalbert with feature `tokenizer`",
            dir.join("tokenizer.json").display()
        );
    }

    #[cfg(feature = "prepare")]
    ensure_safetensors(dir)?;
    #[cfg(not(feature = "prepare"))]
    if !dir.join("model.safetensors").is_file() {
        bail!(
            "missing {} — rebuild rlx-clinicalbert with feature `prepare` \
             (or place a safetensors checkpoint)",
            dir.join("model.safetensors").display()
        );
    }
    Ok(())
}

#[cfg(feature = "tokenizer")]
fn ensure_tokenizer_json(dir: &Path) -> Result<()> {
    let out = dir.join("tokenizer.json");
    if out.is_file() {
        return Ok(());
    }
    let vocab = dir.join("vocab.txt");
    if !vocab.is_file() {
        bail!("missing vocab.txt under {dir:?}");
    }

    let (cls_id, sep_id, pad_id) = special_token_ids(&vocab)?;
    let vocab_str = vocab.to_str().context("vocab path utf8")?;
    let wordpiece = WordPiece::from_file(vocab_str)
        .unk_token("[UNK]".into())
        .build()
        .map_err(|e| anyhow::anyhow!("WordPiece build: {e}"))?;
    let mut tokenizer = Tokenizer::new(wordpiece);
    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
    tokenizer.with_normalizer(Some(BertNormalizer::new(false, true, None, true)));
    let post = TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .map_err(|e| anyhow::anyhow!("tokenizer single: {e}"))?
        .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
        .map_err(|e| anyhow::anyhow!("tokenizer pair: {e}"))?
        .special_tokens(vec![
            ("[CLS]".to_string(), cls_id),
            ("[SEP]".to_string(), sep_id),
            ("[PAD]".to_string(), pad_id),
        ])
        .build()
        .map_err(|e| anyhow::anyhow!("tokenizer post: {e}"))?;
    tokenizer.with_post_processor(Some(post));
    tokenizer
        .save(out.to_str().context("tokenizer path utf8")?, false)
        .map_err(|e| anyhow::anyhow!("save tokenizer.json: {e}"))?;
    eprintln!(
        "[rlx-clinicalbert] wrote {} (vocab={vocab:?})",
        out.display()
    );
    Ok(())
}

#[cfg(feature = "tokenizer")]
fn special_token_ids(vocab: &Path) -> Result<(u32, u32, u32)> {
    let mut ids: HashMap<String, u32> = HashMap::new();
    for (i, line) in std::fs::read_to_string(vocab)?.lines().enumerate() {
        let tok = line.trim();
        if !tok.is_empty() {
            ids.insert(tok.to_string(), i as u32);
        }
    }
    let cls = *ids.get("[CLS]").context("vocab missing [CLS]")?;
    let sep = *ids.get("[SEP]").context("vocab missing [SEP]")?;
    let pad = *ids.get("[PAD]").or_else(|| ids.get("<pad>")).unwrap_or(&0);
    Ok((cls, sep, pad))
}

#[cfg(feature = "prepare")]
fn ensure_safetensors(dir: &Path) -> Result<()> {
    let out = dir.join("model.safetensors");
    if out.is_file() {
        return Ok(());
    }
    let pth = dir.join("pytorch_model.bin");
    if pth.is_file() {
        if is_zip_pytorch(&pth)? {
            eprintln!(
                "[rlx-clinicalbert] converting {} → {}",
                pth.display(),
                out.display()
            );
            convert_pytorch_bin_to_safetensors(&pth, &out)?;
            return Ok(());
        }
        eprintln!(
            "[rlx-clinicalbert] legacy pytorch checkpoint — fetching model.safetensors from HF"
        );
        #[cfg(feature = "hf-download")]
        {
            try_download_safetensors_pr(dir)?;
            if out.is_file() {
                return Ok(());
            }
            bail!(
                "legacy {pth:?} is not zip-format and no safetensors mirror was found on Hugging Face"
            );
        }
        #[cfg(not(feature = "hf-download"))]
        bail!(
            "legacy pytorch checkpoint at {pth:?} requires feature `hf-download` to fetch safetensors"
        );
    }
    bail!("missing model.safetensors and pytorch_model.bin under {dir:?}")
}

#[cfg(feature = "prepare")]
fn is_zip_pytorch(path: &Path) -> Result<bool> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
    let mut magic = [0u8; 2];
    f.read_exact(&mut magic)
        .with_context(|| format!("read magic from {path:?}"))?;
    Ok(magic == *b"PK")
}

/// HF PR branches that host `model.safetensors` when `main` only has legacy pytorch.
#[cfg(all(feature = "prepare", feature = "hf-download"))]
fn safetensors_pr_revisions(variant: crate::ClinicalBertVariant) -> &'static [&'static str] {
    match variant {
        crate::ClinicalBertVariant::BioClinical => &["refs/pr/16"],
        crate::ClinicalBertVariant::BioDischarge => &["refs/pr/7"],
        crate::ClinicalBertVariant::Huang => &["refs/pr/11"],
    }
}

#[cfg(all(feature = "prepare", feature = "hf-download"))]
fn try_download_safetensors_pr(dir: &Path) -> Result<()> {
    use crate::config::ClinicalBertConfig;
    use hf_hub::{Repo, RepoType};

    let cfg_path = ClinicalBertConfig::config_json_path(dir);
    let variant = ClinicalBertConfig::from_file(&cfg_path)
        .ok()
        .and_then(|c| c.variant)
        .unwrap_or(crate::ClinicalBertVariant::BioClinical);

    let cache = crate::download::default_hf_cache_dir();
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache)
        .build()
        .context("hf_hub ApiBuilder")?;

    for rev in safetensors_pr_revisions(variant) {
        let repo = api.repo(Repo::with_revision(
            variant.hf_repo().to_string(),
            RepoType::Model,
            (*rev).to_string(),
        ));
        match repo.get("model.safetensors") {
            Ok(src) => {
                let out = dir.join("model.safetensors");
                std::fs::copy(&src, &out)
                    .with_context(|| format!("copy safetensors {src:?} -> {out:?}"))?;
                eprintln!("[rlx-clinicalbert] downloaded model.safetensors ({variant:?}, {rev})");
                return Ok(());
            }
            Err(e) => {
                eprintln!("[rlx-clinicalbert] skip {rev}: {e}");
            }
        }
    }
    Ok(())
}

#[cfg(feature = "prepare")]
fn convert_pytorch_bin_to_safetensors(_pth: &Path, _out: &Path) -> Result<()> {
    bail!(
        "pytorch_model.bin conversion is not supported; \
         place model.safetensors directly in the checkpoint directory"
    )
}

/// Default materialized model directory name for a variant download.
pub fn default_materialized_dir(base: &Path, variant: crate::ClinicalBertVariant) -> PathBuf {
    base.join(match variant {
        crate::ClinicalBertVariant::Huang => "clinicalbert-huang",
        crate::ClinicalBertVariant::BioClinical => "bio-clinicalbert",
        crate::ClinicalBertVariant::BioDischarge => "bio-discharge-bert",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_bio_clinicalbert_fixture() {
        let dir = Path::new("/tmp/bio-clinicalbert");
        if !dir.join("pytorch_model.bin").is_file() {
            eprintln!("skip: download Bio_ClinicalBERT to /tmp/bio-clinicalbert");
            return;
        }
        prepare_clinicalbert_dir(dir).expect("prepare");
        assert!(dir.join("tokenizer.json").is_file());
        assert!(dir.join("model.safetensors").is_file());
        let len = std::fs::metadata(dir.join("model.safetensors"))
            .expect("stat safetensors")
            .len();
        assert!(len > 400_000_000, "safetensors too small: {len}");
    }
}
