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

//! An exported language model checked against the family's own runner:
//! the package, opened by rlx alone, must give the prompt's logits and
//! greedily decode the tokens the [`LmRunner`] it was exported from does.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_runtime::lm::LmRunner;
use serde_json::json;

/// Where a package's decode left its runner's.
#[derive(Debug, Clone)]
pub struct Divergence {
    /// The prompt's index.
    pub prompt: usize,
    /// The first token the two differ at.
    pub at: usize,
    pub runner: Vec<u32>,
    pub package: Vec<u32>,
}

/// What [`check_generate`] found.
#[derive(Debug, Clone, Default)]
pub struct GenerateCheck {
    pub prompts: usize,
    /// Tokens each side decoded, over every prompt.
    pub tokens: usize,
    /// The largest difference between the two's logits after a prompt,
    /// relative to the runner's largest logit; `None` when the runner gives
    /// no logits (some return the sampled id in their place).
    pub logits_rel_diff: Option<f32>,
    pub divergences: Vec<Divergence>,
}

impl GenerateCheck {
    /// Every prompt decoded the same tokens.
    pub fn passed(&self) -> bool {
        self.divergences.is_empty()
    }
}

impl std::fmt::Display for GenerateCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} prompts, {} tokens: {}; ",
            self.prompts,
            self.tokens,
            if self.passed() {
                "the same tokens".to_string()
            } else {
                format!("{} diverged", self.divergences.len())
            },
        )?;
        match self.logits_rel_diff {
            Some(d) => write!(f, "logits within {d:.2e} of the runner's")?,
            None => write!(f, "the runner gives no logits to compare")?,
        }
        for d in &self.divergences {
            write!(
                f,
                "\n  prompt {} at token {}: runner {:?}, package {:?}",
                d.prompt, d.at, d.runner, d.package
            )?;
        }
        Ok(())
    }
}

/// Decode up to `n_new` tokens greedily after each of `prompts` with
/// `runner` and with `package` (a package's language model: `Model::lm`),
/// each ending at its first id in `stop` (kept), and compare: the tokens,
/// and the logits after each prompt.
pub fn check_generate(
    runner: &mut dyn LmRunner,
    package: &mut dyn LmRunner,
    prompts: &[Vec<u32>],
    n_new: usize,
    stop: &[u32],
) -> Result<GenerateCheck> {
    // A runner may stop at its own end token where the package's
    // `generate` goes on: both end at the first stop.
    let ended = |mut t: Vec<u32>| {
        if let Some(i) = t.iter().position(|x| stop.contains(x)) {
            t.truncate(i + 1);
        }
        t
    };
    let mut out = GenerateCheck {
        prompts: prompts.len(),
        ..Default::default()
    };
    let vocab = runner.vocab_size();
    for (i, p) in prompts.iter().enumerate() {
        // The host-driven pair's logits, else `predict_logits`': whichever
        // is a vocabulary's worth.
        let want = match runner.prefill_logits(p) {
            Ok(l) if l.len() >= vocab => Some(l),
            _ => runner.predict_logits(p).ok().filter(|l| l.len() >= vocab),
        };
        if let Some(want) = want {
            let got = package
                .predict_logits(p)
                .with_context(|| format!("the package's logits after prompt {i}"))?;
            let n = vocab.min(got.len());
            let scale = want[..n].iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
            let worst = want[..n]
                .iter()
                .zip(&got[..n])
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let d = out.logits_rel_diff.unwrap_or(0.0).max(worst / scale);
            out.logits_rel_diff = Some(d);
        }

        let a = ended(runner.generate(p, n_new, &mut |t| !stop.contains(&t))?);
        let b = ended(package.generate(p, n_new, &mut |t| !stop.contains(&t))?);
        out.tokens += a.len();
        if a != b {
            let at = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
            out.divergences.push(Divergence {
                prompt: i,
                at,
                runner: a,
                package: b,
            });
        }
    }
    Ok(out)
}

/// Open the package at `path` on `device` (`"cpu"`, `"metal"`, …) as rlx
/// alone does; its language model is `Model::lm`.
pub fn open_package(path: &Path, device: &str) -> Result<Box<dyn rlx_runtime::model::Model>> {
    let source = path.to_str().context("package path is not UTF-8")?;
    rlx_runtime::model::open_source(source, json!({"device": device}))
}

/// Write a Hugging Face–style checkpoint to `dir` for a test: `config`,
/// `tensors` (f32 safetensors, by name), and a word-level tokenizer of
/// `vocab` ids — `<s>` 1, `</s>` 2 (the stop id), and `t3`, `t4`, … — so a
/// family's exporter can be checked against its runner on random weights
/// with no download.
pub fn write_checkpoint(
    dir: &Path,
    config: &serde_json::Value,
    tensors: &std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    vocab: usize,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for n in &names {
        let (data, shape) = &tensors[*n];
        let bytes = data.len() * 4;
        header.insert(
            (*n).clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        offset += bytes;
    }
    let mut head = serde_json::to_vec(&header)?;
    head.resize(head.len().div_ceil(8) * 8, b' ');
    let mut file = Vec::with_capacity(8 + head.len() + offset);
    file.extend((head.len() as u64).to_le_bytes());
    file.extend(&head);
    for n in &names {
        for x in &tensors[*n].0 {
            file.extend(x.to_le_bytes());
        }
    }
    std::fs::write(dir.join("model.safetensors"), file)?;
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(config)?)?;
    let mut words = serde_json::Map::new();
    for (i, w) in ["[UNK]", "<s>", "</s>"].iter().enumerate() {
        words.insert(w.to_string(), json!(i));
    }
    for i in 3..vocab {
        words.insert(format!("t{i}"), json!(i));
    }
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": words, "unk_token": "[UNK]"},
    });
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tokenizer)?)?;
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_vec(&json!({"bos_token": "<s>", "eos_token": "</s>"}))?,
    )?;
    std::fs::write(
        dir.join("generation_config.json"),
        serde_json::to_vec(&json!({"eos_token_id": 2}))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Next token = last + `step` (mod 50), logits one-hot.
    struct Count(u32);

    impl LmRunner for Count {
        fn family(&self) -> &'static str {
            "count"
        }
        fn vocab_size(&self) -> usize {
            50
        }
        fn predict_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
            let mut l = vec![0f32; 50];
            l[((ids.last().copied().unwrap_or(0) + self.0) % 50) as usize] = 1.0;
            Ok(l)
        }
    }

    #[test]
    fn a_runner_matches_itself_and_a_different_one_diverges_where_it_does() {
        let prompts = [vec![1, 2, 3], vec![7]];
        let same = check_generate(&mut Count(1), &mut Count(1), &prompts, 4, &[]).unwrap();
        assert!(same.passed(), "{same}");
        assert_eq!((same.tokens, same.logits_rel_diff), (8, Some(0.0)));

        let off = check_generate(&mut Count(1), &mut Count(2), &prompts, 4, &[]).unwrap();
        assert_eq!(off.divergences.len(), 2);
        assert_eq!(off.divergences[0].at, 0);
        assert_eq!(off.divergences[0].runner, [4, 5, 6, 7]);
        assert_eq!(off.logits_rel_diff, Some(1.0));
    }

    /// Gives the sampled id where its logits should be.
    struct NoLogits;

    impl LmRunner for NoLogits {
        fn family(&self) -> &'static str {
            "none"
        }
        fn vocab_size(&self) -> usize {
            50
        }
        fn predict_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
            Ok(vec![(ids.last().copied().unwrap_or(0) + 1) as f32])
        }
        fn generate(
            &mut self,
            p: &[u32],
            n: usize,
            _: &mut dyn FnMut(u32) -> bool,
        ) -> Result<Vec<u32>> {
            Count(1).generate(p, n, &mut |_| true)
        }
    }

    #[test]
    fn a_runner_without_logits_is_checked_on_its_tokens_alone() {
        let c = check_generate(&mut NoLogits, &mut Count(1), &[vec![3]], 3, &[]).unwrap();
        assert!(c.passed(), "{c}");
        assert_eq!(c.logits_rel_diff, None);
        assert!(c.to_string().contains("no logits"), "{c}");
    }

    /// [`Count`] that ends at its own `end` token, as runners with an EOS do.
    struct Ends(u32);

    impl LmRunner for Ends {
        fn family(&self) -> &'static str {
            "ends"
        }
        fn vocab_size(&self) -> usize {
            50
        }
        fn predict_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
            Count(1).predict_logits(ids)
        }
        fn generate(
            &mut self,
            p: &[u32],
            n: usize,
            on: &mut dyn FnMut(u32) -> bool,
        ) -> Result<Vec<u32>> {
            let end = self.0;
            Count(1).generate(p, n, &mut |t| on(t) && t != end)
        }
    }

    #[test]
    fn a_runner_ending_at_its_stop_token_matches_a_package_that_would_go_on() {
        // 3 → 4 5 6: the runner ends at 5, the package would go on to 6.
        let c = check_generate(&mut Ends(5), &mut Count(1), &[vec![3]], 3, &[5]).unwrap();
        assert!(c.passed(), "{c}");
        assert_eq!(c.tokens, 2);
        // Without the stop, the two differ.
        let c = check_generate(&mut Ends(5), &mut Count(1), &[vec![3]], 3, &[]).unwrap();
        assert!(!c.passed());
    }
}
