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

//! STEP 2 validation: the rlx Qwen2-0.5B LM greedy-decode must match the HF
//! transformers reference (`scripts/ref.py`) token-for-token. Skips unless the
//! weights (`weights/tts/miratts/model.safetensors`) + fixtures
//! (`tests/fixtures/lm_*`) are present.

use std::path::PathBuf;

use rlx_miratts::{MiraConfig, lm::MiraLm};
use rlx_runtime::Device;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_i64(p: &PathBuf) -> Vec<i64> {
    std::fs::read(p)
        .unwrap_or_else(|_| panic!("read {}", p.display()))
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn qwen2_lm_greedy_matches_transformers() {
    let dir = root().join("weights/tts/miratts");
    let fix = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    if !dir.join("model.safetensors").is_file() || !fix.join("lm_prompt_ids.i64").is_file() {
        eprintln!("skip: MiraTTS weights / LM fixtures not present (run scripts/ref.py)");
        return;
    }
    let prompt: Vec<u32> = read_i64(&fix.join("lm_prompt_ids.i64"))
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let ref_greedy: Vec<u32> = read_i64(&fix.join("lm_greedy_ids.i64"))
        .into_iter()
        .map(|x| x as u32)
        .collect();

    let cfg = MiraConfig::load(&dir).unwrap_or_default();
    let mut lm = MiraLm::load(&dir, &cfg, Device::Cpu).expect("load Qwen2 LM");
    let got = lm
        .generate_greedy(&prompt, ref_greedy.len())
        .expect("greedy decode");

    eprintln!("ref:  {ref_greedy:?}\nrlx:  {got:?}");
    let matched = got
        .iter()
        .zip(&ref_greedy)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!("matched {matched}/{} greedy tokens", ref_greedy.len());
    assert_eq!(
        got, ref_greedy,
        "rlx Qwen2 greedy decode diverged from transformers at token {matched}"
    );
}
