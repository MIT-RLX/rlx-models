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

use anyhow::{Result, anyhow};

/// Read the next argv token after a flag (e.g. `--weights path`).
pub fn req(args: &[String], i: &mut usize) -> Result<String> {
    let flag = args[*i].clone();
    *i += 1;
    let v = args
        .get(*i)
        .ok_or_else(|| anyhow!("missing value for {flag}"))?
        .clone();
    *i += 1;
    Ok(v)
}

/// Remove `--weight-storage auto|f32|f16|bf16` (or `--weight-storage=…`) from
/// `args` and apply it to every later build in this process (see
/// `rlx_core::lm_registry::LmBuildOptions::weight_storage`).
pub fn take_weight_storage(args: &mut Vec<String>) -> Result<()> {
    let mut i = 0;
    while i < args.len() {
        let value = if args[i] == "--weight-storage" {
            let v = args
                .get(i + 1)
                .cloned()
                .ok_or_else(|| anyhow!("missing value for --weight-storage"))?;
            args.drain(i..i + 2);
            v
        } else if let Some(v) = args[i].strip_prefix("--weight-storage=") {
            let v = v.to_string();
            args.remove(i);
            v
        } else {
            i += 1;
            continue;
        };
        let storage: rlx_core::weight_storage::WeightStorage = value
            .parse()
            .map_err(|e: anyhow::Error| anyhow!("--weight-storage: {e}"))?;
        rlx_core::lm_registry::apply_weight_storage(Some(storage))?;
    }
    Ok(())
}

/// Remove `--number-format FMT` (or `--number-format=FMT`) from `args` and
/// apply it to every later compile in this process (see
/// `rlx_core::lm_registry::LmBuildOptions::number_format`). Lets any CLI take
/// the flag without its own parser knowing about it. Also takes the companion
/// storage setting, `--weight-storage` ([`take_weight_storage`]), so every CLI
/// that accepts one accepts both.
pub fn take_number_format(args: &mut Vec<String>) -> Result<()> {
    take_weight_storage(args)?;
    let mut i = 0;
    while i < args.len() {
        let value = if args[i] == "--number-format" {
            let v = args
                .get(i + 1)
                .cloned()
                .ok_or_else(|| anyhow!("missing value for --number-format"))?;
            args.drain(i..i + 2);
            v
        } else if let Some(v) = args[i].strip_prefix("--number-format=") {
            let v = v.to_string();
            args.remove(i);
            v
        } else {
            i += 1;
            continue;
        };
        let format: rlx_runtime::NumberFormat = value
            .parse()
            .map_err(|e: String| anyhow!("--number-format: {e}"))?;
        rlx_core::lm_registry::apply_number_format(Some(format))?;
    }
    Ok(())
}

#[cfg(test)]
mod number_format_tests {
    use super::*;

    #[test]
    fn strips_both_spellings() {
        let mut a: Vec<String> = ["--weights", "w", "--number-format", "f8", "--x"]
            .map(String::from)
            .to_vec();
        take_number_format(&mut a).unwrap();
        assert_eq!(a, ["--weights", "w", "--x"]);
        let mut b: Vec<String> = ["--weight-storage=bf16", "--weights", "w"]
            .map(String::from)
            .to_vec();
        take_number_format(&mut b).unwrap();
        assert_eq!(b, ["--weights", "w"]);
        rlx_core::lm_registry::apply_weight_storage(None).unwrap();
        let mut b: Vec<String> = ["--number-format=bf16"].map(String::from).to_vec();
        take_number_format(&mut b).unwrap();
        assert!(b.is_empty());
        let mut c: Vec<String> = ["--number-format", "f7"].map(String::from).to_vec();
        assert!(take_number_format(&mut c).is_err());
        rlx_core::lm_registry::apply_number_format(None).unwrap();
    }
}
