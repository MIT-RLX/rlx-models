//! Acoustic edit instructions — exact templates the model was trained on.

use anyhow::{Result, bail, ensure};

/// Pitch / speed / volume edit the generation pathway understands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcousticEdit {
    /// Semitone steps in `[-6, -1] ∪ [1, 6]` (zero is invalid).
    Pitch { steps: i32 },
    /// Rate in `[0.5, 2.0]` at 0.1 resolution.
    Speed { rate: f32 },
    /// Gain in `[0.3, 2.0]` at 0.1 resolution.
    Volume { gain: f32 },
}

fn is_tenth(x: f32) -> bool {
    let scaled = (x * 10.0).round();
    (x * 10.0 - scaled).abs() < 1e-4
}

fn format_tenth(x: f32) -> String {
    let scaled = (x * 10.0).round() as i32;
    if scaled % 10 == 0 {
        format!("{}", scaled / 10)
    } else {
        format!("{:.1}", scaled as f32 / 10.0)
    }
}

/// Render a trained acoustic instruction string.
pub fn format_acoustic(edit: AcousticEdit) -> Result<String> {
    match edit {
        AcousticEdit::Pitch { steps } => {
            ensure!(
                (-6..=6).contains(&steps) && steps != 0,
                "pitch steps must be in [-6,-1]∪[1,6], got {steps}"
            );
            let unit = if steps.abs() == 1 { "step" } else { "steps" };
            Ok(format!("shift the pitch by {steps} {unit}"))
        }
        AcousticEdit::Speed { rate } => {
            ensure!(
                (0.5..=2.0).contains(&rate) && is_tenth(rate),
                "speed must be in [0.5, 2.0] with 0.1 steps, got {rate}"
            );
            Ok(format!("adjust the speed to {}", format_tenth(rate)))
        }
        AcousticEdit::Volume { gain } => {
            ensure!(
                (0.3..=2.0).contains(&gain) && is_tenth(gain),
                "volume must be in [0.3, 2.0] with 0.1 steps, got {gain}"
            );
            Ok(format!("adjust the volume to {}", format_tenth(gain)))
        }
    }
}

/// Parse a trained acoustic instruction (trimmed, case-sensitive as upstream).
pub fn parse_acoustic(instruction: &str) -> Result<AcousticEdit> {
    let s = instruction.trim();
    if let Some(rest) = s.strip_prefix("shift the pitch by ") {
        let rest = rest.trim();
        let (num, unit) = rest
            .rsplit_once(' ')
            .ok_or_else(|| anyhow::anyhow!("malformed pitch instruction: {s:?}"))?;
        ensure!(
            unit == "step" || unit == "steps",
            "pitch unit must be step/steps, got {unit:?}"
        );
        let steps: i32 = num
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid pitch steps in {s:?}"))?;
        ensure!(
            (-6..=6).contains(&steps) && steps != 0,
            "pitch steps must be in [-6,-1]∪[1,6], got {steps}"
        );
        ensure!(
            (steps.abs() == 1 && unit == "step") || (steps.abs() != 1 && unit == "steps"),
            "pitch unit must match |steps| (step vs steps)"
        );
        return Ok(AcousticEdit::Pitch { steps });
    }
    if let Some(rest) = s.strip_prefix("adjust the speed to ") {
        let rate: f32 = rest
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid speed in {s:?}"))?;
        ensure!(
            (0.5..=2.0).contains(&rate) && is_tenth(rate),
            "speed must be in [0.5, 2.0] with 0.1 steps, got {rate}"
        );
        return Ok(AcousticEdit::Speed { rate });
    }
    if let Some(rest) = s.strip_prefix("adjust the volume to ") {
        let gain: f32 = rest
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid volume in {s:?}"))?;
        ensure!(
            (0.3..=2.0).contains(&gain) && is_tenth(gain),
            "volume must be in [0.3, 2.0] with 0.1 steps, got {gain}"
        );
        return Ok(AcousticEdit::Volume { gain });
    }
    bail!("unknown acoustic instruction (expected pitch/speed/volume template): {s:?}");
}
